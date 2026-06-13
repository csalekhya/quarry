//! Hash aggregation.
//!
//! Group keys are encoded with Arrow's row format and mapped to dense group
//! ids through a hash map. Accumulators are vectorized: each receives a whole
//! input column plus the per-row group ids and updates flat `Vec`s indexed by
//! group id, so the inner loops never allocate per row. `DISTINCT`
//! aggregates first deduplicate (group id, value) pairs, then feed the
//! surviving values to the ordinary accumulator.
//!
//! Without GROUP BY there is exactly one group, which exists even when the
//! input is empty (`SELECT count(*) FROM empty` returns one row with 0).

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, AsArray, BooleanArray, Float64Array, Int64Array,
    PrimitiveArray, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType as ArrowType, Date32Type, Float64Type, Int64Type, SchemaRef};
use arrow::row::{RowConverter, Rows, SortField};
use rustc_hash::{FxHashMap, FxHashSet};

use super::expr::{canonical_zero, PhysicalExpr};
use super::{make_batch, ExecNode, Operator};
use crate::error::{exec_err, Result};
use crate::plan::expr::AggFunc;

/// Vectorized aggregate state for all groups.
pub trait Accumulator {
    /// Fold `values` (absent for `count(*)`) into the groups named by `groups`.
    fn update(
        &mut self,
        values: Option<&ArrayRef>,
        groups: &[u32],
        num_groups: usize,
    ) -> Result<()>;
    /// Produce one output value per group.
    fn finish(&mut self, num_groups: usize) -> Result<ArrayRef>;
}

struct CountAcc {
    counts: Vec<i64>,
}

impl Accumulator for CountAcc {
    fn update(
        &mut self,
        values: Option<&ArrayRef>,
        groups: &[u32],
        num_groups: usize,
    ) -> Result<()> {
        self.counts.resize(num_groups, 0);
        match values {
            None => groups.iter().for_each(|g| self.counts[*g as usize] += 1),
            Some(v) => {
                for (i, g) in groups.iter().enumerate() {
                    if v.is_valid(i) {
                        self.counts[*g as usize] += 1;
                    }
                }
            }
        }
        Ok(())
    }

    fn finish(&mut self, num_groups: usize) -> Result<ArrayRef> {
        self.counts.resize(num_groups, 0);
        Ok(Arc::new(Int64Array::from(std::mem::take(&mut self.counts))))
    }
}

struct SumIntAcc {
    sums: Vec<i64>,
    seen: Vec<bool>,
}

impl Accumulator for SumIntAcc {
    fn update(
        &mut self,
        values: Option<&ArrayRef>,
        groups: &[u32],
        num_groups: usize,
    ) -> Result<()> {
        self.sums.resize(num_groups, 0);
        self.seen.resize(num_groups, false);
        let Some(v) = values else { return Ok(()) };
        let v = v.as_primitive::<Int64Type>();
        for (i, g) in groups.iter().enumerate() {
            if v.is_valid(i) {
                let g = *g as usize;
                match self.sums[g].checked_add(v.value(i)) {
                    Some(s) => self.sums[g] = s,
                    None => return exec_err("integer overflow in sum()"),
                }
                self.seen[g] = true;
            }
        }
        Ok(())
    }

    fn finish(&mut self, num_groups: usize) -> Result<ArrayRef> {
        self.sums.resize(num_groups, 0);
        self.seen.resize(num_groups, false);
        let out: Int64Array = self
            .sums
            .iter()
            .zip(&self.seen)
            .map(|(s, seen)| seen.then_some(*s))
            .collect();
        Ok(Arc::new(out))
    }
}

/// Floating-point SUM (`avg == false`) or AVG (`avg == true`).
struct SumFloatAcc {
    sums: Vec<f64>,
    counts: Vec<i64>,
    avg: bool,
}

impl Accumulator for SumFloatAcc {
    fn update(
        &mut self,
        values: Option<&ArrayRef>,
        groups: &[u32],
        num_groups: usize,
    ) -> Result<()> {
        self.sums.resize(num_groups, 0.0);
        self.counts.resize(num_groups, 0);
        let Some(v) = values else { return Ok(()) };
        let v = v.as_primitive::<Float64Type>();
        for (i, g) in groups.iter().enumerate() {
            if v.is_valid(i) {
                self.sums[*g as usize] += v.value(i);
                self.counts[*g as usize] += 1;
            }
        }
        Ok(())
    }

    fn finish(&mut self, num_groups: usize) -> Result<ArrayRef> {
        self.sums.resize(num_groups, 0.0);
        self.counts.resize(num_groups, 0);
        let out: Float64Array = self
            .sums
            .iter()
            .zip(&self.counts)
            .map(|(s, c)| match (*c, self.avg) {
                (0, _) => None,
                (c, true) => Some(s / c as f64),
                (_, false) => Some(*s),
            })
            .collect();
        Ok(Arc::new(out))
    }
}

struct MinMaxPrimAcc<T: ArrowPrimitiveType> {
    vals: Vec<Option<T::Native>>,
    is_min: bool,
}

impl<T: ArrowPrimitiveType> Accumulator for MinMaxPrimAcc<T>
where
    T::Native: PartialOrd,
{
    fn update(
        &mut self,
        values: Option<&ArrayRef>,
        groups: &[u32],
        num_groups: usize,
    ) -> Result<()> {
        self.vals.resize(num_groups, None);
        let Some(v) = values else { return Ok(()) };
        let v = v.as_primitive::<T>();
        for (i, g) in groups.iter().enumerate() {
            if v.is_valid(i) {
                let x = v.value(i);
                let slot = &mut self.vals[*g as usize];
                let better = match slot {
                    None => true,
                    Some(cur) => {
                        if self.is_min {
                            x < *cur
                        } else {
                            x > *cur
                        }
                    }
                };
                if better {
                    *slot = Some(x);
                }
            }
        }
        Ok(())
    }

    fn finish(&mut self, num_groups: usize) -> Result<ArrayRef> {
        self.vals.resize(num_groups, None);
        let out: PrimitiveArray<T> = std::mem::take(&mut self.vals).into_iter().collect();
        Ok(Arc::new(out))
    }
}

struct MinMaxStrAcc {
    vals: Vec<Option<String>>,
    is_min: bool,
}

impl Accumulator for MinMaxStrAcc {
    fn update(
        &mut self,
        values: Option<&ArrayRef>,
        groups: &[u32],
        num_groups: usize,
    ) -> Result<()> {
        self.vals.resize(num_groups, None);
        let Some(v) = values else { return Ok(()) };
        let v = v.as_string::<i32>();
        for (i, g) in groups.iter().enumerate() {
            if v.is_valid(i) {
                let x = v.value(i);
                let slot = &mut self.vals[*g as usize];
                let better = match slot {
                    None => true,
                    Some(cur) => {
                        (self.is_min && x < cur.as_str()) || (!self.is_min && x > cur.as_str())
                    }
                };
                if better {
                    *slot = Some(x.to_string());
                }
            }
        }
        Ok(())
    }

    fn finish(&mut self, num_groups: usize) -> Result<ArrayRef> {
        self.vals.resize(num_groups, None);
        Ok(Arc::new(StringArray::from(std::mem::take(&mut self.vals))))
    }
}

struct MinMaxBoolAcc {
    vals: Vec<Option<bool>>,
    is_min: bool,
}

impl Accumulator for MinMaxBoolAcc {
    fn update(
        &mut self,
        values: Option<&ArrayRef>,
        groups: &[u32],
        num_groups: usize,
    ) -> Result<()> {
        self.vals.resize(num_groups, None);
        let Some(v) = values else { return Ok(()) };
        let v = v.as_boolean();
        for (i, g) in groups.iter().enumerate() {
            if v.is_valid(i) {
                let x = v.value(i);
                let slot = &mut self.vals[*g as usize];
                *slot = Some(match slot {
                    None => x,
                    Some(cur) if self.is_min => *cur && x,
                    Some(cur) => *cur || x,
                });
            }
        }
        Ok(())
    }

    fn finish(&mut self, num_groups: usize) -> Result<ArrayRef> {
        self.vals.resize(num_groups, None);
        Ok(Arc::new(BooleanArray::from(std::mem::take(&mut self.vals))))
    }
}

/// Deduplicates (group, value) pairs before delegating to `inner`.
struct DistinctAcc {
    inner: Box<dyn Accumulator>,
    converter: RowConverter,
    seen: FxHashSet<(u32, Box<[u8]>)>,
    rows: Rows,
    groups: Vec<u32>,
}

impl Accumulator for DistinctAcc {
    fn update(
        &mut self,
        values: Option<&ArrayRef>,
        groups: &[u32],
        _num_groups: usize,
    ) -> Result<()> {
        let Some(v) = values else { return Ok(()) };
        // -0.0 and 0.0 are one distinct value.
        let v = &canonical_zero(Arc::clone(v));
        let rows = self.converter.convert_columns(std::slice::from_ref(v))?;
        for (i, g) in groups.iter().enumerate() {
            if v.is_valid(i) && self.seen.insert((*g, rows.row(i).as_ref().into())) {
                self.rows.push(rows.row(i));
                self.groups.push(*g);
            }
        }
        Ok(())
    }

    fn finish(&mut self, num_groups: usize) -> Result<ArrayRef> {
        let cols = self.converter.convert_rows(self.rows.iter())?;
        self.inner.update(cols.first(), &self.groups, num_groups)?;
        self.inner.finish(num_groups)
    }
}

/// Build the accumulator for `func` over values of `input_type`.
pub fn create_accumulator(
    func: AggFunc,
    input_type: Option<&ArrowType>,
    distinct: bool,
) -> Result<Box<dyn Accumulator>> {
    let base: Box<dyn Accumulator> = match (func, input_type) {
        (AggFunc::CountStar, _) | (AggFunc::Count, _) => Box::new(CountAcc { counts: vec![] }),
        (AggFunc::Sum, Some(ArrowType::Int64)) => Box::new(SumIntAcc {
            sums: vec![],
            seen: vec![],
        }),
        (AggFunc::Sum, Some(ArrowType::Float64)) | (AggFunc::Avg, Some(ArrowType::Float64)) => {
            Box::new(SumFloatAcc {
                sums: vec![],
                counts: vec![],
                avg: func == AggFunc::Avg,
            })
        }
        (AggFunc::Min | AggFunc::Max, Some(t)) => {
            let is_min = func == AggFunc::Min;
            match t {
                ArrowType::Int64 => Box::new(MinMaxPrimAcc::<Int64Type> {
                    vals: vec![],
                    is_min,
                }),
                ArrowType::Float64 => Box::new(MinMaxPrimAcc::<Float64Type> {
                    vals: vec![],
                    is_min,
                }),
                ArrowType::Date32 => Box::new(MinMaxPrimAcc::<Date32Type> {
                    vals: vec![],
                    is_min,
                }),
                ArrowType::Utf8 => Box::new(MinMaxStrAcc {
                    vals: vec![],
                    is_min,
                }),
                ArrowType::Boolean => Box::new(MinMaxBoolAcc {
                    vals: vec![],
                    is_min,
                }),
                other => return exec_err(format!("{func}() is not supported for {other}")),
            }
        }
        (f, t) => return exec_err(format!("{f}() is not supported for {t:?}")),
    };
    match (distinct, input_type) {
        (true, Some(t)) => {
            let converter = RowConverter::new(vec![SortField::new(t.clone())])?;
            let rows = converter.empty_rows(0, 0);
            Ok(Box::new(DistinctAcc {
                inner: base,
                converter,
                seen: FxHashSet::default(),
                rows,
                groups: vec![],
            }))
        }
        _ => Ok(base),
    }
}

/// Maps group-key rows to dense ids.
struct GroupTable {
    converter: Option<RowConverter>,
    map: FxHashMap<Box<[u8]>, u32>,
    rows: Option<Rows>,
    num_groups: usize,
}

impl GroupTable {
    fn new(key_types: Vec<ArrowType>) -> Result<Self> {
        if key_types.is_empty() {
            return Ok(GroupTable {
                converter: None,
                map: FxHashMap::default(),
                rows: None,
                num_groups: 1,
            });
        }
        let converter = RowConverter::new(key_types.into_iter().map(SortField::new).collect())?;
        let rows = converter.empty_rows(0, 0);
        Ok(GroupTable {
            converter: Some(converter),
            map: FxHashMap::default(),
            rows: Some(rows),
            num_groups: 0,
        })
    }

    fn group_ids(&mut self, keys: &[ArrayRef], n: usize) -> Result<Vec<u32>> {
        let (Some(converter), Some(store)) = (&self.converter, &mut self.rows) else {
            return Ok(vec![0; n]);
        };
        let rows = converter.convert_columns(keys)?;
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            let row = rows.row(i);
            let id = match self.map.get(row.as_ref()) {
                Some(id) => *id,
                None => {
                    let id = self.num_groups as u32;
                    self.map.insert(row.as_ref().into(), id);
                    store.push(row);
                    self.num_groups += 1;
                    id
                }
            };
            ids.push(id);
        }
        Ok(ids)
    }

    fn key_arrays(&self) -> Result<Vec<ArrayRef>> {
        match (&self.converter, &self.rows) {
            (Some(c), Some(rows)) => Ok(c.convert_rows(rows.iter())?),
            _ => Ok(vec![]),
        }
    }
}

/// An aggregate as executed: function, optional argument, accumulator.
pub struct PhysicalAggregate {
    pub func: AggFunc,
    pub arg: Option<PhysicalExpr>,
    pub acc: Box<dyn Accumulator>,
}

/// Hash aggregation operator (a pipeline breaker).
pub struct HashAggregateExec {
    input: Box<ExecNode>,
    group_exprs: Vec<PhysicalExpr>,
    group_types: Vec<ArrowType>,
    aggs: Vec<PhysicalAggregate>,
    schema: SchemaRef,
    batch_size: usize,
    text: String,
    output: Option<(RecordBatch, usize)>,
    groups: usize,
}

impl HashAggregateExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: ExecNode,
        group_exprs: Vec<PhysicalExpr>,
        group_types: Vec<ArrowType>,
        aggs: Vec<PhysicalAggregate>,
        schema: SchemaRef,
        batch_size: usize,
        text: String,
    ) -> Self {
        HashAggregateExec {
            input: Box::new(input),
            group_exprs,
            group_types,
            aggs,
            schema,
            batch_size,
            text,
            output: None,
            groups: 0,
        }
    }

    fn aggregate_all(&mut self) -> Result<RecordBatch> {
        let mut table = GroupTable::new(self.group_types.clone())?;
        while let Some(batch) = self.input.pull()? {
            let n = batch.num_rows();
            let keys = self
                .group_exprs
                .iter()
                .map(|e| Ok(canonical_zero(e.evaluate(&batch)?.into_array(n))))
                .collect::<Result<Vec<_>>>()?;
            let ids = table.group_ids(&keys, n)?;
            for agg in &mut self.aggs {
                let values = match &agg.arg {
                    Some(e) => Some(e.evaluate(&batch)?.into_array(n)),
                    None => None,
                };
                agg.acc.update(values.as_ref(), &ids, table.num_groups)?;
            }
        }
        let num_groups = table.num_groups;
        self.groups = num_groups;
        let mut cols = table.key_arrays()?;
        for agg in &mut self.aggs {
            cols.push(agg.acc.finish(num_groups)?);
        }
        make_batch(&self.schema, cols, num_groups)
    }
}

impl Operator for HashAggregateExec {
    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.output.is_none() {
            let out = self.aggregate_all()?;
            self.output = Some((out, 0));
        }
        let Some((batch, offset)) = self.output.as_mut() else {
            return Ok(None);
        };
        if *offset >= batch.num_rows() {
            return Ok(None);
        }
        let len = self.batch_size.min(batch.num_rows() - *offset);
        let out = batch.slice(*offset, len);
        *offset += len;
        Ok(Some(out))
    }

    fn children(&self) -> Vec<&ExecNode> {
        vec![&self.input]
    }

    fn describe(&self) -> String {
        format!("HashAggregateExec: {}", self.text)
    }

    fn extra_metrics(&self) -> Vec<(&'static str, String)> {
        vec![("groups", self.groups.to_string())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ints(v: Vec<Option<i64>>) -> ArrayRef {
        Arc::new(Int64Array::from(v))
    }

    #[test]
    fn accumulators_handle_nulls_and_groups() {
        let vals = ints(vec![Some(1), None, Some(3), Some(3), Some(-2)]);
        let groups = [0, 0, 1, 1, 2];
        let run = |func, distinct| {
            let mut acc = create_accumulator(func, Some(&ArrowType::Int64), distinct).unwrap();
            acc.update(Some(&vals), &groups, 4).unwrap();
            acc.finish(4).unwrap()
        };
        let sum = run(AggFunc::Sum, false);
        assert_eq!(
            sum.as_primitive::<Int64Type>().iter().collect::<Vec<_>>(),
            vec![Some(1), Some(6), Some(-2), None]
        );
        let cnt = run(AggFunc::Count, true);
        assert_eq!(
            cnt.as_primitive::<Int64Type>().values().to_vec(),
            vec![1, 1, 1, 0]
        );
        let min = run(AggFunc::Min, false);
        assert_eq!(
            min.as_primitive::<Int64Type>().iter().collect::<Vec<_>>(),
            vec![Some(1), Some(3), Some(-2), None]
        );
        let dsum = run(AggFunc::Sum, true);
        assert_eq!(dsum.as_primitive::<Int64Type>().value(1), 3);
    }

    #[test]
    fn sum_overflow_is_an_error() {
        let vals = ints(vec![Some(i64::MAX), Some(1)]);
        let mut acc = create_accumulator(AggFunc::Sum, Some(&ArrowType::Int64), false).unwrap();
        assert!(acc.update(Some(&vals), &[0, 0], 1).is_err());
    }

    #[test]
    fn string_and_bool_min_max() {
        let s: ArrayRef = Arc::new(StringArray::from(vec![Some("b"), Some("a"), None]));
        let mut acc = create_accumulator(AggFunc::Max, Some(&ArrowType::Utf8), false).unwrap();
        acc.update(Some(&s), &[0, 0, 1], 2).unwrap();
        let out = acc.finish(2).unwrap();
        assert_eq!(
            out.as_string::<i32>().iter().collect::<Vec<_>>(),
            vec![Some("b"), None]
        );
        let b: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        let mut acc = create_accumulator(AggFunc::Min, Some(&ArrowType::Boolean), false).unwrap();
        acc.update(Some(&b), &[0, 0], 1).unwrap();
        assert!(!acc.finish(1).unwrap().as_boolean().value(0));
        assert!(create_accumulator(AggFunc::Sum, Some(&ArrowType::Utf8), false).is_err());
    }
}
