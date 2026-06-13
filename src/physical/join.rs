//! Hash join and nested-loop join.
//!
//! **Hash join.** The right child is the build side: it is drained, its key
//! columns are encoded with Arrow's row format (so multi-column keys of any
//! type compare as byte strings) and indexed in a chained hash table: a map
//! from key hash to the first build row plus a `next` array linking rows with
//! the same hash. The left child is then streamed batch by batch; each probe
//! row walks its chain, compares encoded keys, and the matching
//! (probe, build) index pairs are gathered with Arrow's `take` kernel.
//! Rows with a NULL key never match (SQL equality). For LEFT joins, probe
//! rows without a surviving match are emitted once with NULLs on the right.
//!
//! Build is O(|R|), probe is O(|L| + matches); memory is O(|R|).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch, UInt32Array};
use arrow::compute::{concat_batches, take};
use arrow::datatypes::SchemaRef;
use arrow::row::{RowConverter, Rows, SortField};
use rustc_hash::FxHashMap;
use std::hash::BuildHasher;

use super::expr::{canonical_zero, null_as_false, PhysicalExpr};
use super::{make_batch, ExecNode, Operator};
use crate::error::Result;
use crate::plan::logical::JoinType;

const NONE: u32 = u32::MAX;

struct BuildSide {
    batch: RecordBatch,
    rows: Rows,
    heads: FxHashMap<u64, u32>,
    next: Vec<u32>,
}

/// Equi-join with the right child as the build side.
pub struct HashJoinExec {
    left: Box<ExecNode>,
    right: Box<ExecNode>,
    join_type: JoinType,
    left_keys: Vec<PhysicalExpr>,
    right_keys: Vec<PhysicalExpr>,
    filter: Option<PhysicalExpr>,
    schema: SchemaRef,
    converter: RowConverter,
    build: Option<BuildSide>,
    text: String,
    build_rows: usize,
}

impl HashJoinExec {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        left: ExecNode,
        right: ExecNode,
        join_type: JoinType,
        left_keys: Vec<PhysicalExpr>,
        right_keys: Vec<PhysicalExpr>,
        key_types: Vec<arrow::datatypes::DataType>,
        filter: Option<PhysicalExpr>,
        schema: SchemaRef,
        text: String,
    ) -> Result<Self> {
        let converter = RowConverter::new(key_types.into_iter().map(SortField::new).collect())?;
        Ok(HashJoinExec {
            left: Box::new(left),
            right: Box::new(right),
            join_type,
            left_keys,
            right_keys,
            filter,
            schema,
            converter,
            build: None,
            text,
            build_rows: 0,
        })
    }

    fn build_table(&mut self) -> Result<BuildSide> {
        let schema = Arc::clone(self.right.schema());
        let batches = self.right.collect()?;
        let batch = concat_batches(&schema, &batches)?;
        let keys = eval_keys(&self.right_keys, &batch)?;
        let rows = self.converter.convert_columns(&keys)?;
        let valid = all_valid(&keys, batch.num_rows());
        let mut heads: FxHashMap<u64, u32> = FxHashMap::default();
        heads.reserve(batch.num_rows());
        let mut next = vec![NONE; batch.num_rows()];
        let hasher = rustc_hash::FxBuildHasher;
        // Insert in reverse so chains list build rows in ascending order.
        for i in (0..batch.num_rows()).rev() {
            if !valid[i] {
                continue;
            }
            let h = hasher.hash_one(rows.row(i).as_ref());
            let slot = heads.entry(h).or_insert(NONE);
            next[i] = *slot;
            *slot = i as u32;
        }
        self.build_rows = batch.num_rows();
        Ok(BuildSide {
            batch,
            rows,
            heads,
            next,
        })
    }

    fn probe(&self, build: &BuildSide, probe: &RecordBatch) -> Result<RecordBatch> {
        let keys = eval_keys(&self.left_keys, probe)?;
        let rows = self.converter.convert_columns(&keys)?;
        let valid = all_valid(&keys, probe.num_rows());
        let hasher = rustc_hash::FxBuildHasher;
        let mut probe_idx: Vec<u32> = Vec::new();
        let mut build_idx: Vec<u32> = Vec::new();
        for (i, ok) in valid.iter().enumerate() {
            if !ok {
                continue;
            }
            let row = rows.row(i);
            let Some(&head) = build.heads.get(&hasher.hash_one(row.as_ref())) else {
                continue;
            };
            let mut j = head;
            while j != NONE {
                if build.rows.row(j as usize) == row {
                    probe_idx.push(i as u32);
                    build_idx.push(j);
                }
                j = build.next[j as usize];
            }
        }
        let mut candidates = gather(
            &self.schema,
            probe,
            &build.batch,
            probe_idx.clone(),
            build_idx.into_iter().map(Some).collect(),
        )?;
        let mut matched_probe = probe_idx;
        if let Some(f) = &self.filter {
            let mask = f.evaluate(&candidates)?.into_array(candidates.num_rows());
            let mask = null_as_false(mask.as_boolean());
            matched_probe = matched_probe
                .into_iter()
                .zip(mask.values().iter())
                .filter_map(|(p, keep)| keep.then_some(p))
                .collect();
            candidates = arrow::compute::filter_record_batch(&candidates, &mask)?;
        }
        if self.join_type == JoinType::Left {
            let mut seen = vec![false; probe.num_rows()];
            for p in &matched_probe {
                seen[*p as usize] = true;
            }
            let unmatched: Vec<u32> = (0..probe.num_rows() as u32)
                .filter(|i| !seen[*i as usize])
                .collect();
            if !unmatched.is_empty() {
                let n = unmatched.len();
                let extra = gather(&self.schema, probe, &build.batch, unmatched, vec![None; n])?;
                candidates = concat_batches(&self.schema, &[candidates, extra])?;
            }
        }
        Ok(candidates)
    }
}

/// Evaluate the equi-join keys; `-0.0` is folded into `0.0` so the two zeros
/// encode to the same row bytes and match each other.
fn eval_keys(keys: &[PhysicalExpr], batch: &RecordBatch) -> Result<Vec<ArrayRef>> {
    keys.iter()
        .map(|k| {
            Ok(canonical_zero(
                k.evaluate(batch)?.into_array(batch.num_rows()),
            ))
        })
        .collect()
}

/// Per-row flag: true when no key column is NULL.
fn all_valid(keys: &[ArrayRef], n: usize) -> Vec<bool> {
    let mut v = vec![true; n];
    for k in keys {
        if k.null_count() == 0 {
            continue;
        }
        for (i, slot) in v.iter_mut().enumerate() {
            if k.is_null(i) {
                *slot = false;
            }
        }
    }
    v
}

/// Assemble output rows from probe indices and (optional) build indices.
fn gather(
    schema: &SchemaRef,
    left: &RecordBatch,
    right: &RecordBatch,
    left_idx: Vec<u32>,
    right_idx: Vec<Option<u32>>,
) -> Result<RecordBatch> {
    let n = left_idx.len();
    let li = UInt32Array::from(left_idx);
    let ri = UInt32Array::from(right_idx);
    let mut cols = Vec::with_capacity(schema.fields().len());
    for c in left.columns() {
        cols.push(take(c.as_ref(), &li, None)?);
    }
    for c in right.columns() {
        cols.push(take(c.as_ref(), &ri, None)?);
    }
    make_batch(schema, cols, n)
}

impl Operator for HashJoinExec {
    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.build.is_none() {
            let b = self.build_table()?;
            self.build = Some(b);
        }
        let Some(build) = self.build.take() else {
            return Ok(None);
        };
        // An empty build side means an inner join produces nothing: skip the probe entirely.
        if build.batch.num_rows() == 0 && self.join_type == JoinType::Inner {
            self.build = Some(build);
            return Ok(None);
        }
        let result = match self.left.pull()? {
            None => None,
            Some(p) => Some(self.probe(&build, &p)?),
        };
        self.build = Some(build);
        Ok(result)
    }

    fn children(&self) -> Vec<&ExecNode> {
        vec![&self.left, &self.right]
    }

    fn describe(&self) -> String {
        format!("HashJoinExec: {}", self.text)
    }

    fn extra_metrics(&self) -> Vec<(&'static str, String)> {
        vec![("build_rows", self.build_rows.to_string())]
    }
}

/// Join without equality keys: every left row is paired with every right row,
/// optionally filtered. The right side is materialized; output is produced in
/// chunks of whole left rows to bound batch size.
pub struct NestedLoopJoinExec {
    left: Box<ExecNode>,
    right: Box<ExecNode>,
    join_type: JoinType,
    filter: Option<PhysicalExpr>,
    schema: SchemaRef,
    batch_size: usize,
    text: String,
    build: Option<RecordBatch>,
    current: Option<(RecordBatch, usize)>,
}

impl NestedLoopJoinExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        left: ExecNode,
        right: ExecNode,
        join_type: JoinType,
        filter: Option<PhysicalExpr>,
        schema: SchemaRef,
        batch_size: usize,
        text: String,
    ) -> Self {
        NestedLoopJoinExec {
            left: Box::new(left),
            right: Box::new(right),
            join_type,
            filter,
            schema,
            batch_size,
            text,
            build: None,
            current: None,
        }
    }

    fn chunk(
        &self,
        probe: &RecordBatch,
        start: usize,
        end: usize,
        build: &RecordBatch,
    ) -> Result<RecordBatch> {
        let m = build.num_rows();
        let mut li = Vec::with_capacity((end - start) * m);
        let mut ri = Vec::with_capacity((end - start) * m);
        for i in start..end {
            for j in 0..m {
                li.push(i as u32);
                ri.push(Some(j as u32));
            }
        }
        let mut out = gather(&self.schema, probe, build, li.clone(), ri)?;
        let mut kept = li;
        if let Some(f) = &self.filter {
            let mask = f.evaluate(&out)?.into_array(out.num_rows());
            let mask = null_as_false(mask.as_boolean());
            kept = kept
                .into_iter()
                .zip(mask.values().iter())
                .filter_map(|(p, k)| k.then_some(p))
                .collect();
            out = arrow::compute::filter_record_batch(&out, &mask)?;
        }
        if self.join_type == JoinType::Left {
            let mut seen = vec![false; end - start];
            for p in kept {
                seen[p as usize - start] = true;
            }
            let unmatched: Vec<u32> = (start..end)
                .filter(|i| !seen[i - start])
                .map(|i| i as u32)
                .collect();
            if !unmatched.is_empty() {
                let n = unmatched.len();
                let extra = gather(&self.schema, probe, build, unmatched, vec![None; n])?;
                out = concat_batches(&self.schema, &[out, extra])?;
            }
        }
        Ok(out)
    }
}

impl Operator for NestedLoopJoinExec {
    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.build.is_none() {
            let schema = Arc::clone(self.right.schema());
            let batches = self.right.collect()?;
            self.build = Some(concat_batches(&schema, &batches)?);
        }
        let build = self
            .build
            .take()
            .unwrap_or_else(|| unreachable!("built above"));
        let result = loop {
            if build.num_rows() == 0 && self.join_type == JoinType::Inner {
                break None;
            }
            if self.current.is_none() {
                match self.left.pull()? {
                    None => break None,
                    Some(b) => self.current = Some((b, 0)),
                }
            }
            let Some((probe, offset)) = self.current.take() else {
                break None;
            };
            if offset >= probe.num_rows() {
                continue;
            }
            let rows_per_chunk = (self.batch_size / build.num_rows().max(1)).max(1);
            let end = (offset + rows_per_chunk).min(probe.num_rows());
            let out = self.chunk(&probe, offset, end, &build)?;
            self.current = Some((probe, end));
            break Some(out);
        };
        self.build = Some(build);
        Ok(result)
    }

    fn children(&self) -> Vec<&ExecNode> {
        vec![&self.left, &self.right]
    }

    fn describe(&self) -> String {
        format!("NestedLoopJoinExec: {}", self.text)
    }
}
