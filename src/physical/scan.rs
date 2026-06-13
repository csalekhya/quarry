//! Table scans over in-memory batches, CSV files and Parquet files.
//!
//! Scans apply three pushed-down optimizations: they read only the projected
//! columns (CSV and Parquet readers both support column projection), they
//! evaluate pushed filters before handing batches upstream, and for Parquet
//! they skip whole row groups whose min/max statistics prove that a pushed
//! `column <op> literal` filter cannot match.

use std::fs::File;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::compute::cast;
use arrow::datatypes::SchemaRef;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use parquet::file::metadata::RowGroupMetaData;
use parquet::file::statistics::Statistics;

use super::expr::PhysicalExpr;
use super::{ExecNode, Operator};
use crate::catalog::{TableEntry, TableSource};
use crate::error::{QuarryError, Result};
use crate::plan::expr::BinaryOp;
use crate::types::{DataType, ScalarValue};

type BatchIter = Box<dyn Iterator<Item = Result<RecordBatch>>>;

/// A simple `column <op> literal` predicate usable for row-group pruning.
#[derive(Debug, Clone)]
pub struct PruningPredicate {
    /// Index into the table (file) schema.
    pub column: usize,
    pub op: BinaryOp,
    pub value: ScalarValue,
}

/// Open a stream of normalized batches for `indices` of `table`.
///
/// `pruning` predicates may skip Parquet row groups; the returned count is the
/// number of row groups skipped (always 0 for other sources).
pub fn open_table(
    table: &TableEntry,
    indices: &[usize],
    batch_size: usize,
    pruning: &[PruningPredicate],
) -> Result<(BatchIter, usize, usize)> {
    let target = Arc::new(table.schema.project(indices)?);
    match &table.source {
        TableSource::Memory(batches) => {
            let mut out = Vec::new();
            for b in batches {
                let projected = b.project(indices)?;
                let mut offset = 0;
                while offset < projected.num_rows() {
                    let len = batch_size.min(projected.num_rows() - offset);
                    out.push(Ok(projected.slice(offset, len)));
                    offset += len;
                }
            }
            Ok((Box::new(out.into_iter()), 0, 0))
        }
        TableSource::Csv {
            path,
            has_header,
            delimiter,
            file_schema,
        } => {
            let file = File::open(path).map_err(|e| {
                QuarryError::Catalog(format!("cannot open {}: {e}", path.display()))
            })?;
            let reader = arrow::csv::ReaderBuilder::new(Arc::clone(file_schema))
                .with_header(*has_header)
                .with_delimiter(*delimiter)
                .with_batch_size(batch_size)
                .with_projection(indices.to_vec())
                .build(file)?;
            let iter = reader.map(move |b| normalize(b?, &target));
            Ok((Box::new(iter), 0, 0))
        }
        TableSource::Parquet { path } => {
            let file = File::open(path).map_err(|e| {
                QuarryError::Catalog(format!("cannot open {}: {e}", path.display()))
            })?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            let total_groups = builder.metadata().num_row_groups();
            let keep: Vec<usize> = builder
                .metadata()
                .row_groups()
                .iter()
                .enumerate()
                .filter(|(_, rg)| !pruning.iter().any(|p| row_group_excludes(rg, p, table)))
                .map(|(i, _)| i)
                .collect();
            let pruned = total_groups - keep.len();
            let mask = ProjectionMask::roots(builder.parquet_schema(), indices.iter().copied());
            let reader = builder
                .with_projection(mask)
                .with_row_groups(keep)
                .with_batch_size(batch_size)
                .build()?;
            let iter = reader.map(move |b| normalize(b?, &target));
            Ok((Box::new(iter), total_groups, pruned))
        }
    }
}

/// Cast a source batch to the normalized engine types of `target`.
pub fn normalize(batch: RecordBatch, target: &SchemaRef) -> Result<RecordBatch> {
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (col, field) in batch.columns().iter().zip(target.fields()) {
        if col.data_type() == field.data_type() {
            columns.push(Arc::clone(col));
        } else {
            columns.push(cast(col.as_ref(), field.data_type())?);
        }
    }
    super::make_batch(target, columns, batch.num_rows())
}

/// Can statistics prove that no row in this row group satisfies `p`?
fn row_group_excludes(rg: &RowGroupMetaData, p: &PruningPredicate, table: &TableEntry) -> bool {
    if p.column >= rg.num_columns() {
        return false;
    }
    let Some(stats) = rg.column(p.column).statistics() else {
        return false;
    };
    let t = table.column_type(p.column);
    let (Some(min), Some(max)) = (stat_value(stats, t, true), stat_value(stats, t, false)) else {
        return false;
    };
    let v = &p.value;
    let lt =
        |a: &ScalarValue, b: &ScalarValue| a.partial_cmp_sql(b) == Some(std::cmp::Ordering::Less);
    match p.op {
        BinaryOp::Eq => lt(v, &min) || lt(&max, v),
        BinaryOp::Lt => !lt(&min, v),
        BinaryOp::LtEq => lt(v, &min),
        BinaryOp::Gt => !lt(v, &max),
        BinaryOp::GtEq => lt(&max, v),
        _ => false,
    }
}

fn stat_value(stats: &Statistics, t: DataType, min: bool) -> Option<ScalarValue> {
    let pick = |a: Option<&i64>, b: Option<&i64>| if min { a.copied() } else { b.copied() };
    Some(match (stats, t) {
        (Statistics::Int32(s), DataType::Int64) => ScalarValue::Int64(Some(i64::from(*if min {
            s.min_opt()?
        } else {
            s.max_opt()?
        }))),
        (Statistics::Int32(s), DataType::Date32) => {
            ScalarValue::Date32(Some(*if min { s.min_opt()? } else { s.max_opt()? }))
        }
        (Statistics::Int64(s), DataType::Int64) => {
            ScalarValue::Int64(Some(pick(s.min_opt(), s.max_opt())?))
        }
        (Statistics::Double(s), DataType::Float64) => {
            ScalarValue::Float64(Some(*if min { s.min_opt()? } else { s.max_opt()? }))
        }
        (Statistics::ByteArray(s), DataType::Utf8) => {
            let b = if min { s.min_opt()? } else { s.max_opt()? };
            ScalarValue::Utf8(Some(std::str::from_utf8(b.data()).ok()?.to_string()))
        }
        _ => return None,
    })
}

/// Physical scan operator.
pub struct ScanExec {
    table: Arc<TableEntry>,
    indices: Vec<usize>,
    batch_size: usize,
    pruning: Vec<PruningPredicate>,
    filter: Option<PhysicalExpr>,
    filter_text: Vec<String>,
    iter: Option<BatchIter>,
    row_groups: (usize, usize),
    rows_read: usize,
}

impl ScanExec {
    pub fn new(
        table: Arc<TableEntry>,
        indices: Vec<usize>,
        batch_size: usize,
        filter: Option<PhysicalExpr>,
        filter_text: Vec<String>,
        pruning: Vec<PruningPredicate>,
    ) -> Self {
        ScanExec {
            table,
            indices,
            batch_size,
            pruning,
            filter,
            filter_text,
            iter: None,
            row_groups: (0, 0),
            rows_read: 0,
        }
    }
}

impl Operator for ScanExec {
    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.iter.is_none() {
            let (iter, total, pruned) =
                open_table(&self.table, &self.indices, self.batch_size, &self.pruning)?;
            self.iter = Some(iter);
            self.row_groups = (total, pruned);
        }
        let Some(iter) = self.iter.as_mut() else {
            return Ok(None);
        };
        let Some(batch) = iter.next().transpose()? else {
            return Ok(None);
        };
        self.rows_read += batch.num_rows();
        match &self.filter {
            None => Ok(Some(batch)),
            Some(f) => Ok(Some(super::basic::filter_batch(&batch, f)?)),
        }
    }

    fn children(&self) -> Vec<&ExecNode> {
        Vec::new()
    }

    fn describe(&self) -> String {
        let cols: Vec<&str> = self
            .indices
            .iter()
            .map(|&i| self.table.schema.field(i).name().as_str())
            .collect();
        let mut s = format!(
            "ScanExec: {} ({}) columns=[{}]",
            self.table.name,
            self.table.source.kind(),
            cols.join(", ")
        );
        if !self.filter_text.is_empty() {
            s.push_str(&format!(" filter=[{}]", self.filter_text.join(" AND ")));
        }
        if !self.pruning.is_empty() {
            s.push_str(&format!(" pruning_predicates={}", self.pruning.len()));
        }
        s
    }

    fn extra_metrics(&self) -> Vec<(&'static str, String)> {
        let mut m = vec![("rows_read", self.rows_read.to_string())];
        if let TableSource::Parquet { .. } = self.table.source {
            m.push(("row_groups", self.row_groups.0.to_string()));
            m.push(("row_groups_pruned", self.row_groups.1.to_string()));
        }
        m
    }
}
