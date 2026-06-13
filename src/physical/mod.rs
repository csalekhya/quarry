//! Physical execution: a pull-based (Volcano-style) tree of operators that
//! exchange Arrow [`RecordBatch`]es.
//!
//! Every operator implements [`Operator::next_batch`]. Pipelined operators
//! (scan, filter, project, limit, the probe side of a join) transform one
//! batch at a time; pipeline breakers (hash-join build, aggregation, sort)
//! drain their input on the first call. [`ExecNode`] wraps each operator and
//! records rows, batches and wall time for `EXPLAIN ANALYZE`.

pub mod aggregate;
pub mod basic;
pub mod expr;
pub mod join;
pub mod planner;
pub mod scan;

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};

use crate::error::Result;
use crate::plan::logical::ColumnDef;

pub use planner::create_physical_plan;

/// Tuning knobs for execution.
#[derive(Debug, Clone)]
pub struct ExecConfig {
    /// Target rows per batch produced by scans and pipeline breakers.
    pub batch_size: usize,
}

impl Default for ExecConfig {
    fn default() -> Self {
        ExecConfig { batch_size: 8192 }
    }
}

/// A physical operator.
pub trait Operator {
    /// Produce the next batch, or `None` when exhausted.
    fn next_batch(&mut self) -> Result<Option<RecordBatch>>;
    /// Child nodes, for plan rendering and metrics.
    fn children(&self) -> Vec<&ExecNode>;
    /// One-line description for EXPLAIN.
    fn describe(&self) -> String;
    /// Operator-specific counters shown by EXPLAIN ANALYZE.
    fn extra_metrics(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// Runtime counters for one node.
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub output_rows: usize,
    pub output_batches: usize,
    /// Wall time spent inside this node's `next`, including its children.
    pub elapsed: Duration,
}

/// An operator plus its output schema and metrics.
pub struct ExecNode {
    op: Box<dyn Operator>,
    schema: SchemaRef,
    metrics: Metrics,
}

impl ExecNode {
    pub fn new(op: Box<dyn Operator>, schema: SchemaRef) -> Self {
        ExecNode {
            op,
            schema,
            metrics: Metrics::default(),
        }
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// Pull the next non-empty batch.
    pub fn pull(&mut self) -> Result<Option<RecordBatch>> {
        let start = Instant::now();
        let result = loop {
            match self.op.next_batch() {
                Ok(Some(b)) if b.num_rows() == 0 => continue,
                other => break other,
            }
        };
        self.metrics.elapsed += start.elapsed();
        if let Ok(Some(b)) = &result {
            self.metrics.output_rows += b.num_rows();
            self.metrics.output_batches += 1;
        }
        result
    }

    /// Drain the node into a vector of batches.
    pub fn collect(&mut self) -> Result<Vec<RecordBatch>> {
        let mut out = Vec::new();
        while let Some(b) = self.pull()? {
            out.push(b);
        }
        Ok(out)
    }

    /// Render the physical plan; with `analyze`, include runtime metrics.
    pub fn display(&self, analyze: bool) -> String {
        let mut out = String::new();
        self.write(&mut out, 0, analyze);
        out
    }

    fn write(&self, out: &mut String, depth: usize, analyze: bool) {
        let _ = write!(out, "{}{}", "  ".repeat(depth), self.op.describe());
        if analyze {
            let children: Duration = self.op.children().iter().map(|c| c.metrics.elapsed).sum();
            let self_time = self.metrics.elapsed.saturating_sub(children);
            let _ = write!(
                out,
                "  [rows={} batches={} time={:.3}ms self={:.3}ms",
                self.metrics.output_rows,
                self.metrics.output_batches,
                ms(self.metrics.elapsed),
                ms(self_time)
            );
            for (k, v) in self.op.extra_metrics() {
                let _ = write!(out, " {k}={v}");
            }
            out.push(']');
        }
        out.push('\n');
        for c in self.op.children() {
            c.write(out, depth + 1, analyze);
        }
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Arrow schema for a list of logical output columns. All fields are marked
/// nullable: nullability is tracked logically, not enforced physically.
pub fn arrow_schema(cols: &[ColumnDef]) -> SchemaRef {
    Arc::new(Schema::new(
        cols.iter()
            .map(|c| Field::new(c.name.clone(), c.data_type.to_arrow(), true))
            .collect::<Vec<_>>(),
    ))
}

/// Build a batch, allowing zero columns (the row count is then explicit).
pub fn make_batch(
    schema: &SchemaRef,
    columns: Vec<arrow::array::ArrayRef>,
    rows: usize,
) -> Result<RecordBatch> {
    Ok(RecordBatch::try_new_with_options(
        Arc::clone(schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )?)
}
