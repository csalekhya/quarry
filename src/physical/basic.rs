//! Filter, projection, sort, limit and constant-relation operators.

use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, RecordBatch};
use arrow::compute::SortOptions;
use arrow::compute::{concat_batches, filter_record_batch, lexsort_to_indices, take, SortColumn};
use arrow::datatypes::SchemaRef;

use super::expr::{canonical_zero, null_as_false, PhysicalExpr};
use super::{make_batch, ExecNode, Operator};
use crate::error::Result;

/// Keep rows for which the predicate is TRUE (NULL counts as false).
pub struct FilterExec {
    input: Box<ExecNode>,
    predicate: PhysicalExpr,
    text: String,
}

impl FilterExec {
    pub fn new(input: ExecNode, predicate: PhysicalExpr, text: String) -> Self {
        FilterExec {
            input: Box::new(input),
            predicate,
            text,
        }
    }
}

/// Apply a boolean predicate to a batch.
pub fn filter_batch(batch: &RecordBatch, predicate: &PhysicalExpr) -> Result<RecordBatch> {
    let mask = predicate.evaluate(batch)?.into_array(batch.num_rows());
    Ok(filter_record_batch(
        batch,
        &null_as_false(mask.as_boolean()),
    )?)
}

impl Operator for FilterExec {
    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        match self.input.pull()? {
            None => Ok(None),
            Some(b) => Ok(Some(filter_batch(&b, &self.predicate)?)),
        }
    }

    fn children(&self) -> Vec<&ExecNode> {
        vec![&self.input]
    }

    fn describe(&self) -> String {
        format!("FilterExec: {}", self.text)
    }
}

/// Compute output columns from expressions.
pub struct ProjectExec {
    input: Box<ExecNode>,
    exprs: Vec<PhysicalExpr>,
    schema: SchemaRef,
    text: Vec<String>,
}

impl ProjectExec {
    pub fn new(
        input: ExecNode,
        exprs: Vec<PhysicalExpr>,
        schema: SchemaRef,
        text: Vec<String>,
    ) -> Self {
        ProjectExec {
            input: Box::new(input),
            exprs,
            schema,
            text,
        }
    }
}

impl Operator for ProjectExec {
    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let Some(b) = self.input.pull()? else {
            return Ok(None);
        };
        let n = b.num_rows();
        let cols = self
            .exprs
            .iter()
            .map(|e| Ok(e.evaluate(&b)?.into_array(n)))
            .collect::<Result<Vec<ArrayRef>>>()?;
        Ok(Some(make_batch(&self.schema, cols, n)?))
    }

    fn children(&self) -> Vec<&ExecNode> {
        vec![&self.input]
    }

    fn describe(&self) -> String {
        format!("ProjectExec: {}", self.text.join(", "))
    }
}

/// One sort key: an expression plus direction.
pub struct PhysicalSortKey {
    pub expr: PhysicalExpr,
    pub options: SortOptions,
}

/// Full sort (or top-k when `fetch` is set). Drains its input on first call.
pub struct SortExec {
    input: Box<ExecNode>,
    keys: Vec<PhysicalSortKey>,
    fetch: Option<usize>,
    batch_size: usize,
    text: String,
    output: Option<(RecordBatch, usize)>,
}

impl SortExec {
    pub fn new(
        input: ExecNode,
        keys: Vec<PhysicalSortKey>,
        fetch: Option<usize>,
        batch_size: usize,
        text: String,
    ) -> Self {
        SortExec {
            input: Box::new(input),
            keys,
            fetch,
            batch_size,
            text,
            output: None,
        }
    }

    fn sort_all(&mut self) -> Result<RecordBatch> {
        let schema = Arc::clone(self.input.schema());
        let batches = self.input.collect()?;
        let all = concat_batches(&schema, &batches)?;
        if all.num_rows() == 0 {
            return Ok(all);
        }
        let columns = self
            .keys
            .iter()
            .map(|k| {
                Ok(SortColumn {
                    // Arrow sorts -0.0 before 0.0; SQL treats them as ties.
                    values: canonical_zero(k.expr.evaluate(&all)?.into_array(all.num_rows())),
                    options: Some(k.options),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let indices = lexsort_to_indices(&columns, self.fetch)?;
        let cols = all
            .columns()
            .iter()
            .map(|c| Ok(take(c.as_ref(), &indices, None)?))
            .collect::<Result<Vec<_>>>()?;
        make_batch(&schema, cols, indices.len())
    }
}

impl Operator for SortExec {
    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.output.is_none() {
            let sorted = self.sort_all()?;
            self.output = Some((sorted, 0));
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
        match self.fetch {
            Some(k) => format!("SortExec: TopK(fetch={k}) [{}]", self.text),
            None => format!("SortExec: [{}]", self.text),
        }
    }
}

/// Skip `skip` rows then pass at most `fetch` rows; stops pulling when done.
pub struct LimitExec {
    input: Box<ExecNode>,
    skip: usize,
    fetch: Option<usize>,
    seen: usize,
    emitted: usize,
}

impl LimitExec {
    pub fn new(input: ExecNode, skip: usize, fetch: Option<usize>) -> Self {
        LimitExec {
            input: Box::new(input),
            skip,
            fetch,
            seen: 0,
            emitted: 0,
        }
    }
}

impl Operator for LimitExec {
    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        loop {
            if self.fetch.is_some_and(|f| self.emitted >= f) {
                return Ok(None);
            }
            let Some(b) = self.input.pull()? else {
                return Ok(None);
            };
            let n = b.num_rows();
            let start = self.skip.saturating_sub(self.seen).min(n);
            self.seen += n;
            if start == n {
                continue;
            }
            let mut len = n - start;
            if let Some(f) = self.fetch {
                len = len.min(f - self.emitted);
            }
            self.emitted += len;
            return Ok(Some(b.slice(start, len)));
        }
    }

    fn children(&self) -> Vec<&ExecNode> {
        vec![&self.input]
    }

    fn describe(&self) -> String {
        match self.fetch {
            Some(f) => format!("LimitExec: skip={} fetch={f}", self.skip),
            None => format!("LimitExec: skip={}", self.skip),
        }
    }
}

/// A relation with no rows, or a single row with no columns.
pub struct EmptyExec {
    schema: SchemaRef,
    one_row: bool,
    done: bool,
}

impl EmptyExec {
    pub fn new(schema: SchemaRef, one_row: bool) -> Self {
        EmptyExec {
            schema,
            one_row,
            done: false,
        }
    }
}

impl Operator for EmptyExec {
    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.done || !self.one_row {
            return Ok(None);
        }
        self.done = true;
        Ok(Some(make_batch(&self.schema, vec![], 1)?))
    }

    fn children(&self) -> Vec<&ExecNode> {
        Vec::new()
    }

    fn describe(&self) -> String {
        format!("EmptyExec: rows={}", usize::from(self.one_row))
    }
}
