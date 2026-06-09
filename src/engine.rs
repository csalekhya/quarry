//! The public entry point: register tables, run SQL, explain plans.
//!
//! ```
//! use quarry::Engine;
//! use arrow::array::{Int64Array, RecordBatch};
//! use arrow::datatypes::{DataType, Field, Schema};
//! use std::sync::Arc;
//!
//! let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
//! let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(vec![1, 2, 3]))]).unwrap();
//! let mut engine = Engine::new();
//! engine.register_batches("t", schema, vec![batch]).unwrap();
//! let result = engine.sql("SELECT sum(x) AS total FROM t WHERE x > 1").unwrap();
//! assert_eq!(result.to_string().lines().nth(2).unwrap().trim(), "5");
//! ```

use std::fmt;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType as ArrowType, Field, Schema, SchemaRef};
use log::debug;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::binder::Binder;
use crate::catalog::{normalize_schema, Catalog, StatsCollector, TableEntry, TableSource};
use crate::error::{QuarryError, Result};
use crate::format::format_table;
use crate::optimizer::cardinality::estimate;
use crate::optimizer::Optimizer;
use crate::physical::scan::{normalize, open_table};
use crate::physical::{create_physical_plan, ExecConfig};
use crate::plan::logical::LogicalPlan;
use crate::sql::ast::{Query, Statement};
use crate::sql::parse_statement;
use crate::types::ScalarValue;

/// CSV registration options.
#[derive(Debug, Clone)]
pub struct CsvOptions {
    pub has_header: bool,
    pub delimiter: u8,
    /// Explicit schema; inferred from the first 1000 rows when `None`.
    pub schema: Option<SchemaRef>,
}

impl Default for CsvOptions {
    fn default() -> Self {
        CsvOptions {
            has_header: true,
            delimiter: b',',
            schema: None,
        }
    }
}

/// Rows returned by a query.
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub schema: SchemaRef,
    pub batches: Vec<RecordBatch>,
}

impl QueryResult {
    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(|b| b.num_rows()).sum()
    }

    /// All rows as scalar values (convenient for tests and small results).
    pub fn rows(&self) -> Result<Vec<Vec<ScalarValue>>> {
        let mut out = Vec::with_capacity(self.num_rows());
        for b in &self.batches {
            for i in 0..b.num_rows() {
                out.push(
                    b.columns()
                        .iter()
                        .map(|c| ScalarValue::from_array(c.as_ref(), i))
                        .collect::<Result<_>>()?,
                );
            }
        }
        Ok(out)
    }

    fn from_lines(lines: &str) -> Result<QueryResult> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "plan",
            ArrowType::Utf8,
            false,
        )]));
        let col = StringArray::from(lines.lines().collect::<Vec<_>>());
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(col)])?;
        Ok(QueryResult {
            schema,
            batches: vec![batch],
        })
    }

    /// True for the single-column text produced by EXPLAIN.
    pub fn is_plan_text(&self) -> bool {
        self.schema.fields().len() == 1 && self.schema.field(0).name() == "plan"
    }
}

impl fmt::Display for QueryResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_plan_text() {
            for b in &self.batches {
                let col = b.column(0).as_any().downcast_ref::<StringArray>();
                if let Some(col) = col {
                    for line in col.iter().flatten() {
                        writeln!(f, "{line}")?;
                    }
                }
            }
            return Ok(());
        }
        f.write_str(&format_table(&self.schema, &self.batches))
    }
}

/// A query engine instance: a catalog plus execution settings.
pub struct Engine {
    catalog: Catalog,
    config: ExecConfig,
    disabled_rules: Vec<String>,
}

impl Default for Engine {
    fn default() -> Self {
        Engine::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        Engine {
            catalog: Catalog::new(),
            config: ExecConfig::default(),
            disabled_rules: Vec::new(),
        }
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    pub fn set_batch_size(&mut self, n: usize) {
        self.config.batch_size = n.max(1);
    }

    /// Disable optimizer rules by name (see [`crate::optimizer::RULE_NAMES`]).
    pub fn set_disabled_rules(&mut self, rules: Vec<String>) -> Result<()> {
        Optimizer::without(&rules)?;
        self.disabled_rules = rules;
        Ok(())
    }

    fn optimizer(&self) -> Result<Optimizer> {
        Optimizer::without(&self.disabled_rules)
    }

    fn add_table(&mut self, name: &str, schema: SchemaRef, source: TableSource) -> Result<()> {
        let name = name.to_ascii_lowercase();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(QuarryError::Catalog(format!(
                "invalid table name '{name}' (use letters, digits and _)"
            )));
        }
        let start = Instant::now();
        let mut entry = TableEntry {
            name: name.clone(),
            schema: Arc::clone(&schema),
            source,
            stats: StatsCollector::new(&schema).finish(),
        };
        let indices: Vec<usize> = (0..schema.fields().len()).collect();
        let (iter, _, _) = open_table(&entry, &indices, self.config.batch_size, &[])?;
        let mut collector = StatsCollector::new(&schema);
        for b in iter {
            collector.update(&b?)?;
        }
        entry.stats = collector.finish();
        debug!(
            "registered {name}: {} rows ({}) analyzed in {:?}",
            entry.stats.row_count,
            entry.source.kind(),
            start.elapsed()
        );
        self.catalog.insert(entry);
        Ok(())
    }

    /// Register in-memory batches (converted to engine types).
    pub fn register_batches(
        &mut self,
        name: &str,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> Result<()> {
        let normalized = normalize_schema(&schema)?;
        let batches = batches
            .into_iter()
            .map(|b| normalize(b, &normalized))
            .collect::<Result<Vec<_>>>()?;
        self.add_table(name, normalized, TableSource::Memory(batches))
    }

    /// Register a CSV file. It is read lazily on every scan.
    pub fn register_csv(
        &mut self,
        name: &str,
        path: impl AsRef<Path>,
        opts: CsvOptions,
    ) -> Result<()> {
        let path = path.as_ref().to_path_buf();
        let schema = match &opts.schema {
            Some(s) => Arc::clone(s),
            None => {
                let file = File::open(&path).map_err(|e| {
                    QuarryError::Catalog(format!("cannot open {}: {e}", path.display()))
                })?;
                let format = arrow::csv::reader::Format::default()
                    .with_header(opts.has_header)
                    .with_delimiter(opts.delimiter);
                let (s, _) = format.infer_schema(file, Some(1000))?;
                Arc::new(s)
            }
        };
        let normalized = normalize_schema(&schema)?;
        self.add_table(
            name,
            Arc::clone(&normalized),
            TableSource::Csv {
                path,
                has_header: opts.has_header,
                delimiter: opts.delimiter,
                file_schema: normalized,
            },
        )
    }

    /// Register a Parquet file. It is read lazily on every scan.
    pub fn register_parquet(&mut self, name: &str, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)
            .map_err(|e| QuarryError::Catalog(format!("cannot open {}: {e}", path.display())))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let normalized = normalize_schema(builder.schema())?;
        self.add_table(name, normalized, TableSource::Parquet { path })
    }

    /// Register every `*.csv` and `*.parquet` file in `dir` under its file stem.
    pub fn register_dir(&mut self, dir: impl AsRef<Path>) -> Result<Vec<String>> {
        let mut entries: Vec<_> = std::fs::read_dir(dir.as_ref())?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        entries.sort();
        let mut names = Vec::new();
        for p in entries {
            let (Some(stem), Some(ext)) = (
                p.file_stem().and_then(|s| s.to_str()),
                p.extension().and_then(|s| s.to_str()),
            ) else {
                continue;
            };
            match ext {
                "csv" => self.register_csv(stem, &p, CsvOptions::default())?,
                "parquet" => self.register_parquet(stem, &p)?,
                _ => continue,
            }
            names.push(stem.to_ascii_lowercase());
        }
        Ok(names)
    }

    fn parse_query(sql: &str) -> Result<Box<Query>> {
        match parse_statement(sql)? {
            Statement::Query(q) => Ok(q),
            Statement::Explain { query, .. } => Ok(query),
        }
    }

    /// The bound but unoptimized plan.
    pub fn logical_plan(&self, sql: &str) -> Result<LogicalPlan> {
        let q = Self::parse_query(sql)?;
        Binder::new(&self.catalog).bind(&q)
    }

    /// The optimized logical plan.
    pub fn optimized_plan(&self, sql: &str) -> Result<LogicalPlan> {
        self.optimizer()?.optimize(self.logical_plan(sql)?)
    }

    /// Run one statement (a query or EXPLAIN).
    pub fn sql(&self, sql: &str) -> Result<QueryResult> {
        match parse_statement(sql)? {
            Statement::Query(q) => self.execute(&q),
            Statement::Explain {
                analyze,
                verbose,
                query,
            } => QueryResult::from_lines(&self.explain_query(&query, analyze, verbose)?),
        }
    }

    fn execute(&self, q: &Query) -> Result<QueryResult> {
        let start = Instant::now();
        let plan = Binder::new(&self.catalog).bind(q)?;
        let plan = self.optimizer()?.optimize(plan)?;
        let planned = start.elapsed();
        let mut exec = create_physical_plan(&plan, &self.config)?;
        let schema = Arc::clone(exec.schema());
        let batches = exec.collect()?;
        debug!(
            "query planned in {planned:?}, executed in {:?}, {} rows",
            start.elapsed() - planned,
            batches.iter().map(|b| b.num_rows()).sum::<usize>()
        );
        Ok(QueryResult { schema, batches })
    }

    /// EXPLAIN text for a SQL query string (without the EXPLAIN keyword).
    pub fn explain(&self, sql: &str, analyze: bool, verbose: bool) -> Result<String> {
        let q = Self::parse_query(sql)?;
        self.explain_query(&q, analyze, verbose)
    }

    fn explain_query(&self, q: &Query, analyze: bool, verbose: bool) -> Result<String> {
        let plan = Binder::new(&self.catalog).bind(q)?;
        let mut out = String::new();
        let optimizer = self.optimizer()?;
        let optimized = if verbose {
            out.push_str("== Initial Logical Plan ==\n");
            out.push_str(&annotated(&plan));
            let (p, trace) = optimizer.optimize_traced(plan)?;
            for (rule, step) in trace {
                out.push_str(&format!("== After {rule} ==\n"));
                out.push_str(&annotated(&step));
            }
            p
        } else {
            optimizer.optimize(plan)?
        };
        if !verbose {
            out.push_str("== Optimized Logical Plan ==\n");
            out.push_str(&annotated(&optimized));
        }
        let mut exec = create_physical_plan(&optimized, &self.config)?;
        if analyze {
            let start = Instant::now();
            let rows: usize = exec.collect()?.iter().map(|b| b.num_rows()).sum();
            let elapsed = start.elapsed();
            out.push_str("== Physical Plan (analyzed) ==\n");
            out.push_str(&exec.display(true));
            out.push_str(&format!(
                "== Execution: {rows} rows in {:.3} ms ==\n",
                elapsed.as_secs_f64() * 1000.0
            ));
        } else {
            out.push_str("== Physical Plan ==\n");
            out.push_str(&exec.display(false));
        }
        Ok(out)
    }
}

/// Logical plan text with estimated row counts per node.
pub fn annotated(plan: &LogicalPlan) -> String {
    plan.display_tree(&|p| format!("  [est_rows={}]", estimate(p).rows.round() as u64))
}
