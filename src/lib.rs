//! # quarry
//!
//! A vectorized SQL query engine over Apache Arrow, written from scratch:
//!
//! * [`sql`] — tokenizer and recursive-descent parser for a documented subset
//!   of SQL (SELECT/WHERE/JOIN/GROUP BY/HAVING/ORDER BY/LIMIT, derived tables,
//!   CTEs, EXPLAIN);
//! * [`binder`] — name resolution and type checking into a typed
//!   [`plan::LogicalPlan`];
//! * [`optimizer`] — rule-based rewrites: constant folding, predicate and
//!   projection pushdown, cost-based join reordering, top-k;
//! * [`physical`] — pull-based operators over Arrow record batches: scans
//!   (memory, CSV, Parquet with row-group pruning), filter, projection, hash
//!   join, nested-loop join, hash aggregate, sort, limit;
//! * [`engine::Engine`] — the API tying it together.
//!
//! See `docs/ARCHITECTURE.md` for the execution model.

pub mod binder;
pub mod catalog;
pub mod engine;
pub mod error;
pub mod format;
pub mod optimizer;
pub mod physical;
pub mod plan;
pub mod sql;
pub mod tpch;
pub mod types;

pub use engine::{CsvOptions, Engine, QueryResult};
pub use error::{QuarryError, Result};
