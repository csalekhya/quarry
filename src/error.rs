//! Error type shared by every layer of the engine.

use arrow::error::ArrowError;
use parquet::errors::ParquetError;

/// Every failure the engine can report, tagged by the stage that produced it.
#[derive(Debug, thiserror::Error)]
pub enum QuarryError {
    /// The SQL text could not be tokenized or parsed.
    #[error("syntax error at line {line}, column {column}: {message}")]
    Parse {
        message: String,
        line: usize,
        column: usize,
    },
    /// Name resolution failed (unknown table/column, ambiguity, misuse of aggregates).
    #[error("bind error: {0}")]
    Bind(String),
    /// An expression is not well typed.
    #[error("type error: {0}")]
    Type(String),
    /// The SQL is valid but uses a feature this engine does not implement.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// Planning or optimization invariant was violated (an engine bug).
    #[error("plan error: {0}")]
    Plan(String),
    /// A runtime failure while executing operators (overflow, bad cast, ...).
    #[error("execution error: {0}")]
    Execution(String),
    /// Table registration or lookup failed.
    #[error("catalog error: {0}")]
    Catalog(String),
    #[error("arrow error: {0}")]
    Arrow(#[from] ArrowError),
    #[error("parquet error: {0}")]
    Parquet(#[from] ParquetError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, QuarryError>;

/// Shorthand constructors used by the binder and planner.
pub(crate) fn bind_err<T>(msg: impl Into<String>) -> Result<T> {
    Err(QuarryError::Bind(msg.into()))
}

pub(crate) fn type_err<T>(msg: impl Into<String>) -> Result<T> {
    Err(QuarryError::Type(msg.into()))
}

pub(crate) fn plan_err<T>(msg: impl Into<String>) -> Result<T> {
    Err(QuarryError::Plan(msg.into()))
}

pub(crate) fn exec_err<T>(msg: impl Into<String>) -> Result<T> {
    Err(QuarryError::Execution(msg.into()))
}
