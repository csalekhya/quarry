//! `quarry` command-line interface.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use quarry::tpch::{generate, write_tables, FileFormat};
use quarry::{CsvOptions, Engine, Result};

#[derive(Parser)]
#[command(
    name = "quarry",
    version,
    about = "A vectorized SQL query engine over Apache Arrow"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    Csv,
    Parquet,
}

#[derive(Subcommand)]
enum Command {
    /// Generate deterministic TPC-H-like data files.
    Gen {
        /// Scale factor (1.0 ≈ 6M lineitem rows).
        #[arg(long, default_value_t = 0.01)]
        sf: f64,
        /// Output directory.
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value_t = Format::Parquet)]
        format: Format,
        /// Rows per Parquet row group.
        #[arg(long, default_value_t = 16_384)]
        row_group_rows: usize,
    },
    /// Run a SQL statement (query or EXPLAIN [ANALYZE] [VERBOSE]).
    Query {
        /// Register every .csv/.parquet file in this directory (repeatable).
        #[arg(long = "data")]
        data: Vec<PathBuf>,
        /// Register a single file as NAME=PATH (repeatable).
        #[arg(long = "table")]
        tables: Vec<String>,
        /// Disable an optimizer rule by name (repeatable).
        #[arg(long = "disable-rule")]
        disable: Vec<String>,
        /// Rows per batch.
        #[arg(long, default_value_t = 8192)]
        batch_size: usize,
        /// Read the SQL from a file instead of the argument.
        #[arg(short = 'f', long)]
        file: Option<PathBuf>,
        /// The SQL text.
        sql: Option<String>,
    },
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Gen {
            sf,
            out,
            format,
            row_group_rows,
        } => {
            let start = Instant::now();
            let tables = generate(sf)?;
            let fmt = match format {
                Format::Csv => FileFormat::Csv,
                Format::Parquet => FileFormat::Parquet,
            };
            write_tables(&tables, &out, fmt, row_group_rows)?;
            for t in &tables {
                println!("{:<10} {:>9} rows", t.name, t.batch.num_rows());
            }
            println!(
                "wrote {} tables to {} in {:.2?}",
                tables.len(),
                out.display(),
                start.elapsed()
            );
            Ok(())
        }
        Command::Query {
            data,
            tables,
            disable,
            batch_size,
            file,
            sql,
        } => {
            let mut engine = Engine::new();
            engine.set_batch_size(batch_size);
            engine.set_disabled_rules(disable)?;
            for d in &data {
                engine.register_dir(d)?;
            }
            for t in &tables {
                let (name, path) = t.split_once('=').ok_or_else(|| {
                    quarry::QuarryError::Catalog(format!("--table expects NAME=PATH, got '{t}'"))
                })?;
                if path.ends_with(".parquet") {
                    engine.register_parquet(name, path)?;
                } else {
                    engine.register_csv(name, path, CsvOptions::default())?;
                }
            }
            let text = match (file, sql) {
                (Some(f), _) => std::fs::read_to_string(f)?,
                (None, Some(s)) => s,
                (None, None) => {
                    return Err(quarry::QuarryError::Catalog(
                        "no SQL given (pass it as an argument or with -f)".into(),
                    ))
                }
            };
            let start = Instant::now();
            let result = engine.sql(&text)?;
            print!("{result}");
            if !result.is_plan_text() {
                eprintln!("time: {:.2} ms", start.elapsed().as_secs_f64() * 1000.0);
            }
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
