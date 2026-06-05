//! Benchmark: the 22 TPC-H-like queries on quarry, on quarry with join
//! reordering disabled, and on SQLite (in-memory, no indexes), all over the
//! same generated data.
//!
//! ```text
//! cargo run --release --example bench -- --sf 0.1 --iters 5
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use quarry::tpch::generate;
use quarry::types::{format_date, ScalarValue};
use quarry::Engine;
use rusqlite::types::Value;
use rusqlite::Connection;

#[derive(Parser)]
struct Args {
    /// Scale factor.
    #[arg(long, default_value_t = 0.1)]
    sf: f64,
    /// Timed runs per query for quarry (median reported, after one warm-up).
    #[arg(long, default_value_t = 5)]
    iters: usize,
    /// Timed runs per query for SQLite (0 skips SQLite).
    #[arg(long, default_value_t = 1)]
    sqlite_iters: usize,
    /// Queries not run on SQLite. Q19 repeats its join key inside each OR
    /// branch; SQLite does not factor it out and runs a nested loop over
    /// part x lineitem (about 7 s at SF 0.01, quadratic in SF).
    #[arg(long, value_delimiter = ',', default_value = "19")]
    sqlite_skip: Vec<u32>,
}

fn median(mut v: Vec<Duration>) -> f64 {
    v.sort();
    v[v.len() / 2].as_secs_f64() * 1000.0
}

fn time_quarry(engine: &Engine, sql: &str, iters: usize) -> (usize, f64) {
    let rows = engine.sql(sql).expect("warm-up").num_rows();
    let runs = (0..iters.max(1))
        .map(|_| {
            let t = Instant::now();
            engine.sql(sql).expect("query");
            t.elapsed()
        })
        .collect();
    (rows, median(runs))
}

fn sqlite_dialect(sql: &str) -> String {
    // The benchmark queries only need these rewrites (see tests/common for the full translator).
    let mut s = sql.to_string();
    for (from, to) in [
        ("DATE '1998-12-01' - INTERVAL '90' DAY", "'1998-09-02'"),
        ("DATE '1993-07-01' + INTERVAL '3' MONTH", "'1993-10-01'"),
        ("DATE '1994-01-01' + INTERVAL '1' YEAR", "'1995-01-01'"),
        ("DATE '1993-10-01' + INTERVAL '3' MONTH", "'1994-01-01'"),
        ("DATE '1995-09-01' + INTERVAL '1' MONTH", "'1995-10-01'"),
        ("DATE '1996-01-01' + INTERVAL '3' MONTH", "'1996-04-01'"),
        (
            "EXTRACT(YEAR FROM l_shipdate)",
            "CAST(strftime('%Y', l_shipdate) AS INTEGER)",
        ),
        (
            "EXTRACT(YEAR FROM o_orderdate)",
            "CAST(strftime('%Y', o_orderdate) AS INTEGER)",
        ),
        ("substring(c_phone FROM 1 FOR 2)", "substr(c_phone, 1, 2)"),
        ("DATE '", "'"),
    ] {
        s = s.replace(from, to);
    }
    s
}

fn main() {
    let args = Args::parse();
    let t = Instant::now();
    let tables = generate(args.sf).expect("generate");
    let mut engine = Engine::new();
    let mut no_reorder = Engine::new();
    no_reorder
        .set_disabled_rules(vec!["join_reorder".into()])
        .unwrap();
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("PRAGMA case_sensitive_like = ON;")
        .unwrap();
    for tb in &tables {
        engine
            .register_batches(tb.name, Arc::clone(&tb.schema), vec![tb.batch.clone()])
            .unwrap();
        no_reorder
            .register_batches(tb.name, Arc::clone(&tb.schema), vec![tb.batch.clone()])
            .unwrap();
        if args.sqlite_iters > 0 {
            load(&conn, tb);
        }
    }
    let lineitems = tables
        .iter()
        .find(|t| t.name == "lineitem")
        .map(|t| t.batch.num_rows())
        .unwrap_or(0);
    eprintln!(
        "loaded SF {} ({lineitems} lineitem rows) in {:.1?}",
        args.sf,
        t.elapsed()
    );

    println!("SF {} ({lineitems} lineitem rows), {} on {}, quarry median of {} runs, SQLite median of {} runs\n",
        args.sf, if cfg!(debug_assertions) { "debug build" } else { "release build" }, std::env::consts::ARCH, args.iters, args.sqlite_iters);
    println!("| query | rows | quarry ms | quarry, no join_reorder ms | SQLite ms |");
    println!("|---|---:|---:|---:|---:|");
    let (mut total_q, mut total_s) = (0.0, 0.0);
    for n in 1..=22 {
        let sql = std::fs::read_to_string(format!(
            "{}/queries/tpch/q{n:02}.sql",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let (rows, q_ms) = time_quarry(&engine, &sql, args.iters);
        total_q += q_ms;
        let unordered = no_reorder.optimized_plan(&sql).unwrap().to_string();
        let nr = if unordered.contains("CrossJoin") {
            "cross product, skipped".to_string()
        } else {
            format!("{:.1}", time_quarry(&no_reorder, &sql, args.iters).1)
        };
        let lite = if args.sqlite_iters == 0 {
            "-".to_string()
        } else if args.sqlite_skip.contains(&n) {
            "not run (see note)".to_string()
        } else {
            let lsql = sqlite_dialect(&sql);
            let runs = (0..args.sqlite_iters)
                .map(|_| {
                    let t = Instant::now();
                    let mut stmt = conn.prepare(&lsql).expect("sqlite prepare");
                    let n = stmt.query_map([], |_| Ok(())).unwrap().count();
                    assert_eq!(n, rows, "q{n:02}: SQLite returned a different row count");
                    t.elapsed()
                })
                .collect();
            let ms = median(runs);
            total_s += ms;
            format!("{ms:.1}")
        };
        println!("| q{n:02} | {rows} | {q_ms:.1} | {nr} | {lite} |");
    }
    println!(
        "| total | | {total_q:.1} | | {} |",
        if args.sqlite_iters == 0 {
            "-".into()
        } else {
            format!("{total_s:.1}")
        }
    );
}

fn load(conn: &Connection, t: &quarry::tpch::Table) {
    let cols: Vec<String> = t.schema.fields().iter().map(|f| f.name().clone()).collect();
    conn.execute(
        &format!("CREATE TABLE {} ({})", t.name, cols.join(", ")),
        [],
    )
    .unwrap();
    let ph = vec!["?"; cols.len()].join(", ");
    let tx = conn.unchecked_transaction().unwrap();
    {
        let mut stmt = tx
            .prepare(&format!("INSERT INTO {} VALUES ({ph})", t.name))
            .unwrap();
        for i in 0..t.batch.num_rows() {
            let vals: Vec<Value> = t
                .batch
                .columns()
                .iter()
                .map(|c| match ScalarValue::from_array(c.as_ref(), i).unwrap() {
                    ScalarValue::Int64(Some(v)) => Value::Integer(v),
                    ScalarValue::Float64(Some(v)) => Value::Real(v),
                    ScalarValue::Utf8(Some(s)) => Value::Text(s),
                    ScalarValue::Date32(Some(d)) => Value::Text(format_date(d)),
                    _ => Value::Null,
                })
                .collect();
            stmt.execute(rusqlite::params_from_iter(vals)).unwrap();
        }
    }
    tx.commit().unwrap();
}
