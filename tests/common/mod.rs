//! Shared helpers for differential tests: load the same data into quarry and
//! SQLite, translate quarry's dialect to SQLite's, and compare results.

#![allow(dead_code)]

pub mod features;

use std::sync::{Arc, Mutex, OnceLock};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use quarry::types::{add_months_days, format_date, parse_date, ScalarValue};
use quarry::{Engine, QueryResult};
use rusqlite::types::Value;
use rusqlite::Connection;

/// A normalized value for cross-engine comparison.
#[derive(Debug, Clone)]
pub enum V {
    Null,
    Int(i64),
    Float(f64),
    Text(String),
}

impl V {
    fn sort_key(&self) -> String {
        match self {
            V::Null => "0".into(),
            V::Int(i) => format!("1{:.4}", *i as f64),
            // `+ 0.0` folds -0.0 into 0.0, which compare equal.
            V::Float(f) => format!("1{:.4}", f + 0.0),
            V::Text(s) => format!("2{s}"),
        }
    }
}

pub fn approx_eq(a: &V, b: &V) -> bool {
    let close = |x: f64, y: f64| (x - y).abs() <= 1e-6 * x.abs().max(y.abs()).max(1.0);
    match (a, b) {
        (V::Null, V::Null) => true,
        (V::Int(x), V::Int(y)) => x == y,
        (V::Float(x), V::Float(y)) => close(*x, *y),
        (V::Int(x), V::Float(y)) | (V::Float(y), V::Int(x)) => close(*x as f64, *y),
        (V::Text(x), V::Text(y)) => x == y,
        _ => false,
    }
}

fn from_scalar(v: &ScalarValue) -> V {
    match v {
        _ if v.is_null() => V::Null,
        ScalarValue::Boolean(Some(b)) => V::Int(i64::from(*b)),
        ScalarValue::Int64(Some(i)) => V::Int(*i),
        ScalarValue::Float64(Some(f)) => V::Float(*f),
        ScalarValue::Utf8(Some(s)) => V::Text(s.clone()),
        ScalarValue::Date32(Some(d)) => V::Text(format_date(*d)),
        _ => V::Null,
    }
}

pub fn quarry_rows(r: &QueryResult) -> Vec<Vec<V>> {
    r.rows()
        .expect("rows")
        .iter()
        .map(|row| row.iter().map(from_scalar).collect())
        .collect()
}

pub fn sqlite_rows(conn: &Connection, sql: &str) -> rusqlite::Result<Vec<Vec<V>>> {
    let mut stmt = conn.prepare(sql)?;
    let n = stmt.column_count();
    let rows = stmt.query_map([], |row| {
        (0..n)
            .map(|i| {
                Ok(match row.get::<_, Value>(i)? {
                    Value::Null => V::Null,
                    Value::Integer(i) => V::Int(i),
                    Value::Real(f) => V::Float(f),
                    Value::Text(s) => V::Text(s),
                    Value::Blob(_) => V::Text("<blob>".into()),
                })
            })
            .collect::<rusqlite::Result<Vec<V>>>()
    })?;
    rows.collect()
}

/// Compare two result sets; unordered comparison sorts both canonically.
pub fn compare(label: &str, mut ours: Vec<Vec<V>>, mut theirs: Vec<Vec<V>>, ordered: bool) {
    if !ordered {
        let key = |r: &Vec<V>| r.iter().map(V::sort_key).collect::<Vec<_>>().join("\u{1}");
        ours.sort_by_key(key);
        theirs.sort_by_key(key);
    }
    assert_eq!(
        ours.len(),
        theirs.len(),
        "{label}: row count differs (quarry {} vs sqlite {})\nquarry: {:?}\nsqlite: {:?}",
        ours.len(),
        theirs.len(),
        ours.iter().take(5).collect::<Vec<_>>(),
        theirs.iter().take(5).collect::<Vec<_>>()
    );
    for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
        assert_eq!(a.len(), b.len(), "{label}: column count differs at row {i}");
        for (x, y) in a.iter().zip(b) {
            assert!(
                approx_eq(x, y),
                "{label}: row {i} differs\nquarry: {a:?}\nsqlite: {b:?}"
            );
        }
    }
}

/// Create `name` in SQLite and insert every row of `batches`.
pub fn load_sqlite(conn: &Connection, name: &str, schema: &SchemaRef, batches: &[RecordBatch]) {
    use arrow::datatypes::DataType as T;
    let cols: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| {
            let t = match f.data_type() {
                T::Int64 | T::Int32 | T::Boolean => "INTEGER",
                T::Float64 => "REAL",
                _ => "TEXT",
            };
            format!("{} {t}", f.name())
        })
        .collect();
    conn.execute(&format!("CREATE TABLE {name} ({})", cols.join(", ")), [])
        .expect("create table");
    let placeholders = vec!["?"; schema.fields().len()].join(", ");
    let tx = conn.unchecked_transaction().expect("tx");
    {
        let mut stmt = tx
            .prepare(&format!("INSERT INTO {name} VALUES ({placeholders})"))
            .expect("prepare insert");
        for b in batches {
            for i in 0..b.num_rows() {
                let vals: Vec<Value> = b
                    .columns()
                    .iter()
                    .map(
                        |c| match ScalarValue::from_array(c.as_ref(), i).expect("scalar") {
                            v if v.is_null() => Value::Null,
                            ScalarValue::Boolean(Some(x)) => Value::Integer(i64::from(x)),
                            ScalarValue::Int64(Some(x)) => Value::Integer(x),
                            ScalarValue::Float64(Some(x)) => Value::Real(x),
                            ScalarValue::Utf8(Some(s)) => Value::Text(s),
                            ScalarValue::Date32(Some(d)) => Value::Text(format_date(d)),
                            _ => Value::Null,
                        },
                    )
                    .collect();
                stmt.execute(rusqlite::params_from_iter(vals))
                    .expect("insert");
            }
        }
    }
    tx.commit().expect("commit");
}

pub fn new_sqlite() -> Connection {
    let conn = Connection::open_in_memory().expect("sqlite");
    // quarry's LIKE is case sensitive (as in the SQL standard / PostgreSQL).
    conn.execute_batch("PRAGMA case_sensitive_like = ON;")
        .expect("pragma");
    conn
}

/// Translate quarry SQL to SQLite: DATE literals become strings, constant
/// `DATE ± INTERVAL` is pre-computed, EXTRACT uses strftime and
/// `substring(x FROM a FOR b)` becomes `substr(x, a, b)`.
pub fn to_sqlite_dialect(sql: &str) -> String {
    let mut s = fold_date_intervals(sql);
    s = rewrite_call(&s, "EXTRACT(YEAR FROM ", |inner| {
        format!("CAST(strftime('%Y', {inner}) AS INTEGER)")
    });
    s = rewrite_call(&s, "EXTRACT(MONTH FROM ", |inner| {
        format!("CAST(strftime('%m', {inner}) AS INTEGER)")
    });
    s = rewrite_call(&s, "substring(", |inner| {
        let inner = inner.replacen(" FROM ", ", ", 1).replacen(" FOR ", ", ", 1);
        format!("substr({inner})")
    });
    s.replace("DATE '", "'")
}

fn fold_date_intervals(sql: &str) -> String {
    let mut out = String::new();
    let mut rest = sql;
    while let Some(pos) = rest.find("DATE '") {
        out.push_str(&rest[..pos]);
        let after = &rest[pos + 6..];
        let end = after.find('\'').expect("closing quote");
        let date = &after[..end];
        let tail = &after[end + 1..];
        let trimmed = tail.trim_start();
        let sign = if trimmed.starts_with("+ INTERVAL '") {
            1
        } else if trimmed.starts_with("- INTERVAL '") {
            -1
        } else {
            0
        };
        if sign == 0 {
            out.push_str(&format!("'{date}'"));
            rest = tail;
            continue;
        }
        let iv = &trimmed["+ INTERVAL '".len()..];
        let q = iv.find('\'').expect("interval quote");
        let n: i32 = iv[..q].parse().expect("interval value");
        let unit_rest = iv[q + 1..].trim_start();
        let unit_len = unit_rest
            .find(|c: char| !c.is_ascii_alphabetic())
            .unwrap_or(unit_rest.len());
        let unit = &unit_rest[..unit_len];
        let d = parse_date(date).expect("date");
        let v = sign * n;
        let folded = match unit {
            "DAY" => add_months_days(d, 0, v),
            "MONTH" => add_months_days(d, v, 0),
            "YEAR" => add_months_days(d, 12 * v, 0),
            other => panic!("unknown unit {other}"),
        };
        out.push_str(&format!("'{}'", format_date(folded)));
        rest = &unit_rest[unit_len..];
    }
    out.push_str(rest);
    out
}

/// Replace `prefix ... )` (balanced parentheses) using `f(inner)`.
fn rewrite_call(sql: &str, prefix: &str, f: impl Fn(&str) -> String) -> String {
    let mut out = String::new();
    let mut rest = sql;
    while let Some(pos) = rest.find(prefix) {
        out.push_str(&rest[..pos]);
        let body = &rest[pos + prefix.len()..];
        let mut depth = 1;
        let mut end = 0;
        for (i, c) in body.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push_str(&f(&body[..end]));
        rest = &body[end + 1..];
    }
    out.push_str(rest);
    out
}

/// Engine plus SQLite loaded with TPC-H-like data at `sf`.
pub struct Fixture {
    pub engine: Engine,
    pub sqlite: Mutex<Connection>,
}

pub fn tpch_fixture() -> &'static Fixture {
    static F: OnceLock<Fixture> = OnceLock::new();
    F.get_or_init(|| {
        let tables = quarry::tpch::generate(0.01).expect("generate");
        let mut engine = Engine::new();
        let conn = new_sqlite();
        for t in &tables {
            engine
                .register_batches(t.name, Arc::clone(&t.schema), vec![t.batch.clone()])
                .expect("register");
            load_sqlite(&conn, t.name, &t.schema, std::slice::from_ref(&t.batch));
        }
        Fixture {
            engine,
            sqlite: Mutex::new(conn),
        }
    })
}

/// Run `sql` on both engines and compare.
pub fn check(fx: &Fixture, label: &str, sql: &str, ordered: bool) -> usize {
    let ours = fx
        .engine
        .sql(sql)
        .unwrap_or_else(|e| panic!("{label}: quarry failed: {e}\n{sql}"));
    let lite_sql = to_sqlite_dialect(sql);
    let theirs = {
        let conn = fx.sqlite.lock().unwrap_or_else(|e| e.into_inner());
        sqlite_rows(&conn, &lite_sql)
            .unwrap_or_else(|e| panic!("{label}: sqlite failed: {e}\n{lite_sql}"))
    };
    let n = theirs.len();
    compare(label, quarry_rows(&ours), theirs, ordered);
    n
}
