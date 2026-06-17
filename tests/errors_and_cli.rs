//! Error reporting through the public API and the command-line interface.

use std::process::Command;
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use quarry::{Engine, QuarryError};

fn engine() -> Engine {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(vec![
            Some(i64::MAX),
            Some(1),
            None,
        ]))],
    )
    .unwrap();
    let mut e = Engine::new();
    e.register_batches("t", schema, vec![batch]).unwrap();
    e
}

#[test]
fn errors_name_the_stage_and_position() {
    let e = engine();
    match e.sql("SELECT a\nFROM t WHERE a >") {
        Err(QuarryError::Parse { line, column, .. }) => assert_eq!((line, column), (2, 17)),
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        e.sql("SELECT b FROM t"),
        Err(QuarryError::Bind(_))
    ));
    assert!(matches!(
        e.sql("SELECT a || 1 > 'x' FROM t WHERE a + 'x' = 1"),
        Err(QuarryError::Type(_))
    ));
    assert!(matches!(
        e.sql("SELECT * FROM t RIGHT JOIN t AS u ON true"),
        Err(QuarryError::Unsupported(_))
    ));
    let msg = e.sql("SELECT nope FROM t").unwrap_err().to_string();
    assert_eq!(msg, "bind error: column 'nope' not found");
}

#[test]
fn runtime_errors_surface_from_kernels() {
    let e = engine();
    let err = e.sql("SELECT a + 1 FROM t").unwrap_err();
    assert!(err.to_string().contains("Overflow"), "{err}");
    let err = e.sql("SELECT sum(a) FROM t").unwrap_err();
    assert!(
        err.to_string().contains("integer overflow in sum()"),
        "{err}"
    );
    // Division by zero is NULL, not an error.
    let r = e.sql("SELECT a / 0 FROM t").unwrap();
    assert_eq!(r.num_rows(), 3);
    // An overflowing constant is not folded away at plan time; it fails at runtime.
    assert!(e.sql("SELECT 9223372036854775807 + 1").is_err());
    // ... but only if a row reaches it.
    assert_eq!(
        e.sql("SELECT 9223372036854775807 + 1 FROM t WHERE a = 42")
            .unwrap()
            .num_rows(),
        0
    );
}

#[test]
fn explain_variants_render() {
    let e = engine();
    let text = e
        .sql("EXPLAIN SELECT a FROM t WHERE a > 1 + 1")
        .unwrap()
        .to_string();
    assert!(text.contains("== Optimized Logical Plan =="), "{text}");
    assert!(text.contains("filters=[a > 2]"), "{text}");
    assert!(text.contains("== Physical Plan =="), "{text}");
    let text = e
        .sql("EXPLAIN ANALYZE SELECT count(*) FROM t")
        .unwrap()
        .to_string();
    assert!(text.contains("rows=1 batches=1"), "{text}");
    assert!(text.contains("== Execution: 1 rows"), "{text}");
    let text = e
        .sql("EXPLAIN VERBOSE SELECT a FROM t ORDER BY a LIMIT 1")
        .unwrap()
        .to_string();
    for rule in quarry::optimizer::RULE_NAMES {
        assert!(text.contains(&format!("== After {rule} ==")), "{text}");
    }
    assert!(text.contains("TopK: fetch=1"), "{text}");
}

#[test]
fn disabling_unknown_rule_fails() {
    let mut e = engine();
    assert!(e.set_disabled_rules(vec!["no_such_rule".into()]).is_err());
    e.set_disabled_rules(vec!["predicate_pushdown".into()])
        .unwrap();
    let text = e
        .explain("SELECT a FROM t WHERE a = 1", false, false)
        .unwrap();
    assert!(text.contains("Filter: a = 1"), "{text}");
}

#[test]
fn cli_generates_and_queries_data() {
    let bin = env!("CARGO_BIN_EXE_quarry");
    let dir = tempfile::tempdir().unwrap();
    let out = Command::new(bin)
        .args(["gen", "--sf", "0.001", "--format", "csv", "--out"])
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("lineitem"));
    let out = Command::new(bin)
        .args(["query", "--data"])
        .arg(dir.path())
        .arg("SELECT count(*) AS n FROM nation")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("25") && stdout.contains("(1 row)"),
        "{stdout}"
    );

    let sql_file = dir.path().join("q.sql");
    std::fs::write(&sql_file, "EXPLAIN SELECT r_name FROM region").unwrap();
    let out = Command::new(bin)
        .args(["query", "--table"])
        .arg(format!(
            "region={}",
            dir.path().join("region.csv").display()
        ))
        .arg("-f")
        .arg(&sql_file)
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&out.stdout).contains("ScanExec: region (csv)"));

    let out = Command::new(bin)
        .args(["query", "SELECT x FROM nowhere"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("does not exist"));
    let out = Command::new(bin)
        .args(["query", "--table", "oops"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}
