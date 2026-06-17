//! CSV and Parquet scans: the same data read from memory, CSV and Parquet
//! must give the same answers, Parquet row groups must be pruned by pushed
//! filters, and CSV schema inference must handle NULLs and dates.

mod common;

use std::path::Path;
use std::sync::{Arc, OnceLock};

use quarry::tpch::{generate, write_tables, FileFormat};
use quarry::types::ScalarValue;
use quarry::{CsvOptions, Engine, QuarryError};

struct Engines {
    memory: Engine,
    csv: Engine,
    parquet: Engine,
    _dir: tempfile::TempDir,
}

fn engines() -> &'static Engines {
    static E: OnceLock<Engines> = OnceLock::new();
    E.get_or_init(|| {
        let tables = generate(0.003).unwrap();
        let dir = tempfile::tempdir().unwrap();
        write_tables(&tables, &dir.path().join("csv"), FileFormat::Csv, 1024).unwrap();
        write_tables(
            &tables,
            &dir.path().join("parquet"),
            FileFormat::Parquet,
            1024,
        )
        .unwrap();
        let mut memory = Engine::new();
        for t in &tables {
            memory
                .register_batches(t.name, Arc::clone(&t.schema), vec![t.batch.clone()])
                .unwrap();
        }
        let mut csv = Engine::new();
        let names = csv.register_dir(dir.path().join("csv")).unwrap();
        assert_eq!(names.len(), 8);
        let mut parquet = Engine::new();
        parquet.set_batch_size(700);
        parquet.register_dir(dir.path().join("parquet")).unwrap();
        Engines {
            memory,
            csv,
            parquet,
            _dir: dir,
        }
    })
}

fn query(n: u32) -> String {
    std::fs::read_to_string(format!(
        "{}/queries/tpch/q{n:02}.sql",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

#[test]
fn csv_and_parquet_match_memory_on_all_tpch_queries() {
    let e = engines();
    for n in 1..=22 {
        let sql = query(n);
        let expected = common::quarry_rows(&e.memory.sql(&sql).unwrap());
        let from_csv = common::quarry_rows(
            &e.csv
                .sql(&sql)
                .unwrap_or_else(|err| panic!("csv q{n}: {err}")),
        );
        let from_parquet = common::quarry_rows(
            &e.parquet
                .sql(&sql)
                .unwrap_or_else(|err| panic!("parquet q{n}: {err}")),
        );
        common::compare(&format!("csv q{n:02}"), from_csv, expected.clone(), true);
        common::compare(&format!("parquet q{n:02}"), from_parquet, expected, true);
    }
}

#[test]
fn catalog_statistics_match_across_sources() {
    let e = engines();
    for name in ["lineitem", "orders", "customer"] {
        let m = e.memory.catalog().get(name).unwrap();
        let c = e.csv.catalog().get(name).unwrap();
        let p = e.parquet.catalog().get(name).unwrap();
        assert_eq!(m.stats.row_count, c.stats.row_count);
        assert_eq!(m.stats.row_count, p.stats.row_count);
        for i in 0..m.schema.fields().len() {
            assert_eq!(
                m.stats.columns[i].ndv, p.stats.columns[i].ndv,
                "{name} column {i}"
            );
            assert_eq!(
                m.stats.columns[i].ndv, c.stats.columns[i].ndv,
                "{name} column {i}"
            );
        }
    }
}

#[test]
fn parquet_row_groups_are_pruned_by_date_filters() {
    let e = engines();
    let text = e.parquet.explain(&query(6), true, false).unwrap();
    let scan = text
        .lines()
        .find(|l| l.contains("ScanExec: lineitem (parquet)") && l.contains("row_groups="))
        .unwrap_or_else(|| panic!("no lineitem scan:\n{text}"));
    let field = |k: &str| -> usize {
        let start = scan.find(&format!("{k}=")).unwrap() + k.len() + 1;
        scan[start..]
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    let (groups, pruned, read) = (
        field("row_groups"),
        field("row_groups_pruned"),
        field("rows_read"),
    );
    let total = e.parquet.catalog().get("lineitem").unwrap().stats.row_count as usize;
    // lineitem.parquet is clustered by l_shipdate; a one-year filter keeps ~1/7.
    assert!(
        pruned * 2 > groups,
        "only {pruned} of {groups} row groups pruned:\n{scan}"
    );
    assert!(read * 2 < total, "read {read} of {total} rows:\n{scan}");
}

#[test]
fn csv_inference_handles_nulls_and_dates() {
    let mut engine = Engine::new();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    engine.register_dir(&fixtures).unwrap();
    let emp = engine.catalog().get("employees").unwrap();
    let types: Vec<String> = emp
        .schema
        .fields()
        .iter()
        .map(|f| f.data_type().to_string())
        .collect();
    assert_eq!(
        types,
        ["Int64", "Utf8", "Int64", "Float64", "Date32", "Int64"]
    );
    assert_eq!(emp.stats.columns[3].null_count, 1);
    let r = engine
        .sql(
            "SELECT d.name, count(e.id) AS staff, avg(e.salary) AS avg_salary, min(e.hired) AS first_hire
             FROM departments d LEFT JOIN employees e ON e.dept_id = d.id
             GROUP BY d.name ORDER BY d.name",
        )
        .unwrap()
        .rows()
        .unwrap();
    assert_eq!(r.len(), 5);
    assert_eq!(r[0][0], ScalarValue::Utf8(Some("Engineering".into())));
    assert_eq!(r[0][1], ScalarValue::Int64(Some(4)));
    // Legal has no employees: count 0, avg NULL.
    assert_eq!(r[1][0], ScalarValue::Utf8(Some("Legal".into())));
    assert_eq!(r[1][1], ScalarValue::Int64(Some(0)));
    assert!(r[1][2].is_null());
    // Research: Ken's NULL salary is ignored by avg.
    assert_eq!(r[2][2], ScalarValue::Float64(Some(9000.0)));
    let nulls = engine
        .sql("SELECT count(*) FROM departments WHERE location IS NULL")
        .unwrap()
        .rows()
        .unwrap();
    assert_eq!(nulls[0][0], ScalarValue::Int64(Some(1)));
    // Self join through a manager hierarchy.
    let pairs = engine
        .sql("SELECT e.name, m.name FROM employees e JOIN employees m ON e.manager_id = m.id WHERE m.name = 'Ada'")
        .unwrap();
    assert_eq!(pairs.num_rows(), 3);
}

#[test]
fn explicit_csv_schema_and_delimiter() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.tsv");
    std::fs::write(&path, "1\tx\n2\ty\n3\t\n").unwrap();
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("K", arrow::datatypes::DataType::Int32, true),
        arrow::datatypes::Field::new("v", arrow::datatypes::DataType::Utf8, true),
    ]));
    let mut engine = Engine::new();
    engine
        .register_csv(
            "t",
            &path,
            CsvOptions {
                has_header: false,
                delimiter: b'\t',
                schema: Some(schema),
            },
        )
        .unwrap();
    let r = engine
        .sql("SELECT sum(k), count(v) FROM t")
        .unwrap()
        .rows()
        .unwrap();
    assert_eq!(
        r[0],
        vec![ScalarValue::Int64(Some(6)), ScalarValue::Int64(Some(2))]
    );
}

#[test]
fn registration_errors() {
    let mut engine = Engine::new();
    assert!(matches!(
        engine.register_csv("x", "/nonexistent/file.csv", CsvOptions::default()),
        Err(QuarryError::Catalog(_))
    ));
    assert!(engine
        .register_parquet("x", "/nonexistent/file.parquet")
        .is_err());
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("a", arrow::datatypes::DataType::Int64, true),
    ]));
    assert!(matches!(
        engine.register_batches("bad-name", schema.clone(), vec![]),
        Err(QuarryError::Catalog(_))
    ));
    let bin = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("b", arrow::datatypes::DataType::Binary, true),
    ]));
    assert!(matches!(
        engine.register_batches("b", bin, vec![]),
        Err(QuarryError::Catalog(_))
    ));
    engine.register_batches("empty", schema, vec![]).unwrap();
    let r = engine
        .sql("SELECT count(*), sum(a) FROM empty")
        .unwrap()
        .rows()
        .unwrap();
    assert_eq!(r[0][0], ScalarValue::Int64(Some(0)));
    assert!(r[0][1].is_null());
}
