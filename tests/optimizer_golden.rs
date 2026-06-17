//! Golden tests for the optimizer: the optimized logical plan (with
//! cardinality estimates) and the physical plan of every TPC-H-like query
//! and of focused rewrite examples are compared to files in `tests/golden/`.
//!
//! Regenerate after an intentional planner change with
//! `UPDATE_GOLDEN=1 cargo test --test optimizer_golden` and review the diff.

mod common;

use std::path::PathBuf;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn check_golden(name: &str, sql: &str, verbose: bool) {
    let engine = &common::tpch_fixture().engine;
    let actual = engine
        .explain(sql, false, verbose)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let actual = format!("-- {}\n{actual}", sql.trim().replace('\n', "\n-- "));
    let path = golden_dir().join(format!("{name}.txt"));
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::create_dir_all(golden_dir()).unwrap();
        std::fs::write(&path, &actual).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("missing {}; run with UPDATE_GOLDEN=1", path.display()));
    if expected != actual {
        panic!(
            "{name}: plan changed (UPDATE_GOLDEN=1 to accept)\n--- expected\n{expected}\n--- actual\n{actual}"
        );
    }
}

fn tpch(n: u32) {
    let path = format!("{}/queries/tpch/q{n:02}.sql", env!("CARGO_MANIFEST_DIR"));
    let sql = std::fs::read_to_string(path).unwrap();
    let body: String = sql
        .lines()
        .filter(|l| !l.starts_with("--"))
        .collect::<Vec<_>>()
        .join("\n");
    check_golden(&format!("tpch_q{n:02}"), &body, false);
}

macro_rules! golden_tpch {
    ($($name:ident => $n:expr),* $(,)?) => {
        $(#[test] fn $name() { tpch($n); })*
    };
}

golden_tpch! {
    tpch_q01 => 1, tpch_q02 => 2, tpch_q03 => 3, tpch_q04 => 4, tpch_q05 => 5, tpch_q06 => 6,
    tpch_q07 => 7, tpch_q08 => 8, tpch_q09 => 9, tpch_q10 => 10, tpch_q11 => 11, tpch_q12 => 12,
    tpch_q13 => 13, tpch_q14 => 14, tpch_q15 => 15, tpch_q16 => 16, tpch_q17 => 17, tpch_q18 => 18,
    tpch_q19 => 19, tpch_q20 => 20, tpch_q21 => 21, tpch_q22 => 22,
}

#[test]
fn rule_constant_folding() {
    check_golden(
        "rule_constant_folding",
        "SELECT o_orderkey FROM orders WHERE o_orderdate < DATE '1994-01-01' + INTERVAL '1' YEAR AND 1 = 1 AND o_totalprice > 10 * 100",
        false,
    );
}

#[test]
fn rule_contradiction_becomes_empty() {
    check_golden(
        "rule_contradiction",
        "SELECT count(*) FROM orders WHERE 1 = 0",
        false,
    );
}

#[test]
fn rule_left_join_to_inner() {
    check_golden(
        "rule_left_to_inner",
        "SELECT c_name, o_orderkey FROM customer LEFT JOIN orders ON c_custkey = o_custkey WHERE o_totalprice > 1000",
        false,
    );
}

#[test]
fn rule_left_join_kept_for_is_null() {
    check_golden(
        "rule_left_anti",
        "SELECT c_name FROM customer LEFT JOIN orders ON c_custkey = o_custkey WHERE o_orderkey IS NULL",
        false,
    );
}

#[test]
fn rule_pushdown_through_aggregate_and_projection() {
    check_golden(
        "rule_pushdown_aggregate",
        "SELECT * FROM (SELECT o_custkey, count(*) AS n FROM orders GROUP BY o_custkey) AS c WHERE o_custkey < 100 AND n > 3",
        false,
    );
}

#[test]
fn rule_top_k() {
    check_golden(
        "rule_top_k",
        "SELECT o_orderkey, o_totalprice FROM orders ORDER BY o_totalprice DESC LIMIT 5 OFFSET 2",
        false,
    );
}

#[test]
fn rule_join_reorder_trace() {
    // EXPLAIN VERBOSE shows the plan after every rule, from the binder's
    // left-deep cross products to the reordered hash joins.
    let sql = std::fs::read_to_string(format!(
        "{}/queries/tpch/q05.sql",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let body: String = sql
        .lines()
        .filter(|l| !l.starts_with("--"))
        .collect::<Vec<_>>()
        .join("\n");
    check_golden("verbose_q05", &body, true);
}

/// Structural properties that must hold regardless of exact estimates.
#[test]
fn no_cross_products_in_tpch_plans() {
    let engine = &common::tpch_fixture().engine;
    for n in 1..=22 {
        let path = format!("{}/queries/tpch/q{n:02}.sql", env!("CARGO_MANIFEST_DIR"));
        let sql = std::fs::read_to_string(path).unwrap();
        let plan = engine.optimized_plan(&sql).unwrap().to_string();
        // Q11 and Q22 join against a one-row derived table on a non-equi
        // predicate (the rewritten scalar subquery); Q7 joins the two
        // 25-row nation aliases on its OR predicate first. Those are the only
        // nested-loop joins, and all of them have a tiny input.
        let allowed = [7, 11, 22];
        if !allowed.contains(&n) {
            assert!(
                !plan.contains("CrossJoin") && !plan.contains("Join: filter"),
                "q{n:02} has a cross product:\n{plan}"
            );
        }
    }
}

#[test]
fn build_side_is_the_smaller_input() {
    let engine = &common::tpch_fixture().engine;
    let plan = engine
        .optimized_plan("SELECT count(*) FROM lineitem, orders WHERE l_orderkey = o_orderkey")
        .unwrap();
    let text = plan.to_string();
    let join_line = text.lines().position(|l| l.contains("InnerJoin")).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    // Left (probe) child is printed first: lineitem; right (build): orders.
    assert!(lines[join_line + 1].contains("lineitem"), "{text}");
    assert!(lines[join_line + 2].contains("orders"), "{text}");
}
