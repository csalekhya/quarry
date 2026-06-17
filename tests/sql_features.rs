//! Feature-by-feature differential tests against SQLite on small tables with
//! plenty of NULLs, empty strings, negative numbers and an empty table: the
//! cases where three-valued logic and aggregate semantics usually go wrong.

mod common;

use common::check;
use common::features::{fixture, tables};
use quarry::Engine;

/// (sql, ordered)
const QUERIES: &[(&str, bool)] = &[
    // projection and arithmetic
    ("SELECT * FROM t", false),
    ("SELECT id, a + b, a - b, a * b, a / b, a % b FROM t", false),
    ("SELECT id, f * 2, f / 0.5, -f, abs(f), abs(a), a + f FROM t", false),
    ("SELECT 1 + 2, 'x' || 'y', 7 / 2, 7 % 3, 1 / 0, NULL IS NULL, -7 / 2, -7 % 3", false),
    ("SELECT id, CAST(a AS DOUBLE) / 3, CAST(f AS BIGINT), CAST(a AS VARCHAR) FROM t", false),
    // predicates and three-valued logic
    ("SELECT id, a, b FROM t WHERE a > b", false),
    ("SELECT id FROM t WHERE a IS NULL", false),
    ("SELECT id FROM t WHERE a IS NOT NULL AND b IS NULL", false),
    ("SELECT id FROM t WHERE a BETWEEN 10 AND 50", false),
    ("SELECT id FROM t WHERE a NOT BETWEEN 10 AND 50", false),
    ("SELECT id FROM t WHERE a IN (1, 2, 3, NULL)", false),
    ("SELECT id FROM t WHERE a NOT IN (1, 2, 3)", false),
    ("SELECT id FROM t WHERE a NOT IN (1, NULL)", false),
    ("SELECT id, a > b, a = b OR a IS NULL, NOT (a < b), a <> b AND b > 10 FROM t", false),
    ("SELECT id FROM t WHERE NOT (a > 10 OR b < 20)", false),
    ("SELECT id FROM t WHERE s IS NULL OR s < 'c'", false),
    ("SELECT id FROM t WHERE a = (5 * 2) + 1", false),
    ("SELECT count(*) FROM t WHERE 1 = 0", false),
    ("SELECT count(*) FROM t WHERE 1 = 1 AND a > 3", false),
    ("SELECT id FROM t WHERE d > DATE '1995-06-01' AND d <= DATE '1995-06-01' + INTERVAL '2' MONTH", false),
    // strings
    ("SELECT id, s FROM t WHERE s LIKE 'a%'", false),
    ("SELECT id, s FROM t WHERE s NOT LIKE '%b_'", false),
    ("SELECT id, s FROM t WHERE s LIKE '_'", false),
    ("SELECT id, s || '-' || g FROM t", false),
    ("SELECT id, upper(s), lower(s), length(s), substr(s, 2, 3), substr(s, 2), substring(s FROM 1 FOR 2) FROM t", false),
    // conditional expressions
    ("SELECT id, coalesce(a, b, -1), coalesce(s, g) FROM t", false),
    ("SELECT id, CASE WHEN a > 50 THEN 'big' WHEN a > 10 THEN 'mid' ELSE 'small' END FROM t", false),
    ("SELECT id, CASE g WHEN 'x' THEN 1 WHEN 'y' THEN 2 END FROM t", false),
    ("SELECT id, CASE WHEN a IS NULL THEN f ELSE a END FROM t", false),
    ("SELECT sum(CASE WHEN a IS NULL THEN 1 ELSE 0 END) FROM t", false),
    // dates
    ("SELECT id, EXTRACT(YEAR FROM d), EXTRACT(MONTH FROM d) FROM t", false),
    ("SELECT min(d), max(d) FROM t WHERE g = 'x'", false),
    // aggregates
    ("SELECT count(*), count(a), sum(a), avg(a), min(a), max(a), min(s), max(s), sum(f), avg(f), min(f) FROM t", false),
    ("SELECT count(*), count(a), sum(a), avg(a), min(x) FROM e", false),
    ("SELECT g, count(*), sum(a), max(f), min(s) FROM t GROUP BY g", false),
    ("SELECT g, count(DISTINCT a), count(DISTINCT s), sum(DISTINCT b) FROM t GROUP BY g", false),
    ("SELECT g, sum(a) FROM t GROUP BY g HAVING count(*) > 40", false),
    ("SELECT g, b, count(*) FROM t GROUP BY g, b", false),
    ("SELECT a % 5, count(*) FROM t GROUP BY a % 5", false),
    ("SELECT g FROM t GROUP BY g HAVING g IS NOT NULL", false),
    ("SELECT max(a) - min(a), sum(a) / count(a) FROM t", false),
    ("SELECT x, count(*) FROM e GROUP BY x", false),
    ("SELECT g, sum(a) + 1, avg(b) * 2 FROM t GROUP BY g", false),
    ("SELECT DISTINCT g FROM t", false),
    ("SELECT DISTINCT g, b > 50 FROM t", false),
    // ordering and limits (all orders are total)
    ("SELECT a, b FROM t ORDER BY a, b, id", true),
    ("SELECT a FROM t ORDER BY a DESC, id", true),
    ("SELECT a, s FROM t ORDER BY s DESC, a, id LIMIT 7 OFFSET 3", true),
    ("SELECT id FROM t ORDER BY f * -1, id LIMIT 10", true),
    ("SELECT a AS x, id FROM t ORDER BY x, id", true),
    ("SELECT g, sum(a) AS total FROM t GROUP BY g ORDER BY total DESC, g", true),
    ("SELECT id FROM t ORDER BY 1 DESC LIMIT 5", true),
    ("SELECT id, a FROM t ORDER BY a NULLS LAST, id LIMIT 20", true),
    ("SELECT id FROM t LIMIT 0", true),
    ("SELECT DISTINCT g FROM t ORDER BY g", true),
    // joins
    ("SELECT t.id, u.x FROM t JOIN u ON t.a = u.a", false),
    ("SELECT t.id, u.x FROM t LEFT JOIN u ON t.a = u.a", false),
    ("SELECT t.id, u.x FROM t LEFT JOIN u ON t.a = u.a AND u.x LIKE 'a%'", false),
    ("SELECT t.id, u.x FROM t LEFT JOIN u ON t.a = u.a AND t.b > 50", false),
    ("SELECT t.id FROM t LEFT JOIN u ON t.a = u.a WHERE u.x IS NULL", false),
    ("SELECT t.id, u.x FROM t LEFT JOIN u ON t.a = u.a WHERE u.x > 'c'", false),
    ("SELECT t.id, u.id FROM t JOIN u ON t.a < u.a AND t.b = u.id", false),
    ("SELECT t.id, u.id FROM t JOIN u ON t.a + 1 = u.a", false),
    ("SELECT t.id, u.id FROM t JOIN u ON t.a = u.a AND t.g = u.x", false),
    ("SELECT count(*) FROM t, u", false),
    ("SELECT count(*) FROM t, u WHERE t.b > u.a", false),
    ("SELECT t.id, u.id FROM t LEFT JOIN u ON t.b > u.a + 90", false),
    ("SELECT t.id, e.id FROM t LEFT JOIN e ON t.a = e.a", false),
    ("SELECT count(*) FROM t JOIN e ON t.a = e.a", false),
    ("SELECT t1.id, t2.id FROM t t1 JOIN t t2 ON t1.a = t2.b WHERE t1.id < t2.id", false),
    ("SELECT t.id, u.id, v.id FROM t JOIN u ON t.a = u.a JOIN u v ON v.a = t.b", false),
    ("SELECT u.x, count(t.id) FROM u LEFT JOIN t ON t.a = u.a GROUP BY u.x", false),
    // signed zero: f * 0 is -0.0 for negative f, and SQL treats -0.0 = 0.0
    ("SELECT 0.0 = -0.0, 0.0 <> -0.0, -0.0 < 0.0, 0.0 > -0.0, -0.0 >= 0.0, -0.0 IN (0.0)", false),
    ("SELECT count(*) FROM t WHERE f * 0 = 0.0", false),
    ("SELECT count(*) FROM t WHERE -(f * 0) = 0", false),
    ("SELECT count(*) FROM t WHERE f * 0 < 0.0 OR -(f * 0) > 0.0", false),
    ("SELECT count(*) FROM t WHERE f * 0 <= 0.0 AND f * 0 >= -0.0", false),
    ("SELECT count(*) FROM t WHERE f * 0 IN (0.0, 1.5)", false),
    ("SELECT count(*) FROM t WHERE f * 0 NOT IN (0.0)", false),
    ("SELECT f * 0, count(*) FROM t GROUP BY f * 0", false),
    ("SELECT DISTINCT f * 0 FROM t", false),
    ("SELECT DISTINCT -(f * 0), g FROM t", false),
    ("SELECT g, count(DISTINCT f * 0), count(DISTINCT f) FROM t GROUP BY g", false),
    ("SELECT count(*) FROM t t1 JOIN t t2 ON t1.f * 0 = -(t2.f * 0) WHERE t1.id < 40 AND t2.id < 40", false),
    ("SELECT t1.id, t2.id FROM t t1 LEFT JOIN t t2 ON t1.f * 0 = t2.f * 0 AND t2.id < 4 WHERE t1.id < 30", false),
    ("SELECT id FROM t WHERE f IS NOT NULL ORDER BY f * 0, id LIMIT 30", true),
    ("SELECT id FROM t WHERE f IS NOT NULL ORDER BY -(f * 0) DESC, id LIMIT 30", true),
    // derived tables and CTEs
    ("SELECT * FROM (SELECT g, count(*) AS c FROM t GROUP BY g) AS x WHERE c > 10", false),
    ("SELECT x.g, x.c FROM (SELECT g, count(*) AS c FROM t GROUP BY g) x JOIN (SELECT g, max(a) AS m FROM t GROUP BY g) y ON x.g = y.g WHERE y.m > 0", false),
    ("WITH big AS (SELECT * FROM t WHERE a > 50) SELECT b1.id, b2.id FROM big b1 JOIN big b2 ON b1.b = b2.b", false),
    ("WITH c1 AS (SELECT a FROM t), c2 AS (SELECT a FROM c1 WHERE a > 3) SELECT count(*) FROM c2", false),
    ("WITH s AS (SELECT g, sum(a) AS total FROM t GROUP BY g) SELECT g FROM s WHERE total = (5 - 5) + total", false),
    ("SELECT n FROM (SELECT a + b AS n FROM t) AS q WHERE n > 100", false),
];

#[test]
fn features_match_sqlite() {
    let fx = fixture();
    let mut failures = Vec::new();
    for (i, (sql, ordered)) in QUERIES.iter().enumerate() {
        let label = format!("feature #{i}: {sql}");
        if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check(fx, &label, sql, *ordered)
        })) {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            failures.push(msg);
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} failed:\n{}",
        failures.len(),
        QUERIES.len(),
        failures.join("\n\n")
    );
}

/// The same answers must come back with every optimizer rule disabled: the
/// rules are optimizations, never semantics.
#[test]
fn optimizer_rules_do_not_change_results() {
    let fx = fixture();
    for rule in quarry::optimizer::RULE_NAMES {
        let mut plain = Engine::new();
        plain.set_batch_size(64);
        for (name, schema, batch) in tables() {
            plain.register_batches(name, schema, vec![batch]).unwrap();
        }
        plain.set_disabled_rules(vec![rule.to_string()]).unwrap();
        for (sql, ordered) in QUERIES {
            let a = common::quarry_rows(&fx.engine.sql(sql).unwrap());
            let b = common::quarry_rows(
                &plain
                    .sql(sql)
                    .unwrap_or_else(|e| panic!("{rule}: {sql}: {e}")),
            );
            common::compare(&format!("without {rule}: {sql}"), a, b, *ordered);
        }
    }
}

/// Twelve relations exceed the DP limit, so this exercises the greedy orderer.
#[test]
fn wide_join_uses_greedy_ordering() {
    let fx = fixture();
    let mut sql = String::from("SELECT count(*), sum(u1.a), min(u12.x) FROM u u1");
    for i in 2..=12 {
        sql.push_str(&format!(" JOIN u u{i} ON u{}.id = u{i}.id", i - 1));
    }
    sql.push_str(" WHERE u1.a > 10");
    check(fx, "12-way join", &sql, false);
    let plan = fx.engine.optimized_plan(&sql).unwrap().to_string();
    assert_eq!(plan.matches("InnerJoin").count(), 11, "{plan}");
    assert!(!plan.contains("CrossJoin"), "{plan}");
}

#[test]
fn quarry_only_features() {
    let fx = fixture();
    let rows = |sql: &str| common::quarry_rows(&fx.engine.sql(sql).unwrap());
    // Derived-table column aliases.
    assert_eq!(
        rows("SELECT k FROM (SELECT g FROM t WHERE g = 'x' LIMIT 1) AS d (k)").len(),
        1
    );
    // Date arithmetic on columns, checked against an equivalent day-count predicate.
    let a = rows("SELECT count(*) FROM t WHERE d + INTERVAL '10' DAY > DATE '1995-12-01'");
    let b = rows("SELECT count(*) FROM t WHERE d > DATE '1995-11-21'");
    common::compare("date add", a, b, true);
    // Month arithmetic clamps to month end.
    let r = fx
        .engine
        .sql("SELECT DATE '2024-01-31' + INTERVAL '1' MONTH AS d")
        .unwrap();
    assert!(r.to_string().contains("2024-02-29"), "{r}");
    // round()
    let r = rows("SELECT round(2.5), round(-2.5), round(1.2345, 2)");
    common::compare(
        "round",
        r,
        vec![vec![
            common::V::Float(3.0),
            common::V::Float(-3.0),
            common::V::Float(1.23),
        ]],
        true,
    );
    // count(*) over a zero-column projection.
    assert_eq!(
        rows("SELECT count(*) FROM (SELECT 1 AS one FROM t) AS q")[0].len(),
        1
    );
}
