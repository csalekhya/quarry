//! Differential fuzzing: random queries over the NULL-heavy feature tables are
//! run on quarry and SQLite and must agree.
//!
//! Each proptest case is a seed for a small recursive query generator. The
//! generator sticks to constructs whose semantics the two engines share
//! (integer arithmetic, comparisons, NULL logic, LIKE, IN, BETWEEN, CASE,
//! COALESCE, grouping, inner/left joins), plus a DOUBLE expression family
//! that produces signed zeros (`f * 0`, unary minus) and feeds them to
//! comparisons, IN lists, GROUP BY, DISTINCT, ORDER BY and join keys. Its
//! operands are multiples of 0.25 and its products stay shallow, so every
//! value is exact in binary floating point and comparisons cannot flake on
//! rounding. Set `PROPTEST_CASES` to run more.

mod common;

use proptest::prelude::*;
use quarry::tpch::Rng;

struct Gen {
    rng: Rng,
    /// Column prefix: "" for single-table queries, "t." in joins.
    q: &'static str,
}

impl Gen {
    fn chance(&mut self, pct: i64) -> bool {
        self.rng.range(0, 99) < pct
    }

    fn int_col(&mut self) -> String {
        let c = ["a", "b", "id"][self.rng.range(0, 2) as usize];
        format!("{}{c}", self.q)
    }

    fn num(&mut self, depth: u32) -> String {
        let q = self.q;
        if depth == 0 || self.chance(35) {
            return if self.chance(65) {
                self.int_col()
            } else {
                self.rng.range(-5, 60).to_string()
            };
        }
        match self.rng.range(0, 6) {
            0 => {
                let op = ["+", "-", "*"][self.rng.range(0, 2) as usize];
                format!("({} {op} {})", self.num(depth - 1), self.num(depth - 1))
            }
            1 => {
                let op = ["/", "%"][self.rng.range(0, 1) as usize];
                format!("({} {op} {})", self.num(depth - 1), self.num(depth - 1))
            }
            2 => format!("abs({})", self.num(depth - 1)),
            3 => format!("coalesce({}, {})", self.num(depth - 1), self.num(depth - 1)),
            4 => format!(
                "CASE WHEN {} THEN {} ELSE {} END",
                self.pred(depth - 1),
                self.num(depth - 1),
                self.num(depth - 1)
            ),
            5 => format!("CAST({}f * 4 AS BIGINT)", q),
            _ => format!("(- {})", self.num(depth - 1)),
        }
    }

    /// A DOUBLE expression that often evaluates to -0.0.
    fn fnum(&mut self, depth: u32) -> String {
        let q = self.q;
        if depth == 0 || self.chance(40) {
            return match self.rng.range(0, 4) {
                0 => format!("{q}f"),
                1 => format!("({q}f * 0)"),
                2 => format!("(- ({q}f * 0))"),
                3 => format!("({q}f * -1)"),
                _ => ["0.0", "-0.0", "0.25", "-1.5", "2.0"][self.rng.range(0, 4) as usize]
                    .to_string(),
            };
        }
        match self.rng.range(0, 4) {
            0 => {
                let op = ["+", "-", "*"][self.rng.range(0, 2) as usize];
                format!("({} {op} {})", self.fnum(depth - 1), self.fnum(depth - 1))
            }
            1 => format!("abs({})", self.fnum(depth - 1)),
            2 => format!(
                "coalesce({}, {})",
                self.fnum(depth - 1),
                self.fnum(depth - 1)
            ),
            3 => format!("({} * 0)", self.fnum(depth - 1)),
            _ => format!("(- {})", self.fnum(depth - 1)),
        }
    }

    fn pred(&mut self, depth: u32) -> String {
        let q = self.q;
        let leaf = depth == 0 || self.chance(40);
        if leaf {
            return match self.rng.range(0, 9) {
                0 => {
                    let op = ["=", "<>", "<", "<=", ">", ">="][self.rng.range(0, 5) as usize];
                    format!("{} {op} {}", self.num(1), self.num(1))
                }
                7 => {
                    let op = ["=", "<>", "<", "<=", ">", ">="][self.rng.range(0, 5) as usize];
                    format!("{} {op} {}", self.fnum(1), self.fnum(1))
                }
                8 => format!(
                    "{} {}IN ({}, {})",
                    self.fnum(1),
                    if self.chance(30) { "NOT " } else { "" },
                    self.fnum(0),
                    self.fnum(0)
                ),
                1 => format!(
                    "{}{} IS {}NULL",
                    q,
                    ["a", "b", "s", "g", "f"][self.rng.range(0, 4) as usize],
                    if self.chance(50) { "NOT " } else { "" }
                ),
                2 => {
                    let lo = self.rng.range(-10, 50);
                    let hi = lo + self.rng.range(0, 40);
                    format!(
                        "{} {}BETWEEN {lo} AND {hi}",
                        self.num(1),
                        if self.chance(30) { "NOT " } else { "" }
                    )
                }
                3 => {
                    let n = self.rng.range(1, 4);
                    let items: Vec<String> = (0..n)
                        .map(|_| {
                            if self.chance(15) {
                                "NULL".to_string()
                            } else {
                                self.rng.range(-5, 60).to_string()
                            }
                        })
                        .collect();
                    format!(
                        "{} {}IN ({})",
                        self.int_col(),
                        if self.chance(30) { "NOT " } else { "" },
                        items.join(", ")
                    )
                }
                4 => {
                    let pat =
                        ["a%", "%a%", "_", "%e", "b_c", "", "%"][self.rng.range(0, 6) as usize];
                    format!(
                        "{}s {}LIKE '{pat}'",
                        q,
                        if self.chance(30) { "NOT " } else { "" }
                    )
                }
                5 => format!(
                    "{}g = '{}'",
                    q,
                    ["x", "y", "z"][self.rng.range(0, 2) as usize]
                ),
                6 => format!("{}f > {}", q, self.rng.range(-8, 8)),
                _ => format!("{}f * 0 = 0", q),
            };
        }
        match self.rng.range(0, 2) {
            0 => format!("({} AND {})", self.pred(depth - 1), self.pred(depth - 1)),
            1 => format!("({} OR {})", self.pred(depth - 1), self.pred(depth - 1)),
            _ => format!("NOT ({})", self.pred(depth - 1)),
        }
    }

    fn query(&mut self) -> (String, bool) {
        match self.rng.range(0, 5) {
            0 => {
                self.q = "";
                let order = self.chance(30);
                let mut sql = format!(
                    "SELECT id, {}, {} FROM t WHERE {}",
                    self.num(3),
                    self.pred(2),
                    self.pred(3)
                );
                if order {
                    sql.push_str(&format!(" ORDER BY 2, id LIMIT {}", self.rng.range(0, 40)));
                }
                (sql, order)
            }
            1 => {
                self.q = "";
                let agg = ["sum", "min", "max", "count", "avg"][self.rng.range(0, 4) as usize];
                let distinct = if agg != "avg" && self.chance(20) {
                    "DISTINCT "
                } else {
                    ""
                };
                let mut sql = format!(
                    "SELECT g, {agg}({distinct}{}), count(*), sum({}) FROM t WHERE {} GROUP BY g",
                    self.num(2),
                    self.num(1),
                    self.pred(2)
                );
                if self.chance(40) {
                    sql.push_str(&format!(" HAVING count(*) > {}", self.rng.range(0, 60)));
                }
                (sql, false)
            }
            2 => {
                self.q = "t.";
                let kind = if self.chance(50) { "LEFT JOIN" } else { "JOIN" };
                let extra = if self.chance(40) {
                    format!(" AND u.a > {}", self.rng.range(-10, 60))
                } else {
                    String::new()
                };
                let sql = format!(
                    "SELECT t.id, u.id, u.x FROM t {kind} u ON t.{} = u.a{extra} WHERE {}",
                    ["a", "b"][self.rng.range(0, 1) as usize],
                    self.pred(2)
                );
                (sql, false)
            }
            3 => {
                self.q = "";
                let sql = format!(
                    "SELECT k, count(*) FROM (SELECT {} AS k FROM t WHERE {}) AS d GROUP BY k",
                    self.num(2),
                    self.pred(2)
                );
                (sql, false)
            }
            4 => {
                // Signed zeros as GROUP BY / DISTINCT keys and sort keys.
                self.q = "";
                let sql = match self.rng.range(0, 2) {
                    0 => format!(
                        "SELECT {} AS k, count(*), count(DISTINCT {}) FROM t WHERE {} GROUP BY k",
                        self.fnum(2),
                        self.fnum(1),
                        self.pred(1)
                    ),
                    1 => format!(
                        "SELECT DISTINCT {} FROM t WHERE {}",
                        self.fnum(2),
                        self.pred(1)
                    ),
                    _ => {
                        let sql = format!(
                            "SELECT id, {} AS k FROM t WHERE f IS NOT NULL AND {} ORDER BY k{}, id LIMIT {}",
                            self.fnum(2),
                            self.pred(1),
                            if self.chance(50) { " DESC" } else { "" },
                            self.rng.range(1, 40)
                        );
                        return (sql, true);
                    }
                };
                (sql, false)
            }
            _ => {
                // Signed zeros as equi-join keys.
                self.q = "t.";
                let left = self.fnum(1);
                self.q = "t2.";
                let right = self.fnum(1);
                let kind = if self.chance(50) { "LEFT JOIN" } else { "JOIN" };
                let sql = format!(
                    "SELECT t.id, t2.id FROM t {kind} t t2 ON {left} = {right} AND t2.id < 25 WHERE t.id < 60"
                );
                (sql, false)
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: std::env::var("PROPTEST_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(300),
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    #[test]
    fn random_queries_match_sqlite(seed in any::<u64>()) {
        let mut g = Gen { rng: Rng::new(seed), q: "" };
        let (sql, ordered) = g.query();
        common::check(common::features::fixture(), &format!("seed {seed}"), &sql, ordered);
    }
}

/// The harness itself must notice a wrong answer.
#[test]
#[should_panic(expected = "differs")]
fn harness_detects_mismatches() {
    use common::V;
    common::compare("meta", vec![vec![V::Int(1)]], vec![vec![V::Int(2)]], true);
}
