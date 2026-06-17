//! Correctness: all 22 TPC-H-like queries at SF 0.01 must return the same
//! rows as SQLite on identical data.

mod common;

use common::{check, tpch_fixture};

fn run(n: u32) {
    let path = format!("{}/queries/tpch/q{n:02}.sql", env!("CARGO_MANIFEST_DIR"));
    let sql = std::fs::read_to_string(&path).expect("query file");
    let rows = check(tpch_fixture(), &format!("q{n:02}"), &sql, true);
    assert!(
        rows > 0,
        "q{n:02} returned no rows; the test would be vacuous"
    );
}

macro_rules! tpch_tests {
    ($($name:ident => $n:expr),* $(,)?) => {
        $(#[test] fn $name() { run($n); })*
    };
}

tpch_tests! {
    q01 => 1, q02 => 2, q03 => 3, q04 => 4, q05 => 5, q06 => 6, q07 => 7, q08 => 8,
    q09 => 9, q10 => 10, q11 => 11, q12 => 12, q13 => 13, q14 => 14, q15 => 15, q16 => 16,
    q17 => 17, q18 => 18, q19 => 19, q20 => 20, q21 => 21, q22 => 22,
}
