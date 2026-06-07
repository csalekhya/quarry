#!/usr/bin/env bash
# Offline end-to-end demo: generate TPC-H-like Parquet data, run queries over
# it and over the bundled CSV fixtures, and show the optimizer at work.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --release --locked --quiet
BIN="${CARGO_TARGET_DIR:-target}/release/quarry"
DATA=data/demo
body() { grep -v '^--' "$1"; }

echo "### 1. Generate SF 0.01 TPC-H-like data as Parquet"
"$BIN" gen --sf 0.01 --out "$DATA" --format parquet

echo
echo "### 2. TPC-H Q1 (pricing summary) over Parquet"
"$BIN" query --data "$DATA" -f queries/tpch/q01.sql

echo
echo "### 3. Bundled CSV fixtures: LEFT JOIN, NULL-aware aggregates, self-join"
"$BIN" query --data fixtures "
  SELECT d.name AS department, count(e.id) AS staff, avg(e.salary) AS avg_salary,
         min(e.hired) AS first_hire, coalesce(d.location, '(remote)') AS location
  FROM departments d LEFT JOIN employees e ON e.dept_id = d.id
  GROUP BY d.name, d.location
  ORDER BY staff DESC, department"
"$BIN" query --data fixtures "
  WITH mgr AS (SELECT id, name FROM employees)
  SELECT e.name AS employee, m.name AS manager
  FROM employees e JOIN mgr m ON e.manager_id = m.id
  WHERE e.hired >= DATE '2020-01-01'
  ORDER BY e.hired"

echo
echo "### 4. EXPLAIN: Q5's six-way comma join after predicate pushdown and join reordering"
"$BIN" query --data "$DATA" "EXPLAIN $(body queries/tpch/q05.sql)"

echo
echo "### 5. EXPLAIN ANALYZE: Q6 prunes Parquet row groups with pushed date filters"
"$BIN" query --data "$DATA" "EXPLAIN ANALYZE $(body queries/tpch/q06.sql)"

echo
echo "### 6. Errors carry the stage and position"
"$BIN" query --data "$DATA" "SELECT l_orderkey FROM lineitem WHERE l_shipdate >" || true
"$BIN" query --data "$DATA" "SELECT o_custkey, count(*) FROM orders" || true
