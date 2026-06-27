# Design decisions

Key design decisions in quarry, each with the reasoning and what it would cost if it turned out wrong.

## Scope and stack

- Rust 2021 with `arrow` and `parquet` pinned to exactly `=60.0.0`, default features off (CSV reader on; Parquet with `arrow` + `snap` only) — the engine is built on physical operators over Arrow arrays, and exact pins keep `make check` reproducible from a fresh clone — an upgrade needs a manual bump plus a re-run of the golden tests.
- the SQL parser, binder, optimizer and operators are all written here; no `sqlparser`/`datafusion` crates — building them is the point of the project — the parser covers a documented subset, not full SQL (see `docs/SQL.md`).
- Arrow compute kernels (comparison, arithmetic, `take`, `filter`, `lexsort`, `zip`, `cast`, the row format) are used as building blocks; hash join, hash aggregation, top-k planning, statistics and pruning are implemented here — re-writing SIMD comparison kernels would add nothing to the design story — the engine is only as fast as those kernels on the types it uses.
- a closed set of SQL types (BOOLEAN, BIGINT, DOUBLE, VARCHAR, DATE). Every source column is normalized to one of them at scan time (Int32 to Int64, Float32 to Float64, LargeUtf8/Utf8View to Utf8, Date64 to Date32) — the kernels and the type checker stay small — DECIMAL is approximated by DOUBLE, so money sums are floating point (correctness tests compare floats with a 1e-6 relative tolerance).
- subqueries are supported only in FROM (derived tables) and as CTEs, which is the supported subset. `IN (SELECT ...)`, `EXISTS` and scalar subqueries are rejected with an `Unsupported` error naming the rewrite — decorrelation is a project in itself — TPC-H queries 2, 4, 11, 15-18, 20-22 are rewritten into equivalent joins against derived tables, so the suite is "TPC-H-like", not TPC-H.

## Semantics (chosen to match SQLite, the reference engine)

- SQLite is the correctness oracle (via `rusqlite` with the bundled SQLite 3) — tests must run without network access or Docker, and DuckDB would need a separate install; SQLite is embeddable and deterministic — SQLite is dynamically typed, so the tests normalize values (booleans to 0/1, dates to ISO strings, int/float compared numerically).
- NULLs sort as the smallest value (first in ASC, last in DESC); explicit `NULLS FIRST/LAST` is supported — matches SQLite, making ordered comparisons exact — differs from PostgreSQL's default.
- integer division truncates and `x / 0`, `x % 0` return NULL — matches SQLite — PostgreSQL raises an error instead.
- integer overflow in `+ - *`, `abs()` and `sum()` is a runtime error — silent wrap-around would return wrong answers — SQLite converts some overflows to floating point; the fuzzer keeps values small enough to avoid the difference.
- LIKE is case-sensitive (SQL standard, PostgreSQL); the test SQLite connection sets `PRAGMA case_sensitive_like = ON` — SQLite's default is ASCII case-insensitive — users coming from SQLite may be surprised.
- `CAST` uses Arrow's safe cast: an unparseable string becomes NULL — consistent with the kernels — SQLite would return 0; casts from strings are excluded from the differential tests.
- `substr` with a start below 1 follows PostgreSQL (`substr('hello', 0, 2) = 'h'`) — well defined and documented — SQLite counts negative starts from the end; not fuzzed.
- `CASE` evaluates every branch on the whole batch and then selects (vectorized `zip`) — branch-free and simple — an error in a branch whose condition is false for every row (for example an overflow) still fails the query. Division by zero is NULL, so the common `CASE WHEN x = 0 ...` guard is unaffected.
- `-0.0` equals `0.0` everywhere values are compared: comparisons and IN lists, hash-join keys, GROUP BY and DISTINCT keys, `count(DISTINCT ...)` and sort keys all fold `-0.0` into `0.0` first (`physical::expr::canonical_zero`), as do NDV statistics — Arrow's `cmp` kernels and row format use a total float order that puts `-0.0` below `0.0`, so `SELECT 0.0 = -0.0` was false and a WHERE, JOIN or GROUP BY on a computed float (`-(x * 0)`, `x * negative`) silently disagreed with SQL and SQLite — the fold is applied at those consumers, not to stored or projected values, so `SELECT -0.0` still prints `-0.0` and a group key shows the folded `0.0`; NaN keeps Arrow's total order (NaN equals NaN), which cannot arise from SQL here because `x / 0` is NULL.
- `DATE + INTERVAL 'n' MONTH` clamps to the end of the month (`2024-01-31 + 1 month = 2024-02-29`) — PostgreSQL behaviour — SQLite's `date(..., '+1 month')` overflows into the next month; the test translator only folds constant dates where the two agree.

## Planning

- every column produced anywhere in a plan gets a unique `ColId`; expressions reference ids, not positions or names — moving predicates through projections or reordering joins needs no index arithmetic and self-joins (`nation n1, nation n2`) cannot be confused — physical planning resolves ids to positions once per operator.
- CTEs are inlined: each reference re-binds the CTE body, producing fresh column ids — a CTE referenced twice (TPC-H Q15) can be joined with itself — a CTE used N times is computed N times; there is no materialization.
- aggregates in SELECT/HAVING/ORDER BY are replaced by placeholder identifiers before binding, then grouped sub-expressions are matched structurally — handles `sum(x) / count(*)`, `GROUP BY 1`, `GROUP BY alias` and ORDER BY on aggregates uniformly — the placeholder qualifier contains a NUL byte so user SQL cannot reach it.
- rules run once, in a fixed order (simplify, predicate pushdown, join reorder, projection pushdown, limit pushdown) instead of to a fixed point — each rule's input requirements are met by the earlier ones and plans stay predictable for golden tests — a rewrite enabled by a later rule is not revisited.
- constant folding evaluates expressions with the runtime evaluator on a one-row batch and leaves the expression in place if evaluation fails — folding can never disagree with execution, and an overflow in a filter that no row reaches does not fail the query — folding costs a tiny batch allocation per constant.
- common conjuncts are factored out of ORs (`(a AND b) OR (a AND c)` to `a AND (b OR c)`) — TPC-H Q19 repeats its join key inside every branch; without this the join is a cross product — SQLite lacks the rewrite and needs about 7 s for Q19 at SF 0.01, so the benchmark does not run Q19 on SQLite.
- a WHERE predicate that is strict over the right side of a LEFT JOIN converts it to an INNER JOIN; `IS NULL`, `CASE` and `COALESCE` are treated as non-strict — standard outer-join simplification — the strictness check is conservative, so some convertible joins stay LEFT.
- statistics are exact: registration runs one full pass computing row count, null count, min/max and exact NDV with a hash set of 64-bit value hashes per column — exact NDV keeps estimates and golden tests stable — registration is O(rows) time and O(distinct values) memory per column; a HyperLogLog sketch would bound memory.
- join reordering uses DPsub over connected subsets for clusters of up to 10 relations and a greedy pairing above that, with the C_out cost model (sum of intermediate result sizes); the smaller estimated input becomes the hash-join build side — TPC-H's largest cluster (Q8) has 8 relations — C_out ignores operator costs, so two plans with equal C_out can differ in run time (see the Q17/Q20 benchmark rows).
- LEFT JOIN cardinality uses key containment (`matched = ndv(right)/ndv(left)`) to estimate NULL-extended rows and raises right-side null fractions — makes anti-join (`LEFT JOIN ... IS NULL`) estimates sensible — assumes containment, which fails for unrelated key domains.

## Execution

- single-threaded pull-based (Volcano) execution over batches of 8192 rows — simplest model that still amortizes interpretation over whole batches and supports early termination (LIMIT) — no intra-query parallelism; the machine's other cores are unused.
- the hash join builds its whole right input in memory (concatenated) and indexes it with a hash-to-first-row map plus a `next` chain array; keys use Arrow's row format — multi-column keys of any type compare as bytes, chains avoid per-key vectors — no spilling, so the build side must fit in memory.
- one probe batch produces one output batch, however many matches it has — keeps the join simple — a many-to-many join can produce a very large single batch.
- Parquet row-group pruning applies only to pushed `column <op> literal` filters with min/max statistics — covers range and point filters on clustered data — no page-index or bloom-filter pruning.
- `lineitem.parquet` and `orders.parquet` are written sorted by ship/order date — mirrors a time-partitioned warehouse table and makes pruning observable — CSV files and in-memory tables keep generation order.

## Data, tests and benchmarks

- data comes from a deterministic in-repo generator (SplitMix64), not the official `dbgen` — no download, no C toolchain, byte-identical data on every machine — value distributions differ from TPC-H, so timings are not comparable to published TPC-H results.
- the generator's partsupp supplier formula is simplified to `(part + i * max(S/4, 1)) % S + 1` — the official formula repeats suppliers for some parts at tiny scale factors — the supplier spread differs slightly from TPC-H.
- Q11's threshold fraction (0.001) and Q18's quantity threshold (250) are scaled to the generator — the originals return no rows at SF 0.01 with 1-7 lines per order — those queries differ from the official text.
- golden plan files include cardinality estimates — estimates are part of what the optimizer golden tests should pin down — any statistics or generator change requires `make golden` and a review of the diff.
- the differential fuzzer has a DOUBLE expression family built from `f`, `f * 0`, unary minus and multiples of 0.25 that feeds signed zeros into comparisons, IN, GROUP BY, DISTINCT, ORDER BY and join keys; the harness sorts `-0.0` and `0.0` to the same canonical key — integer-only fuzzing missed the signed-zero bug — operands stay exact in binary floating point, so general float rounding is still covered only by the tolerance-based TPC-H and feature tests.
- coverage floor is 85% of lines across the library (`make coverage`) — the measured figure is above 90%, and 85% leaves room for defensive branches — a regression of a few percent fails CI.
- benchmark numbers in the README are medians from `examples/bench.rs` on a shared laptop with other build jobs running — the only hardware available — they are noisy (single-digit-millisecond differences are not meaningful) and are reported as such.
