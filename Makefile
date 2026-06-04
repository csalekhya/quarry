# Gates: `make check` runs formatting, lint (warnings are errors; clippy also
# type-checks every target), the full test suite under coverage instrumentation
# with a line-coverage floor, and the doc tests.

COVERAGE_MIN ?= 85

.PHONY: check fmt fmt-check lint test coverage doctest demo bench fuzz golden clean-data

check: fmt-check lint coverage doctest

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

lint:
	cargo clippy --all-targets --locked -- -D warnings

test:
	cargo test --locked

coverage:
	@command -v cargo-llvm-cov >/dev/null 2>&1 || { echo "cargo-llvm-cov is required: cargo install cargo-llvm-cov"; exit 1; }
	cargo llvm-cov --locked --summary-only --fail-under-lines $(COVERAGE_MIN)

doctest:
	cargo test --locked --doc

demo:
	./scripts/demo.sh

bench:
	cargo run --release --locked --example bench -- --sf 0.1 --iters 5

fuzz:
	PROPTEST_CASES=20000 cargo test --release --locked --test differential_fuzz

golden:
	UPDATE_GOLDEN=1 cargo test --locked --test optimizer_golden

clean-data:
	rm -rf data
