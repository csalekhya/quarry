#!/usr/bin/env bash
# Line counts quoted in the README: library/CLI source with inline #[cfg(test)]
# modules separated out, integration tests, and query/fixture files.
set -euo pipefail
cd "$(dirname "$0")/.."
src_total=0; src_tests=0
for f in $(find src -name '*.rs' | sort); do
  n=$(wc -l < "$f")
  t=$(awk '/^#\[cfg\(test\)\]/{found=1} found{c++} END{print c+0}' "$f")
  src_total=$((src_total + n)); src_tests=$((src_tests + t))
done
code=0
for f in $(find src -name '*.rs' | sort); do
  c=$(awk '/^#\[cfg\(test\)\]/{exit} !/^[[:space:]]*$/ && !/^[[:space:]]*\/\// {c++} END{print c+0}' "$f")
  code=$((code + c))
done
echo "src (non-test):       $((src_total - src_tests)) lines, $code excluding blank and comment lines"
echo "src inline tests:     $src_tests"
echo "tests/ (integration): $(cat tests/*.rs tests/common/*.rs | wc -l | tr -d ' ')"
echo "examples/:            $(cat examples/*.rs | wc -l | tr -d ' ')"
echo "queries/*.sql:        $(cat queries/tpch/*.sql | wc -l | tr -d ' ')"
