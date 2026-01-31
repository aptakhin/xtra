#!/usr/bin/env bash
set -e

# All tests - includes slow integration tests that load ML models
# For fast tests only, use: scripts/ci/fast-tests.sh

COVERAGE_MIN=${COVERAGE_MIN:-78}

echo "=== Running all tests ==="
echo "Coverage minimum: ${COVERAGE_MIN}%"

uv run pytest \
    --cov=xtra \
    --cov-report=term-missing \
    --cov-fail-under="${COVERAGE_MIN}"

echo "=== All tests passed ==="
