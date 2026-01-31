#!/usr/bin/env bash
set -e

# Fast tests - unit tests only, with timeout enforcement
# Used by pre-commit hook and can be run manually
# For all tests including integration, use: scripts/ci/all-tests.sh

COVERAGE_MIN=${COVERAGE_MIN:-70}
TIMEOUT=${TIMEOUT:-0.5}

echo "=== Running fast tests ==="
echo "Coverage minimum: ${COVERAGE_MIN}%"
echo "Timeout per test: ${TIMEOUT}s"

uv run pytest tests/base tests/ocr tests/llm \
    --timeout="${TIMEOUT}" \
    --cov=xtra \
    --cov-report=term-missing \
    --cov-fail-under="${COVERAGE_MIN}"

echo "=== Fast tests passed ==="
