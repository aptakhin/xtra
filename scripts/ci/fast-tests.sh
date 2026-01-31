#!/usr/bin/env bash
set -e

# Fast tests - unit tests only, with timeout enforcement
# Used by pre-commit hook and can be run manually
# For all tests including integration, use: scripts/ci/all-tests.sh

TIMEOUT=${TIMEOUT:-0.5}

echo "=== Running fast tests ==="
echo "Timeout per test: ${TIMEOUT}s"

uv run pytest tests/base tests/ocr tests/llm --timeout="${TIMEOUT}"

echo "=== Fast tests passed ==="
