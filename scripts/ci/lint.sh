#!/usr/bin/env bash
set -e

echo "=== Running pre-commit hooks (secrets, format, lint, type check, tests) ==="
poetry run pre-commit run --all-files

echo "=== All checks passed ==="
