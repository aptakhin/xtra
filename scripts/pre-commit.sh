#!/usr/bin/env bash

set -e

# Run format and lint
echo "Running ruff format and check..."
poetry run ruff format
poetry run ruff check --fix

# Re-add any files modified by formatting
git add -u

# Run type check
echo "Running ty type check..."
poetry run ty check

# Run tests with coverage (excluding cloud tests that require credentials)
echo "Running tests with coverage..."
poetry run pytest -k "not (azure and test_ocr_extract_pdf) and not (google and test_ocr_extract_pdf) and not test_llm_vcr and not paddle" --cov=xtra --cov-report=term-missing --cov-fail-under=78

# Check if there are any staged files
if [ -z "$(git diff --cached --name-only)" ]; then
    echo "Nothing to commit - no staged files."
    exit 0
fi

# Show staged files
current_branch=$(git branch --show-current)
echo "--------------------------------"
echo "Current branch: $current_branch"
echo "Git staged files:"
echo "--------------------------------"
git status --porcelain | grep -E '^[AMRC]' || true
