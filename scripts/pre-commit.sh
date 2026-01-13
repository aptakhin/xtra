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

# Run tests with coverage
echo "Running tests with coverage..."
poetry run pytest --cov=xtra --cov-report=term-missing --cov-fail-under=85

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
