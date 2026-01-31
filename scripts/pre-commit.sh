#!/usr/bin/env bash

set -e

# Run all pre-commit hooks (gitleaks, ruff, ty, pytest)
echo "Running pre-commit hooks..."
poetry run pre-commit run --all-files

# Re-add any files modified by formatting
git add -u

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
