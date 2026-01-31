#!/usr/bin/env bash
set -e

echo "=== Checking architecture rules ==="
uv run lint-imports
