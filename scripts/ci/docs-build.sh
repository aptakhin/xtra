#!/bin/bash
set -e

echo "Building documentation..."
uv run mkdocs build --strict
echo "Documentation built successfully!"
