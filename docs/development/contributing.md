# Contributing

## Development Setup

```bash
# Clone the repository
git clone https://github.com/aptakhin/unifex.git
cd unifex

# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

## Documentation

Build and serve the documentation locally:

```bash
# Serve docs with live reload
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

Open http://localhost:8000 to view the documentation.

### Testing Documentation Examples

Documentation code examples are tested using Sybil:

```bash
# Run doc tests
uv run pytest docs/
```

## Code Style

- Use type hints for all function signatures
- Follow existing patterns in the codebase
- Imports should be sorted (standard library, third-party, local)

The project uses [ruff](https://github.com/astral-sh/ruff) for formatting and linting.

## Pre-commit Checks

The pre-commit hook runs automatically on `git commit`. To run manually:

```bash
uv run pre-commit run --all-files
```

This runs:

- `ruff format` - Code formatting
- `ruff check --fix` - Linting with auto-fix
- `ty check` - Type checking
- `pytest` - Test suite

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all pre-commit checks pass
4. Submit a pull request

## Design Principles

- Think about correct degree of coupling when designing components
- Prefer exceptions for invalid user input over returning error objects
- Avoid over-engineering - only make changes that are directly requested
