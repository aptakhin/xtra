# Project Guidelines for Claude

## Critical Rules

1. **TDD Always** - Write tests FIRST, then implement. Red → Green → Refactor.
2. **Ask Questions** - Don't blindly execute. Question unclear requirements, suggest better approaches.
3. **Always Commit** - Commit immediately after completing any feature or task.
4. **No Mock Stubs** - Avoid mocks for coverage. If tests need credentials:
   - Ask user to provide credentials to record cassettes
   - Or exclude from coverage if unavailable
   - Never fake coverage with mocks

> See [.claude/guidelines.md](.claude/guidelines.md) for detailed guidelines.

## Build & Test Commands

- Run all tests: `uv run pytest`
- Run specific test: `uv run pytest tests/test_file.py::TestClass::test_method -v`
- Run with coverage: `uv run pytest --cov=xtra`

## Code Style

- Use type hints for all function signatures
- Follow existing patterns in the codebase
- Imports should be sorted (standard library, third-party, local)

## Testing Guidelines

- Integration tests are in `tests/test_integration.py`
- Cloud extractors (Azure, Google) use credential fixtures that skip if credentials aren't configured
- Local OCR tests (EasyOCR, Tesseract, PaddleOCR) run unconditionally
- Use 2-letter ISO 639-1 language codes (e.g., "en", "fr", "de") for all extractors
- Keep tests loosely coupled: test logic with fixtures/mocks, not real files when possible

## Design Principles

- Think about correct degree of coupling when designing components
- Prefer exceptions for invalid user input over returning error objects

## Error Handling

- Raise exceptions for invalid user input (e.g., invalid page number, missing file)
- Use result objects with success/error fields only for expected failures during processing

## Coordinate System

- All extractors support `output_unit` parameter: POINTS, PIXELS, INCHES, NORMALIZED
- PDF extractor doesn't support PIXELS output (PDFs don't have inherent DPI)
- OCR extractors have DPI parameter for pixel conversions
- NORMALIZED coordinates are 0-1 relative to page dimensions

## Workflow

- Always create a git commit after completing a task
