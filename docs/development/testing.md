# Testing

## Running Tests

```bash
# Run all tests
uv run pytest

# Run fast tests only (unit tests, <0.5s per test)
uv run pytest tests/base tests/ocr

# Run integration tests only (slow, load ML models)
uv run pytest tests/integration

# Run with coverage
uv run pytest --cov=xtra --cov-report=term-missing
```

## Test Structure

```
tests/
├── base/           # Fast unit tests (<0.5s each) - run in pre-commit
├── ocr/            # OCR adapter unit tests (mocked) - run in pre-commit
├── llm/            # LLM unit tests (mocked) - run in pre-commit
└── integration/    # Slow tests - NOT in pre-commit
    ├── ocr/        # OCR integration tests (load real ML models)
    └── llm/        # LLM integration tests (call real APIs)
```

**Pre-commit runs:** `tests/base`, `tests/ocr`, and `tests/llm` with 0.5s timeout per test.

**CI runs:** All tests including integration tests.

## Test Data

Test files are located in `tests/data/`:

- `test_pdf_2p_text.pdf` - 2-page PDF with text
- `test_pdf_2p_text_rotated.pdf` - 2-page PDF with rotated text
- `test_pdf_table.pdf` - PDF with tables
- `test_image.png` - Test image for OCR

## Integration Tests

Integration tests load real ML models and call real services.

### Local Extractors (No Credentials Required)

- `PdfExtractor` - Tests PDF text extraction
- `EasyOcrExtractor` - Tests image and PDF OCR with EasyOCR
- `TesseractOcrExtractor` - Tests image and PDF OCR with Tesseract
- `PaddleOcrExtractor` - Tests image and PDF OCR with PaddleOCR

### Cloud Extractors (Require Credentials)

Tests are automatically skipped if credentials are not configured.

#### Azure Setup

```bash
cp .env.example .env
# Edit .env with your credentials:
# XTRA_AZURE_DI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
# XTRA_AZURE_DI_KEY=your-api-key

# Run tests
export $(cat .env | xargs)
uv run pytest tests/integration -v
```

#### Google Setup

```bash
# Edit .env:
# XTRA_GOOGLE_DOCAI_PROCESSOR_NAME=projects/your-project/locations/us/processors/123
# XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH=/path/to/service-account.json

export $(cat .env | xargs)
uv run pytest tests/integration -v
```

## TDD Workflow

This project follows Test-Driven Development:

1. **Red** - Write a failing test first
2. **Green** - Write minimal code to pass the test
3. **Refactor** - Clean up while keeping tests green

## VCR Cassettes

For API tests, we use VCR to record HTTP interactions:

```python
@pytest.mark.vcr()
def test_api_call():
    # First run records the cassette
    # Subsequent runs replay it
    ...
```

Cassettes are stored in `tests/cassettes/`.
