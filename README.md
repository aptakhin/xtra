# xtra

A Python library for document text extraction with local and cloud OCR solutions.

## Features

- **Multiple OCR Backends**: Local (EasyOCR) and cloud (Azure Document Intelligence) OCR support
- **PDF Text Extraction**: Native PDF text extraction using pypdfium2
- **Schema Adapters**: Clean separation of external API schemas from internal models
- **Pydantic Models**: Type-safe document representation with pydantic v1/v2 compatibility

## Installation

```bash
poetry install
```

## Quick Start

### PDF Text Extraction

```python
from pathlib import Path
from xtra import PdfExtractor

with PdfExtractor(Path("document.pdf")) as extractor:
    doc = extractor.extract()
    for page in doc.pages:
        for text in page.texts:
            print(text.text)
```

### OCR Extraction (Local - EasyOCR)

```python
from pathlib import Path
from xtra import OcrExtractor, PdfToImageOcrExtractor

# For images
with OcrExtractor(Path("image.png"), languages=["en"]) as extractor:
    doc = extractor.extract()

# For PDFs via OCR
with PdfToImageOcrExtractor(Path("scanned.pdf"), languages=["en"], dpi=200) as extractor:
    doc = extractor.extract()
```

### OCR Extraction (Cloud - Azure)

```python
from pathlib import Path
from xtra import AzureDocumentIntelligenceExtractor

with AzureDocumentIntelligenceExtractor(
    Path("document.pdf"),
    endpoint="https://your-resource.cognitiveservices.azure.com",
    key="your-api-key",
) as extractor:
    doc = extractor.extract()
```

## CLI Usage

```bash
# PDF extraction
poetry run python -m xtra.cli document.pdf --extractor pdf

# OCR extraction
poetry run python -m xtra.cli image.png --extractor ocr --lang en,it

# PDF via OCR
poetry run python -m xtra.cli scanned.pdf --extractor pdf-ocr

# Azure Document Intelligence
poetry run python -m xtra.cli document.pdf --extractor azure-di \
    --azure-endpoint https://your-resource.cognitiveservices.azure.com \
    --azure-key your-api-key

# JSON output
poetry run python -m xtra.cli document.pdf --extractor pdf --json

# Specific pages
poetry run python -m xtra.cli document.pdf --extractor pdf --pages 0,1,2
```

## Development

### Setup

```bash
# Install dependencies
poetry install

# Install git pre-commit hook
./scripts/install-hooks.sh
```

### Running Tests

```bash
# Run all tests
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=xtra --cov-report=term-missing

# Run specific test file
poetry run pytest tests/test_azure_di.py -v

# Run integration tests only
poetry run pytest tests/test_integration.py -v
```

### Integration Tests

Integration tests run against real files and services without mocking. They are located in `tests/test_integration.py`.

**Local extractors** (no credentials required):
- `PdfExtractor` - Tests PDF text extraction
- `OcrExtractor` - Tests image OCR with EasyOCR
- `PdfToImageOcrExtractor` - Tests PDF-to-image OCR

**Cloud extractors** (require credentials):
- `AzureDocumentIntelligenceExtractor` - Tests Azure Document Intelligence

#### Azure Credentials Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your Azure Document Intelligence credentials:
   ```
   AZURE_DI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
   AZURE_DI_KEY=your-api-key
   ```

3. Load environment variables before running tests:
   ```bash
   # Option 1: Source the .env file
   export $(cat .env | xargs)
   poetry run pytest tests/test_integration.py -v

   # Option 2: Use env command
   env $(cat .env | xargs) poetry run pytest tests/test_integration.py -v
   ```

Azure integration tests are automatically skipped if credentials are not configured.

### Pre-commit Checks

The pre-commit hook runs automatically on `git commit`. To run manually:

```bash
./scripts/pre-commit.sh
```

This runs:
- `ruff format` - Code formatting
- `ruff check --fix` - Linting with auto-fix
- `ty check` - Type checking
- `pytest` with 85% coverage requirement

## Architecture

```
xtra/
├── adapters/           # Schema transformation
│   └── azure_di.py     # Azure AnalyzeResult → internal models
├── extractors/         # Document extraction
│   ├── azure_di.py     # Azure Document Intelligence
│   ├── ocr.py          # EasyOCR (local)
│   └── pdf.py          # Native PDF extraction
└── models.py           # Internal data models
```

### Extractors

Low-level document extraction:
- `PdfExtractor` - Native PDF text extraction via pypdfium2
- `OcrExtractor` - Image OCR via EasyOCR
- `PdfToImageOcrExtractor` - PDF to image + OCR
- `AzureDocumentIntelligenceExtractor` - Azure cloud OCR

### Adapters

Schema transformation from external APIs to internal models:
- `AzureDocumentIntelligenceAdapter` - Converts Azure `AnalyzeResult` to `Page`/`TextBlock`

### Models

Pydantic models for type-safe document representation:
- `Document` - Full document with pages and metadata
- `Page` - Single page with text blocks
- `TextBlock` - Text with bounding box and confidence
- `BBox` - Bounding box coordinates
- `DocumentMetadata` - Source type, fonts, PDF objects

## License

MIT
