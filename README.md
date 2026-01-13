# xtra

A Python library for document text extraction with local and cloud OCR solutions.

## Features

- **Multiple OCR Backends**: Local (EasyOCR, Tesseract, PaddleOCR) and cloud (Azure Document Intelligence, Google Document AI) OCR support
- **PDF Text Extraction**: Native PDF text extraction using pypdfium2
- **Unified Extractors**: Each OCR extractor auto-detects file type (PDF vs image) and handles conversion internally
- **Schema Adapters**: Clean separation of external API schemas from internal models
- **Pydantic Models**: Type-safe document representation with pydantic v1/v2 compatibility

## Installation

```bash
poetry install
```

## Quick Start

### Factory Interface (Recommended)

The simplest way to use xtra is via the factory interface:

```python
from pathlib import Path
from xtra import create_extractor, ExtractorType

# PDF extraction (native text)
with create_extractor(Path("document.pdf"), ExtractorType.PDF) as extractor:
    doc = extractor.extract()

# EasyOCR for images
with create_extractor(Path("image.png"), ExtractorType.EASYOCR, languages=["en"]) as extractor:
    doc = extractor.extract()

# EasyOCR for PDFs (auto-converts to images internally)
with create_extractor(Path("scanned.pdf"), ExtractorType.EASYOCR, dpi=200) as extractor:
    doc = extractor.extract()

# Azure Document Intelligence (credentials from env vars)
with create_extractor(Path("document.pdf"), ExtractorType.AZURE_DI) as extractor:
    doc = extractor.extract()
```

### Example Output

The `extract()` method returns a `Document` object with pages and text blocks:

```python
from pathlib import Path
from xtra import create_extractor, ExtractorType

with create_extractor(Path("document.pdf"), ExtractorType.PDF) as extractor:
    doc = extractor.extract()

# Access extracted data
print(f"Pages: {len(doc.pages)}")  # Pages: 2

for page in doc.pages:
    print(f"Page {page.page + 1} ({page.width:.0f}x{page.height:.0f}):")
    for text in page.texts:
        print(f"  - \"{text.text}\"")
        print(f"    bbox: ({text.bbox.x0:.1f}, {text.bbox.y0:.1f}, {text.bbox.x1:.1f}, {text.bbox.y1:.1f})")
```

Output:
```
Pages: 2
Page 1 (595x842):
  - "First page. First text"
    bbox: (48.3, 57.8, 205.4, 74.6)
  - "First page. Second text"
    bbox: (48.0, 81.4, 231.2, 98.6)
  - "First page. Fourth text"
    bbox: (47.8, 120.5, 221.9, 137.4)
Page 2 (595x842):
  - "Second page. Third text"
    bbox: (47.4, 81.1, 236.9, 98.3)
```

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

### Language Codes

All OCR extractors use **2-letter ISO 639-1 language codes** (e.g., `"en"`, `"fr"`, `"de"`, `"it"`).
Extractors that require different formats (like Tesseract) convert internally.

### OCR Extraction (Local - EasyOCR)

```python
from pathlib import Path
from xtra import EasyOcrExtractor

# For images
with EasyOcrExtractor(Path("image.png"), languages=["en"]) as extractor:
    doc = extractor.extract()

# For PDFs (auto-converts to images)
with EasyOcrExtractor(Path("scanned.pdf"), languages=["en"], dpi=200) as extractor:
    doc = extractor.extract()
```

### OCR Extraction (Local - Tesseract)

Requires Tesseract to be installed on the system:
- macOS: `brew install tesseract`
- Ubuntu: `apt-get install tesseract-ocr`
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

```python
from pathlib import Path
from xtra import TesseractOcrExtractor

# For images
with TesseractOcrExtractor(Path("image.png"), languages=["en"]) as extractor:
    doc = extractor.extract()

# For PDFs (auto-converts to images)
with TesseractOcrExtractor(Path("scanned.pdf"), languages=["en"], dpi=200) as extractor:
    doc = extractor.extract()
```

### OCR Extraction (Local - PaddleOCR)

PaddleOCR provides excellent accuracy for multiple languages, especially Chinese.

```python
from pathlib import Path
from xtra import PaddleOcrExtractor

# For images
with PaddleOcrExtractor(Path("image.png"), lang="en") as extractor:
    doc = extractor.extract()

# For PDFs (auto-converts to images)
with PaddleOcrExtractor(Path("scanned.pdf"), lang="en", dpi=200) as extractor:
    doc = extractor.extract()

# For Chinese text
with PaddleOcrExtractor(Path("chinese_doc.png"), lang="ch") as extractor:
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

### OCR Extraction (Cloud - Google Document AI)

```python
from pathlib import Path
from xtra import GoogleDocumentAIExtractor

with GoogleDocumentAIExtractor(
    Path("document.pdf"),
    processor_name="projects/your-project/locations/us/processors/your-processor-id",
    credentials_path="/path/to/service-account.json",
) as extractor:
    doc = extractor.extract()
```

## CLI Usage

```bash
# PDF extraction
poetry run python -m xtra.cli document.pdf --extractor pdf

# EasyOCR extraction (works for both images and PDFs)
poetry run python -m xtra.cli image.png --extractor easyocr --lang en,it
poetry run python -m xtra.cli scanned.pdf --extractor easyocr --lang en

# Tesseract OCR
poetry run python -m xtra.cli document.pdf --extractor tesseract --lang eng

# PaddleOCR
poetry run python -m xtra.cli document.pdf --extractor paddle --lang en

# Azure Document Intelligence (credentials via CLI or env vars)
poetry run python -m xtra.cli document.pdf --extractor azure-di \
    --azure-endpoint https://your-resource.cognitiveservices.azure.com \
    --azure-key your-api-key

# Or use environment variables
export XTRA_AZURE_DI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
export XTRA_AZURE_DI_KEY=your-api-key
poetry run python -m xtra.cli document.pdf --extractor azure-di

# Google Document AI
poetry run python -m xtra.cli document.pdf --extractor google-docai \
    --google-processor-name projects/your-project/locations/us/processors/123 \
    --google-credentials-path /path/to/credentials.json

# JSON output
poetry run python -m xtra.cli document.pdf --extractor pdf --json

# Specific pages
poetry run python -m xtra.cli document.pdf --extractor pdf --pages 0,1,2
```

## Environment Variables

Cloud extractors support configuration via environment variables:

| Variable | Description |
|----------|-------------|
| `XTRA_AZURE_DI_ENDPOINT` | Azure Document Intelligence endpoint URL |
| `XTRA_AZURE_DI_KEY` | Azure Document Intelligence API key |
| `XTRA_AZURE_DI_MODEL` | Azure model ID (default: `prebuilt-read`) |
| `XTRA_GOOGLE_DOCAI_PROCESSOR_NAME` | Google Document AI processor name |
| `XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH` | Path to Google service account JSON |

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
- `EasyOcrExtractor` - Tests image and PDF OCR with EasyOCR
- `TesseractOcrExtractor` - Tests image and PDF OCR with Tesseract (requires Tesseract installed)
- `PaddleOcrExtractor` - Tests image and PDF OCR with PaddleOCR

**Cloud extractors** (require credentials):
- `AzureDocumentIntelligenceExtractor` - Tests Azure Document Intelligence
- `GoogleDocumentAIExtractor` - Tests Google Document AI

#### Azure Credentials Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your Azure Document Intelligence credentials:
   ```
   XTRA_AZURE_DI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
   XTRA_AZURE_DI_KEY=your-api-key
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

#### Google Document AI Credentials Setup

1. Create a Google Cloud project and enable the Document AI API
2. Create a Document AI processor in the Google Cloud Console
3. Create a service account with Document AI permissions
4. Download the service account JSON key file

5. Edit `.env` with your Google Document AI credentials:
   ```
   XTRA_GOOGLE_DOCAI_PROCESSOR_NAME=projects/your-project/locations/us/processors/your-processor-id
   XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH=/path/to/your/service-account.json
   ```

Google Document AI integration tests are automatically skipped if credentials are not configured.

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
│   ├── azure_di.py     # Azure AnalyzeResult → internal models
│   └── google_docai.py # Google Document → internal models
├── extractors/         # Document extraction
│   ├── azure_di.py     # Azure Document Intelligence
│   ├── google_docai.py # Google Document AI
│   ├── easy_ocr.py     # EasyOCR (local, unified for images/PDFs)
│   ├── tesseract_ocr.py # Tesseract OCR (local, unified for images/PDFs)
│   ├── paddle_ocr.py   # PaddleOCR (local, unified for images/PDFs)
│   ├── pdf.py          # Native PDF extraction
│   └── factory.py      # Unified factory interface
└── models.py           # Internal data models
```

### Extractors

Low-level document extraction:
- `PdfExtractor` - Native PDF text extraction via pypdfium2
- `EasyOcrExtractor` - Image/PDF OCR via EasyOCR (auto-detects file type)
- `TesseractOcrExtractor` - Image/PDF OCR via Tesseract (auto-detects file type)
- `PaddleOcrExtractor` - Image/PDF OCR via PaddleOCR (auto-detects file type)
- `AzureDocumentIntelligenceExtractor` - Azure cloud OCR
- `GoogleDocumentAIExtractor` - Google Cloud Document AI

### Adapters

Schema transformation from external APIs to internal models:
- `AzureDocumentIntelligenceAdapter` - Converts Azure `AnalyzeResult` to `Page`/`TextBlock`
- `GoogleDocumentAIAdapter` - Converts Google `Document` to `Page`/`TextBlock`

### Models

Pydantic models for type-safe document representation:
- `Document` - Full document with pages and metadata
- `Page` - Single page with text blocks
- `TextBlock` - Text with bounding box and confidence
- `BBox` - Bounding box coordinates
- `DocumentMetadata` - Source type, fonts, PDF objects

## License

MIT

## Future plans

- Table extraction for pdf (tabula) and cloud OCR's
- LLM-chat free format result extraction through one interface
