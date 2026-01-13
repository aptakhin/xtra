# xtra

A Python library for document text extraction with local and cloud OCR solutions.

## Features

- **Multiple OCR Backends**: Local (EasyOCR, Tesseract, PaddleOCR) and cloud (Azure Document Intelligence, Google Document AI) OCR support
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
from xtra import EasyOcrExtractor, PdfToImageEasyOcrExtractor

# For images
with EasyOcrExtractor(Path("image.png"), languages=["en"]) as extractor:
    doc = extractor.extract()

# For PDFs via OCR
with PdfToImageEasyOcrExtractor(Path("scanned.pdf"), languages=["en"], dpi=200) as extractor:
    doc = extractor.extract()
```

### OCR Extraction (Local - Tesseract)

Requires Tesseract to be installed on the system:
- macOS: `brew install tesseract`
- Ubuntu: `apt-get install tesseract-ocr`
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

```python
from pathlib import Path
from xtra.extractors.tesseract_ocr import TesseractOcrExtractor, PdfToImageTesseractExtractor

# For images
with TesseractOcrExtractor(Path("image.png"), languages=["eng"]) as extractor:
    doc = extractor.extract()

# For PDFs via OCR
with PdfToImageTesseractExtractor(Path("scanned.pdf"), languages=["eng"], dpi=200) as extractor:
    doc = extractor.extract()
```

### OCR Extraction (Local - PaddleOCR)

PaddleOCR provides excellent accuracy for multiple languages, especially Chinese.

```python
from pathlib import Path
from xtra.extractors.paddle_ocr import PaddleOcrExtractor, PdfToImagePaddleExtractor

# For images
with PaddleOcrExtractor(Path("image.png"), lang="en") as extractor:
    doc = extractor.extract()

# For PDFs via OCR
with PdfToImagePaddleExtractor(Path("scanned.pdf"), lang="en", dpi=200) as extractor:
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
from xtra.extractors.google_docai import GoogleDocumentAIExtractor

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

# EasyOCR extraction
poetry run python -m xtra.cli image.png --extractor easyocr --lang en,it

# PDF via EasyOCR
poetry run python -m xtra.cli scanned.pdf --extractor pdf-easyocr

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
- `EasyOcrExtractor` - Tests image OCR with EasyOCR
- `PdfToImageEasyOcrExtractor` - Tests PDF-to-image OCR with EasyOCR
- `TesseractOcrExtractor` - Tests image OCR with Tesseract (requires Tesseract installed)
- `PdfToImageTesseractExtractor` - Tests PDF-to-image OCR with Tesseract
- `PaddleOcrExtractor` - Tests image OCR with PaddleOCR
- `PdfToImagePaddleExtractor` - Tests PDF-to-image OCR with PaddleOCR

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

#### Google Document AI Credentials Setup

1. Create a Google Cloud project and enable the Document AI API
2. Create a Document AI processor in the Google Cloud Console
3. Create a service account with Document AI permissions
4. Download the service account JSON key file

5. Edit `.env` with your Google Document AI credentials:
   ```
   GOOGLE_DOCAI_PROCESSOR_NAME=projects/your-project/locations/us/processors/your-processor-id
   GOOGLE_DOCAI_CREDENTIALS_PATH=/path/to/your/service-account.json
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
│   ├── ocr.py          # EasyOCR (local)
│   ├── tesseract_ocr.py # Tesseract OCR (local)
│   ├── paddle_ocr.py   # PaddleOCR (local)
│   └── pdf.py          # Native PDF extraction
└── models.py           # Internal data models
```

### Extractors

Low-level document extraction:
- `PdfExtractor` - Native PDF text extraction via pypdfium2
- `EasyOcrExtractor` - Image OCR via EasyOCR
- `PdfToImageEasyOcrExtractor` - PDF to image + EasyOCR
- `TesseractOcrExtractor` - Image OCR via Tesseract
- `PdfToImageTesseractExtractor` - PDF to image + Tesseract OCR
- `PaddleOcrExtractor` - Image OCR via PaddleOCR
- `PdfToImagePaddleExtractor` - PDF to image + PaddleOCR
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
