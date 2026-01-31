# unifex

A Python library for document text extraction with local and cloud OCR solutions.

**Focus:** Built for tasks like fraud detection where precision matters. We needed a universal tool for both PDF and image processing with best-in-class OCR support through local engines (EasyOCR, Tesseract, PaddleOCR) and cloud services (Azure Document Intelligence, Google Document AI).

📖 **[Documentation](https://aptakhin.name/unifex/)**

## Features

- **Multiple OCR Backends**: Local (EasyOCR, Tesseract, PaddleOCR) and cloud (Azure Document Intelligence, Google Document AI) OCR support
- **PDF Text Extraction**: Native PDF text extraction using pypdfium2
- **LLM Extraction**: Extract structured data using GPT-4o, Claude, Gemini, or OpenAI-compatible APIs
- **Parallel Extraction**: Process multiple pages concurrently with thread or process executors
- **Async Support**: Native async/await API for integration with async applications
- **Unified Extractors**: Each OCR extractor auto-detects file type (PDF vs image) and handles conversion internally
- **Schema Adapters**: Clean separation of external API schemas from internal models
- **Pydantic Models**: Type-safe document representation with pydantic v1/v2 compatibility

## Alternatives

For broader document processing, check out [Docling](https://docling-project.github.io) and [Kreuzberg](https://kreuzberg.dev/).

## Installation

```bash
pip install unifex
```

Or with optional dependencies:

```bash
pip install unifex[pdf]       # PDF text extraction
pip install unifex[easyocr]   # EasyOCR support
pip install unifex[tesseract] # Tesseract OCR support
pip install unifex[azure]     # Azure Document Intelligence
pip install unifex[google]    # Google Document AI
pip install unifex[llm-openai]     # OpenAI/GPT-4 extraction
pip install unifex[llm-anthropic]  # Anthropic/Claude extraction
pip install unifex[all]       # All dependencies
```

## Quick Start

### Factory Interface (Recommended)

The simplest way to use unifex is via the factory interface. Both string paths and `Path` objects are accepted:

```python
from unifex import create_extractor, ExtractorType

# PDF extraction (native text) - string path
with create_extractor("document.pdf", ExtractorType.PDF) as extractor:
    result = extractor.extract()
    doc = result.document  # Access the Document

# EasyOCR for images
with create_extractor("image.png", ExtractorType.EASYOCR, languages=["en"]) as extractor:
    result = extractor.extract()

# EasyOCR for PDFs (auto-converts to images internally)
with create_extractor("scanned.pdf", ExtractorType.EASYOCR, dpi=200) as extractor:
    result = extractor.extract()

# Azure Document Intelligence (credentials from env vars)
with create_extractor("document.pdf", ExtractorType.AZURE_DI) as extractor:
    result = extractor.extract()

# Path objects also work
from pathlib import Path
with create_extractor(Path("document.pdf"), ExtractorType.PDF) as extractor:
    result = extractor.extract()
```

### Example Output

The `extract()` method returns an `ExtractionResult` containing the `Document` and per-page results:

```python
from unifex import create_extractor, ExtractorType

with create_extractor("document.pdf", ExtractorType.PDF) as extractor:
    result = extractor.extract()

# Check extraction status
print(f"Success: {result.success}")  # True if all pages extracted

# Access extracted document
doc = result.document
print(f"Pages: {len(doc.pages)}")  # Pages: 2

for page in doc.pages:
    print(f"Page {page.page + 1} ({page.width:.0f}x{page.height:.0f}):")
    for text in page.texts:
        print(f"  - \"{text.text}\"")
        print(f"    bbox: ({text.bbox.x0:.1f}, {text.bbox.y0:.1f}, {text.bbox.x1:.1f}, {text.bbox.y1:.1f})")

# Handle errors if any
if not result.success:
    for page_num, error in result.errors:
        print(f"Page {page_num} failed: {error}")
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

For more detailed examples, see the [documentation](https://aptakhin.name/unifex/).

### PDF Text Extraction

```python
from unifex import PdfExtractor

# String paths work directly
with PdfExtractor("document.pdf") as extractor:
    result = extractor.extract()
    for page in result.document.pages:
        for text in page.texts:
            print(text.text)
```

### Language Codes

All OCR extractors use **2-letter ISO 639-1 language codes** (e.g., `"en"`, `"fr"`, `"de"`, `"it"`).
Extractors that require different formats (like Tesseract) convert internally.

### Parallel Extraction

Extract multiple pages concurrently for faster processing:

```python
from unifex import create_extractor, ExtractorType, ExecutorType

# Thread-based parallelism (recommended for most cases)
with create_extractor("large_document.pdf", ExtractorType.EASYOCR) as extractor:
    result = extractor.extract(max_workers=4)  # 4 parallel workers

# Process-based parallelism (for CPU-bound pure Python workloads)
with create_extractor("large_document.pdf", ExtractorType.EASYOCR) as extractor:
    result = extractor.extract(max_workers=4, executor=ExecutorType.PROCESS)

# Extract specific pages in parallel
with create_extractor("document.pdf", ExtractorType.PDF) as extractor:
    result = extractor.extract(pages=[0, 2, 5, 8], max_workers=4)
```

**Executor Types:**

| Executor | Best For | Notes |
|----------|----------|-------|
| `THREAD` (default) | Most OCR use cases | Shared model cache, low overhead, C libraries release GIL |
| `PROCESS` | CPU-bound pure Python | Models duplicated per worker, higher memory usage |

### Async Extraction

For async applications, use the async API:

```python
import asyncio
from unifex import create_extractor, ExtractorType

async def extract_document():
    with create_extractor("document.pdf", ExtractorType.EASYOCR) as extractor:
        result = await extractor.extract_async(max_workers=4)
        return result.document

doc = asyncio.run(extract_document())
```

### OCR Extraction (Local - EasyOCR)

```python
from unifex import EasyOcrExtractor

# For images
with EasyOcrExtractor("image.png", languages=["en"]) as extractor:
    result = extractor.extract()

# For PDFs (auto-converts to images)
with EasyOcrExtractor("scanned.pdf", languages=["en"], dpi=200) as extractor:
    result = extractor.extract()
```

### OCR Extraction (Local - Tesseract)

Requires Tesseract to be installed on the system:
- macOS: `brew install tesseract`
- Ubuntu: `apt-get install tesseract-ocr`
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

```python
from unifex import TesseractOcrExtractor

# For images
with TesseractOcrExtractor("image.png", languages=["en"]) as extractor:
    result = extractor.extract()

# For PDFs (auto-converts to images)
with TesseractOcrExtractor("scanned.pdf", languages=["en"], dpi=200) as extractor:
    result = extractor.extract()
```

### OCR Extraction (Local - PaddleOCR)

PaddleOCR provides excellent accuracy for multiple languages, especially Chinese.

```python
from unifex import PaddleOcrExtractor

# For images
with PaddleOcrExtractor("image.png", lang="en") as extractor:
    result = extractor.extract()

# For PDFs (auto-converts to images)
with PaddleOcrExtractor("scanned.pdf", lang="en", dpi=200) as extractor:
    result = extractor.extract()

# For Chinese text
with PaddleOcrExtractor("chinese_doc.png", lang="ch") as extractor:
    result = extractor.extract()
```

### OCR Extraction (Cloud - Azure)

```python
from unifex import AzureDocumentIntelligenceExtractor

with AzureDocumentIntelligenceExtractor(
    "document.pdf",
    endpoint="https://your-resource.cognitiveservices.azure.com",
    key="your-api-key",
) as extractor:
    result = extractor.extract()
```

### OCR Extraction (Cloud - Google Document AI)

```python
from unifex import GoogleDocumentAIExtractor

with GoogleDocumentAIExtractor(
    "document.pdf",
    processor_name="projects/your-project/locations/us/processors/your-processor-id",
    credentials_path="/path/to/service-account.json",
) as extractor:
    result = extractor.extract()
```

### LLM Extraction

Extract structured data from documents using vision-capable LLMs. Supports OpenAI, Anthropic, Google, and Azure OpenAI.

```python
from unifex.llm import extract_structured

# Free-form extraction (returns dict)
result = extract_structured(
    "invoice.pdf",
    model="openai/gpt-4o",
)
print(result.data)  # {"invoice_number": "INV-001", "total": 150.00, ...}

# With custom prompt
result = extract_structured(
    "receipt.png",
    model="anthropic/claude-sonnet-4-20250514",
    prompt="Extract the merchant name, date, and total amount",
)
```

#### Structured Extraction with Pydantic Schema

Define a Pydantic model and get type-safe structured output:

```python
from pydantic import BaseModel
from unifex.llm import extract_structured

class Invoice(BaseModel):
    invoice_number: str
    date: str
    total: float
    items: list[dict]

result = extract_structured(
    "invoice.pdf",
    model="openai/gpt-4o",
    schema=Invoice,
)
invoice: Invoice = result.data  # Typed as Invoice
print(f"Invoice {invoice.invoice_number}: ${invoice.total}")
```

#### OpenAI-Compatible APIs (vLLM, Ollama, etc.)

Use custom base URLs for self-hosted or alternative OpenAI-compatible APIs:

```python
from unifex.llm import extract_structured

# vLLM server
result = extract_structured(
    "document.pdf",
    model="openai/meta-llama/Llama-3.2-90B-Vision-Instruct",
    base_url="http://localhost:8000/v1",
)

# Ollama
result = extract_structured(
    "document.pdf",
    model="openai/llava",
    base_url="http://localhost:11434/v1",
)

# With custom headers
result = extract_structured(
    "document.pdf",
    model="openai/gpt-4o",
    base_url="https://your-proxy.com/v1",
    headers={"X-Custom-Auth": "your-token"},
)
```

#### Parallel Extraction

Process multiple pages in parallel for faster extraction:

```python
from unifex.llm import extract_structured

# Sequential: all pages sent in one request (default)
result = extract_structured("document.pdf", model="openai/gpt-4o")

# Parallel: each page processed separately with 4 concurrent workers
result = extract_structured(
    "large_document.pdf",
    model="openai/gpt-4o",
    max_workers=4,
)
# result.data is a list of per-page results
# result.usage contains aggregated token usage
```

#### Async API

```python
import asyncio
from unifex.llm import extract_structured_async

async def extract():
    result = await extract_structured_async(
        "document.pdf",
        model="openai/gpt-4o",
        max_workers=4,  # Concurrent requests limited by semaphore
    )
    return result.data

data = asyncio.run(extract())
```

## CLI Usage

```bash
# PDF extraction
uv run python -m unifex.cli document.pdf --extractor pdf

# EasyOCR extraction (works for both images and PDFs)
uv run python -m unifex.cli image.png --extractor easyocr --lang en,it
uv run python -m unifex.cli scanned.pdf --extractor easyocr --lang en

# Parallel extraction with 4 workers
uv run python -m unifex.cli large_document.pdf --extractor easyocr --workers 4

# Use process executor instead of threads
uv run python -m unifex.cli document.pdf --extractor easyocr --workers 4 --executor process

# Tesseract OCR
uv run python -m unifex.cli document.pdf --extractor tesseract --lang eng

# PaddleOCR
uv run python -m unifex.cli document.pdf --extractor paddle --lang en

# Azure Document Intelligence (credentials via CLI or env vars)
uv run python -m unifex.cli document.pdf --extractor azure-di \
    --azure-endpoint https://your-resource.cognitiveservices.azure.com \
    --azure-key your-api-key

# Or use environment variables
export UNIFEX_AZURE_DI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
export UNIFEX_AZURE_DI_KEY=your-api-key
uv run python -m unifex.cli document.pdf --extractor azure-di

# Google Document AI
uv run python -m unifex.cli document.pdf --extractor google-docai \
    --google-processor-name projects/your-project/locations/us/processors/123 \
    --google-credentials-path /path/to/credentials.json

# JSON output
uv run python -m unifex.cli document.pdf --extractor pdf --json

# Specific pages
uv run python -m unifex.cli document.pdf --extractor pdf --pages 0,1,2

# LLM extraction (free-form)
uv run python -m unifex.cli invoice.pdf --llm openai/gpt-4o

# LLM extraction with custom prompt
uv run python -m unifex.cli receipt.png --llm anthropic/claude-sonnet-4-20250514 \
    --llm-prompt "Extract merchant name, date, and total"

# LLM with parallel workers (each page processed separately)
uv run python -m unifex.cli large_document.pdf --llm openai/gpt-4o --workers 4

# LLM with OpenAI-compatible API (vLLM, Ollama, etc.)
uv run python -m unifex.cli document.pdf --llm openai/llava \
    --llm-base-url http://localhost:11434/v1

# LLM with custom headers
uv run python -m unifex.cli document.pdf --llm openai/gpt-4o \
    --llm-base-url https://your-proxy.com/v1 \
    --llm-header "X-Custom-Auth=your-token"

# LLM JSON output
uv run python -m unifex.cli document.pdf --llm openai/gpt-4o --json
```

## Environment Variables

Cloud extractors and LLM providers support configuration via environment variables:

**OCR Extractors:**

| Variable | Description |
|----------|-------------|
| `UNIFEX_AZURE_DI_ENDPOINT` | Azure Document Intelligence endpoint URL |
| `UNIFEX_AZURE_DI_KEY` | Azure Document Intelligence API key |
| `UNIFEX_AZURE_DI_MODEL` | Azure model ID (default: `prebuilt-read`) |
| `UNIFEX_GOOGLE_DOCAI_PROCESSOR_NAME` | Google Document AI processor name |
| `UNIFEX_GOOGLE_DOCAI_CREDENTIALS_PATH` | Path to Google service account JSON |

**LLM Providers:**

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google AI API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI API version (default: `2024-02-15-preview`) |

## Development

### Setup

```bash
# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run fast tests only (unit tests, <0.5s per test)
uv run pytest tests/base tests/ocr

# Run integration tests only (slow, load ML models)
uv run pytest tests/integration

# Run with coverage
uv run pytest --cov=unifex --cov-report=term-missing
```

### Test Structure

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

### Integration Tests

Integration tests load real ML models and call real services. They are in `tests/integration/`.

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
   UNIFEX_AZURE_DI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
   UNIFEX_AZURE_DI_KEY=your-api-key
   ```

3. Load environment variables before running tests:
   ```bash
   # Option 1: Source the .env file
   export $(cat .env | xargs)
   uv run pytest tests/test_integration.py -v

   # Option 2: Use env command
   env $(cat .env | xargs) uv run pytest tests/test_integration.py -v
   ```

Azure integration tests are automatically skipped if credentials are not configured.

#### Google Document AI Credentials Setup

1. Create a Google Cloud project and enable the Document AI API
2. Create a Document AI processor in the Google Cloud Console
3. Create a service account with Document AI permissions
4. Download the service account JSON key file

5. Edit `.env` with your Google Document AI credentials:
   ```
   UNIFEX_GOOGLE_DOCAI_PROCESSOR_NAME=projects/your-project/locations/us/processors/your-processor-id
   UNIFEX_GOOGLE_DOCAI_CREDENTIALS_PATH=/path/to/your/service-account.json
   ```

Google Document AI integration tests are automatically skipped if credentials are not configured.

### Documentation

Build and serve the documentation locally:

```bash
# Serve docs with live reload
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

Open http://localhost:8000 to view the documentation.

### Pre-commit Checks

The pre-commit hook runs automatically on `git commit`. To run manually:

```bash
uv run pre-commit run --all-files
```

This runs:
- `ruff format` - Code formatting
- `ruff check --fix` - Linting with auto-fix
- `ty check` - Type checking
- `pytest` - Test suite

## Architecture

```
unifex/
├── cli.py              # Command-line interface
├── coordinates.py      # Coordinate unit conversions (POINTS, PIXELS, INCHES, NORMALIZED)
├── models.py           # Core data models (Document, Page, TextBlock, BBox)
├── extractors/         # PDF text extraction
│   ├── base.py         # Base extractor class
│   ├── factory.py      # Unified factory interface
│   ├── pdf.py          # Native PDF extraction via pypdfium2
│   └── character_mergers.py  # Text merging strategies
├── ocr/                # OCR extraction
│   ├── adapters/       # External API → internal models
│   │   ├── azure_di.py
│   │   ├── google_docai.py
│   │   ├── easy_ocr.py
│   │   ├── paddle_ocr.py
│   │   └── tesseract_ocr.py
│   └── extractors/     # OCR extractor implementations
│       ├── azure_di.py
│       ├── google_docai.py
│       ├── easy_ocr.py
│       ├── paddle_ocr.py
│       └── tesseract_ocr.py
├── llm/                # LLM-based extraction
│   ├── factory.py      # LLM extractor factory
│   ├── models.py       # LLM-specific models
│   ├── adapters/
│   │   └── image_encoder.py  # Image encoding for LLM input
│   └── extractors/     # LLM provider implementations
│       ├── anthropic.py
│       ├── openai.py
│       ├── azure_openai.py
│       └── google.py
└── utils/              # Shared utilities
    ├── geometry.py     # Geometric calculations
    └── image_loader.py # Image loading utilities
```

### Extractors

**PDF Extraction:**
- `PdfExtractor` - Native PDF text extraction via pypdfium2

**OCR Extraction:**
- `EasyOcrExtractor` - Image/PDF OCR via EasyOCR
- `TesseractOcrExtractor` - Image/PDF OCR via Tesseract
- `PaddleOcrExtractor` - Image/PDF OCR via PaddleOCR
- `AzureDocumentIntelligenceExtractor` - Azure cloud OCR
- `GoogleDocumentAIExtractor` - Google Cloud Document AI

**LLM Extraction:**
- `AnthropicExtractor` - Claude-based text extraction
- `OpenAIExtractor` - GPT-based text extraction
- `AzureOpenAIExtractor` - Azure OpenAI text extraction
- `GoogleExtractor` - Gemini-based text extraction

### Adapters

Schema transformation from external APIs to internal models:
- **OCR adapters** - Convert Azure, Google, EasyOCR, PaddleOCR, Tesseract results to `Page`/`TextBlock`
- **LLM adapters** - Handle image encoding for LLM input

### Models

Pydantic models for type-safe document representation:
- `Document` - Full document with pages and metadata
- `Page` - Single page with text blocks and tables
- `TextBlock` - Text with bounding box and confidence
- `Table` - Extracted table with rows and columns
- `BBox` - Bounding box coordinates
- `ExtractorMetadata` - Extractor type and processing details

## Work test times

Please keep in mind EasyOCR solution performance slows downs with bigger images and scale. The current overview for small PDF and images with dpi=100 (lower faster).

```bash
11.84s call     tests/test_integration.py::test_ocr_extract_pdf[easyocr]
4.79s call     tests/test_integration.py::test_ocr_extract_pdf[google]
3.64s call     tests/test_integration.py::test_ocr_extract_pdf[azure]
3.58s call     tests/test_integration.py::test_ocr_extract_image[easyocr]
3.01s call     tests/test_integration.py::test_ocr_extract_pdf[paddle]
1.20s call     tests/test_integration.py::test_ocr_extract_image[paddle]
0.94s call     tests/test_factory.py::TestCreateExtractorWithRealFiles::test_creates_paddle_with_gpu_flag
0.94s call     tests/test_factory.py::TestCreateExtractorWithRealFiles::test_creates_paddle_extractor
0.48s call     tests/test_integration.py::test_ocr_extract_pdf[tesseract]
0.15s call     tests/test_integration.py::test_ocr_extract_image[tesseract]
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Future plans

- Detecting language helper
- Performance measurement
