# Architecture

## Project Structure

```
xtra/
├── cli.py              # Command-line interface
├── coordinates.py      # Coordinate unit conversions
├── models.py           # Core data models
├── text_factory.py     # Unified factory interface
├── base/               # Base classes and models
│   ├── base.py         # BaseExtractor class
│   ├── models.py       # Document, Page, TextBlock, etc.
│   └── coordinates.py  # Coordinate conversion logic
├── pdf/                # PDF text extraction
│   ├── pdf.py          # PdfExtractor implementation
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
│   ├── factory.py      # extract_structured functions
│   ├── models.py       # LLM-specific models
│   ├── adapters/
│   │   └── image_encoder.py
│   └── extractors/
│       ├── anthropic.py
│       ├── openai.py
│       ├── azure_openai.py
│       └── google.py
└── utils/              # Shared utilities
    ├── geometry.py
    └── image_loader.py
```

## Layered Architecture

The project follows a layered architecture enforced by import-linter:

```
cli.py
   ↓
text_factory.py
   ↓
pdf/, ocr/, llm/
   ↓
base/
```

### Rules

1. **OCR and LLM are independent** - They don't import from each other
2. **Base has no upward dependencies** - It doesn't import from pdf, ocr, llm, or cli
3. **OCR extractors are independent** - Each OCR extractor is self-contained

## Adapter Pattern

Adapters transform external API responses to internal models:

```
External API Response → Adapter → Page/TextBlock
```

This keeps extractors clean and makes it easy to:
- Add new OCR providers
- Update when APIs change
- Test transformations in isolation

## Extractor Interface

All extractors implement `BaseExtractor`:

```python
class BaseExtractor:
    def extract(
        self,
        pages: Sequence[int] | None = None,
        max_workers: int = 1,
        executor: ExecutorType = ExecutorType.THREAD,
        **kwargs,
    ) -> ExtractionResult: ...

    async def extract_async(
        self,
        pages: Sequence[int] | None = None,
        max_workers: int = 1,
        **kwargs,
    ) -> ExtractionResult: ...

    def extract_page(self, page: int, **kwargs) -> PageExtractionResult: ...

    def get_page_count(self) -> int: ...

    def close(self) -> None: ...
```

## Thread Safety

### PDF Extractor

The PDF extractor uses pypdfium2, which is **not thread-safe**. To enable parallel page extraction, `PdfExtractor` uses an internal `threading.Lock` per instance:

```python
class PdfExtractor:
    def __init__(self, ...):
        self._lock = threading.Lock()

    def extract_page(self, page: int, ...):
        with self._lock:
            pdf_page = self._pdf[page]
            # ... extraction logic
```

This means:

- **Thread executor works** - Multiple threads can call `extract_page()` on the same extractor instance; the lock serializes PDF access
- **Process executor duplicates** - Each process gets its own `PdfExtractor` instance with its own PDF handle
- **Single extractor, multiple threads** - Safe due to internal locking

### OCR Extractors

OCR extractors have varying thread-safety characteristics:

| Extractor | Thread-Safe | Notes |
|-----------|-------------|-------|
| EasyOCR | Yes | Model shared across threads |
| Tesseract | Yes | Subprocess-based |
| PaddleOCR | Yes | Model shared across threads |
| Azure DI | Yes | HTTP client is thread-safe |
| Google DocAI | Yes | gRPC client is thread-safe |

### LLM Extractors

All LLM extractors are thread-safe as they use HTTP/gRPC clients that handle concurrent requests properly.

## Coordinate System

All coordinates flow through a conversion pipeline:

```
Native Unit → Points → Output Unit
```

- PDF uses points natively
- OCR uses pixels natively (at specified DPI)
- Cloud APIs may use normalized coordinates

The `CoordinateConverter` handles all conversions.
