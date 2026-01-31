# xtra

A Python library for document text extraction with local and cloud OCR solutions.

## Overview

**xtra** is built for tasks like fraud detection where precision matters. It provides a universal tool for both PDF and image processing with best-in-class OCR support through local engines and cloud services.

## Key Features

- **Multiple OCR Backends**: Local (EasyOCR, Tesseract, PaddleOCR) and cloud (Azure Document Intelligence, Google Document AI)
- **PDF Text Extraction**: Native PDF text extraction using pypdfium2
- **LLM Extraction**: Extract structured data using GPT-4o, Claude, Gemini, or OpenAI-compatible APIs
- **Parallel Extraction**: Process multiple pages concurrently with thread or process executors
- **Async Support**: Native async/await API for integration with async applications
- **Unified Extractors**: Each OCR extractor auto-detects file type (PDF vs image) and handles conversion internally
- **Pydantic Models**: Type-safe document representation with pydantic v1/v2 compatibility

## Quick Example

```python
from xtra import create_extractor, ExtractorType

# PDF extraction (native text)
with create_extractor("document.pdf", ExtractorType.PDF) as extractor:
    result = extractor.extract()
    print(f"Extracted {len(result.document.pages)} pages")
```

## Alternatives

For broader document processing, check out [Docling](https://docling-project.github.io) and [Kreuzberg](https://kreuzberg.dev/).

## License

BSD 3-Clause License. See [LICENSE](https://github.com/yourorg/xtra/blob/main/LICENSE) for details.
