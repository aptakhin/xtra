# Quick Start

## Factory Interface (Recommended)

The simplest way to use xtra is via the factory interface:

```python
from xtra import create_extractor, ExtractorType

# PDF extraction (native text)
with create_extractor("document.pdf", ExtractorType.PDF) as extractor:
    result = extractor.extract()
    print(f"Extracted {len(result.document.pages)} pages")
```

<!-- skip: next -->
```python
from xtra import create_extractor, ExtractorType

# EasyOCR for images
with create_extractor("image.png", ExtractorType.EASYOCR, languages=["en"]) as extractor:
    result = extractor.extract()

# EasyOCR for PDFs (auto-converts to images internally)
with create_extractor("scanned.pdf", ExtractorType.EASYOCR, dpi=200) as extractor:
    result = extractor.extract()
```

<!-- skip: next -->
```python
from xtra import create_extractor, ExtractorType

# Azure Document Intelligence (credentials from env vars)
with create_extractor("document.pdf", ExtractorType.AZURE_DI) as extractor:
    result = extractor.extract()
```

## Understanding the Result

The `extract()` method returns an `ExtractionResult` containing the `Document` and per-page results:

```python
from xtra import create_extractor, ExtractorType

with create_extractor("document.pdf", ExtractorType.PDF) as extractor:
    result = extractor.extract()

# Check extraction status
print(f"Success: {result.success}")

# Access extracted document
doc = result.document
print(f"Pages: {len(doc.pages)}")

for page in doc.pages:
    print(f"Page {page.page + 1} ({page.width:.0f}x{page.height:.0f}):")
    for text in page.texts[:2]:  # Show first 2 texts per page
        print(f"  - \"{text.text}\"")

# Handle errors if any
if not result.success:
    for page_num, error in result.errors:
        print(f"Page {page_num} failed: {error}")
```

## Direct Extractor Usage

You can also use extractors directly without the factory:

```python
from xtra import PdfExtractor

with PdfExtractor("document.pdf") as extractor:
    result = extractor.extract()
    for page in result.document.pages:
        for text in page.texts[:2]:  # Show first 2 texts per page
            print(text.text)
```

## Next Steps

- [PDF Extraction](../guide/pdf-extraction.md) - Native PDF text extraction
- [OCR Extraction](../guide/ocr-extraction.md) - Local and cloud OCR options
- [LLM Extraction](../guide/llm-extraction.md) - Structured data extraction with LLMs
- [Parallel Processing](../guide/parallel-processing.md) - Speed up extraction with parallelism
