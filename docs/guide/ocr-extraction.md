# OCR Extraction

xtra supports multiple OCR backends for extracting text from images and scanned PDFs.

## Language Codes

All OCR extractors use **2-letter ISO 639-1 language codes** (e.g., `"en"`, `"fr"`, `"de"`, `"it"`).
Extractors that require different formats (like Tesseract) convert internally.

## Local OCR Engines

### EasyOCR

<!-- skip: next -->
```python
from xtra import EasyOcrExtractor

# For images
with EasyOcrExtractor("image.png", languages=["en"]) as extractor:
    result = extractor.extract()

# For PDFs (auto-converts to images)
with EasyOcrExtractor("scanned.pdf", languages=["en"], dpi=200) as extractor:
    result = extractor.extract()
```

### Tesseract

Requires Tesseract to be installed on the system:

- **macOS**: `brew install tesseract`
- **Ubuntu**: `apt-get install tesseract-ocr`
- **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

<!-- skip: next -->
```python
from xtra import TesseractOcrExtractor

# For images
with TesseractOcrExtractor("image.png", languages=["en"]) as extractor:
    result = extractor.extract()

# For PDFs (auto-converts to images)
with TesseractOcrExtractor("scanned.pdf", languages=["en"], dpi=200) as extractor:
    result = extractor.extract()
```

### PaddleOCR

Excellent accuracy for multiple languages, especially Chinese.

<!-- skip: next -->
```python
from xtra import PaddleOcrExtractor

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

## Cloud OCR Services

### Azure Document Intelligence

<!-- skip: next -->
```python
from xtra import AzureDocumentIntelligenceExtractor

with AzureDocumentIntelligenceExtractor(
    "document.pdf",
    endpoint="https://your-resource.cognitiveservices.azure.com",
    key="your-api-key",
) as extractor:
    result = extractor.extract()
```

Or use environment variables:

```bash
export XTRA_AZURE_DI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
export XTRA_AZURE_DI_KEY=your-api-key
```

### Google Document AI

<!-- skip: next -->
```python
from xtra import GoogleDocumentAIExtractor

with GoogleDocumentAIExtractor(
    "document.pdf",
    processor_name="projects/your-project/locations/us/processors/your-processor-id",
    credentials_path="/path/to/service-account.json",
) as extractor:
    result = extractor.extract()
```

Or use environment variables:

```bash
export XTRA_GOOGLE_DOCAI_PROCESSOR_NAME=projects/your-project/locations/us/processors/123
export XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH=/path/to/credentials.json
```

## Parallel Extraction

All OCR extractors support parallel page extraction:

<!-- skip: next -->
```python
from xtra import EasyOcrExtractor

with EasyOcrExtractor("scanned.pdf", languages=["en"]) as extractor:
    result = extractor.extract(max_workers=4)
```

See [Parallel Processing](parallel-processing.md) for more details.

## Coordinate Units

Control the output coordinate system:

<!-- skip: next -->
```python
from xtra import EasyOcrExtractor, CoordinateUnit

# Pixels (default for OCR, uses DPI for conversion)
with EasyOcrExtractor("image.png", languages=["en"],
                       output_unit=CoordinateUnit.PIXELS, dpi=150) as extractor:
    result = extractor.extract()

# Points (1/72 inch)
with EasyOcrExtractor("image.png", languages=["en"],
                       output_unit=CoordinateUnit.POINTS) as extractor:
    result = extractor.extract()

# Normalized (0-1 range)
with EasyOcrExtractor("image.png", languages=["en"],
                       output_unit=CoordinateUnit.NORMALIZED) as extractor:
    result = extractor.extract()
```

## Choosing an OCR Engine

| Engine | Best For | Speed | Accuracy |
|--------|----------|-------|----------|
| **EasyOCR** | General purpose, many languages | Medium | High |
| **Tesseract** | Fast processing, good accuracy | Fast | Medium-High |
| **PaddleOCR** | Chinese text, high accuracy | Medium | Very High |
| **Azure DI** | Production workloads, tables | Fast | Very High |
| **Google DocAI** | Production workloads, forms | Fast | Very High |
