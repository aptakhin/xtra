# Extractors Reference

## PDF Extractor

### PdfExtractor

Native PDF text extraction using pypdfium2.

::: xtra.pdf.PdfExtractor

## Local OCR Extractors

### EasyOcrExtractor

OCR using EasyOCR library.

::: xtra.ocr.extractors.easy_ocr.EasyOcrExtractor

### TesseractOcrExtractor

OCR using Tesseract.

::: xtra.ocr.extractors.tesseract_ocr.TesseractOcrExtractor

### PaddleOcrExtractor

OCR using PaddleOCR.

::: xtra.ocr.extractors.paddle_ocr.PaddleOcrExtractor

## Cloud OCR Extractors

### AzureDocumentIntelligenceExtractor

Azure Document Intelligence OCR.

::: xtra.ocr.extractors.azure_di.AzureDocumentIntelligenceExtractor

### GoogleDocumentAIExtractor

Google Document AI OCR.

::: xtra.ocr.extractors.google_docai.GoogleDocumentAIExtractor

## LLM Extractors

### extract_structured

Synchronous LLM extraction function.

::: xtra.llm_factory.extract_structured

### extract_structured_async

Asynchronous LLM extraction function.

::: xtra.llm_factory.extract_structured_async
