"""OCR extraction module with extractors and adapters."""

from xtra.ocr.extractors.azure_di import AzureDocumentIntelligenceExtractor
from xtra.ocr.extractors.easy_ocr import EasyOcrExtractor
from xtra.ocr.extractors.google_docai import GoogleDocumentAIExtractor
from xtra.ocr.extractors.paddle_ocr import PaddleOcrExtractor
from xtra.ocr.extractors.tesseract_ocr import TesseractOcrExtractor

__all__ = [
    "AzureDocumentIntelligenceExtractor",
    "EasyOcrExtractor",
    "GoogleDocumentAIExtractor",
    "PaddleOcrExtractor",
    "TesseractOcrExtractor",
]
