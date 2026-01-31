"""Adapters for converting OCR results to internal models."""

from xtra.ocr.adapters.azure_di import AzureDocumentIntelligenceAdapter
from xtra.ocr.adapters.easy_ocr import EasyOCRAdapter
from xtra.ocr.adapters.google_docai import GoogleDocumentAIAdapter
from xtra.ocr.adapters.paddle_ocr import PaddleOCRAdapter
from xtra.ocr.adapters.tesseract_ocr import TesseractAdapter

__all__ = [
    "AzureDocumentIntelligenceAdapter",
    "EasyOCRAdapter",
    "GoogleDocumentAIAdapter",
    "PaddleOCRAdapter",
    "TesseractAdapter",
]
