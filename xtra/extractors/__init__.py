from xtra.extractors.azure_di import AzureDocumentIntelligenceExtractor
from xtra.extractors.base import BaseExtractor, ExtractionResult
from xtra.extractors.factory import create_extractor
from xtra.extractors.google_docai import GoogleDocumentAIExtractor
from xtra.extractors.easy_ocr import EasyOcrExtractor
from xtra.extractors.paddle_ocr import PaddleOcrExtractor
from xtra.extractors.pdf import PdfExtractor
from xtra.extractors.tesseract_ocr import TesseractOcrExtractor

# Backward compatibility alias (deprecated)
OcrExtractor = EasyOcrExtractor

__all__ = [
    "AzureDocumentIntelligenceExtractor",
    "BaseExtractor",
    "ExtractionResult",
    "GoogleDocumentAIExtractor",
    "PdfExtractor",
    "EasyOcrExtractor",
    "TesseractOcrExtractor",
    "PaddleOcrExtractor",
    "create_extractor",
    # Deprecated alias
    "OcrExtractor",
]
