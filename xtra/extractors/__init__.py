from .azure_di import AzureDocumentIntelligenceExtractor
from .base import BaseExtractor, ExtractionResult
from .ocr import EasyOcrExtractor, PdfToImageEasyOcrExtractor
from .pdf import PdfExtractor

# Backward compatibility aliases (deprecated)
OcrExtractor = EasyOcrExtractor
PdfToImageOcrExtractor = PdfToImageEasyOcrExtractor

__all__ = [
    "AzureDocumentIntelligenceExtractor",
    "BaseExtractor",
    "ExtractionResult",
    "PdfExtractor",
    "EasyOcrExtractor",
    "PdfToImageEasyOcrExtractor",
    # Deprecated aliases
    "OcrExtractor",
    "PdfToImageOcrExtractor",
]
