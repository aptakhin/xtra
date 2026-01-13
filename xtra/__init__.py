from .adapters import AzureDocumentIntelligenceAdapter
from .extractors import (
    AzureDocumentIntelligenceExtractor,
    BaseExtractor,
    EasyOcrExtractor,
    ExtractionResult,
    GoogleDocumentAIExtractor,
    OcrExtractor,
    PaddleOcrExtractor,
    PdfExtractor,
    TesseractOcrExtractor,
    create_extractor,
)
from .models import (
    BBox,
    CoordinateUnit,
    Document,
    DocumentMetadata,
    ExtractorType,
    FontInfo,
    Page,
    PdfObjectInfo,
    SourceType,
    TextBlock,
)

__all__ = [
    # Adapters
    "AzureDocumentIntelligenceAdapter",
    # Extractors
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
    # Models
    "BBox",
    "CoordinateUnit",
    "FontInfo",
    "TextBlock",
    "Page",
    "PdfObjectInfo",
    "DocumentMetadata",
    "Document",
    "ExtractorType",
    # Deprecated alias
    "SourceType",
]
