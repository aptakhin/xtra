from xtra.adapters import AzureDocumentIntelligenceAdapter
from xtra.extractors import (
    AzureDocumentIntelligenceExtractor,
    BaseExtractor,
    EasyOcrExtractor,
    ExtractionResult,
    GoogleDocumentAIExtractor,
    PaddleOcrExtractor,
    PdfExtractor,
    TesseractOcrExtractor,
    create_extractor,
)
from xtra.models import (
    BBox,
    CoordinateUnit,
    ExtractorMetadata,
    ExtractorType,
    FontInfo,
    Page,
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
    # Models
    "BBox",
    "CoordinateUnit",
    "FontInfo",
    "TextBlock",
    "Page",
    "ExtractorMetadata",
    "ExtractorType",
]
