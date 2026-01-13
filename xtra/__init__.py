from .adapters import AzureDocumentIntelligenceAdapter
from .extractors import (
    AzureDocumentIntelligenceExtractor,
    BaseExtractor,
    EasyOcrExtractor,
    ExtractionResult,
    OcrExtractor,
    PdfExtractor,
    PdfToImageEasyOcrExtractor,
    PdfToImageOcrExtractor,
)
from .models import (
    BBox,
    Document,
    DocumentMetadata,
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
    "PdfExtractor",
    "EasyOcrExtractor",
    "PdfToImageEasyOcrExtractor",
    # Deprecated aliases
    "OcrExtractor",
    "PdfToImageOcrExtractor",
    # Models
    "BBox",
    "FontInfo",
    "TextBlock",
    "Page",
    "PdfObjectInfo",
    "DocumentMetadata",
    "Document",
    "SourceType",
]
