from xtra.extractors.base import (
    BaseExtractor,
    ExecutorType,
    ExtractionResult,
    PageExtractionResult,
)
from xtra.extractors.character_mergers import (
    BasicLineMerger,
    CharacterMerger,
    CharInfo,
    KeepCharacterMerger,
)
from xtra.extractors.factory import create_extractor
from xtra.extractors.pdf import PdfExtractor

__all__ = [
    # Core (always available)
    "BaseExtractor",
    "BasicLineMerger",
    "CharacterMerger",
    "CharInfo",
    "ExecutorType",
    "ExtractionResult",
    "KeepCharacterMerger",
    "PageExtractionResult",
    "PdfExtractor",
    "create_extractor",
    # Optional (lazy loaded)
    "AzureDocumentIntelligenceExtractor",
    "GoogleDocumentAIExtractor",
    "EasyOcrExtractor",
    "TesseractOcrExtractor",
    "PaddleOcrExtractor",
]

# Lazy loading for optional dependencies
_LAZY_IMPORTS = {
    "AzureDocumentIntelligenceExtractor": "xtra.extractors.azure_di",
    "GoogleDocumentAIExtractor": "xtra.extractors.google_docai",
    "EasyOcrExtractor": "xtra.extractors.easy_ocr",
    "TesseractOcrExtractor": "xtra.extractors.tesseract_ocr",
    "PaddleOcrExtractor": "xtra.extractors.paddle_ocr",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
