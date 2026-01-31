"""Factory for creating document extractors by type."""

from __future__ import annotations

import os
from pathlib import Path

from xtra.base import BaseExtractor, CoordinateUnit, ExtractorType
from xtra.pdf import (
    BasicLineMerger,
    CharacterMerger,
    KeepCharacterMerger,
)

# Available character merger choices for PDF extractor
CHARACTER_MERGER_CHOICES = {
    "basic-line": BasicLineMerger,
    "keep-char": KeepCharacterMerger,
}


def get_character_merger(name: str) -> CharacterMerger:
    """Get a character merger instance by name.

    Args:
        name: Merger name - one of: basic-line, keep-char

    Returns:
        CharacterMerger instance.

    Raises:
        ValueError: If name is not a valid merger choice.
    """
    if name not in CHARACTER_MERGER_CHOICES:
        valid = ", ".join(CHARACTER_MERGER_CHOICES.keys())
        raise ValueError(f"Unknown character merger: {name}. Valid choices: {valid}")
    return CHARACTER_MERGER_CHOICES[name]()


def _get_credential(key: str, credentials: dict[str, str] | None) -> str | None:
    """Get credential from dict or environment variable."""
    if credentials and key in credentials:
        return credentials[key]
    return os.environ.get(key)


def create_extractor(  # noqa: PLR0913
    path: Path | str,
    extractor_type: ExtractorType,
    *,
    languages: list[str] | None = None,
    dpi: int = 200,
    use_gpu: bool = False,
    credentials: dict[str, str] | None = None,
    output_unit: CoordinateUnit = CoordinateUnit.POINTS,
    character_merger: str | None = None,
) -> BaseExtractor:
    """Create an extractor by type with unified parameters.

    Args:
        path: Path to document/image file (Path object or string).
        extractor_type: ExtractorType enum value specifying which extractor to use:
            - ExtractorType.PDF - Native PDF extraction
            - ExtractorType.EASYOCR - EasyOCR for images and PDFs (auto-detects)
            - ExtractorType.TESSERACT - Tesseract for images and PDFs (auto-detects)
            - ExtractorType.PADDLE - PaddleOCR for images and PDFs (auto-detects)
            - ExtractorType.AZURE_DI - Azure Document Intelligence
            - ExtractorType.GOOGLE_DOCAI - Google Document AI
        languages: Language codes for OCR (default: ["en"]).
            EasyOCR/Tesseract use full list, PaddleOCR uses first language.
        dpi: DPI for PDF-to-image conversion (default: 200).
        use_gpu: Enable GPU acceleration where supported (default: False).
        credentials: Override credentials dict. If None, reads from env vars:
            - XTRA_AZURE_DI_ENDPOINT, XTRA_AZURE_DI_KEY for Azure
            - XTRA_GOOGLE_DOCAI_PROCESSOR_NAME, XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH for Google
        output_unit: Coordinate unit for output (default: POINTS).
            - CoordinateUnit.POINTS - 1/72 inch (PDF native, resolution-independent)
            - CoordinateUnit.PIXELS - Pixels at the specified DPI
            - CoordinateUnit.INCHES - Imperial inches
            - CoordinateUnit.NORMALIZED - 0-1 range relative to page dimensions
        character_merger: Character merger strategy for PDF extractor (default: basic-line).
            - "basic-line" - Merge characters into lines
            - "keep-char" - Keep each character as separate TextBlock

    Returns:
        Configured extractor instance.

    Raises:
        ValueError: If extractor_type is invalid or required credentials are missing.

    Example:
        >>> from xtra import create_extractor, ExtractorType, CoordinateUnit
        >>> with create_extractor("doc.pdf", ExtractorType.PDF) as ext:
        ...     doc = ext.extract()  # Coordinates in points (default)
        >>> with create_extractor("doc.pdf", ExtractorType.EASYOCR,
        ...                       output_unit=CoordinateUnit.PIXELS) as ext:
        ...     doc = ext.extract()  # Coordinates in pixels
    """
    languages = languages or ["en"]

    if extractor_type == ExtractorType.PDF:
        from xtra.pdf import PdfExtractor

        merger = get_character_merger(character_merger) if character_merger else None
        return PdfExtractor(path, output_unit=output_unit, character_merger=merger)

    elif extractor_type == ExtractorType.EASYOCR:
        from xtra.ocr.extractors.easy_ocr import EasyOcrExtractor

        return EasyOcrExtractor(
            path, languages=languages, gpu=use_gpu, dpi=dpi, output_unit=output_unit
        )

    elif extractor_type == ExtractorType.TESSERACT:
        from xtra.ocr.extractors.tesseract_ocr import TesseractOcrExtractor

        return TesseractOcrExtractor(path, languages=languages, dpi=dpi, output_unit=output_unit)

    elif extractor_type == ExtractorType.PADDLE:
        from xtra.ocr.extractors.paddle_ocr import PaddleOcrExtractor

        # PaddleOCR uses single language string
        lang = languages[0] if languages else "en"
        return PaddleOcrExtractor(
            path, lang=lang, use_gpu=use_gpu, dpi=dpi, output_unit=output_unit
        )

    elif extractor_type == ExtractorType.AZURE_DI:
        from xtra.ocr.extractors.azure_di import AzureDocumentIntelligenceExtractor

        endpoint = _get_credential("XTRA_AZURE_DI_ENDPOINT", credentials)
        key = _get_credential("XTRA_AZURE_DI_KEY", credentials)

        if not endpoint or not key:
            raise ValueError(
                "Azure credentials required. Set XTRA_AZURE_DI_ENDPOINT and XTRA_AZURE_DI_KEY "
                "environment variables or pass credentials dict."
            )

        return AzureDocumentIntelligenceExtractor(
            path, endpoint=endpoint, key=key, output_unit=output_unit
        )

    elif extractor_type == ExtractorType.GOOGLE_DOCAI:
        from xtra.ocr.extractors.google_docai import GoogleDocumentAIExtractor

        processor_name = _get_credential("XTRA_GOOGLE_DOCAI_PROCESSOR_NAME", credentials)
        credentials_path = _get_credential("XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH", credentials)

        if not processor_name:
            raise ValueError(
                "Google Document AI processor name required. Set XTRA_GOOGLE_DOCAI_PROCESSOR_NAME "
                "environment variable or pass credentials dict."
            )

        if not credentials_path:
            raise ValueError(
                "Google Document AI credentials path required. "
                "Set XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH env var or pass credentials dict."
            )

        return GoogleDocumentAIExtractor(
            path,
            processor_name=processor_name,
            credentials_path=credentials_path,
            output_unit=output_unit,
        )

    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
