"""Tesseract OCR extractor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from xtra.adapters.tesseract_ocr import TesseractAdapter
from xtra.extractors._image_loader import ImageLoader
from xtra.extractors.base import BaseExtractor, PageExtractionResult
from xtra.models import (
    CoordinateUnit,
    ExtractorMetadata,
    ExtractorType,
    Page,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _check_pytesseract_installed() -> None:
    """Check if pytesseract is installed, raise ImportError with helpful message if not."""
    try:
        import pytesseract  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pytesseract is not installed. Install it with: pip install xtra[tesseract]"
        ) from e


# ISO 639-1 (2-letter) to Tesseract (3-letter) language code mapping
# Users provide 2-letter codes, we convert internally for Tesseract
LANG_CODE_MAP = {
    "en": "eng",
    "it": "ita",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "pt": "por",
    "nl": "nld",
    "ru": "rus",
    "zh": "chi_sim",
    "ja": "jpn",
    "ko": "kor",
    "ar": "ara",
    "hi": "hin",
    "pl": "pol",
    "uk": "ukr",
    "vi": "vie",
    "th": "tha",
    "tr": "tur",
    "el": "ell",
    "he": "heb",
    "cs": "ces",
    "sv": "swe",
    "da": "dan",
    "fi": "fin",
    "no": "nor",
    "hu": "hun",
    "ro": "ron",
    "bg": "bul",
    "hr": "hrv",
    "sk": "slk",
    "sl": "slv",
    "sr": "srp",
    "id": "ind",
    "ms": "msa",
    "fa": "fas",
}


def _convert_lang_code(code: str) -> str:
    """Convert 2-letter ISO 639-1 code to Tesseract 3-letter code.

    If already a 3-letter code or not in mapping, returns as-is.
    """
    return LANG_CODE_MAP.get(code, code)


class TesseractOcrExtractor(BaseExtractor):
    """Extract text from images or PDFs using Tesseract OCR.

    Composes ImageLoader for image handling, Tesseract for OCR processing,
    and TesseractAdapter for result conversion.
    """

    def __init__(
        self,
        path: Path | str,
        languages: Optional[List[str]] = None,
        dpi: int = 200,
        output_unit: CoordinateUnit = CoordinateUnit.POINTS,
    ) -> None:
        """Initialize Tesseract OCR extractor.

        Args:
            path: Path to the image or PDF file (Path object or string).
            languages: List of 2-letter ISO 639-1 language codes (e.g., ["en", "fr"]).
                       Defaults to ["en"]. Codes are converted to Tesseract format internally.
            dpi: DPI for PDF-to-image conversion. Default 200.
            output_unit: Coordinate unit for output. Default POINTS.
        """
        _check_pytesseract_installed()
        super().__init__(path, output_unit)
        input_languages = languages or ["en"]
        # Store original 2-letter codes for metadata
        self.languages = input_languages
        # Convert to Tesseract format for internal use
        self._tesseract_languages = [_convert_lang_code(lang) for lang in input_languages]
        self.dpi = dpi

        # Compose components
        self._images = ImageLoader(self.path, dpi)
        self._adapter = TesseractAdapter()

    def get_page_count(self) -> int:
        """Return number of pages/images loaded."""
        return self._images.page_count

    def extract_page(self, page: int) -> PageExtractionResult:
        """Extract text from a single image/page."""
        import pytesseract

        try:
            img = self._images.get_page(page)
            width, height = img.size

            # Run OCR pipeline
            lang_str = "+".join(self._tesseract_languages)
            data = pytesseract.image_to_data(
                img, lang=lang_str, output_type=pytesseract.Output.DICT
            )
            text_blocks = self._adapter.convert_result(data)

            result_page = Page(
                page=page,
                width=float(width),
                height=float(height),
                texts=text_blocks,
            )

            # Convert from native PIXELS to output_unit
            result_page = self._convert_page(result_page, CoordinateUnit.PIXELS, self.dpi)
            return PageExtractionResult(page=result_page, success=True)

        except Exception as e:
            logger.warning("Failed to extract page %d: %s", page, e)
            return PageExtractionResult(
                page=Page(page=page, width=0, height=0, texts=[]),
                success=False,
                error=str(e),
            )

    def get_extractor_metadata(self) -> ExtractorMetadata:
        """Return extractor metadata."""
        extra = {"ocr_engine": "tesseract", "languages": self.languages}
        if self._images.is_pdf:
            extra["dpi"] = self.dpi
        return ExtractorMetadata(
            extractor_type=ExtractorType.TESSERACT,
            extra=extra,
        )

    def close(self) -> None:
        """Release resources."""
        self._images.close()
