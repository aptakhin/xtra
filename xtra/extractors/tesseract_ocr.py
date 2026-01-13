"""Tesseract OCR extractor."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytesseract
from PIL import Image

from xtra.models import (
    BBox,
    CoordinateUnit,
    DocumentMetadata,
    ExtractorType,
    TextBlock,
)
from xtra.extractors._ocr_base import ImageBasedExtractor

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


class TesseractOcrExtractor(ImageBasedExtractor):
    """Extract text from images or PDFs using Tesseract OCR.

    Automatically detects file type and handles PDF-to-image conversion internally.
    """

    def __init__(
        self,
        path: Path,
        languages: Optional[List[str]] = None,
        dpi: int = 200,
        output_unit: CoordinateUnit = CoordinateUnit.POINTS,
    ) -> None:
        """Initialize Tesseract OCR extractor.

        Args:
            path: Path to the image or PDF file.
            languages: List of 2-letter ISO 639-1 language codes (e.g., ["en", "fr"]).
                       Defaults to ["en"]. Codes are converted to Tesseract format internally.
            dpi: DPI for PDF-to-image conversion. Default 200.
            output_unit: Coordinate unit for output. Default POINTS.
        """
        input_languages = languages or ["en"]
        # Store original 2-letter codes for metadata
        self.languages = input_languages
        # Convert to Tesseract format for internal use
        self._tesseract_languages = [_convert_lang_code(lang) for lang in input_languages]
        super().__init__(path, dpi, output_unit)

    def _do_ocr(self, img: Image.Image) -> List[TextBlock]:
        """Perform OCR using Tesseract."""
        # Build language string for Tesseract (e.g., "eng+fra+deu")
        lang_str = "+".join(self._tesseract_languages)

        # Get detailed OCR data with bounding boxes
        data = pytesseract.image_to_data(img, lang=lang_str, output_type=pytesseract.Output.DICT)

        return self._convert_results(data)

    def get_metadata(self) -> DocumentMetadata:
        extra = {"ocr_engine": "tesseract", "languages": self.languages}
        if self._is_pdf:
            extra["dpi"] = self.dpi
        return DocumentMetadata(
            source_type=ExtractorType.TESSERACT,
            extra=extra,
        )

    def _convert_results(self, data: dict) -> List[TextBlock]:
        """Convert Tesseract output to TextBlocks."""
        blocks = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i]
            conf = data["conf"][i]

            # Skip empty text and low/negative confidence (non-text elements)
            if not text or not text.strip() or conf < 0:
                continue

            left = data["left"][i]
            top = data["top"][i]
            width = data["width"][i]
            height = data["height"][i]

            bbox = BBox(
                x0=float(left),
                y0=float(top),
                x1=float(left + width),
                y1=float(top + height),
            )

            # Tesseract confidence is 0-100, normalize to 0-1
            confidence = float(conf) / 100.0

            blocks.append(
                TextBlock(
                    text=text,
                    bbox=bbox,
                    rotation=0.0,  # Tesseract doesn't provide rotation per word
                    confidence=confidence,
                )
            )

        return blocks
