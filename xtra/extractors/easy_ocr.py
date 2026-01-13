"""EasyOCR extractor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import easyocr
import numpy as np

from xtra.adapters.easy_ocr import EasyOCRAdapter
from xtra.extractors._image_loader import ImageLoader
from xtra.extractors.base import BaseExtractor, ExtractionResult
from xtra.models import (
    CoordinateUnit,
    DocumentMetadata,
    ExtractorType,
    Page,
)

logger = logging.getLogger(__name__)

_reader_cache: dict[tuple, easyocr.Reader] = {}


def get_reader(languages: List[str], gpu: bool = False) -> easyocr.Reader:
    """Get or create a cached EasyOCR reader."""
    key = (tuple(languages), gpu)
    if key not in _reader_cache:
        _reader_cache[key] = easyocr.Reader(languages, gpu=gpu)
    return _reader_cache[key]


class EasyOcrExtractor(BaseExtractor):
    """Extract text from images or PDFs using EasyOCR.

    Composes ImageLoader for image handling, EasyOCR for OCR processing,
    and EasyOCRAdapter for result conversion.
    """

    def __init__(
        self,
        path: Path,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
        dpi: int = 200,
        output_unit: CoordinateUnit = CoordinateUnit.POINTS,
    ) -> None:
        """Initialize EasyOCR extractor.

        Args:
            path: Path to the image or PDF file.
            languages: List of language codes for OCR. Defaults to ["en"].
            gpu: Whether to use GPU acceleration.
            dpi: DPI for PDF-to-image conversion. Default 200.
            output_unit: Coordinate unit for output. Default POINTS.
        """
        super().__init__(path, output_unit)
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.dpi = dpi

        # Compose components
        self._images = ImageLoader(path, dpi)
        self._adapter = EasyOCRAdapter()

    def get_page_count(self) -> int:
        """Return number of pages/images loaded."""
        return self._images.page_count

    def extract_page(self, page: int) -> ExtractionResult:
        """Extract text from a single image/page."""
        try:
            img = self._images.get_page(page)
            width, height = img.size

            # Run OCR pipeline
            reader = get_reader(self.languages, self.gpu)
            results = reader.readtext(np.array(img))
            text_blocks = self._adapter.convert_result(results)

            result_page = Page(
                page=page,
                width=float(width),
                height=float(height),
                texts=text_blocks,
            )

            # Convert from native PIXELS to output_unit
            result_page = self._convert_page(result_page, CoordinateUnit.PIXELS, self.dpi)
            return ExtractionResult(page=result_page, success=True)

        except Exception as e:
            logger.warning("Failed to extract page %d: %s", page, e)
            return ExtractionResult(
                page=Page(page=page, width=0, height=0, texts=[]),
                success=False,
                error=str(e),
            )

    def get_metadata(self) -> DocumentMetadata:
        """Return extractor metadata."""
        extra = {"ocr_engine": "easyocr", "languages": self.languages}
        if self._images.is_pdf:
            extra["dpi"] = self.dpi
        return DocumentMetadata(
            source_type=ExtractorType.EASYOCR,
            extra=extra,
        )

    def close(self) -> None:
        """Release resources."""
        self._images.close()
