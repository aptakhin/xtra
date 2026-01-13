"""PaddleOCR extractor."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from paddleocr import PaddleOCR

from xtra.adapters.paddle_ocr import PaddleOCRAdapter
from xtra.extractors._image_loader import ImageLoader
from xtra.extractors.base import BaseExtractor, ExtractionResult
from xtra.models import (
    CoordinateUnit,
    DocumentMetadata,
    ExtractorType,
    Page,
)

logger = logging.getLogger(__name__)


class PaddleOcrExtractor(BaseExtractor):
    """Extract text from images or PDFs using PaddleOCR.

    Composes ImageLoader for image handling, PaddleOCR for OCR,
    and PaddleOCRAdapter for result conversion.
    """

    def __init__(
        self,
        path: Path,
        lang: str = "en",
        use_gpu: bool = False,
        dpi: int = 200,
        output_unit: CoordinateUnit = CoordinateUnit.POINTS,
    ) -> None:
        """Initialize PaddleOCR extractor.

        Args:
            path: Path to the image or PDF file.
            lang: Language code for OCR. Common values:
                  - "en" for English
                  - "ch" for Chinese
                  - "fr" for French
                  - "german" for German
                  - "japan" for Japanese
                  - "korean" for Korean
                  See PaddleOCR docs for full list.
            use_gpu: Whether to use GPU acceleration.
            dpi: DPI for PDF-to-image conversion. Default 200.
            output_unit: Coordinate unit for output. Default POINTS.
        """
        super().__init__(path, output_unit)
        self.lang = lang
        self.use_gpu = use_gpu
        self.dpi = dpi

        # Compose components
        self._images = ImageLoader(path, dpi)
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)
        self._adapter = PaddleOCRAdapter()

    def get_page_count(self) -> int:
        """Return number of pages/images loaded."""
        return self._images.page_count

    def extract_page(self, page: int) -> ExtractionResult:
        """Extract text from a single image/page."""
        try:
            img = self._images.get_page(page)
            width, height = img.size

            # Run OCR pipeline
            result = self._ocr.ocr(np.array(img), cls=True)
            text_blocks = self._adapter.convert_result(result)

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
        extra = {"ocr_engine": "paddleocr", "languages": self.lang}
        if self._images.is_pdf:
            extra["dpi"] = self.dpi
        return DocumentMetadata(
            source_type=ExtractorType.PADDLE,
            extra=extra,
        )

    def close(self) -> None:
        """Release resources."""
        self._images.close()
