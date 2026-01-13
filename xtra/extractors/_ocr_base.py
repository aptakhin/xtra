"""Base class for image-based OCR extractors."""

from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import List

from PIL import Image, UnidentifiedImageError

from xtra.models import CoordinateUnit, Page, TextBlock
from xtra.extractors._image_loader import is_pdf, load_images_from_path
from xtra.extractors.base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)


class ImageBasedExtractor(BaseExtractor):
    """Base class for extractors that process images via OCR.

    Provides common functionality for EasyOCR, Tesseract, and PaddleOCR extractors:
    - Automatic PDF-to-image conversion
    - Image loading and management
    - Common error handling in extract_page
    - Resource cleanup

    Subclasses must implement:
    - _do_ocr(img): Perform OCR on a single image and return TextBlocks
    - get_metadata(): Return extractor-specific metadata
    """

    def __init__(
        self,
        path: Path,
        dpi: int = 200,
        output_unit: CoordinateUnit = CoordinateUnit.POINTS,
    ) -> None:
        """Initialize image-based extractor.

        Args:
            path: Path to the image or PDF file.
            dpi: DPI for PDF-to-image conversion. Default 200.
            output_unit: Coordinate unit for output. Default POINTS.
        """
        super().__init__(path, output_unit)
        self.dpi = dpi
        self._images: List[Image.Image] = []
        self._is_pdf = is_pdf(path)
        self._load_images()

    def _load_images(self) -> None:
        """Load image(s) from path."""
        self._images = load_images_from_path(self.path, self.dpi)

    def get_page_count(self) -> int:
        """Return number of pages/images loaded."""
        return len(self._images)

    def extract_page(self, page: int) -> ExtractionResult:
        """Extract text from a single image/page.

        Uses template method pattern - calls _do_ocr() which subclasses implement.
        """
        try:
            if page >= len(self._images):
                raise IndexError(f"Page {page} out of range")

            img = self._images[page]
            width, height = img.size

            text_blocks = self._do_ocr(img)

            result_page = Page(
                page=page,
                width=float(width),
                height=float(height),
                texts=text_blocks,
            )
            # Convert from native PIXELS to output_unit
            result_page = self._convert_page(result_page, CoordinateUnit.PIXELS, self.dpi)
            return ExtractionResult(page=result_page, success=True)

        except (IndexError, UnidentifiedImageError, OSError, RuntimeError) as e:
            logger.warning("Failed to extract page %d: %s", page, e)
            return ExtractionResult(
                page=Page(page=page, width=0, height=0, texts=[]),
                success=False,
                error=str(e),
            )

    @abstractmethod
    def _do_ocr(self, img: Image.Image) -> List[TextBlock]:
        """Perform OCR on a single image.

        Args:
            img: PIL Image to process.

        Returns:
            List of TextBlocks with coordinates in pixels.
        """
        ...

    def close(self) -> None:
        """Close image handles and release resources."""
        for img in self._images:
            try:
                img.close()
            except Exception:  # noqa: S110
                pass
        self._images = []
