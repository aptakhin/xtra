"""PaddleOCR extractor."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Tuple

import pypdfium2 as pdfium
from paddleocr import PaddleOCR
from PIL import Image, UnidentifiedImageError

from ..models import (
    BBox,
    CoordinateUnit,
    DocumentMetadata,
    Page,
    ExtractorType,
    TextBlock,
)
from .base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)


class PaddleOcrExtractor(BaseExtractor):
    """Extract text from images or PDFs using PaddleOCR.

    Automatically detects file type and handles PDF-to-image conversion internally.
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
        self._images: List[Image.Image] = []
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)
        self._is_pdf = path.suffix.lower() == ".pdf"
        self._load_images()

    def _load_images(self) -> None:
        """Load image(s) from path. Auto-detects PDF vs image."""
        if self._is_pdf:
            pdf = pdfium.PdfDocument(self.path)
            scale = self.dpi / 72.0
            for page in pdf:
                bitmap = page.render(scale=scale)
                self._images.append(bitmap.to_pil())
            pdf.close()
        else:
            img = Image.open(self.path)
            self._images = [img]

    def get_page_count(self) -> int:
        return len(self._images)

    def extract_page(self, page: int) -> ExtractionResult:
        """Extract text from a single image/page using PaddleOCR."""
        try:
            if page >= len(self._images):
                raise IndexError(f"Page {page} out of range")

            img = self._images[page]
            width, height = img.size

            # PaddleOCR expects numpy array or file path
            import numpy as np

            img_array = np.array(img)
            result = self._ocr.ocr(img_array, cls=True)

            text_blocks = self._convert_results(result)

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
            logger.warning("Failed to extract page %d with PaddleOCR: %s", page, e)
            return ExtractionResult(
                page=Page(page=page, width=0, height=0, texts=[]),
                success=False,
                error=str(e),
            )

    def get_metadata(self) -> DocumentMetadata:
        extra = {"ocr_engine": "paddleocr", "languages": self.lang}
        if self._is_pdf:
            extra["dpi"] = self.dpi
        return DocumentMetadata(
            source_type=ExtractorType.PADDLE,
            extra=extra,
        )

    def _convert_results(self, result: list) -> List[TextBlock]:
        """Convert PaddleOCR output to TextBlocks.

        PaddleOCR returns: [[[bbox, (text, confidence)], ...]]
        where bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        blocks = []

        if not result or not result[0]:
            return blocks

        for item in result[0]:
            if item is None:
                continue

            bbox_points, (text, confidence) = item

            if not text or not text.strip():
                continue

            bbox, rotation = self._polygon_to_bbox_and_rotation(bbox_points)

            blocks.append(
                TextBlock(
                    text=text,
                    bbox=bbox,
                    rotation=rotation,
                    confidence=float(confidence),
                )
            )

        return blocks

    def _polygon_to_bbox_and_rotation(self, polygon: List[List[float]]) -> Tuple[BBox, float]:
        """Convert PaddleOCR polygon to BBox and rotation.

        Args:
            polygon: List of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        bbox = BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))

        # Calculate rotation from first edge (top-left to top-right)
        dx = polygon[1][0] - polygon[0][0]
        dy = polygon[1][1] - polygon[0][1]
        rotation = math.degrees(math.atan2(dy, dx)) if dx != 0 or dy != 0 else 0.0

        return bbox, rotation

    def close(self) -> None:
        """Close image handles."""
        for img in self._images:
            try:
                img.close()
            except Exception:  # noqa: S110
                pass
        self._images = []
