from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import easyocr
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

_reader_cache: dict[tuple, easyocr.Reader] = {}


def get_reader(languages: List[str], gpu: bool = False) -> easyocr.Reader:
    """Get or create a cached EasyOCR reader."""
    key = (tuple(languages), gpu)
    if key not in _reader_cache:
        _reader_cache[key] = easyocr.Reader(languages, gpu=gpu)
    return _reader_cache[key]


class EasyOcrExtractor(BaseExtractor):
    """Extract text from images or PDFs using EasyOCR.

    Automatically detects file type and handles PDF-to-image conversion internally.
    """

    def __init__(
        self,
        path: Path,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
        dpi: int = 200,
        output_unit: CoordinateUnit = CoordinateUnit.POINTS,
    ) -> None:
        super().__init__(path, output_unit)
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.dpi = dpi
        self._images: List[Image.Image] = []
        self._is_pdf = path.suffix.lower() == ".pdf"
        self._load_images()

    def _load_images(self) -> None:
        """Load image(s) from path. Auto-detects PDF vs image."""
        if self._is_pdf:
            import pypdfium2 as pdfium  # noqa: PLC0415

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
        """Extract text from a single image/page."""
        try:
            if page >= len(self._images):
                raise IndexError(f"Page {page} out of range")

            img = self._images[page]
            width, height = img.size

            reader = get_reader(self.languages, self.gpu)
            import numpy as np  # noqa: PLC0415

            results = reader.readtext(np.array(img))
            text_blocks = self._convert_results(results)

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

    def get_metadata(self) -> DocumentMetadata:
        extra = {"ocr_engine": "easyocr", "languages": self.languages}
        if self._is_pdf:
            extra["dpi"] = self.dpi
        return DocumentMetadata(
            source_type=ExtractorType.EASYOCR,
            extra=extra,
        )

    def _convert_results(self, results: List[Tuple]) -> List[TextBlock]:
        blocks = []
        for result in results:
            polygon, text, confidence = result
            bbox, rotation = self._polygon_to_bbox_and_rotation(polygon)
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
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        bbox = BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))

        dx = polygon[1][0] - polygon[0][0]
        dy = polygon[1][1] - polygon[0][1]
        rotation = math.degrees(math.atan2(dy, dx))

        return bbox, rotation
