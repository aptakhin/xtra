from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import easyocr
from PIL import Image, UnidentifiedImageError

from ..models import (
    BBox,
    DocumentMetadata,
    Page,
    SourceType,
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


class OcrExtractor(BaseExtractor):
    """Extract text from images using EasyOCR."""

    def __init__(
        self,
        path: Path,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
    ) -> None:
        super().__init__(path)
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._images: List[Image.Image] = []
        self._load_images()

    def _load_images(self) -> None:
        """Load image(s) from path. Single image = single page."""
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
            results = reader.readtext(str(self.path) if page == 0 else img)
            print("XX", page, results)
            text_blocks = self._convert_results(results)

            return ExtractionResult(
                page=Page(
                    page=page,
                    width=float(width),
                    height=float(height),
                    texts=text_blocks,
                ),
                success=True,
            )
        except (IndexError, UnidentifiedImageError, OSError, RuntimeError) as e:
            logger.warning("Failed to extract page %d: %s", page, e)
            return ExtractionResult(
                page=Page(page=page, width=0, height=0, texts=[]),
                success=False,
                error=str(e),
            )

    def get_metadata(self) -> DocumentMetadata:
        return DocumentMetadata(
            source_type=SourceType.OCR,
            extra={"ocr_engine": "easyocr", "languages": self.languages},
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


class PdfToImageOcrExtractor(BaseExtractor):
    """Extract text from PDF by converting pages to images and running OCR."""

    def __init__(
        self,
        path: Path,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
        dpi: int = 200,
    ) -> None:
        super().__init__(path)
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.dpi = dpi
        self._images: List[Image.Image] = []
        self._load_pdf_as_images()

    def _load_pdf_as_images(self) -> None:
        """Convert PDF pages to images using pypdfium2."""
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(self.path)
        scale = self.dpi / 72.0
        for page in pdf:
            bitmap = page.render(scale=scale)
            self._images.append(bitmap.to_pil())
        pdf.close()

    def get_page_count(self) -> int:
        return len(self._images)

    def extract_page(self, page: int) -> ExtractionResult:
        """Extract text from a single PDF page via OCR."""
        try:
            if page >= len(self._images):
                raise IndexError(f"Page {page} out of range")

            img = self._images[page]
            width, height = img.size

            reader = get_reader(self.languages, self.gpu)
            import numpy as np

            results = reader.readtext(np.array(img))
            print("XX", page, results)

            text_blocks = self._convert_results(results)

            return ExtractionResult(
                page=Page(
                    page=page,
                    width=float(width),
                    height=float(height),
                    texts=text_blocks,
                ),
                success=True,
            )
        except (IndexError, OSError, RuntimeError) as e:
            logger.warning("Failed to extract page %d via OCR: %s", page, e)
            return ExtractionResult(
                page=Page(page=page, width=0, height=0, texts=[]),
                success=False,
                error=str(e),
            )

    def get_metadata(self) -> DocumentMetadata:
        return DocumentMetadata(
            source_type=SourceType.PDF_OCR,
            extra={
                "ocr_engine": "easyocr",
                "languages": self.languages,
                "dpi": self.dpi,
            },
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
