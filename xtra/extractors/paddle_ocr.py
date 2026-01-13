"""PaddleOCR extractor."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

from xtra.models import (
    CoordinateUnit,
    DocumentMetadata,
    ExtractorType,
    TextBlock,
)
from xtra.utils.geometry import polygon_to_bbox_and_rotation
from xtra.extractors._ocr_base import ImageBasedExtractor


class PaddleOcrExtractor(ImageBasedExtractor):
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
        self.lang = lang
        self.use_gpu = use_gpu
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)
        super().__init__(path, dpi, output_unit)

    def _do_ocr(self, img: Image.Image) -> List[TextBlock]:
        """Perform OCR using PaddleOCR."""
        img_array = np.array(img)
        result = self._ocr.ocr(img_array, cls=True)
        return self._convert_results(result)

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

            bbox, rotation = polygon_to_bbox_and_rotation(bbox_points)

            blocks.append(
                TextBlock(
                    text=text,
                    bbox=bbox,
                    rotation=rotation,
                    confidence=float(confidence),
                )
            )

        return blocks
