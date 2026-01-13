"""EasyOCR extractor."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import easyocr
import numpy as np
from PIL import Image

from xtra.models import (
    CoordinateUnit,
    DocumentMetadata,
    ExtractorType,
    TextBlock,
)
from xtra.utils.geometry import polygon_to_bbox_and_rotation
from xtra.extractors._ocr_base import ImageBasedExtractor

_reader_cache: dict[tuple, easyocr.Reader] = {}


def get_reader(languages: List[str], gpu: bool = False) -> easyocr.Reader:
    """Get or create a cached EasyOCR reader."""
    key = (tuple(languages), gpu)
    if key not in _reader_cache:
        _reader_cache[key] = easyocr.Reader(languages, gpu=gpu)
    return _reader_cache[key]


class EasyOcrExtractor(ImageBasedExtractor):
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
        self.languages = languages or ["en"]
        self.gpu = gpu
        super().__init__(path, dpi, output_unit)

    def _do_ocr(self, img: Image.Image) -> List[TextBlock]:
        """Perform OCR using EasyOCR."""
        reader = get_reader(self.languages, self.gpu)
        results = reader.readtext(np.array(img))
        return self._convert_results(results)

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
            bbox, rotation = polygon_to_bbox_and_rotation(polygon)
            blocks.append(
                TextBlock(
                    text=text,
                    bbox=bbox,
                    rotation=rotation,
                    confidence=float(confidence),
                )
            )
        return blocks
