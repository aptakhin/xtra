from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pypdfium2 as pdfium

from xtra.models import (
    BBox,
    CoordinateUnit,
    DocumentMetadata,
    FontInfo,
    Page,
    PdfObjectInfo,
    ExtractorType,
    TextBlock,
)
from xtra.extractors.base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)


class PdfExtractor(BaseExtractor):
    """Extract text and metadata from PDF files using pypdfium2."""

    def __init__(
        self,
        path: Path,
        output_unit: CoordinateUnit = CoordinateUnit.POINTS,
    ) -> None:
        super().__init__(path, output_unit)
        self._pdf = pdfium.PdfDocument(path)

    def get_page_count(self) -> int:
        return len(self._pdf)

    def extract_page(self, page: int) -> ExtractionResult:
        """Extract a single page by number (0-indexed)."""
        try:
            pdf_page = self._pdf[page]
            width, height = pdf_page.get_size()
            text_blocks = self._extract_text_blocks(pdf_page, height)
            result_page = Page(
                page=page,
                width=width,
                height=height,
                texts=text_blocks,
            )
            # Convert from native POINTS to output_unit
            result_page = self._convert_page(result_page, CoordinateUnit.POINTS)
            return ExtractionResult(page=result_page, success=True)
        except Exception as e:
            return ExtractionResult(
                page=Page(page=page, width=0, height=0, texts=[]),
                success=False,
                error=str(e),
            )

    def get_metadata(self) -> DocumentMetadata:
        metadata_dict = {}
        try:
            for key in ["Title", "Author", "Creator", "Producer", "CreationDate", "ModDate"]:
                val = self._pdf.get_metadata_value(key)
                if val:
                    metadata_dict[key.lower()] = val
        except (KeyError, ValueError, pdfium.PdfiumError) as e:
            logger.warning("Failed to extract PDF metadata: %s", e)

        pdf_objects = self._extract_pdf_objects()

        return DocumentMetadata(
            source_type=ExtractorType.PDF,
            title=metadata_dict.get("title"),
            author=metadata_dict.get("author"),
            creator=metadata_dict.get("creator"),
            producer=metadata_dict.get("producer"),
            creation_date=metadata_dict.get("creationdate"),
            modification_date=metadata_dict.get("moddate"),
            pdf_objects=pdf_objects,
        )

    def close(self) -> None:
        self._pdf.close()

    def _extract_text_blocks(self, page: pdfium.PdfPage, page_height: float) -> List[TextBlock]:
        textpage = page.get_textpage()
        char_count = textpage.count_chars()
        if char_count == 0:
            return []

        blocks: List[TextBlock] = []
        current_chars: List[dict] = []
        prev_char_info: Optional[dict] = None

        for i in range(char_count):
            char = textpage.get_text_range(i, 1)
            bbox = textpage.get_charbox(i)
            rotation = (
                textpage.get_char_rotation(i) if hasattr(textpage, "get_char_rotation") else 0
            )

            char_info = {"char": char, "bbox": bbox, "rotation": rotation, "index": i}

            if self._is_new_block(prev_char_info, char_info):
                if current_chars:
                    block = self._create_text_block(current_chars, textpage, page_height)
                    if block and block.text.strip():
                        blocks.append(block)
                current_chars = []

            current_chars.append(char_info)
            prev_char_info = char_info

        if current_chars:
            block = self._create_text_block(current_chars, textpage, page_height)
            if block and block.text.strip():
                blocks.append(block)

        return blocks

    def _is_new_block(
        self,
        prev: Optional[dict],
        curr: dict,
        line_gap_threshold: float = 5.0,
    ) -> bool:
        if prev is None:
            return False
        vertical_gap = abs(curr["bbox"][1] - prev["bbox"][1])
        return vertical_gap > line_gap_threshold

    def _create_text_block(
        self,
        chars: List[dict],
        textpage: pdfium.PdfTextPage,
        page_height: float,
    ) -> Optional[TextBlock]:
        if not chars:
            return None

        text = "".join(c["char"] for c in chars).strip()
        if not text:
            return None

        x0 = min(c["bbox"][0] for c in chars)
        y0 = min(c["bbox"][1] for c in chars)
        x1 = max(c["bbox"][2] for c in chars)
        y1 = max(c["bbox"][3] for c in chars)

        bbox = BBox(x0=x0, y0=page_height - y1, x1=x1, y1=page_height - y0)
        rotation = chars[0]["rotation"] if chars else 0
        font_info = self._extract_font_info(textpage, chars[0]["index"])

        return TextBlock(
            text=text,
            bbox=bbox,
            rotation=float(rotation),
            font_info=font_info,
        )

    def _extract_font_info(
        self, textpage: pdfium.PdfTextPage, char_index: int
    ) -> Optional[FontInfo]:
        try:
            text_obj = textpage.get_textobj(char_index)
            if text_obj is None:
                return None

            font = text_obj.get_font()
            font_size = text_obj.get_font_size()
            name = font.get_base_name() or font.get_family_name() or None
            weight = font.get_weight()

            return FontInfo(name=name, size=font_size, weight=weight)
        except (AttributeError, IndexError, pdfium.PdfiumError) as e:
            logger.debug("Failed to extract font info for char %d: %s", char_index, e)
            return None

    def _extract_pdf_objects(self) -> List[PdfObjectInfo]:
        objects = []
        try:
            for page_num in range(len(self._pdf)):
                page = self._pdf[page_num]
                for obj in page.get_objects():
                    obj_type = type(obj).__name__
                    objects.append(PdfObjectInfo(obj_id=id(obj), obj_type=obj_type))
        except (IndexError, pdfium.PdfiumError) as e:
            logger.warning("Failed to extract PDF objects: %s", e)
        return objects
