"""Adapter for converting Google Document AI results to internal schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xtra.models import BBox, ExtractorMetadata, ExtractorType, Page, TextBlock
from xtra.utils.geometry import polygon_to_bbox_and_rotation

if TYPE_CHECKING:
    from google.cloud.documentai_v1 import Document


class GoogleDocumentAIAdapter:
    """Converts Google Document AI Document to internal schema."""

    def __init__(self, document: Document | None, processor_name: str) -> None:
        self._document = document
        self._processor_name = processor_name

    @property
    def page_count(self) -> int:
        if self._document is None or not self._document.pages:
            return 0
        return len(self._document.pages)

    def convert_page(self, page: int) -> Page:
        """Convert a single Google Document AI page to internal Page model.

        Args:
            page: Zero-indexed page number.

        Returns:
            Page with converted TextBlocks.

        Raises:
            ValueError: If document is None or has no pages.
            IndexError: If page is out of range.
        """
        if self._document is None:
            raise ValueError("No analysis result available")

        if not self._document.pages or page >= len(self._document.pages):
            raise IndexError(f"Page {page} out of range")

        docai_page = self._document.pages[page]
        width = docai_page.dimension.width if docai_page.dimension else 0.0
        height = docai_page.dimension.height if docai_page.dimension else 0.0
        text_blocks = self._convert_page_to_blocks(docai_page)

        return Page(
            page=page,
            width=float(width),
            height=float(height),
            texts=text_blocks,
        )

    def get_metadata(self) -> ExtractorMetadata:
        """Extract metadata from Google Document AI result."""
        extra: dict = {
            "ocr_engine": "google_document_ai",
            "processor_name": self._processor_name,
        }

        return ExtractorMetadata(
            extractor_type=ExtractorType.GOOGLE_DOCAI,
            extra=extra,
        )

    def _convert_page_to_blocks(self, docai_page) -> list[TextBlock]:
        """Convert Google Document AI page tokens to TextBlocks."""
        blocks: list[TextBlock] = []

        if not docai_page.tokens:
            return blocks

        page_width = docai_page.dimension.width if docai_page.dimension else 1.0
        page_height = docai_page.dimension.height if docai_page.dimension else 1.0

        for token in docai_page.tokens:
            if token.layout is None or token.layout.bounding_poly is None:
                continue

            # Get text from document using text_anchor
            text = self._get_text_from_layout(token.layout)
            if not text:
                continue

            # Use normalized_vertices if available, otherwise vertices
            vertices = token.layout.bounding_poly.normalized_vertices
            if not vertices:
                vertices = token.layout.bounding_poly.vertices

            if not vertices:
                continue

            bbox, rotation = self._vertices_to_bbox_and_rotation(vertices, page_width, page_height)
            confidence = token.layout.confidence if token.layout.confidence else None

            blocks.append(
                TextBlock(
                    text=text,
                    bbox=bbox,
                    rotation=rotation,
                    confidence=confidence,
                )
            )

        return blocks

    def _get_text_from_layout(self, layout) -> str:
        """Extract text from layout using text_anchor."""
        if self._document is None or not self._document.text:
            return ""

        if not layout.text_anchor or not layout.text_anchor.text_segments:
            return ""

        text_parts = []
        for segment in layout.text_anchor.text_segments:
            start_index = int(segment.start_index) if segment.start_index else 0
            end_index = int(segment.end_index) if segment.end_index else 0
            text_parts.append(self._document.text[start_index:end_index])

        return "".join(text_parts)

    @staticmethod
    def _vertices_to_bbox_and_rotation(
        vertices, page_width: float, page_height: float
    ) -> tuple[BBox, float]:
        """Convert Google Document AI vertices to BBox and rotation.

        Google Document AI returns normalized vertices (0-1 range) that need
        to be denormalized using page dimensions before conversion.
        """
        if len(vertices) < 4:
            return BBox(x0=0, y0=0, x1=0, y1=0), 0.0

        # Denormalize vertices to page coordinates
        points = []
        for v in vertices[:4]:
            x = v.x * page_width if hasattr(v, "x") and v.x else 0.0
            y = v.y * page_height if hasattr(v, "y") and v.y else 0.0
            points.append((x, y))

        # Use shared utility for bbox and rotation calculation
        return polygon_to_bbox_and_rotation(points)
