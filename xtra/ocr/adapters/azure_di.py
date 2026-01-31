"""Adapter for converting Azure Document Intelligence results to internal schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xtra.models import ExtractorMetadata, ExtractorType, Page, TextBlock
from xtra.utils.geometry import polygon_to_bbox_and_rotation

if TYPE_CHECKING:
    from azure.ai.documentintelligence.models import AnalyzeResult, DocumentPage


class AzureDocumentIntelligenceAdapter:
    """Converts Azure Document Intelligence AnalyzeResult to internal schema."""

    def __init__(self, result: AnalyzeResult | None, model_id: str = "prebuilt-read") -> None:
        self._result = result
        self._model_id = model_id

    @property
    def page_count(self) -> int:
        if self._result is None or self._result.pages is None:
            return 0
        return len(self._result.pages)

    def convert_page(self, page: int) -> Page:
        """Convert a single Azure page to internal Page model.

        Args:
            page: Zero-indexed page number.

        Returns:
            Page with converted TextBlocks.

        Raises:
            ValueError: If result is None or has no pages.
            IndexError: If page is out of range.
        """
        if self._result is None or self._result.pages is None:
            raise ValueError("No analysis result available")

        if page >= len(self._result.pages):
            raise IndexError(f"Page {page} out of range")

        azure_page = self._result.pages[page]
        width = azure_page.width or 0.0
        height = azure_page.height or 0.0
        text_blocks = self._convert_page_to_blocks(azure_page)

        return Page(
            page=page,
            width=float(width),
            height=float(height),
            texts=text_blocks,
        )

    def get_metadata(self) -> ExtractorMetadata:
        """Extract metadata from Azure result."""
        extra: dict = {
            "ocr_engine": "azure_document_intelligence",
            "model_id": self._model_id,
        }

        if self._result is not None:
            if self._result.model_id:
                extra["azure_model_id"] = self._result.model_id
            if self._result.api_version:
                extra["api_version"] = self._result.api_version

        return ExtractorMetadata(
            extractor_type=ExtractorType.AZURE_DI,
            extra=extra,
        )

    def _convert_page_to_blocks(self, azure_page: DocumentPage) -> list[TextBlock]:
        """Convert Azure DI page words to TextBlocks."""
        blocks: list[TextBlock] = []

        if azure_page.words is None:
            return blocks

        for word in azure_page.words:
            if word.content is None or word.polygon is None:
                continue

            # Azure polygon is flat: [x0, y0, x1, y1, x2, y2, x3, y3]
            bbox, rotation = polygon_to_bbox_and_rotation(word.polygon, flat=True)
            confidence = word.confidence if word.confidence is not None else None

            blocks.append(
                TextBlock(
                    text=word.content,
                    bbox=bbox,
                    rotation=rotation,
                    confidence=confidence,
                )
            )

        return blocks
