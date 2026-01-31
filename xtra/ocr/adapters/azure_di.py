"""Adapter for converting Azure Document Intelligence results to internal schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xtra.models import ExtractorMetadata, ExtractorType, Page, Table, TableCell, TextBlock
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
        tables = self._convert_tables_for_page(page)

        return Page(
            page=page,
            width=float(width),
            height=float(height),
            texts=text_blocks,
            tables=tables,
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

    def _convert_tables_for_page(self, page: int) -> list[Table]:
        """Extract tables that belong to the specified page.

        Azure DI stores tables at the result level with bounding_regions
        indicating which page(s) the table appears on.
        """
        tables: list[Table] = []

        if (
            self._result is None
            or not hasattr(self._result, "tables")
            or self._result.tables is None
        ):
            return tables

        for azure_table in self._result.tables:
            # Check if this table belongs to the current page
            table_page = self._get_table_page(azure_table)
            if table_page != page:
                continue

            cells: list[TableCell] = []
            if hasattr(azure_table, "cells") and azure_table.cells:
                for cell in azure_table.cells:
                    cell_text = cell.content if hasattr(cell, "content") and cell.content else ""
                    row_idx = cell.row_index if hasattr(cell, "row_index") else 0
                    col_idx = cell.column_index if hasattr(cell, "column_index") else 0
                    cells.append(TableCell(text=cell_text, row=row_idx, col=col_idx))

            row_count = azure_table.row_count if hasattr(azure_table, "row_count") else 0
            col_count = azure_table.column_count if hasattr(azure_table, "column_count") else 0

            tables.append(
                Table(
                    page=page,
                    cells=cells,
                    row_count=row_count,
                    col_count=col_count,
                )
            )

        return tables

    def _get_table_page(self, azure_table) -> int:
        """Get the page number for a table from its bounding_regions.

        Azure uses 1-indexed page numbers, we convert to 0-indexed.
        """
        if not hasattr(azure_table, "bounding_regions") or not azure_table.bounding_regions:
            return 0  # Default to first page if no bounding region

        first_region = azure_table.bounding_regions[0]
        if isinstance(first_region, dict):
            page_num = first_region.get("page_number", 1)
        elif hasattr(first_region, "page_number"):
            page_num = first_region.page_number
        else:
            page_num = 1

        return page_num - 1  # Convert to 0-indexed

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
