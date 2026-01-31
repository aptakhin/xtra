from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from xtra.extractors.base import BaseExtractor, PageExtractionResult
from xtra.extractors.character_mergers import (
    BasicLineMerger,
    CharacterMerger,
    CharInfo,
)
from xtra.models import (
    CoordinateUnit,
    ExtractorMetadata,
    ExtractorType,
    Page,
    Table,
    TableCell,
    TextBlock,
)

logger = logging.getLogger(__name__)


class PdfExtractor(BaseExtractor):
    """Extract text and metadata from PDF files using pypdfium2."""

    def __init__(
        self,
        path: Path | str,
        output_unit: CoordinateUnit = CoordinateUnit.POINTS,
        character_merger: CharacterMerger | None = None,
    ) -> None:
        super().__init__(path, output_unit)
        self._pdf = pdfium.PdfDocument(self.path)
        self._merger = character_merger if character_merger is not None else BasicLineMerger()
        self._lock = threading.Lock()

    def get_page_count(self) -> int:
        return len(self._pdf)

    def extract_page(
        self,
        page: int,
        table_options: dict[str, Any] | None = None,
    ) -> PageExtractionResult:
        """Extract a single page by number (0-indexed).

        Thread-safe: uses internal lock for parallel access.

        Args:
            page: Page number (0-indexed).
            table_options: Optional dict of tabula options for table extraction.
                If provided, tables will be extracted and added to Page.tables.
                Common options: lattice, stream, columns, area, guess, multiple_tables.
        """
        try:
            with self._lock:
                pdf_page = self._pdf[page]
                width, height = pdf_page.get_size()
                text_blocks = self._extract_text_blocks(pdf_page, height)

            tables: list[Table] = []
            if table_options is not None:
                tables = self._extract_tables_for_page(page, table_options)

            result_page = Page(
                page=page,
                width=width,
                height=height,
                texts=text_blocks,
                tables=tables,
            )
            # Convert from native POINTS to output_unit
            result_page = self._convert_page(result_page, CoordinateUnit.POINTS)
            return PageExtractionResult(page=result_page, success=True)
        except Exception as e:
            return PageExtractionResult(
                page=Page(page=page, width=0, height=0, texts=[]),
                success=False,
                error=str(e),
            )

    def get_extractor_metadata(self) -> ExtractorMetadata:
        metadata_dict = {}
        try:
            for key in ["Title", "Author", "Creator", "Producer", "CreationDate", "ModDate"]:
                val = self._pdf.get_metadata_value(key)
                if val:
                    metadata_dict[key.lower()] = val
        except (KeyError, ValueError, pdfium.PdfiumError) as e:
            logger.warning("Failed to extract PDF metadata: %s", e)

        return ExtractorMetadata(
            extractor_type=ExtractorType.PDF,
            title=metadata_dict.get("title"),
            author=metadata_dict.get("author"),
            creator=metadata_dict.get("creator"),
            producer=metadata_dict.get("producer"),
            creation_date=metadata_dict.get("creationdate"),
            modification_date=metadata_dict.get("moddate"),
        )

    def close(self) -> None:
        self._pdf.close()

    def _extract_text_blocks(self, page: pdfium.PdfPage, page_height: float) -> list[TextBlock]:
        textpage = page.get_textpage()
        char_count = textpage.count_chars()
        if char_count == 0:
            return []

        # Batch text extraction (206x faster than per-char)
        all_text = textpage.get_text_range(0, char_count)

        # Check rotation support once, not per character
        has_rotation = hasattr(textpage, "get_char_rotation")

        chars: list[CharInfo] = []
        for i in range(char_count):
            bbox = textpage.get_charbox(i)
            rotation = textpage.get_char_rotation(i) if has_rotation else 0
            chars.append(CharInfo(char=all_text[i], bbox=bbox, rotation=rotation, index=i))

        return self._merger.merge(chars, textpage, page_height)

    def extract_tables(
        self,
        pages: Sequence[int] | None = None,
        table_options: dict[str, Any] | None = None,
    ) -> list[Table]:
        """Extract tables from PDF pages using tabula.

        Args:
            pages: Sequence of page numbers to extract (0-indexed).
                   If None, extracts from all pages.
            table_options: Dict of tabula options. Common options:
                - lattice: bool - Use lattice mode (tables with cell borders)
                - stream: bool - Use stream mode (tables without borders)
                - columns: list[float] - Column x-coordinates for splitting
                - area: tuple[float, float, float, float] - (top, left, bottom, right)
                - guess: bool - Guess table areas automatically
                - multiple_tables: bool - Extract multiple tables per page
                - pandas_options: dict - Options for pandas

        Returns:
            List of Table objects with page field indicating source page.
        """
        if pages is None:
            pages = range(self.get_page_count())

        options = table_options or {}
        all_tables: list[Table] = []

        for page_num in pages:
            page_tables = self._extract_tables_for_page(page_num, options)
            all_tables.extend(page_tables)

        return all_tables

    def _extract_tables_for_page(
        self,
        page: int,
        options: dict[str, Any],
    ) -> list[Table]:
        """Extract tables from a single page using tabula.

        Args:
            page: Page number (0-indexed).
            options: Tabula options dict.

        Returns:
            List of Table objects for this page.
        """
        try:
            import tabula
        except ImportError as e:
            raise ImportError(
                "tabula-py is required for table extraction. "
                "Install with: pip install 'xtra[tables]'"
            ) from e

        tabula_opts = self._build_tabula_options(page, options)
        dfs = tabula.read_pdf(str(self.path), **tabula_opts)

        return [self._dataframe_to_table(df, page) for df in dfs if not df.empty]

    def _build_tabula_options(self, page: int, options: dict[str, Any]) -> dict[str, Any]:
        """Build tabula options dict from user options."""
        # Tabula uses 1-indexed pages
        tabula_opts: dict[str, Any] = {
            "pages": page + 1,
            "multiple_tables": options.get("multiple_tables", True),
            "guess": options.get("guess", True),
        }

        # Copy optional settings
        for key in ("lattice", "stream", "columns", "area", "pandas_options"):
            if options.get(key):
                tabula_opts[key] = options[key]

        return tabula_opts

    def _dataframe_to_table(self, df: Any, page: int) -> Table:
        """Convert a pandas DataFrame to a Table model."""
        cells: list[TableCell] = []
        row_count = len(df)
        col_count = len(df.columns)

        # Add header row (column names)
        for col_idx, col_name in enumerate(df.columns):
            cell_text = str(col_name) if col_name is not None else ""
            cells.append(TableCell(text=cell_text, row=0, col=col_idx))

        # Add data rows
        for row_idx, row in enumerate(df.itertuples(index=False), start=1):
            for col_idx, value in enumerate(row):
                cell_text = str(value) if value is not None and str(value) != "nan" else ""
                cells.append(TableCell(text=cell_text, row=row_idx, col=col_idx))

        return Table(
            page=page,
            cells=cells,
            row_count=row_count + 1,  # +1 for header row
            col_count=col_count,
        )
