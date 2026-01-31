"""Tests for table extraction in OCR adapters."""

from __future__ import annotations

from pydantic import BaseModel

from xtra.models import Table

# =============================================================================
# Google Document AI Mock Models
# =============================================================================


class GoogleVertex(BaseModel):
    x: float = 0.0
    y: float = 0.0


class GoogleBoundingPoly(BaseModel):
    normalized_vertices: list[GoogleVertex] = []


class GoogleDimension(BaseModel):
    width: float
    height: float


class GoogleTableCell(BaseModel):
    layout: GoogleCellLayout | None = None
    row_span: int = 1
    col_span: int = 1


class GoogleCellLayout(BaseModel):
    text_anchor: GoogleTextAnchor | None = None
    bounding_poly: GoogleBoundingPoly | None = None


class GoogleTextAnchor(BaseModel):
    text_segments: list[GoogleTextSegment] = []


class GoogleTextSegment(BaseModel):
    start_index: int = 0
    end_index: int = 0


class GoogleTableRow(BaseModel):
    cells: list[GoogleTableCell] = []


class GoogleTable(BaseModel):
    header_rows: list[GoogleTableRow] = []
    body_rows: list[GoogleTableRow] = []
    layout: GoogleCellLayout | None = None


class GooglePage(BaseModel):
    dimension: GoogleDimension
    tokens: list = []
    tables: list[GoogleTable] = []


class GoogleDocument(BaseModel):
    text: str = ""
    pages: list[GooglePage] = []


# Forward reference resolution
GoogleTableCell.model_rebuild()


# =============================================================================
# Azure Document Intelligence Mock Models
# =============================================================================


class AzureTableCell(BaseModel):
    content: str
    row_index: int
    column_index: int
    row_span: int = 1
    column_span: int = 1
    bounding_regions: list | None = None


class AzureTable(BaseModel):
    row_count: int
    column_count: int
    cells: list[AzureTableCell]
    bounding_regions: list | None = None


class AzurePage(BaseModel):
    page_number: int
    width: float
    height: float
    words: list = []


class AzureAnalyzeResult(BaseModel):
    pages: list[AzurePage] | None = None
    tables: list[AzureTable] | None = None
    model_id: str | None = None
    api_version: str | None = None


# =============================================================================
# Google Document AI Table Tests
# =============================================================================


class TestGoogleDocAITableExtraction:
    def test_convert_page_extracts_tables(self) -> None:
        """convert_page should extract tables from Google Document AI response."""
        from xtra.ocr.adapters.google_docai import GoogleDocumentAIAdapter

        # Text: "Header A" (0-8), "Header B" (8-16), "Row 1 A" (16-23), "Row 1 B" (23-30)
        document = GoogleDocument(
            text="Header AHeader BRow 1 ARow 1 B",
            pages=[
                GooglePage(
                    dimension=GoogleDimension(width=612, height=792),
                    tables=[
                        GoogleTable(
                            header_rows=[
                                GoogleTableRow(
                                    cells=[
                                        GoogleTableCell(
                                            layout=GoogleCellLayout(
                                                text_anchor=GoogleTextAnchor(
                                                    text_segments=[
                                                        GoogleTextSegment(
                                                            start_index=0, end_index=8
                                                        )
                                                    ]
                                                )
                                            )
                                        ),
                                        GoogleTableCell(
                                            layout=GoogleCellLayout(
                                                text_anchor=GoogleTextAnchor(
                                                    text_segments=[
                                                        GoogleTextSegment(
                                                            start_index=8, end_index=16
                                                        )
                                                    ]
                                                )
                                            )
                                        ),
                                    ]
                                )
                            ],
                            body_rows=[
                                GoogleTableRow(
                                    cells=[
                                        GoogleTableCell(
                                            layout=GoogleCellLayout(
                                                text_anchor=GoogleTextAnchor(
                                                    text_segments=[
                                                        GoogleTextSegment(
                                                            start_index=16, end_index=23
                                                        )
                                                    ]
                                                )
                                            )
                                        ),
                                        GoogleTableCell(
                                            layout=GoogleCellLayout(
                                                text_anchor=GoogleTextAnchor(
                                                    text_segments=[
                                                        GoogleTextSegment(
                                                            start_index=23, end_index=30
                                                        )
                                                    ]
                                                )
                                            )
                                        ),
                                    ]
                                )
                            ],
                        )
                    ],
                )
            ],
        )

        adapter = GoogleDocumentAIAdapter(document, "test-processor")  # type: ignore[arg-type]
        page = adapter.convert_page(0)

        assert len(page.tables) == 1
        table = page.tables[0]
        assert table.row_count == 2
        assert table.col_count == 2
        assert len(table.cells) == 4

        # Check cell contents
        cell_texts = {(c.row, c.col): c.text for c in table.cells}
        assert cell_texts[(0, 0)] == "Header A"
        assert cell_texts[(0, 1)] == "Header B"
        assert cell_texts[(1, 0)] == "Row 1 A"
        assert cell_texts[(1, 1)] == "Row 1 B"

    def test_convert_page_no_tables(self) -> None:
        """convert_page should return empty tables list when no tables present."""
        from xtra.ocr.adapters.google_docai import GoogleDocumentAIAdapter

        document = GoogleDocument(
            text="Hello",
            pages=[GooglePage(dimension=GoogleDimension(width=612, height=792), tables=[])],
        )

        adapter = GoogleDocumentAIAdapter(document, "test-processor")  # type: ignore[arg-type]
        page = adapter.convert_page(0)

        assert page.tables == []

    def test_convert_page_table_with_empty_cells(self) -> None:
        """convert_page should handle tables with empty cells."""
        from xtra.ocr.adapters.google_docai import GoogleDocumentAIAdapter

        document = GoogleDocument(
            text="A",
            pages=[
                GooglePage(
                    dimension=GoogleDimension(width=612, height=792),
                    tables=[
                        GoogleTable(
                            header_rows=[
                                GoogleTableRow(
                                    cells=[
                                        GoogleTableCell(layout=None),  # Empty cell
                                        GoogleTableCell(
                                            layout=GoogleCellLayout(
                                                text_anchor=GoogleTextAnchor(
                                                    text_segments=[
                                                        GoogleTextSegment(
                                                            start_index=0, end_index=1
                                                        )
                                                    ]
                                                )
                                            )
                                        ),
                                    ]
                                )
                            ],
                            body_rows=[],
                        )
                    ],
                )
            ],
        )

        adapter = GoogleDocumentAIAdapter(document, "test-processor")  # type: ignore[arg-type]
        page = adapter.convert_page(0)

        assert len(page.tables) == 1
        table = page.tables[0]
        assert table.row_count == 1
        assert table.col_count == 2


# =============================================================================
# Azure Document Intelligence Table Tests
# =============================================================================


class TestAzureDocIntelligenceTableExtraction:
    def test_convert_page_extracts_tables(self) -> None:
        """convert_page should extract tables from Azure DI response."""
        from xtra.ocr.adapters.azure_di import AzureDocumentIntelligenceAdapter

        result = AzureAnalyzeResult(
            pages=[AzurePage(page_number=1, width=8.5, height=11.0)],
            tables=[
                AzureTable(
                    row_count=2,
                    column_count=2,
                    cells=[
                        AzureTableCell(content="Header A", row_index=0, column_index=0),
                        AzureTableCell(content="Header B", row_index=0, column_index=1),
                        AzureTableCell(content="Row 1 A", row_index=1, column_index=0),
                        AzureTableCell(content="Row 1 B", row_index=1, column_index=1),
                    ],
                )
            ],
        )

        adapter = AzureDocumentIntelligenceAdapter(result, "prebuilt-layout")  # type: ignore[arg-type]
        page = adapter.convert_page(0)

        assert len(page.tables) == 1
        table = page.tables[0]
        assert table.row_count == 2
        assert table.col_count == 2
        assert len(table.cells) == 4

        cell_texts = {(c.row, c.col): c.text for c in table.cells}
        assert cell_texts[(0, 0)] == "Header A"
        assert cell_texts[(0, 1)] == "Header B"
        assert cell_texts[(1, 0)] == "Row 1 A"
        assert cell_texts[(1, 1)] == "Row 1 B"

    def test_convert_page_no_tables(self) -> None:
        """convert_page should return empty tables list when no tables present."""
        from xtra.ocr.adapters.azure_di import AzureDocumentIntelligenceAdapter

        result = AzureAnalyzeResult(
            pages=[AzurePage(page_number=1, width=8.5, height=11.0)],
            tables=None,
        )

        adapter = AzureDocumentIntelligenceAdapter(result, "prebuilt-read")  # type: ignore[arg-type]
        page = adapter.convert_page(0)

        assert page.tables == []

    def test_convert_page_multi_page_tables(self) -> None:
        """Tables should be assigned to correct pages."""
        from xtra.ocr.adapters.azure_di import AzureDocumentIntelligenceAdapter

        result = AzureAnalyzeResult(
            pages=[
                AzurePage(page_number=1, width=8.5, height=11.0),
                AzurePage(page_number=2, width=8.5, height=11.0),
            ],
            tables=[
                AzureTable(
                    row_count=1,
                    column_count=1,
                    cells=[AzureTableCell(content="Page 1 Table", row_index=0, column_index=0)],
                    bounding_regions=[{"page_number": 1}],
                ),
                AzureTable(
                    row_count=1,
                    column_count=1,
                    cells=[AzureTableCell(content="Page 2 Table", row_index=0, column_index=0)],
                    bounding_regions=[{"page_number": 2}],
                ),
            ],
        )

        adapter = AzureDocumentIntelligenceAdapter(result, "prebuilt-layout")  # type: ignore[arg-type]

        page0 = adapter.convert_page(0)
        page1 = adapter.convert_page(1)

        assert len(page0.tables) == 1
        assert page0.tables[0].cells[0].text == "Page 1 Table"

        assert len(page1.tables) == 1
        assert page1.tables[0].cells[0].text == "Page 2 Table"


# =============================================================================
# PaddleOCR Table Tests (PPStructure)
# =============================================================================


class TestPaddleOCRTableExtraction:
    def test_convert_table_result_basic(self) -> None:
        """convert_table_result should convert PPStructure table output."""
        from xtra.ocr.adapters.paddle_ocr import PaddleOCRAdapter

        table_result = {
            "type": "table",
            "res": {
                "html": "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
            },
        }

        adapter = PaddleOCRAdapter()
        table = adapter.convert_table_result(table_result, page=0)

        assert isinstance(table, Table)
        assert table.page == 0
        assert table.row_count == 2
        assert table.col_count == 2

    def test_convert_table_result_empty_html(self) -> None:
        """convert_table_result should handle empty table HTML."""
        from xtra.ocr.adapters.paddle_ocr import PaddleOCRAdapter

        table_result = {"type": "table", "res": {"html": "<table></table>"}}

        adapter = PaddleOCRAdapter()
        table = adapter.convert_table_result(table_result, page=0)

        assert table.row_count == 0
        assert table.col_count == 0
        assert table.cells == []

    def test_convert_table_result_with_rowspan_colspan(self) -> None:
        """convert_table_result should handle cells with rowspan/colspan."""
        from xtra.ocr.adapters.paddle_ocr import PaddleOCRAdapter

        html = '<table><tr><td colspan="2">Header</td></tr><tr><td>A</td><td>B</td></tr></table>'
        table_result = {"type": "table", "res": {"html": html}}

        adapter = PaddleOCRAdapter()
        table = adapter.convert_table_result(table_result, page=0)

        assert table.row_count == 2
        # Table should have cells even with colspan
        assert len(table.cells) >= 3
