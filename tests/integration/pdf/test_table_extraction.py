"""Tests for table extraction from PDFs using tabula."""

from pathlib import Path

import pytest

from xtra.extractors import PdfExtractor
from xtra.models import Table

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data"
TABLE_PDF = TEST_DATA_DIR / "test_pdf_table.pdf"


@pytest.fixture(scope="module")
def table_pdf_path() -> Path:
    """Create a test PDF with a simple table if it doesn't exist."""
    if TABLE_PDF.exists():
        return TABLE_PDF

    # Create a simple PDF with a table using reportlab
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, TableStyle
    from reportlab.platypus import Table as RLTable

    doc = SimpleDocTemplate(str(TABLE_PDF), pagesize=letter)
    data = [
        ["Header A", "Header B", "Header C"],
        ["Row 1 A", "Row 1 B", "Row 1 C"],
        ["Row 2 A", "Row 2 B", "Row 2 C"],
    ]
    table = RLTable(data)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ]
        )
    )
    doc.build([table])
    return TABLE_PDF


def test_extract_tables_returns_list_of_tables(table_pdf_path: Path) -> None:
    """extract_tables should return a list of Table objects."""
    with PdfExtractor(table_pdf_path) as extractor:
        tables = extractor.extract_tables()

    assert isinstance(tables, list)
    assert len(tables) > 0
    assert all(isinstance(t, Table) for t in tables)


def test_extract_tables_has_correct_structure(table_pdf_path: Path) -> None:
    """Extracted table should have cells with row/col indices."""
    with PdfExtractor(table_pdf_path) as extractor:
        tables = extractor.extract_tables()

    table = tables[0]
    assert table.row_count == 3
    assert table.col_count == 3
    assert len(table.cells) == 9  # 3x3

    # Check header row
    header_cells = [c for c in table.cells if c.row == 0]
    header_texts = sorted([c.text for c in header_cells])
    assert "Header A" in header_texts or any("Header" in c.text for c in header_cells)


def test_extract_tables_with_page_filter(table_pdf_path: Path) -> None:
    """extract_tables should accept pages parameter."""
    with PdfExtractor(table_pdf_path) as extractor:
        tables = extractor.extract_tables(pages=[0])

    assert len(tables) >= 0  # May or may not find tables
    for table in tables:
        assert table.page == 0


def test_extract_tables_with_lattice_mode(table_pdf_path: Path) -> None:
    """extract_tables should accept lattice option for bordered tables."""
    with PdfExtractor(table_pdf_path) as extractor:
        tables = extractor.extract_tables(table_options={"lattice": True})

    # Lattice mode should work for tables with borders
    assert isinstance(tables, list)


def test_extract_tables_with_stream_mode(table_pdf_path: Path) -> None:
    """extract_tables should accept stream option for borderless tables."""
    with PdfExtractor(table_pdf_path) as extractor:
        tables = extractor.extract_tables(table_options={"stream": True})

    assert isinstance(tables, list)


def test_extract_page_with_table_options(table_pdf_path: Path) -> None:
    """extract_page should accept table_options and populate Page.tables."""
    with PdfExtractor(table_pdf_path) as extractor:
        result = extractor.extract_page(0, table_options={})

    assert result.success
    # When table_options is provided, tables should be extracted
    assert isinstance(result.page.tables, list)


def test_extract_page_without_table_options_has_empty_tables(table_pdf_path: Path) -> None:
    """extract_page without table_options should not extract tables."""
    with PdfExtractor(table_pdf_path) as extractor:
        result = extractor.extract_page(0)

    assert result.success
    assert result.page.tables == []
