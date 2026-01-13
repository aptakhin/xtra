from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from xtra.extractors import PdfExtractor
from xtra.models import SourceType


TEST_DATA_DIR = Path(__file__).parent / "data"


def test_extract_test4_pdf_page1_has_3_texts() -> None:
    with PdfExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf") as extractor:
        doc = extractor.extract()
    page1 = doc.pages[0]
    assert len(page1.texts) == 3
    texts = [t.text for t in page1.texts]
    assert "First page. First text" in texts
    assert "First page. Second text" in texts
    assert "First page. Fourth text" in texts


def test_extract_test4_pdf_page2_has_1_text() -> None:
    with PdfExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf") as extractor:
        doc = extractor.extract()
    page2 = doc.pages[1]
    assert len(page2.texts) == 1
    assert page2.texts[0].text == "Second page. Third text"


def test_extract_test4_pdf_texts_have_bbox() -> None:
    with PdfExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf") as extractor:
        doc = extractor.extract()
    for page in doc.pages:
        for text_block in page.texts:
            assert text_block.bbox is not None
            assert text_block.bbox.x0 < text_block.bbox.x1
            assert text_block.bbox.y0 < text_block.bbox.y1


def test_extract_test4_pdf_metadata() -> None:
    with PdfExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf") as extractor:
        doc = extractor.extract()
    assert doc.metadata is not None
    assert doc.metadata.source_type == SourceType.PDF


def test_extract_test4_pdf_has_font_info() -> None:
    with PdfExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf") as extractor:
        doc = extractor.extract()
    page1 = doc.pages[0]
    has_font = any(t.font_info is not None for t in page1.texts)
    assert has_font, "At least one text block should have font info"


def test_extract_single_page() -> None:
    with PdfExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf") as extractor:
        doc = extractor.extract(pages=[0])
    assert len(doc.pages) == 1
    assert doc.pages[0].page == 0


def test_extract_page_method() -> None:
    with PdfExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf") as extractor:
        result = extractor.extract_page(0)
    assert result.success
    assert len(result.page.texts) == 3


def test_get_page_count() -> None:
    with PdfExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf") as extractor:
        assert extractor.get_page_count() == 2
