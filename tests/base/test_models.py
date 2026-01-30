from pathlib import Path

from xtra.models import (
    BBox,
    Document,
    ExtractorMetadata,
    ExtractorType,
    FontInfo,
    Page,
    TextBlock,
)


def test_bbox_creation() -> None:
    bbox = BBox(x0=0.0, y0=0.0, x1=100.0, y1=50.0)
    assert bbox.x0 == 0.0
    assert bbox.y0 == 0.0
    assert bbox.x1 == 100.0
    assert bbox.y1 == 50.0


def test_font_info_defaults() -> None:
    font = FontInfo()
    assert font.name is None
    assert font.size is None
    assert font.flags is None
    assert font.weight is None


def test_font_info_with_values() -> None:
    font = FontInfo(name="Helvetica", size=12.0, weight=400)
    assert font.name == "Helvetica"
    assert font.size == 12.0
    assert font.weight == 400


def test_text_block_creation() -> None:
    bbox = BBox(x0=0.0, y0=0.0, x1=100.0, y1=20.0)
    block = TextBlock(text="Hello", bbox=bbox)
    assert block.text == "Hello"
    assert block.rotation == 0.0
    assert block.confidence is None
    assert block.font_info is None


def test_text_block_with_confidence() -> None:
    bbox = BBox(x0=0.0, y0=0.0, x1=100.0, y1=20.0)
    block = TextBlock(text="Hello", bbox=bbox, confidence=0.95, rotation=5.0)
    assert block.confidence == 0.95
    assert block.rotation == 5.0


def test_page_creation() -> None:
    page = Page(page=0, width=595.0, height=842.0)
    assert page.page == 0
    assert page.width == 595.0
    assert page.height == 842.0
    assert page.texts == []


def test_page_with_texts() -> None:
    bbox = BBox(x0=0.0, y0=0.0, x1=100.0, y1=20.0)
    block = TextBlock(text="Test", bbox=bbox)
    page = Page(page=0, width=595.0, height=842.0, texts=[block])
    assert len(page.texts) == 1
    assert page.texts[0].text == "Test"


def test_document_metadata_pdf() -> None:
    meta = ExtractorMetadata(extractor_type=ExtractorType.PDF, creator="Test")
    assert meta.extractor_type == ExtractorType.PDF
    assert meta.creator == "Test"


def test_document_metadata_easyocr() -> None:
    meta = ExtractorMetadata(
        extractor_type=ExtractorType.EASYOCR,
        extra={"ocr_engine": "easyocr"},
    )
    assert meta.extractor_type == ExtractorType.EASYOCR
    assert meta.extra["ocr_engine"] == "easyocr"


def test_document_creation() -> None:
    path = Path("/tmp/test.pdf")
    doc = Document(path=path)
    assert doc.path == path
    assert doc.pages == []
    assert doc.metadata is None


def test_document_with_pages() -> None:
    path = Path("/tmp/test.pdf")
    page = Page(page=0, width=595.0, height=842.0)
    meta = ExtractorMetadata(extractor_type=ExtractorType.PDF)
    doc = Document(path=path, pages=[page], metadata=meta)
    assert len(doc.pages) == 1
    assert doc.metadata is not None


def test_extractor_type_enum() -> None:
    assert ExtractorType.PDF == "pdf"
    assert ExtractorType.EASYOCR == "easyocr"
    assert ExtractorType.TESSERACT == "tesseract"
    assert ExtractorType.PADDLE == "paddle"
    assert ExtractorType.AZURE_DI == "azure-di"
    assert ExtractorType.GOOGLE_DOCAI == "google-docai"
