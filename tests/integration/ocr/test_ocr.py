"""Integration tests for all extractors.

These tests use real files and services (no mocking).
All extractors are tested via create_extractor factory with parametrization.

Guidelines:
- Local OCR tests (EasyOCR, Tesseract, PaddleOCR) run unconditionally.
- Cloud tests (Azure, Google) require credentials in environment variables.
- Use 2-letter ISO 639-1 language codes (e.g., "en", "fr", "de") for all extractors.
"""

import os
from pathlib import Path

import pytest

from unifex.base import ExecutorType, ExtractorType
from unifex.text_factory import create_extractor

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data"

# All OCR extractors with their expected ocr_engine metadata value
# Note: PaddleOCR is tested separately in test_paddle_ocr.py due to oneDNN/MKL-DNN
# compatibility issues in CI environments. The dedicated tests use different
# initialization that avoids these issues.
OCR_EXTRACTORS = [
    pytest.param(ExtractorType.EASYOCR, "easyocr", id="easyocr"),
    pytest.param(ExtractorType.TESSERACT, "tesseract", id="tesseract"),
    pytest.param(
        ExtractorType.PADDLE,
        "paddleocr",
        id="paddle",
        marks=pytest.mark.skip(reason="Tested in test_paddle_ocr.py; CI has oneDNN issues"),
    ),
    pytest.param(
        ExtractorType.AZURE_DI,
        "azure_document_intelligence",
        id="azure",
        marks=pytest.mark.skipif(
            not os.environ.get("UNIFEX_AZURE_DI_ENDPOINT"),
            reason="Azure credentials not configured",
        ),
    ),
    pytest.param(
        ExtractorType.GOOGLE_DOCAI,
        "google_document_ai",
        id="google",
        marks=pytest.mark.skipif(
            not os.environ.get("UNIFEX_GOOGLE_DOCAI_PROCESSOR_NAME"),
            reason="Google credentials not configured",
        ),
    ),
]

# Local OCR extractors only (for image tests - cloud APIs don't support raw images)
LOCAL_OCR_EXTRACTORS = [
    pytest.param(ExtractorType.EASYOCR, "easyocr", id="easyocr"),
    pytest.param(ExtractorType.TESSERACT, "tesseract", id="tesseract"),
    pytest.param(
        ExtractorType.PADDLE,
        "paddleocr",
        id="paddle",
        marks=pytest.mark.skip(reason="Tested in test_paddle_ocr.py; CI has oneDNN issues"),
    ),
]


def test_pdf_extractor() -> None:
    """Test PDF extractor with real PDF file."""
    with create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", ExtractorType.PDF) as extractor:
        result = extractor.extract()

    doc = result.document
    assert doc.path == TEST_DATA_DIR / "test_pdf_2p_text.pdf"
    assert len(doc.pages) == 2
    assert doc.metadata is not None
    assert doc.metadata.extractor_type == ExtractorType.PDF
    assert result.success is True

    # Verify page 1 content
    page1 = doc.pages[0]
    page1_texts = [t.text for t in page1.texts]
    assert "First page. First text" in page1_texts
    assert "First page. Second text" in page1_texts
    assert "First page. Fourth text" in page1_texts
    assert len(page1.texts) == 3

    # Verify page 2 content
    page2 = doc.pages[1]
    assert len(page2.texts) == 1
    assert page2.texts[0].text == "Second page. Third text"

    # Verify page structure
    for page in doc.pages:
        assert page.width > 0
        assert page.height > 0
        for text in page.texts:
            assert text.bbox is not None
            assert text.bbox.x0 < text.bbox.x1
            assert text.bbox.y0 < text.bbox.y1


@pytest.mark.parametrize("extractor_type,ocr_engine", LOCAL_OCR_EXTRACTORS)
def test_ocr_extract_image(extractor_type: ExtractorType, ocr_engine: str) -> None:
    """Test OCR extraction from image file."""
    with create_extractor(
        TEST_DATA_DIR / "test_image.png",
        extractor_type,
        languages=["en"],
    ) as extractor:
        result = extractor.extract()

    doc = result.document
    assert doc.path == TEST_DATA_DIR / "test_image.png"
    assert len(doc.pages) == 1
    assert doc.metadata is not None
    assert doc.metadata.extractor_type == extractor_type
    assert doc.metadata.extra["ocr_engine"] == ocr_engine

    # Verify OCR detected text
    page = doc.pages[0]
    assert page.width > 0
    assert page.height > 0
    assert len(page.texts) > 0

    all_text = " ".join(t.text for t in page.texts).lower()
    assert len(all_text) > 0, "Expected OCR to extract some text from image"

    # Verify confidence scores
    for text in page.texts:
        assert text.confidence is not None
        assert 0.0 <= text.confidence <= 1.0


@pytest.mark.parametrize("extractor_type,ocr_engine", OCR_EXTRACTORS)
def test_ocr_extract_pdf(extractor_type: ExtractorType, ocr_engine: str) -> None:
    """Test OCR extraction from PDF file."""
    with create_extractor(
        TEST_DATA_DIR / "test_pdf_2p_text.pdf",
        extractor_type,
        languages=["en"],
        dpi=100,
    ) as extractor:
        result = extractor.extract()

    doc = result.document
    assert doc.path == TEST_DATA_DIR / "test_pdf_2p_text.pdf"
    assert len(doc.pages) == 2
    assert doc.metadata is not None
    assert doc.metadata.extractor_type == extractor_type
    assert doc.metadata.extra["ocr_engine"] == ocr_engine

    # Verify pages have content
    for page in doc.pages:
        assert page.width > 0
        assert page.height > 0
        assert len(page.texts) > 0

    # Verify text was extracted
    page1_text = " ".join(t.text for t in doc.pages[0].texts).lower()
    assert len(page1_text) > 0, "Expected OCR to extract some text from page 1"

    page2_text = " ".join(t.text for t in doc.pages[1].texts).lower()
    assert len(page2_text) > 0, "Expected OCR to extract some text from page 2"

    # Verify confidence scores
    for page in doc.pages:
        for text in page.texts:
            assert text.confidence is not None
            assert 0.0 <= text.confidence <= 1.0


# Parallel extraction integration tests


@pytest.mark.parametrize("max_workers", [1, 2])
def test_pdf_extractor_parallel(max_workers: int) -> None:
    """Test PDF extractor with different worker counts."""
    with create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", ExtractorType.PDF) as extractor:
        result = extractor.extract(max_workers=max_workers)

    assert len(result.document.pages) == 2
    # Verify content is identical regardless of worker count
    page1_texts = [t.text for t in result.document.pages[0].texts]
    assert "First page. First text" in page1_texts


def test_pdf_extractor_thread_executor() -> None:
    """Test PDF extractor with thread executor.

    Note: Process executor is not supported for PDF extractor because
    pypdfium2 document handles cannot be pickled across processes.
    Use thread executor (default) for parallel PDF extraction.
    """
    with create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", ExtractorType.PDF) as extractor:
        result = extractor.extract(max_workers=2, executor=ExecutorType.THREAD)

    assert len(result.document.pages) == 2


@pytest.mark.parametrize("extractor_type,_ocr_engine", LOCAL_OCR_EXTRACTORS)
@pytest.mark.parametrize("max_workers", [1, 2])
def test_ocr_extract_parallel(
    extractor_type: ExtractorType, _ocr_engine: str, max_workers: int
) -> None:
    """Test OCR extractors with parallel extraction."""
    with create_extractor(
        TEST_DATA_DIR / "test_pdf_2p_text.pdf",
        extractor_type,
        languages=["en"],
        dpi=100,
    ) as extractor:
        result = extractor.extract(max_workers=max_workers)

    assert len(result.document.pages) == 2
    for page in result.document.pages:
        assert len(page.texts) > 0


@pytest.mark.asyncio
async def test_pdf_extractor_async() -> None:
    """Test PDF extractor with async extraction."""
    with create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", ExtractorType.PDF) as extractor:
        result = await extractor.extract_async(max_workers=2)

    assert len(result.document.pages) == 2
    page1_texts = [t.text for t in result.document.pages[0].texts]
    assert "First page. First text" in page1_texts


@pytest.mark.asyncio
@pytest.mark.parametrize("extractor_type,_ocr_engine", LOCAL_OCR_EXTRACTORS)
async def test_ocr_extract_async(extractor_type: ExtractorType, _ocr_engine: str) -> None:
    """Test OCR extractors with async extraction."""
    with create_extractor(
        TEST_DATA_DIR / "test_pdf_2p_text.pdf",
        extractor_type,
        languages=["en"],
        dpi=100,
    ) as extractor:
        result = await extractor.extract_async(max_workers=2)

    assert len(result.document.pages) == 2
    for page in result.document.pages:
        assert len(page.texts) > 0
