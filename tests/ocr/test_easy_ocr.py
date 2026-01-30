"""Unit tests for EasyOCR extractor with real files."""

from pathlib import Path

from xtra.extractors.ocr.easy_ocr import EasyOcrExtractor

TEST_DATA_DIR = Path(__file__).parent.parent / "data"


class TestEasyOcrExtractorWithPdf:
    """Tests for EasyOcrExtractor with PDF files."""

    def test_init_with_pdf(self) -> None:
        extractor = EasyOcrExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf")
        assert extractor.languages == ["en"]
        assert extractor.gpu is False
        assert extractor.dpi == 200
        assert extractor._images.is_pdf is True
        assert extractor.get_page_count() == 2

    def test_init_with_pdf_custom_dpi(self) -> None:
        extractor = EasyOcrExtractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf", languages=["en", "it"], dpi=300
        )
        assert extractor.dpi == 300
        assert extractor._images.is_pdf is True

    def test_get_page_count_pdf(self) -> None:
        extractor = EasyOcrExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf")
        assert extractor.get_page_count() == 2

    def test_extract_page_out_of_range_pdf(self) -> None:
        extractor = EasyOcrExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf")
        result = extractor.extract_page(5)

        assert not result.success
        assert result.error is not None
        assert "out of range" in result.error.lower()
