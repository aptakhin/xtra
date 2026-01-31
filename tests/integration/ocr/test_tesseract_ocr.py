"""Unit tests for Tesseract OCR extractor with real files."""

from pathlib import Path

from xtra.ocr.extractors.tesseract_ocr import TesseractOcrExtractor, _convert_lang_code

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data"


class TestLangCodeConversion:
    """Tests for language code conversion."""

    def test_convert_known_codes(self) -> None:
        assert _convert_lang_code("en") == "eng"
        assert _convert_lang_code("fr") == "fra"
        assert _convert_lang_code("de") == "deu"

    def test_passthrough_unknown_codes(self) -> None:
        assert _convert_lang_code("eng") == "eng"
        assert _convert_lang_code("unknown") == "unknown"


class TestTesseractOcrExtractorWithPdf:
    """Tests for TesseractOcrExtractor with PDF files."""

    def test_init_with_pdf(self) -> None:
        extractor = TesseractOcrExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf")
        assert extractor.languages == ["en"]
        assert extractor.dpi == 200
        assert extractor._images.is_pdf is True
        assert extractor.get_page_count() == 2

    def test_init_with_pdf_custom_dpi(self) -> None:
        extractor = TesseractOcrExtractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf", languages=["en", "fr"], dpi=300
        )
        assert extractor.dpi == 300
        assert extractor.languages == ["en", "fr"]
        assert extractor._tesseract_languages == ["eng", "fra"]

    def test_get_page_count_pdf(self) -> None:
        extractor = TesseractOcrExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf")
        assert extractor.get_page_count() == 2

    def test_extract_page_out_of_range_pdf(self) -> None:
        extractor = TesseractOcrExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf")
        result = extractor.extract_page(5)

        assert not result.success
        assert result.error is not None
        assert "out of range" in result.error.lower()
