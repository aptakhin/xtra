"""Slow factory tests - tests that load real OCR ML models.

These tests are slow because they instantiate EasyOCR and PaddleOCR
extractors which load ML models into memory.
"""

from pathlib import Path

import pytest

from unifex.base import ExtractorType
from unifex.text_factory import create_extractor

# Check if paddleocr is available
try:
    import paddleocr  # noqa: F401

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data"


class TestCreateExtractorWithRealOCR:
    """Tests for extractor creation that load real OCR models."""

    def test_creates_tesseract_extractor(self) -> None:
        extractor = create_extractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            ExtractorType.TESSERACT,
            languages=["en", "fr"],
            dpi=150,
        )
        assert extractor.languages == ["en", "fr"]  # type: ignore[attr-defined]
        assert extractor.dpi == 150  # type: ignore[attr-defined]
        assert extractor.get_page_count() == 2
        extractor.close()

    def test_creates_easyocr_extractor(self) -> None:
        extractor = create_extractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            ExtractorType.EASYOCR,
            languages=["en", "it"],
            dpi=150,
        )
        assert extractor.languages == ["en", "it"]  # type: ignore[attr-defined]
        assert extractor.dpi == 150  # type: ignore[attr-defined]
        assert extractor.gpu is False  # type: ignore[attr-defined]
        assert extractor.get_page_count() == 2
        extractor.close()

    def test_creates_easyocr_with_gpu_flag(self) -> None:
        extractor = create_extractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            ExtractorType.EASYOCR,
            use_gpu=True,
        )
        assert extractor.gpu is True  # type: ignore[attr-defined]
        extractor.close()

    @pytest.mark.skipif(not HAS_PADDLE, reason="paddleocr not installed")
    def test_creates_paddle_extractor(self) -> None:
        extractor = create_extractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            ExtractorType.PADDLE,
            languages=["en"],
            dpi=150,
        )
        assert extractor.lang == "en"  # type: ignore[attr-defined]
        assert extractor.dpi == 150  # type: ignore[attr-defined]
        assert extractor.use_gpu is False  # type: ignore[attr-defined]
        assert extractor.get_page_count() == 2
        extractor.close()

    @pytest.mark.skipif(not HAS_PADDLE, reason="paddleocr not installed")
    def test_creates_paddle_with_gpu_flag(self) -> None:
        extractor = create_extractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            ExtractorType.PADDLE,
            use_gpu=True,
        )
        assert extractor.use_gpu is True  # type: ignore[attr-defined]
        extractor.close()


class TestCreateExtractorDefaultsWithRealOCR:
    """Tests for default parameter handling that load real OCR models."""

    def test_default_languages(self) -> None:
        extractor = create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", ExtractorType.EASYOCR)
        assert extractor.languages == ["en"]  # type: ignore[attr-defined]
        extractor.close()

    def test_default_dpi(self) -> None:
        extractor = create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", ExtractorType.EASYOCR)
        assert extractor.dpi == 200  # type: ignore[attr-defined]
        extractor.close()

    def test_custom_dpi(self) -> None:
        extractor = create_extractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf", ExtractorType.EASYOCR, dpi=300
        )
        assert extractor.dpi == 300  # type: ignore[attr-defined]
        extractor.close()
