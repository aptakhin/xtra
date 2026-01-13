"""Unit tests for the extractor factory."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from xtra.extractors.factory import create_extractor, _get_credential
from xtra.models import SourceType


TEST_DATA_DIR = Path(__file__).parent / "data"


class TestGetCredential:
    """Tests for _get_credential helper function."""

    def test_returns_from_dict_when_present(self) -> None:
        credentials = {"KEY": "from_dict"}
        result = _get_credential("KEY", credentials)
        assert result == "from_dict"

    def test_returns_from_env_when_not_in_dict(self) -> None:
        with patch.dict(os.environ, {"KEY": "from_env"}):
            result = _get_credential("KEY", None)
            assert result == "from_env"

    def test_dict_takes_precedence_over_env(self) -> None:
        credentials = {"KEY": "from_dict"}
        with patch.dict(os.environ, {"KEY": "from_env"}):
            result = _get_credential("KEY", credentials)
            assert result == "from_dict"

    def test_returns_none_when_not_found(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = _get_credential("NONEXISTENT", None)
            assert result is None


class TestCreateExtractorPdf:
    """Tests for PDF extractor creation."""

    def test_creates_pdf_extractor(self) -> None:
        extractor = create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", SourceType.PDF)
        assert extractor is not None
        assert extractor.get_page_count() == 2
        extractor.close()


class TestCreateExtractorEasyOcr:
    """Tests for EasyOCR extractor creation."""

    def test_creates_easyocr_extractor(self) -> None:
        with patch("xtra.extractors.ocr.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_open.return_value = mock_img

            extractor = create_extractor(
                Path("/fake/image.png"), SourceType.EASYOCR, languages=["en", "it"]
            )
            assert extractor.languages == ["en", "it"]  # type: ignore[attr-defined]
            assert extractor.gpu is False  # type: ignore[attr-defined]

    def test_creates_easyocr_with_gpu(self) -> None:
        with patch("xtra.extractors.ocr.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_open.return_value = mock_img

            extractor = create_extractor(Path("/fake/image.png"), SourceType.EASYOCR, use_gpu=True)
            assert extractor.gpu is True  # type: ignore[attr-defined]

    def test_creates_easyocr_extractor_with_pdf(self) -> None:
        extractor = create_extractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            SourceType.EASYOCR,
            languages=["en"],
            dpi=150,
        )
        assert extractor.dpi == 150  # type: ignore[attr-defined]
        assert extractor.get_page_count() == 2
        extractor.close()


class TestCreateExtractorTesseract:
    """Tests for Tesseract extractor creation."""

    def test_creates_tesseract_extractor(self) -> None:
        with (
            patch("xtra.extractors.tesseract_ocr.Image.open") as mock_open,
            patch("xtra.extractors.tesseract_ocr.pytesseract"),
        ):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_open.return_value = mock_img

            extractor = create_extractor(
                Path("/fake/image.png"), SourceType.TESSERACT, languages=["en", "it"]
            )
            assert extractor.languages == ["en", "it"]  # type: ignore[attr-defined]

    def test_creates_tesseract_extractor_with_pdf(self) -> None:
        extractor = create_extractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            SourceType.TESSERACT,
            languages=["en"],
            dpi=150,
        )
        assert extractor.dpi == 150  # type: ignore[attr-defined]
        assert extractor.get_page_count() == 2
        extractor.close()


class TestCreateExtractorPaddle:
    """Tests for PaddleOCR extractor creation."""

    def test_creates_paddle_extractor(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR"),
            patch("xtra.extractors.paddle_ocr.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_image.open.return_value = mock_img

            extractor = create_extractor(
                Path("/fake/image.png"), SourceType.PADDLE, languages=["ch"]
            )
            assert extractor.lang == "ch"  # type: ignore[attr-defined]
            assert extractor.use_gpu is False  # type: ignore[attr-defined]

    def test_creates_paddle_with_gpu(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR"),
            patch("xtra.extractors.paddle_ocr.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_image.open.return_value = mock_img

            extractor = create_extractor(Path("/fake/image.png"), SourceType.PADDLE, use_gpu=True)
            assert extractor.use_gpu is True  # type: ignore[attr-defined]

    def test_creates_paddle_extractor_with_pdf(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR"),
            patch("xtra.extractors.paddle_ocr.pdfium") as mock_pdfium,
        ):
            mock_pdf = MagicMock()
            mock_page = MagicMock()
            mock_bitmap = MagicMock()
            mock_pil_img = MagicMock()
            mock_pil_img.size = (800, 600)

            mock_bitmap.to_pil.return_value = mock_pil_img
            mock_page.render.return_value = mock_bitmap
            mock_pdf.__iter__ = lambda self: iter([mock_page])
            mock_pdfium.PdfDocument.return_value = mock_pdf

            extractor = create_extractor(
                Path("/fake/document.pdf"), SourceType.PADDLE, languages=["en"], dpi=150
            )
            assert extractor.dpi == 150  # type: ignore[attr-defined]


class TestCreateExtractorAzure:
    """Tests for Azure Document Intelligence extractor creation."""

    def test_creates_azure_extractor_with_credentials_dict(self) -> None:
        with patch("xtra.extractors.azure_di.DocumentIntelligenceClient"):
            extractor = create_extractor(
                TEST_DATA_DIR / "test_pdf_2p_text.pdf",
                SourceType.AZURE_DI,
                credentials={
                    "XTRA_AZURE_DI_ENDPOINT": "https://test.cognitiveservices.azure.com",
                    "XTRA_AZURE_DI_KEY": "test-key",
                },
            )
            assert extractor is not None

    def test_creates_azure_extractor_with_env_vars(self) -> None:
        with (
            patch("xtra.extractors.azure_di.DocumentIntelligenceClient"),
            patch.dict(
                os.environ,
                {
                    "XTRA_AZURE_DI_ENDPOINT": "https://test.cognitiveservices.azure.com",
                    "XTRA_AZURE_DI_KEY": "test-key",
                },
            ),
        ):
            extractor = create_extractor(
                TEST_DATA_DIR / "test_pdf_2p_text.pdf", SourceType.AZURE_DI
            )
            assert extractor is not None

    def test_raises_when_azure_credentials_missing(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="Azure credentials required"),
        ):
            create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", SourceType.AZURE_DI)


class TestCreateExtractorGoogle:
    """Tests for Google Document AI extractor creation."""

    def test_creates_google_extractor_with_credentials_dict(self) -> None:
        with (
            patch("xtra.extractors.google_docai.documentai"),
            patch("xtra.extractors.google_docai.service_account"),
        ):
            extractor = create_extractor(
                TEST_DATA_DIR / "test_pdf_2p_text.pdf",
                SourceType.GOOGLE_DOCAI,
                credentials={
                    "XTRA_GOOGLE_DOCAI_PROCESSOR_NAME": "projects/test/locations/us/processors/123",
                    "XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH": "/path/to/creds.json",
                },
            )
            assert extractor is not None

    def test_creates_google_extractor_with_env_vars(self) -> None:
        with (
            patch("xtra.extractors.google_docai.documentai"),
            patch("xtra.extractors.google_docai.service_account"),
            patch.dict(
                os.environ,
                {
                    "XTRA_GOOGLE_DOCAI_PROCESSOR_NAME": "projects/test/locations/us/processors/123",
                    "XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH": "/path/to/creds.json",
                },
            ),
        ):
            extractor = create_extractor(
                TEST_DATA_DIR / "test_pdf_2p_text.pdf", SourceType.GOOGLE_DOCAI
            )
            assert extractor is not None

    def test_raises_when_google_processor_name_missing(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="Google Document AI processor name required"),
        ):
            create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", SourceType.GOOGLE_DOCAI)

    def test_raises_when_google_credentials_path_missing(self) -> None:
        with (
            patch.dict(
                os.environ,
                {"XTRA_GOOGLE_DOCAI_PROCESSOR_NAME": "projects/test/locations/us/processors/123"},
                clear=True,
            ),
            pytest.raises(ValueError, match="Google Document AI credentials path required"),
        ):
            create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", SourceType.GOOGLE_DOCAI)


class TestCreateExtractorDefaults:
    """Tests for default parameter handling."""

    def test_default_languages(self) -> None:
        with patch("xtra.extractors.ocr.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_open.return_value = mock_img

            extractor = create_extractor(Path("/fake/image.png"), SourceType.EASYOCR)
            assert extractor.languages == ["en"]  # type: ignore[attr-defined]

    def test_default_dpi(self) -> None:
        extractor = create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", SourceType.EASYOCR)
        assert extractor.dpi == 200  # type: ignore[attr-defined]
        extractor.close()

    def test_custom_dpi(self) -> None:
        extractor = create_extractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf", SourceType.EASYOCR, dpi=300
        )
        assert extractor.dpi == 300  # type: ignore[attr-defined]
        extractor.close()
