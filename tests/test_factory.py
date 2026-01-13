"""Unit tests for the extractor factory."""

import os
from pathlib import Path

import pytest

from xtra.extractors.factory import create_extractor, _get_credential
from xtra.models import ExtractorType


TEST_DATA_DIR = Path(__file__).parent / "data"


class TestGetCredential:
    """Tests for _get_credential helper function."""

    def test_returns_from_dict_when_present(self) -> None:
        credentials = {"KEY": "from_dict"}
        result = _get_credential("KEY", credentials)
        assert result == "from_dict"

    def test_returns_from_env_when_not_in_dict(self) -> None:
        original = os.environ.get("KEY")
        try:
            os.environ["KEY"] = "from_env"
            result = _get_credential("KEY", None)
            assert result == "from_env"
        finally:
            if original is None:
                os.environ.pop("KEY", None)
            else:
                os.environ["KEY"] = original

    def test_dict_takes_precedence_over_env(self) -> None:
        credentials = {"KEY": "from_dict"}
        original = os.environ.get("KEY")
        try:
            os.environ["KEY"] = "from_env"
            result = _get_credential("KEY", credentials)
            assert result == "from_dict"
        finally:
            if original is None:
                os.environ.pop("KEY", None)
            else:
                os.environ["KEY"] = original

    def test_returns_none_when_not_found(self) -> None:
        result = _get_credential("NONEXISTENT_KEY_12345", None)
        assert result is None


class TestCreateExtractorWithRealFiles:
    """Tests for extractor creation with real files."""

    def test_creates_pdf_extractor(self) -> None:
        extractor = create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", ExtractorType.PDF)
        assert extractor is not None
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

    def test_creates_paddle_with_gpu_flag(self) -> None:
        extractor = create_extractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            ExtractorType.PADDLE,
            use_gpu=True,
        )
        assert extractor.use_gpu is True  # type: ignore[attr-defined]
        extractor.close()


class TestCreateExtractorDefaults:
    """Tests for default parameter handling."""

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


class TestCreateExtractorCredentialValidation:
    """Tests for credential validation in cloud extractors."""

    def test_raises_when_azure_credentials_missing(self) -> None:
        # Save and clear relevant env vars
        saved = {
            k: os.environ.pop(k, None) for k in ["XTRA_AZURE_DI_ENDPOINT", "XTRA_AZURE_DI_KEY"]
        }
        try:
            with pytest.raises(ValueError, match="Azure credentials required"):
                create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", ExtractorType.AZURE_DI)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_raises_when_google_processor_name_missing(self) -> None:
        saved = {
            k: os.environ.pop(k, None)
            for k in ["XTRA_GOOGLE_DOCAI_PROCESSOR_NAME", "XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH"]
        }
        try:
            with pytest.raises(ValueError, match="Google Document AI processor name required"):
                create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", ExtractorType.GOOGLE_DOCAI)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_raises_when_google_credentials_path_missing(self) -> None:
        saved = {
            k: os.environ.pop(k, None)
            for k in ["XTRA_GOOGLE_DOCAI_PROCESSOR_NAME", "XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH"]
        }
        try:
            os.environ["XTRA_GOOGLE_DOCAI_PROCESSOR_NAME"] = (
                "projects/test/locations/us/processors/123"
            )
            with pytest.raises(ValueError, match="Google Document AI credentials path required"):
                create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", ExtractorType.GOOGLE_DOCAI)
        finally:
            os.environ.pop("XTRA_GOOGLE_DOCAI_PROCESSOR_NAME", None)
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
