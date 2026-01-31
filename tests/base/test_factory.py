"""Unit tests for the extractor factory.

Note: Slow tests that load EasyOCR/PaddleOCR models are in tests/integration/ocr/test_factory.py
"""

import os
from pathlib import Path

import pytest

from xtra.extractors.factory import _get_credential, create_extractor
from xtra.models import ExtractorType

TEST_DATA_DIR = Path(__file__).parent.parent / "data"


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
    """Tests for extractor creation with real files (fast extractors only).

    Note: OCR extractor tests (EasyOCR, PaddleOCR, Tesseract) are in tests/integration/ocr/
    because they load ML models which takes several seconds.
    """

    def test_creates_pdf_extractor(self) -> None:
        extractor = create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", ExtractorType.PDF)
        assert extractor is not None
        assert extractor.get_page_count() == 2
        extractor.close()


class TestStringPathSupport:
    """Tests for string path support in create_extractor."""

    def test_accepts_string_path(self) -> None:
        """Verify create_extractor accepts string paths."""
        str_path = str(TEST_DATA_DIR / "test_pdf_2p_text.pdf")
        extractor = create_extractor(str_path, ExtractorType.PDF)
        assert extractor.path == Path(str_path)
        assert extractor.get_page_count() == 2
        extractor.close()

    def test_string_path_converts_to_path_object(self) -> None:
        """Verify string path is converted to Path internally."""
        str_path = str(TEST_DATA_DIR / "test_pdf_2p_text.pdf")
        extractor = create_extractor(str_path, ExtractorType.PDF)
        assert isinstance(extractor.path, Path)
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
