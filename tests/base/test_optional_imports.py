"""Tests for optional dependency handling and lazy imports."""

import builtins
import sys
from unittest.mock import patch

import pytest


class TestLazyImports:
    """Test lazy import functionality in __init__.py modules."""

    def test_xtra_getattr_returns_extractor(self):
        """Test that __getattr__ returns the correct extractor class."""
        from xtra import EasyOcrExtractor

        assert EasyOcrExtractor is not None
        assert EasyOcrExtractor.__name__ == "EasyOcrExtractor"

    def test_xtra_getattr_raises_attribute_error_for_unknown(self):
        """Test that __getattr__ raises AttributeError for unknown names."""
        import xtra

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = xtra.NonExistentClass

    def test_extractors_getattr_returns_extractor(self):
        """Test that extractors __getattr__ returns the correct extractor class."""
        from xtra.extractors import PaddleOcrExtractor

        assert PaddleOcrExtractor is not None
        assert PaddleOcrExtractor.__name__ == "PaddleOcrExtractor"

    def test_extractors_getattr_raises_attribute_error_for_unknown(self):
        """Test that extractors __getattr__ raises AttributeError for unknown names."""
        from xtra import extractors

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = extractors.NonExistentClass


class TestCheckFunctions:
    """Test the _check_xxx_installed functions with mocked imports."""

    def test_check_easyocr_installed_raises_when_missing(self):
        """Test that _check_easyocr_installed raises ImportError when easyocr is missing."""
        from xtra.extractors.ocr.easy_ocr import _check_easyocr_installed

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "easyocr" in name:
                raise ImportError("No module named easyocr")
            return original_import(name, *args, **kwargs)

        # Clear cached modules
        saved = {k: v for k, v in sys.modules.items() if "easyocr" in k}
        for k in saved:
            del sys.modules[k]

        with patch.object(builtins, "__import__", mock_import):
            with pytest.raises(ImportError, match="pip install xtra\\[easyocr\\]"):
                _check_easyocr_installed()

        # Restore modules
        sys.modules.update(saved)

    def test_check_pytesseract_installed_raises_when_missing(self):
        """Test that _check_pytesseract_installed raises ImportError when pytesseract is missing."""
        from xtra.extractors.ocr.tesseract_ocr import _check_pytesseract_installed

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "pytesseract" in name:
                raise ImportError("No module named pytesseract")
            return original_import(name, *args, **kwargs)

        saved = {k: v for k, v in sys.modules.items() if "pytesseract" in k}
        for k in saved:
            del sys.modules[k]

        with patch.object(builtins, "__import__", mock_import):
            with pytest.raises(ImportError, match="pip install xtra\\[tesseract\\]"):
                _check_pytesseract_installed()

        sys.modules.update(saved)

    def test_check_paddleocr_installed_raises_when_missing(self):
        """Test that _check_paddleocr_installed raises ImportError when paddleocr is missing."""
        from xtra.extractors.ocr.paddle_ocr import _check_paddleocr_installed

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "paddleocr" in name:
                raise ImportError("No module named paddleocr")
            return original_import(name, *args, **kwargs)

        saved = {k: v for k, v in sys.modules.items() if "paddleocr" in k}
        for k in saved:
            del sys.modules[k]

        with patch.object(builtins, "__import__", mock_import):
            with pytest.raises(ImportError, match="pip install xtra\\[paddle\\]"):
                _check_paddleocr_installed()

        sys.modules.update(saved)

    def test_check_azure_installed_raises_when_missing(self):
        """Test that _check_azure_installed raises ImportError when azure SDK is missing."""
        from xtra.extractors.ocr.azure_di import _check_azure_installed

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "azure" in name:
                raise ImportError("No module named azure")
            return original_import(name, *args, **kwargs)

        saved = {k: v for k, v in sys.modules.items() if "azure" in k}
        for k in saved:
            del sys.modules[k]

        with patch.object(builtins, "__import__", mock_import):
            with pytest.raises(ImportError, match="pip install xtra\\[azure\\]"):
                _check_azure_installed()

        sys.modules.update(saved)

    def test_check_google_docai_installed_raises_when_missing(self):
        """Test that _check_google_docai_installed raises ImportError when google SDK is missing."""
        from xtra.extractors.ocr.google_docai import _check_google_docai_installed

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "google" in name:
                raise ImportError("No module named google")
            return original_import(name, *args, **kwargs)

        saved = {k: v for k, v in sys.modules.items() if "google" in k}
        for k in saved:
            del sys.modules[k]

        with patch.object(builtins, "__import__", mock_import):
            with pytest.raises(ImportError, match="pip install xtra\\[google\\]"):
                _check_google_docai_installed()

        sys.modules.update(saved)
