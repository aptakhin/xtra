import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from xtra.cli import main
from xtra.models import SourceType


TEST_DATA_DIR = Path(__file__).parent / "data"


def test_source_type_used_in_cli() -> None:
    # SourceType is used as the extractor choices in CLI
    assert SourceType.PDF == "pdf"
    assert SourceType.EASYOCR == "easyocr"
    assert SourceType.PDF_EASYOCR == "pdf-easyocr"
    assert SourceType.AZURE_DI == "azure-di"


def test_cli_pdf_extractor(capsys: pytest.CaptureFixture) -> None:
    test_args = ["cli", str(TEST_DATA_DIR / "test_pdf_2p_text.pdf"), "--extractor", "pdf"]
    with patch.object(sys, "argv", test_args):
        main()
    captured = capsys.readouterr()
    assert "=== Page 1 ===" in captured.out
    assert "First page. First text" in captured.out


def test_cli_pdf_extractor_json(capsys: pytest.CaptureFixture) -> None:
    test_args = [
        "cli",
        str(TEST_DATA_DIR / "test_pdf_2p_text.pdf"),
        "--extractor",
        "pdf",
        "--json",
    ]
    with patch.object(sys, "argv", test_args):
        main()
    captured = capsys.readouterr()
    assert '"text": "First page. First text"' in captured.out


def test_cli_pdf_extractor_specific_pages(capsys: pytest.CaptureFixture) -> None:
    test_args = [
        "cli",
        str(TEST_DATA_DIR / "test_pdf_2p_text.pdf"),
        "--extractor",
        "pdf",
        "--pages",
        "0",
    ]
    with patch.object(sys, "argv", test_args):
        main()
    captured = capsys.readouterr()
    assert "=== Page 1 ===" in captured.out
    assert "=== Page 2 ===" not in captured.out


def test_cli_file_not_found() -> None:
    test_args = ["cli", "/nonexistent/file.pdf", "--extractor", "pdf"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


def test_cli_languages_parsing(capsys: pytest.CaptureFixture) -> None:
    test_args = [
        "cli",
        str(TEST_DATA_DIR / "test_pdf_2p_text.pdf"),
        "--extractor",
        "pdf",
        "--lang",
        "en,it,de",
    ]
    with patch.object(sys, "argv", test_args):
        main()
    # Should not raise, languages are parsed but only used for OCR
    captured = capsys.readouterr()
    assert "=== Page 1 ===" in captured.out
