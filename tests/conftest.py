"""Shared pytest configuration and fixtures."""

import os
from pathlib import Path

import pytest
from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

# Symlink mappings for clean documentation examples
DOC_SYMLINKS = {
    "document.pdf": TEST_DATA_DIR / "test_pdf_2p_text.pdf",
    "scanned.pdf": TEST_DATA_DIR / "test_pdf_2p_text.pdf",
    "image.png": TEST_DATA_DIR / "test_image.png",
    "table.pdf": TEST_DATA_DIR / "test_pdf_table.pdf",
}


def _setup_doc_symlinks(namespace):
    """Create symlinks for documentation examples."""
    for link_name, target in DOC_SYMLINKS.items():
        link_path = Path(link_name)
        if not link_path.exists():
            link_path.symlink_to(target.resolve())


def _teardown_doc_symlinks(namespace):
    """Remove symlinks created for documentation examples."""
    for link_name in DOC_SYMLINKS:
        link_path = Path(link_name)
        if link_path.is_symlink():
            link_path.unlink()


# Sybil configuration for testing documentation code examples
# Uses SkipParser to allow skipping examples that need external resources
pytest_collect_file = Sybil(
    parsers=[
        PythonCodeBlockParser(),
        SkipParser(),
    ],
    patterns=["*.md"],
    path=str(Path(__file__).parent.parent / "docs"),
    setup=_setup_doc_symlinks,
    teardown=_teardown_doc_symlinks,
).pytest()


@pytest.fixture(scope="module")
def vcr_config():
    """VCR configuration to filter sensitive data from recordings."""
    return {
        "filter_headers": [
            ("authorization", "Bearer REDACTED"),
            ("x-api-key", "REDACTED"),
            ("api-key", "REDACTED"),
            ("x-goog-api-key", "REDACTED"),
        ],
        "filter_post_data_parameters": [
            ("api_key", "REDACTED"),
        ],
        "record_mode": "once",
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
    }
