from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from xtra.extractors.base import BaseExtractor, ExtractionResult
from xtra.models import DocumentMetadata, Page, SourceType


class MockExtractor(BaseExtractor):
    """Mock extractor for testing base class."""

    def __init__(self, path: Path, page_count: int = 2) -> None:
        super().__init__(path)
        self._page_count = page_count

    def get_page_count(self) -> int:
        return self._page_count

    def extract_page(self, page: int) -> ExtractionResult:
        if page >= self._page_count:
            return ExtractionResult(
                page=Page(page=page, width=0, height=0, texts=[]),
                success=False,
                error="Page out of range",
            )
        return ExtractionResult(
            page=Page(page=page, width=100.0, height=100.0, texts=[]),
            success=True,
        )

    def get_metadata(self) -> DocumentMetadata:
        return DocumentMetadata(source_type=SourceType.PDF)


def test_extraction_result_success() -> None:
    page = Page(page=0, width=100.0, height=100.0)
    result = ExtractionResult(page=page, success=True)
    assert result.success
    assert result.error is None


def test_extraction_result_failure() -> None:
    page = Page(page=0, width=0, height=0)
    result = ExtractionResult(page=page, success=False, error="Test error")
    assert not result.success
    assert result.error == "Test error"


def test_base_extractor_extract_all_pages() -> None:
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=3)
    doc = extractor.extract()
    assert len(doc.pages) == 3
    assert doc.metadata is not None


def test_base_extractor_extract_specific_pages() -> None:
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=5)
    doc = extractor.extract(pages=[0, 2, 4])
    assert len(doc.pages) == 3
    assert doc.pages[0].page == 0
    assert doc.pages[1].page == 2
    assert doc.pages[2].page == 4


def test_base_extractor_extract_pages_method() -> None:
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=2)
    results = extractor.extract_pages()
    assert len(results) == 2
    assert all(r.success for r in results)


def test_base_extractor_context_manager() -> None:
    with MockExtractor(Path("/tmp/test.pdf")) as extractor:
        doc = extractor.extract()
    assert len(doc.pages) == 2


def test_base_extractor_close() -> None:
    extractor = MockExtractor(Path("/tmp/test.pdf"))
    extractor.close()  # Should not raise


def test_base_extractor_path_attribute() -> None:
    path = Path("/tmp/test.pdf")
    extractor = MockExtractor(path)
    assert extractor.path == path
