from pathlib import Path

import pytest

from xtra.extractors.base import BaseExtractor, ExtractionResult
from xtra.models import ExtractorMetadata, Page, ExtractorType


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

    def get_extractor_metadata(self) -> ExtractorMetadata:
        return ExtractorMetadata(extractor_type=ExtractorType.PDF)


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


# Parallel extraction tests


def test_extract_pages_parallel_thread_ordering() -> None:
    """Test that parallel extraction with threads maintains page order."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=10)
    results = extractor.extract_pages(pages=[9, 0, 5, 3], max_workers=4, executor="thread")
    assert len(results) == 4
    assert results[0].page.page == 9
    assert results[1].page.page == 0
    assert results[2].page.page == 5
    assert results[3].page.page == 3


def test_extract_pages_parallel_process_ordering() -> None:
    """Test that parallel extraction with processes maintains page order."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=10)
    results = extractor.extract_pages(pages=[9, 0, 5, 3], max_workers=4, executor="process")
    assert len(results) == 4
    assert results[0].page.page == 9
    assert results[1].page.page == 0
    assert results[2].page.page == 5
    assert results[3].page.page == 3


def test_extract_pages_parallel_all() -> None:
    """Test parallel extraction of all pages."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=5)
    results = extractor.extract_pages(max_workers=3)
    assert len(results) == 5
    assert all(r.success for r in results)


def test_extract_parallel_single_page() -> None:
    """Test that single page extraction works with max_workers > 1."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=1)
    results = extractor.extract_pages(max_workers=4)
    assert len(results) == 1


def test_extract_parallel_error_handling() -> None:
    """Test that parallel extraction handles per-page errors."""

    class FailingExtractor(MockExtractor):
        def extract_page(self, page: int) -> ExtractionResult:
            if page == 2:
                raise ValueError("Simulated error")
            return super().extract_page(page)

    extractor = FailingExtractor(Path("/tmp/test.pdf"), page_count=5)
    results = extractor.extract_pages(max_workers=3)
    assert len(results) == 5
    assert results[2].success is False
    assert "Simulated error" in str(results[2].error)
    assert all(r.success for i, r in enumerate(results) if i != 2)


def test_extract_backward_compatible() -> None:
    """Test that default behavior (max_workers=1) matches old behavior."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=3)
    results_default = extractor.extract_pages()
    results_explicit = extractor.extract_pages(max_workers=1)
    assert len(results_default) == len(results_explicit)
    for r1, r2 in zip(results_default, results_explicit):
        assert r1.page.page == r2.page.page


def test_extract_with_parallel_workers() -> None:
    """Test extract() method with max_workers parameter."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=5)
    doc = extractor.extract(max_workers=2)
    assert len(doc.pages) == 5


@pytest.mark.parametrize("executor", ["thread", "process"])
def test_extract_with_executor_types(executor: str) -> None:
    """Test extract() method with different executor types."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=3)
    doc = extractor.extract(max_workers=2, executor=executor)  # type: ignore[arg-type]
    assert len(doc.pages) == 3


# Async extraction tests


@pytest.mark.asyncio
async def test_extract_pages_async() -> None:
    """Test async extraction of pages."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=3)
    results = await extractor.extract_pages_async()
    assert len(results) == 3
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_extract_pages_async_parallel() -> None:
    """Test async parallel extraction maintains order."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=5)
    results = await extractor.extract_pages_async(pages=[4, 2, 0], max_workers=3)
    assert len(results) == 3
    assert results[0].page.page == 4
    assert results[1].page.page == 2
    assert results[2].page.page == 0


@pytest.mark.asyncio
async def test_extract_async() -> None:
    """Test async extract() method."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=3)
    doc = await extractor.extract_async(max_workers=2)
    assert len(doc.pages) == 3


@pytest.mark.asyncio
async def test_extract_async_error_handling() -> None:
    """Test that async extraction handles per-page errors."""

    class FailingExtractor(MockExtractor):
        def extract_page(self, page: int) -> ExtractionResult:
            if page == 1:
                raise ValueError("Async simulated error")
            return super().extract_page(page)

    extractor = FailingExtractor(Path("/tmp/test.pdf"), page_count=3)
    results = await extractor.extract_pages_async(max_workers=2)
    assert len(results) == 3
    assert results[1].success is False
    assert "Async simulated error" in str(results[1].error)
