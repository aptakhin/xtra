from pathlib import Path

import pytest

from xtra.extractors.base import (
    BaseExtractor,
    ExecutorType,
    ExtractionResult,
    PageExtractionResult,
)
from xtra.models import Document, ExtractorMetadata, ExtractorType, Page


class MockExtractor(BaseExtractor):
    """Mock extractor for testing base class."""

    def __init__(self, path: Path, page_count: int = 2) -> None:
        super().__init__(path)
        self._page_count = page_count

    def get_page_count(self) -> int:
        return self._page_count

    def extract_page(self, page: int) -> PageExtractionResult:
        if page >= self._page_count:
            return PageExtractionResult(
                page=Page(page=page, width=0, height=0, texts=[]),
                success=False,
                error="Page out of range",
            )
        return PageExtractionResult(
            page=Page(page=page, width=100.0, height=100.0, texts=[]),
            success=True,
        )

    def get_extractor_metadata(self) -> ExtractorMetadata:
        return ExtractorMetadata(extractor_type=ExtractorType.PDF)


# PageExtractionResult tests


def test_page_extraction_result_success() -> None:
    page = Page(page=0, width=100.0, height=100.0)
    result = PageExtractionResult(page=page, success=True)
    assert result.success
    assert result.error is None


def test_page_extraction_result_failure() -> None:
    page = Page(page=0, width=0, height=0)
    result = PageExtractionResult(page=page, success=False, error="Test error")
    assert not result.success
    assert result.error == "Test error"


# ExtractionResult tests


def test_extraction_result_success_property() -> None:
    """Test ExtractionResult.success returns True when all pages succeed."""
    doc = Document(path=Path("/tmp/test.pdf"), pages=[])
    page_results = [
        PageExtractionResult(page=Page(page=0, width=100, height=100, texts=[]), success=True),
        PageExtractionResult(page=Page(page=1, width=100, height=100, texts=[]), success=True),
    ]
    result = ExtractionResult(document=doc, page_results=page_results)
    assert result.success is True
    assert len(result.failed_pages) == 0
    assert len(result.errors) == 0


def test_extraction_result_failure_property() -> None:
    """Test ExtractionResult.success returns False when any page fails."""
    doc = Document(path=Path("/tmp/test.pdf"), pages=[])
    page_results = [
        PageExtractionResult(page=Page(page=0, width=100, height=100, texts=[]), success=True),
        PageExtractionResult(
            page=Page(page=1, width=0, height=0, texts=[]), success=False, error="Failed"
        ),
    ]
    result = ExtractionResult(document=doc, page_results=page_results)
    assert result.success is False
    assert len(result.failed_pages) == 1
    assert result.errors == [(1, "Failed")]


# BaseExtractor tests


def test_base_extractor_extract_all_pages() -> None:
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=3)
    result = extractor.extract()
    assert len(result.document.pages) == 3
    assert result.document.metadata is not None
    assert result.success is True


def test_base_extractor_extract_specific_pages() -> None:
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=5)
    result = extractor.extract(pages=[0, 2, 4])
    assert len(result.document.pages) == 3
    assert result.document.pages[0].page == 0
    assert result.document.pages[1].page == 2
    assert result.document.pages[2].page == 4


def test_base_extractor_page_results() -> None:
    """Test that extract() returns page_results with all pages."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=2)
    result = extractor.extract()
    assert len(result.page_results) == 2
    assert all(r.success for r in result.page_results)


def test_base_extractor_context_manager() -> None:
    with MockExtractor(Path("/tmp/test.pdf")) as extractor:
        result = extractor.extract()
    assert len(result.document.pages) == 2


def test_base_extractor_close() -> None:
    extractor = MockExtractor(Path("/tmp/test.pdf"))
    extractor.close()  # Should not raise


def test_base_extractor_path_attribute() -> None:
    path = Path("/tmp/test.pdf")
    extractor = MockExtractor(path)
    assert extractor.path == path


# Parallel extraction tests


def test_extract_parallel_thread_ordering() -> None:
    """Test that parallel extraction with threads maintains page order."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=10)
    result = extractor.extract(pages=[9, 0, 5, 3], max_workers=4, executor=ExecutorType.THREAD)
    assert len(result.page_results) == 4
    assert result.page_results[0].page.page == 9
    assert result.page_results[1].page.page == 0
    assert result.page_results[2].page.page == 5
    assert result.page_results[3].page.page == 3


def test_extract_parallel_process_ordering() -> None:
    """Test that parallel extraction with processes maintains page order."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=10)
    result = extractor.extract(pages=[9, 0, 5, 3], max_workers=4, executor=ExecutorType.PROCESS)
    assert len(result.page_results) == 4
    assert result.page_results[0].page.page == 9
    assert result.page_results[1].page.page == 0
    assert result.page_results[2].page.page == 5
    assert result.page_results[3].page.page == 3


def test_extract_parallel_all() -> None:
    """Test parallel extraction of all pages."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=5)
    result = extractor.extract(max_workers=3)
    assert len(result.page_results) == 5
    assert result.success is True


def test_extract_parallel_single_page() -> None:
    """Test that single page extraction works with max_workers > 1."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=1)
    result = extractor.extract(max_workers=4)
    assert len(result.page_results) == 1


def test_extract_parallel_error_handling() -> None:
    """Test that parallel extraction handles per-page errors."""

    class FailingExtractor(MockExtractor):
        def extract_page(self, page: int) -> PageExtractionResult:
            if page == 2:
                raise ValueError("Simulated error")
            return super().extract_page(page)

    extractor = FailingExtractor(Path("/tmp/test.pdf"), page_count=5)
    result = extractor.extract(max_workers=3)
    assert len(result.page_results) == 5
    assert result.page_results[2].success is False
    assert "Simulated error" in str(result.page_results[2].error)
    assert result.success is False
    assert len(result.failed_pages) == 1


def test_extract_backward_compatible() -> None:
    """Test that default behavior (max_workers=1) matches sequential."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=3)
    result_default = extractor.extract()
    result_explicit = extractor.extract(max_workers=1)
    assert len(result_default.page_results) == len(result_explicit.page_results)
    for r1, r2 in zip(result_default.page_results, result_explicit.page_results):
        assert r1.page.page == r2.page.page


def test_extract_with_parallel_workers() -> None:
    """Test extract() method with max_workers parameter."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=5)
    result = extractor.extract(max_workers=2)
    assert len(result.document.pages) == 5


@pytest.mark.parametrize("executor", [ExecutorType.THREAD, ExecutorType.PROCESS])
def test_extract_with_executor_types(executor: ExecutorType) -> None:
    """Test extract() method with different executor types."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=3)
    result = extractor.extract(max_workers=2, executor=executor)
    assert len(result.document.pages) == 3


# Async extraction tests


@pytest.mark.asyncio
async def test_extract_async() -> None:
    """Test async extract() method."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=3)
    result = await extractor.extract_async()
    assert len(result.page_results) == 3
    assert result.success is True


@pytest.mark.asyncio
async def test_extract_async_parallel() -> None:
    """Test async parallel extraction maintains order."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=5)
    result = await extractor.extract_async(pages=[4, 2, 0], max_workers=3)
    assert len(result.page_results) == 3
    assert result.page_results[0].page.page == 4
    assert result.page_results[1].page.page == 2
    assert result.page_results[2].page.page == 0


@pytest.mark.asyncio
async def test_extract_async_with_workers() -> None:
    """Test async extract() method with workers."""
    extractor = MockExtractor(Path("/tmp/test.pdf"), page_count=3)
    result = await extractor.extract_async(max_workers=2)
    assert len(result.document.pages) == 3


@pytest.mark.asyncio
async def test_extract_async_error_handling() -> None:
    """Test that async extraction handles per-page errors."""

    class FailingExtractor(MockExtractor):
        def extract_page(self, page: int) -> PageExtractionResult:
            if page == 1:
                raise ValueError("Async simulated error")
            return super().extract_page(page)

    extractor = FailingExtractor(Path("/tmp/test.pdf"), page_count=3)
    result = await extractor.extract_async(max_workers=2)
    assert len(result.page_results) == 3
    assert result.page_results[1].success is False
    assert "Async simulated error" in str(result.page_results[1].error)
    assert result.success is False
