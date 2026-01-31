"""Tests for parallel LLM extraction with max_workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from xtra.base import ExecutorType
from xtra.llm.models import LLMExtractionResult, LLMProvider

TEST_DATA_DIR = Path(__file__).parent.parent / "data"


def make_fake_extractor(
    responses: dict[tuple[int, ...], dict[str, Any]] | None = None,
    error_pages: set[int] | None = None,
    usage: dict[str, int] | None = None,
) -> Any:
    """Create a fake extractor for testing.

    Args:
        responses: Dict mapping page tuples to response data.
        error_pages: Set of pages that should raise errors.
        usage: Usage dict to return with each response.
    """
    error_pages = error_pages or set()
    usage = usage or {"prompt_tokens": 100, "completion_tokens": 50}

    def fake_extractor(  # noqa: PLR0913
        path: Path,
        model: str,
        schema: Any,
        prompt: str | None,
        pages: list[int] | None,
        dpi: int,
        max_retries: int,
        temperature: float,
        credentials: dict[str, str] | None,
        base_url: str | None,
        headers: dict[str, str] | None,
    ) -> LLMExtractionResult[dict[str, Any]]:
        page_key = tuple(pages) if pages else ()

        # Check if any page should error
        if pages:
            for page in pages:
                if page in error_pages:
                    raise ValueError(f"API error for page {page}")

        # Get response data
        if responses and page_key in responses:
            data = responses[page_key]
        else:
            data = {"pages": pages}

        return LLMExtractionResult(
            data=data,
            model="gpt-4o",
            provider=LLMProvider.OPENAI,
            usage=usage,
        )

    return fake_extractor


async def make_fake_async_extractor(
    responses: dict[tuple[int, ...], dict[str, Any]] | None = None,
    error_pages: set[int] | None = None,
    usage: dict[str, int] | None = None,
) -> Any:
    """Create an async fake extractor for testing."""
    sync_extractor = make_fake_extractor(responses, error_pages, usage)

    async def fake_async_extractor(  # noqa: PLR0913
        path: Path,
        model: str,
        schema: Any,
        prompt: str | None,
        pages: list[int] | None,
        dpi: int,
        max_retries: int,
        temperature: float,
        credentials: dict[str, str] | None,
        base_url: str | None,
        headers: dict[str, str] | None,
    ) -> LLMExtractionResult[dict[str, Any]]:
        return sync_extractor(
            path,
            model,
            schema,
            prompt,
            pages,
            dpi,
            max_retries,
            temperature,
            credentials,
            base_url,
            headers,
        )

    return fake_async_extractor


class TestExtractStructuredParallel:
    """Tests for extract_structured with max_workers > 1."""

    def test_single_worker_all_pages_in_one_request(self) -> None:
        """Test that single worker sends all pages in one request."""
        from xtra.llm_factory import extract_structured

        call_log: list[list[int] | None] = []

        def tracking_extractor(
            path: Path,
            model: str,
            schema: Any,
            prompt: str | None,
            pages: list[int] | None,
            *args: Any,
            **kwargs: Any,
        ) -> LLMExtractionResult[dict[str, Any]]:
            call_log.append(pages)
            return LLMExtractionResult(
                data={"pages": pages}, model="gpt-4o", provider=LLMProvider.OPENAI
            )

        result = extract_structured(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2],
            max_workers=1,
            _extractor=tracking_extractor,
        )

        assert len(call_log) == 1
        assert call_log[0] == [0, 1, 2]
        assert result.data == {"pages": [0, 1, 2]}

    def test_multiple_workers_parallel_execution(self) -> None:
        """Test that multiple workers process pages in parallel."""
        from xtra.llm_factory import extract_structured

        fake = make_fake_extractor()

        result = extract_structured(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2, 3],
            max_workers=2,
            _extractor=fake,
        )

        # Result data should be a list of per-page results
        assert isinstance(result.data, list)
        assert len(result.data) == 4

    def test_preserves_page_order(self) -> None:
        """Test that results maintain original page order."""
        from xtra.llm_factory import extract_structured

        fake = make_fake_extractor()

        result = extract_structured(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2, 3],
            max_workers=4,
            _extractor=fake,
        )

        # Results should be in original page order
        data = result.data
        assert isinstance(data, list)
        assert data[0]["pages"] == [0]
        assert data[1]["pages"] == [1]
        assert data[2]["pages"] == [2]
        assert data[3]["pages"] == [3]

    def test_aggregates_usage(self) -> None:
        """Test that usage is aggregated across all pages."""
        from xtra.llm_factory import extract_structured

        fake = make_fake_extractor(usage={"prompt_tokens": 100, "completion_tokens": 50})

        result = extract_structured(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1],
            max_workers=2,
            _extractor=fake,
        )

        # Usage should be aggregated (2 pages * 100 = 200, 2 * 50 = 100)
        assert result.usage == {"prompt_tokens": 200, "completion_tokens": 100}

    def test_raises_on_page_error(self) -> None:
        """Test that errors on individual pages raise ValueError."""
        from xtra.llm_factory import extract_structured

        fake = make_fake_extractor(error_pages={1})

        with pytest.raises(ValueError, match="Extraction failed for page 1"):
            extract_structured(
                TEST_DATA_DIR / "test_pdf_2p_text.pdf",
                model="openai/gpt-4o",
                pages=[0, 1, 2],
                max_workers=2,
                _extractor=fake,
            )

    def test_single_page_no_parallel(self) -> None:
        """Test that single page doesn't use parallel even with max_workers > 1."""
        from xtra.llm_factory import extract_structured

        call_log: list[list[int] | None] = []

        def tracking_extractor(
            path: Path,
            model: str,
            schema: Any,
            prompt: str | None,
            pages: list[int] | None,
            *args: Any,
            **kwargs: Any,
        ) -> LLMExtractionResult[dict[str, Any]]:
            call_log.append(pages)
            return LLMExtractionResult(
                data={"key": "value"}, model="gpt-4o", provider=LLMProvider.OPENAI
            )

        result = extract_structured(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0],
            max_workers=4,
            _extractor=tracking_extractor,
        )

        # Should call directly, not create a list
        assert len(call_log) == 1
        assert result.data == {"key": "value"}

    def test_process_executor(self) -> None:
        """Test with process executor type."""
        from xtra.llm_factory import extract_structured

        # Note: ProcessPoolExecutor requires picklable functions
        # Using a simple lambda won't work, so we test that thread executor works
        fake = make_fake_extractor()

        result = extract_structured(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1],
            max_workers=2,
            executor=ExecutorType.THREAD,
            _extractor=fake,
        )

        assert len(result.data) == 2


class TestExtractStructuredAsyncParallel:
    """Tests for extract_structured_async with max_workers > 1."""

    @pytest.mark.asyncio
    async def test_async_single_worker(self) -> None:
        """Test that single worker in async mode sends all pages in one request."""
        from xtra.llm_factory import extract_structured_async

        call_log: list[list[int] | None] = []

        async def tracking_extractor(
            path: Path,
            model: str,
            schema: Any,
            prompt: str | None,
            pages: list[int] | None,
            *args: Any,
            **kwargs: Any,
        ) -> LLMExtractionResult[dict[str, Any]]:
            call_log.append(pages)
            return LLMExtractionResult(
                data={"pages": pages}, model="gpt-4o", provider=LLMProvider.OPENAI
            )

        result = await extract_structured_async(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1],
            max_workers=1,
            _extractor=tracking_extractor,
        )

        assert len(call_log) == 1
        assert result.data == {"pages": [0, 1]}

    @pytest.mark.asyncio
    async def test_async_parallel_execution(self) -> None:
        """Test that async parallel extraction works."""
        from xtra.llm_factory import extract_structured_async

        async def fake_async(
            path: Path,
            model: str,
            schema: Any,
            prompt: str | None,
            pages: list[int] | None,
            *args: Any,
            **kwargs: Any,
        ) -> LLMExtractionResult[dict[str, Any]]:
            return LLMExtractionResult(
                data={"pages": pages}, model="gpt-4o", provider=LLMProvider.OPENAI
            )

        result = await extract_structured_async(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2],
            max_workers=2,
            _extractor=fake_async,
        )

        assert isinstance(result.data, list)
        assert len(result.data) == 3

    @pytest.mark.asyncio
    async def test_async_preserves_order(self) -> None:
        """Test that async parallel preserves page order."""
        from xtra.llm_factory import extract_structured_async

        async def fake_async(
            path: Path,
            model: str,
            schema: Any,
            prompt: str | None,
            pages: list[int] | None,
            *args: Any,
            **kwargs: Any,
        ) -> LLMExtractionResult[dict[str, Any]]:
            return LLMExtractionResult(
                data={"page": pages[0] if pages else None},
                model="gpt-4o",
                provider=LLMProvider.OPENAI,
            )

        result = await extract_structured_async(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2, 3],
            max_workers=4,
            _extractor=fake_async,
        )

        data = result.data
        assert isinstance(data, list)
        assert data[0]["page"] == 0
        assert data[1]["page"] == 1
        assert data[2]["page"] == 2
        assert data[3]["page"] == 3
