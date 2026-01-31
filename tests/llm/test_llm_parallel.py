"""Tests for parallel LLM extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from xtra.base import ExecutorType

TEST_DATA_DIR = Path(__file__).parent.parent / "data"


class TestBatchPages:
    """Tests for _batch_pages helper function."""

    def test_single_page_batches(self) -> None:
        """Each page in its own batch."""
        from xtra.llm_factory import _batch_pages

        pages = [0, 1, 2, 3]
        batches = _batch_pages(pages, 1)
        assert batches == [[0], [1], [2], [3]]

    def test_multi_page_batches(self) -> None:
        """Multiple pages per batch."""
        from xtra.llm_factory import _batch_pages

        pages = [0, 1, 2, 3, 4]
        batches = _batch_pages(pages, 2)
        assert batches == [[0, 1], [2, 3], [4]]

    def test_all_in_one_batch(self) -> None:
        """All pages fit in one batch."""
        from xtra.llm_factory import _batch_pages

        pages = [0, 1, 2]
        batches = _batch_pages(pages, 10)
        assert batches == [[0, 1, 2]]

    def test_exact_fit(self) -> None:
        """Pages evenly divide into batches."""
        from xtra.llm_factory import _batch_pages

        pages = [0, 1, 2, 3]
        batches = _batch_pages(pages, 2)
        assert batches == [[0, 1], [2, 3]]

    def test_empty_pages(self) -> None:
        """Empty page list returns empty batches."""
        from xtra.llm_factory import _batch_pages

        batches = _batch_pages([], 2)
        assert batches == []


class TestLLMBatchResult:
    """Tests for LLMBatchResult model."""

    def test_successful_result(self) -> None:
        """Test successful batch result."""
        from xtra.llm.models import LLMBatchResult, LLMProvider

        result = LLMBatchResult(
            data={"key": "value"},
            pages=[0, 1],
            model="gpt-4o",
            provider=LLMProvider.OPENAI,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            success=True,
        )
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.pages == [0, 1]
        assert result.error is None

    def test_failed_result(self) -> None:
        """Test failed batch result."""
        from xtra.llm.models import LLMBatchResult, LLMProvider

        result = LLMBatchResult(
            data=None,
            pages=[2],
            model="gpt-4o",
            provider=LLMProvider.OPENAI,
            success=False,
            error="API rate limit exceeded",
        )
        assert result.success is False
        assert result.data is None
        assert result.error == "API rate limit exceeded"


class TestLLMBatchExtractionResult:
    """Tests for LLMBatchExtractionResult model."""

    def test_successful_batches_property(self) -> None:
        """Test filtering successful batches."""
        from xtra.llm.models import LLMBatchExtractionResult, LLMBatchResult, LLMProvider

        results = [
            LLMBatchResult(
                data={"a": 1}, pages=[0], model="m", provider=LLMProvider.OPENAI, success=True
            ),
            LLMBatchResult(
                data=None,
                pages=[1],
                model="m",
                provider=LLMProvider.OPENAI,
                success=False,
                error="err",
            ),
            LLMBatchResult(
                data={"b": 2}, pages=[2], model="m", provider=LLMProvider.OPENAI, success=True
            ),
        ]
        batch_result = LLMBatchExtractionResult(
            batch_results=results, model="m", provider=LLMProvider.OPENAI
        )

        assert len(batch_result.successful_batches) == 2
        assert batch_result.successful_batches[0].pages == [0]
        assert batch_result.successful_batches[1].pages == [2]

    def test_failed_batches_property(self) -> None:
        """Test filtering failed batches."""
        from xtra.llm.models import LLMBatchExtractionResult, LLMBatchResult, LLMProvider

        results = [
            LLMBatchResult(
                data={"a": 1}, pages=[0], model="m", provider=LLMProvider.OPENAI, success=True
            ),
            LLMBatchResult(
                data=None,
                pages=[1],
                model="m",
                provider=LLMProvider.OPENAI,
                success=False,
                error="err",
            ),
        ]
        batch_result = LLMBatchExtractionResult(
            batch_results=results, model="m", provider=LLMProvider.OPENAI
        )

        assert len(batch_result.failed_batches) == 1
        assert batch_result.failed_batches[0].error == "err"

    def test_all_data_property(self) -> None:
        """Test extracting data from successful batches."""
        from xtra.llm.models import LLMBatchExtractionResult, LLMBatchResult, LLMProvider

        results = [
            LLMBatchResult(
                data={"a": 1}, pages=[0], model="m", provider=LLMProvider.OPENAI, success=True
            ),
            LLMBatchResult(
                data=None,
                pages=[1],
                model="m",
                provider=LLMProvider.OPENAI,
                success=False,
                error="err",
            ),
            LLMBatchResult(
                data={"b": 2}, pages=[2], model="m", provider=LLMProvider.OPENAI, success=True
            ),
        ]
        batch_result = LLMBatchExtractionResult(
            batch_results=results, model="m", provider=LLMProvider.OPENAI
        )

        assert batch_result.all_data == [{"a": 1}, {"b": 2}]

    def test_total_usage_aggregation(self) -> None:
        """Test aggregating usage across batches."""
        from xtra.llm.models import LLMBatchExtractionResult, LLMBatchResult, LLMProvider

        results = [
            LLMBatchResult(
                data={},
                pages=[0],
                model="m",
                provider=LLMProvider.OPENAI,
                usage={"prompt_tokens": 100, "completion_tokens": 50},
            ),
            LLMBatchResult(
                data={},
                pages=[1],
                model="m",
                provider=LLMProvider.OPENAI,
                usage={"prompt_tokens": 150, "completion_tokens": 75},
            ),
        ]
        batch_result = LLMBatchExtractionResult(
            batch_results=results, model="m", provider=LLMProvider.OPENAI
        )

        assert batch_result.total_usage == {"prompt_tokens": 250, "completion_tokens": 125}

    def test_total_usage_with_missing(self) -> None:
        """Test usage aggregation when some batches have no usage."""
        from xtra.llm.models import LLMBatchExtractionResult, LLMBatchResult, LLMProvider

        results = [
            LLMBatchResult(
                data={},
                pages=[0],
                model="m",
                provider=LLMProvider.OPENAI,
                usage={"prompt_tokens": 100},
            ),
            LLMBatchResult(data={}, pages=[1], model="m", provider=LLMProvider.OPENAI, usage=None),
        ]
        batch_result = LLMBatchExtractionResult(
            batch_results=results, model="m", provider=LLMProvider.OPENAI
        )

        assert batch_result.total_usage == {"prompt_tokens": 100}


class TestExtractStructuredParallel:
    """Tests for extract_structured_parallel function."""

    @patch("xtra.llm_factory.extract_structured")
    @patch("xtra.base.ImageLoader")
    def test_sequential_with_single_worker(
        self, mock_loader_class: MagicMock, mock_extract: MagicMock
    ) -> None:
        """Test that single worker runs sequentially."""
        from xtra.llm.models import LLMExtractionResult, LLMProvider
        from xtra.llm_factory import extract_structured_parallel

        # Mock ImageLoader for page count
        mock_loader = MagicMock()
        mock_loader.page_count = 4
        mock_loader_class.return_value = mock_loader

        mock_extract.return_value = LLMExtractionResult(
            data={"key": "value"}, model="gpt-4o", provider=LLMProvider.OPENAI
        )

        result = extract_structured_parallel(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1],
            pages_per_batch=1,
            max_workers=1,
        )

        assert len(result.batch_results) == 2
        assert mock_extract.call_count == 2
        assert all(r.success for r in result.batch_results)

    @patch("xtra.llm_factory.extract_structured")
    @patch("xtra.base.ImageLoader")
    def test_parallel_with_multiple_workers(
        self, mock_loader_class: MagicMock, mock_extract: MagicMock
    ) -> None:
        """Test that multiple workers runs in parallel."""
        from xtra.llm.models import LLMExtractionResult, LLMProvider
        from xtra.llm_factory import extract_structured_parallel

        mock_loader = MagicMock()
        mock_loader.page_count = 4
        mock_loader_class.return_value = mock_loader

        mock_extract.return_value = LLMExtractionResult(
            data={"key": "value"}, model="gpt-4o", provider=LLMProvider.OPENAI
        )

        result = extract_structured_parallel(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2, 3],
            pages_per_batch=1,
            max_workers=2,
        )

        assert len(result.batch_results) == 4
        assert all(r.success for r in result.batch_results)
        assert mock_extract.call_count == 4

    @patch("xtra.llm_factory.extract_structured")
    @patch("xtra.base.ImageLoader")
    def test_handles_batch_errors_gracefully(
        self, mock_loader_class: MagicMock, mock_extract: MagicMock
    ) -> None:
        """Test that errors in one batch don't affect others."""
        from xtra.llm.models import LLMExtractionResult, LLMProvider
        from xtra.llm_factory import extract_structured_parallel

        mock_loader = MagicMock()
        mock_loader.page_count = 3
        mock_loader_class.return_value = mock_loader

        def side_effect(*args: Any, **kwargs: Any) -> LLMExtractionResult[dict[str, Any]]:
            if kwargs.get("pages") == [1]:
                raise ValueError("API error")
            return LLMExtractionResult(data={}, model="gpt-4o", provider=LLMProvider.OPENAI)

        mock_extract.side_effect = side_effect

        result = extract_structured_parallel(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2],
            pages_per_batch=1,
            max_workers=2,
        )

        assert len(result.successful_batches) == 2
        assert len(result.failed_batches) == 1
        assert "API error" in result.failed_batches[0].error  # type: ignore[operator]

    @patch("xtra.llm_factory.extract_structured")
    @patch("xtra.base.ImageLoader")
    def test_preserves_batch_order(
        self, mock_loader_class: MagicMock, mock_extract: MagicMock
    ) -> None:
        """Test that batch results maintain original page order."""
        from xtra.llm.models import LLMExtractionResult, LLMProvider
        from xtra.llm_factory import extract_structured_parallel

        mock_loader = MagicMock()
        mock_loader.page_count = 4
        mock_loader_class.return_value = mock_loader

        def side_effect(*args: Any, **kwargs: Any) -> LLMExtractionResult[dict[str, Any]]:
            pages = kwargs.get("pages", [])
            return LLMExtractionResult(
                data={"pages": pages}, model="gpt-4o", provider=LLMProvider.OPENAI
            )

        mock_extract.side_effect = side_effect

        result = extract_structured_parallel(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2, 3],
            pages_per_batch=1,
            max_workers=4,
        )

        # Results should be in original order
        assert result.batch_results[0].pages == [0]
        assert result.batch_results[1].pages == [1]
        assert result.batch_results[2].pages == [2]
        assert result.batch_results[3].pages == [3]

    @patch("xtra.llm_factory.extract_structured")
    @patch("xtra.base.ImageLoader")
    def test_multi_page_batches(
        self, mock_loader_class: MagicMock, mock_extract: MagicMock
    ) -> None:
        """Test extraction with multiple pages per batch."""
        from xtra.llm.models import LLMExtractionResult, LLMProvider
        from xtra.llm_factory import extract_structured_parallel

        mock_loader = MagicMock()
        mock_loader.page_count = 4
        mock_loader_class.return_value = mock_loader

        mock_extract.return_value = LLMExtractionResult(
            data={"key": "value"}, model="gpt-4o", provider=LLMProvider.OPENAI
        )

        result = extract_structured_parallel(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2, 3],
            pages_per_batch=2,
            max_workers=2,
        )

        assert len(result.batch_results) == 2
        assert result.batch_results[0].pages == [0, 1]
        assert result.batch_results[1].pages == [2, 3]

    @patch("xtra.llm_factory.extract_structured")
    @patch("xtra.base.ImageLoader")
    def test_all_pages_when_none_specified(
        self, mock_loader_class: MagicMock, mock_extract: MagicMock
    ) -> None:
        """Test that all pages are extracted when pages=None."""
        from xtra.llm.models import LLMExtractionResult, LLMProvider
        from xtra.llm_factory import extract_structured_parallel

        mock_loader = MagicMock()
        mock_loader.page_count = 3
        mock_loader_class.return_value = mock_loader

        mock_extract.return_value = LLMExtractionResult(
            data={"key": "value"}, model="gpt-4o", provider=LLMProvider.OPENAI
        )

        result = extract_structured_parallel(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=None,
            pages_per_batch=1,
            max_workers=2,
        )

        assert len(result.batch_results) == 3
        # Verify all pages 0, 1, 2 were processed
        all_pages = [r.pages[0] for r in result.batch_results]
        assert sorted(all_pages) == [0, 1, 2]

    @patch("xtra.llm_factory.extract_structured")
    @patch("xtra.base.ImageLoader")
    def test_process_executor(self, mock_loader_class: MagicMock, mock_extract: MagicMock) -> None:
        """Test with process executor type."""
        from xtra.llm.models import LLMExtractionResult, LLMProvider
        from xtra.llm_factory import extract_structured_parallel

        mock_loader = MagicMock()
        mock_loader.page_count = 2
        mock_loader_class.return_value = mock_loader

        mock_extract.return_value = LLMExtractionResult(
            data={"key": "value"}, model="gpt-4o", provider=LLMProvider.OPENAI
        )

        # Process executor may not work with mocks, but we test the path selection
        # For this test, use single worker to avoid pickling issues
        result = extract_structured_parallel(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1],
            pages_per_batch=1,
            max_workers=1,
            executor=ExecutorType.PROCESS,
        )

        assert len(result.batch_results) == 2
