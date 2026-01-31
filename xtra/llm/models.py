"""Models for LLM-based extraction."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class LLMProvider(StrEnum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure-openai"


class LLMExtractionResult(BaseModel, Generic[T]):
    """Result of LLM extraction."""

    model_config = {"arbitrary_types_allowed": True}

    data: T
    model: str
    provider: LLMProvider
    usage: dict[str, int] | None = None
    raw_response: Any | None = None


class PageExtractionConfig(BaseModel):
    """Configuration for page selection."""

    page_numbers: list[int] | None = None  # None = all pages
    combine_pages: bool = True  # Combine all pages into single extraction


class LLMBatchResult(BaseModel, Generic[T]):
    """Result from a single batch in parallel extraction."""

    model_config = {"arbitrary_types_allowed": True}

    data: T | None
    pages: list[int]
    model: str
    provider: LLMProvider
    usage: dict[str, int] | None = None
    success: bool = True
    error: str | None = None


class LLMBatchExtractionResult(BaseModel, Generic[T]):
    """Result of parallel LLM extraction across multiple batches."""

    model_config = {"arbitrary_types_allowed": True}

    batch_results: list[LLMBatchResult[T]]
    model: str
    provider: LLMProvider

    @property
    def successful_batches(self) -> list[LLMBatchResult[T]]:
        """Return only successful batch results."""
        return [r for r in self.batch_results if r.success]

    @property
    def failed_batches(self) -> list[LLMBatchResult[T]]:
        """Return only failed batch results."""
        return [r for r in self.batch_results if not r.success]

    @property
    def all_data(self) -> list[T]:
        """Return data from all successful batches."""
        return [r.data for r in self.batch_results if r.success and r.data is not None]

    @property
    def total_usage(self) -> dict[str, int]:
        """Aggregate usage across all batches."""
        total: dict[str, int] = {}
        for result in self.batch_results:
            if result.usage:
                for key, value in result.usage.items():
                    total[key] = total.get(key, 0) + value
        return total
