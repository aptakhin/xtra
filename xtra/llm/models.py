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
