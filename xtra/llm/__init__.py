"""LLM-based structured data extraction."""

from xtra.llm.models import (
    LLMBatchExtractionResult,
    LLMBatchResult,
    LLMExtractionResult,
    LLMProvider,
    PageExtractionConfig,
)

__all__ = [
    "LLMBatchExtractionResult",
    "LLMBatchResult",
    "LLMExtractionResult",
    "LLMProvider",
    "PageExtractionConfig",
    "extract_structured",
    "extract_structured_async",
    "extract_structured_parallel",
]


def __getattr__(name: str):
    """Lazy load factory functions to avoid circular imports."""
    if name == "extract_structured":
        from xtra.llm_factory import extract_structured

        return extract_structured
    if name == "extract_structured_async":
        from xtra.llm_factory import extract_structured_async

        return extract_structured_async
    if name == "extract_structured_parallel":
        from xtra.llm_factory import extract_structured_parallel

        return extract_structured_parallel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
