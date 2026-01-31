"""Factory functions for LLM-based extraction."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeVar, overload

from pydantic import BaseModel

from xtra.base import ExecutorType
from xtra.llm.models import (
    LLMBatchExtractionResult,
    LLMBatchResult,
    LLMExtractionResult,
    LLMProvider,
)

T = TypeVar("T", bound=BaseModel)


# Model to provider mapping for inference
MODEL_PROVIDER_MAP = {
    "gpt-": LLMProvider.OPENAI,
    "claude-": LLMProvider.ANTHROPIC,
    "gemini-": LLMProvider.GOOGLE,
}


def _parse_model_string(model: str) -> tuple[LLMProvider, str]:
    """Parse model string into (provider, model_name).

    Supports:
    - "openai/gpt-4o" -> (OPENAI, "gpt-4o")
    - "gpt-4o" -> (OPENAI, "gpt-4o") - inferred from prefix
    """
    if "/" in model:
        provider_str, model_name = model.split("/", 1)
        provider = LLMProvider(provider_str.lower())
        return provider, model_name

    # Infer provider from model name prefix
    for prefix, provider in MODEL_PROVIDER_MAP.items():
        if model.startswith(prefix):
            return provider, model

    raise ValueError(
        f"Cannot infer provider for model '{model}'. "
        f"Use format 'provider/model' (e.g., 'openai/gpt-4o')"
    )


def _get_credential(key: str, credentials: dict[str, str] | None) -> str | None:
    """Get credential from dict or environment variable."""
    if credentials and key in credentials:
        return credentials[key]
    return os.environ.get(key)


@overload
def extract_structured(
    path: Path | str,
    model: str,
    *,
    schema: type[T],
    prompt: str | None = None,
    pages: list[int] | None = None,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    credentials: dict[str, str] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
) -> LLMExtractionResult[T]: ...


@overload
def extract_structured(
    path: Path | str,
    model: str,
    *,
    schema: None = None,
    prompt: str | None = None,
    pages: list[int] | None = None,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    credentials: dict[str, str] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
) -> LLMExtractionResult[dict[str, Any]]: ...


def extract_structured(  # noqa: PLR0913
    path: Path | str,
    model: str,
    *,
    schema: type[T] | None = None,
    prompt: str | None = None,
    pages: list[int] | None = None,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    credentials: dict[str, str] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
) -> LLMExtractionResult[T | dict[str, Any]]:
    """Extract structured data from a document using an LLM.

    Args:
        path: Path to document/image file.
        model: Model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3-5-sonnet").
        schema: Pydantic model for structured output. None for free-form dict.
        prompt: Custom extraction prompt. Auto-generated from schema if None.
        pages: Page numbers to extract from (0-indexed). None for all pages.
        dpi: DPI for PDF-to-image conversion.
        max_retries: Max retry attempts with validation feedback.
        temperature: Sampling temperature (0.0 = deterministic).
        credentials: Override credentials dict (otherwise uses env vars).
        base_url: Custom API base URL for OpenAI-compatible APIs (vLLM, Ollama, etc.).
        headers: Custom HTTP headers for OpenAI-compatible APIs.

    Returns:
        LLMExtractionResult containing extracted data, model info, and provider.
    """
    provider, model_name = _parse_model_string(model)

    if provider == LLMProvider.OPENAI:
        from xtra.llm.extractors.openai import extract_openai

        api_key = _get_credential("OPENAI_API_KEY", credentials)
        return extract_openai(
            path=path,
            model=model_name,
            schema=schema,
            prompt=prompt,
            pages=pages,
            dpi=dpi,
            max_retries=max_retries,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            headers=headers,
        )

    elif provider == LLMProvider.ANTHROPIC:
        from xtra.llm.extractors.anthropic import extract_anthropic

        api_key = _get_credential("ANTHROPIC_API_KEY", credentials)
        return extract_anthropic(
            path=path,
            model=model_name,
            schema=schema,
            prompt=prompt,
            pages=pages,
            dpi=dpi,
            max_retries=max_retries,
            temperature=temperature,
            api_key=api_key,
        )

    elif provider == LLMProvider.GOOGLE:
        from xtra.llm.extractors.google import extract_google

        api_key = _get_credential("GOOGLE_API_KEY", credentials)
        return extract_google(
            path=path,
            model=model_name,
            schema=schema,
            prompt=prompt,
            pages=pages,
            dpi=dpi,
            max_retries=max_retries,
            temperature=temperature,
            api_key=api_key,
        )

    elif provider == LLMProvider.AZURE_OPENAI:
        from xtra.llm.extractors.azure_openai import extract_azure_openai

        # Try AZURE_OPENAI_* first, fall back to XTRA_AZURE_DI_* for consistency
        api_key = _get_credential("AZURE_OPENAI_API_KEY", credentials) or _get_credential(
            "XTRA_AZURE_DI_KEY", credentials
        )
        endpoint = _get_credential("AZURE_OPENAI_ENDPOINT", credentials) or _get_credential(
            "XTRA_AZURE_DI_ENDPOINT", credentials
        )
        api_version = _get_credential("AZURE_OPENAI_API_VERSION", credentials)
        api_version = api_version or "2024-02-15-preview"
        if not endpoint:
            raise ValueError(
                "Azure OpenAI endpoint required (AZURE_OPENAI_ENDPOINT or XTRA_AZURE_DI_ENDPOINT)"
            )
        return extract_azure_openai(
            path=path,
            model=model_name,
            schema=schema,
            prompt=prompt,
            pages=pages,
            dpi=dpi,
            max_retries=max_retries,
            temperature=temperature,
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


@overload
async def extract_structured_async(
    path: Path | str,
    model: str,
    *,
    schema: type[T],
    prompt: str | None = None,
    pages: list[int] | None = None,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    credentials: dict[str, str] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
) -> LLMExtractionResult[T]: ...


@overload
async def extract_structured_async(
    path: Path | str,
    model: str,
    *,
    schema: None = None,
    prompt: str | None = None,
    pages: list[int] | None = None,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    credentials: dict[str, str] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
) -> LLMExtractionResult[dict[str, Any]]: ...


async def extract_structured_async(  # noqa: PLR0913
    path: Path | str,
    model: str,
    *,
    schema: type[T] | None = None,
    prompt: str | None = None,
    pages: list[int] | None = None,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    credentials: dict[str, str] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
) -> LLMExtractionResult[T | dict[str, Any]]:
    """Async version of extract_structured."""
    provider, model_name = _parse_model_string(model)

    if provider == LLMProvider.OPENAI:
        from xtra.llm.extractors.openai import extract_openai_async

        api_key = _get_credential("OPENAI_API_KEY", credentials)
        return await extract_openai_async(
            path=path,
            model=model_name,
            schema=schema,
            prompt=prompt,
            pages=pages,
            dpi=dpi,
            max_retries=max_retries,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            headers=headers,
        )

    elif provider == LLMProvider.ANTHROPIC:
        from xtra.llm.extractors.anthropic import extract_anthropic_async

        api_key = _get_credential("ANTHROPIC_API_KEY", credentials)
        return await extract_anthropic_async(
            path=path,
            model=model_name,
            schema=schema,
            prompt=prompt,
            pages=pages,
            dpi=dpi,
            max_retries=max_retries,
            temperature=temperature,
            api_key=api_key,
        )

    elif provider == LLMProvider.GOOGLE:
        from xtra.llm.extractors.google import extract_google_async

        api_key = _get_credential("GOOGLE_API_KEY", credentials)
        return await extract_google_async(
            path=path,
            model=model_name,
            schema=schema,
            prompt=prompt,
            pages=pages,
            dpi=dpi,
            max_retries=max_retries,
            temperature=temperature,
            api_key=api_key,
        )

    elif provider == LLMProvider.AZURE_OPENAI:
        from xtra.llm.extractors.azure_openai import extract_azure_openai_async

        # Try AZURE_OPENAI_* first, fall back to XTRA_AZURE_DI_* for consistency
        api_key = _get_credential("AZURE_OPENAI_API_KEY", credentials) or _get_credential(
            "XTRA_AZURE_DI_KEY", credentials
        )
        endpoint = _get_credential("AZURE_OPENAI_ENDPOINT", credentials) or _get_credential(
            "XTRA_AZURE_DI_ENDPOINT", credentials
        )
        api_version = _get_credential("AZURE_OPENAI_API_VERSION", credentials)
        api_version = api_version or "2024-02-15-preview"
        if not endpoint:
            raise ValueError(
                "Azure OpenAI endpoint required (AZURE_OPENAI_ENDPOINT or XTRA_AZURE_DI_ENDPOINT)"
            )
        return await extract_azure_openai_async(
            path=path,
            model=model_name,
            schema=schema,
            prompt=prompt,
            pages=pages,
            dpi=dpi,
            max_retries=max_retries,
            temperature=temperature,
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


def _batch_pages(pages: list[int], pages_per_batch: int) -> list[list[int]]:
    """Split pages into batches.

    Args:
        pages: List of page numbers to split.
        pages_per_batch: Number of pages per batch.

    Returns:
        List of page batches.
    """
    if not pages:
        return []
    return [pages[i : i + pages_per_batch] for i in range(0, len(pages), pages_per_batch)]


def _extract_batch(  # noqa: PLR0913
    path: Path,
    model: str,
    batch_pages: list[int],
    provider: LLMProvider,
    model_name: str,
    schema: type[T] | None,
    prompt: str | None,
    dpi: int,
    max_retries: int,
    temperature: float,
    credentials: dict[str, str] | None,
    base_url: str | None,
    headers: dict[str, str] | None,
) -> LLMBatchResult[T | dict[str, Any]]:
    """Extract a single batch of pages.

    Internal helper that wraps extract_structured and returns a batch result.
    """
    try:
        result = extract_structured(
            path=path,
            model=model,
            schema=schema,
            prompt=prompt,
            pages=batch_pages,
            dpi=dpi,
            max_retries=max_retries,
            temperature=temperature,
            credentials=credentials,
            base_url=base_url,
            headers=headers,
        )
        return LLMBatchResult(
            data=result.data,
            pages=batch_pages,
            model=model_name,
            provider=provider,
            usage=result.usage,
            success=True,
        )
    except Exception as e:
        return LLMBatchResult(
            data=None,
            pages=batch_pages,
            model=model_name,
            provider=provider,
            success=False,
            error=str(e),
        )


def extract_structured_parallel(  # noqa: PLR0913
    path: Path | str,
    model: str,
    *,
    schema: type[T] | None = None,
    prompt: str | None = None,
    pages: list[int] | None = None,
    pages_per_batch: int = 1,
    max_workers: int = 1,
    executor: ExecutorType = ExecutorType.THREAD,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    credentials: dict[str, str] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
) -> LLMBatchExtractionResult[T | dict[str, Any]]:
    """Extract structured data with parallel processing across page batches.

    Args:
        path: Path to document/image file.
        model: Model identifier (e.g., "openai/gpt-4o").
        schema: Pydantic model for structured output. None for free-form dict.
        prompt: Custom extraction prompt.
        pages: Page numbers to extract (0-indexed). None for all pages.
        pages_per_batch: Number of pages per LLM request.
        max_workers: Number of parallel workers. 1 means sequential.
        executor: Type of executor (THREAD or PROCESS).
        dpi: DPI for PDF-to-image conversion.
        max_retries: Max retry attempts with validation feedback.
        temperature: Sampling temperature.
        credentials: Override credentials dict.
        base_url: Custom API base URL for OpenAI-compatible APIs.
        headers: Custom HTTP headers.

    Returns:
        LLMBatchExtractionResult containing results from each batch.
    """
    path = Path(path) if isinstance(path, str) else path
    provider, model_name = _parse_model_string(model)

    # Get page count if pages not specified
    if pages is None:
        from xtra.base import ImageLoader

        loader = ImageLoader(path, dpi=dpi)
        pages = list(range(loader.page_count))
        loader.close()

    # Create batches
    batches = _batch_pages(pages, pages_per_batch)

    # Sequential execution for single batch or single worker
    if max_workers <= 1 or len(batches) <= 1:
        results: list[LLMBatchResult[T | dict[str, Any]]] = []
        for batch in batches:
            result = _extract_batch(
                path=path,
                model=model,
                batch_pages=batch,
                provider=provider,
                model_name=model_name,
                schema=schema,
                prompt=prompt,
                dpi=dpi,
                max_retries=max_retries,
                temperature=temperature,
                credentials=credentials,
                base_url=base_url,
                headers=headers,
            )
            results.append(result)
        return LLMBatchExtractionResult(
            batch_results=results,
            model=model_name,
            provider=provider,
        )

    # Parallel execution
    executor_class = ProcessPoolExecutor if executor == ExecutorType.PROCESS else ThreadPoolExecutor

    parallel_results: list[LLMBatchResult[T | dict[str, Any]] | None] = [None] * len(batches)
    with executor_class(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(
                _extract_batch,
                path,
                model,
                batch,
                provider,
                model_name,
                schema,
                prompt,
                dpi,
                max_retries,
                temperature,
                credentials,
                base_url,
                headers,
            ): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                parallel_results[idx] = future.result()
            except Exception as e:
                parallel_results[idx] = LLMBatchResult(
                    data=None,
                    pages=batches[idx],
                    model=model_name,
                    provider=provider,
                    success=False,
                    error=str(e),
                )

    return LLMBatchExtractionResult(
        batch_results=parallel_results,  # type: ignore[arg-type]
        model=model_name,
        provider=provider,
    )
