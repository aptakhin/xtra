"""Factory functions for LLM-based extraction."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeVar, cast, overload

from pydantic import BaseModel

from xtra.base import ExecutorType
from xtra.llm.models import (
    LLMExtractionResult,
    LLMProvider,
)

T = TypeVar("T", bound=BaseModel)

# Type aliases for extractor callables (used for dependency injection in tests)
SingleExtractor = Callable[
    [
        Path,
        str,
        type[T] | None,
        str | None,
        list[int] | None,
        int,
        int,
        float,
        dict[str, str] | None,
        str | None,
        dict[str, str] | None,
    ],
    LLMExtractionResult[T | dict[str, Any]],
]
AsyncSingleExtractor = Callable[
    [
        Path,
        str,
        type[T] | None,
        str | None,
        list[int] | None,
        int,
        int,
        float,
        dict[str, str] | None,
        str | None,
        dict[str, str] | None,
    ],
    Awaitable[LLMExtractionResult[T | dict[str, Any]]],
]


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


def _extract_single(  # noqa: PLR0913
    path: Path,
    model: str,
    schema: type[T] | None,
    prompt: str | None,
    pages: list[int] | None,
    dpi: int,
    max_retries: int,
    temperature: float,
    credentials: dict[str, str] | None,
    base_url: str | None,
    headers: dict[str, str] | None,
) -> LLMExtractionResult[T | dict[str, Any]]:
    """Extract from specified pages in a single request. Internal helper."""
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


async def _extract_single_async(  # noqa: PLR0913
    path: Path,
    model: str,
    schema: type[T] | None,
    prompt: str | None,
    pages: list[int] | None,
    dpi: int,
    max_retries: int,
    temperature: float,
    credentials: dict[str, str] | None,
    base_url: str | None,
    headers: dict[str, str] | None,
) -> LLMExtractionResult[T | dict[str, Any]]:
    """Extract from specified pages in a single async request. Internal helper."""
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


@overload
def extract_structured(
    path: Path | str,
    model: str,
    *,
    schema: type[T],
    prompt: str | None = None,
    pages: list[int] | None = None,
    max_workers: int = 1,
    executor: ExecutorType = ExecutorType.THREAD,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    credentials: dict[str, str] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    _extractor: Any = None,
) -> LLMExtractionResult[T]: ...


@overload
def extract_structured(
    path: Path | str,
    model: str,
    *,
    schema: None = None,
    prompt: str | None = None,
    pages: list[int] | None = None,
    max_workers: int = 1,
    executor: ExecutorType = ExecutorType.THREAD,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    credentials: dict[str, str] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    _extractor: Any = None,
) -> LLMExtractionResult[dict[str, Any]]: ...


def extract_structured(  # noqa: PLR0913
    path: Path | str,
    model: str,
    *,
    schema: type[T] | None = None,
    prompt: str | None = None,
    pages: list[int] | None = None,
    max_workers: int = 1,
    executor: ExecutorType = ExecutorType.THREAD,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    credentials: dict[str, str] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    _extractor: SingleExtractor[T] | None = None,
) -> LLMExtractionResult[T | dict[str, Any]]:
    """Extract structured data from a document using an LLM.

    Args:
        path: Path to document/image file.
        model: Model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3-5-sonnet").
        schema: Pydantic model for structured output. None for free-form dict.
        prompt: Custom extraction prompt. Auto-generated from schema if None.
        pages: Page numbers to extract from (0-indexed). None for all pages.
        max_workers: Number of parallel workers. 1 means sequential (all pages in one request).
                     >1 means parallel (1 page per request, results merged into list).
        executor: Type of executor (THREAD or PROCESS) for parallel extraction.
        dpi: DPI for PDF-to-image conversion.
        max_retries: Max retry attempts with validation feedback.
        temperature: Sampling temperature (0.0 = deterministic).
        credentials: Override credentials dict (otherwise uses env vars).
        base_url: Custom API base URL for OpenAI-compatible APIs (vLLM, Ollama, etc.).
        headers: Custom HTTP headers for OpenAI-compatible APIs.
        _extractor: Internal parameter for dependency injection (testing only).

    Returns:
        LLMExtractionResult containing extracted data, model info, and provider.
        When max_workers > 1, data is a list of per-page results.
    """
    path = Path(path) if isinstance(path, str) else path
    provider, model_name = _parse_model_string(model)
    extractor = _extractor or _extract_single

    # Single-threaded: all pages in one request
    if max_workers <= 1:
        return extractor(
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

    # Parallel: 1 page per request
    from xtra.base import ImageLoader

    # Get all pages if not specified
    if pages is None:
        loader = ImageLoader(path, dpi=dpi)
        pages = list(range(loader.page_count))
        loader.close()

    # Single page - no need for parallel
    if len(pages) <= 1:
        return extractor(
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

    # Parallel execution
    executor_class = ProcessPoolExecutor if executor == ExecutorType.PROCESS else ThreadPoolExecutor

    results: list[LLMExtractionResult[T | dict[str, Any]] | Exception] = [None] * len(pages)  # type: ignore[list-item]
    with executor_class(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(
                extractor,
                path,
                model,
                schema,
                prompt,
                [page],  # Single page per request
                dpi,
                max_retries,
                temperature,
                credentials,
                base_url,
                headers,
            ): i
            for i, page in enumerate(pages)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = e

    # Merge results: collect all data into a list
    merged_data: list[T | dict[str, Any]] = []
    total_usage: dict[str, int] = {}
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            raise ValueError(f"Extraction failed for page {pages[i]}: {res}") from res
        extraction_result: LLMExtractionResult[T | dict[str, Any]] = res
        merged_data.append(extraction_result.data)
        if extraction_result.usage:
            for key, value in extraction_result.usage.items():
                total_usage[key] = total_usage.get(key, 0) + value

    return cast(
        "LLMExtractionResult[T | dict[str, Any]]",
        LLMExtractionResult(
            data=merged_data,
            model=model_name,
            provider=provider,
            usage=total_usage if total_usage else None,
        ),
    )


@overload
async def extract_structured_async(
    path: Path | str,
    model: str,
    *,
    schema: type[T],
    prompt: str | None = None,
    pages: list[int] | None = None,
    max_workers: int = 1,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    credentials: dict[str, str] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    _extractor: Any = None,
) -> LLMExtractionResult[T]: ...


@overload
async def extract_structured_async(
    path: Path | str,
    model: str,
    *,
    schema: None = None,
    prompt: str | None = None,
    pages: list[int] | None = None,
    max_workers: int = 1,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    credentials: dict[str, str] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    _extractor: Any = None,
) -> LLMExtractionResult[dict[str, Any]]: ...


async def extract_structured_async(  # noqa: PLR0913
    path: Path | str,
    model: str,
    *,
    schema: type[T] | None = None,
    prompt: str | None = None,
    pages: list[int] | None = None,
    max_workers: int = 1,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    credentials: dict[str, str] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    _extractor: AsyncSingleExtractor[T] | None = None,
) -> LLMExtractionResult[T | dict[str, Any]]:
    """Async version of extract_structured.

    Args:
        path: Path to document/image file.
        model: Model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3-5-sonnet").
        schema: Pydantic model for structured output. None for free-form dict.
        prompt: Custom extraction prompt. Auto-generated from schema if None.
        pages: Page numbers to extract from (0-indexed). None for all pages.
        max_workers: Number of concurrent requests. 1 means sequential (all pages in one request).
                     >1 means parallel (1 page per request, results merged into list).
        dpi: DPI for PDF-to-image conversion.
        max_retries: Max retry attempts with validation feedback.
        temperature: Sampling temperature (0.0 = deterministic).
        credentials: Override credentials dict (otherwise uses env vars).
        base_url: Custom API base URL for OpenAI-compatible APIs (vLLM, Ollama, etc.).
        headers: Custom HTTP headers for OpenAI-compatible APIs.
        _extractor: Internal parameter for dependency injection (testing only).

    Returns:
        LLMExtractionResult containing extracted data, model info, and provider.
        When max_workers > 1, data is a list of per-page results.
    """
    path = Path(path) if isinstance(path, str) else path
    provider, model_name = _parse_model_string(model)
    extractor = _extractor or _extract_single_async

    # Single request: all pages in one request
    if max_workers <= 1:
        return await extractor(
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

    # Parallel: 1 page per request
    from xtra.base import ImageLoader

    # Get all pages if not specified
    if pages is None:
        loader = ImageLoader(path, dpi=dpi)
        pages = list(range(loader.page_count))
        loader.close()

    # Single page - no need for parallel
    if len(pages) <= 1:
        return await extractor(
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

    # Parallel async execution with semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_workers)

    async def extract_with_limit(page: int) -> LLMExtractionResult[T | dict[str, Any]]:
        async with semaphore:
            return await extractor(
                path,
                model,
                schema,
                prompt,
                [page],
                dpi,
                max_retries,
                temperature,
                credentials,
                base_url,
                headers,
            )

    # Run all extractions concurrently (limited by semaphore)
    tasks = [extract_with_limit(page) for page in pages]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge results: collect all data into a list
    merged_data: list[T | dict[str, Any]] = []
    total_usage: dict[str, int] = {}
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            raise ValueError(f"Extraction failed for page {pages[i]}: {res}") from res
        extraction_result: LLMExtractionResult[T | dict[str, Any]] = res  # type: ignore[assignment]
        merged_data.append(extraction_result.data)
        if extraction_result.usage:
            for key, value in extraction_result.usage.items():
                total_usage[key] = total_usage.get(key, 0) + value

    return cast(
        "LLMExtractionResult[T | dict[str, Any]]",
        LLMExtractionResult(
            data=merged_data,
            model=model_name,
            provider=provider,
            usage=total_usage if total_usage else None,
        ),
    )
