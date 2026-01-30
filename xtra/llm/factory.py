"""Factory functions for LLM-based extraction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar, overload

from pydantic import BaseModel

from xtra.llm.models import LLMExtractionResult, LLMProvider

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
