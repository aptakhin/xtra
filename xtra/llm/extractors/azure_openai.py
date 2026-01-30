"""Azure OpenAI LLM extractor using instructor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from xtra.extractors._image_loader import ImageLoader
from xtra.llm.adapters.image_encoder import ImageEncoder
from xtra.llm.extractors.openai import _build_messages, _build_prompt
from xtra.llm.models import LLMExtractionResult, LLMProvider

T = TypeVar("T", bound=BaseModel)


def extract_azure_openai(  # noqa: PLR0913
    path: Path | str,
    model: str,
    *,
    schema: type[T] | None = None,
    prompt: str | None = None,
    pages: list[int] | None = None,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    api_key: str | None = None,
    endpoint: str | None = None,
    api_version: str = "2024-02-15-preview",
) -> LLMExtractionResult[T | dict[str, Any]]:
    """Extract structured data using Azure OpenAI."""
    try:
        import instructor
        from openai import AzureOpenAI
    except ImportError as e:
        raise ImportError(
            "OpenAI dependencies not installed. Install with: pip install xtra[llm-openai]"
        ) from e

    if not endpoint:
        raise ValueError("Azure OpenAI endpoint required")

    path = Path(path) if isinstance(path, str) else path
    loader = ImageLoader(path, dpi=dpi)
    encoder = ImageEncoder()

    try:
        # Load and encode images
        page_nums = pages if pages is not None else list(range(loader.page_count))
        images = [loader.get_page(p) for p in page_nums]
        encoded_images = encoder.encode_images(images)

        # Build messages
        extraction_prompt = _build_prompt(schema, prompt)
        messages = _build_messages(encoded_images, extraction_prompt)

        # Create Azure client
        azure_client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        client = instructor.from_openai(azure_client)

        # Extract with schema or dict
        if schema is not None:
            response = client.chat.completions.create(
                model=model,  # This is the deployment name in Azure
                response_model=schema,
                max_retries=max_retries,
                messages=cast("Any", messages),
                temperature=temperature,
            )
            data = response
        else:
            response = azure_client.chat.completions.create(
                model=model,
                messages=cast("Any", messages),
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            import json

            data = json.loads(response.choices[0].message.content)

        return LLMExtractionResult(
            data=data,
            model=model,
            provider=LLMProvider.AZURE_OPENAI,
        )
    finally:
        loader.close()


async def extract_azure_openai_async(  # noqa: PLR0913
    path: Path | str,
    model: str,
    *,
    schema: type[T] | None = None,
    prompt: str | None = None,
    pages: list[int] | None = None,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    api_key: str | None = None,
    endpoint: str | None = None,
    api_version: str = "2024-02-15-preview",
) -> LLMExtractionResult[T | dict[str, Any]]:
    """Async extract structured data using Azure OpenAI."""
    try:
        import instructor
        from openai import AsyncAzureOpenAI
    except ImportError as e:
        raise ImportError(
            "OpenAI dependencies not installed. Install with: pip install xtra[llm-openai]"
        ) from e

    if not endpoint:
        raise ValueError("Azure OpenAI endpoint required")

    path = Path(path) if isinstance(path, str) else path
    loader = ImageLoader(path, dpi=dpi)
    encoder = ImageEncoder()

    try:
        # Load and encode images
        page_nums = pages if pages is not None else list(range(loader.page_count))
        images = [loader.get_page(p) for p in page_nums]
        encoded_images = encoder.encode_images(images)

        # Build messages
        extraction_prompt = _build_prompt(schema, prompt)
        messages = _build_messages(encoded_images, extraction_prompt)

        # Create async Azure client
        azure_client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        client = instructor.from_openai(azure_client)

        # Extract with schema or dict
        if schema is not None:
            response = await client.chat.completions.create(
                model=model,
                response_model=schema,
                max_retries=max_retries,
                messages=cast("Any", messages),
                temperature=temperature,
            )
            data = response
        else:
            response = await azure_client.chat.completions.create(
                model=model,
                messages=cast("Any", messages),
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            import json

            data = json.loads(response.choices[0].message.content)

        return LLMExtractionResult(
            data=data,
            model=model,
            provider=LLMProvider.AZURE_OPENAI,
        )
    finally:
        loader.close()
