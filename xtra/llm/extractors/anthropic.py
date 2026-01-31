"""Anthropic LLM extractor using instructor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from xtra.llm.adapters.image_encoder import ImageEncoder
from xtra.llm.extractors.openai import _build_prompt
from xtra.llm.models import LLMExtractionResult, LLMProvider
from xtra.utils.image_loader import ImageLoader

T = TypeVar("T", bound=BaseModel)


def _build_messages_anthropic(
    encoded_images: list[str],
    prompt: str,
) -> list[dict[str, Any]]:
    """Build Anthropic chat messages with images."""
    content: list[dict[str, Any]] = []

    for img_url in encoded_images:
        # Anthropic uses different format for base64 images
        # Extract media type and data from data URL
        if img_url.startswith("data:"):
            parts = img_url.split(",", 1)
            media_type = parts[0].replace("data:", "").replace(";base64", "")
            data = parts[1]
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                }
            )

    content.append(
        {
            "type": "text",
            "text": prompt,
        }
    )

    return [{"role": "user", "content": content}]


def extract_anthropic(  # noqa: PLR0913
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
) -> LLMExtractionResult[T | dict[str, Any]]:
    """Extract structured data using Anthropic."""
    try:
        import instructor
        from anthropic import Anthropic
    except ImportError as e:
        raise ImportError(
            "Anthropic dependencies not installed. Install with: pip install xtra[llm-anthropic]"
        ) from e

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
        messages = _build_messages_anthropic(encoded_images, extraction_prompt)

        # Create instructor client
        client = instructor.from_anthropic(Anthropic(api_key=api_key))  # type: ignore[possibly-missing-attribute]

        # Extract with schema or dict
        if schema is not None:
            response = client.messages.create(
                model=model,
                response_model=schema,
                max_retries=max_retries,
                messages=cast("Any", messages),
                max_tokens=4096,
                temperature=temperature,
            )
            data = response
        else:
            # For dict extraction, use raw client with JSON instruction
            raw_client = Anthropic(api_key=api_key)
            response = raw_client.messages.create(
                model=model,
                messages=messages,
                max_tokens=4096,
                temperature=temperature,
            )
            import json

            data = json.loads(response.content[0].text)

        return LLMExtractionResult(
            data=data,
            model=model,
            provider=LLMProvider.ANTHROPIC,
        )
    finally:
        loader.close()


async def extract_anthropic_async(  # noqa: PLR0913
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
) -> LLMExtractionResult[T | dict[str, Any]]:
    """Async extract structured data using Anthropic."""
    try:
        import instructor
        from anthropic import AsyncAnthropic
    except ImportError as e:
        raise ImportError(
            "Anthropic dependencies not installed. Install with: pip install xtra[llm-anthropic]"
        ) from e

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
        messages = _build_messages_anthropic(encoded_images, extraction_prompt)

        # Create async instructor client
        client = instructor.from_anthropic(AsyncAnthropic(api_key=api_key))  # type: ignore[possibly-missing-attribute]

        # Extract with schema or dict
        if schema is not None:
            response = await client.messages.create(
                model=model,
                response_model=schema,
                max_retries=max_retries,
                messages=cast("Any", messages),
                max_tokens=4096,
                temperature=temperature,
            )
            data = response
        else:
            raw_client = AsyncAnthropic(api_key=api_key)
            response = await raw_client.messages.create(
                model=model,
                messages=messages,
                max_tokens=4096,
                temperature=temperature,
            )
            import json

            data = json.loads(response.content[0].text)

        return LLMExtractionResult(
            data=data,
            model=model,
            provider=LLMProvider.ANTHROPIC,
        )
    finally:
        loader.close()
