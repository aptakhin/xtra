"""OpenAI LLM extractor using instructor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast, get_type_hints

from pydantic import BaseModel

from xtra.extractors._image_loader import ImageLoader
from xtra.llm.adapters.image_encoder import ImageEncoder
from xtra.llm.models import LLMExtractionResult, LLMProvider

T = TypeVar("T", bound=BaseModel)


def _schema_to_field_description(schema: type[BaseModel]) -> str:
    """Convert Pydantic schema to human-readable field description."""
    lines = []
    hints = get_type_hints(schema)

    for field_name, field_type in hints.items():
        # Get field description if available
        field_info = schema.model_fields.get(field_name)
        description = ""
        if field_info and field_info.description:
            description = f" - {field_info.description}"

        # Format type name
        type_name = getattr(field_type, "__name__", str(field_type))
        lines.append(f"  - {field_name}: {type_name}{description}")

    return "\n".join(lines)


def _build_prompt(schema: type[BaseModel] | None, custom_prompt: str | None) -> str:
    """Build extraction prompt with schema field info."""
    if custom_prompt:
        if schema:
            # Append schema info to custom prompt
            fields = _schema_to_field_description(schema)
            return f"{custom_prompt}\n\nExpected fields:\n{fields}"
        # For dict extraction, append JSON instruction (required by OpenAI/Azure)
        return f"{custom_prompt}\n\nRespond with valid JSON."

    if schema:
        fields = _schema_to_field_description(schema)
        return f"Extract structured data from this document.\n\nExpected fields:\n{fields}"

    return "Extract all key-value pairs from this document as JSON."


def _build_messages(
    encoded_images: list[str],
    prompt: str,
) -> list[dict[str, Any]]:
    """Build OpenAI chat messages with images."""
    content: list[dict[str, Any]] = []

    for img_url in encoded_images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": img_url},
            }
        )

    content.append(
        {
            "type": "text",
            "text": prompt,
        }
    )

    return [{"role": "user", "content": content}]


def extract_openai(  # noqa: PLR0913
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
    """Extract structured data using OpenAI."""
    try:
        import instructor
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "OpenAI dependencies not installed. Install with: pip install xtra[llm-openai]"
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
        messages = _build_messages(encoded_images, extraction_prompt)

        # Create instructor client
        client = instructor.from_openai(OpenAI(api_key=api_key))

        # Extract with schema or dict
        if schema is not None:
            response = client.chat.completions.create(
                model=model,
                response_model=schema,
                max_retries=max_retries,
                messages=cast("Any", messages),
                temperature=temperature,
            )
            data = response
        else:
            # For dict extraction, use JSON mode without instructor
            raw_client = OpenAI(api_key=api_key)
            response = raw_client.chat.completions.create(
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
            provider=LLMProvider.OPENAI,
        )
    finally:
        loader.close()


async def extract_openai_async(  # noqa: PLR0913
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
    """Async extract structured data using OpenAI."""
    try:
        import instructor
        from openai import AsyncOpenAI
    except ImportError as e:
        raise ImportError(
            "OpenAI dependencies not installed. Install with: pip install xtra[llm-openai]"
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
        messages = _build_messages(encoded_images, extraction_prompt)

        # Create async instructor client
        client = instructor.from_openai(AsyncOpenAI(api_key=api_key))

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
            raw_client = AsyncOpenAI(api_key=api_key)
            response = await raw_client.chat.completions.create(
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
            provider=LLMProvider.OPENAI,
        )
    finally:
        loader.close()
