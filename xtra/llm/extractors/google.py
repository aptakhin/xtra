"""Google Gemini LLM extractor using instructor."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

from pydantic import BaseModel

from xtra.extractors._image_loader import ImageLoader
from xtra.llm.models import LLMExtractionResult, LLMProvider
from xtra.llm.extractors.openai import _build_prompt

T = TypeVar("T", bound=BaseModel)


def _build_genai_content(images: list[Any], prompt: str) -> list[Any]:
    """Build content list for google.genai API with images and text."""
    from google.genai import types

    content: list[Any] = []
    for img in images:
        # Convert PIL image to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        content.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
    content.append(prompt)
    return content


def extract_google(
    path: Path | str,
    model: str,
    *,
    schema: Optional[Type[T]] = None,
    prompt: Optional[str] = None,
    pages: Optional[List[int]] = None,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
) -> LLMExtractionResult[Union[T, Dict[str, Any]]]:
    """Extract structured data using Google Gemini."""
    try:
        import instructor
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError(
            "Google dependencies not installed. Install with: pip install xtra[llm-google]"
        ) from e

    path = Path(path) if isinstance(path, str) else path
    loader = ImageLoader(path, dpi=dpi)

    try:
        # Create client with API key
        client = genai.Client(api_key=api_key)

        # Load images (Gemini accepts PIL images directly)
        page_nums = pages if pages is not None else list(range(loader.page_count))
        images = [loader.get_page(p) for p in page_nums]

        # Build prompt
        extraction_prompt = _build_prompt(schema, prompt)

        # Build content with images and text
        content = _build_genai_content(images, extraction_prompt)

        # Extract with schema or dict
        data: T | Dict[str, Any]
        if schema is not None:
            # Use instructor with from_genai
            instructor_client = instructor.from_genai(  # type: ignore[possibly-missing-attribute]
                client=client,
                mode=instructor.Mode.GENAI_TOOLS,
            )
            response = instructor_client.chat.completions.create(
                model=model,
                response_model=schema,
                max_retries=max_retries,
                messages=cast(Any, [{"role": "user", "content": content}]),
                generation_config=types.GenerateContentConfig(temperature=temperature),
            )
            data = cast(T, response)
        else:
            # For dict extraction, use raw client with JSON mime type
            response = client.models.generate_content(
                model=model,
                contents=content,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
            if response.text is None:
                raise ValueError("Empty response from Gemini API")
            data = json.loads(response.text)

        return LLMExtractionResult(
            data=data,
            model=model,
            provider=LLMProvider.GOOGLE,
        )
    finally:
        loader.close()


async def extract_google_async(
    path: Path | str,
    model: str,
    *,
    schema: Optional[Type[T]] = None,
    prompt: Optional[str] = None,
    pages: Optional[List[int]] = None,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
) -> LLMExtractionResult[Union[T, Dict[str, Any]]]:
    """Async extract structured data using Google Gemini."""
    try:
        import instructor
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError(
            "Google dependencies not installed. Install with: pip install xtra[llm-google]"
        ) from e

    path = Path(path) if isinstance(path, str) else path
    loader = ImageLoader(path, dpi=dpi)

    try:
        # Create async client with API key
        client = genai.Client(api_key=api_key)

        # Load images
        page_nums = pages if pages is not None else list(range(loader.page_count))
        images = [loader.get_page(p) for p in page_nums]

        # Build prompt
        extraction_prompt = _build_prompt(schema, prompt)

        # Build content with images and text
        content = _build_genai_content(images, extraction_prompt)

        # Extract with schema or dict
        data: T | Dict[str, Any]
        if schema is not None:
            # Use instructor with from_genai
            instructor_client = instructor.from_genai(  # type: ignore[possibly-missing-attribute]
                client=client,
                mode=instructor.Mode.GENAI_TOOLS,
                use_async=True,
            )
            response = await instructor_client.chat.completions.create(
                model=model,
                response_model=schema,
                max_retries=max_retries,
                messages=cast(Any, [{"role": "user", "content": content}]),
                generation_config=types.GenerateContentConfig(temperature=temperature),
            )
            data = cast(T, response)
        else:
            # For dict extraction, use raw async client
            response = await client.aio.models.generate_content(
                model=model,
                contents=content,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
            if response.text is None:
                raise ValueError("Empty response from Gemini API")
            data = json.loads(response.text)

        return LLMExtractionResult(
            data=data,
            model=model,
            provider=LLMProvider.GOOGLE,
        )
    finally:
        loader.close()
