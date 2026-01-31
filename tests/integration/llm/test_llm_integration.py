"""Integration tests for LLM extraction.

Tests skip if the required API key is not configured.
"""

import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from xtra.llm import extract_structured, extract_structured_async
from xtra.llm.models import LLMProvider

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data"


class SimpleExtraction(BaseModel):
    """Simple schema for testing."""

    text_content: str


class Invoice(BaseModel):
    """Nested schema for testing."""

    class LineItem(BaseModel):
        description: str
        amount: float

    vendor: str
    items: list[LineItem]
    total: float


# Fixtures


@pytest.fixture
def openai_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def anthropic_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture
def google_key():
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        pytest.skip("GOOGLE_API_KEY not set")
    return key


# OpenAI tests


def test_openai_with_schema(openai_key) -> None:
    result = extract_structured(
        TEST_DATA_DIR / "test_image.png",
        model="openai/gpt-4o-mini",
        schema=SimpleExtraction,
    )
    assert result.provider == LLMProvider.OPENAI
    assert isinstance(result.data.text_content, str)


def test_openai_without_schema(openai_key) -> None:
    result = extract_structured(
        TEST_DATA_DIR / "test_image.png",
        model="openai/gpt-4o-mini",
        prompt="Extract all visible text as key-value pairs",
    )
    assert isinstance(result.data, dict)


def test_openai_with_prompt(openai_key) -> None:
    result = extract_structured(
        TEST_DATA_DIR / "test_image.png",
        model="openai/gpt-4o-mini",
        schema=SimpleExtraction,
        prompt="Extract the main text content from this image",
    )
    assert isinstance(result.data.text_content, str)


def test_openai_specific_pages(openai_key) -> None:
    result = extract_structured(
        TEST_DATA_DIR / "test_pdf_2p_text.pdf",
        model="openai/gpt-4o-mini",
        schema=SimpleExtraction,
        pages=[0],  # First page only
    )
    assert isinstance(result.data.text_content, str)


def test_openai_nested_schema(openai_key) -> None:
    result = extract_structured(
        TEST_DATA_DIR / "test_pdf_2p_text.pdf",
        model="openai/gpt-4o-mini",
        schema=Invoice,
        prompt="Extract as invoice. Use 'Test' as vendor, create items from text.",
    )
    assert isinstance(result.data.vendor, str)
    assert isinstance(result.data.items, list)


@pytest.mark.asyncio
async def test_openai_async(openai_key) -> None:
    result = await extract_structured_async(
        TEST_DATA_DIR / "test_image.png",
        model="openai/gpt-4o-mini",
        schema=SimpleExtraction,
    )
    assert isinstance(result.data.text_content, str)


# Anthropic tests


def test_anthropic_with_schema(anthropic_key) -> None:
    result = extract_structured(
        TEST_DATA_DIR / "test_image.png",
        model="anthropic/claude-3-5-haiku-latest",
        schema=SimpleExtraction,
    )
    assert result.provider == LLMProvider.ANTHROPIC
    assert isinstance(result.data.text_content, str)


# Google tests


def test_google_with_schema(google_key) -> None:
    result = extract_structured(
        TEST_DATA_DIR / "test_image.png",
        model="google/gemini-1.5-flash",
        schema=SimpleExtraction,
    )
    assert result.provider == LLMProvider.GOOGLE
    assert isinstance(result.data.text_content, str)
