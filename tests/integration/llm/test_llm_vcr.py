"""VCR-based tests for LLM extractors.

These tests record real API responses and replay them. To record:
1. Set API keys in environment (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY)
2. Delete the cassette files in tests/cassettes/llm/
3. Run: uv run pytest tests/integration/llm/test_llm_vcr.py -v

After cassettes are recorded, tests run without credentials.
If cassettes don't exist and credentials aren't set, tests FAIL.
"""

import os
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data"
CASSETTE_DIR = Path(__file__).parent.parent.parent / "cassettes" / "llm"


class DocumentInfo(BaseModel):
    """Schema for extracting document information."""

    title: str = Field(description="The title or heading of the document")
    has_text: bool = Field(description="Whether the document contains readable text")


def get_api_key_or_check_cassette(env_var: str, cassette_name: str) -> str:
    """Get API key from environment, or verify cassette exists.

    Returns a placeholder key if cassette exists (VCR will use recorded response).
    Raises pytest.fail if neither credentials nor cassette are available.
    """
    key = os.getenv(env_var)
    if key:
        return key

    cassette_path = CASSETTE_DIR / f"{cassette_name}.yaml"
    if cassette_path.exists():
        return "cassette-replay-key"

    pytest.fail(
        f"No {env_var} set and no cassette at {cassette_path}. "
        f"Set {env_var} and run tests to record cassette."
    )


class TestOpenAIVCR:
    """VCR-based tests for OpenAI extractor."""

    @pytest.mark.vcr(cassette_library_dir=str(CASSETTE_DIR))
    def test_openai_with_schema(self) -> None:
        """Test OpenAI extraction with schema using recorded response."""
        from xtra.llm.extractors.openai import extract_openai

        api_key = get_api_key_or_check_cassette("OPENAI_API_KEY", "test_openai_with_schema")

        result = extract_openai(
            TEST_DATA_DIR / "test_image.png",
            model="gpt-4o-mini",
            schema=DocumentInfo,
            api_key=api_key,
        )

        assert result.data is not None
        assert isinstance(result.data, DocumentInfo)
        assert result.model == "gpt-4o-mini"

    @pytest.mark.vcr(cassette_library_dir=str(CASSETTE_DIR))
    def test_openai_without_schema(self) -> None:
        """Test OpenAI extraction without schema (JSON mode)."""
        from xtra.llm.extractors.openai import extract_openai

        api_key = get_api_key_or_check_cassette("OPENAI_API_KEY", "test_openai_without_schema")

        result = extract_openai(
            TEST_DATA_DIR / "test_image.png",
            model="gpt-4o-mini",
            schema=None,
            prompt="Extract all text visible in this image as JSON",
            api_key=api_key,
        )

        assert result.data is not None
        assert isinstance(result.data, dict)


class TestAnthropicVCR:
    """VCR-based tests for Anthropic extractor."""

    @pytest.mark.vcr(cassette_library_dir=str(CASSETTE_DIR))
    def test_anthropic_with_schema(self) -> None:
        """Test Anthropic extraction with schema using recorded response."""
        from xtra.llm.extractors.anthropic import extract_anthropic

        api_key = get_api_key_or_check_cassette("ANTHROPIC_API_KEY", "test_anthropic_with_schema")

        result = extract_anthropic(
            TEST_DATA_DIR / "test_image.png",
            model="claude-3-5-haiku-20241022",
            schema=DocumentInfo,
            api_key=api_key,
        )

        assert result.data is not None
        assert isinstance(result.data, DocumentInfo)
        assert result.model == "claude-3-5-haiku-20241022"

    @pytest.mark.vcr(cassette_library_dir=str(CASSETTE_DIR))
    def test_anthropic_without_schema(self) -> None:
        """Test Anthropic extraction without schema."""
        from xtra.llm.extractors.anthropic import extract_anthropic

        api_key = get_api_key_or_check_cassette(
            "ANTHROPIC_API_KEY", "test_anthropic_without_schema"
        )

        result = extract_anthropic(
            TEST_DATA_DIR / "test_image.png",
            model="claude-3-5-haiku-20241022",
            schema=None,
            prompt="Extract all text visible in this image as JSON",
            api_key=api_key,
        )

        assert result.data is not None
        assert isinstance(result.data, dict)


class TestGoogleVCR:
    """VCR-based tests for Google Gemini extractor."""

    @pytest.mark.vcr(cassette_library_dir=str(CASSETTE_DIR))
    def test_google_with_schema(self) -> None:
        """Test Google Gemini extraction with schema using recorded response."""
        from xtra.llm.extractors.google import extract_google

        api_key = get_api_key_or_check_cassette("GOOGLE_API_KEY", "test_google_with_schema")

        result = extract_google(
            TEST_DATA_DIR / "test_image.png",
            model="gemini-2.0-flash",
            schema=DocumentInfo,
            api_key=api_key,
        )

        assert result.data is not None
        assert isinstance(result.data, DocumentInfo)
        assert result.model == "gemini-2.0-flash"

    @pytest.mark.vcr(cassette_library_dir=str(CASSETTE_DIR))
    def test_google_without_schema(self) -> None:
        """Test Google Gemini extraction without schema."""
        from xtra.llm.extractors.google import extract_google

        api_key = get_api_key_or_check_cassette("GOOGLE_API_KEY", "test_google_without_schema")

        result = extract_google(
            TEST_DATA_DIR / "test_image.png",
            model="gemini-2.0-flash",
            schema=None,
            prompt="Extract all text visible in this image as JSON",
            api_key=api_key,
        )

        assert result.data is not None
        assert isinstance(result.data, dict)
