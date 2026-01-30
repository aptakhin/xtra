"""Tests for LLM models."""

from pydantic import BaseModel

from xtra.llm.models import (
    LLMExtractionResult,
    LLMProvider,
    PageExtractionConfig,
)


def test_provider_values() -> None:
    assert LLMProvider.OPENAI == "openai"
    assert LLMProvider.ANTHROPIC == "anthropic"
    assert LLMProvider.GOOGLE == "google"
    assert LLMProvider.AZURE_OPENAI == "azure-openai"


def test_provider_from_string() -> None:
    assert LLMProvider("openai") == LLMProvider.OPENAI
    assert LLMProvider("anthropic") == LLMProvider.ANTHROPIC


def test_extraction_result_with_schema() -> None:
    class TestSchema(BaseModel):
        name: str
        value: int

    data = TestSchema(name="test", value=42)
    result = LLMExtractionResult(
        data=data,
        model="gpt-4o",
        provider=LLMProvider.OPENAI,
    )
    assert result.data.name == "test"
    assert result.data.value == 42


def test_extraction_result_with_dict() -> None:
    data = {"name": "test", "nested": {"key": "value"}}
    result = LLMExtractionResult(
        data=data,
        model="gpt-4o",
        provider=LLMProvider.OPENAI,
    )
    assert isinstance(result.data, dict)
    assert result.data["name"] == "test"
    nested = result.data["nested"]
    assert isinstance(nested, dict)
    assert nested["key"] == "value"


def test_page_config_defaults() -> None:
    config = PageExtractionConfig()
    assert config.page_numbers is None
    assert config.combine_pages is True


def test_page_config_specific_pages() -> None:
    config = PageExtractionConfig(page_numbers=[0, 2, 4])
    assert config.page_numbers == [0, 2, 4]
