"""Tests for prompt building from schema."""

from pydantic import BaseModel, Field

from xtra.llm.extractors.openai import _build_prompt, _schema_to_field_description


class SimpleSchema(BaseModel):
    name: str
    value: int


class SchemaWithDescriptions(BaseModel):
    invoice_number: str = Field(description="The unique invoice identifier")
    total: float = Field(description="Total amount in USD")
    vendor: str | None = Field(default=None, description="Vendor name if available")


class NestedSchema(BaseModel):
    class Item(BaseModel):
        description: str
        amount: float

    title: str
    items: list[Item]


def test_schema_to_field_description_simple() -> None:
    result = _schema_to_field_description(SimpleSchema)
    assert "name" in result
    assert "str" in result
    assert "value" in result
    assert "int" in result


def test_schema_to_field_description_with_descriptions() -> None:
    result = _schema_to_field_description(SchemaWithDescriptions)
    assert "invoice_number" in result
    assert "The unique invoice identifier" in result
    assert "total" in result
    assert "Total amount in USD" in result


def test_build_prompt_with_schema() -> None:
    result = _build_prompt(SimpleSchema, None)
    assert "Extract structured data" in result
    assert "name" in result
    assert "value" in result


def test_build_prompt_with_custom_prompt_and_schema() -> None:
    result = _build_prompt(SimpleSchema, "Extract invoice data")
    assert "Extract invoice data" in result
    assert "name" in result  # Schema fields appended


def test_build_prompt_with_custom_prompt_no_schema() -> None:
    result = _build_prompt(None, "Extract all text")
    # Custom prompt with JSON instruction appended (required by OpenAI/Azure)
    assert "Extract all text" in result
    assert "json" in result.lower()


def test_build_prompt_no_schema_no_prompt() -> None:
    result = _build_prompt(None, None)
    assert "key-value" in result.lower()
    assert "json" in result.lower()
