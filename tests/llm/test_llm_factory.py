"""Tests for LLM extractor factory - model string parsing and routing."""

from pathlib import Path
from unittest.mock import patch

import pytest

from xtra.llm.factory import _get_credential, _parse_model_string
from xtra.llm.models import LLMExtractionResult, LLMProvider

TEST_DATA_DIR = Path(__file__).parent.parent / "data"

# Model string parsing tests


def test_parse_explicit_openai() -> None:
    provider, model = _parse_model_string("openai/gpt-4o")
    assert provider == LLMProvider.OPENAI
    assert model == "gpt-4o"


def test_parse_explicit_anthropic() -> None:
    provider, model = _parse_model_string("anthropic/claude-3-5-sonnet-20241022")
    assert provider == LLMProvider.ANTHROPIC
    assert model == "claude-3-5-sonnet-20241022"


def test_parse_explicit_google() -> None:
    provider, model = _parse_model_string("google/gemini-1.5-pro")
    assert provider == LLMProvider.GOOGLE
    assert model == "gemini-1.5-pro"


def test_parse_explicit_azure() -> None:
    provider, model = _parse_model_string("azure-openai/my-deployment")
    assert provider == LLMProvider.AZURE_OPENAI
    assert model == "my-deployment"


def test_parse_inferred_gpt() -> None:
    provider, model = _parse_model_string("gpt-4o")
    assert provider == LLMProvider.OPENAI


def test_parse_inferred_claude() -> None:
    provider, model = _parse_model_string("claude-3-5-sonnet")
    assert provider == LLMProvider.ANTHROPIC


def test_parse_inferred_gemini() -> None:
    provider, model = _parse_model_string("gemini-1.5-pro")
    assert provider == LLMProvider.GOOGLE


def test_parse_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Cannot infer provider"):
        _parse_model_string("unknown-model")


def test_parse_case_insensitive_provider() -> None:
    provider, _ = _parse_model_string("OpenAI/gpt-4o")
    assert provider == LLMProvider.OPENAI


# Credential helper tests


def test_credential_from_dict() -> None:
    credentials = {"KEY": "from_dict"}
    assert _get_credential("KEY", credentials) == "from_dict"


def test_credential_from_env(monkeypatch) -> None:
    monkeypatch.setenv("TEST_KEY", "from_env")
    assert _get_credential("TEST_KEY", None) == "from_env"


def test_credential_dict_precedence(monkeypatch) -> None:
    monkeypatch.setenv("KEY", "from_env")
    credentials = {"KEY": "from_dict"}
    assert _get_credential("KEY", credentials) == "from_dict"


def test_credential_not_found() -> None:
    assert _get_credential("NONEXISTENT_KEY_12345", None) is None


# Extraction routing tests


class TestExtractStructuredRouting:
    """Tests for extract_structured routing to correct providers."""

    @patch("xtra.llm.extractors.openai.extract_openai")
    def test_routes_to_openai(self, mock_extract_openai) -> None:
        """Test that OpenAI models route to extract_openai."""
        from xtra.llm.factory import extract_structured

        mock_result = LLMExtractionResult(
            data={"key": "value"}, model="gpt-4o", provider=LLMProvider.OPENAI
        )
        mock_extract_openai.return_value = mock_result

        result = extract_structured(
            TEST_DATA_DIR / "test_image.png",
            model="openai/gpt-4o",
        )

        assert result == mock_result
        mock_extract_openai.assert_called_once()

    @patch("xtra.llm.extractors.anthropic.extract_anthropic")
    def test_routes_to_anthropic(self, mock_extract_anthropic) -> None:
        """Test that Anthropic models route to extract_anthropic."""
        from xtra.llm.factory import extract_structured

        mock_result = LLMExtractionResult(
            data={"key": "value"},
            model="claude-3-5-sonnet",
            provider=LLMProvider.ANTHROPIC,
        )
        mock_extract_anthropic.return_value = mock_result

        result = extract_structured(
            TEST_DATA_DIR / "test_image.png",
            model="anthropic/claude-3-5-sonnet",
        )

        assert result == mock_result
        mock_extract_anthropic.assert_called_once()

    @patch("xtra.llm.extractors.google.extract_google")
    def test_routes_to_google(self, mock_extract_google) -> None:
        """Test that Google models route to extract_google."""
        from xtra.llm.factory import extract_structured

        mock_result = LLMExtractionResult(
            data={"key": "value"},
            model="gemini-1.5-pro",
            provider=LLMProvider.GOOGLE,
        )
        mock_extract_google.return_value = mock_result

        result = extract_structured(
            TEST_DATA_DIR / "test_image.png",
            model="google/gemini-1.5-pro",
        )

        assert result == mock_result
        mock_extract_google.assert_called_once()

    @patch("xtra.llm.extractors.azure_openai.extract_azure_openai")
    def test_routes_to_azure_openai(self, mock_extract_azure) -> None:
        """Test that Azure OpenAI models route to extract_azure_openai."""
        from xtra.llm.factory import extract_structured

        mock_result = LLMExtractionResult(
            data={"key": "value"},
            model="my-deployment",
            provider=LLMProvider.AZURE_OPENAI,
        )
        mock_extract_azure.return_value = mock_result

        result = extract_structured(
            TEST_DATA_DIR / "test_image.png",
            model="azure-openai/my-deployment",
            credentials={
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            },
        )

        assert result == mock_result
        mock_extract_azure.assert_called_once()

    def test_azure_openai_raises_without_endpoint(self, monkeypatch) -> None:
        """Test that Azure OpenAI raises error without endpoint."""
        from xtra.llm.factory import extract_structured

        # Clear any endpoint env vars that might be set
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("XTRA_AZURE_DI_ENDPOINT", raising=False)

        with pytest.raises(ValueError, match="endpoint required"):
            extract_structured(
                TEST_DATA_DIR / "test_image.png",
                model="azure-openai/my-deployment",
            )

    @patch("xtra.llm.extractors.openai.extract_openai")
    def test_passes_base_url_to_openai(self, mock_extract_openai) -> None:
        """Test that base_url is passed to OpenAI extractor."""
        from xtra.llm.factory import extract_structured

        mock_result = LLMExtractionResult(data={}, model="gpt-4o", provider=LLMProvider.OPENAI)
        mock_extract_openai.return_value = mock_result

        extract_structured(
            TEST_DATA_DIR / "test_image.png",
            model="openai/gpt-4o",
            base_url="https://custom.api.com",
            headers={"X-Custom": "header"},
        )

        call_kwargs = mock_extract_openai.call_args.kwargs
        assert call_kwargs["base_url"] == "https://custom.api.com"
        assert call_kwargs["headers"] == {"X-Custom": "header"}
