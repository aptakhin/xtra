"""Tests for LLM extractor factory - model string parsing."""

import pytest

from xtra.llm.factory import _get_credential, _parse_model_string
from xtra.llm.models import LLMProvider

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
