"""Tests for Azure Document Intelligence extractor.

Real extractor functionality is tested in integration tests
when AZURE_DI_ENDPOINT and AZURE_DI_KEY environment variables are set.
Adapter logic is tested in test_azure_di_adapter.py.
"""

from xtra.extractors.azure_di import AzureDocumentIntelligenceExtractor
from xtra.adapters.azure_di import AzureDocumentIntelligenceAdapter


def test_extractor_class_exists() -> None:
    """Verify extractor class can be imported."""
    assert AzureDocumentIntelligenceExtractor is not None


def test_adapter_class_exists() -> None:
    """Verify adapter class can be imported."""
    assert AzureDocumentIntelligenceAdapter is not None
