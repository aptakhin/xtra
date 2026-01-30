"""Azure Document Intelligence extractor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from xtra.adapters.azure_di import AzureDocumentIntelligenceAdapter
from xtra.models import CoordinateUnit, ExtractorMetadata, Page
from xtra.extractors.base import BaseExtractor, PageExtractionResult

if TYPE_CHECKING:
    from azure.ai.documentintelligence import DocumentIntelligenceClient

logger = logging.getLogger(__name__)


def _check_azure_installed() -> None:
    """Check if azure-ai-documentintelligence is installed."""
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "azure-ai-documentintelligence is not installed. "
            "Install it with: pip install xtra[azure]"
        ) from e


class AzureDocumentIntelligenceExtractor(BaseExtractor):
    """Extract text from documents using Azure Document Intelligence."""

    def __init__(
        self,
        path: Path | str,
        endpoint: str,
        key: str,
        model_id: str = "prebuilt-read",
        output_unit: CoordinateUnit = CoordinateUnit.POINTS,
    ) -> None:
        _check_azure_installed()
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential

        super().__init__(path, output_unit)
        self.endpoint = endpoint
        self.model_id = model_id
        self._client: DocumentIntelligenceClient = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )
        self._result: Optional[Any] = None
        self._adapter: Optional[AzureDocumentIntelligenceAdapter] = None
        self._analyze_document()

    def _analyze_document(self) -> None:
        """Send document to Azure DI for analysis."""
        try:
            with open(self.path, "rb") as f:
                poller = self._client.begin_analyze_document(
                    model_id=self.model_id,
                    body=f,
                    content_type="application/octet-stream",
                )
                self._result = poller.result()
                self._adapter = AzureDocumentIntelligenceAdapter(self._result, self.model_id)
        except (OSError, ValueError) as e:
            logger.warning("Failed to analyze document with Azure DI: %s", e)
            self._result = None
            self._adapter = AzureDocumentIntelligenceAdapter(None, self.model_id)

    def get_page_count(self) -> int:
        if self._adapter is None:
            return 0
        return self._adapter.page_count

    def extract_page(self, page: int) -> PageExtractionResult:
        """Extract a single page by number (0-indexed)."""
        try:
            if self._adapter is None:
                raise ValueError("Document analysis failed")

            converted_page = self._adapter.convert_page(page)
            # Convert from native INCHES to output_unit
            converted_page = self._convert_page(converted_page, CoordinateUnit.INCHES)
            return PageExtractionResult(page=converted_page, success=True)

        except (IndexError, ValueError, AttributeError) as e:
            logger.warning("Failed to extract page %d from Azure DI result: %s", page, e)
            return PageExtractionResult(
                page=Page(page=page, width=0, height=0, texts=[]),
                success=False,
                error=str(e),
            )

    def get_extractor_metadata(self) -> ExtractorMetadata:
        if self._adapter is None:
            return AzureDocumentIntelligenceAdapter(None, self.model_id).get_metadata()
        return self._adapter.get_metadata()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
