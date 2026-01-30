"""Google Document AI extractor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from xtra.adapters.google_docai import GoogleDocumentAIAdapter
from xtra.models import CoordinateUnit, ExtractorMetadata, Page
from xtra.extractors.base import BaseExtractor, PageExtractionResult

if TYPE_CHECKING:
    from google.cloud.documentai_v1 import Document

logger = logging.getLogger(__name__)


def _check_google_docai_installed() -> None:
    """Check if google-cloud-documentai is installed."""
    try:
        from google.cloud import documentai  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "google-cloud-documentai is not installed. Install it with: pip install xtra[google]"
        ) from e


class GoogleDocumentAIExtractor(BaseExtractor):
    """Extract text from documents using Google Document AI."""

    def __init__(
        self,
        path: Path | str,
        processor_name: str,
        credentials_path: str,
        mime_type: Optional[str] = None,
        output_unit: CoordinateUnit = CoordinateUnit.POINTS,
    ) -> None:
        """Initialize Google Document AI extractor.

        Args:
            path: Path to the document file.
            processor_name: Full processor resource name, e.g.,
                'projects/{project}/locations/{location}/processors/{processor_id}'
            credentials_path: Path to service account JSON credentials file.
            mime_type: Optional MIME type. If not provided, will be inferred from file extension.
            output_unit: Coordinate unit for output. Default POINTS.
        """
        _check_google_docai_installed()
        from google.cloud import documentai
        from google.oauth2 import service_account

        super().__init__(path, output_unit)
        self.processor_name = processor_name
        self.credentials_path = credentials_path
        self.mime_type = mime_type or self._infer_mime_type()

        # Create credentials from service account file
        credentials = service_account.Credentials.from_service_account_file(credentials_path)

        # Extract location from processor name for endpoint
        # Format: projects/{project}/locations/{location}/processors/{processor_id}
        location = self._extract_location_from_processor_name(processor_name)
        opts = {"api_endpoint": f"{location}-documentai.googleapis.com"}

        self._client = documentai.DocumentProcessorServiceClient(
            credentials=credentials, client_options=opts
        )
        self._document: Optional[Document] = None
        self._adapter: Optional[GoogleDocumentAIAdapter] = None
        self._process_document()

    def _infer_mime_type(self) -> str:
        """Infer MIME type from file extension."""
        suffix = self.path.suffix.lower()
        mime_types = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }
        return mime_types.get(suffix, "application/pdf")

    @staticmethod
    def _extract_location_from_processor_name(processor_name: str) -> str:
        """Extract location from processor resource name."""
        # Format: projects/{project}/locations/{location}/processors/{processor_id}
        parts = processor_name.split("/")
        try:
            loc_index = parts.index("locations")
            return parts[loc_index + 1]
        except (ValueError, IndexError):
            return "us"  # Default to US

    def _process_document(self) -> None:
        """Send document to Google Document AI for processing."""
        from google.cloud import documentai

        try:
            with open(self.path, "rb") as f:
                content = f.read()

            raw_document = documentai.RawDocument(
                content=content,
                mime_type=self.mime_type,
            )

            request = documentai.ProcessRequest(
                name=self.processor_name,
                raw_document=raw_document,
            )

            result = self._client.process_document(request=request)
            self._document = result.document
            self._adapter = GoogleDocumentAIAdapter(self._document, self.processor_name)

        except (OSError, ValueError, Exception) as e:
            logger.warning("Failed to process document with Google Document AI: %s", e)
            self._document = None
            self._adapter = GoogleDocumentAIAdapter(None, self.processor_name)

    def get_page_count(self) -> int:
        if self._adapter is None:
            return 0
        return self._adapter.page_count

    def extract_page(self, page: int) -> PageExtractionResult:
        """Extract a single page by number (0-indexed)."""
        try:
            if self._adapter is None:
                raise ValueError("Document processing failed")

            converted_page = self._adapter.convert_page(page)
            # Google DocAI outputs pixels after denormalization
            # Use 72 DPI as standard PDF resolution for conversion
            converted_page = self._convert_page(converted_page, CoordinateUnit.PIXELS, dpi=72.0)
            return PageExtractionResult(page=converted_page, success=True)

        except (IndexError, ValueError, AttributeError) as e:
            logger.warning("Failed to extract page %d from Google Document AI result: %s", page, e)
            return PageExtractionResult(
                page=Page(page=page, width=0, height=0, texts=[]),
                success=False,
                error=str(e),
            )

    def get_extractor_metadata(self) -> ExtractorMetadata:
        if self._adapter is None:
            return GoogleDocumentAIAdapter(None, self.processor_name).get_metadata()
        return self._adapter.get_metadata()

    def close(self) -> None:
        if self._client is not None:
            self._client.transport.close()
