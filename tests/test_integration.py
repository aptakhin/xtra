"""Integration tests for all extractors.

These tests use real files and services (no mocking).
Azure tests require credentials in environment variables - see .env.example.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from xtra.extractors import PdfExtractor
from xtra.extractors.azure_di import AzureDocumentIntelligenceExtractor
from xtra.extractors.ocr import OcrExtractor, PdfToImageOcrExtractor
from xtra.models import SourceType

TEST_DATA_DIR = Path(__file__).parent / "data"


class TestPdfExtractorIntegration:
    """Integration tests for PdfExtractor using real PDF files."""

    def test_extract_full_document(self) -> None:
        """Extract a complete PDF document and verify structure and content."""
        with PdfExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf") as extractor:
            doc = extractor.extract()

        assert doc.path == TEST_DATA_DIR / "test_pdf_2p_text.pdf"
        assert len(doc.pages) == 2
        assert doc.metadata is not None
        assert doc.metadata.source_type == SourceType.PDF

        # Verify page 1 content (exact text checks)
        page1 = doc.pages[0]
        page1_texts = [t.text for t in page1.texts]
        assert "First page. First text" in page1_texts
        assert "First page. Second text" in page1_texts
        assert "First page. Fourth text" in page1_texts
        assert len(page1.texts) == 3

        # Verify page 2 content (exact text checks)
        page2 = doc.pages[1]
        assert len(page2.texts) == 1
        assert page2.texts[0].text == "Second page. Third text"

        # Verify page structure
        for page in doc.pages:
            assert page.width > 0
            assert page.height > 0

            # Verify text blocks have valid bounding boxes
            for text in page.texts:
                assert text.bbox is not None
                assert text.bbox.x0 < text.bbox.x1
                assert text.bbox.y0 < text.bbox.y1


class TestOcrExtractorIntegration:
    """Integration tests for OcrExtractor using real image files."""

    def test_extract_image_with_text(self) -> None:
        """Extract text from an image using EasyOCR."""
        with OcrExtractor(TEST_DATA_DIR / "test_image.png", languages=["en"]) as extractor:
            doc = extractor.extract()

        assert doc.path == TEST_DATA_DIR / "test_image.png"
        assert len(doc.pages) == 1
        assert doc.metadata is not None
        assert doc.metadata.source_type == SourceType.OCR
        assert doc.metadata.extra["ocr_engine"] == "easyocr"

        # Verify OCR detected text
        page = doc.pages[0]
        assert page.width > 0
        assert page.height > 0
        assert len(page.texts) > 0

        # The image contains "Hello Integration Test" - check key words detected
        all_text = " ".join(t.text for t in page.texts).lower()
        detected_keywords = sum(1 for kw in ["hello", "integration", "test"] if kw in all_text)
        assert detected_keywords >= 2, f"Expected at least 2 keywords, got: {all_text}"

        # Verify confidence scores are present
        for text in page.texts:
            assert text.confidence is not None
            assert 0.0 <= text.confidence <= 1.0


class TestPdfToImageOcrExtractorIntegration:
    """Integration tests for PdfToImageOcrExtractor using real PDF files."""

    def test_extract_pdf_via_ocr(self) -> None:
        """Extract text from a PDF by converting to images and running OCR."""
        with PdfToImageOcrExtractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf", languages=["en"], dpi=150
        ) as extractor:
            doc = extractor.extract()

        assert doc.path == TEST_DATA_DIR / "test_pdf_2p_text.pdf"
        assert len(doc.pages) == 2
        assert doc.metadata is not None
        assert doc.metadata.source_type == SourceType.PDF_OCR
        assert doc.metadata.extra["ocr_engine"] == "easyocr"
        assert doc.metadata.extra["dpi"] == 150

        # Verify pages have content
        for page in doc.pages:
            assert page.width > 0
            assert page.height > 0
            assert len(page.texts) > 0

        # Check page 1 detected key words from "First page. First/Second/Fourth text"
        page1_text = " ".join(t.text for t in doc.pages[0].texts).lower()
        assert "first" in page1_text or "page" in page1_text or "text" in page1_text

        # Check page 2 detected key words from "Second page. Third text"
        page2_text = " ".join(t.text for t in doc.pages[1].texts).lower()
        assert "second" in page2_text or "third" in page2_text or "page" in page2_text

        # Verify confidence scores
        for page in doc.pages:
            for text in page.texts:
                assert text.confidence is not None
                assert 0.0 <= text.confidence <= 1.0


class TestAzureDocumentIntelligenceExtractorIntegration:
    """Integration tests for Azure Document Intelligence extractor.

    These tests require Azure credentials in environment variables:
    - AZURE_DI_ENDPOINT: Azure Document Intelligence endpoint URL
    - AZURE_DI_KEY: Azure Document Intelligence API key

    See .env.example for details.
    """

    @pytest.fixture
    def azure_credentials(self) -> tuple[str, str]:
        """Get Azure credentials from environment or skip test."""
        endpoint = os.environ.get("AZURE_DI_ENDPOINT", "")
        key = os.environ.get("AZURE_DI_KEY", "")

        if not endpoint or not key:
            pytest.skip("Azure credentials not configured (AZURE_DI_ENDPOINT, AZURE_DI_KEY)")

        return endpoint, key

    def test_extract_pdf_with_azure(self, azure_credentials: tuple[str, str]) -> None:
        """Extract text from a PDF using Azure Document Intelligence."""
        endpoint, key = azure_credentials

        with AzureDocumentIntelligenceExtractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            endpoint=endpoint,
            key=key,
        ) as extractor:
            doc = extractor.extract()

        assert doc.path == TEST_DATA_DIR / "test_pdf_2p_text.pdf"
        assert len(doc.pages) == 2
        assert doc.metadata is not None
        assert doc.metadata.source_type == SourceType.AZURE_DI
        assert doc.metadata.extra["ocr_engine"] == "azure_document_intelligence"

        # Verify pages have content
        for page in doc.pages:
            assert page.width > 0
            assert page.height > 0
            assert len(page.texts) > 0

            # Verify text blocks have valid structure
            for text in page.texts:
                assert text.bbox is not None
                assert text.confidence is not None
                assert 0.0 <= text.confidence <= 1.0

        # Check page 1 detected key words (fuzzy - Azure models may change)
        page1_text = " ".join(t.text for t in doc.pages[0].texts).lower()
        assert "first" in page1_text or "page" in page1_text or "text" in page1_text

        # Check page 2 detected key words
        page2_text = " ".join(t.text for t in doc.pages[1].texts).lower()
        assert "second" in page2_text or "third" in page2_text or "page" in page2_text
