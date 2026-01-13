"""Integration tests for all extractors.

These tests use real files and services (no mocking).
Azure/Google tests require credentials in environment variables - see .env.example.

Guidelines:
- Local OCR tests (EasyOCR, Tesseract, PaddleOCR) run unconditionally.
- Cloud tests (Azure, Google) are skipped if credentials are not set.
- Use 2-letter ISO 639-1 language codes (e.g., "en", "fr", "de") for all extractors.
  Tesseract converts internally to its 3-letter format.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from xtra.coordinates import CoordinateConverter
from xtra.extractors import PdfExtractor
from xtra.extractors.azure_di import AzureDocumentIntelligenceExtractor
from xtra.extractors.google_docai import GoogleDocumentAIExtractor
from xtra.extractors.easy_ocr import EasyOcrExtractor
from xtra.extractors.tesseract_ocr import TesseractOcrExtractor
from xtra.extractors.paddle_ocr import PaddleOcrExtractor
from xtra.extractors.factory import create_extractor
from xtra.models import CoordinateUnit, Document, ExtractorType, Page

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
        assert doc.metadata.source_type == ExtractorType.PDF

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


class TestEasyOcrExtractorIntegration:
    """Integration tests for EasyOcrExtractor using real image files."""

    def test_extract_image_with_text(self) -> None:
        """Extract text from an image using EasyOCR."""
        with EasyOcrExtractor(TEST_DATA_DIR / "test_image.png", languages=["en"]) as extractor:
            doc = extractor.extract()

        assert doc.path == TEST_DATA_DIR / "test_image.png"
        assert len(doc.pages) == 1
        assert doc.metadata is not None
        assert doc.metadata.source_type == ExtractorType.EASYOCR
        assert doc.metadata.extra["ocr_engine"] == "easyocr"

        # Verify OCR detected text
        page = doc.pages[0]
        assert page.width > 0
        assert page.height > 0
        assert len(page.texts) > 0

        # Note: OCR accuracy varies between environments and model versions.
        # We verify text was extracted; content accuracy tested manually.
        all_text = " ".join(t.text for t in page.texts).lower()
        assert len(all_text) > 0, "Expected OCR to extract some text from image"

        # Verify confidence scores are present
        for text in page.texts:
            assert text.confidence is not None
            assert 0.0 <= text.confidence <= 1.0


class TestEasyOcrExtractorWithPdfIntegration:
    """Integration tests for EasyOcrExtractor with PDF files (unified extractor)."""

    def test_extract_pdf_via_easyocr(self) -> None:
        """Extract text from a PDF by converting to images and running EasyOCR."""
        with EasyOcrExtractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf", languages=["en"], dpi=150
        ) as extractor:
            doc = extractor.extract()

        assert doc.path == TEST_DATA_DIR / "test_pdf_2p_text.pdf"
        assert len(doc.pages) == 2
        assert doc.metadata is not None
        assert doc.metadata.source_type == ExtractorType.EASYOCR
        assert doc.metadata.extra["ocr_engine"] == "easyocr"
        assert doc.metadata.extra["dpi"] == 150

        # Verify pages have content
        for page in doc.pages:
            assert page.width > 0
            assert page.height > 0
            assert len(page.texts) > 0

        # Note: OCR accuracy varies between environments and model versions.
        # We verify text was extracted; content accuracy tested manually.
        page1_text = " ".join(t.text for t in doc.pages[0].texts).lower()
        assert len(page1_text) > 0, "Expected OCR to extract some text from page 1"

        page2_text = " ".join(t.text for t in doc.pages[1].texts).lower()
        assert len(page2_text) > 0, "Expected OCR to extract some text from page 2"

        # Verify confidence scores
        for page in doc.pages:
            for text in page.texts:
                assert text.confidence is not None
                assert 0.0 <= text.confidence <= 1.0


class TestAzureDocumentIntelligenceExtractorIntegration:
    """Integration tests for Azure Document Intelligence extractor.

    These tests require Azure credentials in environment variables:
    - XTRA_AZURE_DI_ENDPOINT: Azure Document Intelligence endpoint URL
    - XTRA_AZURE_DI_KEY: Azure Document Intelligence API key

    See .env.example for details.
    """

    @pytest.fixture
    def azure_credentials(self) -> tuple[str, str]:
        """Get Azure credentials from environment variables. Skip if not set."""
        endpoint = os.environ.get("XTRA_AZURE_DI_ENDPOINT", "")
        key = os.environ.get("XTRA_AZURE_DI_KEY", "")

        if not endpoint or not key:
            pytest.skip("Azure credentials not configured (XTRA_AZURE_DI_ENDPOINT/KEY)")

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
        assert doc.metadata.source_type == ExtractorType.AZURE_DI
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

        # Note: OCR accuracy varies between model versions.
        # We verify text was extracted; content accuracy tested manually.
        page1_text = " ".join(t.text for t in doc.pages[0].texts).lower()
        assert len(page1_text) > 0, "Expected OCR to extract some text from page 1"

        page2_text = " ".join(t.text for t in doc.pages[1].texts).lower()
        assert len(page2_text) > 0, "Expected OCR to extract some text from page 2"


class TestGoogleDocumentAIExtractorIntegration:
    """Integration tests for Google Document AI extractor.

    These tests require Google Cloud credentials:
    - XTRA_GOOGLE_DOCAI_PROCESSOR_NAME: Full processor resource name
    - XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH: Path to service account JSON file

    See .env.example for details.
    """

    @pytest.fixture
    def google_credentials(self) -> tuple[str, str]:
        """Get Google credentials from environment variables. Skip if not set."""
        processor_name = os.environ.get("XTRA_GOOGLE_DOCAI_PROCESSOR_NAME", "")
        credentials_path = os.environ.get("XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH", "")

        if not processor_name or not credentials_path:
            pytest.skip("Google credentials not configured (XTRA_GOOGLE_DOCAI_*)")

        return processor_name, credentials_path

    def test_extract_pdf_with_google_docai(self, google_credentials: tuple[str, str]) -> None:
        """Extract text from a PDF using Google Document AI."""
        processor_name, credentials_path = google_credentials

        with GoogleDocumentAIExtractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            processor_name=processor_name,
            credentials_path=credentials_path,
        ) as extractor:
            doc = extractor.extract()

        assert doc.path == TEST_DATA_DIR / "test_pdf_2p_text.pdf"
        assert len(doc.pages) == 2
        assert doc.metadata is not None
        assert doc.metadata.source_type == ExtractorType.GOOGLE_DOCAI
        assert doc.metadata.extra["ocr_engine"] == "google_document_ai"

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

        # Note: OCR accuracy varies between model versions.
        # We verify text was extracted; content accuracy tested manually.
        page1_text = " ".join(t.text for t in doc.pages[0].texts).lower()
        assert len(page1_text) > 0, "Expected OCR to extract some text from page 1"

        page2_text = " ".join(t.text for t in doc.pages[1].texts).lower()
        assert len(page2_text) > 0, "Expected OCR to extract some text from page 2"


class TestTesseractOcrExtractorIntegration:
    """Integration tests for TesseractOcrExtractor using real image files.

    Requires Tesseract to be installed on the system:
    - macOS: brew install tesseract
    - Ubuntu: apt-get install tesseract-ocr
    - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
    """

    def test_extract_image_with_tesseract(self) -> None:
        """Extract text from an image using Tesseract OCR."""
        with TesseractOcrExtractor(TEST_DATA_DIR / "test_image.png", languages=["en"]) as extractor:
            doc = extractor.extract()

        assert doc.path == TEST_DATA_DIR / "test_image.png"
        assert len(doc.pages) == 1
        assert doc.metadata is not None
        assert doc.metadata.source_type == ExtractorType.TESSERACT
        assert doc.metadata.extra["ocr_engine"] == "tesseract"

        # Verify OCR detected text
        page = doc.pages[0]
        assert page.width > 0
        assert page.height > 0
        assert len(page.texts) > 0

        # Note: OCR accuracy varies between environments and model versions.
        # We verify text was extracted; content accuracy tested manually.
        all_text = " ".join(t.text for t in page.texts).lower()
        assert len(all_text) > 0, "Expected OCR to extract some text from image"

        # Verify confidence scores are present
        for text in page.texts:
            assert text.confidence is not None
            assert 0.0 <= text.confidence <= 1.0


class TestTesseractOcrExtractorWithPdfIntegration:
    """Integration tests for TesseractOcrExtractor with PDF files (unified extractor).

    Requires Tesseract to be installed on the system.
    """

    def test_extract_pdf_via_tesseract(self) -> None:
        """Extract text from a PDF by converting to images and running Tesseract OCR."""
        with TesseractOcrExtractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf", languages=["en"], dpi=150
        ) as extractor:
            doc = extractor.extract()

        assert doc.path == TEST_DATA_DIR / "test_pdf_2p_text.pdf"
        assert len(doc.pages) == 2
        assert doc.metadata is not None
        assert doc.metadata.source_type == ExtractorType.TESSERACT
        assert doc.metadata.extra["ocr_engine"] == "tesseract"
        assert doc.metadata.extra["dpi"] == 150

        # Verify pages have content
        for page in doc.pages:
            assert page.width > 0
            assert page.height > 0
            assert len(page.texts) > 0

        # Note: OCR accuracy varies between environments and model versions.
        # We verify text was extracted; content accuracy tested manually.
        page1_text = " ".join(t.text for t in doc.pages[0].texts).lower()
        assert len(page1_text) > 0, "Expected OCR to extract some text from page 1"

        page2_text = " ".join(t.text for t in doc.pages[1].texts).lower()
        assert len(page2_text) > 0, "Expected OCR to extract some text from page 2"

        # Verify confidence scores
        for page in doc.pages:
            for text in page.texts:
                assert text.confidence is not None
                assert 0.0 <= text.confidence <= 1.0


class TestPaddleOcrExtractorIntegration:
    """Integration tests for PaddleOcrExtractor using real image files.

    Requires PaddleOCR and PaddlePaddle to be installed.
    """

    def test_extract_image_with_paddle(self) -> None:
        """Extract text from an image using PaddleOCR."""
        with PaddleOcrExtractor(TEST_DATA_DIR / "test_image.png", lang="en") as extractor:
            doc = extractor.extract()

        assert doc.path == TEST_DATA_DIR / "test_image.png"
        assert len(doc.pages) == 1
        assert doc.metadata is not None
        assert doc.metadata.source_type == ExtractorType.PADDLE
        assert doc.metadata.extra["ocr_engine"] == "paddleocr"

        # Verify OCR detected text
        page = doc.pages[0]
        assert page.width > 0
        assert page.height > 0
        assert len(page.texts) > 0

        # Note: OCR accuracy varies between environments and model versions.
        # We verify text was extracted; content accuracy tested manually.
        all_text = " ".join(t.text for t in page.texts).lower()
        assert len(all_text) > 0, "Expected OCR to extract some text from image"

        # Verify confidence scores are present
        for text in page.texts:
            assert text.confidence is not None
            assert 0.0 <= text.confidence <= 1.0


class TestPaddleOcrExtractorWithPdfIntegration:
    """Integration tests for PaddleOcrExtractor with PDF files (unified extractor).

    Requires PaddleOCR and PaddlePaddle to be installed.
    """

    def test_extract_pdf_via_paddle(self) -> None:
        """Extract text from a PDF by converting to images and running PaddleOCR."""
        with PaddleOcrExtractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf", lang="en", dpi=150
        ) as extractor:
            doc = extractor.extract()

        assert doc.path == TEST_DATA_DIR / "test_pdf_2p_text.pdf"
        assert len(doc.pages) == 2
        assert doc.metadata is not None
        assert doc.metadata.source_type == ExtractorType.PADDLE
        assert doc.metadata.extra["ocr_engine"] == "paddleocr"
        assert doc.metadata.extra["dpi"] == 150

        # Verify pages have content
        for page in doc.pages:
            assert page.width > 0
            assert page.height > 0
            assert len(page.texts) > 0

        # Note: OCR accuracy varies between environments and model versions.
        # We verify text was extracted; content accuracy tested manually.
        page1_text = " ".join(t.text for t in doc.pages[0].texts).lower()
        assert len(page1_text) > 0, "Expected OCR to extract some text from page 1"

        page2_text = " ".join(t.text for t in doc.pages[1].texts).lower()
        assert len(page2_text) > 0, "Expected OCR to extract some text from page 2"

        # Verify confidence scores
        for page in doc.pages:
            for text in page.texts:
                assert text.confidence is not None
                assert 0.0 <= text.confidence <= 1.0


class TestCoordinateUnitConsistency:
    """Integration tests to verify coordinate transformations are consistent across extractors.

    For each extractor, extracts the same document with different output_unit settings,
    then converts all results to POINTS and verifies coordinates match within tolerance.
    """

    DPI = 150  # Fixed DPI for consistent pixel conversions
    TOLERANCE = 0.01  # Tolerance for floating point comparison

    def _convert_to_points(
        self,
        doc: Document,
        source_unit: CoordinateUnit,
        reference_pages: list[Page] | None = None,
    ) -> Document:
        """Convert all page coordinates to POINTS.

        Args:
            doc: Document to convert.
            source_unit: The unit system of the document coordinates.
            reference_pages: Reference pages with dimensions in POINTS.
                Required when source_unit is NORMALIZED (since normalized
                pages have dimensions 1.0 x 1.0, losing original size info).
        """
        if source_unit == CoordinateUnit.POINTS:
            return doc

        converted_pages = []
        for idx, page in enumerate(doc.pages):
            # For NORMALIZED source, use reference dimensions (in points)
            if source_unit == CoordinateUnit.NORMALIZED and reference_pages:
                ref_page = reference_pages[idx]
                page_width_pts = ref_page.width
                page_height_pts = ref_page.height
            else:
                page_width_pts = page.width
                page_height_pts = page.height

            converter = CoordinateConverter(
                source_unit=source_unit,
                page_width=page_width_pts,
                page_height=page_height_pts,
                dpi=self.DPI,
            )
            converted_pages.append(converter.convert_page(page, CoordinateUnit.POINTS))

        return Document(path=doc.path, pages=converted_pages, metadata=doc.metadata)

    def _assert_documents_similar(
        self, doc1: Document, doc2: Document, unit1: CoordinateUnit, unit2: CoordinateUnit
    ) -> None:
        """Assert two documents have similar coordinates after conversion to points."""
        assert len(doc1.pages) == len(doc2.pages), (
            f"Page count mismatch: {unit1}={len(doc1.pages)}, {unit2}={len(doc2.pages)}"
        )

        for page_idx, (p1, p2) in enumerate(zip(doc1.pages, doc2.pages)):
            # Compare page dimensions
            assert abs(p1.width - p2.width) < self.TOLERANCE, (
                f"Page {page_idx} width mismatch: {unit1}={p1.width}, {unit2}={p2.width}"
            )
            assert abs(p1.height - p2.height) < self.TOLERANCE, (
                f"Page {page_idx} height mismatch: {unit1}={p1.height}, {unit2}={p2.height}"
            )

            # Compare text blocks (same extractor should produce same number of blocks)
            assert len(p1.texts) == len(p2.texts), (
                f"Page {page_idx} text count mismatch: "
                f"{unit1}={len(p1.texts)}, {unit2}={len(p2.texts)}"
            )

            for text_idx, (t1, t2) in enumerate(zip(p1.texts, p2.texts)):
                assert t1.text == t2.text, f"Page {page_idx} text {text_idx} content mismatch"
                assert abs(t1.bbox.x0 - t2.bbox.x0) < self.TOLERANCE, (
                    f"Page {page_idx} text {text_idx} bbox.x0 mismatch: "
                    f"{unit1}={t1.bbox.x0}, {unit2}={t2.bbox.x0}"
                )
                assert abs(t1.bbox.y0 - t2.bbox.y0) < self.TOLERANCE, (
                    f"Page {page_idx} text {text_idx} bbox.y0 mismatch: "
                    f"{unit1}={t1.bbox.y0}, {unit2}={t2.bbox.y0}"
                )
                assert abs(t1.bbox.x1 - t2.bbox.x1) < self.TOLERANCE, (
                    f"Page {page_idx} text {text_idx} bbox.x1 mismatch: "
                    f"{unit1}={t1.bbox.x1}, {unit2}={t2.bbox.x1}"
                )
                assert abs(t1.bbox.y1 - t2.bbox.y1) < self.TOLERANCE, (
                    f"Page {page_idx} text {text_idx} bbox.y1 mismatch: "
                    f"{unit1}={t1.bbox.y1}, {unit2}={t2.bbox.y1}"
                )

    def _test_extractor_coordinate_consistency(
        self, extractor_type: ExtractorType, include_pixels: bool = True
    ) -> None:
        """Test coordinate unit consistency for a given extractor type.

        Args:
            extractor_type: The extractor type to test.
            include_pixels: Whether to test PIXELS unit. Set to False for extractors
                that don't have an inherent DPI (like PDF).
        """
        units = [
            CoordinateUnit.POINTS,
            CoordinateUnit.INCHES,
            CoordinateUnit.NORMALIZED,
        ]
        if include_pixels:
            units.append(CoordinateUnit.PIXELS)

        # Extract document with each output unit
        results: dict[CoordinateUnit, Document] = {}
        for unit in units:
            with create_extractor(
                TEST_DATA_DIR / "test_pdf_2p_text.pdf",
                extractor_type,
                dpi=self.DPI,
                output_unit=unit,
            ) as extractor:
                results[unit] = extractor.extract()

        # Get reference pages (POINTS) for converting NORMALIZED back
        reference_pages = results[CoordinateUnit.POINTS].pages

        # Convert all to POINTS
        converted: dict[CoordinateUnit, Document] = {}
        for unit, doc in results.items():
            converted[unit] = self._convert_to_points(doc, unit, reference_pages)

        # Compare all against POINTS baseline
        baseline_unit = CoordinateUnit.POINTS
        baseline = converted[baseline_unit]
        for unit in units:
            if unit != baseline_unit:
                self._assert_documents_similar(baseline, converted[unit], baseline_unit, unit)

    def test_pdf_coordinate_consistency(self) -> None:
        """Test coordinate unit consistency for PDF extractor.

        Note: PDF extractor doesn't support PIXELS output as PDFs don't have an inherent DPI.
        """
        self._test_extractor_coordinate_consistency(ExtractorType.PDF, include_pixels=False)

    def test_easyocr_coordinate_consistency(self) -> None:
        """Test coordinate unit consistency for EasyOCR extractor."""
        self._test_extractor_coordinate_consistency(ExtractorType.EASYOCR)

    def test_tesseract_coordinate_consistency(self) -> None:
        """Test coordinate unit consistency for Tesseract extractor."""
        self._test_extractor_coordinate_consistency(ExtractorType.TESSERACT)

    def test_paddle_coordinate_consistency(self) -> None:
        """Test coordinate unit consistency for PaddleOCR extractor."""
        self._test_extractor_coordinate_consistency(ExtractorType.PADDLE)

    @pytest.fixture
    def azure_credentials(self) -> tuple[str, str]:
        """Get Azure credentials from environment variables. Skip if not set."""
        endpoint = os.environ.get("XTRA_AZURE_DI_ENDPOINT", "")
        key = os.environ.get("XTRA_AZURE_DI_KEY", "")

        if not endpoint or not key:
            pytest.skip("Azure credentials not configured (XTRA_AZURE_DI_ENDPOINT/KEY)")

        return endpoint, key

    @pytest.fixture
    def google_credentials(self) -> tuple[str, str]:
        """Get Google credentials from environment variables. Skip if not set."""
        processor_name = os.environ.get("XTRA_GOOGLE_DOCAI_PROCESSOR_NAME", "")
        credentials_path = os.environ.get("XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH", "")

        if not processor_name or not credentials_path:
            pytest.skip("Google credentials not configured (XTRA_GOOGLE_DOCAI_*)")

        return processor_name, credentials_path

    def test_azure_coordinate_consistency(self, azure_credentials: tuple[str, str]) -> None:
        """Test coordinate unit consistency for Azure Document Intelligence extractor."""
        self._test_extractor_coordinate_consistency(ExtractorType.AZURE_DI)

    def test_google_coordinate_consistency(self, google_credentials: tuple[str, str]) -> None:
        """Test coordinate unit consistency for Google Document AI extractor."""
        self._test_extractor_coordinate_consistency(ExtractorType.GOOGLE_DOCAI)
