"""Tests for Azure Document Intelligence extractor and adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from xtra.adapters.azure_di import AzureDocumentIntelligenceAdapter
from xtra.extractors.azure_di import AzureDocumentIntelligenceExtractor
from xtra.models import ExtractorType


class TestAzureDocumentIntelligenceExtractor:
    def test_get_metadata(self) -> None:
        with patch("xtra.extractors.azure_di.DocumentIntelligenceClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_result = MagicMock()
            mock_result.pages = []
            mock_result.model_id = "prebuilt-read"
            mock_result.api_version = "2024-02-29-preview"

            mock_poller = MagicMock()
            mock_poller.result.return_value = mock_result
            mock_client.begin_analyze_document.return_value = mock_poller

            with patch("builtins.open", MagicMock()):
                extractor = AzureDocumentIntelligenceExtractor(
                    Path("/tmp/test.pdf"),
                    endpoint="https://test.cognitiveservices.azure.com",
                    key="test-key",
                )

            metadata = extractor.get_metadata()

            assert metadata.source_type == ExtractorType.AZURE_DI
            assert metadata.extra["ocr_engine"] == "azure_document_intelligence"
            assert metadata.extra["model_id"] == "prebuilt-read"
            assert metadata.extra["azure_model_id"] == "prebuilt-read"
            assert metadata.extra["api_version"] == "2024-02-29-preview"

    def test_get_page_count_empty(self) -> None:
        with patch("xtra.extractors.azure_di.DocumentIntelligenceClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_result = MagicMock()
            mock_result.pages = None

            mock_poller = MagicMock()
            mock_poller.result.return_value = mock_result
            mock_client.begin_analyze_document.return_value = mock_poller

            with patch("builtins.open", MagicMock()):
                extractor = AzureDocumentIntelligenceExtractor(
                    Path("/tmp/test.pdf"),
                    endpoint="https://test.cognitiveservices.azure.com",
                    key="test-key",
                )

            assert extractor.get_page_count() == 0

    def test_get_page_count_with_pages(self) -> None:
        with patch("xtra.extractors.azure_di.DocumentIntelligenceClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_page1 = MagicMock()
            mock_page2 = MagicMock()
            mock_result = MagicMock()
            mock_result.pages = [mock_page1, mock_page2]

            mock_poller = MagicMock()
            mock_poller.result.return_value = mock_result
            mock_client.begin_analyze_document.return_value = mock_poller

            with patch("builtins.open", MagicMock()):
                extractor = AzureDocumentIntelligenceExtractor(
                    Path("/tmp/test.pdf"),
                    endpoint="https://test.cognitiveservices.azure.com",
                    key="test-key",
                )

            assert extractor.get_page_count() == 2

    def test_extract_page_success(self) -> None:
        with patch("xtra.extractors.azure_di.DocumentIntelligenceClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Create mock word - Azure DI returns coordinates in inches
            mock_word = MagicMock()
            mock_word.content = "Hello"
            # Polygon in inches: ~(0.14, 0.28) to (0.83, 0.56) inches
            mock_word.polygon = [0.14, 0.28, 0.83, 0.28, 0.83, 0.56, 0.14, 0.56]
            mock_word.confidence = 0.95

            # Create mock page - dimensions in inches (8.5x11 letter size)
            mock_page = MagicMock()
            mock_page.width = 8.5
            mock_page.height = 11.0
            mock_page.words = [mock_word]

            mock_result = MagicMock()
            mock_result.pages = [mock_page]
            mock_result.model_id = "prebuilt-read"
            mock_result.api_version = None

            mock_poller = MagicMock()
            mock_poller.result.return_value = mock_result
            mock_client.begin_analyze_document.return_value = mock_poller

            with patch("builtins.open", MagicMock()):
                extractor = AzureDocumentIntelligenceExtractor(
                    Path("/tmp/test.pdf"),
                    endpoint="https://test.cognitiveservices.azure.com",
                    key="test-key",
                )

            result = extractor.extract_page(0)

            assert result.success
            assert result.page.page == 0
            # Page dimensions converted from inches to points (default)
            assert result.page.width == 612.0  # 8.5 * 72
            assert result.page.height == 792.0  # 11.0 * 72
            assert len(result.page.texts) == 1
            assert result.page.texts[0].text == "Hello"
            assert result.page.texts[0].confidence == 0.95
            # Bbox converted from inches to points
            assert abs(result.page.texts[0].bbox.x0 - 10.08) < 0.01  # 0.14 * 72
            assert abs(result.page.texts[0].bbox.y0 - 20.16) < 0.01  # 0.28 * 72
            assert abs(result.page.texts[0].bbox.x1 - 59.76) < 0.01  # 0.83 * 72
            assert abs(result.page.texts[0].bbox.y1 - 40.32) < 0.01  # 0.56 * 72

    def test_extract_page_out_of_range(self) -> None:
        with patch("xtra.extractors.azure_di.DocumentIntelligenceClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_result = MagicMock()
            mock_result.pages = []

            mock_poller = MagicMock()
            mock_poller.result.return_value = mock_result
            mock_client.begin_analyze_document.return_value = mock_poller

            with patch("builtins.open", MagicMock()):
                extractor = AzureDocumentIntelligenceExtractor(
                    Path("/tmp/test.pdf"),
                    endpoint="https://test.cognitiveservices.azure.com",
                    key="test-key",
                )

            result = extractor.extract_page(5)

            assert not result.success
            assert result.error is not None
            assert "out of range" in result.error.lower()

    def test_extract_page_no_result(self) -> None:
        with patch("xtra.extractors.azure_di.DocumentIntelligenceClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_client.begin_analyze_document.side_effect = ValueError("Failed")

            with patch("builtins.open", MagicMock()):
                extractor = AzureDocumentIntelligenceExtractor(
                    Path("/tmp/test.pdf"),
                    endpoint="https://test.cognitiveservices.azure.com",
                    key="test-key",
                )

            result = extractor.extract_page(0)

            assert not result.success
            assert result.error is not None

    def test_close(self) -> None:
        with patch("xtra.extractors.azure_di.DocumentIntelligenceClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_result = MagicMock()
            mock_result.pages = []

            mock_poller = MagicMock()
            mock_poller.result.return_value = mock_result
            mock_client.begin_analyze_document.return_value = mock_poller

            with patch("builtins.open", MagicMock()):
                extractor = AzureDocumentIntelligenceExtractor(
                    Path("/tmp/test.pdf"),
                    endpoint="https://test.cognitiveservices.azure.com",
                    key="test-key",
                )

            extractor.close()
            mock_client.close.assert_called_once()

    def test_custom_model_id(self) -> None:
        with patch("xtra.extractors.azure_di.DocumentIntelligenceClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_result = MagicMock()
            mock_result.pages = []
            mock_result.model_id = "custom-model"
            mock_result.api_version = None

            mock_poller = MagicMock()
            mock_poller.result.return_value = mock_result
            mock_client.begin_analyze_document.return_value = mock_poller

            with patch("builtins.open", MagicMock()):
                extractor = AzureDocumentIntelligenceExtractor(
                    Path("/tmp/test.pdf"),
                    endpoint="https://test.cognitiveservices.azure.com",
                    key="test-key",
                    model_id="custom-model",
                )

            metadata = extractor.get_metadata()
            assert metadata.extra["model_id"] == "custom-model"


class TestAzureDocumentIntelligenceAdapter:
    def test_polygon_to_bbox_and_rotation_horizontal(self) -> None:
        # Horizontal text polygon
        polygon = [10.0, 10.0, 110.0, 10.0, 110.0, 30.0, 10.0, 30.0]
        bbox, rotation = AzureDocumentIntelligenceAdapter._polygon_to_bbox_and_rotation(polygon)

        assert bbox.x0 == 10.0
        assert bbox.y0 == 10.0
        assert bbox.x1 == 110.0
        assert bbox.y1 == 30.0
        assert rotation == 0.0

    def test_polygon_to_bbox_and_rotation_short_polygon(self) -> None:
        # Too short polygon
        polygon = [10.0, 10.0]
        bbox, rotation = AzureDocumentIntelligenceAdapter._polygon_to_bbox_and_rotation(polygon)

        assert bbox.x0 == 0
        assert bbox.y0 == 0
        assert bbox.x1 == 0
        assert bbox.y1 == 0
        assert rotation == 0.0

    def test_convert_page_to_blocks_empty_words(self) -> None:
        adapter = AzureDocumentIntelligenceAdapter(None)

        mock_page = MagicMock()
        mock_page.words = None

        blocks = adapter._convert_page_to_blocks(mock_page)
        assert blocks == []

    def test_convert_page_to_blocks_skip_invalid_words(self) -> None:
        adapter = AzureDocumentIntelligenceAdapter(None)

        # Word with no content
        mock_word1 = MagicMock()
        mock_word1.content = None
        mock_word1.polygon = [0, 0, 10, 0, 10, 10, 0, 10]

        # Word with no polygon
        mock_word2 = MagicMock()
        mock_word2.content = "Hello"
        mock_word2.polygon = None

        # Valid word
        mock_word3 = MagicMock()
        mock_word3.content = "World"
        mock_word3.polygon = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]
        mock_word3.confidence = 0.9

        mock_page = MagicMock()
        mock_page.words = [mock_word1, mock_word2, mock_word3]

        blocks = adapter._convert_page_to_blocks(mock_page)
        assert len(blocks) == 1
        assert blocks[0].text == "World"

    def test_page_count_with_none_result(self) -> None:
        adapter = AzureDocumentIntelligenceAdapter(None)
        assert adapter.page_count == 0

    def test_page_count_with_result(self) -> None:
        mock_result = MagicMock()
        mock_result.pages = [MagicMock(), MagicMock()]
        adapter = AzureDocumentIntelligenceAdapter(mock_result)
        assert adapter.page_count == 2

    def test_get_metadata_with_none_result(self) -> None:
        adapter = AzureDocumentIntelligenceAdapter(None, model_id="test-model")
        metadata = adapter.get_metadata()

        assert metadata.source_type == ExtractorType.AZURE_DI
        assert metadata.extra["model_id"] == "test-model"
        assert metadata.extra["ocr_engine"] == "azure_document_intelligence"

    def test_convert_page_raises_on_none_result(self) -> None:
        adapter = AzureDocumentIntelligenceAdapter(None)
        try:
            adapter.convert_page(0)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "No analysis result" in str(e)

    def test_convert_page_raises_on_out_of_range(self) -> None:
        mock_result = MagicMock()
        mock_result.pages = []
        adapter = AzureDocumentIntelligenceAdapter(mock_result)
        try:
            adapter.convert_page(0)
            assert False, "Expected IndexError"
        except IndexError as e:
            assert "out of range" in str(e)
