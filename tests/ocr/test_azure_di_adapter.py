"""Tests for Azure Document Intelligence adapter using Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from xtra.adapters.azure_di import AzureDocumentIntelligenceAdapter
from xtra.models import ExtractorType


class DocumentWord(BaseModel):
    """Azure DI word model."""

    content: str | None = None
    polygon: list[float] | None = None
    confidence: float | None = None


class DocumentPage(BaseModel):
    """Azure DI page model."""

    width: float | None = None
    height: float | None = None
    words: list[DocumentWord] | None = None


class AnalyzeResult(BaseModel):
    """Azure DI analyze result model."""

    pages: list[DocumentPage] | None = None
    model_id: str | None = None
    api_version: str | None = None


class TestAzureDocumentIntelligenceAdapter:
    """Tests for AzureDocumentIntelligenceAdapter."""

    def test_page_count_with_none_result(self) -> None:
        adapter = AzureDocumentIntelligenceAdapter(None)
        assert adapter.page_count == 0

    def test_page_count_with_empty_pages(self) -> None:
        result = AnalyzeResult(pages=[])
        adapter = AzureDocumentIntelligenceAdapter(result)  # type: ignore[arg-type]
        assert adapter.page_count == 0

    def test_page_count_with_pages(self) -> None:
        result = AnalyzeResult(
            pages=[
                DocumentPage(width=8.5, height=11.0),
                DocumentPage(width=8.5, height=11.0),
            ]
        )
        adapter = AzureDocumentIntelligenceAdapter(result)  # type: ignore[arg-type]
        assert adapter.page_count == 2

    def test_get_metadata_with_none_result(self) -> None:
        adapter = AzureDocumentIntelligenceAdapter(None, model_id="test-model")
        metadata = adapter.get_metadata()

        assert metadata.extractor_type == ExtractorType.AZURE_DI
        assert metadata.extra["model_id"] == "test-model"
        assert metadata.extra["ocr_engine"] == "azure_document_intelligence"

    def test_get_metadata_with_result(self) -> None:
        result = AnalyzeResult(
            pages=[],
            model_id="prebuilt-read",
            api_version="2024-02-29-preview",
        )
        adapter = AzureDocumentIntelligenceAdapter(result, model_id="prebuilt-read")  # type: ignore[arg-type]
        metadata = adapter.get_metadata()

        assert metadata.extractor_type == ExtractorType.AZURE_DI
        assert metadata.extra["model_id"] == "prebuilt-read"
        assert metadata.extra["azure_model_id"] == "prebuilt-read"
        assert metadata.extra["api_version"] == "2024-02-29-preview"

    def test_convert_page_raises_on_none_result(self) -> None:
        adapter = AzureDocumentIntelligenceAdapter(None)
        with pytest.raises(ValueError, match="No analysis result"):
            adapter.convert_page(0)

    def test_convert_page_raises_on_out_of_range(self) -> None:
        result = AnalyzeResult(pages=[])
        adapter = AzureDocumentIntelligenceAdapter(result)  # type: ignore[arg-type]
        with pytest.raises(IndexError, match="out of range"):
            adapter.convert_page(0)

    def test_convert_page_empty_words(self) -> None:
        page = DocumentPage(width=8.5, height=11.0, words=None)
        result = AnalyzeResult(pages=[page])
        adapter = AzureDocumentIntelligenceAdapter(result)  # type: ignore[arg-type]

        converted = adapter.convert_page(0)

        assert converted.page == 0
        assert converted.width == 8.5
        assert converted.height == 11.0
        assert len(converted.texts) == 0

    def test_convert_page_with_words(self) -> None:
        words = [
            DocumentWord(
                content="Hello",
                polygon=[0.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.5],
                confidence=0.95,
            ),
            DocumentWord(
                content="World",
                polygon=[1.5, 0.0, 2.5, 0.0, 2.5, 0.5, 1.5, 0.5],
                confidence=0.88,
            ),
        ]
        page = DocumentPage(width=8.5, height=11.0, words=words)
        result = AnalyzeResult(pages=[page])
        adapter = AzureDocumentIntelligenceAdapter(result)  # type: ignore[arg-type]

        converted = adapter.convert_page(0)

        assert len(converted.texts) == 2
        assert converted.texts[0].text == "Hello"
        assert converted.texts[0].confidence == 0.95
        assert converted.texts[0].bbox.x0 == 0.0
        assert converted.texts[0].bbox.y0 == 0.0
        assert converted.texts[0].bbox.x1 == 1.0
        assert converted.texts[0].bbox.y1 == 0.5

        assert converted.texts[1].text == "World"
        assert converted.texts[1].confidence == 0.88

    def test_convert_page_skips_invalid_words(self) -> None:
        words = [
            # Word with no content - should be skipped
            DocumentWord(
                content=None,
                polygon=[0.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.5],
                confidence=0.9,
            ),
            # Word with no polygon - should be skipped
            DocumentWord(
                content="Hello",
                polygon=None,
                confidence=0.9,
            ),
            # Valid word
            DocumentWord(
                content="World",
                polygon=[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
                confidence=0.9,
            ),
        ]
        page = DocumentPage(width=8.5, height=11.0, words=words)
        result = AnalyzeResult(pages=[page])
        adapter = AzureDocumentIntelligenceAdapter(result)  # type: ignore[arg-type]

        converted = adapter.convert_page(0)

        assert len(converted.texts) == 1
        assert converted.texts[0].text == "World"

    def test_convert_page_handles_none_confidence(self) -> None:
        words = [
            DocumentWord(
                content="NoConf",
                polygon=[0.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.5],
                confidence=None,
            ),
        ]
        page = DocumentPage(width=8.5, height=11.0, words=words)
        result = AnalyzeResult(pages=[page])
        adapter = AzureDocumentIntelligenceAdapter(result)  # type: ignore[arg-type]

        converted = adapter.convert_page(0)

        assert len(converted.texts) == 1
        assert converted.texts[0].text == "NoConf"
        assert converted.texts[0].confidence is None

    def test_convert_page_handles_none_dimensions(self) -> None:
        page = DocumentPage(width=None, height=None, words=[])
        result = AnalyzeResult(pages=[page])
        adapter = AzureDocumentIntelligenceAdapter(result)  # type: ignore[arg-type]

        converted = adapter.convert_page(0)

        assert converted.width == 0.0
        assert converted.height == 0.0

    def test_multiple_pages(self) -> None:
        pages = [
            DocumentPage(
                width=8.5,
                height=11.0,
                words=[
                    DocumentWord(content="Page1", polygon=[0, 0, 1, 0, 1, 1, 0, 1], confidence=0.9)
                ],
            ),
            DocumentPage(
                width=11.0,
                height=8.5,
                words=[
                    DocumentWord(content="Page2", polygon=[0, 0, 2, 0, 2, 1, 0, 1], confidence=0.85)
                ],
            ),
        ]
        result = AnalyzeResult(pages=pages)
        adapter = AzureDocumentIntelligenceAdapter(result)  # type: ignore[arg-type]

        page0 = adapter.convert_page(0)
        page1 = adapter.convert_page(1)

        assert page0.width == 8.5
        assert page0.texts[0].text == "Page1"
        assert page1.width == 11.0
        assert page1.texts[0].text == "Page2"
