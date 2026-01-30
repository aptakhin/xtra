"""Tests for Google Document AI adapter."""

from __future__ import annotations

from pydantic import BaseModel

from xtra.adapters.google_docai import GoogleDocumentAIAdapter
from xtra.models import ExtractorType

# Pydantic models representing Google Document AI response structure


class Vertex(BaseModel):
    x: float
    y: float


class BoundingPoly(BaseModel):
    normalized_vertices: list[Vertex]


class TextSegment(BaseModel):
    start_index: int
    end_index: int


class TextAnchor(BaseModel):
    text_segments: list[TextSegment]


class Layout(BaseModel):
    bounding_poly: BoundingPoly | None = None
    confidence: float = 0.0
    text_anchor: TextAnchor | None = None


class Token(BaseModel):
    layout: Layout | None = None


class Dimension(BaseModel):
    width: float
    height: float


class Page(BaseModel):
    dimension: Dimension
    tokens: list[Token] = []


class Document(BaseModel):
    text: str = ""
    pages: list[Page] = []


class TestGoogleDocumentAIAdapter:
    def test_normalized_vertices_to_bbox_and_rotation_horizontal(self) -> None:
        vertices = [
            Vertex(x=0.1, y=0.2),
            Vertex(x=0.5, y=0.2),
            Vertex(x=0.5, y=0.3),
            Vertex(x=0.1, y=0.3),
        ]

        bbox, rotation = GoogleDocumentAIAdapter._vertices_to_bbox_and_rotation(
            vertices, page_width=612.0, page_height=792.0
        )

        assert abs(bbox.x0 - 61.2) < 0.1  # 0.1 * 612
        assert abs(bbox.y0 - 158.4) < 0.1  # 0.2 * 792
        assert abs(bbox.x1 - 306.0) < 0.1  # 0.5 * 612
        assert abs(bbox.y1 - 237.6) < 0.1  # 0.3 * 792
        assert rotation == 0.0

    def test_vertices_to_bbox_short_vertices(self) -> None:
        vertices = [Vertex(x=0.1, y=0.1)]
        bbox, rotation = GoogleDocumentAIAdapter._vertices_to_bbox_and_rotation(
            vertices, page_width=612.0, page_height=792.0
        )

        assert bbox.x0 == 0
        assert bbox.y0 == 0
        assert bbox.x1 == 0
        assert bbox.y1 == 0
        assert rotation == 0.0

    def test_page_count_with_none_result(self) -> None:
        adapter = GoogleDocumentAIAdapter(None, "test-processor")
        assert adapter.page_count == 0

    def test_page_count_with_result(self) -> None:
        document = Document(
            pages=[
                Page(dimension=Dimension(width=612, height=792)),
                Page(dimension=Dimension(width=612, height=792)),
            ]
        )
        adapter = GoogleDocumentAIAdapter(document, "test-processor")  # type: ignore[arg-type]
        assert adapter.page_count == 2

    def test_get_metadata_with_none_result(self) -> None:
        adapter = GoogleDocumentAIAdapter(None, "test-processor")
        metadata = adapter.get_metadata()

        assert metadata.extractor_type == ExtractorType.GOOGLE_DOCAI
        assert metadata.extra["processor_name"] == "test-processor"
        assert metadata.extra["ocr_engine"] == "google_document_ai"

    def test_convert_page_raises_on_none_result(self) -> None:
        adapter = GoogleDocumentAIAdapter(None, "test-processor")
        try:
            adapter.convert_page(0)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "No analysis result" in str(e)

    def test_convert_page_raises_on_out_of_range(self) -> None:
        document = Document(pages=[])
        adapter = GoogleDocumentAIAdapter(document, "test-processor")  # type: ignore[arg-type]
        try:
            adapter.convert_page(0)
            assert False, "Expected IndexError"
        except IndexError as e:
            assert "out of range" in str(e)

    def test_convert_page_to_blocks_empty_tokens(self) -> None:
        document = Document(text="")
        adapter = GoogleDocumentAIAdapter(document, "test-processor")  # type: ignore[arg-type]

        page = Page(dimension=Dimension(width=612, height=792), tokens=[])
        blocks = adapter._convert_page_to_blocks(page)
        assert blocks == []

    def test_convert_page_to_blocks_skip_invalid_tokens(self) -> None:
        document = Document(text="Hello World")
        adapter = GoogleDocumentAIAdapter(document, "test-processor")  # type: ignore[arg-type]

        # Token with no layout
        token1 = Token(layout=None)

        # Token with no bounding_poly
        token2 = Token(layout=Layout(bounding_poly=None))

        # Valid token
        token3 = Token(
            layout=Layout(
                bounding_poly=BoundingPoly(
                    normalized_vertices=[
                        Vertex(x=0.0, y=0.0),
                        Vertex(x=0.1, y=0.0),
                        Vertex(x=0.1, y=0.1),
                        Vertex(x=0.0, y=0.1),
                    ]
                ),
                confidence=0.9,
                text_anchor=TextAnchor(text_segments=[TextSegment(start_index=6, end_index=11)]),
            )
        )

        page = Page(dimension=Dimension(width=612, height=792), tokens=[token1, token2, token3])
        blocks = adapter._convert_page_to_blocks(page)

        assert len(blocks) == 1
        assert blocks[0].text == "World"

    def test_convert_page_success(self) -> None:
        document = Document(
            text="Hello",
            pages=[
                Page(
                    dimension=Dimension(width=612, height=792),
                    tokens=[
                        Token(
                            layout=Layout(
                                bounding_poly=BoundingPoly(
                                    normalized_vertices=[
                                        Vertex(x=0.1, y=0.2),
                                        Vertex(x=0.5, y=0.2),
                                        Vertex(x=0.5, y=0.3),
                                        Vertex(x=0.1, y=0.3),
                                    ]
                                ),
                                confidence=0.95,
                                text_anchor=TextAnchor(
                                    text_segments=[TextSegment(start_index=0, end_index=5)]
                                ),
                            )
                        )
                    ],
                )
            ],
        )
        adapter = GoogleDocumentAIAdapter(document, "test-processor")  # type: ignore[arg-type]

        page = adapter.convert_page(0)

        assert page.page == 0
        assert page.width == 612.0
        assert page.height == 792.0
        assert len(page.texts) == 1
        assert page.texts[0].text == "Hello"
        assert page.texts[0].confidence == 0.95
