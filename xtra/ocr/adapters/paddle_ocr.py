"""Adapter for converting PaddleOCR results to internal schema."""

from __future__ import annotations

from html.parser import HTMLParser

from pydantic import BaseModel, field_validator

from xtra.models import Table, TableCell, TextBlock
from xtra.utils.geometry import polygon_to_bbox_and_rotation

POLYGON_POINTS = 4
COORDINATES_PER_POINT = 2

# PaddleOCR 3.x introduced breaking API changes
PADDLEOCR_V3_MAJOR = 3


class PaddleOCRDetection(BaseModel):
    """A single text detection from PaddleOCR.

    PaddleOCR returns detections as [bbox, (text, confidence)] where
    bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] representing 4 corners.
    """

    polygon: list[list[float]]
    text: str
    confidence: float

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != POLYGON_POINTS:
            raise ValueError(f"Polygon must have {POLYGON_POINTS} points, got {len(v)}")
        for point in v:
            if len(point) != COORDINATES_PER_POINT:
                raise ValueError(
                    f"Each point must have {COORDINATES_PER_POINT} coordinates, got {len(point)}"
                )
        return v

    @classmethod
    def from_paddle_format(
        cls, item: tuple[list[list[float]], tuple[str, float]]
    ) -> PaddleOCRDetection:
        """Create from PaddleOCR's native format: [bbox, (text, confidence)]."""
        bbox, (text, confidence) = item
        return cls(polygon=bbox, text=text, confidence=confidence)


class PaddleOCRResult(BaseModel):
    """Validated PaddleOCR result for a single image.

    PaddleOCR 2.x returns results as [[[bbox, (text, conf)], ...]] where the outer
    list is for batch processing (always length 1 for single image).

    PaddleOCR 3.x returns results as [{rec_texts: [...], rec_scores: [...], rec_polys: [...]}]
    """

    detections: list[PaddleOCRDetection]

    @classmethod
    def from_paddle_output(cls, result: list | None) -> PaddleOCRResult:
        """Parse and validate PaddleOCR 2.x raw output format.

        Handles edge cases:
        - None result
        - Empty result [[]]
        - Result with None items [[None]]
        """
        detections: list[PaddleOCRDetection] = []

        if not result or not result[0]:
            return cls(detections=detections)

        for item in result[0]:
            if item is None:
                continue
            detections.append(PaddleOCRDetection.from_paddle_format(item))

        return cls(detections=detections)

    @classmethod
    def from_paddle_v3_output(cls, result: list | None) -> PaddleOCRResult:
        """Parse and validate PaddleOCR 3.x raw output format.

        PaddleOCR 3.x returns: [{rec_texts: [...], rec_scores: [...], rec_polys: [...]}]
        """
        detections: list[PaddleOCRDetection] = []

        if not result or not result[0]:
            return cls(detections=detections)

        page_result = result[0]
        rec_texts = page_result.get("rec_texts", [])
        rec_scores = page_result.get("rec_scores", [])
        rec_polys = page_result.get("rec_polys", [])

        for text, score, poly in zip(rec_texts, rec_scores, rec_polys, strict=False):
            # Convert numpy array to list of lists
            polygon = [[float(p[0]), float(p[1])] for p in poly]
            detections.append(
                PaddleOCRDetection(polygon=polygon, text=text, confidence=float(score))
            )

        return cls(detections=detections)


class TableHTMLParser(HTMLParser):
    """Parse HTML table into rows and cells."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.current_row: list[str] = []
        self.current_cell: str = ""
        self.in_cell: bool = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.current_row = []
        elif tag in ("td", "th"):
            self.in_cell = True
            self.current_cell = ""

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self.in_cell = False
            self.current_row.append(self.current_cell.strip())
        elif tag == "tr":
            if self.current_row:
                self.rows.append(self.current_row)

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell += data


class PaddleOCRAdapter:
    """Converts PaddleOCR output to internal schema."""

    def convert_table_result(self, table_result: dict, page: int = 0) -> Table:
        """Convert PPStructure table output to internal Table model.

        Args:
            table_result: PPStructure result dict with 'type': 'table' and 'res': {'html': '...'}
            page: Page number for the table.

        Returns:
            Table model with cells extracted from HTML.
        """
        cells: list[TableCell] = []
        row_count = 0
        col_count = 0

        if not table_result.get("res") or not table_result["res"].get("html"):
            return Table(page=page, cells=cells, row_count=0, col_count=0)

        html = table_result["res"]["html"]
        parser = TableHTMLParser()
        parser.feed(html)

        for row_idx, row in enumerate(parser.rows):
            col_count = max(col_count, len(row))
            for col_idx, cell_text in enumerate(row):
                cells.append(TableCell(text=cell_text, row=row_idx, col=col_idx))

        row_count = len(parser.rows)

        return Table(page=page, cells=cells, row_count=row_count, col_count=col_count)

    def convert_result(self, result: list | None, major_version: int = 2) -> list[TextBlock]:
        """Convert PaddleOCR output to TextBlocks.

        Args:
            result: Raw PaddleOCR output from ocr() or predict() method.
            major_version: PaddleOCR major version (2 or 3+).

        Returns:
            List of TextBlocks with coordinates in pixels.
        """
        if major_version >= PADDLEOCR_V3_MAJOR:
            validated = PaddleOCRResult.from_paddle_v3_output(result)
        else:
            validated = PaddleOCRResult.from_paddle_output(result)
        return self._detections_to_blocks(validated.detections)

    def _detections_to_blocks(self, detections: list[PaddleOCRDetection]) -> list[TextBlock]:
        """Convert validated detections to TextBlocks."""
        blocks: list[TextBlock] = []

        for detection in detections:
            if not detection.text or not detection.text.strip():
                continue

            bbox, rotation = polygon_to_bbox_and_rotation(detection.polygon)

            blocks.append(
                TextBlock(
                    text=detection.text,
                    bbox=bbox,
                    rotation=rotation,
                    confidence=detection.confidence,
                )
            )

        return blocks
