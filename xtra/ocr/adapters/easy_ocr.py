"""Adapter for converting EasyOCR results to internal schema."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from xtra.models import TextBlock
from xtra.utils.geometry import polygon_to_bbox_and_rotation

POLYGON_POINTS = 4
COORDINATES_PER_POINT = 2


class EasyOCRDetection(BaseModel):
    """A single text detection from EasyOCR.

    EasyOCR returns detections as (bbox, text, confidence) where
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
    def from_easyocr_format(cls, item: tuple) -> EasyOCRDetection:
        """Create from EasyOCR's native format: (bbox, text, confidence)."""
        polygon, text, confidence = item
        return cls(polygon=polygon, text=text, confidence=confidence)


class EasyOCRResult(BaseModel):
    """Validated EasyOCR result for a single image.

    EasyOCR returns results as [(bbox, text, conf), ...] for a single image.
    """

    detections: list[EasyOCRDetection]

    @classmethod
    def from_easyocr_output(cls, result: list | None) -> EasyOCRResult:
        """Parse and validate EasyOCR's raw output format.

        Handles edge cases:
        - Empty result []
        - Result with None items
        """
        detections: list[EasyOCRDetection] = []

        if not result:
            return cls(detections=detections)

        for item in result:
            if item is None:
                continue
            detections.append(EasyOCRDetection.from_easyocr_format(item))

        return cls(detections=detections)


class EasyOCRAdapter:
    """Converts EasyOCR output to internal schema."""

    def convert_result(self, result: list | None) -> list[TextBlock]:
        """Convert EasyOCR output to TextBlocks.

        Args:
            result: Raw EasyOCR output from readtext() method.

        Returns:
            List of TextBlocks with coordinates in pixels.
        """
        validated = EasyOCRResult.from_easyocr_output(result)
        return self._detections_to_blocks(validated.detections)

    def _detections_to_blocks(self, detections: list[EasyOCRDetection]) -> list[TextBlock]:
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
