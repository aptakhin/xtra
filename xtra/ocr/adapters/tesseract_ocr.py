"""Adapter for converting Tesseract OCR results to internal schema."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from xtra.models import BBox, TextBlock


class TesseractDetection(BaseModel):
    """A single text detection from Tesseract.

    Tesseract returns word-level detections with bounding box and confidence.
    """

    left: float
    top: float
    width: float
    height: float
    text: str
    confidence: float

    @field_validator("confidence")
    @classmethod
    def normalize_confidence(cls, v: float) -> float:
        """Normalize confidence from 0-100 to 0-1 scale."""
        return v / 100.0


class TesseractResult(BaseModel):
    """Validated Tesseract result for a single image.

    Tesseract returns results as a dict with parallel arrays for each field.
    """

    detections: list[TesseractDetection]

    @classmethod
    def from_tesseract_output(cls, data: dict[str, list] | None) -> TesseractResult:
        """Parse and validate Tesseract's raw output format.

        Args:
            data: Dict with parallel arrays from pytesseract.image_to_data().
                  Expected keys: left, top, width, height, conf, text.

        Returns:
            TesseractResult with validated detections.
        """
        detections: list[TesseractDetection] = []

        if not data or "text" not in data:
            return cls(detections=detections)

        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i]
            conf = data["conf"][i]

            # Skip empty text and low/negative confidence (non-text elements)
            if not text or not str(text).strip() or conf < 0:
                continue

            detections.append(
                TesseractDetection(
                    left=float(data["left"][i]),
                    top=float(data["top"][i]),
                    width=float(data["width"][i]),
                    height=float(data["height"][i]),
                    text=str(text),
                    confidence=float(conf),
                )
            )

        return cls(detections=detections)


class TesseractAdapter:
    """Converts Tesseract OCR output to internal schema."""

    def convert_result(self, data: dict[str, list] | None) -> list[TextBlock]:
        """Convert Tesseract output to TextBlocks.

        Args:
            data: Raw Tesseract output from image_to_data() method.

        Returns:
            List of TextBlocks with coordinates in pixels.
        """
        validated = TesseractResult.from_tesseract_output(data)
        return self._detections_to_blocks(validated.detections)

    def _detections_to_blocks(self, detections: list[TesseractDetection]) -> list[TextBlock]:
        """Convert validated detections to TextBlocks."""
        blocks: list[TextBlock] = []

        for detection in detections:
            if not detection.text or not detection.text.strip():
                continue

            bbox = BBox(
                x0=detection.left,
                y0=detection.top,
                x1=detection.left + detection.width,
                y1=detection.top + detection.height,
            )

            blocks.append(
                TextBlock(
                    text=detection.text,
                    bbox=bbox,
                    rotation=0.0,  # Tesseract doesn't provide rotation per word
                    confidence=detection.confidence,
                )
            )

        return blocks
