"""Unit tests for EasyOCR adapter and Pydantic models."""

import pytest
from pydantic import ValidationError

from xtra.adapters.easy_ocr import (
    EasyOCRAdapter,
    EasyOCRDetection,
    EasyOCRResult,
)


class TestEasyOCRDetection:
    """Tests for EasyOCRDetection Pydantic model."""

    def test_valid_detection(self) -> None:
        detection = EasyOCRDetection(
            polygon=[[10, 20], [90, 20], [90, 50], [10, 50]],
            text="Hello",
            confidence=0.95,
        )
        assert detection.text == "Hello"
        assert detection.confidence == 0.95
        assert len(detection.polygon) == 4

    def test_from_easyocr_format(self) -> None:
        item = ([[10, 20], [90, 20], [90, 50], [10, 50]], "Hello", 0.95)
        detection = EasyOCRDetection.from_easyocr_format(item)
        assert detection.text == "Hello"
        assert detection.confidence == 0.95

    def test_invalid_polygon_wrong_point_count(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            EasyOCRDetection(
                polygon=[[10, 20], [90, 20], [90, 50]],  # Only 3 points
                text="Hello",
                confidence=0.95,
            )
        assert "points" in str(exc_info.value)

    def test_invalid_polygon_wrong_coordinate_count(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            EasyOCRDetection(
                polygon=[[10], [90, 20], [90, 50], [10, 50]],  # First point has 1 coord
                text="Hello",
                confidence=0.95,
            )
        assert "coordinates" in str(exc_info.value)


class TestEasyOCRResult:
    """Tests for EasyOCRResult Pydantic model."""

    def test_from_easyocr_output_success(self) -> None:
        easyocr_output = [
            ([[10, 20], [90, 20], [90, 50], [10, 50]], "Hello", 0.955),
            ([[100, 20], [190, 20], [190, 50], [100, 50]], "World", 0.872),
        ]
        result = EasyOCRResult.from_easyocr_output(easyocr_output)
        assert len(result.detections) == 2
        assert result.detections[0].text == "Hello"
        assert result.detections[1].text == "World"

    def test_from_easyocr_output_empty(self) -> None:
        result = EasyOCRResult.from_easyocr_output([])
        assert len(result.detections) == 0

    def test_from_easyocr_output_none_items(self) -> None:
        easyocr_output = [
            None,
            ([[10, 20], [90, 20], [90, 50], [10, 50]], "Hello", 0.95),
            None,
        ]
        result = EasyOCRResult.from_easyocr_output(easyocr_output)
        assert len(result.detections) == 1
        assert result.detections[0].text == "Hello"


class TestEasyOCRAdapter:
    """Tests for EasyOCRAdapter conversion logic."""

    def test_convert_result_success(self) -> None:
        adapter = EasyOCRAdapter()
        easyocr_output = [
            ([[10, 20], [90, 20], [90, 50], [10, 50]], "Hello", 0.955),
            ([[100, 20], [190, 20], [190, 50], [100, 50]], "World", 0.872),
        ]

        blocks = adapter.convert_result(easyocr_output)

        assert len(blocks) == 2
        assert blocks[0].text == "Hello"
        assert blocks[0].confidence == pytest.approx(0.955, rel=0.01)
        assert blocks[0].bbox.x0 == pytest.approx(10.0, rel=0.01)
        assert blocks[0].bbox.y0 == pytest.approx(20.0, rel=0.01)
        assert blocks[0].bbox.x1 == pytest.approx(90.0, rel=0.01)
        assert blocks[0].bbox.y1 == pytest.approx(50.0, rel=0.01)

        assert blocks[1].text == "World"
        assert blocks[1].confidence == pytest.approx(0.872, rel=0.01)

    def test_convert_result_empty(self) -> None:
        adapter = EasyOCRAdapter()
        assert adapter.convert_result([]) == []

    def test_convert_result_filters_empty_text(self) -> None:
        adapter = EasyOCRAdapter()
        easyocr_output = [
            ([[10, 20], [90, 20], [90, 50], [10, 50]], "", 0.95),
            ([[100, 20], [190, 20], [190, 50], [100, 50]], "  ", 0.95),
            ([[200, 20], [290, 20], [290, 50], [200, 50]], "Valid", 0.95),
        ]

        blocks = adapter.convert_result(easyocr_output)

        assert len(blocks) == 1
        assert blocks[0].text == "Valid"

    def test_convert_result_with_rotation(self) -> None:
        adapter = EasyOCRAdapter()
        # Rotated text (not axis-aligned)
        easyocr_output = [
            ([[10, 30], [90, 20], [95, 50], [15, 60]], "Rotated", 0.9),
        ]

        blocks = adapter.convert_result(easyocr_output)

        assert len(blocks) == 1
        assert blocks[0].text == "Rotated"
        # Rotation should be detected (non-zero for rotated text)
        assert blocks[0].rotation is not None
