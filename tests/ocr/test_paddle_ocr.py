"""Unit tests for PaddleOCR adapter and Pydantic models."""

import pytest
from pydantic import ValidationError

from xtra.adapters.paddle_ocr import (
    PaddleOCRAdapter,
    PaddleOCRDetection,
    PaddleOCRResult,
)


class TestPaddleOCRDetection:
    """Tests for PaddleOCRDetection Pydantic model."""

    def test_valid_detection(self) -> None:
        detection = PaddleOCRDetection(
            polygon=[[10, 20], [90, 20], [90, 50], [10, 50]],
            text="Hello",
            confidence=0.95,
        )
        assert detection.text == "Hello"
        assert detection.confidence == 0.95
        assert len(detection.polygon) == 4

    def test_from_paddle_format(self) -> None:
        item = ([[10, 20], [90, 20], [90, 50], [10, 50]], ("Hello", 0.95))
        detection = PaddleOCRDetection.from_paddle_format(item)
        assert detection.text == "Hello"
        assert detection.confidence == 0.95

    def test_invalid_polygon_wrong_point_count(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            PaddleOCRDetection(
                polygon=[[10, 20], [90, 20], [90, 50]],  # Only 3 points
                text="Hello",
                confidence=0.95,
            )
        assert "points" in str(exc_info.value)

    def test_invalid_polygon_wrong_coordinate_count(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            PaddleOCRDetection(
                polygon=[[10], [90, 20], [90, 50], [10, 50]],  # First point has 1 coord
                text="Hello",
                confidence=0.95,
            )
        assert "coordinates" in str(exc_info.value)


class TestPaddleOCRResult:
    """Tests for PaddleOCRResult Pydantic model."""

    def test_from_paddle_output_success(self) -> None:
        paddle_output = [
            [
                [[[10, 20], [90, 20], [90, 50], [10, 50]], ("Hello", 0.955)],
                [[[100, 20], [190, 20], [190, 50], [100, 50]], ("World", 0.872)],
            ]
        ]
        result = PaddleOCRResult.from_paddle_output(paddle_output)
        assert len(result.detections) == 2
        assert result.detections[0].text == "Hello"
        assert result.detections[1].text == "World"

    def test_from_paddle_output_none(self) -> None:
        result = PaddleOCRResult.from_paddle_output(None)
        assert len(result.detections) == 0

    def test_from_paddle_output_empty_outer(self) -> None:
        result = PaddleOCRResult.from_paddle_output([])
        assert len(result.detections) == 0

    def test_from_paddle_output_empty_inner(self) -> None:
        result = PaddleOCRResult.from_paddle_output([[]])
        assert len(result.detections) == 0

    def test_from_paddle_output_none_items(self) -> None:
        result = PaddleOCRResult.from_paddle_output([[None]])
        assert len(result.detections) == 0

    def test_from_paddle_output_mixed_none_items(self) -> None:
        paddle_output = [
            [
                None,
                [[[10, 20], [90, 20], [90, 50], [10, 50]], ("Hello", 0.95)],
                None,
            ]
        ]
        result = PaddleOCRResult.from_paddle_output(paddle_output)
        assert len(result.detections) == 1
        assert result.detections[0].text == "Hello"


class TestPaddleOCRAdapter:
    """Tests for PaddleOCRAdapter conversion logic."""

    def test_convert_result_success(self) -> None:
        adapter = PaddleOCRAdapter()
        paddle_output = [
            [
                [[[10, 20], [90, 20], [90, 50], [10, 50]], ("Hello", 0.955)],
                [[[100, 20], [190, 20], [190, 50], [100, 50]], ("World", 0.872)],
            ]
        ]

        blocks = adapter.convert_result(paddle_output)

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
        adapter = PaddleOCRAdapter()
        assert adapter.convert_result(None) == []
        assert adapter.convert_result([]) == []
        assert adapter.convert_result([[]]) == []
        assert adapter.convert_result([[None]]) == []

    def test_convert_result_filters_empty_text(self) -> None:
        adapter = PaddleOCRAdapter()
        paddle_output = [
            [
                [[[10, 20], [90, 20], [90, 50], [10, 50]], ("", 0.95)],
                [[[100, 20], [190, 20], [190, 50], [100, 50]], ("  ", 0.95)],
                [[[200, 20], [290, 20], [290, 50], [200, 50]], ("Valid", 0.95)],
            ]
        ]

        blocks = adapter.convert_result(paddle_output)

        assert len(blocks) == 1
        assert blocks[0].text == "Valid"

    def test_convert_result_with_rotation(self) -> None:
        adapter = PaddleOCRAdapter()
        # Rotated text (not axis-aligned)
        paddle_output = [
            [
                [[[10, 30], [90, 20], [95, 50], [15, 60]], ("Rotated", 0.9)],
            ]
        ]

        blocks = adapter.convert_result(paddle_output)

        assert len(blocks) == 1
        assert blocks[0].text == "Rotated"
        # Rotation should be detected (non-zero for rotated text)
        assert blocks[0].rotation is not None
