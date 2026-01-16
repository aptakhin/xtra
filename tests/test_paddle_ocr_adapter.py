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
            polygon=[[10.0, 20.0], [90.0, 20.0], [90.0, 50.0], [10.0, 50.0]],
            text="Hello",
            confidence=0.95,
        )
        assert detection.text == "Hello"
        assert detection.confidence == 0.95
        assert len(detection.polygon) == 4

    def test_from_paddle_format(self) -> None:
        item = ([[10.0, 20.0], [90.0, 20.0], [90.0, 50.0], [10.0, 50.0]], ("Hello", 0.95))
        detection = PaddleOCRDetection.from_paddle_format(item)
        assert detection.text == "Hello"
        assert detection.confidence == 0.95

    def test_invalid_polygon_wrong_point_count(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            PaddleOCRDetection(
                polygon=[[10.0, 20.0], [90.0, 20.0], [90.0, 50.0]],  # Only 3 points
                text="Hello",
                confidence=0.95,
            )
        assert "points" in str(exc_info.value)

    def test_invalid_polygon_wrong_coordinate_count(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            PaddleOCRDetection(
                polygon=[[10.0], [90.0, 20.0], [90.0, 50.0], [10.0, 50.0]],
                text="Hello",
                confidence=0.95,
            )
        assert "coordinates" in str(exc_info.value)


class TestPaddleOCRResult:
    """Tests for PaddleOCRResult Pydantic model."""

    def test_from_paddle_output_success(self) -> None:
        paddle_output = [
            [
                ([[10.0, 20.0], [90.0, 20.0], [90.0, 50.0], [10.0, 50.0]], ("Hello", 0.955)),
                ([[100.0, 20.0], [190.0, 20.0], [190.0, 50.0], [100.0, 50.0]], ("World", 0.872)),
            ]
        ]
        result = PaddleOCRResult.from_paddle_output(paddle_output)
        assert len(result.detections) == 2
        assert result.detections[0].text == "Hello"
        assert result.detections[1].text == "World"

    def test_from_paddle_output_empty(self) -> None:
        result = PaddleOCRResult.from_paddle_output([])
        assert len(result.detections) == 0

    def test_from_paddle_output_none(self) -> None:
        result = PaddleOCRResult.from_paddle_output(None)
        assert len(result.detections) == 0

    def test_from_paddle_output_empty_inner_list(self) -> None:
        result = PaddleOCRResult.from_paddle_output([[]])
        assert len(result.detections) == 0

    def test_from_paddle_output_none_items(self) -> None:
        paddle_output = [
            [
                None,
                ([[10.0, 20.0], [90.0, 20.0], [90.0, 50.0], [10.0, 50.0]], ("Hello", 0.95)),
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
                ([[10.0, 20.0], [90.0, 20.0], [90.0, 50.0], [10.0, 50.0]], ("Hello", 0.955)),
                ([[100.0, 20.0], [190.0, 20.0], [190.0, 50.0], [100.0, 50.0]], ("World", 0.872)),
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
        assert adapter.convert_result([]) == []

    def test_convert_result_none(self) -> None:
        adapter = PaddleOCRAdapter()
        assert adapter.convert_result(None) == []

    def test_convert_result_filters_empty_text(self) -> None:
        adapter = PaddleOCRAdapter()
        paddle_output = [
            [
                ([[10.0, 20.0], [90.0, 20.0], [90.0, 50.0], [10.0, 50.0]], ("", 0.95)),
                ([[100.0, 20.0], [190.0, 20.0], [190.0, 50.0], [100.0, 50.0]], ("  ", 0.95)),
                ([[200.0, 20.0], [290.0, 20.0], [290.0, 50.0], [200.0, 50.0]], ("Valid", 0.95)),
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
                ([[10.0, 30.0], [90.0, 20.0], [95.0, 50.0], [15.0, 60.0]], ("Rotated", 0.9)),
            ]
        ]

        blocks = adapter.convert_result(paddle_output)

        assert len(blocks) == 1
        assert blocks[0].text == "Rotated"
        # Rotation should be detected (non-zero for rotated text)
        assert blocks[0].rotation is not None
