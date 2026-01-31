"""Unit tests for PaddleOCR adapter and Pydantic models."""

import pytest
from pydantic import ValidationError

from xtra.ocr.adapters.paddle_ocr import (
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

    @pytest.mark.parametrize(
        ("polygon", "error_fragment"),
        [
            ([[10.0, 20.0], [90.0, 20.0], [90.0, 50.0]], "points"),  # 3 points
            ([[10.0], [90.0, 20.0], [90.0, 50.0], [10.0, 50.0]], "coordinates"),  # bad coords
        ],
        ids=["wrong_point_count", "wrong_coordinate_count"],
    )
    def test_invalid_polygon(self, polygon: list, error_fragment: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            PaddleOCRDetection(polygon=polygon, text="Hello", confidence=0.95)
        assert error_fragment in str(exc_info.value)


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

    @pytest.mark.parametrize(
        "paddle_output",
        [[], None, [[]]],
        ids=["empty_list", "none", "empty_inner_list"],
    )
    def test_from_paddle_output_empty_inputs(self, paddle_output: list | None) -> None:
        result = PaddleOCRResult.from_paddle_output(paddle_output)
        assert len(result.detections) == 0

    def test_from_paddle_output_filters_none_items(self) -> None:
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


class TestPaddleOCRResultV3:
    """Tests for PaddleOCRResult v3 format parsing."""

    def test_from_paddle_v3_output_success(self) -> None:
        import numpy as np

        paddle_output = [
            {
                "rec_texts": ["Hello", "World"],
                "rec_scores": [0.955, 0.872],
                "rec_polys": [
                    np.array([[10, 20], [90, 20], [90, 50], [10, 50]], dtype=np.int16),
                    np.array([[100, 20], [190, 20], [190, 50], [100, 50]], dtype=np.int16),
                ],
            }
        ]
        result = PaddleOCRResult.from_paddle_v3_output(paddle_output)
        assert len(result.detections) == 2
        assert result.detections[0].text == "Hello"
        assert result.detections[0].confidence == pytest.approx(0.955, rel=0.01)
        assert result.detections[1].text == "World"
        assert result.detections[1].confidence == pytest.approx(0.872, rel=0.01)

    @pytest.mark.parametrize(
        "paddle_output",
        [[], None, [{}]],
        ids=["empty_list", "none", "empty_dict"],
    )
    def test_from_paddle_v3_output_empty_inputs(self, paddle_output: list | None) -> None:
        result = PaddleOCRResult.from_paddle_v3_output(paddle_output)
        assert len(result.detections) == 0


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

    @pytest.mark.parametrize(
        "paddle_output",
        [[], None],
        ids=["empty_list", "none"],
    )
    def test_convert_result_empty_inputs(self, paddle_output: list | None) -> None:
        adapter = PaddleOCRAdapter()
        assert adapter.convert_result(paddle_output) == []

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
        paddle_output = [
            [
                ([[10.0, 30.0], [90.0, 20.0], [95.0, 50.0], [15.0, 60.0]], ("Rotated", 0.9)),
            ]
        ]

        blocks = adapter.convert_result(paddle_output)

        assert len(blocks) == 1
        assert blocks[0].text == "Rotated"
        assert blocks[0].rotation is not None


class TestPaddleOCRAdapterV3:
    """Tests for PaddleOCRAdapter with v3 format."""

    def test_convert_result_v3_success(self) -> None:
        import numpy as np

        adapter = PaddleOCRAdapter()
        paddle_output = [
            {
                "rec_texts": ["Hello", "World"],
                "rec_scores": [0.955, 0.872],
                "rec_polys": [
                    np.array([[10, 20], [90, 20], [90, 50], [10, 50]], dtype=np.int16),
                    np.array([[100, 20], [190, 20], [190, 50], [100, 50]], dtype=np.int16),
                ],
            }
        ]

        blocks = adapter.convert_result(paddle_output, major_version=3)

        assert len(blocks) == 2
        assert blocks[0].text == "Hello"
        assert blocks[0].confidence == pytest.approx(0.955, rel=0.01)
        assert blocks[0].bbox.x0 == pytest.approx(10.0, rel=0.01)
        assert blocks[0].bbox.y0 == pytest.approx(20.0, rel=0.01)
        assert blocks[1].text == "World"

    @pytest.mark.parametrize(
        "paddle_output",
        [[], None, [{}]],
        ids=["empty_list", "none", "empty_dict"],
    )
    def test_convert_result_v3_empty_inputs(self, paddle_output: list | None) -> None:
        adapter = PaddleOCRAdapter()
        assert adapter.convert_result(paddle_output, major_version=3) == []

    def test_convert_result_v3_filters_empty_text(self) -> None:
        import numpy as np

        adapter = PaddleOCRAdapter()
        paddle_output = [
            {
                "rec_texts": ["", "  ", "Valid"],
                "rec_scores": [0.95, 0.95, 0.95],
                "rec_polys": [
                    np.array([[10, 20], [90, 20], [90, 50], [10, 50]], dtype=np.int16),
                    np.array([[100, 20], [190, 20], [190, 50], [100, 50]], dtype=np.int16),
                    np.array([[200, 20], [290, 20], [290, 50], [200, 50]], dtype=np.int16),
                ],
            }
        ]

        blocks = adapter.convert_result(paddle_output, major_version=3)

        assert len(blocks) == 1
        assert blocks[0].text == "Valid"
