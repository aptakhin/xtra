"""Unit tests for Tesseract OCR adapter and Pydantic models."""

import pytest

from xtra.adapters.tesseract_ocr import (
    TesseractAdapter,
    TesseractDetection,
    TesseractResult,
)


class TestTesseractDetection:
    """Tests for TesseractDetection Pydantic model."""

    def test_valid_detection(self) -> None:
        detection = TesseractDetection(
            left=10,
            top=20,
            width=80,
            height=30,
            text="Hello",
            confidence=95.0,
        )
        assert detection.text == "Hello"
        assert detection.confidence == 0.95  # Normalized to 0-1
        assert detection.left == 10
        assert detection.top == 20

    def test_confidence_normalized(self) -> None:
        detection = TesseractDetection(
            left=0, top=0, width=10, height=10, text="Test", confidence=50.0
        )
        assert detection.confidence == 0.5

    def test_confidence_zero(self) -> None:
        detection = TesseractDetection(
            left=0, top=0, width=10, height=10, text="Test", confidence=0.0
        )
        assert detection.confidence == 0.0

    def test_confidence_hundred(self) -> None:
        detection = TesseractDetection(
            left=0, top=0, width=10, height=10, text="Test", confidence=100.0
        )
        assert detection.confidence == 1.0


class TestTesseractResult:
    """Tests for TesseractResult Pydantic model."""

    def test_from_tesseract_output_success(self) -> None:
        tesseract_output = {
            "level": [5, 5],
            "page_num": [1, 1],
            "block_num": [1, 1],
            "par_num": [1, 1],
            "line_num": [1, 1],
            "word_num": [1, 2],
            "left": [10, 100],
            "top": [20, 20],
            "width": [80, 90],
            "height": [30, 30],
            "conf": [95.5, 87.2],
            "text": ["Hello", "World"],
        }
        result = TesseractResult.from_tesseract_output(tesseract_output)
        assert len(result.detections) == 2
        assert result.detections[0].text == "Hello"
        assert result.detections[1].text == "World"

    def test_from_tesseract_output_none(self) -> None:
        result = TesseractResult.from_tesseract_output(None)
        assert len(result.detections) == 0

    def test_from_tesseract_output_empty(self) -> None:
        result = TesseractResult.from_tesseract_output({})
        assert len(result.detections) == 0

    def test_from_tesseract_output_filters_empty_text(self) -> None:
        tesseract_output = {
            "left": [10, 50, 100],
            "top": [20, 20, 20],
            "width": [30, 40, 50],
            "height": [25, 25, 25],
            "conf": [90.0, 85.0, 80.0],
            "text": ["Hello", "", "World"],
        }
        result = TesseractResult.from_tesseract_output(tesseract_output)
        assert len(result.detections) == 2
        assert result.detections[0].text == "Hello"
        assert result.detections[1].text == "World"

    def test_from_tesseract_output_filters_negative_confidence(self) -> None:
        tesseract_output = {
            "left": [10, 50],
            "top": [20, 20],
            "width": [30, 40],
            "height": [25, 25],
            "conf": [-1, 90.0],
            "text": ["Block", "Word"],
        }
        result = TesseractResult.from_tesseract_output(tesseract_output)
        assert len(result.detections) == 1
        assert result.detections[0].text == "Word"

    def test_from_tesseract_output_filters_whitespace_text(self) -> None:
        tesseract_output = {
            "left": [10, 50],
            "top": [20, 20],
            "width": [30, 40],
            "height": [25, 25],
            "conf": [90.0, 85.0],
            "text": ["  ", "Valid"],
        }
        result = TesseractResult.from_tesseract_output(tesseract_output)
        assert len(result.detections) == 1
        assert result.detections[0].text == "Valid"


class TestTesseractAdapter:
    """Tests for TesseractAdapter conversion logic."""

    def test_convert_result_success(self) -> None:
        adapter = TesseractAdapter()
        tesseract_output = {
            "left": [10, 100],
            "top": [20, 20],
            "width": [80, 90],
            "height": [30, 30],
            "conf": [95.5, 87.2],
            "text": ["Hello", "World"],
        }

        blocks = adapter.convert_result(tesseract_output)

        assert len(blocks) == 2
        assert blocks[0].text == "Hello"
        assert blocks[0].confidence == pytest.approx(0.955, rel=0.01)
        assert blocks[0].bbox.x0 == pytest.approx(10.0, rel=0.01)
        assert blocks[0].bbox.y0 == pytest.approx(20.0, rel=0.01)
        assert blocks[0].bbox.x1 == pytest.approx(90.0, rel=0.01)  # left + width
        assert blocks[0].bbox.y1 == pytest.approx(50.0, rel=0.01)  # top + height

        assert blocks[1].text == "World"
        assert blocks[1].confidence == pytest.approx(0.872, rel=0.01)

    def test_convert_result_empty(self) -> None:
        adapter = TesseractAdapter()
        assert adapter.convert_result(None) == []
        assert adapter.convert_result({}) == []

    def test_convert_result_filters_empty_text(self) -> None:
        adapter = TesseractAdapter()
        tesseract_output = {
            "left": [10, 50, 100],
            "top": [20, 20, 20],
            "width": [30, 40, 50],
            "height": [25, 25, 25],
            "conf": [90.0, -1, 85.0],
            "text": ["", "Block", "Valid"],
        }

        blocks = adapter.convert_result(tesseract_output)

        assert len(blocks) == 1
        assert blocks[0].text == "Valid"

    def test_convert_result_rotation_is_zero(self) -> None:
        adapter = TesseractAdapter()
        tesseract_output = {
            "left": [10],
            "top": [20],
            "width": [80],
            "height": [30],
            "conf": [95.0],
            "text": ["Test"],
        }

        blocks = adapter.convert_result(tesseract_output)

        assert len(blocks) == 1
        assert blocks[0].rotation == 0.0  # Tesseract doesn't provide rotation
