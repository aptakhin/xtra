"""Tests for coordinate conversion utilities."""

import pytest

from xtra.coordinates import POINTS_PER_INCH, CoordinateConverter
from xtra.models import BBox, CoordinateUnit, Page, TextBlock


class TestCoordinateConverter:
    """Tests for CoordinateConverter class."""

    def test_points_to_pixels(self) -> None:
        """72 points at 144 DPI = 144 pixels."""
        converter = CoordinateConverter(
            source_unit=CoordinateUnit.POINTS,
            page_width=72,
            page_height=72,
            dpi=144,
        )
        result = converter.convert_value(72, CoordinateUnit.PIXELS)
        assert result == 144

    def test_pixels_to_points(self) -> None:
        """200 pixels at 200 DPI = 72 points."""
        converter = CoordinateConverter(
            source_unit=CoordinateUnit.PIXELS,
            page_width=200,
            page_height=200,
            dpi=200,
        )
        result = converter.convert_value(200, CoordinateUnit.POINTS)
        assert abs(result - 72) < 0.001

    def test_inches_to_points(self) -> None:
        """1 inch = 72 points."""
        converter = CoordinateConverter(
            source_unit=CoordinateUnit.INCHES,
            page_width=8.5,
            page_height=11,
            dpi=None,
        )
        result = converter.convert_value(1.0, CoordinateUnit.POINTS)
        assert result == POINTS_PER_INCH

    def test_points_to_inches(self) -> None:
        """72 points = 1 inch."""
        converter = CoordinateConverter(
            source_unit=CoordinateUnit.POINTS,
            page_width=612,
            page_height=792,
            dpi=None,
        )
        result = converter.convert_value(72, CoordinateUnit.INCHES)
        assert result == 1.0

    def test_points_to_normalized(self) -> None:
        """306 points on 612pt wide page = 0.5 normalized."""
        converter = CoordinateConverter(
            source_unit=CoordinateUnit.POINTS,
            page_width=612,
            page_height=792,
            dpi=None,
        )
        result = converter.convert_value(306, CoordinateUnit.NORMALIZED, is_x=True)
        assert result == 0.5

    def test_normalized_to_points(self) -> None:
        """0.5 normalized on 612pt page = 306 points."""
        converter = CoordinateConverter(
            source_unit=CoordinateUnit.NORMALIZED,
            page_width=612,  # page dimensions in points for normalized source
            page_height=792,
            dpi=None,
        )
        result = converter.convert_value(0.5, CoordinateUnit.POINTS, is_x=True)
        assert result == 306

    def test_same_unit_no_conversion(self) -> None:
        """Converting to same unit returns same value."""
        converter = CoordinateConverter(
            source_unit=CoordinateUnit.POINTS,
            page_width=612,
            page_height=792,
            dpi=None,
        )
        result = converter.convert_value(100, CoordinateUnit.POINTS)
        assert result == 100

    def test_pixels_conversion_requires_dpi(self) -> None:
        """Converting to pixels without DPI raises error."""
        converter = CoordinateConverter(
            source_unit=CoordinateUnit.INCHES,
            page_width=8.5,
            page_height=11,
            dpi=None,
        )
        with pytest.raises(ValueError, match="DPI required"):
            converter.convert_value(1.0, CoordinateUnit.PIXELS)

    def test_bbox_conversion(self) -> None:
        """BBox conversion preserves all coordinates."""
        converter = CoordinateConverter(
            source_unit=CoordinateUnit.POINTS,
            page_width=612,
            page_height=792,
            dpi=72,
        )
        bbox = BBox(x0=72, y0=72, x1=144, y1=144)
        result = converter.convert_bbox(bbox, CoordinateUnit.INCHES)
        assert result.x0 == 1.0
        assert result.y0 == 1.0
        assert result.x1 == 2.0
        assert result.y1 == 2.0

    def test_text_block_conversion(self) -> None:
        """TextBlock conversion preserves text and metadata."""
        converter = CoordinateConverter(
            source_unit=CoordinateUnit.POINTS,
            page_width=612,
            page_height=792,
            dpi=72,
        )
        block = TextBlock(
            text="Hello",
            bbox=BBox(x0=72, y0=72, x1=144, y1=144),
            rotation=5.0,
            confidence=0.95,
        )
        result = converter.convert_text_block(block, CoordinateUnit.INCHES)
        assert result.text == "Hello"
        assert result.rotation == 5.0
        assert result.confidence == 0.95
        assert result.bbox.x0 == 1.0

    def test_page_conversion(self) -> None:
        """Page conversion updates dimensions and all text blocks."""
        converter = CoordinateConverter(
            source_unit=CoordinateUnit.POINTS,
            page_width=612,
            page_height=792,
            dpi=72,
        )
        page = Page(
            page=0,
            width=612,
            height=792,
            texts=[
                TextBlock(
                    text="Test",
                    bbox=BBox(x0=72, y0=72, x1=144, y1=144),
                )
            ],
        )
        result = converter.convert_page(page, CoordinateUnit.INCHES)
        assert result.width == 8.5
        assert result.height == 11.0
        assert result.texts[0].bbox.x0 == 1.0
        assert result.coordinate_info is not None
        assert result.coordinate_info.unit == CoordinateUnit.INCHES

    def test_page_conversion_preserves_page_number(self) -> None:
        """Page conversion preserves page number."""
        converter = CoordinateConverter(
            source_unit=CoordinateUnit.POINTS,
            page_width=612,
            page_height=792,
            dpi=None,
        )
        page = Page(page=5, width=612, height=792)
        result = converter.convert_page(page, CoordinateUnit.INCHES)
        assert result.page == 5


class TestCoordinateConversionRoundTrip:
    """Test round-trip conversions."""

    def test_points_pixels_roundtrip(self) -> None:
        """points -> pixels -> points should be identity."""
        original = 100.0
        dpi = 200

        # Points to pixels
        converter1 = CoordinateConverter(
            source_unit=CoordinateUnit.POINTS,
            page_width=612,
            page_height=792,
            dpi=dpi,
        )
        pixels = converter1.convert_value(original, CoordinateUnit.PIXELS)

        # Pixels back to points
        converter2 = CoordinateConverter(
            source_unit=CoordinateUnit.PIXELS,
            page_width=612 * dpi / 72,
            page_height=792 * dpi / 72,
            dpi=dpi,
        )
        back_to_points = converter2.convert_value(pixels, CoordinateUnit.POINTS)

        assert abs(back_to_points - original) < 0.001

    def test_inches_points_roundtrip(self) -> None:
        """inches -> points -> inches should be identity."""
        original = 5.5

        converter1 = CoordinateConverter(
            source_unit=CoordinateUnit.INCHES,
            page_width=8.5,
            page_height=11,
            dpi=None,
        )
        points = converter1.convert_value(original, CoordinateUnit.POINTS)

        converter2 = CoordinateConverter(
            source_unit=CoordinateUnit.POINTS,
            page_width=8.5 * 72,
            page_height=11 * 72,
            dpi=None,
        )
        back_to_inches = converter2.convert_value(points, CoordinateUnit.INCHES)

        assert abs(back_to_inches - original) < 0.001


class TestCoordinateEdgeCases:
    """Test edge cases for coordinate conversion."""

    def test_zero_value_conversion(self) -> None:
        """Zero values convert correctly."""
        converter = CoordinateConverter(
            source_unit=CoordinateUnit.POINTS,
            page_width=612,
            page_height=792,
            dpi=72,
        )
        result = converter.convert_value(0, CoordinateUnit.PIXELS)
        assert result == 0

    def test_zero_page_dimensions_normalized(self) -> None:
        """Zero page dimensions return 0 for normalized output."""
        converter = CoordinateConverter(
            source_unit=CoordinateUnit.POINTS,
            page_width=0,
            page_height=0,
            dpi=None,
        )
        result = converter.convert_value(100, CoordinateUnit.NORMALIZED)
        assert result == 0.0

    def test_large_values(self) -> None:
        """Large values convert without overflow."""
        converter = CoordinateConverter(
            source_unit=CoordinateUnit.POINTS,
            page_width=10000,
            page_height=10000,
            dpi=300,
        )
        result = converter.convert_value(5000, CoordinateUnit.PIXELS)
        expected = 5000 * (300 / 72)
        assert abs(result - expected) < 0.001
