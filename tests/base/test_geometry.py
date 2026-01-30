"""Tests for geometry utilities."""

from xtra.utils.geometry import polygon_to_bbox_and_rotation


class TestPolygonToBboxAndRotation:
    """Tests for polygon_to_bbox_and_rotation function."""

    def test_horizontal_polygon(self) -> None:
        """Test horizontal polygon returns 0 rotation."""
        # 100x20 horizontal rectangle
        polygon = [[0, 0], [100, 0], [100, 20], [0, 20]]
        bbox, rotation = polygon_to_bbox_and_rotation(polygon)

        assert bbox.x0 == 0
        assert bbox.y0 == 0
        assert bbox.x1 == 100
        assert bbox.y1 == 20
        assert rotation == 0.0

    def test_rotated_polygon(self) -> None:
        """Test rotated polygon returns correct rotation."""
        # Polygon rotated 45 degrees
        polygon = [[0, 0], [10, 10], [0, 20], [-10, 10]]
        bbox, rotation = polygon_to_bbox_and_rotation(polygon)

        assert bbox.x0 == -10
        assert bbox.y0 == 0
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert abs(rotation - 45.0) < 0.001

    def test_flat_polygon_format(self) -> None:
        """Test flat polygon format (Azure style): x0, y0, x1, y1, x2, y2, x3, y3."""
        flat_polygon = [10, 20, 110, 20, 110, 40, 10, 40]
        bbox, rotation = polygon_to_bbox_and_rotation(flat_polygon, flat=True)

        assert bbox.x0 == 10
        assert bbox.y0 == 20
        assert bbox.x1 == 110
        assert bbox.y1 == 40
        assert rotation == 0.0

    def test_short_polygon_returns_empty_bbox(self) -> None:
        """Test polygon with fewer than 4 points returns empty bbox."""
        short_polygon = [[0, 0], [10, 10]]
        bbox, rotation = polygon_to_bbox_and_rotation(short_polygon)

        assert bbox.x0 == 0
        assert bbox.y0 == 0
        assert bbox.x1 == 0
        assert bbox.y1 == 0
        assert rotation == 0.0

    def test_short_flat_polygon_returns_empty_bbox(self) -> None:
        """Test flat polygon with fewer than 8 values returns empty bbox."""
        short_flat = [0, 0, 10, 10]  # Only 2 points
        bbox, rotation = polygon_to_bbox_and_rotation(short_flat, flat=True)

        assert bbox.x0 == 0
        assert bbox.y0 == 0
        assert bbox.x1 == 0
        assert bbox.y1 == 0
        assert rotation == 0.0

    def test_easyocr_style_polygon(self) -> None:
        """Test EasyOCR-style polygon (list of 2-element lists)."""
        # EasyOCR returns [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        polygon = [[50.0, 10.0], [150.0, 10.0], [150.0, 30.0], [50.0, 30.0]]
        bbox, rotation = polygon_to_bbox_and_rotation(polygon)

        assert bbox.x0 == 50.0
        assert bbox.y0 == 10.0
        assert bbox.x1 == 150.0
        assert bbox.y1 == 30.0
        assert rotation == 0.0
