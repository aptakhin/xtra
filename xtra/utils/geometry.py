"""Geometry utilities for coordinate and polygon conversions."""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple

from xtra.models import BBox

# Type alias for points in different formats
Point = Tuple[float, float]

# Minimum required vertices/coordinates for a valid polygon
MIN_POLYGON_POINTS = 4
MIN_FLAT_COORDS = 8  # 4 points * 2 coordinates each


def polygon_to_bbox_and_rotation(
    polygon: Sequence[Any],
    flat: bool = False,
) -> Tuple[BBox, float]:
    """Convert a polygon to axis-aligned bounding box and rotation angle.

    Supports two polygon formats:
    - Point list: [[x0,y0], [x1,y1], [x2,y2], [x3,y3]] (EasyOCR, PaddleOCR, Google)
    - Flat list: [x0, y0, x1, y1, x2, y2, x3, y3] (Azure)

    Args:
        polygon: Polygon vertices as either list of points or flat coordinate list.
        flat: If True, interpret polygon as flat list [x0, y0, x1, y1, ...].
              If False (default), interpret as list of points [[x0, y0], [x1, y1], ...].

    Returns:
        Tuple of (BBox, rotation_degrees).
        Rotation is calculated from the first edge (top-left to top-right).

    Example:
        >>> # Point list format (EasyOCR/PaddleOCR style)
        >>> points = [[0, 0], [100, 0], [100, 50], [0, 50]]
        >>> bbox, rotation = polygon_to_bbox_and_rotation(points)
        >>> bbox.x0, bbox.y0, bbox.x1, bbox.y1
        (0, 0, 100, 50)
        >>> rotation
        0.0

        >>> # Flat list format (Azure style)
        >>> flat = [0, 0, 100, 0, 100, 50, 0, 50]
        >>> bbox, rotation = polygon_to_bbox_and_rotation(flat, flat=True)
    """
    # Convert to unified point list format
    if flat:
        # Flat format: [x0, y0, x1, y1, ...] -> [(x0, y0), (x1, y1), ...]
        coords = list(polygon)
        if len(coords) < MIN_FLAT_COORDS:
            return BBox(x0=0, y0=0, x1=0, y1=0), 0.0
        points: List[Point] = [(coords[i], coords[i + 1]) for i in range(0, MIN_FLAT_COORDS, 2)]
    else:
        # Point list format: [[x0, y0], [x1, y1], ...] -> [(x0, y0), (x1, y1), ...]
        points = [(float(p[0]), float(p[1])) for p in polygon]
        if len(points) < MIN_POLYGON_POINTS:
            return BBox(x0=0, y0=0, x1=0, y1=0), 0.0

    # Extract min/max for axis-aligned bounding box
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    bbox = BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))

    # Calculate rotation from first edge (assumes top-left to top-right)
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    rotation = math.degrees(math.atan2(dy, dx)) if dx != 0 or dy != 0 else 0.0

    return bbox, rotation
