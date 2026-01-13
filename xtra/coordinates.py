"""Coordinate conversion utilities for unified coordinate output."""

from __future__ import annotations

from typing import Optional

from .models import BBox, CoordinateInfo, CoordinateUnit, Page, TextBlock

POINTS_PER_INCH = 72.0


class CoordinateConverter:
    """Converts coordinates between different unit systems.

    All conversions go through points as the canonical intermediate unit.

    Conversion formulas:
    - Points → Pixels: points * (dpi / 72)
    - Pixels → Points: pixels * (72 / dpi)
    - Points → Inches: points / 72
    - Inches → Points: inches * 72
    - Normalized → Points: normalized * page_dimension_in_points
    - Points → Normalized: points / page_dimension_in_points
    """

    def __init__(
        self,
        source_unit: CoordinateUnit,
        page_width: float,
        page_height: float,
        dpi: Optional[float] = None,
    ) -> None:
        """Initialize converter.

        Args:
            source_unit: The unit system of the source coordinates.
            page_width: Page width in source units.
            page_height: Page height in source units.
            dpi: DPI value, required when converting to/from pixels.
        """
        self.source_unit = source_unit
        self.page_width = page_width
        self.page_height = page_height
        self.dpi = dpi

        # Pre-compute page dimensions in points for efficiency
        self._page_width_pts = self._to_points(page_width, is_x=True)
        self._page_height_pts = self._to_points(page_height, is_x=False)

    def _to_points(self, value: float, is_x: bool) -> float:
        """Convert from source unit to points."""
        if self.source_unit == CoordinateUnit.POINTS:
            return value
        elif self.source_unit == CoordinateUnit.PIXELS:
            if self.dpi is None:
                raise ValueError("DPI required for pixel conversion")
            return value * (POINTS_PER_INCH / self.dpi)
        elif self.source_unit == CoordinateUnit.INCHES:
            return value * POINTS_PER_INCH
        elif self.source_unit == CoordinateUnit.NORMALIZED:
            # For normalized, page dimensions must already be in points
            # This is a special case - normalized coords are 0-1 relative
            page_dim = self.page_width if is_x else self.page_height
            return value * page_dim
        return value

    def _from_points(self, points: float, target: CoordinateUnit, is_x: bool) -> float:
        """Convert from points to target unit."""
        if target == CoordinateUnit.POINTS:
            return points
        elif target == CoordinateUnit.PIXELS:
            if self.dpi is None:
                raise ValueError("DPI required for pixel conversion")
            return points * (self.dpi / POINTS_PER_INCH)
        elif target == CoordinateUnit.INCHES:
            return points / POINTS_PER_INCH
        elif target == CoordinateUnit.NORMALIZED:
            page_dim_pts = self._page_width_pts if is_x else self._page_height_pts
            return points / page_dim_pts if page_dim_pts > 0 else 0.0
        return points

    def convert_value(self, value: float, target_unit: CoordinateUnit, is_x: bool = True) -> float:
        """Convert a single coordinate value to target unit.

        Args:
            value: The coordinate value to convert.
            target_unit: The target unit system.
            is_x: True for x-coordinate (width), False for y-coordinate (height).
                  Used for normalized conversion which depends on page dimension.

        Returns:
            The converted value in target units.
        """
        # Step 1: Convert source to points
        points_value = self._to_points(value, is_x)

        # Step 2: Convert points to target unit
        return self._from_points(points_value, target_unit, is_x)

    def convert_bbox(self, bbox: BBox, target_unit: CoordinateUnit) -> BBox:
        """Convert a BBox to target unit system."""
        return BBox(
            x0=self.convert_value(bbox.x0, target_unit, is_x=True),
            y0=self.convert_value(bbox.y0, target_unit, is_x=False),
            x1=self.convert_value(bbox.x1, target_unit, is_x=True),
            y1=self.convert_value(bbox.y1, target_unit, is_x=False),
        )

    def convert_text_block(self, block: TextBlock, target_unit: CoordinateUnit) -> TextBlock:
        """Convert a TextBlock's bbox to target unit."""
        return TextBlock(
            text=block.text,
            bbox=self.convert_bbox(block.bbox, target_unit),
            rotation=block.rotation,
            confidence=block.confidence,
            font_info=block.font_info,
        )

    def convert_page(
        self, page: Page, target_unit: CoordinateUnit, target_dpi: Optional[float] = None
    ) -> Page:
        """Convert an entire Page to target unit system.

        Args:
            page: The page to convert.
            target_unit: The target unit system.
            target_dpi: DPI for target (used when target is PIXELS).

        Returns:
            A new Page with converted coordinates.
        """
        converted_texts = [self.convert_text_block(t, target_unit) for t in page.texts]

        # Convert page dimensions
        new_width = self.convert_value(page.width, target_unit, is_x=True)
        new_height = self.convert_value(page.height, target_unit, is_x=False)

        # Determine DPI for coordinate info
        coord_dpi: Optional[float] = None
        if target_unit == CoordinateUnit.PIXELS:
            coord_dpi = target_dpi if target_dpi is not None else self.dpi

        return Page(
            page=page.page,
            width=new_width,
            height=new_height,
            texts=converted_texts,
            coordinate_info=CoordinateInfo(
                unit=target_unit,
                dpi=coord_dpi,
            ),
        )
