from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field

    PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseModel, Field

    PYDANTIC_V2 = False


class ExtractorType(StrEnum):
    PDF = "pdf"
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"
    PADDLE = "paddle"
    AZURE_DI = "azure-di"
    GOOGLE_DOCAI = "google-docai"


class CoordinateUnit(StrEnum):
    """Units for coordinate output."""

    PIXELS = "pixels"  # Image pixels at a given DPI
    POINTS = "points"  # 1/72 inch (PDF native, default)
    INCHES = "inches"  # Imperial inches
    NORMALIZED = "normalized"  # 0-1 relative to page dimensions


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class FontInfo(BaseModel):
    name: str | None = None
    size: float | None = None
    flags: int | None = None
    weight: int | None = None


class TextBlock(BaseModel):
    text: str
    bbox: BBox
    rotation: float = 0.0
    confidence: float | None = None
    font_info: FontInfo | None = None


class CoordinateInfo(BaseModel):
    """Information about the coordinate system used."""

    unit: CoordinateUnit
    dpi: float | None = None  # Only meaningful for pixel-based coords


class TableCell(BaseModel):
    """A cell within a table."""

    text: str
    row: int
    col: int
    bbox: BBox | None = None  # Cell bbox if available


class Table(BaseModel):
    """A table extracted from a document page."""

    page: int  # Page number (0-indexed)
    cells: list[TableCell] = Field(default_factory=list)
    row_count: int = 0
    col_count: int = 0
    bbox: BBox | None = None  # Table bbox if available


class Page(BaseModel):
    page: int
    width: float
    height: float
    texts: list[TextBlock] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    coordinate_info: CoordinateInfo | None = None


class ExtractorMetadata(BaseModel):
    extractor_type: ExtractorType
    creator: str | None = None
    producer: str | None = None
    title: str | None = None
    author: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


if PYDANTIC_V2:

    class Document(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        path: Path
        pages: list[Page] = Field(default_factory=list)
        metadata: ExtractorMetadata | None = None
else:

    class Document(BaseModel):
        path: Path
        pages: list[Page] = Field(default_factory=list)
        metadata: ExtractorMetadata | None = None

        class Config:
            arbitrary_types_allowed = True
