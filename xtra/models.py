from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, List, Optional

try:
    from pydantic import BaseModel, ConfigDict, Field

    PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseModel, Field

    PYDANTIC_V2 = False


class SourceType(StrEnum):
    PDF = "pdf"
    EASYOCR = "easyocr"
    PDF_EASYOCR = "pdf-easyocr"
    TESSERACT = "tesseract"
    PDF_TESSERACT = "pdf-tesseract"
    PADDLE = "paddle"
    PDF_PADDLE = "pdf-paddle"
    AZURE_DI = "azure-di"
    GOOGLE_DOCAI = "google-docai"


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class FontInfo(BaseModel):
    name: Optional[str] = None
    size: Optional[float] = None
    flags: Optional[int] = None
    weight: Optional[int] = None
    is_italic: Optional[bool] = None
    is_bold: Optional[bool] = None


class TextBlock(BaseModel):
    text: str
    bbox: BBox
    rotation: float = 0.0
    confidence: Optional[float] = None
    font_info: Optional[FontInfo] = None


class Page(BaseModel):
    page: int
    width: float
    height: float
    texts: List[TextBlock] = Field(default_factory=list)


class PdfObjectInfo(BaseModel):
    obj_id: int
    obj_type: str
    generation: int = 0
    raw: Optional[str] = None


class DocumentMetadata(BaseModel):
    source_type: SourceType
    creator: Optional[str] = None
    producer: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    fonts: List[FontInfo] = Field(default_factory=list)
    pdf_objects: List[PdfObjectInfo] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


if PYDANTIC_V2:

    class Document(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        path: Path
        pages: List[Page] = Field(default_factory=list)
        metadata: Optional[DocumentMetadata] = None
else:

    class Document(BaseModel):
        path: Path
        pages: List[Page] = Field(default_factory=list)
        metadata: Optional[DocumentMetadata] = None

        class Config:
            arbitrary_types_allowed = True
