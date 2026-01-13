from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from ..coordinates import CoordinateConverter
from ..models import CoordinateUnit, Document, DocumentMetadata, Page


@dataclass
class ExtractionResult:
    """Result of extracting a single page."""

    page: Page
    success: bool
    error: Optional[str] = None


class BaseExtractor(ABC):
    """Base class for document extractors."""

    def __init__(
        self,
        path: Path,
        output_unit: CoordinateUnit = CoordinateUnit.POINTS,
    ) -> None:
        self.path = path
        self.output_unit = output_unit

    @abstractmethod
    def get_page_count(self) -> int:
        """Return total number of pages in document."""
        ...

    @abstractmethod
    def extract_page(self, page: int) -> ExtractionResult:
        """Extract a single page by number (0-indexed)."""
        ...

    @abstractmethod
    def get_metadata(self) -> DocumentMetadata:
        """Extract document metadata."""
        ...

    def extract_pages(self, pages: Optional[Sequence[int]] = None) -> List[ExtractionResult]:
        """Extract multiple pages. If pages is None, extract all pages."""
        if pages is None:
            pages = range(self.get_page_count())
        return [self.extract_page(n) for n in pages]

    def extract(self, pages: Optional[Sequence[int]] = None) -> Document:
        """Extract document with optional page selection."""
        results = self.extract_pages(pages)
        extracted_pages = [r.page for r in results if r.success]
        metadata = self.get_metadata()
        return Document(path=self.path, pages=extracted_pages, metadata=metadata)

    def close(self) -> None:
        """Clean up resources. Override in subclasses if needed."""
        pass

    def _convert_page(
        self,
        page: Page,
        source_unit: CoordinateUnit,
        dpi: Optional[float] = None,
    ) -> Page:
        """Convert page coordinates from source unit to output_unit.

        Args:
            page: The page with coordinates in source_unit.
            source_unit: The native unit of the source coordinates.
            dpi: DPI value for pixel conversions (required for PIXELS source/target).

        Returns:
            A new Page with coordinates converted to self.output_unit.
        """
        if source_unit == self.output_unit:
            # No conversion needed
            return page

        converter = CoordinateConverter(
            source_unit=source_unit,
            page_width=page.width,
            page_height=page.height,
            dpi=dpi,
        )
        return converter.convert_page(page, self.output_unit, target_dpi=dpi)

    def __enter__(self) -> "BaseExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
