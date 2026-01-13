from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from ..models import Document, DocumentMetadata, Page


@dataclass
class ExtractionResult:
    """Result of extracting a single page."""

    page: Page
    success: bool
    error: Optional[str] = None


class BaseExtractor(ABC):
    """Base class for document extractors."""

    def __init__(self, path: Path) -> None:
        self.path = path

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

    def __enter__(self) -> "BaseExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
