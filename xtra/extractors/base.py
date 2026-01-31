from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Self

from xtra.coordinates import CoordinateConverter
from xtra.models import CoordinateUnit, Document, ExtractorMetadata, Page


class ExecutorType(str, Enum):
    """Type of executor for parallel extraction."""

    THREAD = "thread"
    PROCESS = "process"


@dataclass
class PageExtractionResult:
    """Result of extracting a single page."""

    page: Page
    success: bool
    error: str | None = None


@dataclass
class ExtractionResult:
    """Result of document extraction with all page results.

    Contains the extracted document (successful pages only) and
    detailed results for each requested page including any errors.
    """

    document: Document
    page_results: list[PageExtractionResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if all requested pages were extracted successfully."""
        return all(r.success for r in self.page_results)

    @property
    def failed_pages(self) -> list[PageExtractionResult]:
        """List of failed page extraction results."""
        return [r for r in self.page_results if not r.success]

    @property
    def errors(self) -> list[tuple[int, str]]:
        """List of (page_number, error_message) for failed pages."""
        return [(r.page.page, r.error or "") for r in self.page_results if not r.success]


class BaseExtractor(ABC):
    """Base class for document extractors."""

    def __init__(
        self,
        path: Path | str,
        output_unit: CoordinateUnit = CoordinateUnit.POINTS,
    ) -> None:
        self.path = Path(path) if isinstance(path, str) else path
        self.output_unit = output_unit

    @abstractmethod
    def get_page_count(self) -> int:
        """Return total number of pages in document."""
        ...

    @abstractmethod
    def extract_page(self, page: int) -> PageExtractionResult:
        """Extract a single page by number (0-indexed)."""
        ...

    @abstractmethod
    def get_extractor_metadata(self) -> ExtractorMetadata:
        """Return metadata about the extractor and processing."""
        ...

    def _extract_pages(
        self,
        pages: Sequence[int] | None = None,
        max_workers: int = 1,
        executor: ExecutorType = ExecutorType.THREAD,
    ) -> list[PageExtractionResult]:
        """Extract multiple pages with optional parallel processing.

        Internal method that returns per-page results.

        Args:
            pages: Sequence of page numbers to extract (0-indexed).
                   If None, extracts all pages.
            max_workers: Number of parallel workers. 1 means sequential.
            executor: Type of executor to use for parallel extraction.

        Returns:
            List of PageExtractionResult in page order.
        """
        if pages is None:
            pages = range(self.get_page_count())
        pages_list = list(pages)

        # Sequential execution (backward compatible)
        if max_workers <= 1 or len(pages_list) <= 1:
            return [self.extract_page(n) for n in pages_list]

        # Select executor class
        executor_class = (
            ProcessPoolExecutor if executor == ExecutorType.PROCESS else ThreadPoolExecutor
        )

        # Parallel execution with ordering preserved
        results: list[PageExtractionResult | None] = [None] * len(pages_list)
        with executor_class(max_workers=max_workers) as pool:
            future_to_idx = {pool.submit(self.extract_page, p): i for i, p in enumerate(pages_list)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = PageExtractionResult(
                        page=Page(page=pages_list[idx], width=0, height=0, texts=[]),
                        success=False,
                        error=str(e),
                    )
        return results  # type: ignore[return-value]

    def extract(
        self,
        pages: Sequence[int] | None = None,
        max_workers: int = 1,
        executor: ExecutorType = ExecutorType.THREAD,
    ) -> ExtractionResult:
        """Extract document with optional page selection and parallel processing.

        Args:
            pages: Sequence of page numbers to extract (0-indexed).
                   If None, extracts all pages.
            max_workers: Number of parallel workers. 1 means sequential.
            executor: Type of executor (ExecutorType.THREAD or ExecutorType.PROCESS).

        Returns:
            ExtractionResult containing the document and all page results.
        """
        page_results = self._extract_pages(pages, max_workers=max_workers, executor=executor)
        extracted_pages = [r.page for r in page_results if r.success]
        metadata = self.get_extractor_metadata()
        document = Document(path=self.path, pages=extracted_pages, metadata=metadata)
        return ExtractionResult(document=document, page_results=page_results)

    async def _extract_pages_async(
        self,
        pages: Sequence[int] | None = None,
        max_workers: int = 1,
    ) -> list[PageExtractionResult]:
        """Async version of _extract_pages - runs extraction in thread pool.

        Internal method that returns per-page results.

        Args:
            pages: Sequence of page numbers to extract (0-indexed).
                   If None, extracts all pages.
            max_workers: Number of parallel workers. 1 means sequential.

        Returns:
            List of PageExtractionResult in page order.
        """
        if pages is None:
            pages = range(self.get_page_count())
        pages_list = list(pages)

        loop = asyncio.get_event_loop()

        # Sequential async execution
        if max_workers <= 1 or len(pages_list) <= 1:
            results = []
            for n in pages_list:
                result = await loop.run_in_executor(None, self.extract_page, n)
                results.append(result)
            return results

        # Parallel async execution
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            tasks = [loop.run_in_executor(pool, self.extract_page, p) for p in pages_list]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to PageExtractionResult failures
        results = []
        for i, r in enumerate(raw_results):
            if isinstance(r, PageExtractionResult):
                results.append(r)
            else:
                results.append(
                    PageExtractionResult(
                        page=Page(page=pages_list[i], width=0, height=0, texts=[]),
                        success=False,
                        error=str(r),
                    )
                )
        return results

    async def extract_async(
        self,
        pages: Sequence[int] | None = None,
        max_workers: int = 1,
    ) -> ExtractionResult:
        """Async version of extract - runs extraction in thread pool.

        Args:
            pages: Sequence of page numbers to extract (0-indexed).
                   If None, extracts all pages.
            max_workers: Number of parallel workers. 1 means sequential.

        Returns:
            ExtractionResult containing the document and all page results.
        """
        page_results = await self._extract_pages_async(pages, max_workers=max_workers)
        extracted_pages = [r.page for r in page_results if r.success]
        metadata = self.get_extractor_metadata()
        document = Document(path=self.path, pages=extracted_pages, metadata=metadata)
        return ExtractionResult(document=document, page_results=page_results)

    def close(self) -> None:  # noqa: B027 - Intentionally empty, subclasses override if needed
        """Clean up resources. Override in subclasses if needed."""

    def _convert_page(
        self,
        page: Page,
        source_unit: CoordinateUnit,
        dpi: float | None = None,
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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
