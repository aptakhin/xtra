from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence

from xtra.coordinates import CoordinateConverter
from xtra.models import CoordinateUnit, Document, ExtractorMetadata, Page

ExecutorType = Literal["thread", "process"]


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
    def extract_page(self, page: int) -> ExtractionResult:
        """Extract a single page by number (0-indexed)."""
        ...

    @abstractmethod
    def get_extractor_metadata(self) -> ExtractorMetadata:
        """Return metadata about the extractor and processing."""
        ...

    def extract_pages(
        self,
        pages: Optional[Sequence[int]] = None,
        max_workers: int = 1,
        executor: ExecutorType = "thread",
    ) -> List[ExtractionResult]:
        """Extract multiple pages with optional parallel processing.

        Args:
            pages: Sequence of page numbers to extract (0-indexed).
                   If None, extracts all pages.
            max_workers: Number of parallel workers. 1 means sequential
                         (default, backward compatible). Values > 1 enable
                         parallel extraction.
            executor: Type of executor to use for parallel extraction.
                      "thread" (default) uses ThreadPoolExecutor - best for OCR
                      since C libraries release GIL and model cache is shared.
                      "process" uses ProcessPoolExecutor - true parallelism but
                      models are duplicated per worker (high memory usage).
                      Note: "process" requires the extractor to be picklable,
                      which may not work with all extractors (e.g., PDF extractor
                      with native file handles). Use "thread" for maximum
                      compatibility.

        Returns:
            List of ExtractionResult in page order (matching input pages order).
        """
        if pages is None:
            pages = range(self.get_page_count())
        pages_list = list(pages)

        # Sequential execution (backward compatible)
        if max_workers <= 1 or len(pages_list) <= 1:
            return [self.extract_page(n) for n in pages_list]

        # Select executor class
        executor_class = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor

        # Parallel execution with ordering preserved
        results: List[Optional[ExtractionResult]] = [None] * len(pages_list)
        with executor_class(max_workers=max_workers) as pool:
            future_to_idx = {pool.submit(self.extract_page, p): i for i, p in enumerate(pages_list)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = ExtractionResult(
                        page=Page(page=pages_list[idx], width=0, height=0, texts=[]),
                        success=False,
                        error=str(e),
                    )
        return results  # type: ignore[return-value]

    def extract(
        self,
        pages: Optional[Sequence[int]] = None,
        max_workers: int = 1,
        executor: ExecutorType = "thread",
    ) -> Document:
        """Extract document with optional page selection and parallel processing.

        Args:
            pages: Sequence of page numbers to extract (0-indexed).
                   If None, extracts all pages.
            max_workers: Number of parallel workers. 1 means sequential.
            executor: Type of executor ("thread" or "process").

        Returns:
            Document containing extracted pages and metadata.
        """
        results = self.extract_pages(pages, max_workers=max_workers, executor=executor)
        extracted_pages = [r.page for r in results if r.success]
        metadata = self.get_extractor_metadata()
        return Document(path=self.path, pages=extracted_pages, metadata=metadata)

    async def extract_pages_async(
        self,
        pages: Optional[Sequence[int]] = None,
        max_workers: int = 1,
    ) -> List[ExtractionResult]:
        """Async version of extract_pages - runs extraction in thread pool.

        Uses ThreadPoolExecutor internally since OCR operations are CPU-bound
        and release the GIL. This provides an async interface for integration
        with async applications.

        Args:
            pages: Sequence of page numbers to extract (0-indexed).
                   If None, extracts all pages.
            max_workers: Number of parallel workers. 1 means sequential.

        Returns:
            List of ExtractionResult in page order.
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

        # Convert exceptions to ExtractionResult failures
        results = []
        for i, r in enumerate(raw_results):
            if isinstance(r, ExtractionResult):
                results.append(r)
            else:
                results.append(
                    ExtractionResult(
                        page=Page(page=pages_list[i], width=0, height=0, texts=[]),
                        success=False,
                        error=str(r),
                    )
                )
        return results

    async def extract_async(
        self,
        pages: Optional[Sequence[int]] = None,
        max_workers: int = 1,
    ) -> Document:
        """Async version of extract - runs extraction in thread pool.

        Args:
            pages: Sequence of page numbers to extract (0-indexed).
                   If None, extracts all pages.
            max_workers: Number of parallel workers. 1 means sequential.

        Returns:
            Document containing extracted pages and metadata.
        """
        results = await self.extract_pages_async(pages, max_workers=max_workers)
        extracted_pages = [r.page for r in results if r.success]
        metadata = self.get_extractor_metadata()
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
