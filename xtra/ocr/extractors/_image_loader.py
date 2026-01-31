"""Image loading for OCR extractors."""

from __future__ import annotations

import threading
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


class ImageLoader:
    """Loads and manages images from files for OCR processing.

    Handles both single images and PDFs (converting pages to images lazily).
    Use as a composable component in OCR extractors.

    PDF pages are only converted to images when requested via get_page(),
    and cached for subsequent access.
    """

    def __init__(self, path: Path, dpi: int = 200) -> None:
        """Initialize image loader.

        Args:
            path: Path to the image or PDF file.
            dpi: DPI for PDF-to-image conversion. Default 200.
        """
        self.path = path
        self.dpi = dpi
        self.is_pdf = path.suffix.lower() == ".pdf"

        # Lazy loading state
        self._pdf: pdfium.PdfDocument | None = None
        self._page_count: int | None = None
        self._image_cache: dict[int, Image.Image] = {}
        self._lock = threading.Lock()

    @property
    def page_count(self) -> int:
        """Return number of pages/images (lazy loaded for PDFs)."""
        with self._lock:
            if self._page_count is None:
                if self.is_pdf:
                    self._open_pdf()
                    self._page_count = len(self._pdf)  # type: ignore[arg-type]
                else:
                    self._page_count = 1
            return self._page_count

    def _open_pdf(self) -> None:
        """Open PDF document if not already open."""
        if self._pdf is None and self.is_pdf:
            self._pdf = pdfium.PdfDocument(self.path)

    def _render_page(self, page: int) -> Image.Image:
        """Render a single PDF page to PIL Image."""
        self._open_pdf()
        if self._pdf is None:
            raise RuntimeError("PDF not loaded")

        scale = self.dpi / 72.0
        pdf_page = self._pdf[page]
        bitmap = pdf_page.render(scale=scale)
        return bitmap.to_pil()

    def get_page(self, page: int) -> Image.Image:
        """Get a specific page image (lazy loaded and cached).

        Thread-safe: uses internal lock for parallel access.

        Args:
            page: Zero-indexed page number.

        Returns:
            PIL Image for the requested page.

        Raises:
            IndexError: If page is out of range.
        """
        if page >= self.page_count:
            raise IndexError(f"Page {page} out of range (have {self.page_count} pages)")

        with self._lock:
            # Return cached image if available
            if page in self._image_cache:
                return self._image_cache[page]

            # Load and cache image
            if self.is_pdf:
                img = self._render_page(page)
            else:
                img = Image.open(self.path)

            self._image_cache[page] = img
            return img

    def close(self) -> None:
        """Close image handles and release resources."""
        # Close cached images
        for img in self._image_cache.values():
            try:
                img.close()
            except Exception:  # noqa: S110
                pass
        self._image_cache = {}

        # Close PDF document
        if self._pdf is not None:
            try:
                self._pdf.close()
            except Exception:  # noqa: S110
                pass
            self._pdf = None

        self._page_count = None
