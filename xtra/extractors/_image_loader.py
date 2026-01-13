"""Image loading for OCR extractors."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pypdfium2 as pdfium
from PIL import Image


class ImageLoader:
    """Loads and manages images from files for OCR processing.

    Handles both single images and PDFs (converting pages to images).
    Use as a composable component in OCR extractors.
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
        self._images: List[Image.Image] = []
        self._load()

    def _load(self) -> None:
        """Load image(s) from path."""
        if self.is_pdf:
            self._images = self._load_pdf_as_images()
        else:
            self._images = [Image.open(self.path)]

    def _load_pdf_as_images(self) -> List[Image.Image]:
        """Convert PDF pages to PIL Images at the specified DPI."""
        images: List[Image.Image] = []
        pdf = pdfium.PdfDocument(self.path)
        scale = self.dpi / 72.0

        for page in pdf:
            bitmap = page.render(scale=scale)
            images.append(bitmap.to_pil())

        pdf.close()
        return images

    @property
    def page_count(self) -> int:
        """Return number of pages/images loaded."""
        return len(self._images)

    def get_page(self, page: int) -> Image.Image:
        """Get a specific page image.

        Args:
            page: Zero-indexed page number.

        Returns:
            PIL Image for the requested page.

        Raises:
            IndexError: If page is out of range.
        """
        if page >= len(self._images):
            raise IndexError(f"Page {page} out of range (have {len(self._images)} pages)")
        return self._images[page]

    def close(self) -> None:
        """Close image handles and release resources."""
        for img in self._images:
            try:
                img.close()
            except Exception:  # noqa: S110
                pass
        self._images = []
