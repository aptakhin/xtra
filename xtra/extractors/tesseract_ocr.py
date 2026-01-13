"""Tesseract OCR extractor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pypdfium2 as pdfium
import pytesseract
from PIL import Image, UnidentifiedImageError

from ..models import (
    BBox,
    DocumentMetadata,
    Page,
    SourceType,
    TextBlock,
)
from .base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)


class TesseractOcrExtractor(BaseExtractor):
    """Extract text from images using Tesseract OCR."""

    def __init__(
        self,
        path: Path,
        languages: Optional[List[str]] = None,
    ) -> None:
        """Initialize Tesseract OCR extractor.

        Args:
            path: Path to the image file.
            languages: List of language codes (e.g., ["eng", "fra"]).
                       Defaults to ["eng"]. Use Tesseract language codes.
        """
        super().__init__(path)
        self.languages = languages or ["eng"]
        self._images: List[Image.Image] = []
        self._load_images()

    def _load_images(self) -> None:
        """Load image(s) from path. Single image = single page."""
        img = Image.open(self.path)
        self._images = [img]

    def get_page_count(self) -> int:
        return len(self._images)

    def extract_page(self, page: int) -> ExtractionResult:
        """Extract text from a single image/page using Tesseract."""
        try:
            if page >= len(self._images):
                raise IndexError(f"Page {page} out of range")

            img = self._images[page]
            width, height = img.size

            # Build language string for Tesseract (e.g., "eng+fra+deu")
            lang_str = "+".join(self.languages)

            # Get detailed OCR data with bounding boxes
            data = pytesseract.image_to_data(
                img, lang=lang_str, output_type=pytesseract.Output.DICT
            )

            text_blocks = self._convert_results(data)

            return ExtractionResult(
                page=Page(
                    page=page,
                    width=float(width),
                    height=float(height),
                    texts=text_blocks,
                ),
                success=True,
            )
        except (IndexError, UnidentifiedImageError, OSError, RuntimeError) as e:
            logger.warning("Failed to extract page %d with Tesseract: %s", page, e)
            return ExtractionResult(
                page=Page(page=page, width=0, height=0, texts=[]),
                success=False,
                error=str(e),
            )

    def get_metadata(self) -> DocumentMetadata:
        return DocumentMetadata(
            source_type=SourceType.TESSERACT,
            extra={"ocr_engine": "tesseract", "languages": self.languages},
        )

    def _convert_results(self, data: dict) -> List[TextBlock]:
        """Convert Tesseract output to TextBlocks."""
        blocks = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i]
            conf = data["conf"][i]

            # Skip empty text and low/negative confidence (non-text elements)
            if not text or not text.strip() or conf < 0:
                continue

            left = data["left"][i]
            top = data["top"][i]
            width = data["width"][i]
            height = data["height"][i]

            bbox = BBox(
                x0=float(left),
                y0=float(top),
                x1=float(left + width),
                y1=float(top + height),
            )

            # Tesseract confidence is 0-100, normalize to 0-1
            confidence = float(conf) / 100.0

            blocks.append(
                TextBlock(
                    text=text,
                    bbox=bbox,
                    rotation=0.0,  # Tesseract doesn't provide rotation per word
                    confidence=confidence,
                )
            )

        return blocks

    def close(self) -> None:
        """Close image handles."""
        for img in self._images:
            try:
                img.close()
            except Exception:  # noqa: S110
                pass
        self._images = []


class PdfToImageTesseractExtractor(BaseExtractor):
    """Extract text from PDF by converting pages to images and running Tesseract OCR."""

    def __init__(
        self,
        path: Path,
        languages: Optional[List[str]] = None,
        dpi: int = 200,
    ) -> None:
        """Initialize PDF to image Tesseract extractor.

        Args:
            path: Path to the PDF file.
            languages: List of language codes (e.g., ["eng", "fra"]).
                       Defaults to ["eng"]. Use Tesseract language codes.
            dpi: Resolution for rendering PDF pages to images. Default 200.
        """
        super().__init__(path)
        self.languages = languages or ["eng"]
        self.dpi = dpi
        self._images: List[Image.Image] = []
        self._load_pdf_as_images()

    def _load_pdf_as_images(self) -> None:
        """Convert PDF pages to images using pypdfium2."""
        pdf = pdfium.PdfDocument(self.path)
        scale = self.dpi / 72.0
        for page in pdf:
            bitmap = page.render(scale=scale)
            self._images.append(bitmap.to_pil())
        pdf.close()

    def get_page_count(self) -> int:
        return len(self._images)

    def extract_page(self, page: int) -> ExtractionResult:
        """Extract text from a single PDF page via Tesseract OCR."""
        try:
            if page >= len(self._images):
                raise IndexError(f"Page {page} out of range")

            img = self._images[page]
            width, height = img.size

            # Build language string for Tesseract
            lang_str = "+".join(self.languages)

            # Get detailed OCR data with bounding boxes
            data = pytesseract.image_to_data(
                img, lang=lang_str, output_type=pytesseract.Output.DICT
            )

            text_blocks = self._convert_results(data)

            return ExtractionResult(
                page=Page(
                    page=page,
                    width=float(width),
                    height=float(height),
                    texts=text_blocks,
                ),
                success=True,
            )
        except (IndexError, OSError, RuntimeError) as e:
            logger.warning("Failed to extract page %d via Tesseract OCR: %s", page, e)
            return ExtractionResult(
                page=Page(page=page, width=0, height=0, texts=[]),
                success=False,
                error=str(e),
            )

    def get_metadata(self) -> DocumentMetadata:
        return DocumentMetadata(
            source_type=SourceType.PDF_TESSERACT,
            extra={
                "ocr_engine": "tesseract",
                "languages": self.languages,
                "dpi": self.dpi,
            },
        )

    def _convert_results(self, data: dict) -> List[TextBlock]:
        """Convert Tesseract output to TextBlocks."""
        blocks = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i]
            conf = data["conf"][i]

            # Skip empty text and low/negative confidence
            if not text or not text.strip() or conf < 0:
                continue

            left = data["left"][i]
            top = data["top"][i]
            width = data["width"][i]
            height = data["height"][i]

            bbox = BBox(
                x0=float(left),
                y0=float(top),
                x1=float(left + width),
                y1=float(top + height),
            )

            confidence = float(conf) / 100.0

            blocks.append(
                TextBlock(
                    text=text,
                    bbox=bbox,
                    rotation=0.0,
                    confidence=confidence,
                )
            )

        return blocks

    def close(self) -> None:
        """Close image handles."""
        for img in self._images:
            try:
                img.close()
            except Exception:  # noqa: S110
                pass
        self._images = []
