"""PaddleOCR extractor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xtra.extractors.base import BaseExtractor, PageExtractionResult
from xtra.models import (
    CoordinateUnit,
    ExtractorMetadata,
    ExtractorType,
    Page,
    Table,
)
from xtra.ocr.adapters.paddle_ocr import PaddleOCRAdapter
from xtra.utils.image_loader import ImageLoader

if TYPE_CHECKING:
    from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

_ocr_cache: dict[tuple, Any] = {}
_ppstructure_cache: dict[tuple, Any] = {}
_paddle_major_version: dict[str, int] = {}

# PaddleOCR 3.x introduced breaking API changes
PADDLEOCR_V3_MAJOR = 3


def _get_paddle_major_version() -> int:
    """Get the major version of PaddleOCR."""
    if "version" not in _paddle_major_version:
        import paddleocr

        _paddle_major_version["version"] = int(paddleocr.__version__.split(".")[0])
    return _paddle_major_version["version"]


def _check_paddleocr_installed() -> None:
    """Check if paddleocr is installed, raise ImportError with helpful message if not."""
    try:
        from paddleocr import PaddleOCR  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PaddleOCR is not installed. Install it with: pip install xtra[paddle]"
        ) from e


def _check_ppstructure_installed() -> None:
    """Check if PPStructure is available."""
    try:
        from paddleocr import PPStructure  # type: ignore[attr-defined] # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PPStructure is not available. Install with: pip install 'paddleocr>=2.6'"
        ) from e


def get_ppstructure(lang: str, use_gpu: bool) -> Any:
    """Get or create a cached PPStructure instance for table extraction."""
    _check_ppstructure_installed()
    from paddleocr import PPStructure  # type: ignore[attr-defined]

    key = ("ppstructure", lang, use_gpu)
    if key not in _ppstructure_cache:
        major_version = _get_paddle_major_version()
        if major_version >= PADDLEOCR_V3_MAJOR:
            # PaddleOCR 3.x API
            _ppstructure_cache[key] = PPStructure(lang=lang, return_ocr_result_in_table=True)
        else:
            # PaddleOCR 2.x API
            _ppstructure_cache[key] = PPStructure(
                lang=lang, use_gpu=use_gpu, show_log=False, return_ocr_result_in_table=True
            )
    return _ppstructure_cache[key]


def get_paddle_ocr(lang: str, use_gpu: bool) -> PaddleOCR:
    """Get or create a cached PaddleOCR instance."""
    from paddleocr import PaddleOCR

    key = (lang, use_gpu)
    if key not in _ocr_cache:
        major_version = _get_paddle_major_version()
        if major_version >= PADDLEOCR_V3_MAJOR:
            # PaddleOCR 3.x API: use_angle_cls renamed to use_textline_orientation, no show_log
            _ocr_cache[key] = PaddleOCR(use_textline_orientation=True, lang=lang)
        else:
            # PaddleOCR 2.x API
            _ocr_cache[key] = PaddleOCR(
                use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False
            )
    return _ocr_cache[key]


class PaddleOcrExtractor(BaseExtractor):
    """Extract text from images or PDFs using PaddleOCR.

    Composes ImageLoader for image handling, PaddleOCR for OCR,
    and PaddleOCRAdapter for result conversion.

    PaddleOCR model is loaded lazily on first extraction and cached globally.
    """

    def __init__(
        self,
        path: Path | str,
        lang: str = "en",
        use_gpu: bool = False,
        dpi: int = 200,
        output_unit: CoordinateUnit = CoordinateUnit.POINTS,
    ) -> None:
        """Initialize PaddleOCR extractor.

        Args:
            path: Path to the image or PDF file (Path object or string).
            lang: Language code for OCR. Common values:
                  - "en" for English
                  - "ch" for Chinese
                  - "fr" for French
                  - "german" for German
                  - "japan" for Japanese
                  - "korean" for Korean
                  See PaddleOCR docs for full list.
            use_gpu: Whether to use GPU acceleration.
            dpi: DPI for PDF-to-image conversion. Default 200.
            output_unit: Coordinate unit for output. Default POINTS.
        """
        _check_paddleocr_installed()
        super().__init__(path, output_unit)
        self.lang = lang
        self.use_gpu = use_gpu
        self.dpi = dpi

        # Compose components (lazy - OCR loaded on first use)
        self._images = ImageLoader(self.path, dpi)
        self._adapter = PaddleOCRAdapter()

    def get_page_count(self) -> int:
        """Return number of pages/images loaded."""
        return self._images.page_count

    def extract_page(self, page: int) -> PageExtractionResult:
        """Extract text from a single image/page."""
        import numpy as np

        try:
            img = self._images.get_page(page)
            width, height = img.size

            # Run OCR pipeline (lazy load model)
            ocr = get_paddle_ocr(self.lang, self.use_gpu)
            img_array = np.array(img)

            # Use version-specific API
            major_version = _get_paddle_major_version()
            if major_version >= PADDLEOCR_V3_MAJOR:
                result = ocr.predict(img_array)
            else:
                result = ocr.ocr(img_array, cls=True)

            text_blocks = self._adapter.convert_result(result, major_version)

            result_page = Page(
                page=page,
                width=float(width),
                height=float(height),
                texts=text_blocks,
            )

            # Convert from native PIXELS to output_unit
            result_page = self._convert_page(result_page, CoordinateUnit.PIXELS, self.dpi)
            return PageExtractionResult(page=result_page, success=True)

        except Exception as e:
            logger.warning("Failed to extract page %d: %s", page, e)
            return PageExtractionResult(
                page=Page(page=page, width=0, height=0, texts=[]),
                success=False,
                error=str(e),
            )

    def extract_tables(
        self,
        pages: list[int] | None = None,
    ) -> list[Table]:
        """Extract tables from document using PPStructure.

        Args:
            pages: List of page numbers to extract (0-indexed).
                   If None, extracts from all pages.

        Returns:
            List of Table objects with page field indicating source page.
        """
        import numpy as np

        if pages is None:
            pages = list(range(self.get_page_count()))

        all_tables: list[Table] = []
        engine = get_ppstructure(self.lang, self.use_gpu)

        for page_num in pages:
            try:
                img = self._images.get_page(page_num)
                img_array = np.array(img)

                # PPStructure returns list of layout elements
                result = engine(img_array)

                for element in result:
                    if element.get("type") == "table":
                        table = self._adapter.convert_table_result(element, page=page_num)
                        all_tables.append(table)

            except Exception as e:
                logger.warning("Failed to extract tables from page %d: %s", page_num, e)

        return all_tables

    def get_extractor_metadata(self) -> ExtractorMetadata:
        """Return extractor metadata."""
        extra = {"ocr_engine": "paddleocr", "languages": self.lang}
        if self._images.is_pdf:
            extra["dpi"] = self.dpi
        return ExtractorMetadata(
            extractor_type=ExtractorType.PADDLE,
            extra=extra,
        )

    def close(self) -> None:
        """Release resources."""
        self._images.close()
