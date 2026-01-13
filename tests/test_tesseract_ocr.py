"""Unit tests for Tesseract OCR extractor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xtra.models import ExtractorType


class TestTesseractOcrExtractor:
    """Unit tests for TesseractOcrExtractor."""

    def test_get_metadata(self) -> None:
        with (
            patch("xtra.extractors.tesseract_ocr.pytesseract"),
            patch("xtra.extractors._image_loader.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_image.open.return_value = mock_img

            from xtra.extractors.tesseract_ocr import TesseractOcrExtractor

            extractor = TesseractOcrExtractor(Path("/fake/image.png"))
            metadata = extractor.get_metadata()

            assert metadata.source_type == ExtractorType.TESSERACT
            assert metadata.extra["ocr_engine"] == "tesseract"
            assert "languages" in metadata.extra

    def test_get_page_count(self) -> None:
        with (
            patch("xtra.extractors.tesseract_ocr.pytesseract"),
            patch("xtra.extractors._image_loader.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_image.open.return_value = mock_img

            from xtra.extractors.tesseract_ocr import TesseractOcrExtractor

            extractor = TesseractOcrExtractor(Path("/fake/image.png"))
            assert extractor.get_page_count() == 1

    def test_extract_page_success(self) -> None:
        with (
            patch("xtra.extractors.tesseract_ocr.pytesseract") as mock_pytesseract,
            patch("xtra.extractors._image_loader.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (800, 600)
            mock_image.open.return_value = mock_img

            # Mock pytesseract.image_to_data output (TSV format as dict)
            mock_pytesseract.image_to_data.return_value = {
                "level": [1, 2, 3, 4, 5, 5],
                "page_num": [1, 1, 1, 1, 1, 1],
                "block_num": [0, 1, 1, 1, 1, 1],
                "par_num": [0, 0, 1, 1, 1, 1],
                "line_num": [0, 0, 0, 1, 1, 1],
                "word_num": [0, 0, 0, 0, 1, 2],
                "left": [0, 10, 10, 10, 10, 100],
                "top": [0, 20, 20, 20, 20, 20],
                "width": [800, 200, 200, 200, 80, 90],
                "height": [600, 50, 50, 50, 30, 30],
                "conf": [-1, -1, -1, -1, 95.5, 87.2],
                "text": ["", "", "", "", "Hello", "World"],
            }
            mock_pytesseract.Output.DICT = "dict"

            from xtra.extractors.tesseract_ocr import TesseractOcrExtractor

            extractor = TesseractOcrExtractor(Path("/fake/image.png"))
            result = extractor.extract_page(0)

            assert result.success is True
            assert result.page.page == 0
            # Dimensions converted from pixels to points (default) at 200 DPI
            assert result.page.width == 288.0  # 800 * (72/200)
            assert result.page.height == 216.0  # 600 * (72/200)
            assert len(result.page.texts) == 2

            # Check first word - bbox converted from pixels to points
            assert result.page.texts[0].text == "Hello"
            assert result.page.texts[0].bbox.x0 == pytest.approx(3.6, rel=0.01)  # 10 * (72/200)
            assert result.page.texts[0].bbox.y0 == pytest.approx(7.2, rel=0.01)  # 20 * (72/200)
            assert result.page.texts[0].bbox.x1 == pytest.approx(32.4, rel=0.01)  # 90 * (72/200)
            assert result.page.texts[0].bbox.y1 == pytest.approx(18.0, rel=0.01)  # 50 * (72/200)
            assert result.page.texts[0].confidence == pytest.approx(0.955, rel=0.01)

            # Check second word
            assert result.page.texts[1].text == "World"
            assert result.page.texts[1].confidence == pytest.approx(0.872, rel=0.01)

    def test_extract_page_out_of_range(self) -> None:
        with (
            patch("xtra.extractors.tesseract_ocr.pytesseract"),
            patch("xtra.extractors._image_loader.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_image.open.return_value = mock_img

            from xtra.extractors.tesseract_ocr import TesseractOcrExtractor

            extractor = TesseractOcrExtractor(Path("/fake/image.png"))
            result = extractor.extract_page(5)

            assert result.success is False
            assert result.error is not None
            assert "out of range" in result.error.lower()

    def test_extract_page_filters_empty_text(self) -> None:
        with (
            patch("xtra.extractors.tesseract_ocr.pytesseract") as mock_pytesseract,
            patch("xtra.extractors._image_loader.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (800, 600)
            mock_image.open.return_value = mock_img

            # Mock with empty text entries
            mock_pytesseract.image_to_data.return_value = {
                "level": [5, 5, 5],
                "page_num": [1, 1, 1],
                "block_num": [1, 1, 1],
                "par_num": [1, 1, 1],
                "line_num": [1, 1, 1],
                "word_num": [1, 2, 3],
                "left": [10, 50, 100],
                "top": [20, 20, 20],
                "width": [30, 40, 50],
                "height": [25, 25, 25],
                "conf": [90.0, -1, 85.0],
                "text": ["Hello", "", "World"],
            }
            mock_pytesseract.Output.DICT = "dict"

            from xtra.extractors.tesseract_ocr import TesseractOcrExtractor

            extractor = TesseractOcrExtractor(Path("/fake/image.png"))
            result = extractor.extract_page(0)

            assert result.success is True
            # Only non-empty text should be included
            assert len(result.page.texts) == 2
            texts = [t.text for t in result.page.texts]
            assert "Hello" in texts
            assert "World" in texts

    def test_extract_page_filters_low_confidence(self) -> None:
        with (
            patch("xtra.extractors.tesseract_ocr.pytesseract") as mock_pytesseract,
            patch("xtra.extractors._image_loader.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (800, 600)
            mock_image.open.return_value = mock_img

            # Mock with negative confidence (Tesseract uses -1 for non-text elements)
            mock_pytesseract.image_to_data.return_value = {
                "level": [5, 5],
                "page_num": [1, 1],
                "block_num": [1, 1],
                "par_num": [1, 1],
                "line_num": [1, 1],
                "word_num": [1, 2],
                "left": [10, 50],
                "top": [20, 20],
                "width": [30, 40],
                "height": [25, 25],
                "conf": [-1, 90.0],
                "text": ["Block", "Word"],
            }
            mock_pytesseract.Output.DICT = "dict"

            from xtra.extractors.tesseract_ocr import TesseractOcrExtractor

            extractor = TesseractOcrExtractor(Path("/fake/image.png"))
            result = extractor.extract_page(0)

            assert result.success is True
            # Only positive confidence text should be included
            assert len(result.page.texts) == 1
            assert result.page.texts[0].text == "Word"

    def test_custom_languages(self) -> None:
        with (
            patch("xtra.extractors.tesseract_ocr.pytesseract") as mock_pytesseract,
            patch("xtra.extractors._image_loader.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_image.open.return_value = mock_img

            mock_pytesseract.image_to_data.return_value = {
                "level": [],
                "page_num": [],
                "block_num": [],
                "par_num": [],
                "line_num": [],
                "word_num": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": [],
                "text": [],
            }
            mock_pytesseract.Output.DICT = "dict"

            from xtra.extractors.tesseract_ocr import TesseractOcrExtractor

            extractor = TesseractOcrExtractor(Path("/fake/image.png"), languages=["en", "fr", "de"])
            extractor.extract_page(0)

            # Check that pytesseract was called with converted language string
            call_args = mock_pytesseract.image_to_data.call_args
            assert call_args[1]["lang"] == "eng+fra+deu"

    def test_close(self) -> None:
        with (
            patch("xtra.extractors.tesseract_ocr.pytesseract"),
            patch("xtra.extractors._image_loader.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_image.open.return_value = mock_img

            from xtra.extractors.tesseract_ocr import TesseractOcrExtractor

            extractor = TesseractOcrExtractor(Path("/fake/image.png"))
            # close() should not raise
            extractor.close()

    def test_init_with_dpi(self) -> None:
        with (
            patch("xtra.extractors.tesseract_ocr.pytesseract"),
            patch("xtra.extractors._image_loader.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_image.open.return_value = mock_img

            from xtra.extractors.tesseract_ocr import TesseractOcrExtractor

            extractor = TesseractOcrExtractor(Path("/fake/image.png"), dpi=300)
            assert extractor.dpi == 300
            assert not extractor._images.is_pdf


class TestTesseractOcrExtractorWithPdf:
    """Unit tests for TesseractOcrExtractor with PDF files."""

    def test_get_metadata_with_pdf(self) -> None:
        with (
            patch("xtra.extractors.tesseract_ocr.pytesseract"),
            patch("xtra.extractors._image_loader.pdfium") as mock_pdfium,
        ):
            mock_pdf = MagicMock()
            mock_page = MagicMock()
            mock_bitmap = MagicMock()
            mock_pil_img = MagicMock()
            mock_pil_img.size = (800, 600)

            mock_bitmap.to_pil.return_value = mock_pil_img
            mock_page.render.return_value = mock_bitmap
            mock_pdf.__iter__ = lambda self: iter([mock_page])
            mock_pdfium.PdfDocument.return_value = mock_pdf

            from xtra.extractors.tesseract_ocr import TesseractOcrExtractor

            extractor = TesseractOcrExtractor(Path("/fake/document.pdf"))
            metadata = extractor.get_metadata()

            assert metadata.source_type == ExtractorType.TESSERACT
            assert metadata.extra["ocr_engine"] == "tesseract"
            assert metadata.extra["dpi"] == 200

    def test_get_page_count_pdf(self) -> None:
        with (
            patch("xtra.extractors.tesseract_ocr.pytesseract"),
            patch("xtra.extractors._image_loader.pdfium") as mock_pdfium,
        ):
            mock_pdf = MagicMock()
            mock_pages = [MagicMock(), MagicMock(), MagicMock()]
            for page in mock_pages:
                mock_bitmap = MagicMock()
                mock_pil_img = MagicMock()
                mock_pil_img.size = (800, 600)
                mock_bitmap.to_pil.return_value = mock_pil_img
                page.render.return_value = mock_bitmap

            mock_pdf.__iter__ = lambda self: iter(mock_pages)
            mock_pdfium.PdfDocument.return_value = mock_pdf

            from xtra.extractors.tesseract_ocr import TesseractOcrExtractor

            extractor = TesseractOcrExtractor(Path("/fake/document.pdf"))
            assert extractor.get_page_count() == 3

    def test_extract_page_success_pdf(self) -> None:
        with (
            patch("xtra.extractors.tesseract_ocr.pytesseract") as mock_pytesseract,
            patch("xtra.extractors._image_loader.pdfium") as mock_pdfium,
        ):
            mock_pdf = MagicMock()
            mock_page = MagicMock()
            mock_bitmap = MagicMock()
            mock_pil_img = MagicMock()
            mock_pil_img.size = (1200, 900)

            mock_bitmap.to_pil.return_value = mock_pil_img
            mock_page.render.return_value = mock_bitmap
            mock_pdf.__iter__ = lambda self: iter([mock_page])
            mock_pdfium.PdfDocument.return_value = mock_pdf

            mock_pytesseract.image_to_data.return_value = {
                "level": [5],
                "page_num": [1],
                "block_num": [1],
                "par_num": [1],
                "line_num": [1],
                "word_num": [1],
                "left": [50],
                "top": [100],
                "width": [200],
                "height": [50],
                "conf": [92.5],
                "text": ["PDF"],
            }
            mock_pytesseract.Output.DICT = "dict"

            from xtra.extractors.tesseract_ocr import TesseractOcrExtractor

            extractor = TesseractOcrExtractor(Path("/fake/document.pdf"))
            result = extractor.extract_page(0)

            assert result.success is True
            assert result.page.page == 0
            # Dimensions converted from pixels to points (default) at 200 DPI
            assert result.page.width == 432.0  # 1200 * (72/200)
            assert result.page.height == 324.0  # 900 * (72/200)
            assert len(result.page.texts) == 1
            assert result.page.texts[0].text == "PDF"

    def test_custom_dpi_pdf(self) -> None:
        with (
            patch("xtra.extractors.tesseract_ocr.pytesseract"),
            patch("xtra.extractors._image_loader.pdfium") as mock_pdfium,
        ):
            mock_pdf = MagicMock()
            mock_page = MagicMock()
            mock_bitmap = MagicMock()
            mock_pil_img = MagicMock()
            mock_pil_img.size = (800, 600)

            mock_bitmap.to_pil.return_value = mock_pil_img
            mock_page.render.return_value = mock_bitmap
            mock_pdf.__iter__ = lambda self: iter([mock_page])
            mock_pdfium.PdfDocument.return_value = mock_pdf

            from xtra.extractors.tesseract_ocr import TesseractOcrExtractor

            extractor = TesseractOcrExtractor(Path("/fake/document.pdf"), dpi=300)
            metadata = extractor.get_metadata()

            assert metadata.extra["dpi"] == 300
            # Check render was called with correct scale (300/72)
            mock_page.render.assert_called_once()
            call_kwargs = mock_page.render.call_args[1]
            assert call_kwargs["scale"] == pytest.approx(300 / 72, rel=0.01)
