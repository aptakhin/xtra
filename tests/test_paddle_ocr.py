"""Unit tests for PaddleOCR extractor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xtra.models import ExtractorType


class TestPaddleOcrExtractor:
    """Unit tests for PaddleOcrExtractor."""

    def test_get_metadata(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR"),
            patch("xtra.extractors.paddle_ocr.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_image.open.return_value = mock_img

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/image.png"))
            metadata = extractor.get_metadata()

            assert metadata.source_type == ExtractorType.PADDLE
            assert metadata.extra["ocr_engine"] == "paddleocr"
            assert "languages" in metadata.extra

    def test_get_page_count(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR"),
            patch("xtra.extractors.paddle_ocr.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_image.open.return_value = mock_img

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/image.png"))
            assert extractor.get_page_count() == 1

    def test_extract_page_success(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR") as mock_paddle_class,
            patch("xtra.extractors.paddle_ocr.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (800, 600)
            mock_image.open.return_value = mock_img

            # Mock PaddleOCR output format: list of [bbox, (text, confidence)]
            # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (4 corners)
            mock_paddle = MagicMock()
            mock_paddle.ocr.return_value = [
                [
                    [[[10, 20], [90, 20], [90, 50], [10, 50]], ("Hello", 0.955)],
                    [[[100, 20], [190, 20], [190, 50], [100, 50]], ("World", 0.872)],
                ]
            ]
            mock_paddle_class.return_value = mock_paddle

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/image.png"))
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
            patch("xtra.extractors.paddle_ocr.PaddleOCR"),
            patch("xtra.extractors.paddle_ocr.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_image.open.return_value = mock_img

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/image.png"))
            result = extractor.extract_page(5)

            assert result.success is False
            assert result.error is not None
            assert "out of range" in result.error.lower()

    def test_extract_page_handles_none_result(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR") as mock_paddle_class,
            patch("xtra.extractors.paddle_ocr.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (800, 600)
            mock_image.open.return_value = mock_img

            # PaddleOCR can return None or [[None]] when no text detected
            mock_paddle = MagicMock()
            mock_paddle.ocr.return_value = [[None]]
            mock_paddle_class.return_value = mock_paddle

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/image.png"))
            result = extractor.extract_page(0)

            assert result.success is True
            assert len(result.page.texts) == 0

    def test_extract_page_handles_empty_result(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR") as mock_paddle_class,
            patch("xtra.extractors.paddle_ocr.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (800, 600)
            mock_image.open.return_value = mock_img

            mock_paddle = MagicMock()
            mock_paddle.ocr.return_value = [[]]
            mock_paddle_class.return_value = mock_paddle

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/image.png"))
            result = extractor.extract_page(0)

            assert result.success is True
            assert len(result.page.texts) == 0

    def test_custom_languages(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR") as mock_paddle_class,
            patch("xtra.extractors.paddle_ocr.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_image.open.return_value = mock_img

            mock_paddle = MagicMock()
            mock_paddle.ocr.return_value = [[]]
            mock_paddle_class.return_value = mock_paddle

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/image.png"), lang="ch")
            metadata = extractor.get_metadata()

            assert metadata.extra["languages"] == "ch"
            # Check PaddleOCR was initialized with correct language
            mock_paddle_class.assert_called_once()
            call_kwargs = mock_paddle_class.call_args[1]
            assert call_kwargs["lang"] == "ch"

    def test_close(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR"),
            patch("xtra.extractors.paddle_ocr.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_image.open.return_value = mock_img

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/image.png"))
            # close() should not raise
            extractor.close()

    def test_init_with_dpi(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR"),
            patch("xtra.extractors.paddle_ocr.Image") as mock_image,
        ):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_image.open.return_value = mock_img

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/image.png"), dpi=300)
            assert extractor.dpi == 300
            assert not extractor._is_pdf


class TestPaddleOcrExtractorWithPdf:
    """Unit tests for PaddleOcrExtractor with PDF files."""

    def test_get_metadata_with_pdf(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR"),
            patch("xtra.extractors.paddle_ocr.pdfium") as mock_pdfium,
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

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/document.pdf"))
            metadata = extractor.get_metadata()

            assert metadata.source_type == ExtractorType.PADDLE
            assert metadata.extra["ocr_engine"] == "paddleocr"
            assert metadata.extra["dpi"] == 200

    def test_get_page_count_pdf(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR"),
            patch("xtra.extractors.paddle_ocr.pdfium") as mock_pdfium,
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

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/document.pdf"))
            assert extractor.get_page_count() == 3

    def test_extract_page_success_pdf(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR") as mock_paddle_class,
            patch("xtra.extractors.paddle_ocr.pdfium") as mock_pdfium,
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

            mock_paddle = MagicMock()
            mock_paddle.ocr.return_value = [
                [
                    [[[50, 100], [250, 100], [250, 150], [50, 150]], ("PDF", 0.925)],
                ]
            ]
            mock_paddle_class.return_value = mock_paddle

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/document.pdf"))
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
            patch("xtra.extractors.paddle_ocr.PaddleOCR"),
            patch("xtra.extractors.paddle_ocr.pdfium") as mock_pdfium,
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

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/document.pdf"), dpi=300)
            metadata = extractor.get_metadata()

            assert metadata.extra["dpi"] == 300
            # Check render was called with correct scale (300/72)
            mock_page.render.assert_called_once()
            call_kwargs = mock_page.render.call_args[1]
            assert call_kwargs["scale"] == pytest.approx(300 / 72, rel=0.01)
