import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


from xtra.extractors.ocr import (
    OcrExtractor,
    PdfToImageOcrExtractor,
    _reader_cache,
    get_reader,
)
from xtra.models import SourceType


TEST_DATA_DIR = Path(__file__).parent / "data"


def test_get_reader_caches() -> None:
    # Clear cache first
    _reader_cache.clear()

    with patch("xtra.extractors.ocr.easyocr.Reader") as mock_reader:
        mock_reader.return_value = MagicMock()

        reader1 = get_reader(["en"], gpu=False)
        reader2 = get_reader(["en"], gpu=False)

        # Should only create reader once
        assert mock_reader.call_count == 1
        assert reader1 is reader2


def test_get_reader_different_languages() -> None:
    _reader_cache.clear()

    with patch("xtra.extractors.ocr.easyocr.Reader") as mock_reader:
        mock_reader.return_value = MagicMock()

        get_reader(["en"], gpu=False)
        get_reader(["it"], gpu=False)

        # Should create separate readers for different languages
        assert mock_reader.call_count == 2


class TestOcrExtractor:
    def test_init_default_languages(self) -> None:
        with patch("xtra.extractors.ocr.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_open.return_value = mock_img

            extractor = OcrExtractor(Path("/tmp/test.png"))
            assert extractor.languages == ["en"]
            assert extractor.gpu is False

    def test_init_custom_languages(self) -> None:
        with patch("xtra.extractors.ocr.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_open.return_value = mock_img

            extractor = OcrExtractor(Path("/tmp/test.png"), languages=["en", "it"], gpu=True)
            assert extractor.languages == ["en", "it"]
            assert extractor.gpu is True

    def test_get_page_count(self) -> None:
        with patch("xtra.extractors.ocr.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_open.return_value = mock_img

            extractor = OcrExtractor(Path("/tmp/test.png"))
            assert extractor.get_page_count() == 1

    def test_get_metadata(self) -> None:
        with patch("xtra.extractors.ocr.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_open.return_value = mock_img

            extractor = OcrExtractor(Path("/tmp/test.png"), languages=["en", "it"])
            metadata = extractor.get_metadata()

            assert metadata.source_type == SourceType.OCR
            assert metadata.extra["ocr_engine"] == "easyocr"
            assert metadata.extra["languages"] == ["en", "it"]

    def test_extract_page_success(self) -> None:
        _reader_cache.clear()

        with patch("xtra.extractors.ocr.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.size = (800, 600)
            mock_open.return_value = mock_img

            with patch("xtra.extractors.ocr.easyocr.Reader") as mock_reader_cls:
                mock_reader = MagicMock()
                mock_reader.readtext.return_value = [
                    ([[0, 0], [100, 0], [100, 20], [0, 20]], "Hello", 0.95),
                ]
                mock_reader_cls.return_value = mock_reader

                extractor = OcrExtractor(Path("/tmp/test.png"))
                result = extractor.extract_page(0)

                assert result.success
                assert len(result.page.texts) == 1
                assert result.page.texts[0].text == "Hello"
                assert result.page.texts[0].confidence == 0.95

    def test_extract_page_out_of_range(self) -> None:
        with patch("xtra.extractors.ocr.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_open.return_value = mock_img

            extractor = OcrExtractor(Path("/tmp/test.png"))
            result = extractor.extract_page(5)

            assert not result.success
            assert result.error is not None
            assert "out of range" in result.error.lower()

    def test_polygon_to_bbox_and_rotation(self) -> None:
        with patch("xtra.extractors.ocr.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_open.return_value = mock_img

            extractor = OcrExtractor(Path("/tmp/test.png"))
            polygon = [[0, 0], [100, 0], [100, 20], [0, 20]]
            bbox, rotation = extractor._polygon_to_bbox_and_rotation(polygon)

            assert bbox.x0 == 0
            assert bbox.y0 == 0
            assert bbox.x1 == 100
            assert bbox.y1 == 20
            assert rotation == 0.0

    def test_convert_results(self) -> None:
        with patch("xtra.extractors.ocr.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_open.return_value = mock_img

            extractor = OcrExtractor(Path("/tmp/test.png"))
            results = [
                ([[0, 0], [50, 0], [50, 10], [0, 10]], "Text1", 0.9),
                ([[0, 20], [60, 20], [60, 30], [0, 30]], "Text2", 0.8),
            ]
            blocks = extractor._convert_results(results)

            assert len(blocks) == 2
            assert blocks[0].text == "Text1"
            assert blocks[0].confidence == 0.9
            assert blocks[1].text == "Text2"


class TestPdfToImageOcrExtractor:
    def test_init_default_values(self) -> None:
        extractor = PdfToImageOcrExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf")
        assert extractor.languages == ["en"]
        assert extractor.gpu is False
        assert extractor.dpi == 200
        assert extractor.get_page_count() == 2

    def test_get_metadata(self) -> None:
        extractor = PdfToImageOcrExtractor(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf", languages=["en", "it"], dpi=300
        )
        metadata = extractor.get_metadata()

        assert metadata.source_type == SourceType.PDF_OCR
        assert metadata.extra["ocr_engine"] == "easyocr"
        assert metadata.extra["dpi"] == 300

    def test_get_page_count(self) -> None:
        extractor = PdfToImageOcrExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf")
        assert extractor.get_page_count() == 2

    def test_extract_page_out_of_range(self) -> None:
        extractor = PdfToImageOcrExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf")
        result = extractor.extract_page(5)

        assert not result.success
        assert result.error is not None
        assert "out of range" in result.error.lower()

    def test_convert_results(self) -> None:
        extractor = PdfToImageOcrExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf")
        results = [
            ([[0, 0], [50, 0], [50, 10], [0, 10]], "Text1", 0.9),
        ]
        blocks = extractor._convert_results(results)

        assert len(blocks) == 1
        assert blocks[0].text == "Text1"
        assert blocks[0].confidence == 0.9

    def test_polygon_to_bbox_and_rotation(self) -> None:
        extractor = PdfToImageOcrExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf")
        polygon = [[10, 10], [110, 10], [110, 30], [10, 30]]
        bbox, rotation = extractor._polygon_to_bbox_and_rotation(polygon)

        assert bbox.x0 == 10
        assert bbox.y0 == 10
        assert bbox.x1 == 110
        assert bbox.y1 == 30
        assert rotation == 0.0
