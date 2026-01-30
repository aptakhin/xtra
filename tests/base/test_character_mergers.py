"""Tests for character merger strategies using fixture data."""

import pytest

from xtra.extractors.character_mergers import (
    BasicLineMerger,
    CharInfo,
    KeepCharacterMerger,
)

# Fixtures for testing - simulating characters extracted from a PDF page
# Page height is 792 points (standard US Letter)
PAGE_HEIGHT = 792.0


def make_char(char: str, bbox: tuple[float, float, float, float], index: int = 0) -> CharInfo:
    """Helper to create CharInfo with default rotation."""
    return CharInfo(char=char, bbox=bbox, rotation=0.0, index=index)


@pytest.fixture
def single_line_chars() -> list[CharInfo]:
    """Characters forming a single line: 'Hello'."""
    return [
        make_char("H", (10.0, 100.0, 20.0, 112.0), 0),
        make_char("e", (20.0, 100.0, 28.0, 112.0), 1),
        make_char("l", (28.0, 100.0, 34.0, 112.0), 2),
        make_char("l", (34.0, 100.0, 40.0, 112.0), 3),
        make_char("o", (40.0, 100.0, 50.0, 112.0), 4),
    ]


@pytest.fixture
def two_line_chars() -> list[CharInfo]:
    """Characters forming two lines: 'Hi' and 'There'."""
    return [
        # First line: "Hi" at y=100
        make_char("H", (10.0, 100.0, 20.0, 112.0), 0),
        make_char("i", (20.0, 100.0, 26.0, 112.0), 1),
        # Second line: "There" at y=130 (30 point gap)
        make_char("T", (10.0, 130.0, 20.0, 142.0), 2),
        make_char("h", (20.0, 130.0, 28.0, 142.0), 3),
        make_char("e", (28.0, 130.0, 36.0, 142.0), 4),
        make_char("r", (36.0, 130.0, 44.0, 142.0), 5),
        make_char("e", (44.0, 130.0, 52.0, 142.0), 6),
    ]


@pytest.fixture
def chars_with_whitespace() -> list[CharInfo]:
    """Characters with spaces: 'A B'."""
    return [
        make_char("A", (10.0, 100.0, 20.0, 112.0), 0),
        make_char(" ", (20.0, 100.0, 25.0, 112.0), 1),
        make_char("B", (25.0, 100.0, 35.0, 112.0), 2),
    ]


@pytest.fixture
def rotated_char() -> list[CharInfo]:
    """Single rotated character."""
    return [CharInfo(char="R", bbox=(10.0, 100.0, 20.0, 112.0), rotation=45.0, index=0)]


class TestBasicLineMerger:
    """Tests for BasicLineMerger."""

    def test_empty_chars_returns_empty_list(self) -> None:
        """Empty input should return empty output."""
        merger = BasicLineMerger()
        result = merger.merge([], None, PAGE_HEIGHT)
        assert result == []

    def test_single_line_merged_into_one_block(self, single_line_chars: list[CharInfo]) -> None:
        """Characters on same line should merge into single TextBlock."""
        merger = BasicLineMerger()
        result = merger.merge(single_line_chars, None, PAGE_HEIGHT)

        assert len(result) == 1
        assert result[0].text == "Hello"

    def test_two_lines_create_two_blocks(self, two_line_chars: list[CharInfo]) -> None:
        """Characters on different lines should create separate TextBlocks."""
        merger = BasicLineMerger()
        result = merger.merge(two_line_chars, None, PAGE_HEIGHT)

        assert len(result) == 2
        assert result[0].text == "Hi"
        assert result[1].text == "There"

    def test_bbox_encompasses_all_chars(self, single_line_chars: list[CharInfo]) -> None:
        """TextBlock bbox should encompass all characters."""
        merger = BasicLineMerger()
        result = merger.merge(single_line_chars, None, PAGE_HEIGHT)

        bbox = result[0].bbox
        # x0 should be leftmost char x0 (10.0)
        assert bbox.x0 == 10.0
        # x1 should be rightmost char x1 (50.0)
        assert bbox.x1 == 50.0

    def test_y_coordinates_flipped_for_page(self, single_line_chars: list[CharInfo]) -> None:
        """Y coordinates should be flipped (PDF origin is bottom-left)."""
        merger = BasicLineMerger()
        result = merger.merge(single_line_chars, None, PAGE_HEIGHT)

        bbox = result[0].bbox
        # Original y0=100, y1=112, page_height=792
        # Flipped: y0 = 792 - 112 = 680, y1 = 792 - 100 = 692
        assert bbox.y0 == PAGE_HEIGHT - 112.0
        assert bbox.y1 == PAGE_HEIGHT - 100.0

    def test_custom_line_gap_threshold(self, two_line_chars: list[CharInfo]) -> None:
        """Custom threshold should affect line detection."""
        # With large threshold (50), both lines should merge
        merger = BasicLineMerger(line_gap_threshold=50.0)
        result = merger.merge(two_line_chars, None, PAGE_HEIGHT)

        assert len(result) == 1
        assert result[0].text == "HiThere"

    def test_small_threshold_splits_more(self, two_line_chars: list[CharInfo]) -> None:
        """Smaller threshold should detect more line breaks."""
        # Default threshold (5.0) should split the two lines
        merger = BasicLineMerger(line_gap_threshold=5.0)
        result = merger.merge(two_line_chars, None, PAGE_HEIGHT)

        assert len(result) == 2

    def test_whitespace_preserved_in_text(self, chars_with_whitespace: list[CharInfo]) -> None:
        """Whitespace should be preserved in merged text."""
        merger = BasicLineMerger()
        result = merger.merge(chars_with_whitespace, None, PAGE_HEIGHT)

        assert len(result) == 1
        assert result[0].text == "A B"

    def test_rotation_from_first_char(self, rotated_char: list[CharInfo]) -> None:
        """TextBlock should use rotation from first character."""
        merger = BasicLineMerger()
        result = merger.merge(rotated_char, None, PAGE_HEIGHT)

        assert result[0].rotation == 45.0

    def test_whitespace_only_chars_filtered(self) -> None:
        """Lines with only whitespace should be filtered out."""
        chars = [
            make_char(" ", (10.0, 100.0, 15.0, 112.0), 0),
            make_char(" ", (15.0, 100.0, 20.0, 112.0), 1),
        ]
        merger = BasicLineMerger()
        result = merger.merge(chars, None, PAGE_HEIGHT)

        assert len(result) == 0

    def test_no_font_info_when_textpage_none(self, single_line_chars: list[CharInfo]) -> None:
        """Font info should be None when textpage is not provided."""
        merger = BasicLineMerger()
        result = merger.merge(single_line_chars, None, PAGE_HEIGHT)

        assert result[0].font_info is None


class TestKeepCharacterMerger:
    """Tests for KeepCharacterMerger."""

    def test_empty_chars_returns_empty_list(self) -> None:
        """Empty input should return empty output."""
        merger = KeepCharacterMerger()
        result = merger.merge([], None, PAGE_HEIGHT)
        assert result == []

    def test_each_char_becomes_separate_block(self, single_line_chars: list[CharInfo]) -> None:
        """Each character should become its own TextBlock."""
        merger = KeepCharacterMerger()
        result = merger.merge(single_line_chars, None, PAGE_HEIGHT)

        assert len(result) == 5
        assert [b.text for b in result] == ["H", "e", "l", "l", "o"]

    def test_whitespace_chars_preserved(self, chars_with_whitespace: list[CharInfo]) -> None:
        """Whitespace characters should be preserved as separate blocks."""
        merger = KeepCharacterMerger()
        result = merger.merge(chars_with_whitespace, None, PAGE_HEIGHT)

        assert len(result) == 3
        assert result[1].text == " "
        assert result[1].text.isspace()

    def test_bbox_per_character(self, single_line_chars: list[CharInfo]) -> None:
        """Each character should have its own bbox."""
        merger = KeepCharacterMerger()
        result = merger.merge(single_line_chars, None, PAGE_HEIGHT)

        # First char 'H' has bbox (10, 100, 20, 112)
        assert result[0].bbox.x0 == 10.0
        assert result[0].bbox.x1 == 20.0

        # Second char 'e' has bbox (20, 100, 28, 112)
        assert result[1].bbox.x0 == 20.0
        assert result[1].bbox.x1 == 28.0

    def test_y_coordinates_flipped(self, single_line_chars: list[CharInfo]) -> None:
        """Y coordinates should be flipped for each character."""
        merger = KeepCharacterMerger()
        result = merger.merge(single_line_chars, None, PAGE_HEIGHT)

        bbox = result[0].bbox
        # Original y0=100, y1=112
        assert bbox.y0 == PAGE_HEIGHT - 112.0
        assert bbox.y1 == PAGE_HEIGHT - 100.0

    def test_rotation_preserved(self, rotated_char: list[CharInfo]) -> None:
        """Rotation should be preserved for each character."""
        merger = KeepCharacterMerger()
        result = merger.merge(rotated_char, None, PAGE_HEIGHT)

        assert result[0].rotation == 45.0

    def test_no_font_info_when_textpage_none(self, single_line_chars: list[CharInfo]) -> None:
        """Font info should be None when textpage is not provided."""
        merger = KeepCharacterMerger()
        result = merger.merge(single_line_chars, None, PAGE_HEIGHT)

        for block in result:
            assert block.font_info is None

    def test_handles_multi_line_input(self, two_line_chars: list[CharInfo]) -> None:
        """Should create blocks for all characters regardless of line."""
        merger = KeepCharacterMerger()
        result = merger.merge(two_line_chars, None, PAGE_HEIGHT)

        assert len(result) == 7
        assert "".join(b.text for b in result) == "HiThere"
