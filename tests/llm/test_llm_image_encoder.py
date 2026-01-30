"""Tests for LLM image encoder adapter."""

import base64

from PIL import Image

from xtra.llm.adapters.image_encoder import ImageEncoder


def test_encode_png() -> None:
    encoder = ImageEncoder(image_format="PNG")
    img = Image.new("RGB", (100, 100), color="red")
    result = encoder.encode_image(img)
    assert result.startswith("data:image/png;base64,")


def test_encode_jpeg() -> None:
    encoder = ImageEncoder(image_format="JPEG")
    img = Image.new("RGB", (100, 100), color="blue")
    result = encoder.encode_image(img)
    assert result.startswith("data:image/jpeg;base64,")


def test_encoded_is_valid_base64() -> None:
    encoder = ImageEncoder()
    img = Image.new("RGB", (50, 50), color="green")
    result = encoder.encode_image(img)
    base64_data = result.split(",", 1)[1]
    decoded = base64.b64decode(base64_data)
    assert len(decoded) > 0


def test_encode_multiple() -> None:
    encoder = ImageEncoder()
    images = [Image.new("RGB", (50, 50), color=c) for c in ["red", "green", "blue"]]
    results = encoder.encode_images(images)
    assert len(results) == 3


def test_no_resize_when_disabled() -> None:
    encoder = ImageEncoder(max_dimension=None)
    img = Image.new("RGB", (2000, 1000), color="white")
    resized = encoder._resize_if_needed(img)
    assert resized.size == (2000, 1000)


def test_resize_large_image() -> None:
    encoder = ImageEncoder(max_dimension=100)
    img = Image.new("RGB", (200, 100), color="white")
    resized = encoder._resize_if_needed(img)
    assert resized.size == (100, 50)  # Scaled by 0.5, aspect preserved


def test_no_resize_small_image() -> None:
    encoder = ImageEncoder(max_dimension=500)
    img = Image.new("RGB", (100, 100), color="white")
    resized = encoder._resize_if_needed(img)
    assert resized.size == (100, 100)


def test_preserve_aspect_ratio() -> None:
    encoder = ImageEncoder(max_dimension=100)
    img = Image.new("RGB", (400, 200), color="white")  # 2:1 ratio
    resized = encoder._resize_if_needed(img)
    assert resized.size == (100, 50)  # Still 2:1


def test_default_max_dimension() -> None:
    encoder = ImageEncoder()
    assert encoder.max_dimension == 2048
