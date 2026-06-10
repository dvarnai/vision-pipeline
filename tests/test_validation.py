from io import BytesIO

import pytest
from PIL import Image

from src.inference.validation import InvalidInputError, validate_image_bytes


def _image_bytes(mode="RGB", image_format="PNG"):
    image = Image.new(mode, (4, 4), color=128)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def test_validate_image_bytes_accepts_supported_image_and_converts_to_rgb():
    image = validate_image_bytes(
        _image_bytes(mode="L"),
        filename="scene.png",
        content_type="image/png",
    )

    assert image.mode == "RGB"
    assert image.size == (4, 4)


def test_validate_image_bytes_rejects_extension_content_type_mismatch():
    with pytest.raises(InvalidInputError) as exc_info:
        validate_image_bytes(
            _image_bytes(),
            filename="scene.jpg",
            content_type="image/png",
        )

    assert exc_info.value.code == "extension_content_type_mismatch"


def test_validate_image_bytes_rejects_empty_file():
    with pytest.raises(InvalidInputError) as exc_info:
        validate_image_bytes(b"", filename="scene.png", content_type="image/png")

    assert exc_info.value.code == "empty_file"
