from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError


DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
SUPPORTED_CONTENT_TYPES = {
    "image/bmp": {".bmp"},
    "image/jpeg": {".jpeg", ".jpg"},
    "image/jpg": {".jpeg", ".jpg"},
    "image/png": {".png"},
    "image/webp": {".webp"},
}
SUPPORTED_IMAGE_MODES = {"L", "RGB", "RGBA"}


class InvalidInputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _format_values(values):
    return ", ".join(sorted(values))


def _extension_for(filename: str | None) -> str | None:
    if not filename:
        return None
    return Path(filename).suffix.lower()


def _validate_extension(filename: str | Path | None):
    extension = _extension_for(str(filename)) if filename is not None else None
    if extension is None:
        return None
    if extension not in SUPPORTED_EXTENSIONS:
        raise InvalidInputError(
            "unsupported_extension",
            f"unsupported image extension {extension!r}; supported extensions: {_format_values(SUPPORTED_EXTENSIONS)}",
        )
    return extension


def _validate_content_type(content_type: str | None, extension: str | None):
    if content_type is None:
        return

    normalized_type = content_type.split(";", 1)[0].strip().lower()
    expected_extensions = SUPPORTED_CONTENT_TYPES.get(normalized_type)
    if expected_extensions is None:
        raise InvalidInputError(
            "unsupported_content_type",
            f"unsupported content type {content_type!r}; supported content types: {_format_values(SUPPORTED_CONTENT_TYPES)}",
        )

    if extension is not None and extension not in expected_extensions:
        raise InvalidInputError(
            "extension_content_type_mismatch",
            f"extension {extension!r} does not match content type {content_type!r}",
        )


def _validate_size(size_bytes: int, max_bytes: int):
    if size_bytes == 0:
        raise InvalidInputError("empty_file", "image file is empty")
    if size_bytes > max_bytes:
        raise InvalidInputError(
            "file_too_large",
            f"image file is {size_bytes} bytes, which exceeds the {max_bytes} byte limit",
        )


def _decode_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            if image.mode not in SUPPORTED_IMAGE_MODES:
                raise InvalidInputError(
                    "unsupported_image_mode",
                    f"unsupported image mode {image.mode!r}; supported modes: {_format_values(SUPPORTED_IMAGE_MODES)}",
                )
            return image.convert("RGB").copy()
    except InvalidInputError:
        raise
    except (OSError, UnidentifiedImageError) as exc:
        raise InvalidInputError(
            "undecodable_image",
            f"image could not be decoded: {exc}",
        ) from exc


def validate_image_bytes(
    image_bytes: bytes,
    *,
    filename: str | None = None,
    content_type: str | None = None,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> Image.Image:
    extension = _validate_extension(filename)
    _validate_content_type(content_type, extension)
    _validate_size(len(image_bytes), max_bytes)
    return _decode_image(image_bytes)


def validate_image_path(
    image_path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise InvalidInputError("file_not_found", f"image file does not exist: {path}")
    if not path.is_file():
        raise InvalidInputError("not_a_file", f"image path is not a file: {path}")

    _validate_extension(path)
    _validate_size(path.stat().st_size, max_bytes)
    return _decode_image(path.read_bytes())
