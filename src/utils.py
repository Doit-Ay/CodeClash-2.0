"""
EVA Guardian — Utility Functions
Image processing helpers used across the application.
"""
from PIL import Image
from typing import Tuple

from src.config import PAD_COLOR, MAX_IMAGE_SIZE


def resize_image(
    image: Image.Image,
    max_size: Tuple[int, int] = MAX_IMAGE_SIZE,
) -> Image.Image:
    """Resizes a PIL image if it exceeds *max_size*, preserving aspect ratio."""
    if image.width > max_size[0] or image.height > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def pad_image_to_square(image: Image.Image) -> Image.Image:
    """Pads a PIL image with a dark background to make it square."""
    width, height = image.size
    if width == height:
        return image

    bigger = max(width, height)
    result = Image.new(image.mode, (bigger, bigger), PAD_COLOR)
    result.paste(image, ((bigger - width) // 2, (bigger - height) // 2))
    return result
