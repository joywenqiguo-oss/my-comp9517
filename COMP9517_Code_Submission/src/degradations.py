"""Image degradation functions for robustness testing.

These functions are intended for test-time robustness evaluation only.
Do not use degraded test images for training or hyperparameter tuning.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _to_float_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _from_float_array(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def gaussian_noise(image: Image.Image, sigma: float = 0.10, seed: int | None = None) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = _to_float_array(image)
    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape)
    return _from_float_array(arr + noise)


def gaussian_blur(image: Image.Image, kernel_size: int = 5) -> Image.Image:
    # PIL uses radius rather than kernel size. A simple approximation is kernel_size / 2.
    radius = max((int(kernel_size) - 1) / 2.0, 0.0)
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))


def brightness_change(image: Image.Image, factor: float = 0.6) -> Image.Image:
    return ImageEnhance.Brightness(image.convert("RGB")).enhance(float(factor))


def contrast_change(image: Image.Image, factor: float = 0.6) -> Image.Image:
    return ImageEnhance.Contrast(image.convert("RGB")).enhance(float(factor))


def jpeg_compression(image: Image.Image, quality: int = 40) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


DEGRADATION_LEVELS = {
    "noise": [0.05, 0.10, 0.20],
    "blur": [3, 5, 7],
    "brightness": [0.8, 0.6, 0.4],
    "contrast": [0.8, 0.6, 0.4],
    "jpeg": [70, 40, 20],
}


def apply_degradation(image: Image.Image, degradation: str, severity) -> Image.Image:
    degradation = degradation.lower()
    if degradation == "noise":
        return gaussian_noise(image, sigma=float(severity))
    if degradation == "blur":
        return gaussian_blur(image, kernel_size=int(severity))
    if degradation == "brightness":
        return brightness_change(image, factor=float(severity))
    if degradation == "contrast":
        return contrast_change(image, factor=float(severity))
    if degradation == "jpeg":
        return jpeg_compression(image, quality=int(severity))
    raise ValueError(f"Unknown degradation: {degradation}")
