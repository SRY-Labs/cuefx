"""sRGB <-> linear light, and luminance. Images are BGR (OpenCV order), float32 in linear light.

The 8-bit decode and encode are table lookups, and they round-trip exactly: `encode8(decode8(x)) == x` for every
8-bit value, so a pixel the maths leaves alone comes back byte-identical.
"""
from __future__ import annotations

import numpy as np

LUMA_BGR = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)  # Rec.709 weights, in B, G, R order
_ENC_STEPS = 65535  # linear values are quantised to this many steps before the encode lookup


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    """sRGB-encoded values in 0..1 to linear light."""
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    """Linear light in 0..1 to sRGB-encoded values."""
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)


_DECODE = srgb_to_linear(np.arange(256) / 255.0).astype(np.float32)
_ENCODE = np.clip(np.rint(linear_to_srgb(np.arange(_ENC_STEPS + 1) / _ENC_STEPS) * 255.0), 0, 255).astype(np.uint8)


def decode8(img: np.ndarray) -> np.ndarray:
    """8-bit sRGB image to float32 linear light."""
    return _DECODE[img]


def encode8(lin: np.ndarray) -> np.ndarray:
    """Float linear light to 8-bit sRGB, clipped to 0..1."""
    idx = np.rint(np.clip(lin, 0.0, 1.0) * _ENC_STEPS).astype(np.int32)
    return _ENCODE[idx]


def luminance(lin: np.ndarray) -> np.ndarray:
    """Luminance of a linear BGR image (h, w, 3) -> (h, w)."""
    return lin @ LUMA_BGR
