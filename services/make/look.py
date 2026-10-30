"""Place Look: put the look image (the effect on pure black) where the first layout ball is.

The orb in the look image is found by thresholding luminance (Otsu) and taking the largest blob.
`find_orb` raises LookError when there is none; `place_look` degrades to a warning instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from services.common.canvas import Rect, plate_to_canvas
from services.cue.route import OrbSample

MIN_PEAK = 16  # a look image whose brightest pixel is below this (of 255) is treated as empty


class LookError(ValueError):
    """The look image has no usable orb (empty, or cut off by the image border)."""


@dataclass(frozen=True)
class Orb:
    """The orb found in an image: centre in pixel-edge coordinates, area-equivalent diameter."""
    x: float
    y: float
    diameter: float


def luminance(image: np.ndarray) -> np.ndarray:
    """Float32 luminance of a grey, BGR or BGRA image, in 0..255 (8-bit input) ."""
    if image.ndim == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(np.ascontiguousarray(image[..., :3]), cv2.COLOR_BGR2GRAY).astype(np.float32)


def find_orb(image: np.ndarray) -> Orb:
    """Locate the orb: Otsu threshold on luminance, largest blob, then
    diameter = that of the circle with the blob's area, centre = the blob's luminance-weighted centroid.

    Raises LookError if there is no blob or the blob touches the image border.
    """
    lum = luminance(image)
    if lum.size == 0 or float(lum.max()) < MIN_PEAK:
        raise LookError("the look image is empty: no pixel is brighter than black")
    grey8 = np.clip(lum, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(grey8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (grey8 > thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        raise LookError("no orb found in the look image: nothing above the Otsu threshold")
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[k]
    ih, iw = mask.shape
    if x == 0 or y == 0 or x + w == iw or y + h == ih:
        raise LookError(f"the orb touches the border of the look image (blob {w}x{h} at {x},{y} in {iw}x{ih}); "
                        f"use an image with the orb fully inside, on black")
    ys, xs = np.nonzero(labels == k)
    wts = lum[ys, xs]
    cx = float((xs + 0.5) @ wts / wts.sum())
    cy = float((ys + 0.5) @ wts / wts.sum())
    return Orb(cx, cy, float(2.0 * np.sqrt(area / np.pi)))


def place_look(look_image: np.ndarray, path: list[OrbSample], canvas_rect: Rect,
               canvas_size: tuple[int, int], at: int = 0) -> tuple[np.ndarray, list[str]]:
    """The look image scaled so its orb has the diameter of the ball in layout frame `at` (the first one by default;
    with an entrance, the first frame with the whole orb, `services.cue.route.first_full`), and moved so the two centres
    match, on a black canvas of `canvas_size`. Returns (8-bit BGR image, warnings).

    If no usable orb is found (the image is empty, or the orb touches the image border so its size and centre
    are unknown) the whole image is centred on the canvas instead, shrunk to fit if it is larger, and a warning
    says so: the effect then starts wherever the image has it, not on the ball.
    """
    src = look_image[..., :3] if look_image.ndim == 3 else cv2.cvtColor(look_image, cv2.COLOR_GRAY2BGR)
    src = np.ascontiguousarray(src)
    try:
        orb = find_orb(look_image)
    except LookError as e:
        return _centred(src, canvas_size), [f"look: {e}; the whole image is centred on the canvas, not placed on the ball. "
                                             f"Use a look image with the orb fully inside, on pure black."]
    xy, d = plate_to_canvas([path[at][1], path[at][2]], path[at][3], canvas_rect, canvas_size)
    return orb_onto(src, orb, float(xy[0]), float(xy[1]), float(d), canvas_size), []


def orb_onto(src: np.ndarray, orb: Orb, x: float, y: float, d: float, canvas_size: tuple[int, int]) -> np.ndarray:
    """`src` scaled so `orb` has diameter d and moved so its centre is at (x, y), on a black canvas.

    Works on any dtype OpenCV warps (8-bit, or float32 linear light). Pixel-edge coordinates.
    """
    s = d / orb.diameter
    if s < 1:  # warpAffine does not filter: low-pass first so shrinking does not alias
        src = cv2.GaussianBlur(src, (0, 0), 0.5 / s)
    # x_dst = s * (x_src - orb.x) + x, in pixel-edge coordinates; warpAffine counts pixel centres at integers
    m = np.array([[s, 0, x - s * orb.x + 0.5 * (s - 1)],
                  [0, s, y - s * orb.y + 0.5 * (s - 1)]], dtype=np.float64)
    return cv2.warpAffine(src, m, canvas_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _centred(img: np.ndarray, canvas_size: tuple[int, int]) -> np.ndarray:
    """The whole image centred on a black canvas; shrunk (never enlarged) to fit."""
    cw, ch = canvas_size
    h, w = img.shape[:2]
    k = min(1.0, cw / w, ch / h)
    if k < 1:
        img = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    out = np.zeros((ch, cw, 3), dtype=np.uint8)
    x0, y0 = (cw - w) // 2, (ch - h) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def standin_look(size: int = 512, tint: str = "#9fd4ff", core_diameter: float = 0.28, glow_sigma: float = 0.11) -> np.ndarray:
    """A stand-in look image: bright blue-white core with a soft glow of `tint` on black (8-bit BGR, size x size).

    `core_diameter` and `glow_sigma` are fractions of `size`.
    """
    rgb = np.array([int(tint[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255.0
    bgr = rgb[::-1]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    r = np.hypot(xx - size / 2, yy - size / 2)
    rc = core_diameter * size / 2
    core = np.clip((rc + 0.5 - r) / 2.0 + 0.5, 0.0, 1.0)  # soft edge, about 2 px wide
    glow = 0.55 * np.exp(-0.5 * (r / (glow_sigma * size)) ** 2)
    white = np.array([1.0, 0.98, 0.95], dtype=np.float32)  # BGR: blue-white
    core_colour = 0.35 * bgr + 0.65 * white
    img = glow[..., None] * bgr + core[..., None] * (core_colour - glow[..., None] * bgr)
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
