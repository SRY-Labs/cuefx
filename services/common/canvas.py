"""The effect canvas: which plate region it covers (`canvas_rect`) and the maps between plate and canvas.

Coordinates are continuous, pixel edges at integers (pixel i covers [i, i + 1)), the same as the tracker's.
Plate pixels are the conformed plate; canvas pixels are the frame LTX works on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from services.cue.route import OrbSample

Rect = tuple[int, int, int, int]  # x, y, w, h in plate pixels


@dataclass(frozen=True)
class CanvasConfig:
    max_magnify: float = 2.0     # the rect is never smaller than canvas_size / this (an arbitrary cap)
    aspect_tol: float = 0.005    # the rect's aspect may differ from the canvas aspect by this much
    search: int = 32             # widths tried above the exact one when rounding to whole pixels


def propose_canvas_rect(path: list[OrbSample], canvas_size: tuple[int, int], plate_size: tuple[int, int],
                        margin: float = 1.0, cfg: CanvasConfig = CanvasConfig()) -> Rect:
    """The plate region to use as the effect canvas, as whole-pixel [x, y, w, h].

    1. Bounding box of every orb disc in `path` (centre +- diameter / 2), plus `margin` x the largest
       diameter on every side.
    2. Grown to the canvas aspect ratio, staying centred.
    3. Never smaller than canvas_size / cfg.max_magnify.
    4. Moved inside the plate; shrunk only if it is larger than the plate (so the margin can be lost there).
    5. Rounded to whole pixels, the aspect within cfg.aspect_tol of the canvas aspect, still covering the box.
    """
    if not path:
        raise ValueError("cannot propose a canvas rect for an empty path")
    cw, ch = canvas_size
    pw, ph = plate_size
    aspect = cw / ch
    pts = np.array([(s[1], s[2], s[3]) for s in path], dtype=float)
    r = pts[:, 2] / 2
    pad = margin * float(pts[:, 2].max())
    x0, x1 = float((pts[:, 0] - r).min() - pad), float((pts[:, 0] + r).max() + pad)
    y0, y1 = float((pts[:, 1] - r).min() - pad), float((pts[:, 1] + r).max() + pad)
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    if w < cw / cfg.max_magnify:
        w = cw / cfg.max_magnify
        h = w / aspect
    if w > pw or h > ph:
        k = min(pw / w, ph / h)
        w, h = w * k, h * k
    left = min(max(cx - w / 2, 0.0), pw - w)
    top = min(max(cy - h / 2, 0.0), ph - h)
    return _round_rect(left, top, w, h, aspect, plate_size, cfg)


def _round_rect(left: float, top: float, w: float, h: float, aspect: float, plate_size: tuple[int, int],
                cfg: CanvasConfig) -> Rect:
    """Whole-pixel rect with the best aspect, at least as large as (w, h) where the plate allows."""
    pw, ph = plate_size
    first = min(math.ceil(w) + 1, pw)  # one pixel of slack, so centring on whole pixels still covers the box
    best: tuple[float, int, int] | None = None
    for iw in range(first, min(first + cfg.search, pw) + 1):
        ih = round(iw / aspect)
        err = abs(iw / ih / aspect - 1) if ih > 0 else math.inf
        if ih <= ph and ih >= min(h + 1, ph) and err <= cfg.aspect_tol and (best is None or err < best[0] - 1e-12):
            best = (err, iw, ih)
    if best is None:  # nothing at or above the box fits the plate: take the best that fits below it
        for iw in range(first, 0, -1):
            ih = round(iw / aspect)
            if 0 < ih <= ph and abs(iw / ih / aspect - 1) <= cfg.aspect_tol:
                best = (0.0, iw, ih)
                break
        else:
            raise ValueError(f"no whole-pixel rect with aspect {aspect:.4f} fits the plate {pw}x{ph}")
    _, iw, ih = best
    x = min(max(round(left + w / 2 - iw / 2), 0), pw - iw)
    y = min(max(round(top + h / 2 - ih / 2), 0), ph - ih)
    return x, y, iw, ih


def canvas_scale(canvas_rect: Rect, canvas_size: tuple[int, int]) -> float:
    """Canvas pixels per plate pixel (by width; the aspects agree to within 0.5 %)."""
    return canvas_size[0] / canvas_rect[2]


def plate_to_canvas(xy, d, canvas_rect: Rect, canvas_size: tuple[int, int]):
    """Plate position(s) `xy` ((2,) or (n, 2)) and diameter(s) `d` to canvas pixels: (xy, d)."""
    x, y, w, h = canvas_rect
    cw, ch = canvas_size
    xy = np.asarray(xy, dtype=float)
    return (xy - (x, y)) * (cw / w, ch / h), np.asarray(d, dtype=float) * (cw / w)


def canvas_to_plate(xy, d, canvas_rect: Rect, canvas_size: tuple[int, int]):
    """Exact inverse of `plate_to_canvas`: canvas position(s) and diameter(s) to plate pixels."""
    x, y, w, h = canvas_rect
    cw, ch = canvas_size
    xy = np.asarray(xy, dtype=float)
    return xy / (cw / w, ch / h) + (x, y), np.asarray(d, dtype=float) / (cw / w)
