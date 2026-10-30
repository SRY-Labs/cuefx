"""Ingest LTX output: check it against the cue, write the Make files Merge reads, and a temporary matte.

The LTX effect video (an mp4, or a folder of PNGs sorted by name) must be exactly the cue's canvas_size: a
different size is a hard error, because we never stretch. A different frame count degrades: the overlap is
used, missing frames are black (no effect, no matte) and a warning gives the counts.

TEMPORARY MATTE (until Alpha Gen is built): a smooth ramp on the effect's own luminance (linear light) between
two thresholds that are fractions of the clip's bright level (a high percentile of luminance over the whole
clip). Nothing in it is specific to one clip, so a clip at half the brightness gets the same matte shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from services.common.colour import decode8, luminance
from services.make.layout import count_frames, to_bgr8, write_video
from services.delivery.video import VideoError, read_frames

MATTE_WARNING = "temporary luminance matte — replace with Alpha Gen"


class IngestError(VideoError):
    """The LTX frames cannot be used: unreadable, or not the canvas size."""


@dataclass(frozen=True)
class MatteConfig:
    bright_percentile: float = 99.9  # the clip's bright level: this percentile of luminance over all its frames
    low: float = 0.35                # the matte starts to rise at this x the bright level
    high: float = 0.80               # and is 1 from this x the bright level
    stride: int = 4                  # the percentile is measured on every stride-th pixel in x and y
    crf: int = 12


@dataclass
class IngestResult:
    effect: list[np.ndarray]         # 8-bit BGR, one per frame of the (trimmed) effect_range
    matte: list[np.ndarray]          # 8-bit grey as BGR
    bright: float                    # the clip's bright level (linear luminance)
    warnings: list[str]


def read_ltx(src: Path) -> list[np.ndarray]:
    """The frames of an mp4 (ffmpeg) or of a folder of PNGs (sorted by name), 8-bit BGR."""
    src = Path(src)
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() == ".png")
        if not files:
            raise IngestError(f"{src} has no PNG files")
        frames = []
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise IngestError(f"cannot read {p}")
            frames.append(img)
        return frames
    if not src.exists():
        raise IngestError(f"{src} not found")
    return [np.array(f) for f in read_frames(src)]


def fit_to_range(frames: list[np.ndarray], count: int, canvas_size: tuple[int, int],
                 name: str = "ltx") -> tuple[list[np.ndarray], int, list[str]]:
    """`count` frames at canvas size, the number of real frames among them, and warnings.

    A frame of another size raises IngestError. Too many frames: the extra ones at the end are dropped. Too few:
    the missing ones at the end are black. Either way a warning says how many there were and how many are used.
    """
    if not frames:
        raise IngestError(f"{name}: no frames")
    w, h = canvas_size
    for i, fr in enumerate(frames):
        if (fr.shape[1], fr.shape[0]) != (w, h):
            raise IngestError(f"{name}: frame {i} is {fr.shape[1]}x{fr.shape[0]}, the cue's canvas is {w}x{h}; "
                              f"never stretched: make again at the canvas size")
    warnings: list[str] = []
    real = min(len(frames), count)
    if len(frames) != count:
        what = "extra frames at the end are dropped" if len(frames) > count else \
            f"the last {count - len(frames)} are empty (no effect)"
        warnings.append(f"{name}: {len(frames)} frames, effect_range has {count}; using {real}, {what}")
    out = list(frames[:real]) + [np.zeros((h, w, 3), np.uint8) for _ in range(count - real)]
    return out, real, warnings


def bright_level(effect: list[np.ndarray], cfg: MatteConfig = MatteConfig()) -> float:
    """The clip's bright level: `bright_percentile` of linear luminance over every frame (every stride-th pixel)."""
    s = cfg.stride
    sample = np.concatenate([luminance(decode8(fr[::s, ::s])).ravel() for fr in effect])
    return float(np.percentile(sample, cfg.bright_percentile))


def luminance_matte(lum: np.ndarray, bright: float, cfg: MatteConfig = MatteConfig()) -> np.ndarray:
    """Matte 0..1 (float32) from linear luminance: a smoothstep from low x bright to high x bright.
    0 on black, 1 on the brightest parts; an all-black clip (bright 0) gives an all-0 matte."""
    if bright <= 0:
        return np.zeros(lum.shape, np.float32)
    t = np.clip((lum - cfg.low * bright) / ((cfg.high - cfg.low) * bright), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def temporary_matte(effect: list[np.ndarray], real: int,
                    cfg: MatteConfig = MatteConfig()) -> tuple[list[np.ndarray], float]:
    """Matte frames (8-bit grey as BGR) for all frames, and the bright level measured on the first `real` ones.
    Frames after `real` (the empty ones) get an all-0 matte."""
    bright = bright_level(effect[:real], cfg)
    matte = [to_bgr8(luminance_matte(luminance(decode8(fr)), bright, cfg)) for fr in effect[:real]]
    matte += [np.zeros_like(effect[0]) for _ in effect[real:]]
    return matte, bright


def ingest(frames: list[np.ndarray], count: int, canvas_size: tuple[int, int],
           cfg: MatteConfig = MatteConfig()) -> IngestResult:
    """Check `frames` against the cue (see `fit_to_range`) and make the temporary matte."""
    effect, real, warnings = fit_to_range(frames, count, canvas_size)
    matte, bright = temporary_matte(effect, real, cfg)
    return IngestResult(effect, matte, bright, warnings + [MATTE_WARNING])


def ingest_with_matte(frames: list[np.ndarray], matte_frames: list[np.ndarray], count: int,
                      canvas_size: tuple[int, int], cfg: MatteConfig = MatteConfig()) -> IngestResult:
    """`ingest` with a matte made elsewhere (Alpha Gen) instead of the temporary one: both are checked and fitted
    to the range the same way (`fit_to_range`); the matte is used as it is (grey, 0..255 = coverage)."""
    effect, real, warnings = fit_to_range(frames, count, canvas_size)
    matte, _, matte_warnings = fit_to_range(matte_frames, count, canvas_size, "matte")
    return IngestResult(effect, matte, bright_level(effect[:real], cfg), warnings + matte_warnings)


def write_ingest(result: IngestResult, fps: float, out_dir: Path, cfg: MatteConfig = MatteConfig()) -> None:
    """Write effect.mp4 and matte.mp4 (H.264 CRF 12, like the stand-in) into out_dir.
    Raises VideoError if a decoded file does not have one frame per frame of the result."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frames in (("effect.mp4", result.effect), ("matte.mp4", result.matte)):
        write_video(frames, out_dir / name, fps, cfg.crf)
        n = count_frames(out_dir / name)
        if n != len(frames):
            raise VideoError(f"{name} has {n} frames, expected {len(frames)}")
