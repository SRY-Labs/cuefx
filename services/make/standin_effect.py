"""Stand-in Make: a code-drawn effect that writes the same two files LTX will write.

    effect.mp4  the look image on black, moved and scaled onto the ball in every frame, plus a constant black floor
    matte.mp4   grey, white = the look's solid core, moved and scaled the same way

With an entrance the orb grows from nothing over it (diameter, light and matte x `services.cue.route.entrance_gain`).
Both are at canvas size, one frame per frame of the effect window (`services.cue.window.effect_window`), H.264 8-bit like LTX's output, so
Merge cannot tell them from the real thing. Swapping in LTX later replaces only these two files.
Light is moved and scaled in linear light (one orb per frame, like the layout's one ball: no motion-blur copies); the effect is stored sRGB-encoded, the matte as is.

A cue with a `shape` other than the sphere (task 31) is not a look image placed on a ball: `render_standin_shape` draws the
shape with the layout's own renderer, tinted from the look image's mean colour, with the sphere look's glow.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from services.common.canvas import Rect
from services.common.colour import decode8, encode8, luminance
from services.make.effects import EndingPlan, burst_factors, render_shards, shatter_alpha
from services.make.layout import (LayoutConfig, LayoutError, apply_gains, canvas_path, count_frames, render_shape, to_bgr8,
                          write_video)
from services.make.look import LookError, Orb, find_orb, orb_onto
from services.cue.route import OrbSample

MIN_DRAW_D = 0.25  # canvas px: a smaller orb (only ever in the first frames of an entrance) is not drawn
# the shape stand-in's look, the numbers of `services.make.look.standin_look` (size 512): glow sigma 0.11 and core 0.28 of the size
GLOW_SIGMA = 0.11 / 0.28   # glow sigma as a share of the shape's diameter
GLOW_AMOUNT = 0.55         # glow light at the centre, x the tint
CORE_WHITE = (1.0, 0.98, 0.95)  # BGR: the core is this much white ...
CORE_WHITE_SHARE = 0.65         # ... mixed with the tint
SHARD_GLOW_SIGMA = 0.12    # the shatter's glow: Gaussian of the shard coverage, sigma as a share of the effect's diameter
SHARD_GLOW_AMOUNT = 0.9    # x the tint, where the shards cover everything within the sigma
TINT_MIN_LUM = 0.1        # the look's tint is the mean colour of the pixels at least this x its peak


@dataclass(frozen=True)
class StandinConfig:
    black_floor: float = 0.02     # linear, added everywhere: LTX's "black" is not quite black
    core_threshold: float = 0.5   # the matte's core: look luminance >= this x its peak
    core_feather_px: float = 1.5  # Gaussian sigma of the core's edge, in look-image pixels
    crf: int = 12


@dataclass(frozen=True)
class StandinLook:
    """The look image ready to place: its light (linear BGR), its core (0..1) and where its orb is."""
    light: np.ndarray
    core: np.ndarray
    orb: Orb


def _to_bgr8(image: np.ndarray) -> np.ndarray:
    """Grey, BGR or BGRA, 8- or 16-bit, to 8-bit BGR."""
    img = image[..., :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = np.rint(img.astype(np.float64) * (255.0 / np.iinfo(img.dtype).max)).astype(np.uint8)
    return np.ascontiguousarray(img)


def prepare_look(look_image: np.ndarray, cfg: StandinConfig = StandinConfig()) -> tuple[StandinLook, list[str]]:
    """Linear light, feathered core and orb of a look image, and warnings.

    The core is the pixels whose (linear) luminance is at least `core_threshold` x the brightest, feathered by a
    Gaussian of `core_feather_px`. If the orb cannot be found the whole image is treated as the orb (centred,
    as wide as its shorter side) and a warning says so.
    """
    src = _to_bgr8(look_image)
    h, w = src.shape[:2]
    warnings: list[str] = []
    try:
        orb = find_orb(src)
    except LookError as e:
        orb = Orb(w / 2, h / 2, float(min(w, h)))
        warnings.append(f"look: {e}; the whole image is used as the orb, so the effect may not match the ball's size")
    light = decode8(src)
    lum = luminance(light)
    peak = float(lum.max())
    core = (lum >= cfg.core_threshold * peak).astype(np.float32) if peak > 0 else np.zeros((h, w), np.float32)
    if cfg.core_feather_px > 0:
        core = cv2.GaussianBlur(core, (0, 0), cfg.core_feather_px)
    return StandinLook(light, core, orb), warnings


def _placed(src: np.ndarray, orb: Orb, x: float, y: float, d: float, canvas_size: tuple[int, int]) -> np.ndarray:
    """`orb_onto`, or black when the orb is under MIN_DRAW_D canvas px wide (its light is far below one 8-bit step,
    and shrinking the look that far would need an enormous anti-alias blur)."""
    if d < MIN_DRAW_D:
        return np.zeros((canvas_size[1], canvas_size[0]) + src.shape[2:], dtype=src.dtype)
    return orb_onto(src, orb, x, y, d, canvas_size)


def render_standin(look: StandinLook, path: list[OrbSample], canvas_rect: Rect, canvas_size: tuple[int, int],
                   cfg: StandinConfig = StandinConfig(), gains: list[float] | None = None,
                   ending: EndingPlan | None = None,
                   lights: list[float] | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(effect frames, matte frames), 8-bit BGR at canvas size, one per sample of `path`.

    Each frame places the look once on the ball, at the same canvas centre and diameter as the layout draws it
    (`canvas_path`, `apply_gains`): one orb per frame, no motion-blur copies. Visibility is not applied: that is
    Merge's job. With `gains` (the entrance, one per sample) the diameter, the light and the matte are all scaled by
    the frame's gain, like the layout's ball, so the orb grows from nothing and its core never blocks more plate than
    it lights.
    With an `ending` (a burst or shatter, task 36) the frames from the vanish on are the ending's: a grown, flashing,
    fading look (burst), shards tinted from the look (shatter), or black once it is over. `lights` (one per sample,
    `services.cue.route.orb_lights`) scales the light only, not the matte: the charge-up pulse.
    """
    cpath = apply_gains(canvas_path(path, canvas_rect, canvas_size), gains)
    effect, matte = [], []
    tint = look_tint(look.light) if ending is not None else None
    for i, (x, y, d) in enumerate(cpath):
        st = None if ending is None else ending.states[i]
        if st is not None and not (st.kind == "shatter" and st.k == 0):
            g = np.float32(1.0 if gains is None else gains[i])
            if st.kind == "gone":
                e = np.zeros((canvas_size[1], canvas_size[0], 3), np.float32)
                m = np.zeros((canvas_size[1], canvas_size[0]), np.float32)
            elif st.kind == "burst":
                scale, light, alpha = burst_factors(st.t)
                e = _placed(look.light, look.orb, x, y, d * scale, canvas_size) * g * np.float32(light)
                m = _placed(look.core, look.orb, x, y, d * scale, canvas_size) * g * np.float32(alpha)
            else:
                e, m = _shard_frame(tint, ending, st, x, y, float(cpath[st.ref][2]), canvas_size)
                e, m = e * g, m * g
            effect.append(encode8((e if lights is None else e * np.float32(lights[i])) + cfg.black_floor))
            matte.append(to_bgr8(m))
            continue
        if gains is None:
            e = orb_onto(look.light, look.orb, x, y, d, canvas_size)
            m = orb_onto(look.core, look.orb, x, y, d, canvas_size)
        else:
            g = np.float32(gains[i])
            e = _placed(look.light, look.orb, x, y, d, canvas_size) * g
            m = _placed(look.core, look.orb, x, y, d, canvas_size) * g
        effect.append(encode8((e if lights is None else e * np.float32(lights[i])) + cfg.black_floor))
        matte.append(to_bgr8(m))
    return effect, matte


def look_tint(light: np.ndarray) -> np.ndarray:
    """Mean colour (linear BGR, brightest channel 1) of the lit pixels of a look image (`StandinLook.light`): those
    with a luminance of at least TINT_MIN_LUM x the peak. White if the image has none."""
    lum = luminance(light)
    peak = float(lum.max())
    if peak <= 0:
        return np.ones(3, dtype=np.float32)
    mean = light[lum >= TINT_MIN_LUM * peak].mean(axis=0)
    return (mean / max(float(mean.max()), 1e-9)).astype(np.float32)


def _glow(shape: tuple[int, int], x: float, y: float, d: float) -> np.ndarray:
    """GLOW_AMOUNT x exp(-r^2 / 2 sigma^2) around (x, y), sigma = GLOW_SIGMA x d, computed over 4 sigma only."""
    out = np.zeros(shape, dtype=np.float32)
    sigma = GLOW_SIGMA * d
    pad = int(np.ceil(4 * sigma)) + 1
    x0, x1 = max(0, int(x) - pad), min(shape[1], int(x) + pad + 1)
    y0, y1 = max(0, int(y) - pad), min(shape[0], int(y) + pad + 1)
    if x0 >= x1 or y0 >= y1:
        return out
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    out[y0:y1, x0:x1] = GLOW_AMOUNT * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / np.float32(sigma * sigma))
    return out


def _shard_frame(tint: np.ndarray, plan: EndingPlan, st, x: float, y: float, d_ref: float,
                 canvas_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(light (h, w, 3) linear BGR, matte (h, w)) of the shards of a shatter at the ending's frame `st`: each shard lit
    in the core colour (the shape stand-in's `CORE_WHITE_SHARE` white + the `tint`, by its grey over the albedo), with a
    soft glow of the tint round the cloud, all scaled by the shatter's fade. The light is not clipped."""
    lcfg = LayoutConfig()
    cw, ch = canvas_size
    core = ((1 - CORE_WHITE_SHARE) * tint + CORE_WHITE_SHARE * np.array(CORE_WHITE, dtype=np.float32)).astype(np.float32)
    grey, cover = render_shards((ch, cw), x, y, d_ref, st.k, plan.fps, plan.seed, lcfg.albedo, lcfg.ambient, lcfg.light, plan.picture)
    glow = cv2.GaussianBlur(cover, (0, 0), max(SHARD_GLOW_SIGMA * d_ref, 1.0)) * np.float32(SHARD_GLOW_AMOUNT)
    e = glow[..., None] * tint * (1 - cover[..., None]) + core * (grey / lcfg.albedo)[..., None]
    alpha = np.float32(shatter_alpha(st.t))
    return (e * alpha).astype(np.float32), cover * alpha


def render_standin_shape(kind: str, tint: np.ndarray, path: list[OrbSample], canvas_rect: Rect,
                         canvas_size: tuple[int, int], spin: float, cfg: StandinConfig = StandinConfig(),
                         gains: list[float] | None = None, ending: EndingPlan | None = None,
                         lights: list[float] | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(effect frames, matte frames) like `render_standin`, with the ring, crystal or cube of the layout in place of
    the look image: `render_shape` (same centre, size and spin per frame as the layout), lit in the core colour
    (`CORE_WHITE_SHARE` white + the `tint`, scaled by the shape's grey over the albedo) over a Gaussian glow of the
    tint, as in the sphere's look. The matte is the shape's coverage. With `gains` the diameter, the light and the
    matte are scaled by the frame's gain, as in `render_standin`. `spin` is degrees per frame. An `ending` (task 36) is
    drawn as in `render_standin`: the shape grown and flashing (burst), or the shards (shatter)."""
    lcfg = LayoutConfig()
    core = ((1 - CORE_WHITE_SHARE) * tint + CORE_WHITE_SHARE * np.array(CORE_WHITE, dtype=np.float32)).astype(np.float32)
    cpath = apply_gains(canvas_path(path, canvas_rect, canvas_size), gains)
    cw, ch = canvas_size
    effect, matte = [], []
    for i, (x, y, d) in enumerate(cpath):
        g = np.float32(1.0 if gains is None else gains[i])
        st = None if ending is None else ending.states[i]
        light, alpha = np.float32(1.0 if lights is None else lights[i]), np.float32(1.0)
        if st is not None and st.kind == "gone":
            d = 0.0
        elif st is not None and st.kind == "burst":
            scale, burst_light, alpha = (np.float32(v) for v in burst_factors(st.t))
            light, d = light * burst_light, d * float(scale)
        if st is not None and st.kind == "shatter" and st.k > 0:
            e, m = _shard_frame(tint, ending, st, x, y, float(cpath[st.ref][2]), canvas_size)
        elif d < MIN_DRAW_D:
            e, m = np.zeros((ch, cw, 3), np.float32), np.zeros((ch, cw), np.float32)
        else:
            grey, m = render_shape(kind, (ch, cw), x, y, d, spin * i, lcfg)
            e = _glow((ch, cw), x, y, d)[..., None] * tint * (1 - m[..., None]) + core * (grey / lcfg.albedo)[..., None]
        effect.append(encode8(e * (g * light) + cfg.black_floor))
        matte.append(to_bgr8(m * (g * alpha)))
    return effect, matte


def write_standin_files(effect: list[np.ndarray], matte: list[np.ndarray], frames: int, fps: float, out_dir: Path,
                        crf: int = StandinConfig.crf) -> None:
    """Write effect.mp4 and matte.mp4 into out_dir. Raises LayoutError if a decoded file does not have `frames` frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, images in (("effect.mp4", effect), ("matte.mp4", matte)):
        write_video(images, out_dir / name, fps, crf)
        n = count_frames(out_dir / name)
        if n != frames:
            raise LayoutError(f"{name} has {n} frames, expected {frames}")


def write_standin(look: StandinLook, path: list[OrbSample], canvas_rect: Rect, canvas_size: tuple[int, int],
                  fps: float, out_dir: Path, cfg: StandinConfig = StandinConfig(),
                  gains: list[float] | None = None, ending: EndingPlan | None = None,
                  lights: list[float] | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Write effect.mp4 and matte.mp4 into out_dir; returns their frames (`render_standin`, with the entrance `gains`).

    Raises LayoutError if a decoded file does not have exactly one frame per sample of `path`.
    """
    effect, matte = render_standin(look, path, canvas_rect, canvas_size, cfg, gains, ending, lights)
    write_standin_files(effect, matte, len(path), fps, out_dir, cfg.crf)
    return effect, matte
