"""How an effect ends and how it charges (task 36): pure numbers and a shard renderer, shared by the layout and the stand-in.

An ending (cue v2.6 `ending`) starts at `vanish.start` and lasts until `vanish_of(cue).end`; for a burst or shatter the
effect fades itself, so Merge applies no fade-out of its own (`services.cue.route.visibility`). Frame k of the ending (k = 0 at
`vanish.start`, the effect still whole) has progress t = k / length in 0..1:

    burst    the shape grows to 1.8 x over the first 40 %, at 1.6 x the light (near white), held to 60 %, then fades   `burst_factors`
    shatter  12..20 grey shards fly outward, spin, fall a little; fully bright for the first half,
             then they fade                                                                          `shards`, `render_shards`

The ending is the whole vanish (task 38): a longer vanish stretches the burst and the fade, and lets the shards drift
further (their speed is fixed, so the way they travel is proportional to the vanish), kept inside the picture.
After the ending nothing is drawn. Everything here is a function of the cue's seed and the frame, so a rerun is identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from services.common.canvas import plate_to_canvas
from services.cue.cue import Cue, ending_of, vanish_of
from services.cue.route import OrbSample, smoothstep

BURST_SCALE = 1.8        # the burst grows to this x the size it had at vanish.start ...
BURST_GROW_T = 0.4       # ... over this share of the ending (ease-out)
BURST_FLASH = 1.6        # light x at the peak of the flash: the shape's grey of 0.6 is then 0.96, near white
BURST_FLASH_PEAK_T = BURST_GROW_T  # the light reaches it with the size and holds it until BURST_FADE_FROM
BURST_FADE_FROM = 0.6    # the burst fades out from here to the end of the ending
SHATTER_FADE_FROM = 0.5  # the shards are fully bright until here, then fade out
SHARD_COUNT = (12, 20)   # inclusive
SHARD_SIZE = (0.25, 0.42)  # a shard's bounding diameter, x the effect's diameter at vanish.start; never below SHARD_MIN_SIZE
SHARD_MIN_SIZE = 0.25    # so LTX can still read every shard (task 38: was 0.15)
SHARD_SPEED = (1.4, 3.2)   # outward speed, effect diameters per second
SHARD_SPIN = (180.0, 540.0)  # degrees per second in the picture plane (either way)
SHARD_TUMBLE = (240.0, 600.0)  # degrees per second about an in-plane axis: it changes the shard's grey
SHARD_GRAVITY = 2.0      # diameters per second squared, downwards ("slight")
SHARD_START_SPREAD = 0.3   # shards start this far (x diameter) from the centre, along their direction: about the old footprint
SHARD_AA_SHIFT = 4       # cv2 sub-pixel bits for the anti-aliased polygon edge
SHARD_SOFT_FROM = 0.7    # a shard that would leave the picture slows down from this share of the way to its edge


@dataclass(frozen=True)
class EndingState:
    """Where one path sample is in the ending. kind: "burst", "shatter", or "gone" (after the ending: draw nothing).
    k: frames since vanish.start; t = k / length; ref: index (into the path) of the vanish.start sample, whose size and
    position the ending is measured from."""
    kind: str
    k: int
    length: int
    ref: int

    @property
    def t(self) -> float:
        return self.k / self.length


@dataclass(frozen=True)
class EndingPlan:
    """The ending of a cue over a path: one `EndingState` per sample (None before the ending starts), and what the
    shards need (`fps`, `seed`)."""
    states: list[EndingState | None]
    fps: float
    seed: int | None
    picture: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1) canvas px the shards stay inside; None = anywhere


def ending_plan(cue: Cue, path: list[OrbSample]) -> EndingPlan | None:
    """The `EndingPlan` of a burst or shatter over `path`, or None for a fade ending: the layout and the stand-in then
    draw exactly as they always did."""
    kind = ending_of(cue)
    if kind not in ("burst", "shatter"):
        return None
    v = vanish_of(cue)
    length = v.end - v.start
    ref = min(max(v.start - path[0][0], 0), len(path) - 1)
    states = [None if s[0] < v.start else EndingState("gone" if s[0] >= v.end else kind, max(s[0] - v.start, 0), length, ref)
              for s in path]
    return EndingPlan(states, cue.fps, cue.seed, picture_on_canvas(cue))


def picture_on_canvas(cue: Cue) -> tuple[float, float, float, float] | None:
    """The part of the canvas that shows picture, as (x0, y0, x1, y1) in canvas pixels: the conformed picture (the plate
    minus the black padding of the conform, as `cue/new` reads it) where it meets the canvas. None without a canvas."""
    if cue.canvas_rect is None or cue.canvas_size is None:
        return None
    pw, ph = cue.plate_size
    ox, oy = (0, 0) if cue.transform is None else (max(cue.transform.offset[0], 0), max(cue.transform.offset[1], 0))
    (x0, y0), _ = plate_to_canvas((ox, oy), 0.0, cue.canvas_rect, cue.canvas_size)
    (x1, y1), _ = plate_to_canvas((pw - ox, ph - oy), 0.0, cue.canvas_rect, cue.canvas_size)
    cw, ch = cue.canvas_size
    return max(float(x0), 0.0), max(float(y0), 0.0), min(float(x1), float(cw)), min(float(y1), float(ch))


# --------------------------------------------------------------------------- burst

def burst_factors(t: float) -> tuple[float, float, float]:
    """(size x, light x, alpha) of a burst at progress t in 0..1: the size grows to BURST_SCALE (ease-out) over the first
    BURST_GROW_T, the light rises to BURST_FLASH with it and holds there until BURST_FADE_FROM, the alpha (what the matte
    is scaled by) is the fade alone, from BURST_FADE_FROM (the light fades with it). At t = 0 all are 1 (the effect as it was)."""
    grow = min(t / BURST_GROW_T, 1.0)
    scale = 1.0 + (BURST_SCALE - 1.0) * (1.0 - (1.0 - grow) ** 2)
    flash = min(t / BURST_FLASH_PEAK_T, 1.0)
    alpha = 1.0 - smoothstep((t - BURST_FADE_FROM) / (1.0 - BURST_FADE_FROM))
    return scale, (1.0 + (BURST_FLASH - 1.0) * flash) * alpha, alpha


# --------------------------------------------------------------------------- shatter

def shatter_alpha(t: float) -> float:
    """What the shards are scaled by at progress t: 1 (fully bright) until SHATTER_FADE_FROM, then smoothly to 0."""
    return 1.0 - smoothstep((t - SHATTER_FADE_FROM) / (1.0 - SHATTER_FADE_FROM))


@dataclass(frozen=True)
class Shard:
    """One fragment. Lengths are in effect diameters (d) at vanish.start, angles in degrees unless said."""
    angle: float                                  # outward direction, radians (picture: x right, y down)
    speed: float                                  # d per second
    size: float                                   # bounding diameter, >= SHARD_MIN_SIZE
    corners: tuple[tuple[float, float], ...]      # the polygon, in a circle of radius 1 around its centre
    spin: float                                   # in-plane, degrees per second
    phase: float                                  # in-plane angle at k = 0
    tumble: float                                 # degrees per second about the in-plane axis `axis`
    axis: float                                   # radians
    tilt0: float                                  # tumble angle at k = 0


def shards(seed: int | None) -> list[Shard]:
    """The fragments of a shatter: 12..20 of them, their number, directions, sizes and motion all from `seed` (None = 0).
    Directions are spread round the circle (one per 1/n of it, jittered), so the break reads as an explosion."""
    rng = np.random.default_rng([abs(int(seed or 0)), 36])
    n = int(rng.integers(SHARD_COUNT[0], SHARD_COUNT[1] + 1))
    out = []
    for i in range(n):
        angle = 2 * math.pi * (i + rng.uniform(-0.4, 0.4)) / n
        m = int(rng.integers(3, 5))  # a triangle or a four-cornered shard
        a0 = rng.uniform(0, 2 * math.pi)
        angles = np.sort(a0 + 2 * math.pi * (np.arange(m) + rng.uniform(-0.25, 0.25, m)) / m)
        radii = rng.uniform(0.55, 1.0, m)
        radii[int(rng.integers(m))] = 1.0  # the polygon touches its circle: `size` is its true bounding diameter
        out.append(Shard(angle=angle, speed=float(rng.uniform(*SHARD_SPEED)),
                         size=max(float(rng.uniform(*SHARD_SIZE)), SHARD_MIN_SIZE),
                         corners=tuple((float(r * math.cos(a)), float(r * math.sin(a))) for a, r in zip(angles, radii)),
                         spin=float(rng.uniform(*SHARD_SPIN) * rng.choice((-1, 1))), phase=float(rng.uniform(0, 360)),
                         tumble=float(rng.uniform(*SHARD_TUMBLE) * rng.choice((-1, 1))),
                         axis=float(rng.uniform(0, math.pi)), tilt0=float(rng.uniform(0, 360))))
    return out


def shard_grey(shard: Shard, tau: float, albedo: float, ambient: float, light: tuple[float, float, float]) -> float:
    """The flat grey of a shard `tau` seconds into the ending: the sphere's matte Lambert shading of the shard's normal,
    which tumbles about `shard.axis` (so shards flicker as they turn)."""
    a = math.radians(shard.tilt0 + shard.tumble * tau)
    ux, uy = math.cos(shard.axis), math.sin(shard.axis)
    n = np.array([uy * math.sin(a), -ux * math.sin(a), math.cos(a)])
    if n[2] < 0:
        n = -n  # a shard has two faces
    lv = np.asarray(light, dtype=np.float64)
    lambert = float(np.clip(n @ (lv / np.linalg.norm(lv)), 0.0, 1.0))
    return albedo * (ambient + (1.0 - ambient) * lambert)


def confine(cx: float, cy: float, dx: float, dy: float, r: float,
            bounds: tuple[float, float, float, float] | None) -> tuple[float, float]:
    """The centre (cx + dx, cy + dy) of a shard of radius `r`, kept so the whole shard is inside `bounds` (x0, y0, x1, y1).
    The way to the edge along (dx, dy) is the limit: a shard that would pass SHARD_SOFT_FROM of it slows smoothly and
    never passes it (so shards drift towards the edge and stay inside, they do not stop dead). `bounds` None leaves it alone."""
    if bounds is None:
        return cx + dx, cy + dy
    x0, y0, x1, y1 = bounds
    lx, hx, ly, hy = x0 + r, x1 - r, y0 + r, y1 - r
    cx = (lx + hx) / 2 if lx > hx else min(max(cx, lx), hx)  # a picture too small for the shard: its centre
    cy = (ly + hy) / 2 if ly > hy else min(max(cy, ly), hy)
    room_x, room_y = (hx - cx) if dx > 0 else (cx - lx), (hy - cy) if dy > 0 else (cy - ly)
    ratio = max(abs(dx) / room_x if room_x > 1e-9 else (math.inf if dx else 0.0),
                abs(dy) / room_y if room_y > 1e-9 else (math.inf if dy else 0.0))
    if ratio <= SHARD_SOFT_FROM:
        return cx + dx, cy + dy
    if not math.isfinite(ratio):  # no room at all on that side
        return cx, cy
    soft = SHARD_SOFT_FROM + (1 - SHARD_SOFT_FROM) * math.tanh((ratio - SHARD_SOFT_FROM) / (1 - SHARD_SOFT_FROM))
    return cx + dx * soft / ratio, cy + dy * soft / ratio


def shard_polygon(shard: Shard, tau: float, cx: float, cy: float, d: float,
                  bounds: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """(m, 2) canvas pixels of a shard's corners `tau` seconds into the ending, for an effect of diameter `d` at (cx, cy)
    (the effect's centre in that frame): moved outward along its direction, pulled down by gravity, and turned. With
    `bounds` (the picture on the canvas) the shard is kept wholly inside it (`confine`)."""
    ca, sa = math.cos(shard.angle), math.sin(shard.angle)
    reach = SHARD_START_SPREAD + shard.speed * tau
    px, py = confine(cx, cy, d * reach * ca, d * (reach * sa + 0.5 * SHARD_GRAVITY * tau * tau), shard.size * d / 2, bounds)
    r = math.radians(shard.phase + shard.spin * tau)
    cr, sr = math.cos(r), math.sin(r)
    c = np.asarray(shard.corners, dtype=np.float64) * (shard.size * d / 2)
    return np.stack([px + c[:, 0] * cr - c[:, 1] * sr, py + c[:, 0] * sr + c[:, 1] * cr], axis=1)


def render_shards(shape: tuple[int, int], cx: float, cy: float, d: float, k: int, fps: float, seed: int | None,
                  albedo: float, ambient: float, light: tuple[float, float, float],
                  bounds: tuple[float, float, float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(grey, coverage), float32 (h, w) in 0..1: the shards of a shatter at frame k of the ending (k >= 1), drawn with
    anti-aliased edges, each in its flat grey. `d` and (cx, cy) are the effect's diameter at vanish.start and its centre
    in this frame (canvas pixels); `bounds` is the picture the shards stay inside (`EndingPlan.picture`). Neither
    is faded: the caller scales by the ending's alpha."""
    h, w = shape
    grey, cover = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    tau = k / fps
    for s in shards(seed):
        poly = shard_polygon(s, tau, cx, cy, d, bounds)
        x0, y0 = int(np.floor(poly[:, 0].min())) - 1, int(np.floor(poly[:, 1].min())) - 1
        x1, y1 = int(np.ceil(poly[:, 0].max())) + 2, int(np.ceil(poly[:, 1].max())) + 2
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            continue
        patch = np.zeros((y1 - y0, x1 - x0), np.uint8)
        pts = np.rint((poly - (x0, y0)) * (1 << SHARD_AA_SHIFT)).astype(np.int32)
        cv2.fillPoly(patch, [pts], 255, lineType=cv2.LINE_AA, shift=SHARD_AA_SHIFT)
        ix0, iy0, ix1, iy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        cov = patch[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0].astype(np.float32) / 255.0
        g = np.float32(shard_grey(s, tau, albedo, ambient, light))
        grey[iy0:iy1, ix0:ix1] = grey[iy0:iy1, ix0:ix1] * (1 - cov) + g * cov
        cover[iy0:iy1, ix0:ix1] = 1 - (1 - cover[iy0:iy1, ix0:ix1]) * (1 - cov)
    return grey, cover
