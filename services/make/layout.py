"""The grey-ball layout video: one frame per plate frame in effect_range, on the effect canvas.

A matte Lambert sphere with faint rotating meridian stripes (or, with a cue `shape`, a ring, crystal or cube drawn by
the same renderer, `shape_layer`): exactly one clean ball per frame, with no motion-blur
copies or trails (LTX copies what it sees, so sub-frame copies on the throw became ghost orbs in the render). It is
drawn in every frame; visibility and fades belong to Merge, except the entrance: with entrance gains the ball grows from
nothing (diameter and brightness x gain) and is absent before it. Positions come from `orb_path` (plate pixels) and are
mapped to the canvas with `plate_to_canvas`. Coordinates are continuous, pixel i covers [i, i + 1).
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from services.common.canvas import Rect, plate_to_canvas
from services.cue.cue import Cue, shape_of
from services.make.effects import EndingPlan, burst_factors, render_shards, shatter_alpha
from services.cue.route import OrbSample


class LayoutError(RuntimeError):
    """A problem writing the layout video (no ffmpeg, encoder failure, wrong frame count)."""


@dataclass(frozen=True)
class LayoutConfig:
    albedo: float = 0.6
    ambient: float = 0.2                           # share of the light that reaches the shadow side
    light: tuple[float, float, float] = (-0.5, -0.5, 0.7)  # towards the light: upper left, in front (x right, y down, z to the viewer)
    stripes: int = 3                               # meridian bands around the ball
    stripe_amount: float = 0.08                    # +- brightness
    spin_deg_per_frame: float = 12.0               # rotation about the vertical axis
    crf: int = 12


def ball_layer(shape: tuple[int, int], cx: float, cy: float, diameter: float, spin_deg: float,
               cfg: LayoutConfig = LayoutConfig()) -> np.ndarray:
    """Float32 (h, w) image in 0..1 of one shaded ball on black, edge anti-aliased by pixel coverage.

    The disc is `clean_orb_proxy.render_proxy_frame`'s (coverage clip(r + 0.5 - distance) in a local box), in this
    module's pixel convention (pixel centres at i + 0.5, so the ball is not shifted half a pixel), times the shading.
    """
    h, w = shape
    r = diameter / 2
    out = np.zeros(shape, dtype=np.float32)
    x0, x1 = max(0, int(np.floor(cx - r - 1))), min(w, int(np.ceil(cx + r + 1)))
    y0, y1 = max(0, int(np.floor(cy - r - 1))), min(h, int(np.ceil(cy + r + 1)))
    if x0 >= x1 or y0 >= y1 or r <= 0:
        return out
    px = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5 - cx
    py = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5 - cy
    dist = np.sqrt(px * px + py * py)
    coverage = np.clip(r + 0.5 - dist, 0.0, 1.0)  # one pixel wide soft edge
    nx, ny = px / r, py / r
    nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, None))
    n = np.stack(np.broadcast_arrays(nx, ny, nz), axis=-1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    light = np.asarray(cfg.light, dtype=np.float32)
    lambert = np.clip(n @ (light / np.linalg.norm(light)), 0.0, 1.0)
    lon = np.arctan2(nx, nz) + np.deg2rad(spin_deg)  # longitude on the ball, turned by the spin
    stripe = 1.0 + cfg.stripe_amount * np.cos(cfg.stripes * lon)
    shade = cfg.albedo * (cfg.ambient + (1.0 - cfg.ambient) * lambert) * stripe
    out[y0:y1, x0:x1] = np.clip(shade, 0.0, 1.0) * coverage
    return out


# --------------------------------------------------------------------------- the other shapes (task 31)

@dataclass(frozen=True)
class ShapeSpec:
    """Constants of one shape, in the cue's units. `spin` is the default `spin_deg_per_s`."""
    spin: float


SHAPE_SPECS: dict[str, ShapeSpec] = {
    "sphere": ShapeSpec(288.0),   # today's stripe spin (LayoutConfig.spin_deg_per_frame 12 x 24 fps); the sphere keeps using the config
    "ring": ShapeSpec(90.0),      # degrees per second about the ring's own axis
    "crystal": ShapeSpec(60.0),   # about its vertical axis
    "cube": ShapeSpec(60.0),      # about an axis tilted CUBE_TILT_DEG from vertical
}
SUPERSAMPLE = 4          # n x n samples per pixel; the edge is their average (no motion-blur copies)
RING_TILT_DEG = 60.0     # the ring's plane is turned this far from facing the camera (about the x axis)
RING_TUBE = 0.18         # tube thickness, as a share of the diameter
RING_STRIPES = 6         # bands around the ring: a torus turned about its own axis would otherwise show no spin
RING_STRIPE_AMOUNT = 0.25
CRYSTAL_WIDTH = 0.6      # octahedron width at its middle, as a share of its height (height = the diameter)
CUBE_TILT_DEG = 20.0     # the cube's spin axis leans this far from vertical, towards the camera
MARCH_EPS = 5e-4         # sphere tracing: a ray hits the ring when within this share of the diameter of its surface
MARCH_STEPS = 64
CHUNK = 250_000          # rays per numpy pass, to bound memory


def _rot_x(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _shade(world_normal: np.ndarray, cfg: LayoutConfig) -> np.ndarray:
    """Matte Lambert grey of unit normals (n, 3): the sphere's albedo, ambient and light."""
    light = np.asarray(cfg.light, dtype=np.float64)
    lambert = np.clip(world_normal @ (light / np.linalg.norm(light)), 0.0, 1.0)
    return cfg.albedo * (cfg.ambient + (1.0 - cfg.ambient) * lambert)


def _polytope_planes(kind: str, size: float) -> tuple[np.ndarray, np.ndarray]:
    """(unit normals (k, 3), offsets (k,)) of the convex solid n . p <= h, centred on the origin, in its own frame.
    cube: side `size`. crystal: an octahedron `size` high and CRYSTAL_WIDTH x `size` wide, its axis along y."""
    if kind == "cube":
        n = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
        return n, np.full(6, size / 2)
    a, b = CRYSTAL_WIDTH * size / 2, size / 2
    signs = np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)], dtype=np.float64)
    raw = signs / np.array([a, b, a])  # the facet x/a + y/b + z/a = 1 and its seven mirror images
    length = np.linalg.norm(raw, axis=1)
    return raw / length[:, None], 1.0 / length


def _rotation(kind: str, spin_deg: float) -> np.ndarray:
    """Object -> world rotation of a ring, crystal or cube turned `spin_deg` about its spin axis."""
    if kind == "ring":
        return _rot_x(RING_TILT_DEG) @ _rot_z(spin_deg)   # the ring's own axis is its z axis
    if kind == "cube":
        return _rot_x(-CUBE_TILT_DEG) @ _rot_y(spin_deg)  # spin axis (the object's y) leans towards the camera
    return _rot_y(spin_deg)                                # crystal: vertical axis


def _cube_size(rot: np.ndarray, diameter: float) -> float:
    """Side of the cube whose smallest enclosing circle on screen (its "outer size") is `diameter` in this pose.
    A cube's silhouette breathes by about 10 % as it turns; the side follows it so the outer size stays put."""
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) @ rot.T
    return diameter / (2.0 * cv2.minEnclosingCircle(corners[:, :2].astype(np.float32))[1])


def _polytope_pass(kind: str, px: np.ndarray, py: np.ndarray, size: float, rot: np.ndarray, cfg: LayoutConfig
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(shade, hit) for rays through the screen offsets px, py (flat arrays): the facet of a cube or crystal nearest the
    viewer. Rays run along z (orthographic); a facet has one flat grey."""
    normals, offsets = _polytope_planes(kind, size)
    origin = np.stack([px, py, np.zeros_like(px)], axis=1) @ rot  # ray start, in the solid's frame
    denom = normals @ rot[2]                                      # world +z (towards the viewer) in the solid's frame
    num = offsets[None, :] - origin @ normals.T                   # n . p(t) <= h  <=>  t denom <= num
    with np.errstate(divide="ignore", invalid="ignore"):
        t = num / denom[None, :]
    near = np.where(denom[None, :] < 0, t, -np.inf).max(axis=1)   # the ray enters through these planes ...
    far_t = np.where(denom[None, :] > 0, t, np.inf)
    face = far_t.argmin(axis=1)                                   # ... and leaves through the nearest of these: what is seen
    far = far_t[np.arange(len(face)), face]
    hit = (near <= far) & ~((denom[None, :] == 0) & (num < 0)).any(axis=1)
    return _shade(normals[face] @ rot.T, cfg), hit


def _ring_pass(px: np.ndarray, py: np.ndarray, diameter: float, rot: np.ndarray, cfg: LayoutConfig
               ) -> tuple[np.ndarray, np.ndarray]:
    """(shade, hit) of a torus (outer diameter `diameter`, tube RING_TUBE x diameter) by sphere tracing along z, with
    RING_STRIPES faint bands around its axis so that its spin shows."""
    tube = RING_TUBE * diameter / 2
    major = diameter / 2 - tube
    bound = diameter / 2
    shade, hit = np.zeros(px.shape), np.zeros(px.shape, bool)
    inside = np.nonzero(px * px + py * py < bound * bound)[0]  # only rays through the bounding circle can hit
    if len(inside) == 0:
        return shade, hit
    z0 = np.sqrt(bound * bound - px[inside] ** 2 - py[inside] ** 2)
    start = np.stack([px[inside], py[inside], z0], axis=1) @ rot  # on the bounding sphere, in the ring's frame
    step = rot.T @ np.array([0.0, 0.0, -1.0])                     # away from the viewer, in the ring's frame
    travel, limit = np.zeros(len(inside)), 2 * z0
    active = np.arange(len(inside))
    eps = MARCH_EPS * diameter
    hits = []
    for _ in range(MARCH_STEPS):
        if len(active) == 0:
            break
        p = start[active] + travel[active, None] * step
        dist = np.hypot(np.hypot(p[:, 0], p[:, 1]) - major, p[:, 2]) - tube
        done = dist < eps
        if done.any():
            hits.append(active[done])
        travel[active] += dist
        active = active[~done & (travel[active] < limit[active])]
    if not hits:
        return shade, hit
    idx = np.concatenate(hits)
    p = start[idx] + travel[idx, None] * step
    rho = np.maximum(np.hypot(p[:, 0], p[:, 1]), 1e-9)
    n_obj = np.stack([p[:, 0] * (1 - major / rho), p[:, 1] * (1 - major / rho), p[:, 2]], axis=1)
    n_obj /= np.maximum(np.linalg.norm(n_obj, axis=1, keepdims=True), 1e-9)
    stripe = 1.0 + RING_STRIPE_AMOUNT * np.cos(RING_STRIPES * np.arctan2(p[:, 1], p[:, 0]))
    shade[inside[idx]] = _shade(n_obj @ rot.T, cfg) * stripe
    hit[inside[idx]] = True
    return shade, hit


def render_shape(kind: str, shape: tuple[int, int], cx: float, cy: float, diameter: float, spin_deg: float,
                 cfg: LayoutConfig = LayoutConfig()) -> tuple[np.ndarray, np.ndarray]:
    """(image, mask), float32 (h, w) in 0..1: one ring, crystal or cube on black, and its anti-aliased coverage.

    Same position and `diameter` (the shape's outer size on screen: the smallest circle that holds it), the same
    pixel-centre convention (pixel i covers [i, i + 1)), grey shading and light as `ball_layer`; `spin_deg` turns the
    shape about its spin axis (SHAPE_SPECS, RING_*, CRYSTAL_*, CUBE_*). SUPERSAMPLE x SUPERSAMPLE samples per pixel.
    """
    if kind not in ("ring", "crystal", "cube"):
        raise ValueError(f"render_shape draws a ring, crystal or cube, not {kind!r}")
    h, w = shape
    image, mask = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    r = diameter / 2
    x0, x1 = max(0, int(np.floor(cx - r - 1))), min(w, int(np.ceil(cx + r + 1)))
    y0, y1 = max(0, int(np.floor(cy - r - 1))), min(h, int(np.ceil(cy + r + 1)))
    if x0 >= x1 or y0 >= y1 or r <= 0:
        return image, mask
    ss = SUPERSAMPLE
    sx = x0 + (np.arange((x1 - x0) * ss) + 0.5) / ss - cx
    sy = y0 + (np.arange((y1 - y0) * ss) + 0.5) / ss - cy
    px, py = (a.ravel() for a in np.meshgrid(sx, sy))
    rot = _rotation(kind, spin_deg)
    size = _cube_size(rot, diameter) if kind == "cube" else diameter
    shade, hit = np.zeros(px.shape), np.zeros(px.shape, bool)
    for a in range(0, len(px), CHUNK):
        s = slice(a, a + CHUNK)
        if kind == "ring":
            shade[s], hit[s] = _ring_pass(px[s], py[s], diameter, rot, cfg)
        else:
            shade[s], hit[s] = _polytope_pass(kind, px[s], py[s], size, rot, cfg)
    grid = (len(sy), len(sx))

    def average(a: np.ndarray) -> np.ndarray:
        return a.reshape(y1 - y0, ss, x1 - x0, ss).mean(axis=(1, 3))

    image[y0:y1, x0:x1] = average((np.clip(shade, 0.0, 1.0) * hit).reshape(grid))
    mask[y0:y1, x0:x1] = average(hit.reshape(grid).astype(np.float64))
    return image, mask


def shape_layer(kind: str, shape: tuple[int, int], cx: float, cy: float, diameter: float, spin_deg: float,
                cfg: LayoutConfig = LayoutConfig()) -> np.ndarray:
    """One shape on black, float32 (h, w) in 0..1: `ball_layer` for the sphere (untouched), `render_shape` otherwise."""
    if kind == "sphere":
        return ball_layer(shape, cx, cy, diameter, spin_deg, cfg)
    return render_shape(kind, shape, cx, cy, diameter, spin_deg, cfg)[0]


def spin_per_frame(cue: Cue) -> float | None:
    """Degrees the cue's shape turns per frame: `spin_deg_per_s` / fps, or the shape's default; None for a sphere
    without a spin of its own (the layout then uses `LayoutConfig.spin_deg_per_frame`, as it always did)."""
    if cue.spin_deg_per_s is not None:
        return cue.spin_deg_per_s / cue.fps
    kind = shape_of(cue)
    return None if kind == "sphere" else SHAPE_SPECS[kind].spin / cue.fps


def canvas_path(path: list[OrbSample], canvas_rect: Rect, canvas_size: tuple[int, int]) -> np.ndarray:
    """(n, 3): x, y, diameter on the canvas for every sample of `path`."""
    xy, d = plate_to_canvas(np.array([[s[1], s[2]] for s in path]), np.array([s[3] for s in path]),
                            canvas_rect, canvas_size)
    return np.column_stack([xy, d])


def render_frame(cpath: np.ndarray, i: int, canvas_size: tuple[int, int],
                 cfg: LayoutConfig = LayoutConfig(), kind: str = "sphere", spin: float | None = None) -> np.ndarray:
    """Layout frame i (index into `cpath`) as float32 (h, w) in 0..1: one shape (`kind`, a ball by default) at cpath[i],
    spun `spin` (default `cfg.spin_deg_per_frame`) degrees x i. No motion blur, however far it moves to the next frame."""
    cw, ch = canvas_size
    x, y, d = cpath[i]
    return shape_layer(kind, (ch, cw), x, y, d, (cfg.spin_deg_per_frame if spin is None else spin) * i, cfg)


def to_bgr8(frame: np.ndarray) -> np.ndarray:
    """Float (h, w) in 0..1 to a grey 8-bit BGR image."""
    g = np.clip(np.rint(frame * 255.0), 0, 255).astype(np.uint8)
    return np.dstack([g, g, g])


def apply_gains(cpath: np.ndarray, gains: list[float] | None) -> np.ndarray:
    """`cpath` with each diameter scaled by its frame's entrance gain (`services.cue.route.entrance_gain`); as is without."""
    if gains is None:
        return cpath
    out = cpath.copy()
    out[:, 2] *= np.asarray(gains, dtype=float)
    return out


def ending_frame(cpath: np.ndarray, i: int, canvas_size: tuple[int, int], cfg: LayoutConfig, kind: str,
                 spin: float | None, gain: float, plan: EndingPlan) -> np.ndarray:
    """Layout frame i of a cue with a burst or shatter ending (task 36): the plain frame before the ending starts (and
    in its first frame), the grown and flashing shape of a burst, the grey shards of a shatter, black after it."""
    cw, ch = canvas_size
    st = plan.states[i]
    g = np.float32(gain)
    if st is None or (st.kind == "shatter" and st.k == 0):
        return render_frame(cpath, i, canvas_size, cfg, kind, spin) * g
    if st.kind == "gone":
        return np.zeros((ch, cw), np.float32)
    x, y, d = cpath[i]
    if st.kind == "burst":
        scale, light, _ = burst_factors(st.t)
        return shape_layer(kind, (ch, cw), x, y, d * scale, (cfg.spin_deg_per_frame if spin is None else spin) * i, cfg) * np.float32(light) * g
    grey, _ = render_shards((ch, cw), x, y, float(cpath[st.ref][2]), st.k, plan.fps, plan.seed, cfg.albedo, cfg.ambient,
                            cfg.light, plan.picture)
    return grey * np.float32(shatter_alpha(st.t)) * g


def render_layout(path: list[OrbSample], canvas_rect: Rect, canvas_size: tuple[int, int],
                  cfg: LayoutConfig = LayoutConfig(), gains: list[float] | None = None, kind: str = "sphere",
                  spin: float | None = None, ending: EndingPlan | None = None,
                  lights: list[float] | None = None) -> list[np.ndarray]:
    """All layout frames as 8-bit BGR images, one per sample of `path`.

    With `gains` (one per sample, 0..1: the entrance) the ball's diameter and brightness are both scaled by its
    frame's gain, so it grows from nothing; a gain of 0 is a black frame. `kind` and `spin` (degrees per frame) are
    the cue's shape and its spin (`spin_per_frame`); the defaults draw the ball as always. `ending` (a burst or
    shatter, `services.make.effects.ending_plan`) draws that ending from the vanish on; None draws nothing of it. `lights` (one
    per sample, `services.cue.route.orb_lights`) scales the brightness only, on top of the gains: the charge-up pulse.
    """
    cpath = apply_gains(canvas_path(path, canvas_rect, canvas_size), gains)
    bright = gains if lights is None else [(1.0 if gains is None else g) * x for g, x in zip(gains or [1.0] * len(path), lights)]
    if ending is not None:
        return [to_bgr8(ending_frame(cpath, i, canvas_size, cfg, kind, spin, 1.0 if bright is None else bright[i], ending))
                for i in range(len(path))]
    if bright is None:
        return [to_bgr8(render_frame(cpath, i, canvas_size, cfg, kind, spin)) for i in range(len(path))]
    return [to_bgr8(render_frame(cpath, i, canvas_size, cfg, kind, spin) * np.float32(g)) for i, g in enumerate(bright)]


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise LayoutError("ffmpeg is not on PATH; it is needed to write the H.264 layout video")
    return exe


SWS_ACCURATE = ["-sws_flags", "accurate_rnd+full_chroma_int+bitexact"]  # ffmpeg's default RGB->YUV darkens grey 128 to ~124


def write_video(frames: list[np.ndarray], out: Path, fps: float, crf: int = 12) -> None:
    """Encode 8-bit BGR frames as H.264, yuv420p, with ffmpeg (libx264)."""
    h, w = frames[0].shape[:2]
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [find_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", f"{fps:g}", "-i", "-", *SWS_ACCURATE, "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    proc = subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames),
                          capture_output=True)
    if proc.returncode:
        raise LayoutError(f"ffmpeg failed: {proc.stderr.decode(errors='replace').strip()}")


def count_frames(video: Path) -> int:
    """Number of frames actually decoded from `video`."""
    cap = cv2.VideoCapture(str(video))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def write_layout(path: list[OrbSample], canvas_rect: Rect, canvas_size: tuple[int, int], fps: float, out_dir: Path,
                 cfg: LayoutConfig = LayoutConfig(), gains: list[float] | None = None, kind: str = "sphere",
                 spin: float | None = None, ending: EndingPlan | None = None,
                 lights: list[float] | None = None) -> list[np.ndarray]:
    """Write layout.mp4 and layout_png/ into out_dir; returns the frames (`render_layout`, with the entrance `gains`).

    Raises LayoutError if the decoded video does not have exactly one frame per sample of `path`.
    """
    frames = render_layout(path, canvas_rect, canvas_size, cfg, gains, kind, spin, ending, lights)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_video(frames, out_dir / "layout.mp4", fps, cfg.crf)
    png_dir = out_dir / "layout_png"
    png_dir.mkdir(exist_ok=True)
    for old in png_dir.glob("*.png"):
        old.unlink()
    for s, frame in zip(path, frames):
        cv2.imwrite(str(png_dir / f"layout_{s[0]:04d}.png"), frame)
    n = count_frames(out_dir / "layout.mp4")
    if n != len(path):
        raise LayoutError(f"layout.mp4 has {n} frames, expected {len(path)}")
    return frames
