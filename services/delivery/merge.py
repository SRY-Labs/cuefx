"""Merge: composite the effect layers onto the untouched plate (BRIEF section 2.3).

All maths is in linear light, float32, at plate size. The plate is decoded as sRGB; it is really Rec.709 (the
transfer curves differ a little near black), and that difference is acceptable for now.

Per frame in effect_range, with E = effect, m = matte, g = grade.glow, o = grade.opacity:
    1. effect and matte mapped canvas -> plate (bilinear, `canvas_to_plate`); 0 outside canvas_rect
    2. black clamp: E = max(E - floor, 0), floor = the 5th-percentile luminance of the effect frame
    3. visibility: E and m times the frame's visibility
    4. E = E x brightness
    5. out = plate (1 - m o) + o (E m + g E (1 - m))
    6. behind ranges: out = out (1 - M) + (plate + s g blur(E)) M, M = occluder mask, s = spill, blur sigma
       = spill_blur x the orb's diameter on the plate (soft light on the hand, not a copy of the orb's edge)
    7. grain: Gaussian noise of the plate's own noise level, weighted by min(1, 4 x luminance of E)
Frames outside effect_range are the plate, untouched. Registration (section 2.3 step 3) comes in a later task.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from services.common.canvas import Rect, canvas_to_plate
from services.common.colour import decode8, encode8, linear_to_srgb, luminance
from services.cue.cue import Cue, Grade
from services.delivery.debug import compare_frame, mask_tile, merge_tile, write_masks_sheet, write_merge_sheet
from services.analysis.masks import MaskConfig, OccluderTrack, masks_for_frame, occluder_mask, occluder_tracks
from services.cue.route import OrbSample
from services.delivery.video import BT709_OUT, VideoError, VideoWriter, add_audio, probe, read_frames


@dataclass(frozen=True)
class MergeConfig:
    spill: float = 0.15            # glow left on the occluder in behind ranges, as a share of the glow
    spill_blur: float = 0.3        # the spill is the effect blurred by this x the orb's diameter (plate px)
    black_percentile: float = 5.0  # the effect's black floor is this percentile of its luminance
    grain_gain: float = 4.0        # grain weight = min(1, effect luminance x this)
    grain_seed: int = 0            # with the frame number: the same grain on every rerun
    crf: int = 16


# --------------------------------------------------------------------------- pure maths, one frame

def canvas_to_plate_image(img: np.ndarray, canvas_rect: Rect, canvas_size: tuple[int, int],
                          plate_size: tuple[int, int]) -> np.ndarray:
    """A canvas-size image mapped onto the plate through `canvas_to_plate` (bilinear); 0 outside canvas_rect."""
    (ox, oy), _ = canvas_to_plate([0.0, 0.0], 0.0, canvas_rect, canvas_size)
    (ex, ey), _ = canvas_to_plate([1.0, 1.0], 0.0, canvas_rect, canvas_size)
    sx, sy = ex - ox, ey - oy  # plate px per canvas px
    x, y, w, h = canvas_rect
    # plate = canvas * s + o in pixel-edge coordinates; warpAffine counts pixel centres at integers.
    # Drawn straight into the rect (so the shift is relative to its corner), then pasted.
    m = np.array([[sx, 0, 0.5 * sx + ox - 0.5 - x], [0, sy, 0.5 * sy + oy - 0.5 - y]], dtype=np.float64)
    inside = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    pw, ph = plate_size
    out = np.zeros((ph, pw) + img.shape[2:], dtype=img.dtype)
    out[y:y + h, x:x + w] = inside
    return out


def black_floor(effect: np.ndarray, percentile: float = 5.0) -> float:
    """The effect frame's black level: a low percentile of its luminance (linear)."""
    return float(np.percentile(luminance(effect), percentile))


def black_clamp(effect: np.ndarray, floor: float) -> np.ndarray:
    """E = max(E - floor, 0): LTX's not-quite-black would otherwise brighten the whole plate."""
    return np.maximum(effect - np.float32(floor), 0.0)


def composite(plate: np.ndarray, E: np.ndarray, m: np.ndarray, glow: float, opacity: float) -> np.ndarray:
    """out = plate (1 - m o) + o (E m + g E (1 - m)). The matte blocks the plate only under the solid core;
    the orb adds light. With g = 1 the two parts sum to E, so no edge gets darker twice."""
    m3 = m[..., None]
    return plate * (1 - m3 * opacity) + opacity * (E * m3 + glow * E * (1 - m3))


def spill_light(E: np.ndarray, orb_d: float, glow: float, spill: float, blur: float = 0.3,
                rect: Rect | None = None) -> np.ndarray:
    """s g blur(E): the soft light the orb throws onto an occluder in front of it. Gaussian sigma = blur x orb_d
    (the orb's diameter in plate px). Blurred only over the effect's bounding box plus 4 sigma; 0 outside `rect`
    (canvas_rect), like the effect itself, so the plate outside it stays untouched."""
    out = np.zeros_like(E)
    ys, xs = np.nonzero(E.max(axis=2) > 0)
    if len(ys) == 0 or spill == 0 or glow == 0:
        return out
    sigma = max(blur * orb_d, 0.5)
    pad = int(np.ceil(4 * sigma))
    h, w = E.shape[:2]
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    out[y0:y1, x0:x1] = cv2.GaussianBlur(E[y0:y1, x0:x1], (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
    out *= np.float32(spill * glow)
    if rect is not None:
        x, y, w, h = rect
        keep = np.zeros(E.shape[:2], bool)
        keep[y:y + h, x:x + w] = True
        out[~keep] = 0
    return out


def restore_behind(out: np.ndarray, plate: np.ndarray, spill: np.ndarray, M: np.ndarray) -> np.ndarray:
    """out (1 - M) + (plate + spill) M: the occluder's own pixels on top, plus the soft `spill_light` on it."""
    M3 = M[..., None]
    return out * (1 - M3) + (plate + spill) * M3


def noise_std(plate: np.ndarray) -> np.ndarray:
    """The plate's noise std per channel, (3,): median absolute deviation of the finest diagonal Haar high-pass.

    HH = (a - b - c + d) / 2 over each 2x2 block has the same std as white noise in the image, and almost no
    signal except at edges, which the median ignores. std = median(|HH|) / 0.6745.
    """
    h, w = plate.shape[0] // 2 * 2, plate.shape[1] // 2 * 2
    p = plate[:h, :w]
    hh = (p[0::2, 0::2] - p[0::2, 1::2] - p[1::2, 0::2] + p[1::2, 1::2]) / 2
    return np.median(np.abs(hh).reshape(-1, plate.shape[2]), axis=0).astype(np.float32) / np.float32(0.6745)


def grain(std: np.ndarray, weight: np.ndarray, seed: int, f: int) -> np.ndarray:
    """Gaussian noise of `std` (per channel) times `weight` (h, w), the same for the same (seed, f) every run.
    Zero wherever weight is 0. Noise is drawn only over the bounding box of the non-zero weight."""
    out = np.zeros(weight.shape + (len(std),), dtype=np.float32)
    ys, xs = np.nonzero(weight)
    if len(ys) == 0:
        return out
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rng = np.random.default_rng([seed, f])
    noise = rng.standard_normal((y1 - y0, x1 - x0, len(std)), dtype=np.float32) * std
    out[y0:y1, x0:x1] = noise * weight[y0:y1, x0:x1, None]
    return out


@dataclass
class MergedFrame:
    """One merged frame: the 8-bit result, and the layers (linear, plate size) for export and the sheets."""
    f: int
    out8: np.ndarray
    plate8: np.ndarray | None = None
    E: np.ndarray | None = None      # effect after clamp, visibility and brightness
    m: np.ndarray | None = None      # matte after visibility
    masks: list[np.ndarray] = field(default_factory=list)
    seconds: float = 0.0             # time spent on this frame's maths


def merge_frame(f: int, plate8: np.ndarray, effect8: np.ndarray, matte8: np.ndarray, vis: float, grade: Grade,
                canvas_rect: Rect, canvas_size: tuple[int, int], masks: list[np.ndarray], orb_d: float,
                cfg: MergeConfig = MergeConfig()) -> MergedFrame:
    """Merge one plate frame (8-bit BGR) with its effect and matte frames (8-bit, canvas size).
    `orb_d` is the orb's diameter on the plate this frame (it sets the spill blur)."""
    t0 = time.perf_counter()
    plate_size = (plate8.shape[1], plate8.shape[0])
    plate = decode8(plate8)
    effect_c = decode8(effect8)
    floor = black_floor(effect_c, cfg.black_percentile)  # measured on the canvas frame, before mapping
    E = canvas_to_plate_image(effect_c, canvas_rect, canvas_size, plate_size)
    matte_c = (matte8[..., 1] if matte8.ndim == 3 else matte8).astype(np.float32) / 255.0  # coverage: no transfer curve
    m = np.clip(canvas_to_plate_image(matte_c, canvas_rect, canvas_size, plate_size), 0.0, 1.0)
    E = black_clamp(E, floor)
    E *= np.float32(vis)
    m *= np.float32(vis)
    # TODO(Gate 1): decide whether grade.tint is applied to LTX output. Not applied here: the stand-in is
    # already drawn in the tint.
    E *= np.float32(grade.brightness)
    out = composite(plate, E, m, grade.glow, grade.opacity)
    if masks:
        spill = spill_light(E, orb_d, grade.glow, cfg.spill, cfg.spill_blur, canvas_rect)
        for M in masks:
            out = restore_behind(out, plate, spill, M)
    weight = np.minimum(1.0, luminance(E) * np.float32(cfg.grain_gain))
    out += grain(noise_std(plate), weight, cfg.grain_seed, f)
    # TODO: light on the person (grade.person_light, BRIEF 2.3 step 6) is not built yet; ignored.
    out8 = encode8(out)
    return MergedFrame(f, out8, plate8, E, m, masks, time.perf_counter() - t0)


# --------------------------------------------------------------------------- the clip

def merge_frames(cue: Cue, path: list[OrbSample], plates: Iterable[np.ndarray], effects: Iterator[np.ndarray],
                 mattes: Iterator[np.ndarray], tracks: list[OccluderTrack], cfg: MergeConfig = MergeConfig(),
                 mask_cfg: MaskConfig = MaskConfig()) -> Iterator[MergedFrame]:
    """Merge a whole clip. `path` gives the frames of the (trimmed) effect range and their visibility;
    `effects` and `mattes` yield one canvas frame per sample of `path`. Frames outside the range are the plate
    itself (the same array, untouched). If the effect or matte runs out early the remaining frames are the plate."""
    vis, diam = {s[0]: s[4] for s in path}, {s[0]: s[3] for s in path}
    for f, plate8 in enumerate(plates):
        if f not in vis:
            yield MergedFrame(f, plate8, plate8)
            continue
        e8, m8 = next(effects, None), next(mattes, None)
        if e8 is None or m8 is None:
            yield MergedFrame(f, plate8, plate8)
            continue
        masks = masks_for_frame(tracks, f, plate8.shape[:2], mask_cfg)
        yield merge_frame(f, plate8, e8, m8, vis[f], cue.grade, cue.canvas_rect, cue.canvas_size, masks, diam[f], cfg)


def layer_images(fr: MergedFrame, glow: float) -> tuple[np.ndarray, np.ndarray]:
    """16-bit BGRA export layers of one frame, sRGB-encoded colour:
    core = E m with alpha m; glow = g E (1 - m) with alpha = its luminance."""
    m3 = fr.m[..., None]
    core = fr.E * m3
    g = glow * fr.E * (1 - m3)

    def bgra16(rgb_lin: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        c = np.rint(linear_to_srgb(rgb_lin) * 65535).astype(np.uint16)
        a = np.rint(np.clip(alpha, 0, 1) * 65535).astype(np.uint16)
        return np.dstack([c, a])

    return bgra16(core, fr.m), bgra16(g, luminance(g))


# --------------------------------------------------------------------------- files

def merge_dirs(make: Path, out: Path, key: str, effect_dir: Path | None) -> tuple[Path, Path]:
    """(folder with effect.mp4 and matte.mp4, folder Merge writes to). With `effect_dir` (an ingested LTX run) its
    files are used and the output goes to <out>/<key>/<name of effect_dir>/, so runs do not overwrite each other."""
    if effect_dir is None:
        return make / key, out / key
    return effect_dir, out / key / effect_dir.name


@dataclass
class MergeReport:
    warnings: list[str]
    frames: int                    # plate frames written
    merged: int                    # of those, merged with an effect frame
    seconds_per_merged: float      # mean time of the maths per merged frame
    seconds_total: float           # wall time of the whole run
    noise_std: np.ndarray | None   # the plate's noise std (linear, B G R) on the first merged frame
    outputs: list[Path]


def sheet_frames(wanted: list[int], frames: int, path: list[OrbSample], n: int = 12) -> list[int]:
    """`wanted` if every one of them is in the clip, else `n` evenly spaced frames of the effect range."""
    if wanted and all(0 <= f < frames for f in wanted):
        return sorted(wanted)
    first, last = path[0][0], path[-1][0]
    return sorted(set(np.linspace(first, last, min(n, last - first + 1)).round().astype(int).tolist()))


def _canvas_frames(frames: Iterator[np.ndarray], canvas_size: tuple[int, int], name: str,
                   warnings: list[str]) -> Iterator[np.ndarray]:
    """Frames at canvas size: a file of another size is resized to it, with one warning."""
    warned = False
    for fr in frames:
        if (fr.shape[1], fr.shape[0]) != tuple(canvas_size):
            if not warned:
                warnings.append(f"{name}: frames are {fr.shape[1]}x{fr.shape[0]}, not the canvas "
                                f"{canvas_size[0]}x{canvas_size[1]}; resized")
                warned = True
            fr = cv2.resize(fr, tuple(canvas_size), interpolation=cv2.INTER_LINEAR)
        yield fr


def _clear(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.png"):
        old.unlink()
    return d


def run_merge(plate_video: Path, cue: Cue, path: list[OrbSample], make_dir: Path, out_dir: Path,
              landmarks: dict[str, np.ndarray] | None, wanted_sheet: list[int], export_layers: bool = False,
              cfg: MergeConfig = MergeConfig(), mask_cfg: MaskConfig = MaskConfig()) -> MergeReport:
    """Merge a clip from its Make files and write final.mp4, compare.mp4, merge_sheet.png, masks_sheet.png
    (and with `export_layers`, layers/core, layers/glow, layers/masks) into out_dir.

    `make_dir` holds effect.mp4 and matte.mp4: the Make folder, or an ingested LTX run's folder (`--effect-dir`).

    Raises VideoError only when there is nothing to merge: the plate cannot be read or has the wrong size, or
    effect.mp4 / matte.mp4 are missing. Everything else is a warning in the report.
    """
    t_start = time.perf_counter()
    warnings: list[str] = []
    info = probe(plate_video)
    if (info.width, info.height) != tuple(cue.plate_size):
        raise VideoError(f"{plate_video.name} is {info.width}x{info.height}, the cue's plate is "
                         f"{cue.plate_size[0]}x{cue.plate_size[1]}")
    files = {name: make_dir / f"{name}.mp4" for name in ("effect", "matte")}
    for name, p in files.items():
        if not p.exists():
            raise VideoError(f"{p} is missing: run Make first (scripts/make_standin_effect.py, or "
                             f"scripts/ingest_ltx.py for --effect-dir)")
    if abs(info.fps - cue.fps) > 0.01:
        warnings.append(f"fps: the plate is {info.fps:g} fps, the cue says {cue.fps:g}; writing at the plate's")
    tracks, w = occluder_tracks(cue, landmarks, mask_cfg)
    warnings += w
    out_dir.mkdir(parents=True, exist_ok=True)
    effects = _canvas_frames(read_frames(files["effect"]), cue.canvas_size, "effect.mp4", warnings)
    mattes = _canvas_frames(read_frames(files["matte"]), cue.canvas_size, "matte.mp4", warnings)
    sheet_at = set(sheet_frames(wanted_sheet, cue.frames, path))
    layers = {k: _clear(out_dir / "layers" / k) for k in ("core", "glow", "masks")} if export_layers else {}
    final_path = out_dir / ("final_video_only.mp4" if info.has_audio else "final.mp4")
    pw, ph = cue.plate_size
    sheet, mask_tiles = [], {t.index: [] for t in tracks}
    merged, seconds, std, n = 0, 0.0, None, 0
    with VideoWriter(final_path, (pw, ph), info.fps, cfg.crf, BT709_OUT) as final, \
            VideoWriter(out_dir / "compare.mp4", (pw, ph // 2 // 2 * 2), info.fps, cfg.crf, BT709_OUT) as compare:
        for fr in merge_frames(cue, path, read_frames(plate_video), effects, mattes, tracks, cfg, mask_cfg):
            n += 1
            final.write(fr.out8)
            compare.write(compare_frame(fr.plate8, fr.out8, fr.f)[:ph // 2 // 2 * 2])
            if fr.E is not None:
                merged += 1
                seconds += fr.seconds
                if std is None:
                    std = noise_std(decode8(fr.plate8))
            if fr.f in sheet_at:
                sheet.append(merge_tile(fr.out8, cue.canvas_rect, f"frame {fr.f}"))
            for t in tracks:
                if fr.f in t.points:
                    mask = occluder_mask(t.points[fr.f], (ph, pw), mask_cfg, t.sizes.get(fr.f))
                    mask_tiles[t.index].append(mask_tile(fr.plate8, mask, cue.canvas_rect,
                                                         f"behind[{t.index}] frame {fr.f}"))
                    if export_layers and mask is not None:
                        cv2.imwrite(str(layers["masks"] / f"mask_{fr.f:04d}_b{t.index}.png"),
                                    np.rint(mask * 255).astype(np.uint8))
            if export_layers and fr.E is not None:
                core, glow = layer_images(fr, cue.grade.glow)
                cv2.imwrite(str(layers["core"] / f"core_{fr.f:04d}.png"), core)
                cv2.imwrite(str(layers["glow"] / f"glow_{fr.f:04d}.png"), glow)
    if n != cue.frames:
        warnings.append(f"plate: decoded {n} frames, the cue says {cue.frames}; wrote {n}")
    if merged < len(path):
        warnings.append(f"effect: only {merged} of the {len(path)} effect frames were merged (the effect or matte "
                        f"file is shorter, or the plate is); the rest are the plate")
    outputs = [out_dir / "final.mp4", out_dir / "compare.mp4"]
    if info.has_audio:
        warnings += add_audio(final_path, plate_video, out_dir / "final.mp4")
        final_path.unlink()
    if sheet:
        write_merge_sheet(sheet, out_dir / "merge_sheet.png")
        outputs.append(out_dir / "merge_sheet.png")
    tiles = [tile for t in tracks for tile in mask_tiles[t.index]]
    stale = out_dir / "masks_sheet.png"
    if tiles:
        write_masks_sheet(tiles, stale)
        outputs.append(stale)
    elif stale.exists():
        stale.unlink()
    outputs += [layers[k] for k in layers]
    return MergeReport(warnings, n, merged, seconds / max(merged, 1), time.perf_counter() - t_start, std, outputs)
