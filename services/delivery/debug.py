"""Debug outputs a human can check by eye: seed_check.png, track_debug.mp4, track_sheet.png, route_debug.mp4,
route_sheet.png, layout_overlay.mp4, layout_sheet.png, look_check.png, standin_sheet.png, and the pieces of
merge_sheet.png, masks_sheet.png and compare.mp4."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from services.common.canvas import canvas_to_plate
from services.cue.cue import Cue, HandSeed, lost_runs, no_flight
from services.cue.route import OrbSample, corrected_track, release_point
from services.analysis.track import Hand, Selection, TrackError, TrackResult
from services.cue.window import effect_window

# MediaPipe hand skeleton: wrist, then thumb / index / middle / ring / pinky chains.
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
         (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
GREEN, YELLOW, RED, MAGENTA, GREY, WHITE, CYAN = (0, 220, 0), (0, 230, 255), (0, 0, 230), (255, 0, 255), \
    (150, 150, 150), (255, 255, 255), (255, 200, 0)  # BGR
SHEET_COLS, SHEET_MAX, SHEET_W = 5, 25, 1920


def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _skeleton(img: np.ndarray, xyz: np.ndarray, color, thick: int) -> None:
    for a, b in BONES:
        cv2.line(img, _pt(xyz[a]), _pt(xyz[b]), color, thick, cv2.LINE_AA)
    for p in xyz:
        cv2.circle(img, _pt(p), thick + 1, color, -1, cv2.LINE_AA)


def _text(img: np.ndarray, s: str, org: tuple[int, int], scale: float, color=WHITE) -> None:
    t = max(1, round(scale * 2))
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), t + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, t, cv2.LINE_AA)


def seed_check_image(frame: np.ndarray, hands: list[Hand], seed: HandSeed, chosen: int) -> np.ndarray:
    """Seed frame with every detected skeleton, its label and score, the click (cross), the chosen palm (circle)."""
    img, s = frame.copy(), frame.shape[0] / 1080
    for i, h in enumerate(hands):
        _skeleton(img, h.xyz, GREEN if i == chosen else CYAN, max(1, round(3 * s)))
        c = h.palm_center
        _text(img, f"{h.label} {h.score:.2f} x={c[0]:.0f}", (int(c[0]) - 60, int(c[1] + 1.2 * h.palm_width)), 0.8 * s)
    r = int(0.9 * hands[chosen].palm_width)
    cv2.circle(img, _pt(hands[chosen].palm_center), r, YELLOW, max(2, round(4 * s)), cv2.LINE_AA)
    cv2.drawMarker(img, _pt((seed.x, seed.y)), RED, cv2.MARKER_CROSS, int(60 * s), max(2, round(4 * s)))
    _text(img, f"seed frame {seed.f}: cross = click, circle = chosen palm", (20, int(50 * s)), 1.0 * s)
    return img


def _debug_frame(frame: np.ndarray, f: int, res: TrackResult, lost: np.ndarray) -> np.ndarray:
    img, s = frame.copy(), frame.shape[0] / 1080
    other = res.other[f]
    if not np.isnan(other).any():
        _skeleton(img, other, GREY, max(1, round(2 * s)))
    p = res.points[f]
    color = RED if lost[f] else GREEN if p.source == "tracked" else YELLOW
    cv2.circle(img, _pt((p.x, p.y)), max(4, int(0.12 * p.palm_w)), color, -1, cv2.LINE_AA)
    if res.ambiguous[f]:
        cv2.circle(img, _pt((p.x, p.y)), max(8, int(0.3 * p.palm_w)), MAGENTA, max(2, round(3 * s)), cv2.LINE_AA)
    _text(img, f"frame {f}", (20, int(50 * s)), 1.2 * s)
    return img


def write_candidate_table(hands: list[list[Hand]], sel: Selection, path: Path) -> None:
    """Every detected hand in every frame with its tracklet and the chain it ended up in."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "x", "y", "palm_w", "label", "score", "tracklet", "chain"])
        for f, cands in enumerate(hands):
            for j, h in enumerate(cands):
                t = sel.cand_tracklet[f][j]
                c = h.palm_center
                w.writerow([f, f"{c[0]:.0f}", f"{c[1]:.0f}", f"{h.palm_width:.0f}", h.label, f"{h.score:.2f}",
                            t, sel.tracklets[t].chain])


def write_debug_outputs(video: Path, hands: list[list[Hand]], sel: Selection, seed: HandSeed,
                        res: TrackResult, out_dir: Path) -> None:
    """Write seed_check.png, track_debug.mp4, track_sheet.png and candidates.csv into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_candidate_table(hands, sel, out_dir / "candidates.csv")
    n, info = res.info.frames, res.info
    lost = np.zeros(n, dtype=bool)
    for a, b in res.lost:
        lost[a:b] = True
    sheet_frames = sorted(set(np.linspace(0, n - 1, min(SHEET_MAX, n)).round().astype(int).tolist()))
    tile_w = SHEET_W // SHEET_COLS
    tile_h = round(tile_w * info.height / info.width)
    tiles = []
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(out_dir / "track_debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             info.fps, (info.width, info.height))
    if not writer.isOpened():
        raise TrackError("cannot write track_debug.mp4")
    for f in range(n):
        ok, frame = cap.read()
        if not ok:
            raise TrackError(f"re-reading {video}: frame {f} missing")
        if f == seed.f:
            cv2.imwrite(str(out_dir / "seed_check.png"),
                        seed_check_image(frame, hands[f], seed, sel.chosen[f]))
        img = _debug_frame(frame, f, res, lost)
        writer.write(img)
        if f in sheet_frames:
            tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            _text(tile, str(f), (6, tile_h - 8), 0.8)
            tiles.append(tile)
    cap.release()
    writer.release()
    _write_sheet(tiles, out_dir / "track_sheet.png")


def _write_sheet(tiles: list[np.ndarray], path: Path, cols: int = SHEET_COLS) -> None:
    """Tiles in rows of `cols`, the last row padded with black."""
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    blank = np.zeros_like(tiles[0])
    rows = [np.hstack(r + [blank] * (cols - len(r))) for r in rows]
    cv2.imwrite(str(path), np.vstack(rows))


def _route_frame(frame: np.ndarray, f: int, cue: Cue, sample: OrbSample | None, palm: tuple[float, float, bool],
                 crosses: list[tuple[float, float]]) -> np.ndarray:
    """Plate with the orb (circle of its diameter, dimmed by visibility), route crosses, palm dot, frame number."""
    img, s = frame.copy(), frame.shape[0] / 1080
    for x, y in crosses:
        cv2.drawMarker(img, _pt((x, y)), YELLOW, cv2.MARKER_CROSS, int(40 * s), max(2, round(3 * s)))
    px, py, lost = palm
    cv2.circle(img, _pt((px, py)), max(4, int(6 * s)), RED if lost else GREEN, -1, cv2.LINE_AA)
    phase = "outside effect_range"
    if sample is not None:
        _, x, y, d, vis = sample
        glow = img.copy()
        cv2.circle(glow, _pt((x, y)), max(1, int(d / 2)), (255, 255, 255), -1, cv2.LINE_AA)
        img = cv2.addWeighted(glow, 0.35 * vis, img, 1 - 0.35 * vis, 0)
        cv2.circle(img, _pt((x, y)), max(1, int(d / 2)), tuple(int(c * vis) for c in CYAN), max(2, round(4 * s)), cv2.LINE_AA)
        held = no_flight(cue) or f < cue.release_frame
        phase = ("held" if held else "flying") + f", visibility {vis:.2f}"
    _text(img, f"frame {f}  {phase}", (20, int(50 * s)), 1.0 * s)
    return img


def write_route_debug(video: Path, cue: Cue, path: list[OrbSample], out_dir: Path) -> None:
    """Write route_debug.mp4 (every plate frame) and route_sheet.png (up to 25 frames inside effect_range)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    er, n = effect_window(cue), cue.frames
    by_frame = {s[0]: s for s in path}
    palms = corrected_track(cue)
    lost = np.zeros(n, dtype=bool)
    for a, b in lost_runs(cue.track):
        lost[a:b] = True
    crosses = [] if no_flight(cue) else [release_point(cue)] + [(p.x, p.y) for p in cue.route[1:]]
    sheet_frames = set(np.linspace(er.start, er.end - 1, min(SHEET_MAX, er.length)).round().astype(int).tolist())
    w, h = cue.plate_size
    tile_w = SHEET_W // SHEET_COLS
    tile_h = round(tile_w * h / w)
    tiles = []
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(out_dir / "route_debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), cue.fps, (w, h))
    if not writer.isOpened():
        raise TrackError("cannot write route_debug.mp4")
    for f in range(n):
        ok, frame = cap.read()
        if not ok:
            raise TrackError(f"re-reading {video}: frame {f} missing")
        img = _route_frame(frame, f, cue, by_frame.get(f), (palms[f].x, palms[f].y, bool(lost[f])), crosses)
        writer.write(img)
        if f in sheet_frames:
            tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            _text(tile, str(f), (6, tile_h - 8), 0.8)
            tiles.append(tile)
    cap.release()
    writer.release()
    _write_sheet(tiles, out_dir / "route_sheet.png")


def write_layout_overlay(video: Path, cue: Cue, layout: list[np.ndarray], out: Path) -> None:
    """layout_overlay.mp4: the plate for every frame in effect_range with the layout mapped back to plate pixels
    (through `canvas_to_plate`) and added at 50 %, the canvas_rect outline and the frame number."""
    er, (w, h) = effect_window(cue), cue.plate_size
    cw, ch = cue.canvas_size
    (x0, y0), d = canvas_to_plate([0, 0], cw, cue.canvas_rect, cue.canvas_size)  # canvas origin and 1 canvas px in plate px
    k = float(d) / cw
    m = np.array([[k, 0, x0], [0, k, y0]], dtype=np.float64)  # canvas -> plate: pure scale and shift
    rx, ry, rw, rh = cue.canvas_rect
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), cue.fps, (w, h))
    if not writer.isOpened():
        raise TrackError(f"cannot write {out.name}")
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, er.start)
    for i, f in enumerate(range(er.start, er.end)):
        ok, frame = cap.read()
        if not ok:
            raise TrackError(f"re-reading {video}: frame {f} missing")
        mapped = cv2.warpAffine(layout[i], m, (w, h), flags=cv2.INTER_LINEAR)
        img = cv2.add(frame, (mapped * 0.5).astype(np.uint8))
        cv2.rectangle(img, (rx, ry), (rx + rw - 1, ry + rh - 1), YELLOW, max(2, round(3 * h / 1080)))
        _text(img, f"frame {f}  layout {i}", (20, int(50 * h / 1080)), 1.0 * h / 1080)
        writer.write(img)
    cap.release()
    writer.release()


def write_layout_sheet(cue: Cue, layout: list[np.ndarray], out: Path) -> None:
    """layout_sheet.png: up to 25 evenly spaced layout frames, each labelled with its plate frame."""
    er = effect_window(cue)
    idx = sorted(set(np.linspace(0, len(layout) - 1, min(SHEET_MAX, len(layout))).round().astype(int).tolist()))
    tile_w = SHEET_W // SHEET_COLS
    tile_h = round(tile_w * layout[0].shape[0] / layout[0].shape[1])
    tiles = []
    for i in idx:
        tile = cv2.resize(layout[i], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        _text(tile, f"{er.start + i}", (6, tile_h - 8), 0.8)
        tiles.append(tile)
    _write_sheet(tiles, out)


def write_standin_sheet(path: list[OrbSample], effect: list[np.ndarray], matte: list[np.ndarray], out: Path) -> None:
    """standin_sheet.png: up to 25 evenly spaced stand-in effect frames, the matte's 50 % edge drawn in yellow,
    each labelled with its plate frame."""
    idx = sorted(set(np.linspace(0, len(effect) - 1, min(SHEET_MAX, len(effect))).round().astype(int).tolist()))
    tile_w = SHEET_W // SHEET_COLS
    tile_h = round(tile_w * effect[0].shape[0] / effect[0].shape[1])
    tiles = []
    for i in idx:
        img = effect[i].copy()
        edge = (matte[i][..., 1] >= 128).astype(np.uint8)
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, YELLOW, 1, cv2.LINE_AA)
        tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        _text(tile, f"{path[i][0]}", (6, tile_h - 8), 0.8)
        tiles.append(tile)
    _write_sheet(tiles, out)


def entrance_crop(cue: Cue, sample: OrbSample, scale: float = 4.0) -> tuple[int, int, int, int]:
    """[x, y, w, h] on the plate: a square `scale` x the orb's held diameter wide around `sample`'s centre, kept inside
    the plate (and no larger than it)."""
    pw, ph = cue.plate_size
    side = int(min(round(scale * cue.diameter_px), pw, ph))
    x = int(min(max(round(sample[1] - side / 2), 0), pw - side))
    y = int(min(max(round(sample[2] - side / 2), 0), ph - side))
    return x, y, side, side


def write_entrance_sheet(frames: list[tuple[int, np.ndarray, str]], crop: tuple[int, int, int, int], out: Path,
                         cols: int = 6) -> None:
    """entrance_sheet.png: each (plate frame, comp frame, label) cropped to `crop`, side by side in rows of `cols`,
    labelled, so the stand-in can be seen growing."""
    x, y, w, h = crop
    tile_w = SHEET_W // cols
    tile_h = round(tile_w * h / w)
    tiles = []
    for f, img, label in frames:
        tile = cv2.resize(img[y:y + h, x:x + w], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        _text(tile, f"{f}", (6, 26), 0.8)
        _text(tile, label, (6, tile_h - 8), 0.55)
        tiles.append(tile)
    _write_sheet(tiles, out, cols)


def write_ingest_sheet(first_frame: int, effect: list[np.ndarray], matte: list[np.ndarray], out: Path,
                       n: int = 8) -> None:
    """ingest_sheet.png: `n` evenly spaced frames, each as effect | matte side by side, labelled with its plate
    frame (`first_frame` is the plate frame of the first effect frame). Two pairs per row."""
    idx = sorted(set(np.linspace(0, len(effect) - 1, min(n, len(effect))).round().astype(int).tolist()))
    tile_w = SHEET_W // 4
    tile_h = round(tile_w * effect[0].shape[0] / effect[0].shape[1])
    tiles = []
    for i in idx:
        pair = [cv2.resize(img[i], (tile_w, tile_h), interpolation=cv2.INTER_AREA) for img in (effect, matte)]
        _text(pair[0], f"{first_frame + i}", (6, tile_h - 8), 0.8)
        tiles.append(np.hstack(pair))
    _write_sheet(tiles, out, 2)


def _crop(img: np.ndarray, rect) -> np.ndarray:
    x, y, w, h = rect
    return img[y:y + h, x:x + w].copy()


def merge_tile(final8: np.ndarray, canvas_rect, label: str) -> np.ndarray:
    """A merged frame cropped to canvas_rect at full resolution, labelled."""
    tile = _crop(final8, canvas_rect)
    _text(tile, label, (12, tile.shape[0] - 16), tile.shape[0] / 720)
    return tile


def write_merge_sheet(tiles: list[np.ndarray], out: Path, cols: int = 4) -> None:
    """merge_sheet.png: merge tiles in rows of `cols`."""
    _write_sheet(tiles, out, cols)


def mask_tile(plate8: np.ndarray, mask: np.ndarray | None, canvas_rect, label: str, width: int = 640) -> np.ndarray:
    """The plate cropped to canvas_rect with the occluder mask over it in red (its 50 % edge in yellow);
    'no mask' when there is none."""
    img = _crop(plate8, canvas_rect)
    if mask is None:
        label += "  NO MASK"
    else:
        m = _crop(mask, canvas_rect)[..., None]
        img = np.clip(img * (1 - 0.5 * m) + np.array(RED, np.float32) * 0.5 * m, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours((m[..., 0] >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, YELLOW, max(1, round(img.shape[0] / 720)), cv2.LINE_AA)
    tile = cv2.resize(img, (width, round(width * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    _text(tile, label, (8, tile.shape[0] - 10), 0.7)
    return tile


def write_masks_sheet(tiles: list[np.ndarray], out: Path) -> None:
    """masks_sheet.png: mask tiles in rows of SHEET_COLS."""
    _write_sheet(tiles, out)


def compare_frame(plate8: np.ndarray, final8: np.ndarray, f: int) -> np.ndarray:
    """Plate and final side by side, each at half size, so the result is plate-wide and half as tall."""
    h, w = plate8.shape[:2]
    half = (w // 2, h // 2)
    left = cv2.resize(plate8, half, interpolation=cv2.INTER_AREA)
    right = cv2.resize(final8, half, interpolation=cv2.INTER_AREA)
    s = h / 2 / 1080
    _text(left, f"plate  frame {f}", (14, int(44 * s) + 10), max(0.5, 1.4 * s))
    _text(right, f"final  frame {f}", (14, int(44 * s) + 10), max(0.5, 1.4 * s))
    out = np.hstack([left, right])
    if out.shape[1] < w:  # odd plate width
        out = np.pad(out, ((0, 0), (0, w - out.shape[1]), (0, 0)))
    return out


def write_look_check(first_layout: np.ndarray, look_placed: np.ndarray, out: Path) -> None:
    """look_check.png: first layout frame, placed look, and a 50/50 blend, side by side."""
    blend = cv2.addWeighted(first_layout, 0.5, look_placed, 0.5, 0)
    tiles = [first_layout.copy(), look_placed.copy(), blend]
    for t, label in zip(tiles, ("first layout frame", "look_placed", "50/50 blend")):
        _text(t, label, (12, 34), 0.9)
    cv2.imwrite(str(out), np.hstack(tiles))
