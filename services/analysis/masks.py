"""Occluder masks for the behind ranges, from the hand landmarks the tracker saved (landmarks.npz).

`left_hand` / `right_hand` are the **person's** hands, matched to the cue's `hand`, never to MediaPipe's
handedness label:
- the chosen hand itself: the `chosen` chain (an orb behind its own hand);
- the other hand: the `other` chain where it has the frame. The other chain is strict (it exists for identity
  and drops tracklets with mixed labels), so elsewhere the mask uses every detected candidate that is not the
  chosen hand and whose palm centre is more than `near_chosen` x palm width from the chosen palm (`cand_lm`,
  `cand_is_chosen`). A mask needs no identity, only "the hand that is not the chosen one, in this frame".
Per frame the mask is the union of the convex hulls of those landmark sets, dilated by `dilate` x hand size and
feathered by a Gaussian of `feather` x hand size. Landmarks sit on the joints, not the skin, so the hull is
thinner than the hand, most of all side-on, where the palm width (5 to 17) collapses: hand size is
max(|lm5 - lm17|, |lm0 - lm9|), the median over the hand's last `size_frames` frames with landmarks. Short gaps are interpolated; longer ones get no
mask and a warning (degrade, don't fail). `person` and point occluders need SAM: not yet, they warn too.
Coordinates are plate pixels, pixel i covers [i, i + 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from services.cue.cue import Behind, Cue, TrackPoint
from services.analysis.track import PALM_IDS, missing_runs

Sets = list[np.ndarray]  # the landmark sets (21, 2) of one frame's occluder; empty = no mask


@dataclass(frozen=True)
class MaskConfig:
    dilate: float = 0.25       # x hand size
    feather: float = 0.06      # Gaussian sigma, x hand size
    size_frames: int = 5       # hand size is the median over this many of the hand's latest frames with landmarks
    max_gap: int = 6           # landmark gaps up to this many frames are interpolated
    near_chosen: float = 0.5   # a candidate whose palm is closer than this x palm_w to the chosen palm is ignored


@dataclass
class OccluderTrack:
    """One behind range and, for each of its frames, the occluder's landmark sets ((21, 2) each); [] = no mask."""
    index: int
    behind: Behind
    points: dict[int, Sets] = field(default_factory=dict)
    sizes: dict[int, float] = field(default_factory=dict)  # hand size per frame that has landmarks


def load_landmarks(path: Path) -> dict[str, np.ndarray]:
    """The arrays of landmarks.npz the masks use: `chosen`, `other` ((frames, 21, 3), NaN where missing) and,
    from trackers that save them, `cand_lm` ((frames, 2, 21, 3)) and `cand_is_chosen` ((frames, 2))."""
    with np.load(path) as z:
        return {k: z[k] for k in ("chosen", "other", "cand_lm", "cand_is_chosen") if k in z}


def occluder_chain(occluder, hand: str) -> str | None:
    """'chosen' or 'other': which of the person's hands `occluder` names, given the cue's `hand`.
    None for occluders that are not a hand."""
    if occluder == f"{hand}_hand":
        return "chosen"
    if occluder in ("left_hand", "right_hand"):
        return "other"
    return None


def fill_landmark_gaps(lms: np.ndarray, max_gap: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Landmarks (frames, 21, k) with missing runs of up to `max_gap` frames linearly interpolated, landmark by
    landmark, between the frames on either side. Returns (filled, the runs left missing [start, end))."""
    out = lms.copy()
    valid = ~np.isnan(lms).any(axis=(1, 2))
    left = []
    for a, b in missing_runs(valid):
        if a == 0 or b == len(lms) or b - a > max_gap:  # too long, or no frame on one side to interpolate from
            left.append((a, b))
            continue
        for f in range(a, b):
            t = (f - a + 1) / (b - a + 1)
            out[f] = (1 - t) * lms[a - 1] + t * lms[b]
    return out, left


def fill_set_gaps(sets: list[Sets], max_gap: int) -> tuple[list[Sets], list[tuple[int, int]]]:
    """Per-frame landmark sets with empty runs of up to `max_gap` frames interpolated, where the frames on
    either side have exactly one set each. Returns (filled, the runs left empty [start, end))."""
    out = [list(x) for x in sets]
    left = []
    for a, b in missing_runs(np.array([len(x) > 0 for x in sets])):
        if a == 0 or b == len(sets) or b - a > max_gap or len(sets[a - 1]) != 1 or len(sets[b]) != 1:
            left.append((a, b))
            continue
        for f in range(a, b):
            t = (f - a + 1) / (b - a + 1)
            out[f] = [(1 - t) * sets[a - 1][0] + t * sets[b][0]]
    return out, left


def _valid(lm: np.ndarray) -> bool:
    return not np.isnan(lm).any()


def chosen_sets(landmarks: dict[str, np.ndarray]) -> list[Sets]:
    """The chosen hand's landmarks, one set per frame where it was detected."""
    return [[lm[:, :2].astype(np.float64)] if _valid(lm) else [] for lm in landmarks["chosen"]]


def nonchosen_sets(landmarks: dict[str, np.ndarray], track: list[TrackPoint],
                   cfg: MaskConfig = MaskConfig()) -> list[Sets]:
    """Per frame: the other chain's landmarks if it has the frame, else every candidate that is not the chosen
    hand and whose palm centre is more than `near_chosen` x palm_w from the chosen palm (the cue's track)."""
    other = landmarks["other"]
    cand, is_chosen = landmarks.get("cand_lm"), landmarks.get("cand_is_chosen")
    out: list[Sets] = []
    for f in range(len(other)):
        if _valid(other[f]):
            out.append([other[f, :, :2].astype(np.float64)])
            continue
        sets = []
        if cand is not None and f < len(track):
            p = track[f]
            for j in range(cand.shape[1]):
                lm = cand[f, j]
                if is_chosen[f, j] or not _valid(lm):
                    continue
                palm = lm[list(PALM_IDS), :2].mean(axis=0)
                if np.hypot(palm[0] - p.x, palm[1] - p.y) > cfg.near_chosen * p.palm_w:
                    sets.append(lm[:, :2].astype(np.float64))
        out.append(sets)
    return out


def _frames(runs: list[tuple[int, int]]) -> str:
    return ", ".join(f"{a}..{b - 1}" if b - a > 1 else f"{a}" for a, b in runs)


def occluder_tracks(cue: Cue, landmarks: dict[str, np.ndarray] | None,
                    cfg: MaskConfig = MaskConfig()) -> tuple[list[OccluderTrack], list[str]]:
    """The occluder landmarks for every frame of every behind range, and warnings for frames without a mask."""
    tracks, warnings = [], []
    filled: dict[str, tuple[list[Sets], list[tuple[int, int]]]] = {}
    for i, b in enumerate(cue.behind):
        t = OccluderTrack(i, b, {f: [] for f in range(b.start, b.end)})
        tracks.append(t)
        where = f"behind[{i}] ({b.start}..{b.end - 1})"
        chain = occluder_chain(b.occluder, cue.hand)
        if chain is None:
            warnings.append(f"{where}: occluder {b.occluder!r} needs SAM, which is not built yet; no mask, "
                            f"the orb stays in front there")
            continue
        if landmarks is None:
            warnings.append(f"{where}: no landmarks file; no mask, the orb stays in front there")
            continue
        if chain not in filled:
            sets = chosen_sets(landmarks) if chain == "chosen" else nonchosen_sets(landmarks, cue.track, cfg)
            filled[chain] = fill_set_gaps(sets, cfg.max_gap)
        sets, gaps = filled[chain]
        sizes = smoothed_sizes(sets, cfg.size_frames)
        for f in t.points:
            if 0 <= f < len(sets) and sets[f]:
                t.points[f], t.sizes[f] = sets[f], sizes[f]
        missing = missing_runs(np.array([len(t.points[f]) > 0 for f in t.points]))
        if missing:
            runs = [(b.start + a, b.start + e) for a, e in missing]
            long = [g for g in gaps if any(g[0] < e and a < g[1] for a, e in runs)]
            why = (f" (missing {_frames(long)}: more than {cfg.max_gap} frames, or at the clip edge)" if long
                   else " (outside the landmarks file)")
            src = "chosen chain" if chain == "chosen" else "other chain or non-chosen candidates"
            warnings.append(f"{where}: no {b.occluder} landmarks ({src}) in frames {_frames(runs)}{why}; "
                            f"no mask there, the orb stays in front")
    return tracks, warnings


def palm_width(xy: np.ndarray) -> float:
    """Distance from landmark 5 (index knuckle) to 17 (pinky knuckle)."""
    return float(np.linalg.norm(xy[5] - xy[17]))


def hand_size(xy: np.ndarray) -> float:
    """max(|lm5 - lm17|, |lm0 - lm9|): the palm width, or the wrist to middle knuckle length when the hand is
    side-on and the width collapses."""
    return max(palm_width(xy), float(np.linalg.norm(xy[0] - xy[9])))


def smoothed_sizes(sets: list[Sets], n: int = 5) -> list[float | None]:
    """Per frame with landmarks: the median hand size over the last n frames (this one included) with landmarks,
    so one odd frame cannot shrink the mask. A frame's size is that of its largest set. None where empty."""
    out: list[float | None] = []
    recent: list[float] = []
    for x in sets:
        if not x:
            out.append(None)
            continue
        recent = (recent + [max(hand_size(xy) for xy in x)])[-n:]
        out.append(float(np.median(recent)))
    return out


def hull(xy: np.ndarray) -> np.ndarray:
    """Convex hull of the landmarks, (k, 2) float, pixel-edge coordinates."""
    return cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2).astype(np.float64)


def hand_mask(xy: np.ndarray, shape: tuple[int, int], cfg: MaskConfig = MaskConfig(),
              size: float | None = None) -> np.ndarray:
    """Float32 (h, w) mask in 0..1: the landmarks' convex hull, dilated and feathered by fractions of the hand size
    (`size`, or this frame's own `hand_size` when not given)."""
    h, w = shape
    out = np.zeros(shape, dtype=np.float32)
    hs = max(hand_size(xy) if size is None else size, 1.0)
    r, sigma = cfg.dilate * hs, cfg.feather * hs
    pad = r + 3 * sigma + 2
    poly = hull(xy)
    x0, y0 = max(0, int(np.floor(poly[:, 0].min() - pad))), max(0, int(np.floor(poly[:, 1].min() - pad)))
    x1, y1 = min(w, int(np.ceil(poly[:, 0].max() + pad))), min(h, int(np.ceil(poly[:, 1].max() + pad)))
    if x0 >= x1 or y0 >= y1:
        return out
    inside = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shift = 4  # sub-pixel vertices; fillPoly counts pixel centres at integers
    pts = np.rint((poly - (x0 + 0.5, y0 + 0.5)) * (1 << shift)).astype(np.int32)
    cv2.fillPoly(inside, [pts], 1, cv2.LINE_8, shift)
    dist = cv2.distanceTransform(1 - inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)  # px to the nearest hull pixel
    grown = np.clip(r + 0.5 - dist, 0.0, 1.0).astype(np.float32)
    if sigma > 0:
        grown = cv2.GaussianBlur(grown, (0, 0), sigma)
    out[y0:y1, x0:x1] = grown
    return out


def occluder_mask(sets: Sets, shape: tuple[int, int], cfg: MaskConfig = MaskConfig(),
                  size: float | None = None) -> np.ndarray | None:
    """Union (per-pixel max) of the hand masks of one frame's landmark sets; None if there are none."""
    if not sets:
        return None
    return np.maximum.reduce([hand_mask(xy, shape, cfg, size) for xy in sets])


def masks_for_frame(tracks: list[OccluderTrack], f: int, shape: tuple[int, int],
                    cfg: MaskConfig = MaskConfig()) -> list[np.ndarray]:
    """The occluder masks of frame f: one per behind range that covers f and has landmarks there."""
    return [occluder_mask(t.points[f], shape, cfg, t.sizes.get(f)) for t in tracks if t.points.get(f)]
