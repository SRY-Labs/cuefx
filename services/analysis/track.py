"""Hand tracking (Analyze step): detect, pick the artist's hand, fill gaps, smooth, correct.

Pass 1 (`detect_hands`) runs MediaPipe once, forward only, and stores every candidate.
Pass 2 (`select_hands`) works offline on those candidates and keeps two identities (the chosen
hand and the other hand) so the two hands cannot swap when they overlap.
All thresholds are in palm widths; nothing here depends on a particular clip.
"""
from __future__ import annotations

import itertools
import math
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from services.cue.cue import Correction, Cue, HandSeed, TrackPoint, lost_runs, save_cue

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = REPO_ROOT / "models" / "hand_landmarker.task"
PALM_IDS = (0, 5, 9, 13, 17)
MAX_HANDS = 2  # the Hand Landmarker's num_hands


class TrackError(RuntimeError):
    """Output is impossible (unreadable clip, seed frame outside the clip or without any hand); the CLI prints it and stops.
    `code` is the contract error code (contract/schemas/error_codes.json)."""

    def __init__(self, message: str, code: str = "clip_unreadable") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class OneEuroConfig:
    """One Euro filter. Positions are filtered in palm widths, so beta is in palm widths/s."""
    min_cutoff: float = 1.0  # Hz; smoothing when the hand is still
    beta: float = 1.0        # cutoff gained per (palm width / s) of speed
    d_cutoff: float = 1.0    # Hz; smoothing of the speed estimate


@dataclass(frozen=True)
class TrackConfig:
    seed_radius: float = 2.0       # a seed click farther than this many palm widths from every palm warns
    # tracklets
    link_dist: float = 1.5         # link frames if closer than this many palm widths ...
    link_ratio: tuple[float, float] = (0.7, 1.4)  # ... and the palm-width ratio is inside this range
    merge_dist: float = 1.0        # two candidates closer than this many palm widths are a merge
    # stitching
    max_step: float = 1.5          # allowed jump, palm widths per frame of gap
    max_step_frames: int = 6       # cap on the gap multiplier; also how long a velocity is held
    vel_frames: int = 5            # velocity is measured over this many last frames
    min_len: int = 3               # shorter tracklets are weak: they need a fitting palm width
    label_cost: float = 3.0
    width_cost: float = 2.0
    reacquire_match: float = 0.8   # label agreement needed to jump a gap the gate rejects
    # hard gate for joining a chain (the costs below only choose among tracklets that pass). There is no gate on
    # palm width: the palm legitimately shrinks 3x across a blurred throw; width only enters through width_cost.
    vote_gate: float = 0.5         # a tracklet whose "Right" vote differs from the chain's by more than this ...
    vote_gate_len: int = 8         # ... is rejected if it is shorter than this many frames
    ambiguous_near: float = 1.5    # chosen hand missing: a candidate this close to it is ambiguous
    max_gap: int = 6               # a run of more missing frames than this is reported as lost
    one_euro: OneEuroConfig = field(default_factory=OneEuroConfig)
    correction_falloff: int = 12   # frames over which a correction fades out beyond its keyframes


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int           # frames actually decoded
    meta_frames: int      # frame count the container claims


@dataclass(frozen=True)
class Hand:
    """One detected hand in one frame."""
    xyz: np.ndarray  # (21, 3): x, y in pixels, z as MediaPipe gives it
    label: str       # MediaPipe handedness; assumes a mirrored image, so only a tie-breaker
    score: float

    @property
    def palm_center(self) -> np.ndarray:
        return self.xyz[list(PALM_IDS), :2].mean(axis=0)

    @property
    def palm_width(self) -> float:
        return float(np.linalg.norm(self.xyz[5, :2] - self.xyz[17, :2]))


@dataclass
class Tracklet:
    """A short chain of candidates in consecutive frames that is very likely one hand."""
    id: int
    frames: list[int]
    cands: list[int]      # index into each frame's candidate list
    pos: np.ndarray       # (n, 2) palm centres
    widths: np.ndarray    # (n,) palm widths
    w_right: float        # sum of handedness scores voting "Right"
    w_all: float          # sum of all handedness scores
    chain: str = "dropped"  # "chosen" | "other" | "dropped"

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def palm_w(self) -> float:
        return float(np.median(self.widths))

    @property
    def vote(self) -> float:
        """Score-weighted share of "Right" labels, 0..1."""
        return self.w_right / self.w_all if self.w_all > 0 else 0.5


@dataclass
class Selection:
    chosen: list[int | None]       # per frame: index into that frame's candidates, or None
    other: list[int | None]
    tracklets: list[Tracklet]
    cand_tracklet: list[list[int]]  # per frame, per candidate: tracklet id
    reference_vote: float          # "Right" vote of the seed tracklet (measured on the clip)
    reacquired: list[tuple[int, int]]  # frame ranges [start, end) jumped by reacquisition (always lost)
    warnings: list[str] = field(default_factory=list)  # uncertainty the artist can fix (e.g. a seed that missed)

    @property
    def reference_label(self) -> str:
        return "Right" if self.reference_vote >= 0.5 else "Left"


@dataclass
class TrackResult:
    info: VideoInfo
    points: list[TrackPoint]      # smoothed
    raw_xy: np.ndarray            # (frames, 2) before smoothing (gaps filled)
    smoothed_xy: np.ndarray
    ambiguous: np.ndarray
    chosen: np.ndarray            # (frames, 21, 3), NaN where missing
    other: np.ndarray
    gaps: list[tuple[int, int]]   # every run of missing frames, [start, end)
    lost: list[tuple[int, int]]   # the runs longer than max_gap
    cand_lm: np.ndarray | None = None         # (frames, 2, 21, 3): every candidate, NaN where empty (for masks)
    cand_is_chosen: np.ndarray | None = None  # (frames, 2): that candidate is the chosen hand


# --------------------------------------------------------------------------- pass 1: detection

def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download hand_landmarker.task (float16) if it is missing."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    return path


def detect_hands(video: Path, model: Path) -> tuple[VideoInfo, list[list[Hand]]]:
    """Run the Hand Landmarker once, forward only (VIDEO mode), and keep every candidate."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise TrackError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    meta_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        raise TrackError(f"video reports fps={fps}")
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO, num_hands=MAX_HANDS)
    hands: list[list[Hand]] = []
    height = width = 0
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            height, width = bgr.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(image, round(len(hands) * 1000 / fps))
            frame_hands = []
            for lms, cats in zip(res.hand_landmarks, res.handedness):
                xyz = np.array([[p.x * width, p.y * height, p.z] for p in lms], dtype=np.float64)
                frame_hands.append(Hand(xyz, cats[0].category_name, float(cats[0].score)))
            hands.append(frame_hands)
    cap.release()
    if not hands:
        raise TrackError(f"no frames decoded from {video}")
    return VideoInfo(width, height, fps, len(hands), meta_frames), hands


# --------------------------------------------------------------------------- pass 2: selection

def _merged(cands: list[Hand], cfg: TrackConfig) -> set[int]:
    """Candidates of a frame that sit closer than merge_dist palm widths to another candidate."""
    out: set[int] = set()
    for i, j in itertools.combinations(range(len(cands)), 2):
        gap = float(np.linalg.norm(cands[i].palm_center - cands[j].palm_center))
        if gap < cfg.merge_dist * max(cands[i].palm_width, cands[j].palm_width):
            out |= {i, j}
    return out


def _best_links(prev: list[Hand], cands: list[Hand], free: list[int], cfg: TrackConfig) -> dict[int, int]:
    """Best one-to-one links (most links, then least distance) from prev hands to free candidates."""
    lo, hi = cfg.link_ratio
    valid: dict[tuple[int, int], float] = {}
    for a, p in enumerate(prev):
        for j in free:
            d = float(np.linalg.norm(cands[j].palm_center - p.palm_center))
            ratio = cands[j].palm_width / p.palm_width if p.palm_width > 0 else 0.0
            if d < cfg.link_dist * p.palm_width and lo <= ratio <= hi:
                valid[(a, j)] = d
    best_score, best_links = (0, 0.0), {}
    for pick in itertools.product([None, *free], repeat=len(prev)):
        used = [j for j in pick if j is not None]
        if len(used) != len(set(used)) or any((a, j) not in valid for a, j in enumerate(pick) if j is not None):
            continue
        score = (len(used), -sum(valid[(a, j)] for a, j in enumerate(pick) if j is not None))
        if score > best_score:
            best_score, best_links = score, {a: j for a, j in enumerate(pick) if j is not None}
    return best_links


def build_tracklets(hands: list[list[Hand]], cfg: TrackConfig = TrackConfig()) -> tuple[list[Tracklet], list[list[int]]]:
    """Link candidates of consecutive frames into tracklets; a merge ends them (merged candidates stand alone)."""
    tracklets: list[Tracklet] = []
    cand_tracklet: list[list[int]] = [[-1] * len(h) for h in hands]

    def start(f: int, j: int) -> Tracklet:
        h = hands[f][j]
        right = h.score if h.label == "Right" else 0.0
        t = Tracklet(len(tracklets), [f], [j], h.palm_center[None], np.array([h.palm_width]), right, h.score)
        tracklets.append(t)
        cand_tracklet[f][j] = t.id
        return t

    def extend(t: Tracklet, f: int, j: int) -> None:
        h = hands[f][j]
        t.frames.append(f)
        t.cands.append(j)
        t.pos = np.vstack([t.pos, h.palm_center])
        t.widths = np.append(t.widths, h.palm_width)
        t.w_right += h.score if h.label == "Right" else 0.0
        t.w_all += h.score
        cand_tracklet[f][j] = t.id

    open_: list[Tracklet] = []
    for f, cands in enumerate(hands):
        merged = _merged(cands, cfg)
        free = [j for j in range(len(cands)) if j not in merged]
        prev = [hands[f - 1][t.cands[-1]] for t in open_]
        links = _best_links(prev, cands, free, cfg)
        now = [open_[a] for a in links]
        for a, j in links.items():
            extend(open_[a], f, j)
        now += [start(f, j) for j in free if j not in links.values()]
        for j in sorted(merged):
            start(f, j)  # stands alone: not open for linking in either direction
        open_ = now
    return tracklets, cand_tracklet


class _Chain:
    """The tracklets assigned to one hand. Never holds two tracklets that overlap in time."""

    def __init__(self, tls: list[Tracklet]) -> None:
        self.tls = list(tls)

    def overlaps(self, t: Tracklet) -> bool:
        return any(t.start <= a.end and a.start <= t.end for a in self.tls)

    def end(self, step: int) -> int:
        """Last frame of the chain going forward, first frame going backward."""
        return max(a.end for a in self.tls) if step > 0 else min(a.start for a in self.tls)

    @property
    def vote(self) -> float:
        return sum(t.w_right for t in self.tls) / max(sum(t.w_all for t in self.tls), 1e-9)

    @property
    def palm_w(self) -> float:
        return float(np.median(np.concatenate([t.widths for t in self.tls])))

    def extrapolate(self, step: int, gap: int, cfg: TrackConfig) -> np.ndarray:
        """Where the chain end should be `gap` frames on: last velocity, held at most max_step_frames."""
        pts = dict(sorted((f, p) for t in self.tls for f, p in zip(t.frames, t.pos)))
        e = self.end(step)
        window = [f for f in pts if abs(f - e) < cfg.vel_frames]
        vel = np.zeros(2)
        if len(window) >= 2:
            vel = (pts[max(window)] - pts[min(window)]) / (max(window) - min(window))
        return pts[e] + vel * step * min(gap, cfg.max_step_frames)


def _fit(chain: _Chain, t: Tracklet, step: int, cfg: TrackConfig) -> tuple[float, bool] | None:
    """(cost, passes the distance gate) of adding t to chain, or None if a hard rule (label gate, overlap, short-tracklet width) forbids it."""
    if not chain.tls or chain.overlaps(t):
        return None
    if abs(t.vote - chain.vote) > cfg.vote_gate and t.n < cfg.vote_gate_len:
        return None  # hard gate: a short tracklet with the other label; long ones outvote single wrong labels
    pw = chain.palm_w
    ratio = t.palm_w / pw
    if t.n < cfg.min_len and not cfg.link_ratio[0] <= ratio <= cfg.link_ratio[1]:
        return None  # a short tracklet with the wrong palm width is a merged blob or a false detection
    entry_f, entry = (t.start, t.pos[0]) if step > 0 else (t.end, t.pos[-1])
    gap = abs(entry_f - chain.end(step))
    d = float(np.linalg.norm(chain.extrapolate(step, gap, cfg) - entry))
    cost = d / pw + cfg.label_cost * abs(t.vote - chain.vote) + cfg.width_cost * abs(math.log(ratio))
    return cost, d <= cfg.max_step * pw * min(gap, cfg.max_step_frames)


def _stitch(tracklets: list[Tracklet], seed_tl: Tracklet, cfg: TrackConfig) -> list[tuple[int, int]]:
    """Grow the chosen and other chains outwards from the seed tracklet; returns the reacquired gaps."""
    chosen, other = _Chain([seed_tl]), _Chain([])
    chains = {"chosen": chosen, "other": other}
    seed_tl.chain = "chosen"
    reacquired: list[tuple[int, int]] = []
    for step in (1, -1):
        pending = [t for t in tracklets if t is not seed_tl and ((t.start >= seed_tl.start) == (step > 0))]
        while pending:
            front = min(t.start for t in pending) if step > 0 else max(t.end for t in pending)
            tied = [t for t in pending if (t.start if step > 0 else t.end) == front]
            # simultaneous tracklets: the best fit for the chosen hand goes first, the rest overlap it
            t = min(tied, key=lambda x: (_fit(chosen, x, step, cfg) or (math.inf, False))[0])
            pending.remove(t)
            if chosen.overlaps(t) and not other.tls and t.n >= cfg.min_len:
                other.tls.append(t)  # simultaneous with the chosen hand, so it is the other hand
                t.chain = "other"
                continue
            fits = {name: _fit(chain, t, step, cfg) for name, chain in chains.items()}
            take = {name: f[0] for name, f in fits.items() if f and f[1]}
            jump = [name for name, f in fits.items() if f and not f[1] and t.n >= cfg.min_len
                    and abs(t.vote - chains[name].vote) <= 1 - cfg.reacquire_match]
            if not take and len(jump) == 1:  # the gate says no, but the label agrees and nobody else can
                take[jump[0]] = fits[jump[0]][0]
                e = chains[jump[0]].end(step)
                reacquired.append((e + 1, t.start) if step > 0 else (t.end + 1, e))
            if take:
                name = min(take, key=take.get)
                chains[name].tls.append(t)
                t.chain = name
    return reacquired


def select_hands(hands: list[list[Hand]], seed: HandSeed, cfg: TrackConfig = TrackConfig()) -> Selection:
    """Pick the seeded hand: tracklets first, then stitch them into a chosen and an other chain."""
    n = len(hands)
    if not 0 <= seed.f < n:
        raise TrackError(f"seed frame {seed.f} is outside the clip (0..{n - 1})", "input_invalid")
    cands = hands[seed.f]
    if not cands:
        raise TrackError(f"no hand detected in seed frame {seed.f}", "no_hand")
    click = np.array([seed.x, seed.y])
    k = min(range(len(cands)), key=lambda i: np.linalg.norm(cands[i].palm_center - click))
    dist, width = float(np.linalg.norm(cands[k].palm_center - click)), cands[k].palm_width
    warnings = []
    if dist > cfg.seed_radius * width:  # degrade: the nearest hand anywhere in the frame is used
        c = cands[k].palm_center
        warnings.append(f"hand_seed: ({seed.x:g}, {seed.y:g}) is {dist:.0f} px from the nearest palm in frame {seed.f}, "
                        f"more than {cfg.seed_radius:g} x its palm width ({width:.0f} px); using that hand at "
                        f"({c[0]:.0f}, {c[1]:.0f}). Click on the hand you want to track.")
    tracklets, cand_tracklet = build_tracklets(hands, cfg)
    seed_tl = tracklets[cand_tracklet[seed.f][k]]
    reacquired = _stitch(tracklets, seed_tl, cfg)
    chosen: list[int | None] = [None] * n
    other: list[int | None] = [None] * n
    for t in tracklets:
        for f, j in zip(t.frames, t.cands):
            if t.chain == "chosen":
                chosen[f] = j
            elif t.chain == "other":
                other[f] = j
    return Selection(chosen, other, tracklets, cand_tracklet, seed_tl.vote, sorted(reacquired), warnings)


# --------------------------------------------------------------------------- gaps, smoothing

def missing_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """Runs of False in `valid` as [start, end) frame ranges."""
    runs, start = [], None
    for f, v in enumerate(valid):
        if not v and start is None:
            start = f
        if v and start is not None:
            runs.append((start, f))
            start = None
    if start is not None:
        runs.append((start, len(valid)))
    return runs


def fill_gaps(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linear interpolation over invalid frames; hold the nearest value before the first / after the last."""
    frames = np.arange(len(valid))
    cols = np.atleast_2d(values.T).T if values.ndim == 1 else values
    out = np.stack([np.interp(frames, frames[valid], cols[valid, j]) for j in range(cols.shape[1])], axis=1)
    return out[:, 0] if values.ndim == 1 else out


def one_euro(xy: np.ndarray, fps: float, scale: float, cfg: OneEuroConfig) -> np.ndarray:
    """One Euro filter on (frames, 2) positions, forward in time; `scale` (a palm width) makes it size independent."""
    dt = 1.0 / fps
    alpha = lambda cutoff: 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff) / dt)  # noqa: E731
    p = xy / scale
    out = np.empty_like(p)
    out[0], d_hat = p[0], np.zeros(p.shape[1])
    for i in range(1, len(p)):
        d_hat = alpha(cfg.d_cutoff) * (p[i] - out[i - 1]) / dt + (1 - alpha(cfg.d_cutoff)) * d_hat
        a = alpha(cfg.min_cutoff + cfg.beta * np.abs(d_hat))
        out[i] = a * p[i] + (1 - a) * out[i - 1]
    return out * scale


def smooth_track(xy: np.ndarray, tracked: np.ndarray, fps: float, scale: float, cfg: OneEuroConfig) -> np.ndarray:
    """Zero-lag smoothing, offline: One Euro forward and backward, averaged.

    Every run of consecutive tracked frames is smoothed on its own, so nothing is smoothed across a gap.
    The frames in between stay linear (between the smoothed ends of the neighbouring runs).
    """
    out = xy.copy()
    for a, b in missing_runs(~tracked):
        if b - a > 1:
            fwd = one_euro(xy[a:b], fps, scale, cfg)
            bwd = one_euro(xy[a:b][::-1], fps, scale, cfg)[::-1]
            out[a:b] = (fwd + bwd) / 2
    return fill_gaps(out, tracked)


def ambiguity(hands: list[list[Hand]], chosen: list[int | None], xy: np.ndarray, pw: np.ndarray,
              cfg: TrackConfig) -> np.ndarray:
    """Frames where the chosen hand may be confused with another: a merge, or an occluded hand with a neighbour."""
    flags = np.zeros(len(hands), dtype=bool)
    for f, cands in enumerate(hands):
        flags[f] = bool(_merged(cands, cfg))
        if chosen[f] is None:  # the interpolated position is a guess: any candidate near it could be the hand
            flags[f] |= any(np.linalg.norm(c.palm_center - xy[f]) < cfg.ambiguous_near * pw[f] for c in cands)
    return flags


def smooth_and_flag(fps: float, xy: np.ndarray, pw: np.ndarray, conf: np.ndarray, valid: np.ndarray,
                    reacquired: list[tuple[int, int]], cfg: TrackConfig = TrackConfig()
                    ) -> tuple[list[TrackPoint], np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    """Gaps filled, One Euro smoothing and lost flags for per-frame positions `xy` and sizes `pw` (NaN where `valid` is False).

    Shared by the hand tracker and the pose tracker. Returns (points, raw_xy, pw filled, smoothed_xy, gaps, lost).
    """
    n = len(valid)
    raw_xy, pw = fill_gaps(xy, valid), fill_gaps(pw, valid)
    smoothed = smooth_track(xy, valid, fps, float(np.median(pw)), cfg.one_euro)
    gaps = missing_runs(valid)
    # a gap is lost if it is long, or if the hand was only found again by jumping across it
    lost = [(a, b) for a, b in gaps
            if b - a > cfg.max_gap or any(a < rb and ra < b for ra, rb in reacquired)]
    in_lost = np.zeros(n, dtype=bool)
    for a, b in lost:
        in_lost[a:b] = True
    points = [TrackPoint(f, float(smoothed[f, 0]), float(smoothed[f, 1]),
                         "tracked" if valid[f] else "interpolated", float(pw[f]), float(conf[f]), bool(in_lost[f]))
              for f in range(n)]
    return points, raw_xy, pw, smoothed, gaps, lost


def build_track(info: VideoInfo, hands: list[list[Hand]], sel: Selection,
                cfg: TrackConfig = TrackConfig()) -> TrackResult:
    """Turn a selection into the smoothed per-frame track, with gaps filled and flagged."""
    n = info.frames
    valid = np.array([c is not None for c in sel.chosen])
    xy, pw, conf = np.full((n, 2), np.nan), np.full(n, np.nan), np.zeros(n)
    chosen, other = np.full((n, 21, 3), np.nan), np.full((n, 21, 3), np.nan)
    cand_lm, cand_is_chosen = np.full((n, MAX_HANDS, 21, 3), np.nan), np.zeros((n, MAX_HANDS), dtype=bool)
    for f in range(n):
        for j, h in enumerate(hands[f][:MAX_HANDS]):
            cand_lm[f, j] = h.xyz
            cand_is_chosen[f, j] = sel.chosen[f] == j
        if sel.chosen[f] is not None:
            h = hands[f][sel.chosen[f]]
            xy[f], pw[f], conf[f] = h.palm_center, h.palm_width, h.score  # MediaPipe gives no per-hand
            chosen[f] = h.xyz                                             # presence score; score = handedness
        if sel.other[f] is not None:
            other[f] = hands[f][sel.other[f]].xyz
    points, raw_xy, pw, smoothed, gaps, lost = smooth_and_flag(info.fps, xy, pw, conf, valid, sel.reacquired, cfg)
    return TrackResult(info, points, raw_xy, smoothed, ambiguity(hands, sel.chosen, raw_xy, pw, cfg),
                       chosen, other, gaps, lost, cand_lm, cand_is_chosen)


def _offsets(track: list[TrackPoint], keys: list[Correction], lo: int, hi: int, k: int) -> dict[int, tuple[float, float]]:
    """Offsets for frames [lo, hi) from tracked-frame keyframes `keys` (all inside [lo, hi), sorted).

    Linear between keyframes; before the first and after the last it fades to 0 over k frames,
    but never beyond lo / hi (the edges of the clip or of a lost gap).
    """
    frames = [c.f for c in keys]
    dx = [c.x - track[c.f].x for c in keys]
    dy = [c.y - track[c.f].y for c in keys]
    out = {}
    for f in range(max(lo, frames[0] - k), min(hi, frames[-1] + k + 1)):
        away = frames[0] - f if f < frames[0] else f - frames[-1] if f > frames[-1] else 0
        fade = 1.0 if away == 0 else max(0.0, 1.0 - away / k) if k > 0 else 0.0
        out[f] = (fade * float(np.interp(f, frames, dx)), fade * float(np.interp(f, frames, dy)))
    return out


def apply_corrections(track: list[TrackPoint], corrections: list[Correction],
                      cfg: TrackConfig = TrackConfig()) -> list[TrackPoint]:
    """Artist keyframes, applied by where they land.

    A correction on a **tracked** frame shifts the track by an offset that is linear between such keyframes.
    Before the first and after the last keyframe the offset fades linearly to 0 over `cfg.correction_falloff`
    frames, so one correction moves a short window around it. The window stops at the edge of any lost gap
    (`TrackPoint.lost`): a gap is never shifted by, and nothing beyond it is reached from, a correction on
    the other side. Only frames whose offset is non-zero are marked `corrected`.

    A correction on a frame **inside a lost gap** is an extra anchor for that gap only: the gap is
    interpolated again, linearly, through its anchors and its two (corrected) neighbours, and all its
    frames are marked `corrected`. No falloff. A keyframe lands exactly on (x, y).
    """
    if not corrections:
        return list(track)
    keys = sorted(corrections, key=lambda c: c.f)
    if len({c.f for c in keys}) != len(keys):
        raise ValueError("two corrections for the same frame")
    n = len(track)
    out = list(track)
    gaps = lost_runs(track)
    edges = [0] + [e for g in gaps for e in g] + [n]  # tracked stretches are [edges[0], edges[1]), [edges[2], edges[3]), ...
    stretches = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2) if edges[i] < edges[i + 1]]
    for lo, hi in stretches:
        mine = [c for c in keys if lo <= c.f < hi]
        if not mine:
            continue
        for f, (ox, oy) in _offsets(track, mine, lo, hi, cfg.correction_falloff).items():
            if ox != 0 or oy != 0:
                out[f] = replace(track[f], x=track[f].x + ox, y=track[f].y + oy, source="corrected")
    for a, b in gaps:
        anchors = [c for c in keys if a <= c.f < b]
        if not anchors:
            continue
        pts = [(c.f, c.x, c.y) for c in anchors]
        if a > 0:
            pts.insert(0, (a - 1, out[a - 1].x, out[a - 1].y))
        if b < n:
            pts.append((b, out[b].x, out[b].y))
        fs, xs, ys = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
        for f in range(a, b):
            out[f] = replace(track[f], x=float(np.interp(f, fs, xs)), y=float(np.interp(f, fs, ys)), source="corrected")
    return out


# --------------------------------------------------------------------------- top level + saving

def track_video(video: Path, hand: str, seed: HandSeed, cfg: TrackConfig = TrackConfig(),
                model: Path | None = None) -> tuple[TrackResult, list[list[Hand]], Selection]:
    """Detect, select and build the track for one clip."""
    info, hands = detect_hands(video, ensure_model() if model is None else model)
    sel = select_hands(hands, seed, cfg)
    return build_track(info, hands, sel, cfg), hands, sel


def save_track(result: TrackResult, video: Path, hand: str, seed: HandSeed, out_dir: Path) -> None:
    """Write track.json (a cue file with the track filled in) and landmarks.npz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = result.info
    cue = Cue(clip=video.name, fps=info.fps, frames=info.frames, plate_size=(info.width, info.height),
              hand=hand, hand_seed=seed, track=result.points)
    save_cue(cue, out_dir / "track.json")
    save_landmarks(result, out_dir / "landmarks.npz")


def save_landmarks(result: TrackResult, path: Path) -> None:
    """Write the hand landmarks Merge's occluder masks read (`services.analysis.masks.load_landmarks`) as one .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, chosen=result.chosen, other=result.other,
                        raw_xy=result.raw_xy, smoothed_xy=result.smoothed_xy, ambiguous=result.ambiguous,
                        cand_lm=result.cand_lm, cand_is_chosen=result.cand_is_chosen)
