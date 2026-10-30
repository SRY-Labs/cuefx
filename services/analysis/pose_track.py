"""Body-point tracking (Analyze step, task 32): the effect follows a fist, foot, knee, shoulder or head.

MediaPipe Pose Landmarker runs once per clip (`get_pose_landmarks`, the only function that calls a pose model) and
all 33 landmarks of every frame are kept, so switching the body point never runs Pose again. A body point is then a
position and a size measure computed from those landmarks (`PART_SPECS`). The size measure is written into
`TrackPoint.palm_w`, so everything after tracking (gaps, One Euro smoothing, corrections, start timings, Make,
Merge) works on a pose track exactly as on a palm track: the gap/lost/smoothing code is the hand tracker's own.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from services.cue.cue import BodyPoint, HandSeed, TrackPoint
from services.analysis.track import REPO_ROOT, TrackConfig, TrackError, VideoInfo, smooth_and_flag

POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
POSE_MODEL_PATH = REPO_ROOT / "models" / "pose_landmarker_heavy.task"
POSE_SOURCE = "mediapipe_pose_heavy_f16"  # names the landmark source in the cache key; change it with the model
N_LANDMARKS = 33
MIN_CONF = 0.5     # a frame is normal when a pose was found and the mean visibility of its landmarks is at least this
LOW_VIS_MIN = 0.2  # from 0.2 up to MIN_CONF the raw position is used (source "tracked"); set from two clips (kick, punch)
JUMP_LIMIT_SIZES = 3.0  # a low-visibility frame farther than this many size measures per frame from the last accepted one is missing; two clips
EDGE_MARGIN_SIZES = 1.0  # a point this many size measures (the clip's median) past the plate edge still counts as inside it; two clips (punch fist at the bottom edge)
SEED_RADIUS = 3.0  # a seed click farther than this many size measures from the body point warns


@dataclass(frozen=True)
class PartSpec:
    """Landmarks (MediaPipe Pose indices, L/R = the person's left/right) of one body part.

    `points[side]` are averaged for the position; `size[side]` is the pair whose distance, times `size_scale`, is the
    size measure. `conf` is the mean visibility of every landmark used for either.
    """
    points: dict[str | None, tuple[int, ...]]
    size: dict[str | None, tuple[int, int]]
    size_scale: float = 1.0


PART_SPECS: dict[str, PartSpec] = {
    # fist: mean of wrist, index, pinky; size = wrist -> index
    "fist": PartSpec({"left": (15, 19, 17), "right": (16, 20, 18)}, {"left": (15, 19), "right": (16, 20)}),
    # foot: mean of heel and foot index; size = heel -> foot index
    "foot": PartSpec({"left": (29, 31), "right": (30, 32)}, {"left": (29, 31), "right": (30, 32)}),
    # knee: the knee; size = 0.4 x hip -> knee
    "knee": PartSpec({"left": (25,), "right": (26,)}, {"left": (23, 25), "right": (24, 26)}, 0.4),
    # shoulder: the shoulder; size = 0.35 x shoulder width (11 <-> 12)
    "shoulder": PartSpec({"left": (11,), "right": (12,)}, {"left": (11, 12), "right": (11, 12)}, 0.35),
    # head: midpoint of the ears; size = ear <-> ear distance
    "head": PartSpec({None: (7, 8)}, {None: (7, 8)}),
}
POSE_PARTS = tuple(PART_SPECS)  # the parts this module tracks; "palm" is the hand tracker's


@dataclass
class PoseLandmarks:
    """The Pose result of a clip, in MediaPipe's 33-point order: plate-pixel x, y and visibility per landmark.

    `xy` is NaN and `visibility` 0 in a frame without a pose. Any pose source can fill this (RTMPose, DWPose, ...)
    as long as it returns this shape in this landmark order.
    """
    info: VideoInfo
    xy: np.ndarray          # (frames, 33, 2)
    visibility: np.ndarray  # (frames, 33)

    @property
    def present(self) -> np.ndarray:
        """(frames,) True where a pose was found."""
        return ~np.isnan(self.xy[:, 0, 0])


@dataclass
class BodySeries:
    """One body point as per-frame arrays, before gaps are filled."""
    xy: np.ndarray     # (frames, 2), NaN where there is no pose
    size: np.ndarray   # (frames,) the size measure
    conf: np.ndarray   # (frames,) mean visibility of the landmarks used
    valid: np.ndarray  # (frames,) the position is used: a normal frame (conf >= MIN_CONF) or an accepted `low` one
    low: np.ndarray    # (frames,) valid with LOW_VIS_MIN <= conf < MIN_CONF: the raw position of a low-visibility landmark set


@dataclass
class PoseTrackResult:
    info: VideoInfo
    points: list[TrackPoint]   # smoothed
    raw_xy: np.ndarray         # (frames, 2) before smoothing (gaps filled)
    smoothed_xy: np.ndarray
    gaps: list[tuple[int, int]]
    lost: list[tuple[int, int]]
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- the pose source (swappable)

def ensure_pose_model(path: Path = POSE_MODEL_PATH) -> Path:
    """Download pose_landmarker_heavy.task (float16) if it is missing."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(POSE_MODEL_URL, tmp)
        tmp.replace(path)
    return path


def get_pose_landmarks(clip: Path, model: Path | None = None) -> PoseLandmarks:
    """Run the Pose Landmarker once over `clip` (VIDEO mode, one pose) and keep all 33 landmarks and their visibility.

    The one place a pose model is called: another model replaces this function and nothing else.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    model = ensure_pose_model() if model is None else model
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise TrackError(f"cannot open video: {clip}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    meta_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        raise TrackError(f"video reports fps={fps}")
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    xy: list[np.ndarray] = []
    vis: list[np.ndarray] = []
    height = width = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            height, width = bgr.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(image, round(len(xy) * 1000 / fps))
            if res.pose_landmarks:
                lms = res.pose_landmarks[0]
                xy.append(np.array([[p.x * width, p.y * height] for p in lms], dtype=np.float64))
                vis.append(np.array([p.visibility or 0.0 for p in lms], dtype=np.float64))
            else:
                xy.append(np.full((N_LANDMARKS, 2), np.nan))
                vis.append(np.zeros(N_LANDMARKS))
    cap.release()
    if not xy:
        raise TrackError(f"no frames decoded from {clip}")
    return PoseLandmarks(VideoInfo(width, height, fps, len(xy), meta_frames), np.stack(xy), np.stack(vis))


def save_pose_landmarks(pose: PoseLandmarks, path: Path) -> None:
    """Write the landmarks as one .npz file (through a .part file renamed at the end)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    i = pose.info
    part = path.with_name(path.stem + ".part.npz")
    np.savez_compressed(part, xy=pose.xy, visibility=pose.visibility,
                        info=np.array([i.width, i.height, i.fps, i.frames, i.meta_frames], dtype=np.float64))
    part.replace(path)


def load_pose_landmarks(path: Path) -> PoseLandmarks:
    with np.load(path) as z:
        w, h, fps, frames, meta = z["info"]
        return PoseLandmarks(VideoInfo(int(w), int(h), float(fps), int(frames), int(meta)), z["xy"], z["visibility"])


def cached_pose_landmarks(clip: Path, cache_file: Path | None, model: Path | None = None) -> PoseLandmarks:
    """`get_pose_landmarks`, read from `cache_file` if it is there and written there if not (None: no cache)."""
    if cache_file is not None and cache_file.is_file():
        try:
            return load_pose_landmarks(cache_file)
        except (OSError, ValueError, KeyError):
            pass  # unreadable cache: run Pose again
    pose = get_pose_landmarks(clip, model)
    if cache_file is not None:
        save_pose_landmarks(pose, cache_file)
    return pose


# --------------------------------------------------------------------------- body point from landmarks

def part_side(part: str, side: str | None) -> str | None:
    """The side a part is looked up by: the head has none, every other part needs "left" or "right"."""
    if part == "head":
        return None
    if part not in PART_SPECS or side not in ("left", "right"):
        raise ValueError(f"no body part {part!r} with side {side!r}")
    return side


def body_point_series(pose: PoseLandmarks, part: str, side: str | None) -> BodySeries:
    """Position (plate px), size measure and confidence of one body point in every frame (see `PART_SPECS`)."""
    side = part_side(part, side)
    spec = PART_SPECS[part]
    ids = list(spec.points[side])
    a, b = spec.size[side]
    xy = pose.xy[:, ids].mean(axis=1)
    size = spec.size_scale * np.linalg.norm(pose.xy[:, a] - pose.xy[:, b], axis=1)
    used = sorted({*ids, a, b})
    conf = pose.visibility[:, used].mean(axis=1)
    with np.errstate(invalid="ignore"):
        base = pose.present & (size > 0)
        inside = _inside_plate(pose, used, 0.0)
        solid = base & inside & (conf >= MIN_CONF)
        # one size per clip for both checks below: a wild landmark inflates the per-frame size measure and with it the limits
        ref = float(np.median(size[solid])) if solid.any() else float(np.median(size[base])) if base.any() else 0.0
        ok = base & _inside_plate(pose, used, EDGE_MARGIN_SIZES * ref)
        normal, weak = ok & (conf >= MIN_CONF), ok & (conf >= LOW_VIS_MIN) & (conf < MIN_CONF)
    low = _without_jumps(xy, ref, normal, weak)
    return BodySeries(xy, size, conf, normal | low, low)


def _inside_plate(pose: PoseLandmarks, ids: list[int], margin: float) -> np.ndarray:
    """(frames,) True where every landmark of `ids` lies on the plate or at most `margin` px outside it. One farther
    out is a guess by the model, however visible it claims to be, so the body point counts as missing there."""
    x, y = pose.xy[:, ids, 0], pose.xy[:, ids, 1]
    return ((x >= -margin) & (x < pose.info.width + margin) & (y >= -margin) & (y < pose.info.height + margin)).all(axis=1)


def _without_jumps(xy: np.ndarray, ref: float, normal: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """(frames,) the `weak` frames kept: one is dropped when its position is more than JUMP_LIMIT_SIZES x `ref` (the
    clip's size measure) x (frames since the last accepted frame) away from that frame's position, so a weak guess
    cannot jump to another limb. The first accepted frame of the clip has nothing to be compared with and is kept."""
    keep = np.zeros(len(weak), dtype=bool)
    last = None
    for f in range(len(weak)):
        if weak[f]:
            keep[f] = last is None or float(np.linalg.norm(xy[f] - xy[last])) <= JUMP_LIMIT_SIZES * ref * (f - last)
        if normal[f] or keep[f]:
            last = f
    return keep


def side_from_click(pose: PoseLandmarks, part: str, seed: HandSeed) -> str | None:
    """The person's side ("left" or "right") of `part` whose point is nearer the artist's click in the seed frame, or
    None when that cannot be told: the part has no side (the head), or one of the two sides is missing in the seed
    frame (no pose, low visibility, off the plate), or the frame is outside the clip. An exact tie is "left"."""
    if part == "head" or not 0 <= seed.f < pose.info.frames:
        return None
    dist = {}
    for side in ("left", "right"):
        s = body_point_series(pose, part, side)
        if not s.valid[seed.f]:
            return None
        dist[side] = float(np.linalg.norm(s.xy[seed.f] - np.array([seed.x, seed.y])))
    return "left" if dist["left"] <= dist["right"] else "right"


def build_pose_track(pose: PoseLandmarks, bp: BodyPoint, seed: HandSeed | None, cfg: TrackConfig = TrackConfig()
                     ) -> PoseTrackResult:
    """The smoothed per-frame track of a body point, with gaps filled and flagged by the hand tracker's own rules.

    Raises TrackError: a supplied seed frame is outside the clip (input_invalid), or the body point is missing in
    every frame (track_lost). A supplied click far from the body point is a warning, not an error. With no seed,
    the explicitly requested side is tracked and no click warning is made.
    """
    n = pose.info.frames
    if seed is not None and not 0 <= seed.f < n:
        raise TrackError(f"seed frame {seed.f} is outside the clip (0..{n - 1})", "input_invalid")
    s = body_point_series(pose, bp.part, bp.side)
    if not s.valid.any():
        raise TrackError(f"the {bp.part} was not found in any frame", "track_lost")
    xy, size = np.where(s.valid[:, None], s.xy, np.nan), np.where(s.valid, s.size, np.nan)
    conf = np.where(s.valid, s.conf, 0.0)
    points, raw_xy, _, smoothed, gaps, lost = smooth_and_flag(pose.info.fps, xy, size, conf, s.valid, [], cfg)
    return PoseTrackResult(pose.info, points, raw_xy, smoothed, gaps, lost,
                           [] if seed is None else seed_warnings(s, bp, seed))


def seed_warnings(s: BodySeries, bp: BodyPoint, seed: HandSeed) -> list[str]:
    """`seed_far_from_body_point` detail text if no pose is within SEED_RADIUS size measures of the click, else []."""
    name = bp.part if bp.side is None or bp.part == "head" else f"{bp.side} {bp.part}"
    if not s.valid[seed.f]:
        return [f"hand_seed: no pose with a visible {name} in frame {seed.f}; the click cannot be checked. "
                f"Click on the {name} in a frame where it is visible."]
    dist = float(np.linalg.norm(s.xy[seed.f] - np.array([seed.x, seed.y])))
    if dist > SEED_RADIUS * s.size[seed.f]:
        c = s.xy[seed.f]
        return [f"hand_seed: ({seed.x:g}, {seed.y:g}) is {dist:.0f} px from the {name} in frame {seed.f}, more than "
                f"{SEED_RADIUS:g} x its size ({s.size[seed.f]:.0f} px); the {name} is at ({c[0]:.0f}, {c[1]:.0f}). "
                f"Click on the {name} you want to track."]
    return []
