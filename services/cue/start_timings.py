"""Starting timings for a new cue, from the clip's own track (task 28, rule A; task 32 for body points).

A pure function of the track: where the palm holds still, the orb builds up and is held; where the hand starts to
move fast, it is released. A clip with a steady run but no throw after it starts as a "hold": the effect stays on the
hand to the end of the run (task 33). The artist drags everything afterwards, so this is only a starting point.
A body-point track (fist, foot, ...) is read the same way: its speeds are relative to its size measure (`palm_w`), so
a kick or a punch is the throw. Only the default size and offset depend on the part (`PART_SIZING`).

Task 36: a fist, foot or knee is steady only while the person stands, so for those parts the strike rule is tried first
(`strike_release`): a fist's fastest frame, if it is faster than 1.0 size measures per frame, is the release; a foot or
knee (bigger size measure, slower swing, and a swing out and back) takes the peak of the first run of at least 3 frames
above 0.4. The cue then starts with a short flight along the movement, kept inside the picture. Palm, shoulder and head
never use it.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from services.cue.cue import MAX_EFFECT_FRAMES, Entrance, TrackPoint, seconds_to_frames
from services.cue.window import LEAD_IN, _longest, propose_effect_range

STEADY_SPEED = 0.06     # palm speed (palm widths per frame) below which a tracked frame is steady
MAX_GAP = 2             # non-steady frames allowed inside one steady run
THROW_SPEED = 1.0       # palm speed (palm widths per frame) that marks the throw; a wind-up stays well below it
THROW_SEARCH_S = 1.5    # seconds after the steady run in which the throw is looked for
MIN_HOLD_EXTRA = 8      # a steady run must be at least N_in + this many frames, else the fallback is used
DIAMETER_PER_PALM = 1.15
ANCHOR_UP_PER_DIAMETER = 0.65
VANISH_START_S = 0.25   # release -> vanish.start
VANISH_LEN_S = 0.25     # vanish.start -> vanish.end
FALLBACK_APPEAR = 0.25  # fractions of the clip when there is no steady run
FALLBACK_RELEASE = 0.60
FALLBACK_DIAMETER = 110.0  # when the track has no palm width at all (the golden cue's)
STRIKE_PARTS = ("fist", "foot", "knee")  # the parts the strike rule is tried on (task 36)
STRIKE_SPEED = {"fist": 1.0, "foot": 0.4, "knee": 0.4}  # size measures per frame (smoothed over STRIKE_SMOOTH frames) a strike beats
STRIKE_MIN_RUN = 3      # foot and knee: the first run of frames above their threshold must last this long (a fist: its fastest frame)
STRIKE_MARGIN_D = 0.9   # the default route's end stays this x diameter inside the picture (1.8 x the radius: a burst fits)
STRIKE_SMOOTH = 3       # frames the speed is averaged over (centred)
STRIKE_LEAD_S = 1.0     # appear_frame = release - this
STRIKE_ROUTE_S = 0.5    # the default route's second point is this long after the release ...
STRIKE_ROUTE_SIZES = 6.0  # ... and this many size measures along the movement, kept inside the plate
STRIKE_DIRECTION_FRAMES = 3  # the movement direction is the mean velocity over this many frames before the release
# default effect size per body part, starting values not tuned to any clip: (diameter / size measure,
# anchor_offset up as a share of the diameter, anchor_offset up in size measures); anchor_offset = [0, -(d_share * d + s_share * size)]
PART_SIZING: dict[str, tuple[float, float, float]] = {
    "palm": (DIAMETER_PER_PALM, ANCHOR_UP_PER_DIAMETER, 0.0),  # 1.15 x size, up 0.65 d (as before task 32)
    "fist": (1.6, 0.0, 0.0),       # 1.6 x size, on the fist
    "foot": (1.0, 0.5, 0.0),       # 1.0 x size, up 0.5 d (task 36: on the shoe, not in the floor)
    "knee": (1.0, 0.0, 0.0),       # 1.0 x size, on the knee
    "shoulder": (1.0, 0.5, 0.0),   # 1.0 x size, up 0.5 d
    "head": (1.0, 0.0, 1.1),       # 1.0 x size, 1.1 sizes above the head
}


@dataclass(frozen=True)
class StartTimings:
    appear_frame: int
    release_frame: int | None  # None for a "hold"
    vanish_start: int
    vanish_end: int
    diameter_px: int
    anchor_offset: tuple[int, int]
    fallback: bool  # True: no steady palm found, the timings are fractions of the clip
    behaviour: str = "hold_throw"  # "hold": a steady run and no throw after it
    hold_shortened: bool = False   # a hold cut to MAX_EFFECT_FRAMES
    route_end: tuple[int, int, int] | None = None  # strike rule: (frame, x, y) of the default route's second point


def steady_runs(track: list[TrackPoint], steady_speed: float = STEADY_SPEED, max_gap: int = MAX_GAP
                ) -> list[tuple[int, int]]:
    """Runs [first, last] (inclusive) of steady frames, merged across gaps of at most `max_gap` frames.

    A frame is steady if it and the one before it are tracked (not lost) and the palm moved less than
    `steady_speed` x the median palm width since the frame before.
    """
    if len(track) < 2:
        return []
    widths = [p.palm_w for p in track if p.palm_w > 0]
    if not widths:
        return []
    scale = statistics.median(widths)
    flags = [False] * len(track)
    for i in range(1, len(track)):
        a, b = track[i - 1], track[i]
        flags[i] = not (a.lost or b.lost) and math.hypot(b.x - a.x, b.y - a.y) / scale < steady_speed
    runs: list[list[int]] = []
    for i, ok in enumerate(flags):
        if not ok:
            continue
        if runs and i - runs[-1][1] - 1 <= max_gap:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [(a, b) for a, b in runs]


def find_throw(track: list[TrackPoint], run_end: int, fps: float) -> int | None:
    """The release frame after a steady run ending at `run_end`: the first frame whose palm speed is above
    `THROW_SPEED` palm widths, or the first frame of a lost gap, within `THROW_SEARCH_S` after the run.
    None when neither happens in that window.
    """
    widths = [p.palm_w for p in track if p.palm_w > 0]
    if not widths:
        return None
    scale = statistics.median(widths)
    for f in range(run_end + 1, min(len(track), run_end + 1 + round(THROW_SEARCH_S * fps))):
        a, b = track[f - 1], track[f]
        if b.lost and not a.lost:
            return f
        if not (a.lost or b.lost) and math.hypot(b.x - a.x, b.y - a.y) / scale > THROW_SPEED:
            return f
    return None


def throw_frame(track: list[TrackPoint], run_end: int, fps: float) -> int:
    """`find_throw`, or `run_end + 1` when there is no throw in the search window."""
    found = find_throw(track, run_end, fps)
    return run_end + 1 if found is None else found


def _window_fits(appear: int, vanish_end: int, n_in: int, frames: int) -> bool:
    """True if the effect window over the entrance .. vanish.end is within the window limit (services.cue.window)."""
    start = max(0, appear - max(min(n_in, appear), LEAD_IN))
    span = min(frames, vanish_end) - start
    return (span + 6) // 8 * 8 + 1 <= _longest(min(MAX_EFFECT_FRAMES, frames))


def _size(track: list[TrackPoint], first: int, last: int, part: str = "palm") -> tuple[int, tuple[int, int]]:
    d_per_size, up_d, up_size = PART_SIZING[part]
    widths = [p.palm_w for p in track[first:last + 1] if p.palm_w > 0]
    size = statistics.median(widths) if widths else None
    diameter = round(d_per_size * size) if widths else round(FALLBACK_DIAMETER)
    up = up_d * diameter + (up_size * size if widths else 0.0)
    return diameter, (0, -round(up))


def strike_speeds(track: list[TrackPoint]) -> list[float]:
    """Speed of the point in size measures per frame, averaged over `STRIKE_SMOOTH` frames (centred; fewer at the ends).
    The size measure is the track's median `palm_w`. A step into or out of a lost frame has no speed (0)."""
    widths = [p.palm_w for p in track if p.palm_w > 0]
    if len(track) < 2 or not widths:
        return [0.0] * len(track)
    scale = statistics.median(widths)
    raw = [0.0] + [0.0 if a.lost or b.lost else math.hypot(b.x - a.x, b.y - a.y) / scale
                   for a, b in zip(track, track[1:])]
    half = STRIKE_SMOOTH // 2
    return [statistics.fmean(raw[max(1, f - half):min(len(raw), f + half + 1)]) if f else 0.0 for f in range(len(raw))]


def strike_release(track: list[TrackPoint], part: str = "fist") -> int | None:
    """The release frame of a strike, or None. fist: the fastest frame (the first such frame), if its smoothed speed is
    above `STRIKE_SPEED["fist"]`. foot, knee: the peak (first such frame) of the first run of at least `STRIKE_MIN_RUN`
    frames above `STRIKE_SPEED[part]`: the kick out, not the swing back."""
    speeds, limit = strike_speeds(track), STRIKE_SPEED[part]
    if not speeds:
        return None
    if part == "fist":
        return speeds.index(max(speeds)) if max(speeds) > limit else None
    f = 0
    while f < len(speeds):
        if speeds[f] <= limit:
            f += 1
            continue
        end = f
        while end + 1 < len(speeds) and speeds[end + 1] > limit:
            end += 1
        if end - f + 1 >= STRIKE_MIN_RUN:
            return max(range(f, end + 1), key=lambda i: (speeds[i], -i))
        f = end + 1
    return None


def _direction(track: list[TrackPoint], release: int) -> tuple[float, float]:
    """Unit vector of the mean velocity over the `STRIKE_DIRECTION_FRAMES` frames before `release`; straight up where
    the point did not move at all."""
    first, last = max(0, release - STRIKE_DIRECTION_FRAMES - 1), max(0, release - 1)
    dx, dy = track[last].x - track[first].x, track[last].y - track[first].y
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else (0.0, -1.0)


def _strike(track: list[TrackPoint], release: int, fps: float, frames: int, n_in: int, part: str,
            picture: tuple[int, int, int, int]) -> StartTimings:
    """Starting timings of a strike released at `release`: appear 1 s before it, the vanish of any throw, and a default
    route whose second point is 0.5 s later and 6 sizes along the movement, inside `picture` (x0, y0, x1, y1: the conformed
    picture without its black padding) by `STRIKE_MARGIN_D` x the diameter, so a burst or shatter there is fully in frame."""
    appear = max(0, release - seconds_to_frames(STRIKE_LEAD_S, fps))
    v_start = release + seconds_to_frames(VANISH_START_S, fps)
    v_end = min(frames, v_start + seconds_to_frames(VANISH_LEN_S, fps))
    v_start = min(v_start, v_end - 1)
    appear = min(appear, release - 1) if release > 0 else 0
    while appear < release - 1 and not _window_fits(appear, v_end, n_in, frames):
        appear += 1
    diameter, offset = _size(track, 0, len(track) - 1, part)
    widths = [p.palm_w for p in track if p.palm_w > 0]
    size = statistics.median(widths) if widths else diameter / PART_SIZING[part][0]
    window, _ = propose_effect_range(appear, v_end, frames)  # the route's last point must be inside the window
    last = min(release + seconds_to_frames(STRIKE_ROUTE_S, fps), window.end - 1, frames - 1)
    end = None
    if last > release:
        ux, uy = _direction(track, release)
        x = track[release].x + offset[0] + STRIKE_ROUTE_SIZES * size * ux
        y = track[release].y + offset[1] + STRIKE_ROUTE_SIZES * size * uy
        m = STRIKE_MARGIN_D * diameter
        x0, y0, x1, y1 = picture
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x = cx if x1 - x0 < 2 * m else min(max(x, x0 + m), x1 - m)  # a picture too small for the margin: its centre
        y = cy if y1 - y0 < 2 * m else min(max(y, y0 + m), y1 - m)
        end = (last, round(x), round(y))
    return StartTimings(appear, release, v_start, v_end, diameter, offset, fallback=False, route_end=end)


def start_timings(track: list[TrackPoint], fps: float, frames: int, entrance_seconds: float | None = None,
                  part: str = "palm", plate_size: tuple[int, int] = (1920, 1080),
                  picture: tuple[int, int, int, int] | None = None) -> StartTimings:
    """Appear, release, vanish, size and anchor offset for a new cue from the analyze `track` (see the task 28 rule).

    The longest steady run [a, b] gives appear = a + N_in (the entrance builds up while the palm already holds) and
    release = the throw after it (`throw_frame`: first fast or lost frame within 1.5 s, else b + 1). Without a run
    of N_in + 8 frames, appear and release are fractions of the clip (`fallback`).
    `part` (a body point's part) sets the default diameter and anchor offset (`PART_SIZING`); for a fist, foot or knee
    the strike rule (`strike_release`) goes first: a fast enough movement is the release and gets a two-point route
    inside `picture` (default: the whole plate of `plate_size`); a track with no such movement is read as below.
    Vanish follows the release; appear moves later (never the release) until the effect window fits the limit.
    With a run but no throw (`find_throw`) the result is a "hold": appear as above, vanish right after the run (length
    `VANISH_LEN_S`), and where the window is too long the vanish moves earlier (appear stays) until it fits.
    """
    n_in = seconds_to_frames(Entrance().seconds if entrance_seconds is None else entrance_seconds, fps)
    if part in STRIKE_PARTS:
        strike = strike_release(track, part)
        if strike is not None:
            return _strike(track, min(strike, frames - 1), fps, frames, n_in, part,
                           picture or (0, 0, plate_size[0], plate_size[1]))
    runs = [r for r in steady_runs(track) if r[1] - r[0] + 1 >= n_in + MIN_HOLD_EXTRA]
    if runs:
        first, last = max(runs, key=lambda r: (r[1] - r[0], -r[0]))
        appear = min(first + n_in, last)
        throw = find_throw(track, last, fps)
        if throw is None:
            return _hold(track, first, last, appear, n_in, fps, frames, part)
        release = min(throw, frames - 1)
    else:
        first, last = 0, len(track) - 1
        appear, release = round(FALLBACK_APPEAR * frames), round(FALLBACK_RELEASE * frames)
        release = min(max(release, 1), frames - 1)
    v_start = release + seconds_to_frames(VANISH_START_S, fps)
    v_end = min(frames, v_start + seconds_to_frames(VANISH_LEN_S, fps))
    v_start = min(v_start, v_end - 1)
    appear = min(appear, release - 1) if release > 0 else 0
    while appear < release - 1 and not _window_fits(appear, v_end, n_in, frames):
        appear += 1
    diameter, offset = _size(track, first, last, part)
    return StartTimings(max(appear, 0), release, v_start, v_end, diameter, offset, fallback=not runs)


def _hold(track: list[TrackPoint], first: int, last: int, appear: int, n_in: int, fps: float, frames: int,
          part: str = "palm") -> StartTimings:
    """Starting timings of a "hold": no release, vanish from the end of the steady run [first, last]."""
    length = max(1, seconds_to_frames(VANISH_LEN_S, fps))
    v_start = min(last + 1, frames - 1)
    v_end = min(frames, v_start + length)
    start = max(0, appear - max(min(n_in, appear), LEAD_IN))  # the window starts at the entrance
    shortened = v_end - start > MAX_EFFECT_FRAMES
    if shortened:  # longer than the limit: the vanish moves earlier, appear stays
        v_end = max(start + MAX_EFFECT_FRAMES, appear + 1 + length)
        v_start = v_end - length
    diameter, offset = _size(track, first, last, part)
    return StartTimings(appear, None, v_start, v_end, diameter, offset, fallback=False, behaviour="hold",
                        hold_shortened=shortened)
