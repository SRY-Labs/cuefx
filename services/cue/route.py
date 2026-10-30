"""Where the orb is in every frame: held on the palm, then flying along the route.

All positions are plate pixels; frames are plate frames. Only the cue is read, nothing else.
"""
from __future__ import annotations

import math
import statistics

import numpy as np

from services.cue.cue import (CHARGE_UP, Cue, TrackPoint, behaviour_of, ending_of, entrance_frames, entrance_start, no_flight,
                       open_lost_runs, orbit_of, validate, vanish_of)
from services.analysis.track import TrackConfig, apply_corrections
from services.cue.window import effect_window

OrbSample = tuple[int, float, float, float, float]  # frame, x, y, diameter_px, visibility
ORBIT_SIZE_SWING = 0.15  # an orbiting effect is this much bigger at the very front and smaller at the very back
ORBIT_BACK_LIGHT = 0.5   # and this bright at the very back (1 at the very front)

# route[i].easing_in: applied to the time fraction s (0..1) of the segment route[i-1] -> route[i], for position and
# size alike. The same curves as the UI's EASE table; "linear" returns s itself, so it routes exactly as v2.1 did.
EASE = {
    "linear": lambda s: s,
    "ease_in": lambda s: s * s,
    "ease_out": lambda s: 1 - (1 - s) * (1 - s),
    "ease_in_out": lambda s: s * s * (3 - 2 * s),
}


def corrected_track(cue: Cue) -> list[TrackPoint]:
    """The track with the artist's corrections applied (with the cue's falloff)."""
    return apply_corrections(cue.track, cue.corrections, TrackConfig(correction_falloff=cue.correction_falloff))


def held_xy(cue: Cue) -> np.ndarray:
    """(frames, 2): corrected palm centre plus `anchor_offset` for every plate frame."""
    xy = np.array([[p.x, p.y] for p in corrected_track(cue)], dtype=float)
    return xy + np.asarray(cue.anchor_offset, dtype=float)


def release_source_frame(cue: Cue) -> int:
    """The frame whose held position the release point is taken from.

    Normally `release_frame`. If that frame is interpolated inside a lost tracking gap the position is a
    guess halfway across a jump, so the last tracked frame before the gap is used (the first one after it,
    if the gap starts the clip). A gap the artist has corrected is trusted: its frames are not guesses.
    """
    for a, b in open_lost_runs(cue):
        if a <= cue.release_frame < b:
            return a - 1 if a > 0 else b
    return cue.release_frame


def release_point(cue: Cue) -> tuple[float, float]:
    """Position of `route[0]`: its own x, y if the cue gives them, else the held position at the source frame."""
    first = cue.route[0]
    if first.x is not None and first.y is not None:
        return float(first.x), float(first.y)
    x, y = held_xy(cue)[release_source_frame(cue)]
    return float(x), float(y)


def visibility(cue: Cue, f: int) -> float:
    """0 before the entrance start C (appear_frame when the entrance is off), ramp up over fade_in_frames from C,
    1 until vanish.start, linear down to 0 at vanish.end. The build-up itself is in the effect (`entrance_gain`).
    With a burst or shatter ending (v2.6) there is no linear fade-out: 1 until `vanish_of(cue).end`, then 0; the effect
    fades itself."""
    start, vanish = entrance_start(cue), vanish_of(cue)
    if f < start or f >= vanish.end:
        return 0.0
    fade = cue.grade.fade_in_frames
    up = min(1.0, (f - start + 1) / fade) if fade > 0 else 1.0
    if ending_of(cue) != "fade":  # a burst or shatter ends itself in the layout and the stand-in: no fade-out here
        return up
    down = 1.0 if f <= vanish.start else (vanish.end - f) / (vanish.end - vanish.start)
    return min(up, down)


def smoothstep(t: float) -> float:
    """3t^2 - 2t^3 for t clipped to 0..1: 0 at 0, 1 at 1, flat at both ends."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def entrance_gain(cue: Cue, f: int) -> float:
    """g: how much of the orb is there in frame f; the stand-in's diameter and brightness are scaled by it.

    With an entrance of N_in > 0 frames: 0 before C, smoothstep((f - C) / N_in) over C .. appear_frame - 1 (0 at C),
    1 from appear_frame on. With the entrance off (N_in = 0) it is 1 in every frame, as before the entrance existed.
    """
    n = entrance_frames(cue)
    if n == 0:
        return 1.0
    return smoothstep((f - (cue.appear_frame - n)) / n)


def orb_gains(cue: Cue, path: list[OrbSample]) -> list[float]:
    """`entrance_gain` for every sample of `path`."""
    return [entrance_gain(cue, s[0]) for s in path]


def first_full(gains: list[float] | None) -> int:
    """Index of the first sample with the whole orb (gain 1); 0 without gains."""
    if not gains:
        return 0
    return next((i for i, g in enumerate(gains) if g >= 1.0), len(gains) - 1)


def charge_growth(cue: Cue, f: int) -> float:
    """How big a "charge_up" is as a share of `diameter_px` in a held frame: `CHARGE_UP.start_scale` at appear_frame (and
    before it), then smoothstep to `end_scale` at release_frame. 1 for every other behaviour."""
    if behaviour_of(cue) != "charge_up":
        return 1.0
    lo, hi = CHARGE_UP["start_scale"], CHARGE_UP["end_scale"]
    return lo + (hi - lo) * smoothstep((f - cue.appear_frame) / max(cue.release_frame - cue.appear_frame, 1))


def charge_pulse(cue: Cue, f: int) -> float:
    """The "charge_up" pulse in frame f: 1 + `pulse_amount` x sin(2 pi `pulse_hz` t), t in seconds from appear_frame, from
    appear_frame up to (not including) release_frame; 1 otherwise and for every other behaviour. It scales the size and the
    brightness alike."""
    if behaviour_of(cue) != "charge_up" or not cue.appear_frame <= f < cue.release_frame:
        return 1.0
    return 1.0 + CHARGE_UP["pulse_amount"] * math.sin(2 * math.pi * CHARGE_UP["pulse_hz"] * (f - cue.appear_frame) / cue.fps)


def orbit_size(cue: Cue) -> float:
    """The size measure an orbit's radius is counted in: the median `palm_w` of the track (the body point's own size
    measure, as the strike rule uses), or `diameter_px` where the track has none."""
    widths = [p.palm_w for p in cue.track if p.palm_w > 0]
    return statistics.median(widths) if widths else float(cue.diameter_px)


def orbit_state(cue: Cue, f: int) -> tuple[float, float, float, float]:
    """(dx, dy, size x, light x) of an "orbit" effect in frame f, relative to the held position (body point + anchor_offset).

    The angle is `speed x t`, t in seconds from appear_frame (0 there). The circle has radius `radius_sizes` x the size
    measure; its vertical axis is squashed by cos(tilt). The side towards the viewer is the lower one (positive y): there
    the effect is `ORBIT_SIZE_SWING` bigger and at full light; the upper (far) side is that much smaller and
    `ORBIT_BACK_LIGHT` bright, smoothly with the angle. There is no occlusion by the body (known limit)."""
    o = orbit_of(cue)
    theta = math.radians(o.speed_deg_per_s * (f - cue.appear_frame) / cue.fps)
    r = o.radius_sizes * orbit_size(cue)
    depth = math.sin(theta)  # +1 at the very front, -1 at the very back
    light = (1 + ORBIT_BACK_LIGHT) / 2 + (1 - ORBIT_BACK_LIGHT) / 2 * depth
    return (r * math.cos(theta), r * depth * math.cos(math.radians(o.tilt_deg)), 1 + ORBIT_SIZE_SWING * depth, light)


def orb_lights(cue: Cue, path: list[OrbSample]) -> list[float]:
    """Brightness of the orb in every sample of `path`, on top of the entrance gain (which scales size and brightness):
    the charge-up pulse, or an orbit's front and back. 1 for every other behaviour."""
    if behaviour_of(cue) == "orbit":
        return [orbit_state(cue, s[0])[3] for s in path]
    return [charge_pulse(cue, s[0]) for s in path]


def catmull_rom(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, s: float,
                alpha: float = 0.5) -> np.ndarray:
    """Point at fraction s (0..1) of the segment p1 -> p2 on a centripetal Catmull-Rom spline (Barry-Goldman)."""
    knots = [0.0]
    for a, b in ((p0, p1), (p1, p2), (p2, p3)):
        knots.append(knots[-1] + max(float(np.linalg.norm(b - a)) ** alpha, 1e-9))
    t0, t1, t2, t3 = knots
    t = t1 + s * (t2 - t1)
    a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
    b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
    b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
    return (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2


def _flying(frames: list[int], xy: np.ndarray, diam: list[float], eases: list[str], f: int) -> tuple[float, float, float]:
    """Position and diameter at frame f >= frames[0]: spline between route points, still after the last one.
    `eases[i]` is route[i].easing_in, the shape of the time from route[i-1] to route[i]."""
    if f >= frames[-1]:
        return float(xy[-1, 0]), float(xy[-1, 1]), float(diam[-1])
    i = max(k for k in range(len(frames) - 1) if frames[k] <= f)
    s = EASE[eases[i + 1]]((f - frames[i]) / (frames[i + 1] - frames[i]))
    p0 = xy[i - 1] if i > 0 else 2 * xy[0] - xy[1]  # mirrored phantom points at both ends
    p3 = xy[i + 2] if i + 2 < len(xy) else 2 * xy[-1] - xy[-2]
    x, y = catmull_rom(p0, xy[i], xy[i + 1], p3, s)
    return float(x), float(y), float(diam[i] + s * (diam[i + 1] - diam[i]))


def orb_path(cue: Cue) -> list[OrbSample]:
    """(f, x, y, diameter_px, visibility) for every frame of `effect_window` (effect_range grown to cover the
    entrance, trimmed by `make_range` if too long).

    Before release_frame the orb is held: corrected track plus anchor_offset, diameter = `diameter_px` (the full
    size; the entrance scales what is drawn, see `entrance_gain`). In entrance frames where the track is lost (an
    open lost gap) the orb holds its appear_frame position; `validate` warns. From release_frame it flies through
    the route points (position: centripetal Catmull-Rom, size: linear, timing: by frame, shaped by each point's
    `easing_in`, see EASE) and stays at the last point afterwards. A "hold" cue never releases: every frame is the
    held position at `diameter_px` (the vanish fades it out). A "charge_up" cue is held like a hold_throw, but its
    held diameter grows from 0.6 to 1.0 x `diameter_px` and pulses (`charge_growth`, `charge_pulse`). An "orbit" cue
    never releases and travels on a circle round the held position (`orbit_state`). Raises ValueError if the cue has
    errors.
    """
    errors, _ = validate(cue)
    if errors:
        raise ValueError("cannot route an invalid cue:\n" + "\n".join(errors))
    held = held_xy(cue)
    c, a = entrance_start(cue), cue.appear_frame
    for lo, hi in open_lost_runs(cue):
        held[max(lo, c):min(hi, a)] = held[a]
    hold = no_flight(cue)
    orbit = behaviour_of(cue) == "orbit"
    if not hold:
        frames = [p.f for p in cue.route]
        xy = np.array([release_point(cue)] + [(p.x, p.y) for p in cue.route[1:]], dtype=float)
        diam = [p.diameter_px for p in cue.route]
        eases = [p.easing_in for p in cue.route]
    er = effect_window(cue)
    out: list[OrbSample] = []
    for f in range(er.start, er.end):
        if hold or f < cue.release_frame:
            x, y, d = float(held[f, 0]), float(held[f, 1]), float(cue.diameter_px)
            d *= charge_growth(cue, f) * charge_pulse(cue, f)  # 1 unless the behaviour is "charge_up"
            if orbit:
                dx, dy, size, _ = orbit_state(cue, f)
                x, y, d = x + dx, y + dy, d * size
        else:
            x, y, d = _flying(frames, xy, diam, eases, f)
        out.append((f, x, y, d, visibility(cue, f)))
    return out


def propose_route_timing(points_xy_d: list[tuple[float, float, float]], release_frame: int,
                         end_frame: int) -> list[int]:
    """A frame for each route point, to offer the artist as a default. `points_xy_d` is (x, y, diameter_px)
    for every route point, release point first; the result starts at release_frame and ends at end_frame.

    Model: the orb moves at a constant speed in the room, and the camera is a pinhole. Then a thing's
    on-screen size is proportional to 1 / depth, and so is its on-screen speed. So on-screen speed is
    proportional to on-screen size, and the time a segment takes is proportional to
    (on-screen length of the segment) / (mean diameter over the segment).
    The times are scaled so the last point lands on end_frame, rounded to whole frames, and nudged so the
    frames are strictly increasing. This assumes constant speed in the room and size proportional to
    1 / depth; an orb that speeds up or slows down, or whose size is not set by distance, will look off.
    Segment length is the straight line between points, not the curve through them. The proportionality is exact
    for sideways motion; for motion along the line of sight on-screen speed falls faster than size, so a throw
    straight away from the camera is given too few frames near the far end.
    """
    n = len(points_xy_d)
    if n < 2:
        raise ValueError("need at least two route points to propose timing")
    if end_frame - release_frame < n - 1:
        raise ValueError(f"{n} points need at least {n - 1} frames between release {release_frame} and end {end_frame}")
    if any(d <= 0 for _, _, d in points_xy_d):
        raise ValueError("diameters must be > 0")
    times = [0.0]
    for (x0, y0, d0), (x1, y1, d1) in zip(points_xy_d, points_xy_d[1:]):
        times.append(times[-1] + max(math.hypot(x1 - x0, y1 - y0) / ((d0 + d1) / 2), 1e-9))
    scale = (end_frame - release_frame) / times[-1]
    frames = [release_frame + round(t * scale) for t in times]
    for i in range(1, n):  # strictly increasing going forward ...
        frames[i] = max(frames[i], frames[i - 1] + 1)
    frames[-1] = end_frame
    for i in range(n - 2, 0, -1):  # ... and back from the fixed last point
        frames[i] = min(frames[i], frames[i + 1] - 1)
    return frames
