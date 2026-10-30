"""Cue file v2.6 (docs/BRIEF.md section 4, contract/CONTRACT.md): every field, load/save, and `validate`.

Field names are the contract with the UI team. Keys this module does not know are kept in
`Cue.extra` and written back unchanged, so loading and saving a full cue file loses nothing.
A cue written by the tracking step has only the first block of fields; the rest are None
until the artist (or an example file) fills them in.

v2.1 files load as v2.2: ranges were already [start, end) in v2.1, so only the version changes and the v2.2
fields (`route[].easing_in`, `route[].note`, `grade.spill`) get their defaults. v2.2 files load as they are: v2.3 adds
only `behaviour` ("hold_throw" when missing), so a v2.2 file is a v2.3 file that does not use it. v2.4 adds only
`body_point` (the person's palm on `hand` when missing), so v2.2 and v2.3 files are v2.4 files that do not use it.
v2.5 adds only `shape` ("sphere" when missing) and `spin_deg_per_s` (the shape's default when missing), so older files
are v2.5 files that do not use them. v2.6 adds only `ending` ("fade" when missing), so older files are v2.6 files that do
not use it.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

VERSION = "2.3"            # what new cues are written as; a cue that sets `body_point` is written as BODY_POINT_VERSION
BODY_POINT_VERSION = "2.4"
SHAPE_VERSION = "2.5"      # a cue that sets `shape` or `spin_deg_per_s` is written as this
EFFECTS_VERSION = "2.6"    # a cue that sets a non-fade `ending` is written as this
SUPPORTED_VERSIONS = ("2.2", "2.3", "2.4", "2.5", "2.6")  # a file keeps its version when it is loaded, so v2.2 files stay byte for byte
UPGRADES = {"2.1": "2.2"}            # loaded as the newer version
BEHAVIOURS = ("hold_throw", "hold", "charge_up", "orbit")
NO_FLIGHT = ("hold", "orbit")  # the behaviours with no release and no route (they stay with the body point)
ENDINGS = ("fade", "burst", "shatter")  # how the effect ends at `vanish.start`; "fade" (Merge's linear fade) is the default
SHAPES = ("sphere", "ring", "crystal", "cube")  # what the layout and the stand-in draw; the sphere is the default
BODY_PARTS = ("palm", "fist", "foot", "knee", "shoulder", "head")  # the head has no side; palm is the hand tracker's
SOURCES = ("tracked", "interpolated", "corrected")
HANDS = ("right", "left")
OCCLUDERS = ("left_hand", "right_hand", "person")
EASINGS = ("linear", "ease_in", "ease_out", "ease_in_out")
MAX_EFFECT_FRAMES = 193  # 8k+1: 8 s at 24 fps, the longest window Make and Merge use
# defaults the UI reads from GET /api/defaults (task 36); the renderers and `check` read the same constants
ENDING_MIN_SECONDS = {"fade": 0.0, "burst": 0.3, "shatter": 0.5}  # the shortest vanish each ending needs
ENDING_DEFAULT_SECONDS = {"burst": 0.6, "shatter": 1.0}  # the vanish length the UI sets when the artist picks that ending (task 38)
CHARGE_UP = {"start_scale": 0.6, "end_scale": 1.0, "pulse_hz": 3.5, "pulse_amount": 0.08}  # size 0.6 -> 1.0, +-8 % pulse
ORBIT_DEFAULTS = {"radius_sizes": 1.5, "speed_deg_per_s": 180, "tilt_deg": 60}  # what a missing `orbit` field means
ORBIT_TILT_BY_PART = {"head": 80}  # a missing orbit.tilt_deg is this for the part, else ORBIT_DEFAULTS (the head's circle is nearly edge on)
ORBIT_LIMITS = {"radius_sizes": (0.0, 10.0), "speed_deg_per_s": (-3600.0, 3600.0), "tilt_deg": (0.0, 90.0)}  # orbit_invalid beyond
_TINT = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class TrackPoint:
    """Palm centre of the chosen hand in one frame (plate pixels)."""
    f: int
    x: float
    y: float
    source: str  # "tracked" | "interpolated" | "corrected"
    palm_w: float
    conf: float
    lost: bool = False  # inside a lost gap: a run the tracker could not bridge, or one it jumped across
    # (addition to v2.1; files without it load as False)


@dataclass(frozen=True)
class Correction:
    """Artist keyframe: the palm centre should be at (x, y) in frame f."""
    f: int
    x: float
    y: float


@dataclass(frozen=True)
class HandSeed:
    """The artist's click on the chosen hand in frame f."""
    f: int
    x: float
    y: float


@dataclass(frozen=True)
class BodyPoint:
    """What the effect follows (v2.4): a part of the body and the person's side of it (None for the head)."""
    part: str
    side: str | None = None


@dataclass(frozen=True)
class Transform:
    """How the upload was conformed to the plate."""
    scale: float
    offset: tuple[float, float]


@dataclass(frozen=True)
class EffectRange:
    """Plate frames the effect is made for, [start, end)."""
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class RoutePoint:
    """A point the orb flies through at frame f; x and y are None where they are derived from the track.
    `easing_in` (v2.2) shapes the time of the segment that ends at this point; `note` (v2.2) is the artist's label."""
    f: int
    x: float | None
    y: float | None
    diameter_px: float
    easing_in: str = "linear"  # "linear" | "ease_in" | "ease_out" | "ease_in_out"
    note: str | None = None


@dataclass(frozen=True)
class Vanish:
    """The orb fades out linearly from `start` (fully visible) to `end` (gone)."""
    start: int
    end: int


@dataclass(frozen=True)
class Behind:
    """In [start, end) the plate's `occluder` is restored on top of the effect."""
    start: int
    end: int
    occluder: str | dict  # "left_hand" | "right_hand" | "person" | {"point": [f, x, y]}


@dataclass(frozen=True)
class Entrance:
    """The build-up before appear_frame: the stand-in grows from nothing to full over `seconds`.
    0 turns the entrance off (the orb is drawn in every frame and Merge's fade-in starts at appear_frame)."""
    seconds: float = 0.5


@dataclass(frozen=True)
class Orbit:
    """The circle an "orbit" effect travels on round its body point (v2.6): the radius in size measures of the body point,
    the speed in degrees per second (the sign is the direction), and the tilt of the circle's plane in degrees (0 = seen
    from the front, a full circle; 90 = edge on, a line). A missing field is None here and is read as `ORBIT_DEFAULTS` (the
    tilt of the head: `ORBIT_TILT_BY_PART`) by `orbit_of`."""
    radius_sizes: float | None = None
    speed_deg_per_s: float | None = None
    tilt_deg: float | None = None


@dataclass(frozen=True)
class Interaction:
    """Timing of the interaction pass segment: the cross-fade into and out of it, and the tail after the orb is gone."""
    crossfade_seconds: float = 0.25
    exit_tail_seconds: float = 0.25


@dataclass(frozen=True)
class Grade:
    tint: str
    brightness: float
    glow: float
    person_light: float
    opacity: float
    fade_in_frames: int
    spill: float | None = None  # v2.2: light on whatever passes in front; stored, not used by Merge yet


@dataclass
class Cue:
    clip: str
    fps: float
    frames: int
    plate_size: tuple[int, int]
    hand: str  # the person's hand: "right" | "left"
    hand_seed: HandSeed
    track: list[TrackPoint] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)
    correction_falloff: int = 12  # frames a correction fades out beyond its first/last keyframe
    version: str = VERSION
    transform: Transform | None = None
    effect_range: EffectRange | None = None
    canvas_size: tuple[int, int] | None = None
    canvas_rect: tuple[int, int, int, int] | None = None  # [x, y, w, h] in plate pixels
    anchor_offset: tuple[float, float] | None = None
    appear_frame: int | None = None
    release_frame: int | None = None
    diameter_px: float | None = None
    route: list[RoutePoint] | None = None
    vanish: Vanish | None = None
    behind: list[Behind] = field(default_factory=list)
    look: str | None = None
    seed: int | None = None
    grade: Grade | None = None
    behaviour: str | None = None            # v2.3; None (missing) means "hold_throw", see `behaviour_of`
    body_point: BodyPoint | None = None     # v2.4; None (missing) means the palm of `hand`, see `body_point_of`
    shape: str | None = None                # v2.5; None (missing) means "sphere", see `shape_of`
    spin_deg_per_s: float | None = None     # v2.5; None (missing) means the shape's default spin (services.make.layout.SHAPE_SPECS)
    ending: str | None = None               # v2.6; None (missing) means "fade", see `ending_of`
    orbit: Orbit | None = None              # v2.6; used by behaviour "orbit"; None (missing) means ORBIT_DEFAULTS, see `orbit_of`
    entrance: Entrance | None = None        # addition to v2.1; None (missing) means the defaults, see `entrance_of`
    interaction: Interaction | None = None  # addition to v2.1; None (missing) means the defaults
    extra: dict = field(default_factory=dict)  # fields owned by other steps, preserved as-is


def behaviour_of(cue: Cue) -> str:
    """The cue's behaviour: "hold_throw" (held, released, flies along the route), "charge_up" (the same, and it grows and
    pulses while it is held, v2.6), "hold" (stays on the hand) or "orbit" (circles the body point, v2.6)."""
    return cue.behaviour if cue.behaviour is not None else "hold_throw"


def orbit_of(cue: Cue) -> Orbit:
    """The circle of the cue, every field filled: the cue's orbit block, each missing field from the defaults (the tilt
    from `ORBIT_TILT_BY_PART` for the body point's part first)."""
    o = cue.orbit if cue.orbit is not None else Orbit()
    tilt = ORBIT_TILT_BY_PART.get(body_point_of(cue).part, ORBIT_DEFAULTS["tilt_deg"])
    return Orbit(o.radius_sizes if o.radius_sizes is not None else ORBIT_DEFAULTS["radius_sizes"],
                 o.speed_deg_per_s if o.speed_deg_per_s is not None else ORBIT_DEFAULTS["speed_deg_per_s"],
                 o.tilt_deg if o.tilt_deg is not None else tilt)


def orbit_issues(orbit: Orbit) -> list[str]:
    """What is wrong with the values of an orbit block (each starts with its path), [] if it is usable: every field is a
    number inside `ORBIT_LIMITS`, and the radius is above 0."""
    out = []
    for name, (lo, hi) in ORBIT_LIMITS.items():
        v = getattr(orbit, name)
        if v is None:
            continue  # missing: the default
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            out.append(f"orbit.{name}: must be a number, got {v!r}")
        elif not lo <= v <= hi or (name == "radius_sizes" and v <= 0):
            out.append(f"orbit.{name}: {v} is outside {'>' if name == 'radius_sizes' else ''}{lo:g}..{hi:g}")
    return out


def no_flight(cue: Cue) -> bool:
    """True for a behaviour that never releases (so `release_frame` is null and `route` is [])."""
    return behaviour_of(cue) in NO_FLIGHT


def shape_of(cue: Cue) -> str:
    """The cue's shape: "sphere", "ring", "crystal" or "cube" ("sphere" when the cue has none)."""
    return cue.shape if cue.shape is not None else "sphere"


def ending_of(cue: Cue) -> str:
    """The cue's ending: "fade" (Merge fades the effect out over `vanish`), "burst" or "shatter" (the effect ends itself)."""
    return cue.ending if cue.ending is not None else "fade"


def ending_frames(cue: Cue) -> int:
    """Frames the ending needs at least: ceil(ENDING_MIN_SECONDS x fps); 0 for a fade."""
    seconds = ENDING_MIN_SECONDS.get(ending_of(cue), 0.0)
    return math.ceil(seconds * cue.fps - 1e-9) if seconds > 0 else 0


def vanish_of(cue: Cue) -> Vanish:
    """`cue.vanish`, with its end moved later where a burst or shatter needs more frames than it has (`normalize` writes
    this into the cue, and `validate` warns `vanish_extended` until it does). A fade is as stored."""
    v, need = cue.vanish, ending_frames(cue)
    return v if v is None or v.end - v.start >= need else Vanish(v.start, v.start + need)


def min_version(cue: Cue) -> str | None:
    """The oldest cue version that can hold the cue's newer fields ("2.6" for a non-fade `ending`, a "charge_up" or "orbit"
    or an `orbit` block, "2.5" for `shape`/`spin_deg_per_s`, "2.4" for `body_point`, "2.3" for a "hold"), or None when it
    uses none of them. A file is written as at least this."""
    if cue.ending not in (None, "fade") or cue.behaviour in ("charge_up", "orbit") or cue.orbit is not None:
        return EFFECTS_VERSION
    if cue.shape is not None or cue.spin_deg_per_s is not None:
        return SHAPE_VERSION
    if cue.body_point is not None:
        return BODY_POINT_VERSION
    return "2.3" if cue.behaviour == "hold" else None


def body_point_of(cue: Cue) -> BodyPoint:
    """The body point the effect follows: the cue's, or the palm of `hand` if it has none."""
    return cue.body_point if cue.body_point is not None else BodyPoint("palm", cue.hand)


def is_palm(bp: BodyPoint | None) -> bool:
    """True for no body point or the palm: the hand tracker's path, which no body point changes."""
    return bp is None or bp.part == "palm"


def body_point_issue(bp: BodyPoint, hand: str | None) -> str | None:
    """Why `bp` is not a usable body point for a cue with this `hand` (None: not known, the palm's side is not
    compared), or None. The palm's side must be the hand; every other part but the head needs left or right; the
    head's side is ignored."""
    if bp.part not in BODY_PARTS:
        return f"part: {bp.part!r} is not one of {', '.join(BODY_PARTS)}"
    if bp.part == "head":
        return None
    if bp.side not in HANDS:
        return f"side: {bp.side!r} must be 'left' or 'right' for the {bp.part}"
    if bp.part == "palm" and hand is not None and bp.side != hand:
        return f"side: {bp.side!r} must equal hand {hand!r} for the palm"
    return None


def entrance_of(cue: Cue) -> Entrance:
    """The cue's entrance block, or the defaults if it has none."""
    return cue.entrance if cue.entrance is not None else Entrance()


def interaction_of(cue: Cue) -> Interaction:
    """The cue's interaction block, or the defaults if it has none."""
    return cue.interaction if cue.interaction is not None else Interaction()


def seconds_to_frames(seconds: float, fps: float) -> int:
    """round(seconds x fps), never below 0."""
    return max(0, round(seconds * fps))


def entrance_frames(cue: Cue) -> int:
    """N_in: frames of the entrance, round(entrance.seconds x fps), cut to appear_frame so it never starts before
    frame 0 (`validate` warns when it is cut)."""
    return min(seconds_to_frames(entrance_of(cue).seconds, cue.fps), max(cue.appear_frame or 0, 0))


def entrance_start(cue: Cue) -> int:
    """C = appear_frame - N_in: the first frame of the entrance (appear_frame itself when the entrance is off)."""
    return cue.appear_frame - entrance_frames(cue)


# --------------------------------------------------------------------------- lost tracking gaps

def lost_runs(track: list[TrackPoint]) -> list[tuple[int, int]]:
    """Runs [start, end) of consecutive frames the tracker flagged `lost`."""
    runs, start = [], None
    for f, p in enumerate(track):
        if p.lost and start is None:
            start = f
        elif not p.lost and start is not None:
            runs.append((start, f))
            start = None
    if start is not None:
        runs.append((start, len(track)))
    return runs


def open_lost_runs(cue: Cue) -> list[tuple[int, int]]:
    """The lost runs no correction lies in. A gap the artist has clicked into is handled: those clicks are its anchors."""
    return [(a, b) for a, b in lost_runs(cue.track) if not any(a <= c.f < b for c in cue.corrections)]


# --------------------------------------------------------------------------- validation

@dataclass(frozen=True)
class Issue:
    """One problem `check` found. `message` starts with its field path (this is what `validate` returns); `code` is
    the contract code (contract/schemas/error_codes.json); `frames` is the plate range [start, end) it is about."""
    severity: str  # "error" | "warning"
    code: str
    message: str
    frames: tuple[int, int] | None = None


class _Issues(list):
    """A list of Issue with shorthands for adding errors and warnings."""

    def error(self, code: str, message: str, frames: tuple[int, int] | None = None) -> None:
        self.append(Issue("error", code, message, frames))

    def warning(self, code: str, message: str, frames: tuple[int, int] | None = None) -> None:
        self.append(Issue("warning", code, message, frames))

    def any_error(self, *prefixes: str) -> bool:
        return any(i.severity == "error" and i.message.startswith(prefixes) for i in self)


def _structure_issues(cue: Cue) -> _Issues:
    """Problems with the fields the tracking step writes; used when saving and loading."""
    n, out = cue.frames, _Issues()
    if cue.hand not in HANDS:
        out.error("hand_invalid", f"hand: must be 'left' or 'right', got {cue.hand!r}")
    if not 0 <= cue.hand_seed.f < n:
        out.error("frame_outside_clip", f"hand_seed.f: {cue.hand_seed.f} is outside 0..{n - 1}")
    for i, p in enumerate(cue.track):
        if p.source not in SOURCES:
            out.error("track_invalid", f"track[{i}].source: {p.source!r} is not one of {', '.join(SOURCES)}", (i, i + 1))
    if cue.track and [p.f for p in cue.track] != list(range(n)):
        out.error("track_invalid", f"track: must have exactly one point per frame 0..{n - 1}, in order; "
                                   f"got {len(cue.track)} points")
    if cue.correction_falloff < 0:
        out.error("value_out_of_range", f"correction_falloff: must be >= 0, got {cue.correction_falloff}")
    seen: set[int] = set()
    for i, c in enumerate(cue.corrections):
        if not 0 <= c.f < n:
            out.error("frame_outside_clip", f"corrections[{i}].f: {c.f} is outside 0..{n - 1}")
        if c.f in seen:
            out.error("correction_duplicate", f"corrections[{i}].f: a second correction for frame {c.f}", (c.f, c.f + 1))
        seen.add(c.f)
    return out


def validate_cue(cue: Cue) -> None:
    """Raise ValueError if the tracking fields break the section 4 conventions."""
    errors = [i.message for i in _structure_issues(cue)]
    if errors:
        raise ValueError("; ".join(errors))


def validate(cue: Cue) -> tuple[list[str], list[str]]:
    """Every problem in the cue as (errors, warnings). Each message starts with its field path."""
    issues = check(cue)
    return ([i.message for i in issues if i.severity == "error"],
            [i.message for i in issues if i.severity == "warning"])


def check(cue: Cue) -> list[Issue]:
    """Every problem in the cue, with its contract code: the errors in the order found, then the warnings."""
    out = _structure_issues(cue)
    n = cue.frames
    if behaviour_of(cue) not in BEHAVIOURS:
        out.error("behaviour_invalid", f"behaviour: must be one of {', '.join(BEHAVIOURS)}, got {cue.behaviour!r}")
    if shape_of(cue) not in SHAPES:
        out.error("shape_invalid", f"shape: must be one of {', '.join(SHAPES)}, got {cue.shape!r}")
    if cue.orbit is not None:
        for why in orbit_issues(cue.orbit):
            out.error("orbit_invalid", why)
    if ending_of(cue) not in ENDINGS:
        out.error("ending_invalid", f"ending: must be one of {', '.join(ENDINGS)}, got {cue.ending!r}")
    if cue.spin_deg_per_s is not None and (not isinstance(cue.spin_deg_per_s, (int, float))
                                           or isinstance(cue.spin_deg_per_s, bool)):
        out.error("value_out_of_range", f"spin_deg_per_s: must be a number, got {cue.spin_deg_per_s!r}")
    if cue.body_point is not None:
        why = body_point_issue(cue.body_point, cue.hand)
        if why:
            out.error("body_point_invalid", f"body_point.{why}")
    hold = no_flight(cue)  # no release and no route to check
    for name in ("effect_range", "canvas_size", "canvas_rect", "anchor_offset", "appear_frame", "release_frame",
                 "diameter_px", "route", "vanish", "grade"):
        if getattr(cue, name) is None and not (hold and name in ("release_frame", "route")):
            out.error("missing_field", f"{name}: missing")
    if cue.version not in SUPPORTED_VERSIONS:
        out.error("version_unsupported", f"version: must be one of {', '.join(SUPPORTED_VERSIONS)}, got {cue.version!r}")
    if not cue.track:
        out.error("no_track", f"track: empty; needs one point per frame 0..{n - 1}", (0, n))
    er = cue.effect_range
    if er is not None:
        if er.length < 1 or (er.length - 1) % 8:
            out.error("effect_range_length", f"effect_range: length {er.length} is not 8k+1", (er.start, er.end))
        if er.start < 0 or er.end > n:
            out.error("effect_range_outside", f"effect_range: {er.start}..{er.end} is outside the clip 0..{n}",
                      (er.start, er.end))

    def in_range(start: int, end: int) -> bool:
        return er is None or (er.start <= start and end <= er.end)

    if cue.canvas_size is not None:
        for axis, v in zip("wh", cue.canvas_size):
            if v <= 0 or v % 64:
                out.error("canvas_invalid", f"canvas_size: {axis} = {v} is not divisible by 64")
    if cue.canvas_rect is not None:
        x, y, w, h = cue.canvas_rect
        pw, ph = cue.plate_size
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > pw or y + h > ph:
            out.error("canvas_invalid", f"canvas_rect: {list(cue.canvas_rect)} is outside the plate {pw}x{ph}")
        elif cue.canvas_size is not None and cue.canvas_size[0] > 0 and cue.canvas_size[1] > 0:
            ratio = (w / h) / (cue.canvas_size[0] / cue.canvas_size[1])
            if abs(ratio - 1) > 0.005:
                out.error("canvas_invalid", f"canvas_rect: aspect {w / h:.4f} differs from canvas_size aspect "
                                            f"{cue.canvas_size[0] / cue.canvas_size[1]:.4f} by more than 0.5 %")

    appear, release = cue.appear_frame, cue.release_frame
    for name, v in (("appear_frame", appear), ("release_frame", None if hold else release)):
        if v is not None and er is not None and not er.start <= v < er.end:
            out.error("frame_outside_effect_range", f"{name}: {v} is outside effect_range {er.start}..{er.end}",
                      (v, v + 1))
    if appear is not None and release is not None and appear >= release and not hold:
        out.error("release_not_after_appear", f"release_frame: {release} is not after appear_frame {appear}",
                  (release, release + 1))

    ent, inter = entrance_of(cue), interaction_of(cue)
    for path, v in (("entrance.seconds", ent.seconds), ("interaction.crossfade_seconds", inter.crossfade_seconds),
                    ("interaction.exit_tail_seconds", inter.exit_tail_seconds)):
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not v >= 0:
            out.error("value_out_of_range", f"{path}: must be a number >= 0, got {v!r}")
    entrance_ok = not out.any_error("entrance.", "appear_frame", "effect_range")
    if entrance_ok and appear is not None:
        wanted = seconds_to_frames(ent.seconds, cue.fps)
        if wanted > appear:
            out.warning("entrance_shortened",
                        f"entrance.seconds: {ent.seconds:g} s is {wanted} frames, more than appear_frame {appear}; "
                        f"the entrance is cut to {appear} frames so it starts at frame 0", (0, appear))
    if er is not None:
        from services.cue.window import cover_from  # here, not at the top: services.cue.window imports this module
        c = entrance_start(cue) if entrance_ok and appear is not None else er.start
        w = er if out.any_error("effect_range") else cover_from(er, c, n)
        if w != er:
            out.warning("window_grown",
                        f"effect_range: {er.start}..{er.end} starts after the entrance start {c} (appear_frame "
                        f"{appear} - {appear - c} entrance frames); Make and Merge use {w.start}..{w.end} "
                        f"({w.length} frames)", (w.start, w.end))
            if w.end - w.start < er.end - c:
                out.warning("window_cut_short",
                            f"effect_range: the clip is too short to grow the window to cover {c}..{er.end}; "
                            f"frames {w.end}..{er.end} get no effect", (w.end, er.end))
        if w.length > MAX_EFFECT_FRAMES:  # Make trims it (services.cue.window.make_range): degrade, don't fail
            keep = (MAX_EFFECT_FRAMES - 1) // 8 * 8 + 1
            out.warning("window_trimmed",
                        f"effect_range: length {w.length} is more than {MAX_EFFECT_FRAMES}; Make trims it to "
                        f"{w.start}..{w.start + keep}, so the last {w.length - keep} frames "
                        f"({w.start + keep}..{w.end}) get no effect", (w.start + keep, w.end))

    route = cue.route
    if hold:
        if route or release is not None:
            out.warning("hold_path_unused", "behaviour: hold; release_frame and route are not used")
    elif route is not None:
        if not route:
            out.error("route_empty", "route: empty; needs at least the release point")
        else:
            if release is not None and route[0].f != release:
                out.error("route_start", f"route[0].f: {route[0].f} is not release_frame {release}",
                          (route[0].f, route[0].f + 1))
            for i, p in enumerate(route):
                if i and p.f <= route[i - 1].f:
                    out.error("route_order", f"route[{i}].f: {p.f} is not after route[{i - 1}].f {route[i - 1].f}",
                              (p.f, p.f + 1))
                if er is not None and not er.start <= p.f < er.end:
                    out.error("frame_outside_effect_range",
                              f"route[{i}].f: {p.f} is outside effect_range {er.start}..{er.end}", (p.f, p.f + 1))
                if i and p.x is None:
                    out.error("route_position_missing", f"route[{i}].x: null; only route[0] may derive its position")
                if i and p.y is None:
                    out.error("route_position_missing", f"route[{i}].y: null; only route[0] may derive its position")
                if p.diameter_px is None or p.diameter_px <= 0:
                    out.error("value_out_of_range", f"route[{i}].diameter_px: must be > 0, got {p.diameter_px}")
                if p.easing_in not in EASINGS:
                    out.error("easing_invalid",
                              f"route[{i}].easing_in: {p.easing_in!r} is not one of {', '.join(EASINGS)}")

    v = cue.vanish
    if v is not None:
        if v.start >= v.end:
            out.error("range_reversed", f"vanish: start {v.start} is not before end {v.end}")
        elif not in_range(v.start, v.end):
            out.error("range_outside_effect", f"vanish: {v.start}..{v.end} is outside effect_range "
                                              f"{er.start}..{er.end}", (v.start, v.end))
    if v is not None and v.start < v.end and ending_of(cue) in ("burst", "shatter") and not out.any_error("vanish"):
        wanted = vanish_of(cue)
        if wanted.end > v.end:
            lost = 0 if er is None else wanted.end - min(wanted.end, n, er.start + MAX_EFFECT_FRAMES)
            if er is not None and (wanted.end > n or wanted.end - er.start > MAX_EFFECT_FRAMES):
                # the window cannot grow to hold the ending (the 8 s cap, or the end of the clip): it is cut
                out.warning("window_trimmed",
                            f"vanish: a {cue.ending} needs {wanted.end - v.start} frames from {v.start}, to {wanted.end}; "
                            f"the effect window can reach {wanted.end - lost} at most, so the last {lost} frames of it "
                            f"get no effect", (wanted.end - lost, wanted.end))
            else:
                out.warning("vanish_extended",
                            f"vanish: a {cue.ending} needs at least {wanted.end - v.start} frames, the vanish has "
                            f"{v.end - v.start}; Make and Merge use {v.start}..{wanted.end}", (v.end, wanted.end))
    for i, b in enumerate(cue.behind):
        if b.start >= b.end:
            out.error("range_reversed", f"behind[{i}]: start {b.start} is not before end {b.end}")
        elif not in_range(b.start, b.end):
            out.error("range_outside_effect", f"behind[{i}]: {b.start}..{b.end} is outside effect_range "
                                              f"{er.start}..{er.end}", (b.start, b.end))
        if not _valid_occluder(b.occluder):
            out.error("occluder_invalid", f"behind[{i}].occluder: {b.occluder!r} is not left_hand, right_hand, person "
                                          f"or {{\"point\": [f, x, y]}}")

    if cue.diameter_px is not None and cue.diameter_px <= 0:
        out.error("value_out_of_range", f"diameter_px: must be > 0, got {cue.diameter_px}")
    g = cue.grade
    if g is not None:
        if not 0 <= g.opacity <= 1:
            out.error("value_out_of_range", f"grade.opacity: {g.opacity} is outside 0..1")
        if g.glow < 0:
            out.error("value_out_of_range", f"grade.glow: must be >= 0, got {g.glow}")
        if g.brightness < 0:
            out.error("value_out_of_range", f"grade.brightness: must be >= 0, got {g.brightness}")
        if not isinstance(g.tint, str) or not _TINT.match(g.tint):
            out.error("tint_invalid", f"grade.tint: {g.tint!r} is not #rrggbb")
        if g.fade_in_frames < 0:
            out.error("value_out_of_range", f"grade.fade_in_frames: must be >= 0, got {g.fade_in_frames}")
        if g.spill is not None and not g.spill >= 0:
            out.error("value_out_of_range", f"grade.spill: must be >= 0, got {g.spill}")

    # warnings: the cue is legal but the result will probably not look right
    # a fade is still on show until vanish.end; a burst or shatter (v2.6) ends itself from vanish.start, in place
    still_to = None if v is None else (v.end if ending_of(cue) == "fade" else v.start) - 1
    if route and not hold and still_to is not None and still_to > route[-1].f:
        out.warning("orb_holds_after_route",
                    f"vanish.end: the orb is still visible for {still_to - route[-1].f} frames after the last "
                    f"route point (frame {route[-1].f}, vanish ends at {v.end}); it will hold still there",
                    (route[-1].f + 1, still_to + 1))
    if len(cue.track) == n:
        lost = open_lost_runs(cue)
        if entrance_ok and appear is not None:
            c = entrance_start(cue)
            for a, b in lost:
                lo, hi = max(a, c), min(b, appear)
                if lo < hi:
                    out.warning("track_uncertain",
                                f"track[{lo}..{hi - 1}]: the hand is lost during the entrance ({c}..{appear - 1}); "
                                f"the orb holds its appear_frame position there", (lo, hi))
        held_end = (v.end if v is not None else None) if hold else release  # the orb is held until here
        if appear is not None and held_end is not None:
            for a, b in lost:
                lo, hi = max(a, appear), min(b, held_end)
                if lo < hi:
                    out.warning("track_uncertain",
                                f"track[{lo}..{hi - 1}]: the held orb follows a position interpolated across a "
                                f"lost tracking gap ({a}..{b - 1}); it will slide there", (lo, hi))
        if release is not None and not hold:
            for a, b in lost:
                if a <= release < b:
                    out.warning("release_in_lost_gap",
                                f"release_frame: {release} lies inside a lost tracking gap ({a}..{b - 1}); the "
                                f"orb is released from the hand's position at frame {a - 1 if a else b}", (a, b))
    errors = [i for i in out if i.severity == "error"]
    return errors + [i for i in out if i.severity == "warning"]


def _valid_occluder(o) -> bool:
    if isinstance(o, str):
        return o in OCCLUDERS
    if isinstance(o, dict) and set(o) == {"point"}:
        p = o["point"]
        return isinstance(p, (list, tuple)) and len(p) == 3 and all(isinstance(v, (int, float)) for v in p)
    return False


# --------------------------------------------------------------------------- JSON

# the order of section 4; None fields are left out
_KNOWN = ("version", "clip", "fps", "frames", "plate_size", "transform", "effect_range", "canvas_size",
          "canvas_rect", "hand", "hand_seed", "body_point", "track", "corrections", "correction_falloff", "anchor_offset",
          "behaviour", "orbit", "appear_frame", "release_frame", "diameter_px", "route", "vanish", "behind", "look", "shape",
          "spin_deg_per_s", "ending", "seed", "grade", "entrance", "interaction")


def cue_to_dict(cue: Cue) -> dict:
    """Cue as a JSON-ready dict in section 4 field order. `route[].note` and `grade.spill` are left out when None."""
    def dc(o, optional: tuple[str, ...] = ("note", "spill")):
        if o is None:
            return None
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(o).items()
                if not (k in optional and v is None)}

    d = {
        "version": cue.version,
        "clip": cue.clip,
        "fps": int(cue.fps) if float(cue.fps).is_integer() else cue.fps,
        "frames": cue.frames,
        "plate_size": list(cue.plate_size),
        "transform": dc(cue.transform),
        "effect_range": dc(cue.effect_range),
        "canvas_size": None if cue.canvas_size is None else list(cue.canvas_size),
        "canvas_rect": None if cue.canvas_rect is None else list(cue.canvas_rect),
        "hand": cue.hand,
        "hand_seed": dc(cue.hand_seed),
        "body_point": dc(cue.body_point),
        "track": [dc(p) for p in cue.track],
        "corrections": [dc(c) for c in cue.corrections],
        "correction_falloff": cue.correction_falloff,
        "anchor_offset": None if cue.anchor_offset is None else list(cue.anchor_offset),
        "behaviour": cue.behaviour,
        "orbit": dc(cue.orbit, optional=("radius_sizes", "speed_deg_per_s", "tilt_deg")),
        "appear_frame": cue.appear_frame,
        "release_frame": cue.release_frame,
        "diameter_px": cue.diameter_px,
        "route": None if cue.route is None else [dc(p) for p in cue.route],
        "vanish": dc(cue.vanish),
        "behind": [dc(b) for b in cue.behind],
        "look": cue.look,
        "shape": cue.shape,
        "spin_deg_per_s": cue.spin_deg_per_s,
        "ending": cue.ending,
        "seed": cue.seed,
        "grade": dc(cue.grade),
        "entrance": dc(cue.entrance),
        "interaction": dc(cue.interaction),
    }
    d = {k: v for k, v in d.items() if v is not None or (k == "release_frame" and no_flight(cue))}
    d.update(cue.extra)
    return d


def _block(cls, name: str, d):
    """An optional settings block from JSON; its missing fields get their defaults. ValueError (with the field path)
    if it is not an object or has a field `cls` does not know."""
    if not isinstance(d, dict):
        raise ValueError(f"{name}: must be an object, got {d!r}")
    known = set(cls.__dataclass_fields__)
    for k in d:
        if k not in known:
            raise ValueError(f"{name}.{k}: unknown field; expected {', '.join(sorted(known))}")
    return cls(**d)


def cue_from_dict(d: dict) -> Cue:
    """Inverse of `cue_to_dict`; keys it does not know go to `Cue.extra`."""
    def opt(key, make):
        return make(d[key]) if d.get(key) is not None else None

    version = d.get("version", VERSION)
    version = UPGRADES.get(version, version)  # v2.1 ranges are already [start, end): nothing to convert but the version
    return Cue(
        clip=d["clip"], fps=d["fps"], frames=d["frames"], plate_size=tuple(d["plate_size"]),
        hand=d["hand"], hand_seed=HandSeed(**d["hand_seed"]),
        body_point=opt("body_point", lambda b: _block(BodyPoint, "body_point", b)),
        track=[TrackPoint(**p) for p in d.get("track", [])],
        corrections=[Correction(**c) for c in d.get("corrections", [])],
        correction_falloff=d.get("correction_falloff", 12),
        version=version,
        transform=opt("transform", lambda t: Transform(t["scale"], tuple(t["offset"]))),
        effect_range=opt("effect_range", lambda r: EffectRange(**r)),
        canvas_size=opt("canvas_size", tuple),
        canvas_rect=opt("canvas_rect", tuple),
        anchor_offset=opt("anchor_offset", tuple),
        behaviour=d.get("behaviour"),
        orbit=opt("orbit", lambda o: _block(Orbit, "orbit", o)),
        appear_frame=d.get("appear_frame"),
        release_frame=d.get("release_frame"),
        diameter_px=d.get("diameter_px"),
        route=opt("route", lambda r: [RoutePoint(**p) for p in r]),
        vanish=opt("vanish", lambda v: Vanish(**v)),
        behind=[Behind(**b) for b in d.get("behind", [])],
        look=d.get("look"),
        shape=d.get("shape"),
        spin_deg_per_s=d.get("spin_deg_per_s"),
        ending=d.get("ending"),
        seed=d.get("seed"),
        grade=opt("grade", lambda g: Grade(**g)),
        entrance=opt("entrance", lambda e: _block(Entrance, "entrance", e)),
        interaction=opt("interaction", lambda i: _block(Interaction, "interaction", i)),
        extra={k: v for k, v in d.items() if k not in _KNOWN},
    )


def save_cue(cue: Cue, path: Path) -> None:
    """Check the tracking fields (ValueError if broken) and write the cue as JSON."""
    validate_cue(cue)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cue_to_dict(cue), indent=1) + "\n", encoding="utf-8")


def load_cue(path: Path) -> Cue:
    """Read a cue JSON file and check its tracking fields (ValueError if broken). Use `validate` for the rest."""
    cue = cue_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    validate_cue(cue)
    return cue
