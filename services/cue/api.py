"""The backend side of the app contract (contract/CONTRACT.md), as pure functions on JSON-shaped data.

    analyze(clip_path, seed)      -> analyze_result      (raises CueFXError)
    preview_path(cue, track)      -> [preview]           (raises CueFXError)
    validate(cue, track=None)     -> [warning]           (never raises for a bad cue: problems are warnings)
    propose_timing(cue, track)    -> cue                 (raises CueFXError)
    new_cue(clip, ..., track)     -> cue                 a starter cue for an analyzed clip (raises CueFXError)
    normalize(cue)                -> cue                 derived fields recomputed (raises CueFXError)

Cues and tracks come in as dicts / lists exactly as in contract/schemas; nothing here keeps state. The shapes and the
codes (contract/schemas/error_codes.json) are read from the contract folder, so the contract is the one source.
"""
from __future__ import annotations

import copy
import json
import statistics
from dataclasses import asdict, replace
from functools import cache
from pathlib import Path

from services.cue.cue import (CHARGE_UP, ENDING_DEFAULT_SECONDS, ENDING_MIN_SECONDS, MAX_EFFECT_FRAMES, NO_FLIGHT, ORBIT_DEFAULTS, ORBIT_TILT_BY_PART, SUPPORTED_VERSIONS,
                       VERSION, BodyPoint, Cue, EffectRange, HandSeed, Issue, TrackPoint, body_point_issue, check,
                       cue_from_dict, cue_to_dict, ending_of, entrance_start, is_palm, lost_runs, min_version, no_flight,
                       vanish_of)
from services.make.layout import SHAPE_SPECS
from services.cue.start_timings import start_timings
from services.cue.route import entrance_gain, held_xy, orb_path, propose_route_timing, release_point

CONTRACT_DIR = Path(__file__).resolve().parents[2] / "contract"
SCHEMA_DIR = CONTRACT_DIR / "schemas"
SCHEMA_BASE = "https://schemas.cuefx.invalid/"

MAX_CLIP_FRAMES = 1440  # 60 s at 24 fps
MAX_LOST_SHARE = 0.5    # analyze fails (track_lost) when more than this share of the frames is inside lost gaps
PHASES = ("before", "entrance", "held", "flight", "vanish", "after")

# UI fields the backend stores but does not use yet: validate reports each one it finds (setting_not_used)
NOT_USED = {
    "advanced.track_smoothing": "Track smoothing",
    "advanced.export_layers": "The export layers choice (every render writes all layers)",
    "grade.spill": "Spill on occluders (Merge uses a fixed spill for now)",
    "grade.person_light": "Light on the person",
    "route[0].easing_in": "Easing on the first path point (no segment ends there)",
}
CAUSE = {  # cause kind of each cue.check code; every other code is about one field
    "no_track": "track", "track_invalid": "track", "track_uncertain": "track", "release_in_lost_gap": "track",
    "entrance_shortened": "timing", "window_grown": "timing", "window_cut_short": "timing",
    "window_trimmed": "timing", "orb_holds_after_route": "timing", "vanish_extended": "timing",
}


class CueFXError(Exception):
    """A call could not produce its output. `code` is a key of error_codes.json; `message` is plain language."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


# --------------------------------------------------------------------------- the contract files

@cache
def codes() -> dict[str, dict]:
    """error_codes.json: code -> {call, severity, message template}."""
    return json.loads((SCHEMA_DIR / "error_codes.json").read_text(encoding="utf-8"))["codes"]


def message(code: str, **params) -> str:
    """The plain-language message of `code`, its template filled with `params`."""
    return codes()[code]["message"].format(**params)


def error(code: str, **params) -> CueFXError:
    """A CueFXError with the code's message."""
    return CueFXError(code, message(code, **params))


@cache
def _validator(name: str):
    import jsonschema
    from referencing import Registry, Resource

    resources = [(SCHEMA_BASE + p.name, Resource.from_contents(json.loads(p.read_text(encoding="utf-8"))))
                 for p in SCHEMA_DIR.glob("*.schema.json")]
    registry = Registry().with_resources(resources)
    schema = registry.contents(SCHEMA_BASE + f"{name}.schema.json")
    return jsonschema.Draft202012Validator(schema, registry=registry)


def schema_errors(name: str, doc) -> list[tuple[str, str]]:
    """(field path, message) for every way `doc` breaks contract/schemas/<name>.schema.json; [] if it fits."""
    out = []
    for e in sorted(_validator(name).iter_errors(doc), key=lambda e: list(map(str, e.absolute_path))):
        path = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in e.absolute_path).lstrip(".")
        out.append((path or "(top level)", e.message))
    return out


def check_schema(name: str, doc) -> None:
    """Raise CueFXError(input_invalid) if `doc` does not fit its schema."""
    errs = schema_errors(name, doc)
    if errs:
        raise error("input_invalid", detail="; ".join(f"{p}: {m}" for p, m in errs[:5]) +
                    (f" (and {len(errs) - 5} more)" if len(errs) > 5 else ""))


# --------------------------------------------------------------------------- track: API form <-> cue form

def track_from_points(points: list[TrackPoint]) -> list[dict]:
    """The API track (track.schema.json) of a cue-file track: frame=f, confidence=conf, uncertain=lost."""
    return [{"frame": p.f, "x": p.x, "y": p.y, "confidence": min(max(p.conf, 0.0), 1.0), "uncertain": p.lost,
             "source": p.source, "palm_w": p.palm_w} for p in points]


def points_from_track(track: list[dict]) -> list[TrackPoint]:
    """Inverse of `track_from_points`. Without `source` a frame is tracked unless uncertain; without palm_w, 0."""
    return [TrackPoint(t["frame"], t["x"], t["y"], t.get("source", "interpolated" if t["uncertain"] else "tracked"),
                       t.get("palm_w", 0.0), t["confidence"], t["uncertain"]) for t in track]


def _cue(cue: dict, track: list[dict] | None) -> Cue:
    """The Cue of a contract cue dict, with `track` (API form) in place of its own track if given."""
    check_schema("cue", cue)
    if track is not None:
        check_schema("track", track)
    c = cue_from_dict(cue)
    return c if track is None else replace(c, track=points_from_track(track))


def _routable(cue: dict, track: list[dict]) -> Cue:
    """`_cue`, and CueFXError(input_invalid) if the cue has errors (nothing can be routed then)."""
    c = _cue(cue, track)
    errors = [i for i in check(c) if i.severity == "error"]
    if errors:
        raise error("input_invalid", detail="the cue has errors; run validate. " + "; ".join(i.message for i in errors[:3]))
    return c


# --------------------------------------------------------------------------- analyze

def analyze_result(points: list[TrackPoint], warnings: list[dict] | None = None, body_point: BodyPoint | None = None,
                   body_point_seed: HandSeed | None = None) -> dict:
    """analyze_result.schema.json of a finished track. `body_point` (v2.7) is the part and side that was tracked, for
    a body point other than the palm: the side may be the one the artist's click chose (`side_from_click`)."""
    detected = [p.palm_w for p in points if p.source == "tracked"]
    out = {
        "track": track_from_points(points),
        "summary": {"tracked": len(detected), "total": len(points),
                    "uncertain_ranges": [[a, b] for a, b in lost_runs(points)]},
        "hand_size_px": round(statistics.median(detected or [p.palm_w for p in points]), 1),
        "warnings": warnings or [],
    }
    if body_point is not None:
        out["body_point"] = {"part": body_point.part, "side": body_point.side}
    if body_point_seed is not None:
        out["body_point_seed"] = {"frame": body_point_seed.f, "x": body_point_seed.x, "y": body_point_seed.y}
    return out


def _probe_frames(clip: Path) -> int:
    """The frame count the container claims; CueFXError(clip_unreadable) if it cannot be opened."""
    import cv2

    cap = cv2.VideoCapture(str(clip))
    try:
        if not clip.is_file() or not cap.isOpened():
            raise error("clip_unreadable")
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def parse_body_point(raw: dict | BodyPoint | None, hand: str | None = None) -> BodyPoint | None:
    """The BodyPoint of a request's `body_point` ({part, side}), None if there is none (the palm of the cue's hand).

    The head has no side (it is stored as null). CueFXError(body_point_invalid) for an unknown part, a missing side,
    or a palm whose side is not `hand` (compared when `hand` is given).
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        if (not isinstance(raw.get("part"), str) or set(raw) - {"part", "side"}
                or not isinstance(raw.get("side"), (str, type(None)))):
            raise error("body_point_invalid")
        raw = BodyPoint(raw["part"], raw.get("side"))
    if body_point_issue(raw, hand):
        raise error("body_point_invalid")
    return BodyPoint(raw.part, None if raw.part == "head" else raw.side)


def body_point_label(bp: BodyPoint) -> str:
    """"right fist", "head": how a message names a body point."""
    return bp.part if bp.side is None else f"{bp.side} {bp.part}"


def track_lost_body_point(bp: BodyPoint, lost: int, total: int) -> CueFXError:
    """track_lost for a body point other than the palm: the message names it ("the right foot", "the head")."""
    text = codes()["track_lost"]["message_body_point"].format(body_point=body_point_label(bp), lost=lost, total=total)
    return CueFXError("track_lost", text)


def analyze(clip_path: Path, seed: dict | None, model: Path | None = None, landmarks_out: Path | None = None,
            body_point: dict | BodyPoint | None = None, hand: str | None = None, pose=None,
            pose_cache: Path | None = None) -> dict:
    """Track the hand the artist clicked (`seed` = {frame, x, y}, plate px) through the clip; analyze_result.

    `body_point` (v2.6, {part, side}): missing or "palm" is the hand tracker described here, unchanged. Another part
    (fist, foot, knee, shoulder, head) is tracked by `analyze_body_point` from the Pose Landmarker instead; `hand`
    (the cue's hand) is only used to check a palm body point's side. `pose` / `pose_cache`: see `analyze_body_point`.

    Wraps services.analysis.track.track_video. Raises CueFXError: clip_unreadable, clip_too_long, no_hand (no hand in the seed
    frame), track_lost (more than MAX_LOST_SHARE of the frames inside lost gaps), input_invalid (a bad seed).
    A click that misses the hand is not an error: the nearest hand is used and a seed_far warning says so.
    With `landmarks_out`, the hand landmarks Merge needs for behind ranges are written there (.npz), as
    scripts/track_clip.py writes landmarks.npz.
    """
    from services.analysis.track import TrackError, save_landmarks, track_video

    clip = Path(clip_path)
    n = _probe_frames(clip)
    if n > MAX_CLIP_FRAMES:
        raise error("clip_too_long", frames=n, max_frames=MAX_CLIP_FRAMES, max_seconds=MAX_CLIP_FRAMES // 24)
    bp = parse_body_point(body_point, hand)
    if seed is None:
        if is_palm(bp):
            raise error("input_invalid", detail="hand_seed is required for the palm")
        hs = None
    elif (isinstance(seed, dict) and all(isinstance(seed.get(k), (int, float)) and not isinstance(seed.get(k), bool)
                                        for k in ("frame", "x", "y")) and float(seed["frame"]).is_integer()):
        hs = HandSeed(int(seed["frame"]), float(seed["x"]), float(seed["y"]))
    else:
        raise error("input_invalid", detail="seed must be {frame, x, y} with numbers (frame a whole number)")
    if not is_palm(bp):
        return analyze_body_point(clip, hs, bp, pose, pose_cache, n)
    try:
        result, _, sel = track_video(clip, "right", hs, model=model)  # the hand label only names files; unused here
    except TrackError as e:
        if e.code == "no_hand":
            raise error("no_hand", frame=hs.f) from e
        if e.code == "input_invalid":
            raise error("input_invalid", detail=str(e)) from e
        raise error(e.code) from e
    if result.info.frames > MAX_CLIP_FRAMES:
        raise error("clip_too_long", frames=result.info.frames, max_frames=MAX_CLIP_FRAMES,
                    max_seconds=MAX_CLIP_FRAMES // 24)
    lost = sum(p.lost for p in result.points)
    if lost > MAX_LOST_SHARE * len(result.points):
        raise error("track_lost", lost=lost, total=len(result.points))
    if landmarks_out is not None:
        save_landmarks(result, Path(landmarks_out))
    warnings = [_warning("seed_far", (hs.f, hs.f + 1), {"kind": "track"}, {}, detail=w) for w in sel.warnings]
    return analyze_result(result.points, _unique_ids(warnings))


def analyze_body_point(clip: Path, seed: HandSeed | None, bp: BodyPoint, pose=None, pose_cache: Path | None = None,
                       frames: int | None = None) -> dict:
    """Track a body point (fist, foot, knee, shoulder, head) through the clip; analyze_result (task 32).

    The Pose Landmarker runs once per clip: `pose` (services.analysis.pose_track.PoseLandmarks) is used if given, else the landmarks
    in `pose_cache` (a .npz, written there on the first run). The track is built with the hand tracker's own gap, lost
    and smoothing rules; `palm_w` is the body point's size measure and `hand_size_px` its median. Raises CueFXError:
    clip_unreadable, clip_too_long, track_lost (the point is missing in more than MAX_LOST_SHARE of the frames),
    input_invalid (a seed frame outside the clip). A click far from the body point is the `seed_far_from_body_point`
    warning.
    """
    from services.analysis.pose_track import build_pose_track, cached_pose_landmarks, side_from_click
    from services.analysis.track import TrackError

    n = _probe_frames(clip) if frames is None else frames
    if n > MAX_CLIP_FRAMES:
        raise error("clip_too_long", frames=n, max_frames=MAX_CLIP_FRAMES, max_seconds=MAX_CLIP_FRAMES // 24)
    asked, warnings = bp, []
    try:
        pose = cached_pose_landmarks(clip, pose_cache) if pose is None else pose
        side = side_from_click(pose, bp.part, seed) if seed is not None else None
        if side is not None and side != bp.side:
            bp = BodyPoint(bp.part, side)
            warnings.append(_warning("side_from_click", (seed.f, seed.f + 1), {"kind": "track"},
                                     {"body_point": body_point_label(bp), "asked_side": asked.side}))
        result = build_pose_track(pose, bp, seed)
    except TrackError as e:
        if e.code == "input_invalid":
            raise error("input_invalid", detail=str(e)) from e
        if e.code == "track_lost":
            raise track_lost_body_point(bp, n, n) from e
        raise error(e.code) from e
    if result.info.frames > MAX_CLIP_FRAMES:
        raise error("clip_too_long", frames=result.info.frames, max_frames=MAX_CLIP_FRAMES,
                    max_seconds=MAX_CLIP_FRAMES // 24)
    lost = sum(p.lost for p in result.points)
    if lost > MAX_LOST_SHARE * len(result.points):
        raise track_lost_body_point(bp, lost, len(result.points))
    if seed is not None:
        warnings += [_warning("seed_far_from_body_point", (seed.f, seed.f + 1), {"kind": "track"},
                              {"body_point": body_point_label(bp)}, detail=w) for w in result.warnings]
        return analyze_result(result.points, _unique_ids(warnings), bp)
    tracked = [p for p in result.points if p.source == "tracked"]
    sample = max(tracked, key=lambda p: (p.conf, -p.f))
    return analyze_result(result.points, _unique_ids(warnings), bp, HandSeed(sample.f, sample.x, sample.y))


# --------------------------------------------------------------------------- preview

def phase_at(cue: Cue, f: int) -> str:
    """before | entrance | held | flight | vanish | after; the first rule that fits wins. A "hold" cue has no flight:
    it is held until the vanish starts."""
    if f < entrance_start(cue):
        return "before"
    release = cue.vanish.start if no_flight(cue) else cue.release_frame
    for name, end in (("entrance", cue.appear_frame), ("held", release), ("flight", cue.vanish.start),
                      ("vanish", vanish_of(cue).end)):
        if f < end:
            return name
    return "after"


def preview_path(cue: dict, track: list[dict], with_held: bool = False) -> list[dict]:
    """[{frame, x, y, diameter, visibility, visible, phase}] for every frame 0..frames-1 (preview.schema.json).

    Inside the effect window every value comes from the render's own code: orb_path (x, y, diameter, visibility)
    and entrance_gain (the drawn diameter is diameter x gain, as the layout draws it). Outside it nothing is drawn.
    `with_held` (v2.3) adds `held {x, y}` to every entry: the corrected track plus anchor_offset (route.held_xy),
    for the UI's hand marker. Raises CueFXError(input_invalid) if the cue or track does not fit its schema or the
    cue has errors.
    """
    c = _routable(cue, track)
    held = held_xy(c) if with_held else None
    path = orb_path(c)
    by_f = {s[0]: s for s in path}
    first, last = path[0], path[-1]
    out = []
    for f in range(c.frames):
        s = by_f.get(f)
        if s is None:
            _, x, y, _, _ = first if f < first[0] else last
            d = vis = 0.0
        else:
            _, x, y, d, vis = s
            d *= entrance_gain(c, f)
        out.append({"frame": f, "x": round(x, 3), "y": round(y, 3), "diameter": round(d, 3),
                    "visibility": round(vis, 4), "visible": d > 0 and vis > 0, "phase": phase_at(c, f)})
        if held is not None:
            out[-1]["held"] = {"x": round(float(held[f, 0]), 3), "y": round(float(held[f, 1]), 3)}
    return out


# --------------------------------------------------------------------------- validate

def _warning(code: str, frames: tuple[int, int] | None, cause: dict, params: dict, detail: str | None = None,
             fix: dict | None = None) -> dict:
    """A warning.schema.json object; `first`/`last` (inclusive, for display) are filled from `frames`."""
    if frames is not None:
        params = {"first": frames[0], "last": frames[1] - 1, **params}
    w = {"id": code if frames is None else f"{code}_{frames[0]}", "severity": codes()[code]["severity"], "code": code,
         "message": message(code, **params), "frames": None if frames is None else list(frames),
         "jump": None if frames is None else frames[0], "cause": cause}
    if detail is not None:
        w["detail"] = detail
    if fix is not None:
        w["fix"] = fix
    return w


def _unique_ids(warnings: list[dict]) -> list[dict]:
    """Make ids unique within one response by suffixing repeats with _2, _3, ..."""
    seen: dict[str, int] = {}
    for w in warnings:
        seen[w["id"]] = seen.get(w["id"], 0) + 1
        if seen[w["id"]] > 1:
            w["id"] = f"{w['id']}_{seen[w['id']]}"
    return warnings


def _issue_warning(cue: Cue, i: Issue) -> dict:
    """The contract warning of one cue.check issue."""
    path, _, _ = i.message.partition(":")
    params: dict = {"path": path}
    fix = None
    cause = {"kind": CAUSE.get(i.code, "field")}
    if cause["kind"] == "field":
        cause["path"] = path
    if i.code == "track_uncertain":
        a, b = i.frames
        k = next((k for k, r in enumerate(cue.behind) if a < r.end and r.start < b), None)
        params["reason"] = "the other hand crosses in front" if k is not None else "the tracker lost the palm"
        if k is not None:
            cause = {"kind": "pass", "index": k}
    if i.code == "hold_path_unused" and cue.behaviour == "orbit":
        w = _warning(i.code, i.frames, cause, params, detail=i.message)
        w["message"] = codes()[i.code]["message_orbit"]
        return w
    if i.code == "vanish_extended":
        params.update(ending=cue.ending, seconds=f"{ENDING_MIN_SECONDS[cue.ending]:g}")
    if i.code == "entrance_shortened":
        secs = round(cue.appear_frame / cue.fps, 2)
        params["seconds"] = f"{secs:g}"
        fix = {"label": f"Use {secs:g} s", "op": "set", "path": "entrance.seconds", "value": secs}
    w = _warning(i.code, i.frames, cause, params, detail=i.message, fix=fix)
    if i.frames is None and cause["kind"] == "field":
        w["id"] = f"{i.code}_{path}"
    return w


def _present(cue: dict, dotted: str) -> bool:
    """True if the dotted path (e.g. advanced.track_smoothing, route[0].easing_in) is set to a value that matters."""
    if dotted == "route[0].easing_in":
        route = cue.get("route") or []
        return bool(route) and route[0].get("easing_in", "linear") != "linear"
    head, _, key = dotted.partition(".")
    block = cue.get(head)
    if not isinstance(block, dict) or block.get(key) is None:
        return False
    return not (dotted == "grade.person_light" and block[key] == 0)


def validate(cue: dict, track: list[dict] | None = None) -> list[dict]:
    """Every problem with the cue as warnings (warning.schema.json), errors first. Never raises for a bad cue.

    Schema problems come back alone, as severity error, code invalid_field, with the field path in `cause`. Otherwise:
    the backend's own checks (services.cue.cue.check, the same rules the render applies), then the checks only the app
    needs: no_path, check_pass_edge, setting_not_used (for each UI field stored but not used yet) and unknown_field.
    `track` (API form) replaces the cue's own track when given.
    """
    errs = schema_errors("cue", cue) + ([("track" + p.removeprefix("(top level)"), m)
                                         for p, m in schema_errors("track", track)] if track is not None else [])
    if errs:
        return _unique_ids([_warning("invalid_field", None, {"kind": "field", "path": p}, {"path": p},
                                     detail=f"{p}: {m}") for p, m in errs])
    c = cue_from_dict(cue)
    if track is not None:
        c = replace(c, track=points_from_track(track))
    out = [_issue_warning(c, i) for i in check(c)]
    if not no_flight(c) and c.route is not None and len(c.route) == 1 and c.release_frame is not None:
        out.append(_warning("no_path", (c.release_frame, c.release_frame + 1), {"kind": "tool-draw"}, {}))
    for k, b in enumerate(c.behind):
        if b.start < b.end:
            edge = max(b.start, b.end - 2)
            out.append(_warning("check_pass_edge", (edge, b.end), {"kind": "pass", "index": k}, {}))
    for dotted, label in NOT_USED.items():
        if _present(cue, dotted):
            out.append(_warning("setting_not_used", None, {"kind": "setting", "path": dotted}, {"label": label}))
            out[-1]["id"] = f"setting_not_used_{dotted}"
    known = _validator("cue").schema["properties"]
    for key in cue:
        if key not in known:
            out.append(_warning("unknown_field", None, {"kind": "field", "path": key}, {"path": key}))
            out[-1]["id"] = f"unknown_field_{key}"
    order = {"error": 0, "warning": 1, "info": 2}
    return _unique_ids(sorted(out, key=lambda w: order[w["severity"]]))


# --------------------------------------------------------------------------- new cue, normalize

# fields a new cue takes from the contract's canonical example (golden/throw_71319.cue.json): the backend has no rule
# that picks timings, size or look for a clip, so these are starting values the artist edits, not derived ones
STARTER_FIELDS = ("canvas_size", "anchor_offset", "appear_frame", "release_frame", "diameter_px", "vanish", "look",
                  "seed", "grade", "entrance")  # timings, size and offset: only normalize's fallback; new_cue sets them
STARTER_CUE = CONTRACT_DIR / "golden" / "throw_71319.cue.json"


@cache
def _starter() -> dict:
    d = json.loads(STARTER_CUE.read_text(encoding="utf-8"))
    return {k: d[k] for k in STARTER_FIELDS}


def _defaults(cue: dict) -> dict:
    """Values for the fields a cue needs but does not have: the starter fields, no conform (transform identity), the
    canvas on the whole plate (BRIEF section 4: "full frame now"), no corrections or behind ranges, and a route that
    is only the release point."""
    w, h = cue["plate_size"]
    out = {"transform": {"scale": 1.0, "offset": [0, 0]}, "canvas_rect": [0, 0, w, h], "corrections": [],
           "behind": [], **copy.deepcopy(_starter())}
    if cue.get("behaviour") in NO_FLIGHT:  # no release and no flight path
        out["release_frame"], out["route"] = None, []
        return out
    release = cue.get("release_frame", out["release_frame"])
    diameter = cue.get("diameter_px", out["diameter_px"])
    out["route"] = [{"f": release, "x": None, "y": None, "diameter_px": diameter, "easing_in": "linear"}]
    return out


def derived_effect_range(cue: Cue) -> EffectRange | None:
    """effect_range from the timings, by services.cue.window.propose_effect_range (appear_frame - 4 .. vanish.end, 8k+1,
    inside the clip; the entrance is covered later by Make, `effect_window`, with a window_grown info). None if the
    timings it needs are missing or cover nothing (validate reports those)."""
    from services.cue.window import propose_effect_range  # here: services.cue.window imports services.cue.cue, as api does

    if cue.appear_frame is None or cue.vanish is None:
        return None
    try:
        er, _ = propose_effect_range(cue.appear_frame, cue.vanish.end, cue.frames)
    except ValueError:
        return None
    return er


def computed_effect_range(cue: dict, track: list[dict] | None = None) -> dict | None:
    """The `effect_range` {start, end} that `normalize` would give this cue (`derived_effect_range`), or None when the
    cue does not fit its schema or its timings cover nothing. For the validate route, whose answer is a list."""
    if schema_errors("cue", cue) or (track is not None and schema_errors("track", track)):
        return None
    c = cue_from_dict(cue)
    er = derived_effect_range(c)
    return None if er is None else {"start": er.start, "end": er.end}


def defaults() -> dict:
    """Every default the UI needs, read from the backend's own constants (GET /api/defaults, task 36)."""
    return {"spin_deg_per_s": {k: spec.spin for k, spec in SHAPE_SPECS.items()},
            "ending_seconds": dict(ENDING_MIN_SECONDS), "ending_default_seconds": dict(ENDING_DEFAULT_SECONDS),
            "charge_up": dict(CHARGE_UP), "orbit": dict(ORBIT_DEFAULTS),
            "max_window_frames": MAX_EFFECT_FRAMES, "orbit_tilt_deg_by_part": dict(ORBIT_TILT_BY_PART)}


def normalize(cue: dict) -> dict:
    """`normalize_with_warnings`, without the warnings."""
    return normalize_with_warnings(cue)[0]


def normalize_with_warnings(cue: dict) -> tuple[dict, list[dict]]:
    """The cue with every derived field recomputed by the backend, so the UI never computes one itself:

    - `version` (a v2.1 cue is upgraded to "2.2" as on load; v2.2 and v2.3 cues keep theirs, except that a "hold"
      cue is "2.3"), `route[].easing_in` "linear" where missing;
    - missing fields filled (`_defaults`; a "hold" cue gets `release_frame` null and `route` []); present ones are
      kept as they are;
    - `effect_range` from appear_frame and vanish (`derived_effect_range`); kept if they are missing;
    - `vanish.end` moved later where a burst or shatter `ending` needs more frames than the vanish has (v2.6): the
      warning `vanish_extended` says so (`window_trimmed` instead where the 8 s window or the clip ends first: the
      vanish then ends where the window does). A fade ending never changes the vanish.
    The warnings are the second result; the cue is the same as before for a cue without those fields. The input is not changed. CueFXError(input_invalid) if the cue does not fit its schema. A cue with value errors
    comes back normalized as far as possible; validate reports what is left.
    """
    check_schema("cue", cue)
    out = copy.deepcopy(cue)
    for k, v in _defaults(out).items():
        if k not in out:  # an explicit null (look, seed) is the artist's choice and is kept
            out[k] = v
    try:
        c = cue_from_dict(out)
    except (TypeError, ValueError) as e:
        raise error("input_invalid", detail=str(e)) from e
    warnings = []
    if c.vanish is not None and c.vanish.start < c.vanish.end and ending_of(c) in ("burst", "shatter"):
        wanted = vanish_of(c)
        if wanted.end > c.vanish.end:
            old_end, c = c.vanish.end, replace(c, vanish=wanted)
            er = derived_effect_range(c)
            if er is not None and er.end < wanted.end:  # the window cannot hold the whole ending: cut it, as `validate` says
                c = replace(c, vanish=replace(c.vanish, end=max(er.end, c.vanish.start + 1)))
                warnings.append(_warning("window_trimmed", (c.vanish.end, wanted.end), {"kind": "timing"}, {}))
            else:
                warnings.append(_warning("vanish_extended", (old_end, wanted.end), {"kind": "timing"},
                                         {"ending": c.ending, "seconds": f"{ENDING_MIN_SECONDS[c.ending]:g}"}))
    er = derived_effect_range(c)
    if er is not None:
        c = replace(c, effect_range=er)
    oldest = min_version(c)  # a "hold" cue is 2.3, one that sets body_point 2.4, one that sets shape or spin 2.5
    if oldest is not None and c.version in SUPPORTED_VERSIONS and SUPPORTED_VERSIONS.index(c.version) < SUPPORTED_VERSIONS.index(oldest):
        c = replace(c, version=oldest)
    return cue_to_dict(c), warnings


def new_cue_with_warnings(clip: str, fps: float, frames: int, plate_size: list[int], hand: str, hand_seed: dict,
                          track: list[dict], transform: dict | None = None, body_point: dict | None = None
                          ) -> tuple[dict, list[dict]]:
    """`new_cue` and its warnings (warning.schema.json): `placeholder_timings` when the track has no steady palm and
    the starting timings are fractions of the clip; `started_as_hold` (info) when a steady palm has no throw after it
    and the cue starts as a "hold", plus `hold_shortened` (info) when that hold was cut to the window limit.
    `body_point` (v2.6): the cue gets it (version 2.4) and the default size and offset of its part; the timings rule
    is the same, read on the body-point track. Without it everything is as before."""
    check_schema("track", track)
    bp = parse_body_point(body_point, hand)
    points = points_from_track(track)
    # the conformed picture is the plate minus the black padding the conform added: the centred offset on each side
    ox, oy = (0, 0) if transform is None else (max(transform["offset"][0], 0), max(transform["offset"][1], 0))
    picture = (ox, oy, plate_size[0] - ox, plate_size[1] - oy)
    t = start_timings(points, fps, frames, part="palm" if bp is None else bp.part, plate_size=tuple(plate_size),
                      picture=picture)
    base = {"version": VERSION, "clip": clip, "fps": int(fps) if float(fps).is_integer() else fps, "frames": frames,
            "plate_size": list(plate_size), "hand": hand,
            "hand_seed": {"f": int(hand_seed["f"]), "x": hand_seed["x"], "y": hand_seed["y"]},
            "track": [asdict(p) for p in points],
            "behaviour": t.behaviour, "appear_frame": t.appear_frame, "release_frame": t.release_frame,
            "diameter_px": t.diameter_px, "anchor_offset": list(t.anchor_offset),
            "vanish": {"start": t.vanish_start, "end": t.vanish_end}}
    if t.route_end is not None:  # the strike rule's flight; every other start has only the release point (normalize adds it)
        f, x, y = t.route_end
        base["route"] = [{"f": t.release_frame, "x": None, "y": None, "diameter_px": t.diameter_px, "easing_in": "linear"},
                         {"f": f, "x": x, "y": y, "diameter_px": t.diameter_px, "easing_in": "linear"}]
    if transform is not None:
        base["transform"] = {"scale": transform["scale"], "offset": list(transform["offset"])}
    if bp is not None:
        base["body_point"] = {"part": bp.part, "side": bp.side}
    warnings = [_warning("placeholder_timings", (t.appear_frame, t.release_frame), {"kind": "timing"}, {})] if t.fallback else []
    if t.behaviour == "hold":
        warnings.append(_warning("started_as_hold", (t.appear_frame, t.vanish_start), {"kind": "timing"}, {}))
        if t.hold_shortened:
            warnings.append(_warning("hold_shortened", (t.vanish_start, t.vanish_end), {"kind": "timing"}, {}))
    return normalize(base), warnings


def new_cue(clip: str, fps: float, frames: int, plate_size: list[int], hand: str, hand_seed: dict,
            track: list[dict], transform: dict | None = None, body_point: dict | None = None) -> dict:
    """A full, normalized v2.3 cue for a clip just analyzed: the analyze `track` (API form, stored in cue form),
    `hand`, `hand_seed` {f, x, y}, the starting timings and size from the clip's own track (`services.cue.start_timings`),
    the other starter values (`STARTER_FIELDS`), the release point as the only route point and no behind ranges.
    `behaviour` is "hold_throw", or "hold" (no release, empty route) where the clip has a steady palm and no throw.
    `transform` ({scale, offset}, `services.analysis.conform.transform_for`) is how the upload was conformed to the plate;
    identity when left out. `body_point` ({part, side}, v2.6): see `new_cue_with_warnings`.
    CueFXError(input_invalid) if the track does not fit its schema, body_point_invalid for a bad `body_point`."""
    return new_cue_with_warnings(clip, fps, frames, plate_size, hand, hand_seed, track, transform, body_point)[0]


# --------------------------------------------------------------------------- timing

def propose_timing(cue: dict, track: list[dict]) -> dict:
    """A copy of `cue` with route[1:-1] re-timed by services.cue.route.propose_route_timing (constant speed in the room,
    pinhole camera); route[0] stays on release_frame and the last point keeps its frame. The input is not changed.
    A route of fewer than 2 points (or a "hold" cue) comes back as it is (validate reports no_path). CueFXError(input_invalid) if the
    cue has errors or the points do not fit into their frames."""
    c = _routable(cue, track)
    out = copy.deepcopy(cue)
    if no_flight(c) or c.route is None or len(c.route) < 2:
        return out
    pts = [(*release_point(c), c.route[0].diameter_px)] + [(p.x, p.y, p.diameter_px) for p in c.route[1:]]
    try:
        frames = propose_route_timing(pts, c.release_frame, c.route[-1].f)
    except ValueError as e:
        raise error("input_invalid", detail=str(e)) from e
    for p, f in zip(out["route"], frames):
        p["f"] = f
    return out
