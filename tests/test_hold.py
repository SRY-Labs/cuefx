"""Task 33: the Hold behaviour (cue v2.3), the 193-frame (8 s) window, and the starting timings that choose between
Hold and Hold + throw."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from conftest import PALM_W, make_cue
from services.cue import api
from services.cue.cue import (MAX_EFFECT_FRAMES, Behind, EffectRange, RoutePoint, TrackPoint, Vanish, behaviour_of,
                       cue_from_dict, cue_to_dict, validate)
from services.cue.make_key import make_key
from services.cue.route import held_xy, orb_path
from services.cue.start_timings import start_timings, steady_runs
from services.cue.window import effect_window, propose_effect_range

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "contract" / "golden"
HOLD_MESSAGE = "This effect is set to Hold, so the flight path is not used."
LIMIT_MESSAGE = "The effect can run up to 8 seconds (193 frames). Move the vanish earlier or remove late path points."


def hold_cue(**changes):
    """The synthetic cue as a Hold: no release, no route (unless `changes` leaves one in)."""
    return make_cue(**{"behaviour": "hold", "release_frame": None, "route": [], **changes})


def long_clip(frames: int) -> dict:
    return dict(frames=frames, track=[TrackPoint(f, 1000.0, 600.0, "tracked", PALM_W, 0.9) for f in range(frames)])


def codes_of(cue) -> set[str]:
    return {w["code"] for w in api.validate(cue_to_dict(cue))}


# --------------------------------------------------------------------------- old files are hold_throw

def test_a_v22_cue_loads_as_hold_throw_and_is_written_back_unchanged():
    d = json.loads((GOLDEN / "throw_71319.cue.json").read_text(encoding="utf-8"))
    assert d["version"] == "2.2" and "behaviour" not in d
    cue = cue_from_dict(d)
    assert cue.behaviour is None and behaviour_of(cue) == "hold_throw" and cue.version == "2.2"
    assert cue_to_dict(cue) == d                       # nothing added, nothing lost
    assert api.normalize(d) == d                       # the golden cue is its own normalization
    track = json.loads((GOLDEN / "throw_71319.track.json").read_text(encoding="utf-8"))
    assert api.validate(d, track) == json.loads((GOLDEN / "throw_71319.validate.json").read_text(encoding="utf-8"))


def test_a_cue_without_a_behaviour_is_hold_throw_for_every_step(tmp_path):
    look = tmp_path / "look.png"
    look.write_bytes(b"look image bytes")
    plain, explicit = make_cue(), make_cue(behaviour="hold_throw")
    assert behaviour_of(plain) == behaviour_of(explicit) == "hold_throw"
    assert make_key(plain, look) == make_key(explicit, look) == "d5fa87152707"
    assert orb_path(plain) == orb_path(explicit)


def test_a_bad_behaviour_is_an_error():
    w = [w for w in api.validate(cue_to_dict(make_cue(behaviour="swing"))) if w["code"] == "behaviour_invalid"]
    assert len(w) == 1 and w[0]["severity"] == "error" and w[0]["cause"] == {"kind": "field", "path": "behaviour"}
    assert api.schema_errors("warning", w[0]) == []


# --------------------------------------------------------------------------- validate

def test_a_hold_cue_has_no_release_or_route_errors_or_warnings():
    cue = hold_cue()
    assert [w for w in api.validate(cue_to_dict(cue)) if w["severity"] != "info"] == []
    assert cue_to_dict(cue)["release_frame"] is None and cue_to_dict(cue)["route"] == []
    errors, warnings = validate(cue)
    assert errors == [] and warnings == []
    # the same cue as hold_throw would be an error ("route: empty") and a missing release
    assert any("route" in e for e in validate(replace(cue, behaviour="hold_throw"))[0])


def test_a_hold_does_not_check_where_the_release_would_be():
    lost = [TrackPoint(f, 1000.0, 600.0, "interpolated" if 100 <= f < 110 else "tracked", PALM_W, 0.9, lost=100 <= f < 110)
            for f in range(200)]
    held = hold_cue(track=lost)
    assert "release_in_lost_gap" not in codes_of(held)
    assert "release_in_lost_gap" in codes_of(make_cue(track=lost, release_frame=105, route=[
        RoutePoint(105, None, None, 110), RoutePoint(116, 700, 420, 60)]))
    assert "track_uncertain" in codes_of(held)  # a lost track while held is still reported (the orb slides there)


def test_a_leftover_route_or_release_is_one_plain_warning_and_not_used():
    for cue in (hold_cue(route=[RoutePoint(107, None, None, 110), RoutePoint(116, 700, 420, 60)]),
                hold_cue(release_frame=107)):
        w = [w for w in api.validate(cue_to_dict(cue)) if w["code"] == "hold_path_unused"]
        assert len(w) == 1 and w[0]["severity"] == "warning" and w[0]["message"] == HOLD_MESSAGE
        assert api.schema_errors("warning", w[0]) == []
        assert not {"no_path", "orb_holds_after_route", "route_start"} & codes_of(cue)
    assert "hold_path_unused" not in codes_of(hold_cue())
    # and the path is not used: a leftover route does not move the orb
    left = hold_cue(route=[RoutePoint(107, None, None, 110), RoutePoint(116, 700, 420, 60)], release_frame=107)
    assert orb_path(left) == orb_path(hold_cue())


def test_the_other_checks_still_apply_to_a_hold():
    assert "frame_outside_effect_range" in codes_of(hold_cue(appear_frame=20))
    assert "effect_range_length" in codes_of(hold_cue(effect_range=EffectRange(32, 154)))
    assert "range_outside_effect" in codes_of(hold_cue(behind=[Behind(20, 40, "left_hand")]))
    assert "missing_field" in codes_of(hold_cue(vanish=None))
    assert "release_not_after_appear" not in codes_of(hold_cue(appear_frame=107, release_frame=100))


# --------------------------------------------------------------------------- the window limit

@pytest.mark.parametrize("hold", [False, True])
def test_a_window_of_193_is_allowed_and_201_gets_the_8_second_message(hold):
    make = (lambda **c: hold_cue(**c)) if hold else make_cue
    ok = make(effect_range=EffectRange(32, 225), vanish=Vanish(200, 210), **long_clip(300))
    assert MAX_EFFECT_FRAMES == 193 and ok.effect_range.length == 193
    assert "window_trimmed" not in codes_of(ok)
    over = make(effect_range=EffectRange(32, 233), vanish=Vanish(200, 210), **long_clip(300))   # 201 frames
    w = [w for w in api.validate(cue_to_dict(over)) if w["code"] == "window_trimmed"]
    assert len(w) == 1 and w[0]["message"] == LIMIT_MESSAGE and w[0]["severity"] == "warning"
    assert (w[0]["frames"], w[0]["cause"]["kind"]) == ([225, 233], "timing")   # degrade: the last 8 frames get no effect
    assert [s[0] for s in orb_path(over)] == list(range(32, 225))


def test_a_window_of_121_or_less_keeps_its_make_key_and_a_longer_one_has_a_new_one(tmp_path):
    look = tmp_path / "look.png"
    look.write_bytes(b"look image bytes")
    assert make_key(make_cue(effect_range=EffectRange(32, 153)), look) == "d5fa87152707"   # the key it always had
    long = make_cue(effect_range=EffectRange(32, 161), vanish=Vanish(150, 160))   # 129 frames: no longer trimmed to 121
    assert make_key(long, look) != "d5fa87152707" and make_key(long, look) == make_key(replace(long), look)


# --------------------------------------------------------------------------- the path, the key, the maths

def test_a_hold_stays_on_the_hand_in_every_frame_of_the_window():
    cue = hold_cue()
    cue = replace(cue, track=[TrackPoint(f, 1000.0 + 5 * f, 600.0 + f, "tracked", PALM_W, 0.9) for f in range(200)])
    path = orb_path(cue)
    held = held_xy(cue)
    assert [s[0] for s in path] == list(range(32, 153))
    for f, x, y, d, vis in path:
        assert (x, y, d) == (held[f, 0], held[f, 1], cue.diameter_px)       # never released, never resized
    assert [round(s[4], 3) for s in path if s[0] in (128, 129, 130, 131)] == [1.0, round(2 / 3, 3), round(1 / 3, 3), 0.0]


def test_a_hold_corrections_follow_like_a_held_orb():
    from services.cue.cue import Correction
    cue = hold_cue(corrections=[Correction(100, 1200.0, 500.0)], vanish=Vanish(140, 146), effect_range=EffectRange(32, 153))
    by_f = {s[0]: s for s in orb_path(cue)}
    assert (by_f[100][1], by_f[100][2]) == (1200.0, 500.0 - 70)              # the correction + anchor_offset, mid-window


def test_the_hold_make_key_differs_and_every_hold_throw_key_stays(tmp_path):
    look = tmp_path / "look.png"
    look.write_bytes(b"look image bytes")
    throw = make_cue()
    hold = replace(throw, behaviour="hold", release_frame=None, route=[])
    assert make_key(throw, look) == "d5fa87152707"
    assert make_key(hold, look) != make_key(throw, look)
    assert make_key(hold, look) == make_key(replace(hold, route=[]), look)
    assert make_key(replace(hold, release_frame=107, route=[RoutePoint(107, None, None, 110)]), look) != make_key(hold, look)


def test_preview_path_of_a_hold_has_no_flight_phase():
    cue = cue_to_dict(hold_cue())
    track = api.track_from_points(hold_cue().track)
    out = api.preview_path(cue, track, with_held=True)
    assert {p["phase"] for p in out} == {"before", "held", "vanish", "after"}   # no entrance (0 s), no flight
    assert api.schema_errors("preview", out) == []
    inside = [p for p in out if 40 <= p["frame"] < 131]
    assert all((p["x"], p["y"]) == (p["held"]["x"], p["held"]["y"]) for p in inside)
    assert api.propose_timing(cue, track) == cue


def test_normalize_handles_a_hold():
    cue = cue_to_dict(hold_cue())
    del cue["effect_range"], cue["canvas_rect"]
    out = api.normalize(cue)
    assert out["version"] == "2.3" and out["behaviour"] == "hold"
    assert out["release_frame"] is None and out["route"] == []
    er, _ = propose_effect_range(40, 131, 200)                               # appear_frame .. vanish.end, as for a throw
    assert out["effect_range"] == {"start": er.start, "end": er.end}
    assert api.normalize(out) == out
    assert api.schema_errors("cue", out) == []
    bare = {k: v for k, v in cue.items() if k not in ("release_frame", "route")}  # missing, not null
    again = api.normalize(bare)
    assert again["release_frame"] is None and again["route"] == []


def test_a_hold_cue_is_written_as_2_3():
    d = cue_to_dict(hold_cue(version="2.3"))
    assert d["behaviour"] == "hold" and "release_frame" in d and d["release_frame"] is None
    assert cue_from_dict(d) == hold_cue(version="2.3")


# --------------------------------------------------------------------------- the stand-in draws it

def test_the_standin_draws_a_hold_on_the_hand_in_every_frame():
    from services.make.look import standin_look
    from services.make.standin_effect import prepare_look, render_standin
    cue = hold_cue()
    path = orb_path(cue)[:12]
    look = prepare_look(standin_look())[0]
    effect, matte = render_standin(look, path, (0, 0, 1920, 1080), (256, 144))
    assert len(effect) == len(matte) == 12
    assert all(e.max() > e.min() for e in effect)                           # something is drawn in every frame


# --------------------------------------------------------------------------- starting timings

def steady(n: int, throw_at: int | None = None, palm_w: float = 80.0) -> list[TrackPoint]:
    """Still from frame 1; from `throw_at` on 2.5 palm widths per frame."""
    x, out = 600.0, []
    for f in range(n):
        x += 2.5 * palm_w if throw_at is not None and f >= throw_at else 0.0
        out.append(TrackPoint(f, x, 500.0, "tracked", palm_w, 0.9))
    return out


def cue_for(points, frames=None, fps=24):
    return api.new_cue_with_warnings("c.mov", fps, frames or len(points), [1920, 1080], "right",
                                     {"f": 5, "x": 600, "y": 500}, api.track_from_points(points))


def errors(cue):
    return [w for w in api.validate(cue) if w["severity"] == "error"]


def test_a_steady_run_with_no_throw_starts_as_hold():
    pts = steady(100)
    t = start_timings(pts, 24, 100)
    assert (t.behaviour, t.release_frame, t.fallback, t.hold_shortened) == ("hold", None, False, False)
    assert (t.appear_frame, t.vanish_start, t.vanish_end) == (13, 99, 100)  # the run reaches the end: vanish.start is the last frame
    cue, warnings = cue_for(pts)
    assert cue["behaviour"] == "hold" and cue["release_frame"] is None and cue["route"] == []
    assert cue["version"] == "2.3" and errors(cue) == [] and api.schema_errors("cue", cue) == []
    assert [w["code"] for w in warnings] == ["started_as_hold"]
    w = warnings[0]
    assert w["severity"] == "info" and api.schema_errors("warning", w) == []
    assert w["message"] == "No throw found in this clip, so the effect starts as Hold. Switch to Hold + throw to draw a flight path."
    assert [x["code"] for x in api.validate(cue) if x["code"] in ("no_path", "hold_path_unused")] == []


def test_the_hold_vanishes_right_after_the_run_when_the_clip_goes_on():
    pts = [TrackPoint(p.f, p.x + (0 if p.f < 71 else 20.0 * (p.f - 70)), p.y, p.source, p.palm_w, p.conf)
           for p in steady(150)]                                            # still 1..70, then 0.25 palm widths a frame
    assert steady_runs(pts)[0] == (1, 70)
    t = start_timings(pts, 24, 150)
    assert (t.behaviour, t.appear_frame, t.vanish_start, t.vanish_end) == ("hold", 13, 71, 77)   # 0.25 s long
    assert errors(cue_for(pts)[0]) == []


def test_a_hold_longer_than_8_seconds_is_cut_to_193_frames_with_the_shortened_info():
    pts = steady(400)                                                       # a 400-frame hold, 16.7 s
    t = start_timings(pts, 24, 400)
    assert t.behaviour == "hold" and t.hold_shortened and t.appear_frame == 13   # appear kept, the end cut
    cue, warnings = cue_for(pts)
    assert [w["code"] for w in warnings] == ["started_as_hold", "hold_shortened"]
    assert warnings[1]["severity"] == "info" and warnings[1]["message"] == "The hold was shortened to the 8-second limit."
    assert api.schema_errors("warning", warnings[1]) == []
    assert errors(cue) == [] and effect_window(cue_from_dict(cue)).length == 193   # what Make draws, entrance included
    assert cue["vanish"]["end"] - cue["vanish"]["start"] == 6 and cue["vanish"]["end"] <= cue["effect_range"]["end"]
    assert not {"window_trimmed", "window_cut_short"} & {w["code"] for w in api.validate(cue)}
    assert cue["vanish"]["start"] > cue["appear_frame"]


def test_a_throw_found_is_hold_throw_as_before():
    pts = steady(150, throw_at=71)
    t = start_timings(pts, 24, 150)
    assert (t.behaviour, t.release_frame, t.hold_shortened) == ("hold_throw", 71, False)
    assert (t.appear_frame, t.vanish_start, t.vanish_end) == (13, 77, 83)
    cue, warnings = cue_for(pts)
    assert cue["behaviour"] == "hold_throw" and cue["release_frame"] == 71 and warnings == []


def test_no_steady_run_is_still_the_placeholder_hold_throw():
    pts = [TrackPoint(f, 600.0 + 200 * f, 500.0, "tracked", 80.0, 0.9) for f in range(120)]
    t = start_timings(pts, 24, 120)
    assert t.fallback and t.behaviour == "hold_throw" and (t.appear_frame, t.release_frame) == (30, 72)
    assert [w["code"] for w in cue_for(pts)[1]] == ["placeholder_timings"]


def test_a_hold_on_the_golden_track_starts_after_the_hold_ends_when_cut_at_the_throw():
    golden = json.loads((GOLDEN / "throw_71319.track.json").read_text(encoding="utf-8"))
    cue, warnings = api.new_cue_with_warnings("throw_71319.mov", 24, 193, [1920, 1080], "right",
                                              {"f": 60, "x": 1160, "y": 550}, golden)
    assert warnings == [] and cue["behaviour"] == "hold_throw" and cue["release_frame"] == 107
    assert np.isfinite(cue["appear_frame"])
