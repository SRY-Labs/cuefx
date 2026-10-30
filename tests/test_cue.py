"""Cue v2.1: round trip and every validation rule, on a synthetic cue."""
from dataclasses import replace

import pytest

from conftest import PALM_W, make_cue
from services.cue.cue import (Behind, Correction, EffectRange, Grade, RoutePoint, TrackPoint, Vanish, cue_from_dict,
                       cue_to_dict, load_cue, lost_runs, open_lost_runs, save_cue, validate)


def with_track(*runs: tuple[int, int]) -> list[TrackPoint]:
    """200 tracked frames at x = 10 f, with each run [a, b) interpolated and flagged lost."""
    pts = [TrackPoint(f, 10.0 * f, 0.0, "tracked", PALM_W, 0.9) for f in range(200)]
    for a, b in runs:
        pts[a:b] = [replace(p, source="interpolated", lost=True) for p in pts[a:b]]
    return pts


def test_valid_cue_has_no_errors_and_no_warnings(cue):
    assert validate(cue) == ([], [])


def test_full_cue_roundtrips_without_loss(cue, tmp_path):
    cue.corrections = [Correction(84, 1002.5, 598.25)]
    cue.behind = [Behind(80, 91, "left_hand"), Behind(95, 99, {"point": [95, 10.5, 20]})]
    cue.extra = {"ui_note": "kept"}
    cue.track[3] = TrackPoint(3, 1000.123456789, 600.987654321, "interpolated", 97.123456, 0.123456)
    save_cue(cue, tmp_path / "c.json")
    assert load_cue(tmp_path / "c.json") == cue


# (field path, changes that break one rule, other paths that break as a consequence)
BREAKS = [
    ("version", dict(version="2.0"), ()),
    ("effect_range", dict(effect_range=EffectRange(32, 152)), ()),                    # 120: not 8k+1
    ("effect_range", dict(effect_range=EffectRange(32, 162)), ()),                    # 130 > 121 and not 8k+1
    ("effect_range", dict(effect_range=EffectRange(-4, 117)), ("route[2].f", "vanish")),   # before the clip
    ("effect_range", dict(effect_range=EffectRange(80, 201), appear_frame=90), ()),   # after the clip
    ("canvas_size", dict(canvas_size=(1920, 1080)), ()),                              # 1080 % 64 != 0
    ("canvas_rect", dict(canvas_rect=(100, 0, 1920, 1080)), ()),                      # outside the plate
    ("canvas_rect", dict(canvas_rect=(0, 0, 1920, 1000)), ()),                        # aspect off by 8 %
    ("track", dict(track=with_track()[:199]), ()),
    ("track[5].source", dict(track=[*with_track()[:5], TrackPoint(5, 0, 0, "guessed", 1, 1), *with_track()[6:]]), ()),
    ("appear_frame", dict(appear_frame=20), ()),                                      # before effect_range
    ("release_frame", dict(appear_frame=107), ()),                                    # appear not before release
    ("release_frame", dict(release_frame=153, route=[RoutePoint(153, None, None, 110)]), ("route[0].f",)),
    ("route", dict(route=[]), ()),
    ("route[0].f", dict(route=[RoutePoint(108, None, None, 110), RoutePoint(116, 700, 420, 60)]), ()),
    ("route[2].f", dict(route=[RoutePoint(107, None, None, 110), RoutePoint(116, 700, 420, 60),
                               RoutePoint(116, 420, 360, 25)]), ()),
    ("route[1].f", dict(route=[RoutePoint(107, None, None, 110), RoutePoint(160, 700, 420, 60)]), ()),
    ("route[1].x", dict(route=[RoutePoint(107, None, None, 110), RoutePoint(116, None, 420, 60)]), ()),
    ("route[1].y", dict(route=[RoutePoint(107, None, None, 110), RoutePoint(116, 700, None, 60)]), ()),
    ("route[1].diameter_px", dict(route=[RoutePoint(107, None, None, 110), RoutePoint(116, 700, 420, 0)]), ()),
    ("vanish", dict(vanish=Vanish(130, 130)), ()),
    ("vanish", dict(vanish=Vanish(128, 160)), ()),
    ("behind[0]", dict(behind=[Behind(91, 80, "left_hand")]), ()),
    ("behind[0]", dict(behind=[Behind(20, 40, "left_hand")]), ()),
    ("behind[0].occluder", dict(behind=[Behind(80, 91, "foot")]), ()),
    ("behind[0].occluder", dict(behind=[Behind(80, 91, {"point": [80, 10]})]), ()),
    ("hand", dict(hand="middle"), ()),
    ("diameter_px", dict(diameter_px=0), ()),
    ("grade.opacity", dict(grade=Grade("#9fd4ff", 1.0, 0.6, 0.4, 1.5, 3)), ()),
    ("grade.glow", dict(grade=Grade("#9fd4ff", 1.0, -0.1, 0.4, 1.0, 3)), ()),
    ("grade.brightness", dict(grade=Grade("#9fd4ff", -1.0, 0.6, 0.4, 1.0, 3)), ()),
    ("grade.tint", dict(grade=Grade("9fd4ff", 1.0, 0.6, 0.4, 1.0, 3)), ()),
    ("grade.fade_in_frames", dict(grade=Grade("#9fd4ff", 1.0, 0.6, 0.4, 1.0, -1)), ()),
    ("correction_falloff", dict(correction_falloff=-1), ()),
]


@pytest.mark.parametrize("path,changes,also", BREAKS, ids=[f"{i}-{b[0]}" for i, b in enumerate(BREAKS)])
def test_each_rule_fires_on_a_cue_broken_in_that_one_way(path, changes, also):
    errors, _ = validate(make_cue(**changes))
    assert any(e.startswith(path + ":") for e in errors), errors
    assert {e.split(":")[0] for e in errors} <= {path, *also}, errors  # nothing else fired


def test_all_problems_are_collected_not_just_the_first():
    errors, _ = validate(make_cue(version="1", hand="middle", diameter_px=-1, canvas_size=(1000, 576)))
    assert {e.split(":")[0] for e in errors} >= {"version", "hand", "diameter_px", "canvas_size"}


def test_missing_fields_are_reported_with_their_path():
    errors, _ = validate(make_cue(route=None, grade=None))
    assert "route: missing" in errors and "grade: missing" in errors


# --------------------------------------------------------------------------- warnings

def test_orb_visible_after_the_last_route_point_warns():
    errors, warnings = validate(make_cue(vanish=Vanish(131, 138)))  # last route point is frame 130
    assert not errors
    assert len(warnings) == 1 and warnings[0].startswith("vanish.end:") and "7 frames" in warnings[0]
    assert validate(make_cue(vanish=Vanish(128, 131)))[1] == []  # last visible frame is 130: nothing held after


def test_an_effect_range_over_193_frames_is_a_warning_not_an_error():
    long_clip = dict(frames=300, track=[TrackPoint(f, 1000.0, 600.0, "tracked", PALM_W, 0.9) for f in range(300)])
    errors, warnings = validate(make_cue(effect_range=EffectRange(32, 241), **long_clip))  # 209 = 8k+1, but over 193
    assert errors == []
    assert len(warnings) == 1 and warnings[0].startswith("effect_range:") and "32..225" in warnings[0]
    assert "last 16 frames" in warnings[0]
    errors, _ = validate(make_cue(effect_range=EffectRange(32, 162)))         # 130: the 8k+1 rule is still an error
    assert any("not 8k+1" in e for e in errors)


def test_lost_runs_are_the_flagged_runs():
    assert lost_runs(with_track((10, 20), (50, 52))) == [(10, 20), (50, 52)]
    assert lost_runs(with_track((190, 200))) == [(190, 200)]  # a run at the end of the clip
    assert lost_runs(with_track()) == []


def test_interpolated_frames_without_the_flag_are_not_lost():
    track = with_track((60, 70))
    unflagged = [replace(p, lost=False) for p in track]  # still source = interpolated: a long gap, but not flagged
    assert lost_runs(unflagged) == []
    assert validate(make_cue(track=unflagged))[1] == []
    assert validate(make_cue(track=track))[1] != []


def test_the_lost_flag_alone_drives_the_warnings():
    short_flagged = with_track((60, 62))  # 2 frames and no jump: only the flag says it is lost
    assert any(w.startswith("track[60..61]:") for w in validate(make_cue(track=short_flagged))[1])
    assert validate(make_cue(track=with_track()))[1] == []


def test_old_cue_files_without_lost_load_as_false():
    d = cue_to_dict(make_cue())
    for p in d["track"]:
        del p["lost"]
    assert all(p["lost"] is False for p in cue_to_dict(cue_from_dict(d))["track"])
    d["track"][5]["lost"] = True
    assert cue_from_dict(d).track[5].lost is True


def test_a_gap_with_a_correction_in_it_is_handled():
    cue = make_cue(track=with_track((60, 70)), corrections=[Correction(65, 650, 0)])
    assert open_lost_runs(cue) == []
    assert validate(cue) == ([], [])
    cue.corrections = [Correction(71, 710, 0)]   # next to the gap, not in it
    assert open_lost_runs(cue) == [(60, 70)]


def test_held_frames_interpolated_in_a_lost_gap_warn():
    _, warnings = validate(make_cue(track=with_track((60, 70))))  # 10 frames: lost, inside the held range 40..106
    assert any(w.startswith("track[60..69]:") for w in warnings), warnings
    short = [replace(p, lost=False) if 60 <= p.f < 63 else p for p in with_track((60, 63))]
    assert validate(make_cue(track=short))[1] == []  # short and smooth, so the tracker did not flag it


def test_release_inside_a_lost_gap_warns():
    errors, warnings = validate(make_cue(track=with_track((105, 110))))  # flagged lost across 5 frames
    assert not errors
    release = [w for w in warnings if w.startswith("release_frame:")]
    assert len(release) == 1 and "frame 104" in release[0]
    assert any(w.startswith("track[105..106]:") for w in warnings)  # the held part of that gap


def test_route_points_may_lie_outside_the_plate():
    cue = make_cue(route=[RoutePoint(107, None, None, 110), RoutePoint(110, 170, 345, 100),
                          RoutePoint(113, -120, 300, 90)], vanish=Vanish(112, 114))
    assert validate(cue) == ([], [])
