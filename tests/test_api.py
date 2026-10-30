"""services.cue.api: validate (one small synthetic cue per code), preview_path, propose_timing and analyze (tracker faked)."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from conftest import make_cue, moving_track, with_lost_gap
from services.cue import api
from services.cue.cue import (Behind, BodyPoint, Correction, EffectRange, Entrance, Grade, HandSeed, Orbit, RoutePoint, TrackPoint,
                       Vanish, cue_to_dict)
from services.cue.route import propose_route_timing, release_point

GRADE = dict(tint="#9fd4ff", brightness=1.0, glow=0.6, person_light=0.0, opacity=1.0, fade_in_frames=3)


def cue_dict(**changes) -> dict:
    """The synthetic cue (conftest) as contract JSON, with person_light 0 so a clean cue reports nothing unused."""
    changes.setdefault("grade", Grade(**GRADE))
    return cue_to_dict(make_cue(**changes))


def codes_of(warnings):
    return {w["code"] for w in warnings}


def test_a_clean_cue_gives_only_the_pass_edge_reminder():
    warnings = api.validate(cue_dict())
    assert [w["code"] for w in warnings] == ["check_pass_edge"]
    w = warnings[0]
    assert w["severity"] == "info" and w["frames"] == [89, 91] and w["jump"] == 89 and w["cause"] == {"kind": "pass", "index": 0}
    assert "89–90" in w["message"]  # frames are shown inclusive


def _setitem(path, value):
    def edit(d):
        node = d
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
    return edit


# (code, cue changes for make_cue, edit of the JSON afterwards)
CASES = [
    ("hand_invalid", dict(hand="middle"), None),
    ("frame_outside_clip", dict(hand_seed=HandSeed(300, 1000, 600)), None),
    ("track_invalid", dict(track=moving_track()[:199]), None),
    ("value_out_of_range", dict(diameter_px=0), None),
    ("correction_duplicate", dict(corrections=[Correction(60, 1000, 600), Correction(60, 1001, 600)]), None),
    ("missing_field", {}, lambda d: d.pop("vanish")),
    ("version_unsupported", dict(version="2.0"), None),
    ("no_track", dict(track=[]), None),
    ("effect_range_length", dict(effect_range=EffectRange(32, 152)), None),
    ("effect_range_outside", dict(effect_range=EffectRange(80, 201), appear_frame=90), None),
    ("canvas_invalid", dict(canvas_size=(1920, 1080)), None),
    ("frame_outside_effect_range", dict(appear_frame=20), None),
    ("release_not_after_appear", dict(appear_frame=107), None),
    ("entrance_shortened", dict(entrance=Entrance(2.0)), None),
    ("window_grown", dict(entrance=Entrance(0.5)), None),
    ("window_cut_short", dict(effect_range=EffectRange(4, 197), appear_frame=10, entrance=Entrance(0.5)), None),
    ("window_trimmed", dict(effect_range=EffectRange(32, 241)), None),
    ("behaviour_invalid", dict(behaviour="swing"), None),
    ("body_point_invalid", dict(body_point=BodyPoint("fist", None)), None),
    ("shape_invalid", dict(shape="pyramid"), None),
    ("ending_invalid", dict(ending="explode"), None),
    ("orbit_invalid", dict(behaviour="orbit", release_frame=None, route=[], orbit=Orbit(radius_sizes=0.0)), None),
    ("vanish_extended", dict(ending="shatter"), None),
    ("hold_path_unused", dict(behaviour="hold"), None),
    ("route_empty", dict(route=[]), None),
    ("route_start", dict(route=[RoutePoint(108, None, None, 110), RoutePoint(116, 700, 420, 60)]), None),
    ("route_order", dict(route=[RoutePoint(107, None, None, 110), RoutePoint(116, 700, 420, 60),
                                RoutePoint(116, 420, 360, 25)]), None),
    ("route_position_missing", dict(route=[RoutePoint(107, None, None, 110), RoutePoint(116, None, 420, 60)]), None),
    ("easing_invalid", {}, _setitem(["route", 1, "easing_in"], "bounce")),
    ("range_reversed", dict(vanish=Vanish(130, 130)), None),
    ("range_outside_effect", dict(behind=[Behind(20, 40, "left_hand")]), None),
    ("occluder_invalid", dict(behind=[Behind(80, 91, "foot")]), None),
    ("tint_invalid", {}, _setitem(["grade", "tint"], "9fd4ff")),
    ("orb_holds_after_route", dict(vanish=Vanish(131, 138)), None),
    ("track_uncertain", dict(track=with_lost_gap(moving_track(), 60, 70)), None),
    ("release_in_lost_gap", dict(track=with_lost_gap(moving_track(), 105, 110)), None),
    ("invalid_field", {}, _setitem(["route", 1, "x"], "left")),
    ("unknown_field", {}, _setitem(["colour"], "blue")),
    ("setting_not_used", {}, _setitem(["advanced"], {"track_smoothing": 0.4})),
    ("no_path", dict(route=[RoutePoint(107, None, None, 110)]), None),
    ("check_pass_edge", {}, None),
]


@pytest.mark.parametrize("code,changes,edit", CASES, ids=[c[0] for c in CASES])
def test_each_validate_code_comes_from_a_small_synthetic_cue(code, changes, edit):
    d = cue_dict(**changes)
    if edit:
        edit(d)
    warnings = api.validate(d)
    assert code in codes_of(warnings), [w["detail"] if "detail" in w else w["message"] for w in warnings]
    for w in warnings:
        assert api.schema_errors("warning", w) == [], w
        assert w["severity"] == api.codes()[w["code"]]["severity"]
    assert len({w["id"] for w in warnings}) == len(warnings)
    order = {"error": 0, "warning": 1, "info": 2}
    assert [order[w["severity"]] for w in warnings] == sorted(order[w["severity"]] for w in warnings)


def test_every_validate_code_has_a_case():
    validate_codes = {k for k, v in api.codes().items() if v["call"] == "validate"}
    assert validate_codes == {c[0] for c in CASES}


def test_schema_errors_come_alone_with_the_field_path():
    d = cue_dict()
    d["route"][1]["x"] = "left"
    d["behind"][0]["occluder"] = 3
    warnings = api.validate(d)
    assert codes_of(warnings) == {"invalid_field"}
    assert {w["cause"]["path"] for w in warnings} == {"route[1].x", "behind[0].occluder"}
    assert all(w["severity"] == "error" and w["cause"]["kind"] == "field" for w in warnings)


def test_a_bad_track_argument_is_an_invalid_field_under_track():
    track = api.track_from_points(moving_track())
    track[3]["confidence"] = 2.0
    warnings = api.validate(cue_dict(), track)
    assert [w["cause"]["path"] for w in warnings] == ["track[3].confidence"]


@pytest.mark.parametrize("field,edit", [
    ("advanced.track_smoothing", _setitem(["advanced"], {"track_smoothing": 0.4})),
    ("advanced.export_layers", _setitem(["advanced"], {"export_layers": {"matte": True}})),
    ("grade.spill", _setitem(["grade", "spill"], 0.4)),
    ("grade.person_light", _setitem(["grade", "person_light"], 0.4)),
    ("route[0].easing_in", _setitem(["route", 0, "easing_in"], "ease_out")),
])
def test_every_accepted_but_unused_field_says_so(field, edit):
    d = cue_dict()
    edit(d)
    unused = [w for w in api.validate(d) if w["code"] == "setting_not_used"]
    assert [w["cause"] for w in unused] == [{"kind": "setting", "path": field}]
    assert unused[0]["severity"] == "info"
    assert set(api.NOT_USED) == {"advanced.track_smoothing", "advanced.export_layers", "grade.spill",
                                 "grade.person_light", "route[0].easing_in"}


def test_fields_accepted_for_generate_are_silent():
    d = cue_dict()
    d.update(look="fireball", prompt="A fireball, camera stays fixed.", reference_image="still.png")
    assert codes_of(api.validate(d)) == {"check_pass_edge"}


def test_track_uncertain_names_the_pass_when_the_other_hand_is_in_front():
    w = next(w for w in api.validate(cue_dict(track=with_lost_gap(moving_track(), 82, 88))) if w["code"] == "track_uncertain")
    assert w["cause"] == {"kind": "pass", "index": 0} and "crosses in front" in w["message"]
    assert w["frames"] == [82, 88] and "82–87" in w["message"]
    w = next(w for w in api.validate(cue_dict(track=with_lost_gap(moving_track(), 60, 70))) if w["code"] == "track_uncertain")
    assert w["cause"] == {"kind": "track"}


def test_entrance_shortened_offers_a_fix_that_clears_it():
    d = cue_dict(entrance=Entrance(2.0))
    w = next(w for w in api.validate(d) if w["code"] == "entrance_shortened")
    assert w["fix"] == {"label": "Use 1.67 s", "op": "set", "path": "entrance.seconds", "value": 1.67}
    d["entrance"]["seconds"] = w["fix"]["value"]
    assert "entrance_shortened" not in codes_of(api.validate(d))


def test_validate_uses_the_track_argument_over_the_cue_track():
    d = cue_dict(track=[])
    assert "no_track" in codes_of(api.validate(d))
    assert "no_track" not in codes_of(api.validate(d, api.track_from_points(moving_track())))


def test_validate_does_not_change_its_input():
    d = cue_dict()
    before = copy.deepcopy(d)
    api.validate(d)
    assert d == before


# --------------------------------------------------------------------------- preview_path

def test_preview_covers_every_frame_with_phases_in_order():
    track = api.track_from_points(moving_track(speed=1.0))
    p = api.preview_path(cue_dict(track=[]), track)
    assert [q["frame"] for q in p] == list(range(200))
    assert api.schema_errors("preview", p) == []
    phases = [q["phase"] for q in p]
    assert phases == ["before"] * 40 + ["held"] * 67 + ["flight"] * 21 + ["vanish"] * 3 + ["after"] * 69
    assert not any(q["visible"] for q in p[:40]) and all(q["visible"] for q in p[40:131]) and not p[131]["visible"]


def test_preview_entrance_phase_and_growth():
    track = api.track_from_points(moving_track())
    p = api.preview_path(cue_dict(entrance=Entrance(0.5), track=[]), track)
    assert [q["phase"] for q in p[27:41]] == ["before"] + ["entrance"] * 12 + ["held"]
    d = [q["diameter"] for q in p[28:41]]
    assert d[0] == 0 and d[-1] == 110 and all(np.diff(d) > 0)          # grows from nothing, whole at appear_frame


def test_preview_follows_the_easing():
    d = cue_dict(track=[])
    d["route"][1]["easing_in"] = "ease_in"
    track = api.track_from_points(moving_track())
    eased, plain = api.preview_path(d, track), api.preview_path(cue_dict(track=[]), track)
    assert eased[111]["x"] != plain[111]["x"] and eased[116] == plain[116]  # same points, different timing


def test_preview_of_a_cue_with_errors_is_input_invalid():
    with pytest.raises(api.CueFXError) as e:
        api.preview_path(cue_dict(appear_frame=107), api.track_from_points(moving_track()))
    assert e.value.code == "input_invalid" and "release_frame" in e.value.message


def test_preview_of_a_cue_that_breaks_the_schema_is_input_invalid():
    d = cue_dict()
    d["frames"] = "many"
    with pytest.raises(api.CueFXError) as e:
        api.preview_path(d, api.track_from_points(moving_track()))
    assert e.value.code == "input_invalid" and "frames" in e.value.message


# --------------------------------------------------------------------------- propose_timing

def test_propose_timing_retimes_the_middle_points_and_leaves_the_input_alone():
    d = cue_dict(route=[RoutePoint(107, None, None, 110), RoutePoint(110, 900, 500, 90),
                        RoutePoint(112, 700, 420, 60), RoutePoint(130, 420, 360, 25)], track=[])
    track = api.track_from_points(moving_track())
    before = copy.deepcopy(d)
    out = api.propose_timing(d, track)
    assert d == before
    cue = replace(make_cue(), route=[RoutePoint(**{k: p[k] for k in ("f", "x", "y", "diameter_px")}) for p in d["route"]])
    expected = propose_route_timing([(*release_point(cue), 110), (900, 500, 90), (700, 420, 60), (420, 360, 25)], 107, 130)
    assert [p["f"] for p in out["route"]] == expected and expected[0] == 107 and expected[-1] == 130
    assert expected != [107, 110, 112, 130]
    out["route"] = d["route"]
    assert out == d  # nothing else changed


def test_propose_timing_with_one_point_returns_the_cue():
    d = cue_dict(route=[RoutePoint(107, None, None, 110)], vanish=Vanish(105, 107))
    assert api.propose_timing(d, api.track_from_points(moving_track())) == d


def test_propose_timing_errors_are_input_invalid():
    d = cue_dict(route=[RoutePoint(107, None, None, 110), RoutePoint(108, 900, 500, 90), RoutePoint(109, 700, 420, 60),
                        RoutePoint(110, 420, 360, 25)], vanish=Vanish(108, 111))
    d["route"][3]["f"] = 109  # the cue now has an error (route order)
    with pytest.raises(api.CueFXError) as e:
        api.propose_timing(d, api.track_from_points(moving_track()))
    assert e.value.code == "input_invalid"


# --------------------------------------------------------------------------- track forms

def test_track_forms_round_trip():
    pts = with_lost_gap(moving_track(speed=0.5), 60, 70)
    track = api.track_from_points(pts)
    assert api.schema_errors("track", track) == []
    assert api.points_from_track(track) == pts
    assert [t["uncertain"] for t in track[58:72]] == [False] * 2 + [True] * 10 + [False] * 2
    bare = [{k: t[k] for k in ("frame", "x", "y", "confidence", "uncertain")} for t in track]
    assert [p.source for p in api.points_from_track(bare)[58:62]] == ["tracked", "tracked", "interpolated", "interpolated"]


# --------------------------------------------------------------------------- analyze (tracker faked)

class FakeInfo:
    def __init__(self, frames):
        self.frames = frames


class FakeResult:
    def __init__(self, points):
        self.points, self.info = points, FakeInfo(len(points))


class FakeSel:
    def __init__(self, warnings=()):
        self.warnings = list(warnings)


@pytest.fixture
def fake_clip(tmp_path, monkeypatch):
    """A clip path whose probe reports 200 frames; set `state["points"]` / `state["raise"]` to drive track_video."""
    import services.analysis.track

    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"not really a video")
    state = {"points": moving_track(), "warnings": [], "raise": None, "frames": 200}
    monkeypatch.setattr(api, "_probe_frames", lambda p: state["frames"])

    def track_video(video, hand, seed, cfg=None, model=None):
        if state["raise"]:
            raise state["raise"]
        return FakeResult(state["points"]), None, FakeSel(state["warnings"])

    monkeypatch.setattr(services.analysis.track, "track_video", track_video)
    return clip, state


def test_analyze_returns_the_contract_result(fake_clip):
    clip, state = fake_clip
    state["points"] = with_lost_gap(moving_track(), 60, 70)
    res = api.analyze(clip, {"frame": 60, "x": 1000, "y": 600})
    assert api.schema_errors("analyze_result", res) == []
    assert res["summary"] == {"tracked": 190, "total": 200, "uncertain_ranges": [[60, 70]]}
    assert res["hand_size_px"] == 100.0 and res["warnings"] == []


def test_analyze_passes_a_missed_click_on_as_a_warning(fake_clip):
    clip, state = fake_clip
    state["warnings"] = ["hand_seed: (5, 5) is 1200 px from the nearest palm in frame 60, ..."]
    w = api.analyze(clip, {"frame": 60, "x": 5, "y": 5})["warnings"]
    assert [x["code"] for x in w] == ["seed_far"] and w[0]["frames"] == [60, 61] and w[0]["detail"].startswith("hand_seed")


@pytest.mark.parametrize("raised,code", [("no hand detected in seed frame 60", "no_hand"),
                                         ("seed frame 300 is outside the clip (0..199)", "input_invalid"),
                                         ("cannot open video: x", "clip_unreadable")])
def test_analyze_maps_tracker_errors_to_contract_codes(fake_clip, raised, code):
    from services.analysis.track import TrackError

    clip, state = fake_clip
    state["raise"] = TrackError(raised, code) if code != "clip_unreadable" else TrackError(raised)
    with pytest.raises(api.CueFXError) as e:
        api.analyze(clip, {"frame": 60, "x": 1000, "y": 600})
    assert e.value.code == code


def test_analyze_fails_when_most_of_the_clip_is_lost(fake_clip):
    clip, state = fake_clip
    state["points"] = with_lost_gap(moving_track(), 20, 121, jump=0)  # 101 of 200 frames lost
    with pytest.raises(api.CueFXError) as e:
        api.analyze(clip, {"frame": 10, "x": 1000, "y": 600})
    assert e.value.code == "track_lost" and "101 of 200" in e.value.message
    state["points"] = with_lost_gap(moving_track(), 20, 120, jump=0)  # exactly half: still a track
    assert api.analyze(clip, {"frame": 10, "x": 1000, "y": 600})["summary"]["uncertain_ranges"] == [[20, 120]]


def test_analyze_rejects_a_long_clip_before_tracking(fake_clip):
    clip, state = fake_clip
    state["frames"], state["raise"] = api.MAX_CLIP_FRAMES + 1, AssertionError("must not track")
    with pytest.raises(api.CueFXError) as e:
        api.analyze(clip, {"frame": 0, "x": 0, "y": 0})
    assert e.value.code == "clip_too_long" and str(api.MAX_CLIP_FRAMES) in e.value.message


@pytest.mark.parametrize("seed", [{"frame": 1.5, "x": 0, "y": 0}, {"frame": 1, "x": "a", "y": 0}, [1, 2, 3]])
def test_analyze_rejects_a_bad_seed(fake_clip, seed):
    with pytest.raises(api.CueFXError) as e:
        api.analyze(fake_clip[0], seed)
    assert e.value.code == "input_invalid"


def test_analyze_of_a_missing_clip_is_unreadable(tmp_path):
    with pytest.raises(api.CueFXError) as e:
        api.analyze(tmp_path / "nope.mov", {"frame": 0, "x": 0, "y": 0})
    assert e.value.code == "clip_unreadable"


def test_track_points_keep_their_confidence_in_range():
    pts = [TrackPoint(0, 1.0, 2.0, "tracked", 90.0, 1.0000001)]
    assert api.track_from_points(pts)[0]["confidence"] == 1.0
