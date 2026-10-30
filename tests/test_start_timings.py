"""Tasks 28-29: starting timings from the clip's own track (services.cue.start_timings, used by api.new_cue)."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from services.cue import api
from services.cue.cue import TrackPoint
from services.cue.start_timings import STEADY_SPEED, THROW_SEARCH_S, THROW_SPEED, find_throw, start_timings, steady_runs, throw_frame

ROOT = Path(__file__).resolve().parents[1]
SEED = {"f": 5, "x": 600, "y": 500}


def track(n: int, moving: set[int] = frozenset(), palm_w: float = 80.0) -> list[TrackPoint]:
    """x advances 2.5 palm widths per frame (a throw: above THROW_SPEED) in `moving` frames and stays put otherwise."""
    x, out = 600.0, []
    for f in range(n):
        x += 2.5 * palm_w if f in moving else 0.0
        out.append(TrackPoint(f, x, 500.0, "tracked", palm_w, 0.9))
    return out


def speed_track(speeds: dict[int, float], n: int, palm_w: float = 80.0, lost: frozenset[int] = frozenset()
                ) -> list[TrackPoint]:
    """Frame f moves speeds.get(f, 0) palm widths from frame f - 1; frames in `lost` are lost."""
    x, out = 600.0, []
    for f in range(n):
        x += speeds.get(f, 0.0) * palm_w
        out.append(TrackPoint(f, x, 500.0, "tracked", palm_w, 0.9, lost=f in lost))
    return out


def cue_for(points: list[TrackPoint], frames: int | None = None, fps: float = 24):
    frames = frames or len(points)
    return api.new_cue_with_warnings("c.mov", fps, frames, [1920, 1080], "right", SEED, api.track_from_points(points))


def errors(cue: dict) -> list[dict]:
    return [w for w in api.validate(cue) if w["severity"] == "error"]


def test_one_long_steady_run():
    t = start_timings(track(200, moving=set(range(71, 200))), 24, 200)
    assert steady_runs(track(200, moving=set(range(71, 200)))) == [(1, 70)]
    assert not t.fallback
    assert (t.appear_frame, t.release_frame) == (1 + 12, 71)  # appear = run start + N_in (0.5 s), release = run end + 1
    assert (t.vanish_start, t.vanish_end) == (77, 83)         # 0.25 s after the release, 0.25 s long
    assert t.diameter_px == round(1.15 * 80) and t.anchor_offset == (0, -round(0.65 * 92))


def test_the_longer_of_two_steady_runs_wins():
    pts = track(200, moving=set(range(30, 40)) | set(range(121, 200)))  # runs 1..29 and 40..120
    assert steady_runs(pts) == [(1, 29), (40, 120)]
    t = start_timings(pts, 24, 200)
    assert (t.appear_frame, t.release_frame) == (40 + 12, 121)


def test_a_gap_of_two_frames_does_not_split_a_run_but_three_do():
    two = steady_runs(track(100, moving={40, 41}))
    three = steady_runs(track(100, moving={40, 41, 42}))
    assert two == [(1, 99)] and three == [(1, 39), (43, 99)]


def test_lost_frames_are_not_steady():
    pts = [TrackPoint(p.f, p.x, p.y, p.source, p.palm_w, p.conf, lost=20 <= p.f < 30) for p in track(60)]
    assert steady_runs(pts, max_gap=0) == [(1, 19), (31, 59)]


def test_no_steady_run_falls_back_to_fractions_of_the_clip_with_a_warning():
    pts = track(120, moving=set(range(1, 120)))
    t = start_timings(pts, 24, 120)
    assert t.fallback and (t.appear_frame, t.release_frame) == (30, 72) and (t.vanish_start, t.vanish_end) == (78, 84)
    cue, warnings = cue_for(pts)
    assert [w["code"] for w in warnings] == ["placeholder_timings"]
    assert api.schema_errors("warning", warnings[0]) == []
    assert "No steady palm found; starting timings are placeholders" in warnings[0]["message"]
    assert errors(cue) == []


def test_a_short_steady_run_is_not_enough():
    pts = track(120, moving=set(range(1, 120)) - set(range(50, 60)))  # 10 steady frames < N_in + 8
    assert start_timings(pts, 24, 120).fallback


def test_a_30_frame_clip_has_no_validate_errors():
    for pts in (track(30), track(30, moving=set(range(1, 30)))):
        cue, _ = cue_for(pts)
        assert errors(cue) == [], errors(cue)
        release = cue["release_frame"] if cue["behaviour"] == "hold_throw" else cue["appear_frame"] + 1  # a hold has none
        assert 0 <= cue["appear_frame"] < release < cue["vanish"]["end"] <= 30


def test_a_1000_frame_clip_fits_the_window():
    pts = track(1000, moving=set(range(900, 1000)))
    cue, warnings = cue_for(pts)
    assert warnings == [] and errors(cue) == []
    er = cue["effect_range"]
    assert er["end"] - er["start"] <= 193
    assert not [w for w in api.validate(cue) if w["code"] in ("window_cut_short", "window_trimmed")]
    assert cue["appear_frame"] > 1 + 12 and cue["release_frame"] == 900  # appear moved later, the release did not


def test_same_track_same_cue():
    pts = track(150, moving=set(range(90, 150)))
    assert cue_for(pts) == cue_for(pts)


def test_only_the_release_point_is_in_the_route_and_the_rest_are_defaults():
    cue, _ = cue_for(track(150, moving=set(range(90, 150))))
    assert cue["route"] == [{"f": 90, "x": None, "y": None, "diameter_px": cue["diameter_px"], "easing_in": "linear"}]
    assert cue["behind"] == [] and cue["look"] == "energy_orb" and cue["seed"] == 42
    assert "no_path" in {w["code"] for w in api.validate(cue)}


def test_the_golden_track_gets_a_valid_cue_with_the_hold_found():
    golden = json.loads((ROOT / "contract" / "golden" / "throw_71319.track.json").read_text(encoding="utf-8"))
    cue, warnings = api.new_cue_with_warnings("throw_71319.mov", 24, 193, [1920, 1080], "right",
                                              {"f": 60, "x": 1160, "y": 550}, golden)
    assert warnings == [] and errors(cue) == []
    assert steady_runs(api.points_from_track(golden))[0] == (26, 96)
    assert 36 <= cue["appear_frame"] <= 106  # the palm is held by eye ~40-106
    # the real throw is at 107-108, after a wind-up (frames 99-106, 0.12-0.42 palm widths per frame): the hand is lost
    # from 107 (task 29; task 28 released at the end of the hold, 97)
    assert 106 <= cue["release_frame"] <= 108 and cue["release_frame"] == 107
    assert cue["vanish"] == {"start": 113, "end": 119}
    assert (cue["diameter_px"], cue["anchor_offset"]) == (100, [0, -65])


# --------------------------------------------------------------------------- task 29: the release is the throw

def test_a_wind_up_then_a_throw_releases_on_the_throw():
    pts = speed_track({**{f: 0.4 for f in range(71, 81)}, 81: 2.5, 82: 2.5}, 150)  # steady 1..70, wind-up, throw
    assert steady_runs(pts)[0] == (1, 70)
    assert start_timings(pts, 24, 150).release_frame == 81


def test_a_lost_gap_after_the_run_releases_on_its_first_lost_frame():
    pts = speed_track({f: 0.2 for f in range(71, 76)}, 150, lost=frozenset(range(76, 90)))
    assert start_timings(pts, 24, 150).release_frame == 76


def test_only_slow_drift_after_the_run_is_a_hold_that_vanishes_after_the_run():
    pts = speed_track({f: 0.5 for f in range(71, 150)}, 150)  # task 33: no throw, so a hold (was: release at 71)
    t = start_timings(pts, 24, 150)
    assert (t.behaviour, t.release_frame, t.vanish_start) == ("hold", None, 71)


def test_a_fast_frame_two_seconds_after_the_run_is_ignored():
    pts = speed_track({f: 0.3 for f in range(71, 150)} | {71 + 48: 2.5}, 150)
    assert 2.0 > THROW_SEARCH_S
    t = start_timings(pts, 24, 150)  # task 33: no throw in the search window, so a hold (was: release at 71)
    assert (t.behaviour, t.release_frame, t.vanish_start) == ("hold", None, 71)


def test_the_speed_limit_is_strict_and_the_search_window_includes_its_last_frame():
    assert THROW_SPEED == 1.0
    at = speed_track({81: THROW_SPEED}, 150)  # exactly the limit: not a throw
    assert throw_frame(at, 70, 24) == 71 and find_throw(at, 70, 24) is None
    last = 70 + round(THROW_SEARCH_S * 24)
    assert throw_frame(speed_track({last: 2.5}, 150), 70, 24) == last
    assert throw_frame(speed_track({last + 1: 2.5}, 150), 70, 24) == 71


def test_the_throw_rule_is_relative_to_the_palm_width():
    base = speed_track({f: 0.4 for f in range(71, 81)} | {81: 2.5}, 150)
    big = [TrackPoint(p.f, 2 * p.x, 2 * p.y, p.source, 2 * p.palm_w, p.conf, lost=p.lost) for p in base]
    assert start_timings(big, 24, 150) == replace(start_timings(base, 24, 150), diameter_px=2 * 92,
                                                  anchor_offset=(0, -round(0.65 * 2 * 92)))
    assert start_timings(big, 24, 150).release_frame == 81


def test_the_throw_moves_appear_later_when_the_window_is_too_long():
    pts = speed_track({f: 0.3 for f in range(971, 980)} | {f: 2.5 for f in range(980, 1100)}, 1100)  # hold, throw at 980
    t = start_timings(pts, 24, 1100)
    assert t.release_frame == 980 and t.appear_frame > 1 + 12  # appear moved later, the release did not
    assert t.appear_frame < t.release_frame < t.vanish_end <= 1100


def test_the_steady_threshold_is_relative_to_the_palm_width():
    for w in (40.0, 160.0):
        slow = [TrackPoint(f, 600 + f * STEADY_SPEED * 0.9 * w, 500, "tracked", w, 0.9) for f in range(60)]
        assert steady_runs(slow) == [(1, 59)]


@pytest.mark.slow
def test_throw_71320_real_analyze_has_no_validate_errors():
    clip = ROOT / "data" / "input" / "throw_71320.mov"
    if not clip.exists():
        pytest.skip("data/input/throw_71320.mov is not here")
    result = api.analyze(clip, {"frame": 80, "x": 1133, "y": 540})
    cue, warnings = api.new_cue_with_warnings("throw_71320.mov", 24, 121, [1920, 1080], "right",
                                              {"f": 80, "x": 1133, "y": 540}, result["track"])
    print("71320", {k: cue[k] for k in ("appear_frame", "release_frame", "vanish", "diameter_px", "anchor_offset",
                                        "effect_range")}, [w["code"] for w in warnings])
    assert errors(cue) == []
