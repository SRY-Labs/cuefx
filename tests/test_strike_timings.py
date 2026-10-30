"""Task 36 part B: the strike rule for a fist, foot or knee (services.cue.start_timings.strike_release, used by api.new_cue).
A fist: its fastest frame above 1.0 size/frame. A foot or knee: the peak of the first run of 3 frames above 0.4."""
import math
from dataclasses import replace

import pytest

from services.cue import api
from services.cue.cue import TrackPoint, cue_from_dict, check
from services.cue.start_timings import (STRIKE_PARTS, STRIKE_SPEED, start_timings, strike_release, strike_speeds)

SIZE = 100.0
SEED = {"f": 5, "x": 600, "y": 500}


def moving(n: int, steps: dict[int, tuple[float, float]], start=(900.0, 600.0), size: float = SIZE,
           lost: frozenset[int] = frozenset()) -> list[TrackPoint]:
    """Frame f moves steps[f] (in size measures) from frame f - 1; still otherwise."""
    x, y = start
    out = []
    for f in range(n):
        dx, dy = steps.get(f, (0.0, 0.0))
        x, y = x + dx * size, y + dy * size
        out.append(TrackPoint(f, x, y, "tracked", size, 0.9, lost=f in lost))
    return out


KICK = {f: (1.2 * s, 0.5 * s) for f, s in zip(range(100, 106), (0.4, 1.0, 2.0, 3.0, 2.0, 0.6))}  # peak 3.0 at frame 103


def new_cue(pts, part, side="right", frames=150, plate=(1920, 1080), transform=None):
    return api.new_cue_with_warnings("c.mp4", 24, frames, list(plate), "right", SEED, api.track_from_points(pts),
                                     transform, body_point={"part": part, "side": side})


def test_the_release_is_the_fastest_smoothed_frame_above_one_size_per_frame():
    pts = moving(150, {70: (2.0, 0.0), 71: (3.0, 0.0), 72: (2.0, 0.0)})
    sp = strike_speeds(pts)
    assert sp[71] == pytest.approx((2 + 3 + 2) / 3) and strike_release(pts) == 71
    assert STRIKE_SPEED == {"fist": 1.0, "foot": 0.4, "knee": 0.4} and strike_speeds(pts)[0] == 0.0


def test_one_fast_frame_is_averaged_down_and_does_not_count():
    assert strike_release(moving(150, {70: (2.9, 0.0)})) is None  # 2.9 / 3 < 1
    assert strike_release(moving(150, {70: (3.1, 0.0)})) == 69    # 3.1 / 3 > 1; the first of the equal peaks


def test_a_lost_step_has_no_speed():
    pts = moving(150, {70: (6.0, 0.0)}, lost=frozenset({70}))
    assert strike_release(pts) is None


@pytest.mark.parametrize("part", STRIKE_PARTS)
def test_a_strike_starts_as_a_throw_with_a_short_flight(part):
    cue, warnings = new_cue(moving(150, KICK, start=(300.0, 400.0)), part)
    assert [w["code"] for w in warnings] == []
    assert cue["behaviour"] == "hold_throw" and cue["release_frame"] == 103
    assert cue["appear_frame"] == 103 - 24                                     # one second before the release
    assert cue["vanish"] == {"start": 109, "end": 115}                          # release + 0.25 s for 0.25 s, as for any throw
    r = cue["route"]
    assert [p["f"] for p in r] == [103, 115] and r[0]["x"] is None and r[1]["diameter_px"] == cue["diameter_px"]
    assert r[0]["diameter_px"] == cue["diameter_px"]
    assert not [w for w in api.validate(cue) if w["severity"] == "error"], api.validate(cue)
    assert "orb_holds_after_route" not in {w["code"] for w in api.validate(cue)}


def test_the_route_goes_six_sizes_along_the_mean_velocity_before_the_release():
    pts = moving(150, KICK, start=(300.0, 250.0))
    cue, _ = new_cue(pts, "fist")
    # mean velocity over frames 100..102 (the three before the release 103): steps (0.48, 0.2), (1.2, 0.5), (2.4, 1.0)
    vx, vy = sum(KICK[f][0] for f in (100, 101, 102)), sum(KICK[f][1] for f in (100, 101, 102))
    n = math.hypot(vx, vy)
    ox, oy = cue["anchor_offset"]
    end = cue["route"][1]
    assert (end["x"], end["y"]) == (round(pts[103].x + ox + 6 * SIZE * vx / n), round(pts[103].y + oy + 6 * SIZE * vy / n))


def test_the_route_is_kept_inside_the_plate_by_0_9_diameters():
    """The margin 1.8 x the radius (0.9 x the diameter) keeps a burst or shatter at the end of the route fully in frame."""
    cue, _ = new_cue(moving(150, {f: (-3.0, 0.0) for f in range(70, 75)}, start=(500.0, 600.0)), "fist")
    assert cue["diameter_px"] == 160 and cue["route"][1]["x"] == round(0.9 * 160) and cue["route"][1]["y"] == 600
    cue, _ = new_cue(moving(150, {f: (0.0, 3.0) for f in range(70, 75)}, start=(900.0, 400.0)), "foot")
    assert cue["diameter_px"] == 100 and cue["route"][1]["y"] == 1080 - 90 and cue["route"][1]["x"] == 900
    cue, _ = new_cue(moving(150, {f: (3.0, 0.0) for f in range(70, 75)}, start=(900.0, 400.0)), "fist")
    assert cue["route"][1]["x"] == 1920 - 144


def test_the_route_stays_inside_the_conformed_picture_not_its_padding():
    """A 4:3 clip conformed to the plate has 240 px of black each side (transform offset [240, 0]): the route's end is
    kept 0.9 diameters inside the picture 240..1680, and 0.9 diameters from the top and bottom."""
    t = {"scale": 1.406, "offset": [240, 0]}
    cue, _ = new_cue(moving(150, {f: (-3.0, 0.0) for f in range(70, 75)}, start=(700.0, 600.0)), "fist", transform=t)
    assert cue["route"][1]["x"] == 240 + 144 and cue["route"][1]["y"] == 600
    cue, _ = new_cue(moving(150, {f: (3.0, 0.0) for f in range(70, 75)}, start=(1400.0, 600.0)), "fist", transform=t)
    assert cue["route"][1]["x"] == 1680 - 144
    cue, _ = new_cue(moving(150, {f: (0.0, -3.0) for f in range(70, 75)}, start=(900.0, 500.0)), "foot", transform=t)
    assert cue["route"][1]["y"] == 90
    for c in (cue,):
        assert not [w for w in api.validate(c) if w["severity"] == "error"]


def test_a_picture_too_small_for_the_margin_gets_its_centre():
    pts = moving(150, {f: (3.0, 0.0) for f in range(70, 75)}, start=(900.0, 400.0))
    t = start_timings(pts, 24, 150, part="fist", picture=(800, 300, 960, 460))  # 160 px square, margin 144 each side
    assert t.route_end[1:] == (880, 380)


def test_the_appear_frame_is_not_below_zero():
    cue, _ = new_cue(moving(150, {f: (3.0, 0.0) for f in range(10, 13)}), "knee")
    assert cue["release_frame"] == 11 and cue["appear_frame"] == 0
    assert not [w for w in api.validate(cue) if w["severity"] == "error"]


def test_a_strike_at_the_end_of_the_clip_still_makes_a_valid_cue():
    for peak in (147, 148, 149):
        cue, _ = new_cue(moving(150, {f: (3.0, 0.0) for f in range(peak - 1, 150)}), "fist", frames=150)
        assert not [w for w in api.validate(cue) if w["severity"] == "error"], (peak, api.validate(cue))
        assert cue["route"][-1]["f"] < cue["effect_range"]["end"]


def test_a_foot_or_knee_takes_the_peak_of_the_first_run_above_0_4_and_not_the_swing_back():
    out = {f: (0.5, 0.0) for f in range(60, 66)}                       # the kick out: 0.5 size/frame for 6 frames
    out[62] = (0.7, 0.0)                                               # its peak
    back = {f: (-0.65, 0.0) for f in range(80, 86)}                    # the swing back is faster
    pts = moving(150, {**out, **back})
    sp = strike_speeds(pts)
    assert max(sp) > sp[62] and sp.index(max(sp)) >= 80                # the global peak is the swing back
    for part in ("foot", "knee"):
        assert strike_release(pts, part) == 61                         # the first run's peak (a tie of 61, 62, 63: the first)
    assert strike_release(pts, "fist") is None                        # a fist stays at 1.0
    cue, _ = new_cue(pts, "foot")
    assert cue["release_frame"] == 61 and cue["appear_frame"] == 61 - 24 and cue["behaviour"] == "hold_throw"


def test_a_run_must_last_three_frames():
    def above(steps):
        return [f for f, v in enumerate(strike_speeds(moving(150, steps))) if v > 0.4]

    two = {70: (0.9, 0.0), 71: (0.9, 0.0)}                             # smoothed 0.3 0.6 0.6 0.3: a run of 2
    assert above(two) == [70, 71] and strike_release(moving(150, two), "foot") is None
    three = {**two, 72: (0.9, 0.0)}                                    # smoothed 0.3 0.6 0.9 0.6 0.3: a run of 3
    assert above(three) == [70, 71, 72] and strike_release(moving(150, three), "foot") == 71
    one = {70: (1.0, 0.0)}                                             # 1/3 = 0.33 everywhere
    assert above(one) == [] and strike_release(moving(150, one), "knee") is None
    later = {**two, **{f: (0.6, 0.0) for f in range(90, 96)}}          # a run of 2, then a run of 4 (91..94)
    assert above(later) == [70, 71, 91, 92, 93, 94]
    assert strike_release(moving(150, later), "foot") == 91            # the short run is skipped


def test_no_movement_above_the_threshold_keeps_the_old_rule():
    slow = moving(150, {f: (0.9, 0.0) for f in range(90, 150)})  # faster than a drift, never above 1.0
    assert strike_release(slow, "fist") is None
    for part in ("fist",):
        assert strike_release(slow) is None
        a, b = start_timings(slow, 24, 150, part=part), start_timings(slow, 24, 150, part="palm")
        assert (a.appear_frame, a.release_frame, a.vanish_start, a.vanish_end, a.behaviour, a.route_end) == (
            b.appear_frame, b.release_frame, b.vanish_start, b.vanish_end, b.behaviour, None)
        cue, _ = new_cue(slow, part)
        assert len(cue["route"]) <= 1  # at most the release point, as before (a drift with no throw starts as a hold)
        assert cue["behaviour"] == b.behaviour


def test_a_slow_foot_or_knee_never_reaches_0_4():
    slow = moving(150, {f: (0.3, 0.0) for f in range(90, 150)})
    for part in ("foot", "knee"):
        assert strike_release(slow, part) is None
        a, b = start_timings(slow, 24, 150, part=part), start_timings(slow, 24, 150, part="palm")
        assert (a.appear_frame, a.release_frame, a.vanish_start, a.vanish_end, a.behaviour, a.route_end) == (
            b.appear_frame, b.release_frame, b.vanish_start, b.vanish_end, b.behaviour, None)


@pytest.mark.parametrize("part", ["palm", "shoulder", "head"])
def test_palm_shoulder_and_head_never_use_the_strike_rule(part):
    pts = moving(150, KICK, start=(300.0, 400.0))
    t = start_timings(pts, 24, 150, part=part)
    old = start_timings(pts, 24, 150, part="palm")
    assert (t.appear_frame, t.release_frame, t.vanish_start, t.vanish_end, t.fallback, t.behaviour, t.route_end) == (
        old.appear_frame, old.release_frame, old.vanish_start, old.vanish_end, old.fallback, old.behaviour, None)
    cue, _ = new_cue(pts, part) if part != "palm" else api.new_cue_with_warnings(
        "c.mp4", 24, 150, [1920, 1080], "right", SEED, api.track_from_points(pts))
    assert len(cue["route"]) == 1 and cue["release_frame"] == old.release_frame


def test_a_hand_throw_on_a_palm_track_is_unchanged_by_the_rule():
    """The same track as a palm: the steady-run rule, the one-point route. (The 71319 golden test covers the real one.)"""
    pts = moving(200, {f: (2.5, 0.0) for f in range(71, 200)})
    t = start_timings(pts, 24, 200)
    assert (t.appear_frame, t.release_frame, t.vanish_start, t.vanish_end) == (1 + 12, 71, 77, 83) and t.route_end is None
