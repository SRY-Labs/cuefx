"""Orb path: held, release, flying, visibility, and the proposed route timing."""
import math

import numpy as np
import pytest

from conftest import PALM_W, make_cue, moving_track, with_lost_gap
from services.cue.cue import Correction, RoutePoint, Vanish, validate
from services.cue.route import (corrected_track, orb_path, propose_route_timing, release_point, release_source_frame,
                         visibility)


def at(path, f):
    return next(s for s in path if s[0] == f)


def dist(a, b) -> float:
    return math.hypot(a[1] - b[1], a[2] - b[2])


# --------------------------------------------------------------------------- held and release

def test_path_covers_exactly_the_effect_range(cue):
    path = orb_path(cue)
    assert [s[0] for s in path] == list(range(32, 153))


def test_held_orb_follows_the_track_plus_anchor(cue):
    cue.track = moving_track(speed=2.0)
    path = orb_path(cue)
    for f in (32, 60, 106):
        _, x, y, d, _ = at(path, f)
        assert (x, y, d) == (1000 + 2.0 * f, 600 - 70, 110)


def test_correction_falloff_carries_through_to_the_held_path(cue):
    cue.corrections = [Correction(60, 1020, 600)]  # +20 px in x at frame 60
    cue.correction_falloff = 12
    path = orb_path(cue)
    assert at(path, 60)[1] == 1020
    assert at(path, 66)[1] == pytest.approx(1010)   # half way through the fade
    assert at(path, 72)[1] == pytest.approx(1000) and at(path, 80)[1] == 1000
    assert at(path, 48)[1] == pytest.approx(1000) and at(path, 54)[1] == pytest.approx(1010)
    assert [p.source for p in corrected_track(cue)[49:72]] == ["corrected"] * 23
    cue.correction_falloff = 3
    assert at(orb_path(cue), 66)[1] == 1000  # a shorter falloff reaches the held path too


def test_no_jump_at_release_while_the_hand_moves(cue):
    cue.track = moving_track(speed=8.0)
    path = orb_path(cue)
    step = dist(at(path, 106), at(path, 107))  # last held frame to first flying frame
    assert step < PALM_W
    assert step == pytest.approx(8.0)  # the release point is the held position at release_frame
    assert (at(path, 107)[1], at(path, 107)[2]) == (1000 + 8.0 * 107, 530)


def test_release_inside_a_lost_gap_uses_the_last_tracked_frame_before_it(cue):
    cue.track = with_lost_gap(moving_track(speed=8.0), 107, 111)  # 107..110 interpolated across a 900 px jump
    assert release_source_frame(cue) == 106
    path = orb_path(cue)
    assert release_point(cue) == (1000 + 8.0 * 106, 530)
    assert (at(path, 107)[1], at(path, 107)[2]) == (1000 + 8.0 * 106, 530)  # not partway across the jump
    assert dist(at(path, 106), at(path, 107)) < PALM_W


def test_a_correction_inside_a_lost_gap_anchors_the_held_path_and_moves_nothing_outside_it(cue):
    base = with_lost_gap(moving_track(speed=8.0), 100, 104)
    cue.track = base
    plain = {s[0]: s for s in orb_path(cue)}
    cue.corrections = [Correction(102, 1700, 400)]
    path = {s[0]: s for s in orb_path(cue)}
    assert path[102][1:3] == (1700, 400 - 70)                              # the anchor, plus the anchor offset
    assert [p.source for p in corrected_track(cue)[100:104]] == ["corrected"] * 4
    for f in list(range(32, 100)) + list(range(104, 153)):
        assert path[f][1:] == plain[f][1:], f                              # no falloff: nothing outside the gap moved


def test_release_inside_a_corrected_lost_gap_is_taken_from_the_release_frame(cue):
    cue.track = with_lost_gap(moving_track(speed=8.0), 107, 111)
    assert release_source_frame(cue) == 106                                # not corrected: the guess is not trusted
    cue.corrections = [Correction(108, 1500, 600)]
    assert release_source_frame(cue) == 107                                # the artist has been into the gap
    assert validate(cue)[1] == []


def test_release_source_frame_is_the_release_frame_when_it_is_tracked(cue):
    assert release_source_frame(cue) == 107
    cue.track = with_lost_gap(moving_track(), 90, 94)  # a lost gap, but not at release
    assert release_source_frame(cue) == 107


def test_a_given_release_position_is_used_as_it_is(cue):
    cue.route[0] = RoutePoint(107, 900, 500, 110)
    assert at(orb_path(cue), 107)[1:3] == (900, 500)


def test_invalid_cue_cannot_be_routed(cue):
    cue.release_frame = 20
    with pytest.raises(ValueError, match="release_frame"):
        orb_path(cue)


# --------------------------------------------------------------------------- flying

def test_route_passes_exactly_through_every_point_at_its_frame(cue):
    path = orb_path(cue)
    for p in cue.route[1:]:
        _, x, y, d, _ = at(path, p.f)
        assert (x, y, d) == pytest.approx((p.x, p.y, p.diameter_px), abs=1e-9)
    assert at(path, 107)[1:4] == pytest.approx((1000, 530, 110))


def test_diameter_is_linear_between_points(cue):
    path = orb_path(cue)
    assert at(path, 111)[3] == pytest.approx(110 + (60 - 110) * 4 / 9)
    assert at(path, 123)[3] == pytest.approx(60 + (25 - 60) * 7 / 14)


def test_orb_holds_still_after_the_last_route_point(cue):
    cue.vanish = Vanish(140, 150)
    path = orb_path(cue)
    for f in range(130, 153):
        assert at(path, f)[1:4] == (420, 360, 25)


def test_two_points_make_a_straight_line_with_uniform_timing(cue):
    cue.route = [RoutePoint(107, None, None, 110), RoutePoint(117, 400, 530, 60)]  # level with the release point
    path = orb_path(cue)
    xs = [at(path, f)[1] for f in range(107, 118)]
    np.testing.assert_allclose(xs, np.linspace(1000, 400, 11), atol=1e-9)
    assert all(at(path, f)[2] == pytest.approx(530) for f in range(107, 118))


def test_curve_is_smooth_through_a_middle_point(cue):
    cue.route = [RoutePoint(107, None, None, 110), RoutePoint(117, 700, 400, 60), RoutePoint(127, 600, 100, 30)]
    path = orb_path(cue)
    v_in = np.array(at(path, 117)[1:3]) - np.array(at(path, 116)[1:3])
    v_out = np.array(at(path, 118)[1:3]) - np.array(at(path, 117)[1:3])
    cosine = v_in @ v_out / (np.linalg.norm(v_in) * np.linalg.norm(v_out))
    assert cosine > 0.9  # no corner at the middle point


# --------------------------------------------------------------------------- visibility

def test_visibility_fade_in_vanish_ramp_and_zero_outside(cue):
    cue.vanish = Vanish(128, 136)
    cue.grade = type(cue.grade)(**{**vars(cue.grade), "fade_in_frames": 4})
    vis = {f: visibility(cue, f) for f in range(32, 153)}
    assert all(vis[f] == 0 for f in range(32, 40))                       # before appear_frame
    assert [vis[f] for f in range(40, 45)] == pytest.approx([0.25, 0.5, 0.75, 1.0, 1.0])
    assert all(vis[f] == 1 for f in range(44, 129))                      # up to vanish.start
    assert [vis[f] for f in range(128, 136)] == pytest.approx([1 - k / 8 for k in range(8)])
    assert all(vis[f] == 0 for f in range(136, 153))                     # from vanish.end
    assert [s[4] for s in orb_path(cue)] == [vis[f] for f in range(32, 153)]


def test_no_fade_in_means_full_visibility_at_appear_frame(cue):
    cue.grade = type(cue.grade)(**{**vars(cue.grade), "fade_in_frames": 0})
    assert visibility(cue, 39) == 0 and visibility(cue, 40) == 1


# --------------------------------------------------------------------------- route timing

def test_timing_straight_route_with_constant_diameter_is_equal_speed():
    pts = [(0, 0, 50), (100, 0, 50), (200, 0, 50), (300, 0, 50)]
    assert propose_route_timing(pts, 10, 40) == [10, 20, 30, 40]


def test_timing_gives_a_shrinking_orb_more_frames_per_pixel():
    pts = [(0, 0, 100), (100, 0, 50), (200, 0, 25), (300, 0, 12.5)]  # same 100 px steps, shrinking
    frames = propose_route_timing(pts, 0, 60)
    assert frames[0] == 0 and frames[-1] == 60
    gaps = np.diff(frames)
    assert all(gaps > 0) and gaps[0] < gaps[1] < gaps[2]  # later points take more frames for the same pixels


def test_timing_keeps_frames_strictly_increasing_when_points_are_close():
    frames = propose_route_timing([(0, 0, 10), (0.001, 0, 10), (0.002, 0, 10), (1000, 0, 10)], 0, 6)
    assert frames[0] == 0 and frames[-1] == 6
    assert all(np.diff(frames) >= 1)


def test_timing_errors():
    with pytest.raises(ValueError, match="at least"):
        propose_route_timing([(0, 0, 10), (5, 0, 10), (9, 0, 10)], 0, 1)
    with pytest.raises(ValueError, match="two route points"):
        propose_route_timing([(0, 0, 10)], 0, 10)
    with pytest.raises(ValueError, match="diameters"):
        propose_route_timing([(0, 0, 10), (5, 0, 0)], 0, 10)
