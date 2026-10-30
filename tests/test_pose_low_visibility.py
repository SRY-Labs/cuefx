"""Task 35: low-visibility Pose positions are used, with a jump guard and a small margin past the plate edge.
Synthetic landmarks only; the real-clip numbers are in test_pose_low_visibility_real.py (slow)."""
import numpy as np

from services.cue.cue import BodyPoint, HandSeed
from services.analysis.pose_track import (EDGE_MARGIN_SIZES, JUMP_LIMIT_SIZES, LOW_VIS_MIN, MIN_CONF, body_point_series,
                              build_pose_track)
from test_pose_track import FRAMES, d, mean_of, static_pose

FIST = (16, 18, 20)                       # fist right: position and size landmarks
SEED = HandSeed(0, *mean_of(*FIST))
SIZE = d(16, 20)


def set_vis(pose, frames, v):
    """Visibility `v` for the fist-right landmarks in `frames`."""
    for f in frames:
        pose.visibility[f, list(FIST)] = v
    return pose


def shift(pose, f, dx, dy):
    pose.xy[f, list(FIST)] += [dx, dy]


def test_the_constants():
    assert (LOW_VIS_MIN, MIN_CONF, JUMP_LIMIT_SIZES, EDGE_MARGIN_SIZES) == (0.2, 0.5, 3.0, 1.0)


def test_visibility_0_3_uses_the_raw_position_as_tracked_with_that_conf():
    pose = set_vis(static_pose(), range(10, 14), 0.3)
    shift(pose, 11, 0.5 * SIZE, 0)          # a small real move: the raw position must come through
    r = build_pose_track(pose, BodyPoint("fist", "right"), SEED)
    assert r.gaps == [] and r.lost == []
    assert all(p.source == "tracked" and not p.lost for p in r.points)
    assert [round(p.conf, 3) for p in r.points[10:14]] == [0.3] * 4 and r.points[0].conf == 1.0
    s = body_point_series(pose, "fist", "right")
    assert list(np.flatnonzero(s.low)) == [10, 11, 12, 13] and s.valid.all()
    np.testing.assert_allclose(s.xy[11], mean_of(*FIST) + [0.5 * SIZE, 0])  # the raw position, not a fill


def test_the_boundaries_0_2_is_low_and_0_5_is_normal():
    pose = static_pose()
    pose.visibility[5, list(FIST)] = 0.2
    pose.visibility[6, list(FIST)] = 0.5
    pose.visibility[7, list(FIST)] = 0.19
    s = body_point_series(pose, "fist", "right")
    assert s.low[5] and not s.low[6] and s.valid[5] and s.valid[6] and not s.valid[7]


def test_visibility_0_1_is_filled_in_as_before():
    pose = set_vis(static_pose(), range(10, 13), 0.1)
    r = build_pose_track(pose, BodyPoint("fist", "right"), SEED)
    assert r.gaps == [(10, 13)]
    assert [p.source for p in r.points[10:13]] == ["interpolated"] * 3 and all(p.conf == 0 for p in r.points[10:13])


def test_a_weak_frame_that_jumps_five_sizes_is_filled_in_and_three_is_kept():
    pose = set_vis(static_pose(), [10], 0.3)
    shift(pose, 10, 5 * SIZE, 0)
    assert not body_point_series(pose, "fist", "right").valid[10]
    r = build_pose_track(pose, BodyPoint("fist", "right"), SEED)
    assert r.points[10].source == "interpolated" and r.gaps == [(10, 11)]

    pose = set_vis(static_pose(), [10], 0.3)
    shift(pose, 10, 2.9 * SIZE, 0)
    assert body_point_series(pose, "fist", "right").valid[10]


def test_the_jump_limit_is_per_frame_of_distance_from_the_last_accepted_frame():
    """Frames 10-12 are missing; a weak frame 13 may be 3 sizes x 4 frames away from frame 9, not more."""
    pose = static_pose()
    for f in (10, 11, 12):
        pose.visibility[f, list(FIST)] = 0.0
    pose.visibility[13, list(FIST)] = 0.3
    shift(pose, 13, 8 * SIZE, 0)
    assert body_point_series(pose, "fist", "right").valid[13]
    shift(pose, 13, 5 * SIZE, 0)            # now 13 sizes: over 3 x 4
    assert not body_point_series(pose, "fist", "right").valid[13]


def test_a_dropped_weak_frame_does_not_become_the_reference():
    """A weak guess that was refused cannot make the next weak frame (near the real position) look like a jump."""
    pose = set_vis(static_pose(), range(10, 12), 0.3)
    shift(pose, 10, 9 * SIZE, 0)
    s = body_point_series(pose, "fist", "right")
    assert not s.valid[10] and s.valid[11]


def test_a_point_half_a_size_outside_the_plate_is_kept_and_two_sizes_is_missing():
    pose = static_pose()
    pose.xy[10, list(FIST), 0] = np.array([-0.4, -0.5, -0.3]) * SIZE
    assert body_point_series(pose, "fist", "right").valid[10]
    pose.xy[11, list(FIST), 0] = np.array([-2.0, -2.0, -2.0]) * SIZE
    assert not body_point_series(pose, "fist", "right").valid[11]


def test_the_bottom_edge_uses_the_same_margin():
    pose = static_pose()
    h = pose.info.height
    pose.xy[10, list(FIST), 1] = h + 0.8 * SIZE
    pose.xy[11, list(FIST), 1] = h + 1.5 * SIZE
    s = body_point_series(pose, "fist", "right")
    assert s.valid[10] and not s.valid[11]


def test_a_normal_frame_is_unchanged_and_not_low():
    s = body_point_series(static_pose(), "fist", "right")
    assert s.valid.all() and not s.low.any()
