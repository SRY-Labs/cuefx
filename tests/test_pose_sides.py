"""Pose fixes and the side by click (tasks 31 and 32 follow-ups): a landmark off the plate is missing, the track_lost
message names the body point, the click picks the side of a fist, foot, knee or shoulder."""
import numpy as np
import pytest

from services.cue import api
from services.cue.cue import BodyPoint, HandSeed
from services.analysis.pose_track import body_point_series, build_pose_track, side_from_click
from test_pose_track import FRAMES, lm_xy, mean_of, static_pose, hide

CLIP = "clip.mp4"  # never opened: the pose is given


# --------------------------------------------------------------------------- landmarks off the plate

@pytest.mark.parametrize("off", [(-500.0, 500.0), (500.0, -500.0), (2500.0, 500.0), (500.0, 1600.0), (5000.0, 5000.0)])  # all beyond the task 35 edge margin
def test_a_landmark_outside_the_plate_is_missing_whatever_its_visibility(off):
    pose = static_pose()
    pose.xy[10:13, 16] = off              # landmark 16 (right wrist) off the plate, visibility still 1
    assert (pose.visibility[10:13, 16] == 1).all()
    valid = body_point_series(pose, "fist", "right").valid
    assert not valid[10:13].any() and valid[:10].all() and valid[13:].all()
    assert body_point_series(pose, "fist", "left").valid.all()           # only the right fist uses landmark 16


def test_the_size_landmark_counts_too_and_the_plate_edge_is_inside():
    pose = static_pose()
    pose.xy[5, 20] = (-500.0, 400.0)                                       # landmark 20 is in the right fist's size pair
    pose.xy[6, 16] = (0.0, 0.0)                                          # exactly the corner: on the plate
    valid = body_point_series(pose, "fist", "right").valid
    assert not valid[5] and valid[6]


def test_an_off_plate_gap_is_filled_and_flagged_like_any_missing_frames():
    pose = static_pose()
    pose.xy[20:40, 16] = (-500.0, 300.0)                                  # 20 frames: a lost gap
    res = build_pose_track(pose, BodyPoint("fist", "right"), HandSeed(5, *mean_of(16, 20, 18)))
    assert res.lost == [(20, 40)] and all(res.points[f].lost for f in range(20, 40))


# --------------------------------------------------------------------------- the track_lost message

PALM_TEXT = ("CueFX lost the hand on 101 of 200 frames. Pick the hand again on a frame in the middle of the hold, "
             "where the palm is clear.")


def test_the_palm_message_is_the_same_text_as_before():
    assert api.error("track_lost", lost=101, total=200).message == PALM_TEXT


@pytest.mark.parametrize("bp,name", [(BodyPoint("head", None), "the head"), (BodyPoint("foot", "right"), "the right foot"),
                                     (BodyPoint("knee", "left"), "the left knee")])
def test_track_lost_names_the_body_point(bp, name):
    pose = hide(static_pose(), range(FRAMES), no_pose=True)              # never found
    with pytest.raises(api.CueFXError) as e:
        api.analyze_body_point(CLIP, HandSeed(5, 100, 300), bp, pose=pose, frames=FRAMES)
    assert e.value.code == "track_lost" and f"lost {name} on 60 of 60" in e.value.message
    assert "the hand" not in e.value.message and "palm" not in e.value.message


def test_track_lost_by_share_names_the_body_point_too():
    pose = hide(static_pose(), range(10, FRAMES), no_pose=True)          # found in 10 of 60 frames
    with pytest.raises(api.CueFXError) as e:
        api.analyze_body_point(CLIP, HandSeed(5, *mean_of(7, 8)), BodyPoint("head", None), pose=pose, frames=FRAMES)
    assert e.value.code == "track_lost" and "lost the head on 50 of 60" in e.value.message


# --------------------------------------------------------------------------- side by click

LEFT_FIST, RIGHT_FIST = mean_of(15, 19, 17), mean_of(16, 20, 18)


def click(p, dx=0.0):
    return HandSeed(5, float(p[0]) + dx, float(p[1]))


def test_the_nearer_side_wins_for_every_part_with_sides():
    pose = static_pose()
    for part, left, right in (("fist", (15, 19, 17), (16, 20, 18)), ("foot", (29, 31), (30, 32)),
                              ("knee", (25,), (26,)), ("shoulder", (11,), (12,))):
        assert side_from_click(pose, part, click(mean_of(*left))) == "left", part
        assert side_from_click(pose, part, click(mean_of(*right))) == "right", part
    assert side_from_click(pose, "head", click(LEFT_FIST)) is None       # the head has no side


def test_a_click_that_disagrees_with_the_given_side_wins_with_an_info_warning():
    pose = static_pose()
    res = api.analyze_body_point(CLIP, click(RIGHT_FIST), BodyPoint("fist", "left"), pose=pose, frames=FRAMES)
    assert res["body_point"] == {"part": "fist", "side": "right"}
    [w] = res["warnings"]
    assert w["code"] == "side_from_click" and w["severity"] == "info" and w["frames"] == [5, 6]
    assert "right fist" in w["message"] and "left" in w["message"] and api.schema_errors("warning", w) == []
    assert res["track"][0]["x"] == pytest.approx(RIGHT_FIST[0], abs=0.01)  # the right fist is what is tracked
    assert api.schema_errors("analyze_result", res) == []


def test_a_click_that_agrees_changes_nothing_and_says_nothing():
    res = api.analyze_body_point(CLIP, click(LEFT_FIST), BodyPoint("fist", "left"), pose=static_pose(), frames=FRAMES)
    assert res["body_point"] == {"part": "fist", "side": "left"} and res["warnings"] == []


def test_the_head_and_the_palm_are_unchanged():
    res = api.analyze_body_point(CLIP, click(mean_of(7, 8)), BodyPoint("head", None), pose=static_pose(), frames=FRAMES)
    assert res["body_point"] == {"part": "head", "side": None} and res["warnings"] == []
    from services.cue.cue import TrackPoint
    assert "body_point" not in api.analyze_result([TrackPoint(0, 1.0, 2.0, "tracked", 90.0, 0.9)])


def test_if_a_side_cannot_be_judged_the_given_side_stays():
    pose = hide(static_pose(), range(FRAMES), landmarks=[16, 18, 20])    # the right fist is not visible anywhere
    assert side_from_click(pose, "fist", click(RIGHT_FIST)) is None
    pose2 = hide(static_pose(), [5], landmarks=[15, 17, 19])             # the left fist is hidden in the seed frame only
    assert side_from_click(pose2, "fist", click(LEFT_FIST)) is None
    assert side_from_click(static_pose(), "fist", HandSeed(500, 0, 0)) is None  # a seed frame outside the clip


def test_cue_new_carries_the_side_the_click_chose():
    res = api.analyze_body_point(CLIP, click(RIGHT_FIST), BodyPoint("fist", "left"), pose=static_pose(), frames=FRAMES)
    cue, _ = api.new_cue_with_warnings("c.mp4", 24, FRAMES, [1920, 1080], "right", {"f": 5, "x": 1.0, "y": 2.0},
                                       res["track"], body_point=res["body_point"])
    assert cue["body_point"] == {"part": "fist", "side": "right"} and cue["version"] == "2.4"
