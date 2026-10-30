"""Selection on real detections: frames 60-140 of clip 71319 (raw MediaPipe candidates, no selection results).

The tracker once followed his other hand after the left-hand pass (frames 84-88) and no test noticed,
because every test used synthetic hands.
"""
import csv
import math
from pathlib import Path

import pytest

from services.cue.cue import HandSeed
from services.analysis.track import VideoInfo, build_track, select_hands
from test_track import hand

FIXTURE = Path(__file__).parent / "fixtures" / "throw_71319_candidates_60_140.csv"
FRAMES = 141


@pytest.fixture(scope="module")
def clip():
    hands = [[] for _ in range(FRAMES)]  # frames 0-59 are empty: the seed is at frame 60
    for r in csv.DictReader(open(FIXTURE, encoding="utf-8")):
        hands[int(r["frame"])].append(hand(float(r["x"]), float(r["y"]), r["label"], float(r["palm_w"]), float(r["score"])))
    sel = select_hands(hands, HandSeed(60, 1160, 550))
    return hands, sel, build_track(VideoInfo(1920, 1080, 24.0, FRAMES, FRAMES), hands, sel)


def test_frames_85_to_88_are_not_taken_from_the_other_hand(clip):
    _, _, res = clip
    assert all(res.points[f].source != "tracked" for f in range(85, 89))


def test_the_chosen_hand_stays_on_his_right_hand_until_the_throw(clip):
    _, _, res = clip
    xs = [res.points[f].x for f in range(89, 105) if res.points[f].source == "tracked"]
    assert len(xs) >= 12 and all(1150 <= x <= 1350 for x in xs)


def test_the_hand_is_reacquired_by_frame_111_near_the_arm_end(clip):
    _, _, res = clip
    first = next(p for p in res.points if p.f >= 107 and p.source == "tracked")
    assert first.f <= 111
    p = res.points[111]
    assert p.source == "tracked" and math.hypot(p.x - 425, p.y - 393) < 40


def test_no_tracked_frame_is_on_the_other_chains_tracklets(clip):
    hands, sel, res = clip
    for f in range(60, FRAMES):
        if res.points[f].source == "tracked":
            c = sel.chosen[f]
            assert sel.tracklets[sel.cand_tracklet[f][c]].chain == "chosen", f
            assert c != sel.other[f], f
