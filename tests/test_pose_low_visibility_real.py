"""Task 35 on the real clips kick.mp4 and punch.mp4 (slow: runs the Pose Landmarker once per clip unless
data/debug/pose_<clip>.npz is there).

    pytest -m slow tests/test_pose_low_visibility_real.py -s
"""
from pathlib import Path

import pytest

from services.cue.cue import BodyPoint, HandSeed
from services.analysis.pose_track import POSE_MODEL_PATH, body_point_series, build_pose_track, cached_pose_landmarks

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow

# clip, part, side: frames missing and longest gap before task 35 (measured at commit 165f06c)
BEFORE = {
    ("kick", "foot", "right"): (28, 28),
    ("kick", "foot", "left"): (11, 6),
    ("punch", "fist", "left"): (56, 44),
    ("punch", "fist", "right"): (29, 29),
}


@pytest.mark.parametrize("key", BEFORE, ids=[" ".join(k) for k in BEFORE])
def test_no_clip_gets_worse_and_the_long_gaps_are_gone(key):
    clip, part, side = key
    path = ROOT / "data" / "input" / f"{clip}.mp4"
    if not path.is_file() or not POSE_MODEL_PATH.is_file():
        pytest.skip(f"{path} or the pose model is not here")
    pose = cached_pose_landmarks(path, ROOT / "data" / "debug" / f"pose_{clip}.npz")
    res = build_pose_track(pose, BodyPoint(part, side), HandSeed(0, 0, 0))
    missing = sum(p.source != "tracked" for p in res.points)
    longest = max((b - a for a, b in res.gaps), default=0)
    lost = sum(b - a for a, b in res.lost)
    low = int(body_point_series(pose, part, side).low.sum())
    print(f"\n{clip} {part} {side}: missing {BEFORE[key][0]} -> {missing}, longest gap {BEFORE[key][1]} -> {longest}, "
          f"lost frames {lost}, low-visibility frames {low}")
    assert missing <= BEFORE[key][0] and longest <= BEFORE[key][1]
    assert lost == 0
    assert missing <= 2  # the punching fist keeps two frames (53, 54) that jump
