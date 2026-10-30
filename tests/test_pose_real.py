"""Body points on the real clip 71319 (slow: runs the Pose Landmarker, about 20 s, and one stand-in render).

    pytest -m slow tests/test_pose_real.py -s
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from services.cue import api
from services.cue.cue import BodyPoint, HandSeed
from services.analysis.pose_track import POSE_MODEL_PATH, build_pose_track, get_pose_landmarks
from services import config
from app import create_app

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "data" / "input" / "throw_71319.mov"
LOOK = ROOT / "data" / "look" / "standin_orb.png"
GOLDEN_TRACK = ROOT / "contract" / "golden" / "throw_71319.track.json"  # the palm track the hand tracker made of this clip
CROSS_CHECK_LIMIT = 1.5  # palm widths

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def pose():
    if not CLIP.is_file() or not POSE_MODEL_PATH.is_file():
        pytest.skip("data/input/throw_71319.mov or models/pose_landmarker_heavy.task is not here")
    t0 = time.perf_counter()
    p = get_pose_landmarks(CLIP)
    dt = time.perf_counter() - t0
    print(f"\nPose: {p.info.frames} frames in {dt:.1f} s = {1000 * dt / p.info.frames:.0f} ms/frame")
    return p


def test_pose_finds_the_person_in_every_frame(pose):
    assert pose.info.frames == 193 and pose.xy.shape == (193, 33, 2) and pose.visibility.shape == (193, 33)
    assert pose.present.all()


def test_the_right_fist_follows_the_right_palm(pose):
    """Cross-check: the right fist from Pose against the palm from the hand tracker, frames 40-99."""
    palm = json.loads(GOLDEN_TRACK.read_text(encoding="utf-8"))
    r = build_pose_track(pose, BodyPoint("fist", "right"), HandSeed(60, 1160, 550))
    d = np.array([np.hypot(r.points[f].x - palm[f]["x"], r.points[f].y - palm[f]["y"]) / palm[f]["palm_w"]
                  for f in range(40, 100)])
    print(f"fist right vs palm, frames 40-99: median {np.median(d):.2f}, worst {d.max():.2f} (frame {40 + d.argmax()}) palm widths")
    assert d.max() < CROSS_CHECK_LIMIT
    assert r.warnings == []  # the click on the palm is on the fist


def test_the_left_fist_is_the_other_hand(pose):
    """The person's left is not the mirrored side: at frame 85 the left hand is crossing over the right palm."""
    right = build_pose_track(pose, BodyPoint("fist", "right"), HandSeed(60, 1160, 550)).points
    left = build_pose_track(pose, BodyPoint("fist", "left"), HandSeed(85, 1160, 550)).points
    assert abs(left[40].x - right[40].x) > 100  # apart before the pass (his right hand is on the image's left side)


def test_the_feet_are_out_of_frame_and_that_is_an_error_not_a_crash(pose):
    with pytest.raises(api.CueFXError) as e:
        api.analyze_body_point(CLIP, HandSeed(60, 1000, 1000), BodyPoint("foot", "right"), pose=pose, frames=193)
    assert e.value.code == "track_lost"


def test_a_hold_stand_in_on_the_head_renders(pose, tmp_path):
    """cue/new with the head starts as a hold (the head does not move); the stand-in and Merge render it."""
    if not LOOK.is_file():
        pytest.skip("data/look/standin_orb.png is not here")
    seed = HandSeed(60, 1176, 296)
    res = api.analyze_body_point(CLIP, seed, BodyPoint("head", None), pose=pose, frames=193)
    cue, warnings = api.new_cue_with_warnings("throw_71319.mov", 24, 193, [1920, 1080], "right",
                                              {"f": 60, "x": 1176, "y": 296}, res["track"],
                                              body_point={"part": "head", "side": None})
    assert cue["behaviour"] == "hold" and cue["version"] == "2.4" and [w["code"] for w in warnings] == ["started_as_hold"]
    assert [w for w in api.validate(cue) if w["severity"] == "error"] == []

    c = TestClient(create_app(config.Settings(data_dir=tmp_path / "data", look_image=LOOK)))
    up = c.post("/api/clips", files={"file": (CLIP.name, CLIP.read_bytes(), "video/quicktime")})
    assert up.status_code == 201, up.text
    t0 = time.perf_counter()
    r = c.post("/api/standin?wait=1", json={"cue": cue, "track": res["track"], "clip_id": up.json()["clip_id"]})
    print(f"hold stand-in on the head: {time.perf_counter() - t0:.0f} s")
    assert r.status_code == 200, r.text
    assert api.schema_errors("standin_result", r.json()) == []
    out = ROOT / "data" / "debug"
    out.mkdir(parents=True, exist_ok=True)
    video = out / "task32_throw_71319_head_hold_hardcomp.mp4"
    video.write_bytes(c.get(r.json()["files"]["hardcomp_download"]).content)
    (out / "task32_throw_71319_head_hold_cue.json").write_text(json.dumps(cue, indent=1), encoding="utf-8")
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for f in range(0, 193, 16):  # every 16th frame, with the frame number
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            img = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)
            cv2.putText(img, f"f{f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(img)
    cap.release()
    assert len(tiles) >= 12
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
    cv2.imwrite(str(out / "task32_throw_71319_head_hold_sheet.jpg"),
                np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
