"""Body points over HTTP (task 32): /api/analyze and /api/cue/new with `body_point`, the Pose cache shared by all body
points of a clip, and the palm path (a missing or explicit palm body point) staying on the hand tracker."""
import numpy as np
import pytest

from server_helpers import FRAMES, assert_error, assert_fits, client, clip_file, golden, settings, uploaded, uploaded_template  # noqa: F401
from services.cue import api
from services.analysis import pose_track
from services.analysis.pose_track import PoseLandmarks
from services.analysis.track import VideoInfo

SEED = {"frame": 6, "x": 700.0, "y": 400.0}
EAR = float(np.hypot(40, 9))  # ear <-> ear in fake_pose: the head's size measure


def fake_pose(n: int = FRAMES) -> PoseLandmarks:
    """A still person (landmark i at (100 + 40 i, 300 + 9 i)), in every frame, visibility 1."""
    one = np.array([[100.0 + 40 * i, 300.0 + 9 * i] for i in range(33)])
    return PoseLandmarks(VideoInfo(1920, 1080, 24.0, n, n), np.tile(one, (n, 1, 1)), np.ones((n, 33)))


@pytest.fixture
def pose_calls(monkeypatch):
    """pose_track.get_pose_landmarks replaced by a counter; the palm tracker is replaced by one that fails."""
    calls = []

    def fake(clip, model=None):
        calls.append(clip)
        return fake_pose()

    monkeypatch.setattr(pose_track, "get_pose_landmarks", fake)
    return calls


def body(clip_id, part, side, **changes):
    return {"clip_id": clip_id, "hand": "right", "hand_seed": SEED, "body_point": {"part": part, "side": side}, **changes}


def test_analyze_a_body_point_returns_a_contract_result(client, uploaded, pose_calls):
    r = client.post("/api/analyze", json=body(uploaded["clip_id"], "head", None))
    assert r.status_code == 200
    res = r.json()
    assert_fits("analyze_result", res)
    assert len(res["track"]) == FRAMES
    assert res["track"][0]["x"] == pytest.approx(100 + 40 * 7.5, abs=0.01) and res["track"][0]["y"] == pytest.approx(300 + 9 * 7.5, abs=0.01)
    assert res["track"][0]["palm_w"] == pytest.approx(EAR, abs=0.01)  # ear <-> ear, in plate px
    assert len(pose_calls) == 1 and pose_calls[0].name == "plate.mp4"   # Pose reads the conformed plate


@pytest.mark.parametrize("part,side", [("foot", "left"), ("foot", "right"), ("head", None)])
def test_a_body_point_needs_no_click_and_returns_a_real_tracked_seed(client, uploaded, pose_calls, part, side):
    request = {"clip_id": uploaded["clip_id"], "hand": "right", "body_point": {"part": part, "side": side}}
    analyzed = client.post("/api/analyze", json=request)
    assert analyzed.status_code == 200
    result = analyzed.json()
    assert_fits("analyze_result", result)
    assert result["body_point"] == {"part": part, "side": side} and result["warnings"] == []
    seed = result["body_point_seed"]
    point = result["track"][seed["frame"]]
    assert point["source"] == "tracked" and seed["x"] == point["x"] and seed["y"] == point["y"]

    made = client.post("/api/cue/new", json=request)
    assert made.status_code == 200
    cue = made.json()
    assert_fits("cue", cue)
    assert cue["body_point"] == {"part": part, "side": side}
    assert cue["hand_seed"] == {"f": seed["frame"], "x": seed["x"], "y": seed["y"]}


def test_a_palm_still_requires_a_click(client, uploaded, pose_calls):
    cid = uploaded["clip_id"]
    assert_error(client.post("/api/analyze", json={"clip_id": cid, "hand": "right"}), 400, "input_invalid")
    assert_error(client.post("/api/cue/new", json={"clip_id": cid, "hand": "right",
                                                    "body_point": {"part": "palm", "side": "right"}}),
                 400, "input_invalid")
    assert pose_calls == []


def test_pose_runs_once_per_clip_whatever_the_body_point(client, uploaded, pose_calls, settings):
    cid = uploaded["clip_id"]
    for part, side in [("head", None), ("fist", "right"), ("fist", "left"), ("knee", "right"), ("shoulder", "left")]:
        assert client.post("/api/analyze", json=body(cid, part, side)).status_code == 200
    assert len(pose_calls) == 1
    assert len(list((settings.cache_dir / cid).glob("pose_*.npz"))) == 1


def test_a_body_point_analyze_is_cached_by_part_and_side(client, uploaded, pose_calls, monkeypatch):
    cid, runs = uploaded["clip_id"], []
    real = api.analyze

    def counting(*a, **k):
        runs.append(k.get("body_point"))
        return real(*a, **k)

    monkeypatch.setattr(api, "analyze", counting)
    first = client.post("/api/analyze", json=body(cid, "fist", "right")).json()
    assert client.post("/api/analyze", json=body(cid, "fist", "right")).json() == first
    assert len(runs) == 1
    client.post("/api/analyze", json=body(cid, "fist", "left"))
    client.post("/api/analyze", json=body(cid, "head", None))
    assert len(runs) == 3 and len(pose_calls) == 1


def test_a_palm_body_point_is_the_hand_trackers_and_has_the_old_cache_key(client, uploaded, pose_calls, monkeypatch):
    runs = []

    def fake(clip_path, seed, model=None, **kw):
        runs.append(kw)
        return api.analyze_result(api.points_from_track(golden("throw_71319.track.json")))

    monkeypatch.setattr(api, "analyze", fake)
    cid = uploaded["clip_id"]
    plain = {"clip_id": cid, "hand": "right", "hand_seed": SEED}
    first = client.post("/api/analyze", json=plain).json()
    explicit = client.post("/api/analyze", json={**plain, "body_point": {"part": "palm", "side": "right"}}).json()
    assert explicit == first and len(runs) == 1 and "body_point" not in runs[0]  # one cache entry, the palm path
    assert pose_calls == []


def test_a_bad_body_point_is_body_point_invalid(client, uploaded, pose_calls):
    cid = uploaded["clip_id"]
    for bp in ({"part": "palm", "side": "left"}, {"part": "fist", "side": None}, {"part": "elbow", "side": "left"},
               {"part": "fist", "side": "up"}):
        r = client.post("/api/analyze", json={"clip_id": cid, "hand": "right", "hand_seed": SEED, "body_point": bp})
        assert_error(r, 400, "body_point_invalid")
        r = client.post("/api/cue/new", json={"clip_id": cid, "hand": "right", "hand_seed": SEED, "body_point": bp})
        assert_error(r, 400, "body_point_invalid")
    assert pose_calls == []
    r = client.post("/api/analyze", json={"clip_id": cid, "hand": "right", "hand_seed": SEED, "body_point": {"side": "left"}})
    assert_error(r, 400, "input_invalid")  # a malformed body (no part) is an ordinary bad request


def test_a_far_click_is_a_warning_in_the_analyze_result(client, uploaded, pose_calls):
    r = client.post("/api/analyze", json=body(uploaded["clip_id"], "fist", "right", hand_seed={**SEED, "x": 1800.0}))
    [w] = r.json()["warnings"]
    assert w["code"] == "seed_far_from_body_point" and w["severity"] == "warning" and "right fist" in w["message"]
    assert_fits("warning", w)


def test_cue_new_with_a_body_point(client, uploaded, pose_calls):
    import json

    r = client.post("/api/cue/new", json=body(uploaded["clip_id"], "head", None))
    assert r.status_code == 200
    cue = r.json()
    assert_fits("cue", cue)
    assert cue["version"] == "2.4" and cue["body_point"] == {"part": "head", "side": None} and cue["hand"] == "right"
    assert cue["diameter_px"] == round(EAR) and cue["anchor_offset"] == [0, -round(1.1 * EAR)]  # head: 1.0 x size, up 1.1 x size
    codes = {w["code"] for w in json.loads(r.headers["X-CueFX-Warnings"])}
    assert codes <= {"started_as_hold", "placeholder_timings", "hold_shortened"}
    assert [w for w in api.validate(cue) if w["severity"] == "error"] == []
    assert cue["track"][0]["palm_w"] == pytest.approx(EAR, abs=0.01)


def test_cue_new_without_a_body_point_is_a_v23_cue_with_no_body_point(client, uploaded, monkeypatch):
    monkeypatch.setattr(api, "analyze", lambda *a, **k: api.analyze_result(
        api.points_from_track(golden("throw_71319.track.json")[:FRAMES])))
    r = client.post("/api/cue/new", json={"clip_id": uploaded["clip_id"], "hand": "right", "hand_seed": SEED})
    assert r.status_code == 200 and r.json()["version"] == "2.3" and "body_point" not in r.json()


def test_the_click_picks_the_side_in_analyze_and_cue_new(client, uploaded, pose_calls):
    """Task 31: the fists of fake_pose are at x 780 (left) and 820 (right); a click at x 830 is on the right one."""
    import json

    click = {"frame": 6, "x": 830.0, "y": 460.0}
    r = client.post("/api/analyze", json=body(uploaded["clip_id"], "fist", "left", hand_seed=click))
    assert r.status_code == 200
    res = r.json()
    assert_fits("analyze_result", res)
    assert res["body_point"] == {"part": "fist", "side": "right"}
    [w] = res["warnings"]
    assert w["code"] == "side_from_click" and w["severity"] == "info" and "right fist" in w["message"]
    assert res["track"][0]["x"] == pytest.approx(820, abs=0.01)

    r = client.post("/api/cue/new", json=body(uploaded["clip_id"], "fist", "left", hand_seed=click))
    assert r.status_code == 200 and r.json()["body_point"] == {"part": "fist", "side": "right"}
    assert [w["code"] for w in json.loads(r.headers["X-CueFX-Warnings"]) if w["code"] == "side_from_click"] == ["side_from_click"]

    agree = client.post("/api/analyze", json=body(uploaded["clip_id"], "fist", "right", hand_seed=click)).json()
    assert agree["body_point"] == {"part": "fist", "side": "right"} and agree["warnings"] == []
    head = client.post("/api/analyze", json=body(uploaded["clip_id"], "head", None)).json()
    assert head["body_point"] == {"part": "head", "side": None}
