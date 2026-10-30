"""Body-point tracking (task 32) on synthetic landmarks: position and size per part, missing frames, gaps and `lost`,
smoothing, the seed check, body_point validation, default size per part, and that the palm path is not touched.
No Pose model and no clip is needed here; the real-clip checks are in test_pose_real.py (slow)."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from conftest import make_cue
from services.cue import api
from services.analysis import pose_track
from services.cue.cue import (BodyPoint, Cue, HandSeed, TrackPoint, body_point_of, check, cue_from_dict, cue_to_dict,
                       is_palm, load_cue)
from services.analysis.pose_track import (MIN_CONF, PART_SPECS, PoseLandmarks, body_point_series, build_pose_track,
                              cached_pose_landmarks, load_pose_landmarks, save_pose_landmarks)
from services.cue.start_timings import PART_SIZING, start_timings
from services.analysis.track import TrackError, VideoInfo

ROOT = Path(__file__).resolve().parents[1]
FRAMES = 60


def lm_xy(i: int) -> tuple[float, float]:
    """Landmark i of the static synthetic person: every landmark at its own place."""
    return 100.0 + 13.0 * i, 300.0 + 7.0 * i * (1 + i % 3)


def static_pose(frames: int = FRAMES, **hide) -> PoseLandmarks:
    """The same pose in every frame, visibility 1."""
    one = np.array([lm_xy(i) for i in range(33)])
    xy, vis = np.tile(one, (frames, 1, 1)), np.ones((frames, 33))
    return PoseLandmarks(VideoInfo(1920, 1080, 24.0, frames, frames), xy, vis)


def hide(pose: PoseLandmarks, frames, landmarks=None, no_pose=False) -> PoseLandmarks:
    """Frames where the pose is gone (`no_pose`) or the given landmarks are not visible."""
    for f in frames:
        if no_pose:
            pose.xy[f] = np.nan
            pose.visibility[f] = 0.0
        else:
            pose.visibility[f, landmarks] = 0.0
    return pose


def d(a: int, b: int) -> float:
    return float(np.hypot(lm_xy(a)[0] - lm_xy(b)[0], lm_xy(a)[1] - lm_xy(b)[1]))


def mean_of(*ids: int) -> np.ndarray:
    return np.mean([lm_xy(i) for i in ids], axis=0)


# part, side, landmark numbers averaged for the position, expected size: written out here from the task table, not from the code
EXPECTED = [
    ("fist", "left", (15, 19, 17), d(15, 19)),
    ("fist", "right", (16, 20, 18), d(16, 20)),
    ("foot", "left", (29, 31), d(29, 31)),
    ("foot", "right", (30, 32), d(30, 32)),
    ("knee", "left", (25,), 0.4 * d(23, 25)),
    ("knee", "right", (26,), 0.4 * d(24, 26)),
    ("shoulder", "left", (11,), 0.35 * d(11, 12)),
    ("shoulder", "right", (12,), 0.35 * d(11, 12)),
    ("head", None, (7, 8), d(7, 8)),
]


@pytest.mark.parametrize("part,side,ids,size", EXPECTED, ids=[f"{e[0]}-{e[1]}" for e in EXPECTED])
def test_position_and_size_measure_of_every_part(part, side, ids, size):
    s = body_point_series(static_pose(), part, side)
    assert s.valid.all()
    np.testing.assert_allclose(s.xy, np.tile(mean_of(*ids), (FRAMES, 1)))
    np.testing.assert_allclose(s.size, size)


def test_every_part_in_the_table_is_covered():
    assert {(e[0], e[1]) for e in EXPECTED} == {(p, s) for p, spec in PART_SPECS.items() for s in spec.points}


def test_the_head_has_no_side_and_the_other_parts_need_one():
    assert body_point_series(static_pose(), "head", "left").valid.all()  # a side given for the head is ignored
    with pytest.raises(ValueError):
        body_point_series(static_pose(), "fist", None)
    with pytest.raises(ValueError):
        body_point_series(static_pose(), "palm", "right")  # the palm is the hand tracker's, not a pose part


def test_conf_is_the_mean_visibility_of_the_landmarks_used():
    pose = static_pose()
    pose.visibility[:, 20] = 0.4  # fist right uses 16, 18, 20 (position) and 16, 20 (size): 3 landmarks
    s = body_point_series(pose, "fist", "right")
    np.testing.assert_allclose(s.conf, (1 + 1 + 0.4) / 3)
    assert s.valid.all()
    pose.visibility[:, 20] = 0.0
    assert np.allclose(body_point_series(pose, "fist", "right").conf, 2 / 3)  # still above MIN_CONF
    pose.visibility[:, 18] = 0.0
    s = body_point_series(pose, "fist", "right")
    assert s.valid.all() and s.low.all()                                      # 1/3: below MIN_CONF, a low-visibility position (task 35)
    pose.visibility[:, 16] = 0.0
    assert not body_point_series(pose, "fist", "right").valid.any()           # 0 < LOW_VIS_MIN
    assert MIN_CONF == 0.5


def test_knee_conf_includes_the_hip_the_size_comes_from():
    pose = static_pose()
    pose.visibility[:, 23] = 0.0  # left hip
    assert np.allclose(body_point_series(pose, "knee", "left").conf, 0.5)
    assert np.allclose(body_point_series(pose, "knee", "right").conf, 1.0)


def test_a_frame_without_a_pose_is_missing():
    pose = hide(static_pose(), [10, 11], no_pose=True)
    s = body_point_series(pose, "head", None)
    assert list(np.flatnonzero(~s.valid)) == [10, 11]


# --------------------------------------------------------------------------- the track: gaps, lost, smoothing

SEED = HandSeed(30, *mean_of(16, 20, 18))


def test_short_gaps_are_interpolated_and_long_ones_are_lost():
    pose = hide(static_pose(), range(10, 13), landmarks=[16, 18, 20])      # 3 frames: interpolated
    pose = hide(pose, range(30, 40), no_pose=True)                          # 10 frames: lost (> 6)
    r = build_pose_track(pose, BodyPoint("fist", "right"), SEED)
    assert r.gaps == [(10, 13), (30, 40)] and r.lost == [(30, 40)]
    assert [p.source for p in r.points[10:13]] == ["interpolated"] * 3 and not any(p.lost for p in r.points[10:13])
    assert all(p.lost and p.source == "interpolated" for p in r.points[30:40])
    assert all(p.source == "tracked" and not p.lost for p in r.points[:10] + r.points[13:30] + r.points[40:])
    assert all(p.conf == 0 for p in r.points[10:13]) and r.points[0].conf == 1.0
    assert all(abs(p.palm_w - d(16, 20)) < 1e-6 for p in r.points)         # size measure also fills gaps


def test_a_gap_of_exactly_six_frames_is_not_lost_and_seven_is():
    for length, lost in ((6, False), (7, True)):
        r = build_pose_track(hide(static_pose(), range(20, 20 + length), no_pose=True), BodyPoint("head", None), SEED)
        assert any(p.lost for p in r.points) is lost


def test_the_gap_and_smoothing_code_is_the_hand_trackers():
    """No copy: both trackers call services.analysis.track.smooth_and_flag."""
    import inspect
    from services.analysis import track
    assert "smooth_and_flag" in inspect.getsource(track.build_track) and "smooth_and_flag" in inspect.getsource(pose_track)
    assert not hasattr(pose_track, "fill_gaps") and not hasattr(pose_track, "one_euro")


def test_smoothing_is_applied():
    pose = static_pose()
    rng = np.random.default_rng(3)
    pose.xy[:, [16, 18, 20]] += rng.normal(0, 4.0, (FRAMES, 3, 2))
    raw = body_point_series(pose, "fist", "right").xy
    r = build_pose_track(pose, BodyPoint("fist", "right"), HandSeed(30, *raw[30]))
    xs = np.array([p.x for p in r.points])
    assert xs.std() < 0.6 * raw[:, 0].std()
    assert not np.allclose(xs, raw[:, 0])


def test_a_moving_point_is_followed_without_lag():
    pose = static_pose()
    pose.xy[:, 7:9, 0] += (10.0 * np.arange(FRAMES))[:, None]
    r = build_pose_track(pose, BodyPoint("head", None), HandSeed(30, *mean_of(7, 8) + [300, 0]))
    expected = mean_of(7, 8)[0] + 10.0 * np.arange(FRAMES)
    assert np.abs(np.array([p.x for p in r.points]) - expected).max() < 3.0  # zero-lag smoothing


def test_no_pose_in_any_frame_is_track_lost():
    pose = hide(static_pose(), range(FRAMES), no_pose=True)
    with pytest.raises(TrackError) as e:
        build_pose_track(pose, BodyPoint("head", None), SEED)
    assert e.value.code == "track_lost"
    with pytest.raises(api.CueFXError) as e2:
        api.analyze_body_point(Path("clip.mp4"), SEED, BodyPoint("head", None), pose=pose, frames=FRAMES)
    assert e2.value.code == "track_lost"


def test_a_body_point_missing_in_most_frames_is_track_lost():
    pose = hide(static_pose(), range(10, FRAMES), landmarks=[7, 8])
    with pytest.raises(api.CueFXError) as e:
        api.analyze_body_point(Path("clip.mp4"), SEED, BodyPoint("head", None), pose=pose, frames=FRAMES)
    assert e.value.code == "track_lost"


def test_seed_frame_outside_the_clip_is_input_invalid():
    with pytest.raises(TrackError) as e:
        build_pose_track(static_pose(), BodyPoint("head", None), HandSeed(FRAMES, 0, 0))
    assert e.value.code == "input_invalid"


# --------------------------------------------------------------------------- the seed click

def test_a_click_near_the_body_point_gives_no_warning():
    near = mean_of(16, 20, 18) + [2.5 * d(16, 20), 0]  # inside 3 size measures
    r = build_pose_track(static_pose(), BodyPoint("fist", "right"), HandSeed(30, *near))
    assert r.warnings == []


def test_a_click_far_from_the_body_point_warns():
    far = mean_of(16, 20, 18) + [3.5 * d(16, 20), 0]
    res = api.analyze_body_point(Path("clip.mp4"), HandSeed(30, *far), BodyPoint("fist", "right"), pose=static_pose(),
                                 frames=FRAMES)
    [w] = res["warnings"]
    assert w["code"] == "seed_far_from_body_point" and w["severity"] == "warning" and w["frames"] == [30, 31]
    assert "right fist" in w["message"] and w["detail"].startswith("hand_seed") and len(res["track"]) == FRAMES
    assert api.schema_errors("warning", w) == [] and api.schema_errors("analyze_result", res) == []


def test_a_click_in_a_frame_without_the_pose_warns_and_the_track_is_still_made():
    pose = hide(static_pose(), [30], no_pose=True)
    res = api.analyze_body_point(Path("clip.mp4"), SEED, BodyPoint("fist", "right"), pose=pose, frames=FRAMES)
    assert [w["code"] for w in res["warnings"]] == ["seed_far_from_body_point"] and len(res["track"]) == FRAMES


def test_the_side_comes_from_body_point_not_from_the_click():
    """A click on the left fist while the right fist is asked for: the right fist is tracked, with a warning."""
    pose = static_pose()
    pose.xy[:, [15, 19, 17], 0] -= 1000.0  # the left fist is far from the right one
    res = api.analyze_body_point(Path("clip.mp4"), HandSeed(30, *mean_of(15, 19, 17) - [1000, 0]),
                                 BodyPoint("fist", "right"), pose=pose, frames=FRAMES)
    assert res["track"][0]["x"] == pytest.approx(mean_of(16, 20, 18)[0]) and res["warnings"]


def test_analyze_result_size_is_the_median_size_measure():
    res = api.analyze_body_point(Path("clip.mp4"), SEED, BodyPoint("fist", "right"), pose=static_pose(), frames=FRAMES)
    assert res["hand_size_px"] == pytest.approx(d(16, 20), abs=0.05)
    assert res["summary"] == {"tracked": FRAMES, "total": FRAMES, "uncertain_ranges": []}
    assert all(t["palm_w"] == pytest.approx(d(16, 20)) for t in res["track"])


# --------------------------------------------------------------------------- the swappable pose source and its cache

def test_the_pose_cache_keeps_all_landmarks_and_pose_runs_once(tmp_path, monkeypatch):
    calls = []
    pose = static_pose()
    pose.xy[5] = np.nan
    pose.visibility[5] = 0

    def fake(clip, model=None):
        calls.append(clip)
        return pose

    monkeypatch.setattr(pose_track, "get_pose_landmarks", fake)
    cache = tmp_path / "pose.npz"
    first = cached_pose_landmarks(Path("a.mp4"), cache)
    again = cached_pose_landmarks(Path("a.mp4"), cache)
    assert len(calls) == 1 and cache.is_file()
    for p in (first, again):
        assert p.xy.shape == (FRAMES, 33, 2) and p.visibility.shape == (FRAMES, 33)
    np.testing.assert_array_equal(again.xy, pose.xy)
    np.testing.assert_array_equal(again.visibility, pose.visibility)
    assert again.info == pose.info and not again.present[5]
    for part, side in (("fist", "right"), ("head", None), ("knee", "left")):  # other body points: still one Pose run
        build_pose_track(cached_pose_landmarks(Path("a.mp4"), cache), BodyPoint(part, side), SEED)
    assert len(calls) == 1


def test_an_unreadable_cache_runs_pose_again(tmp_path, monkeypatch):
    cache = tmp_path / "pose.npz"
    cache.write_bytes(b"not an npz")
    monkeypatch.setattr(pose_track, "get_pose_landmarks", lambda clip, model=None: static_pose())
    assert cached_pose_landmarks(Path("a.mp4"), cache).xy.shape == (FRAMES, 33, 2)
    assert load_pose_landmarks(cache).xy.shape == (FRAMES, 33, 2)  # and rewrote it


def test_only_one_function_calls_a_pose_model():
    src = (ROOT / "services" / "analysis" / "pose_track.py").read_text(encoding="utf-8")
    assert src.count("PoseLandmarker") == src.count("vision.PoseLandmarker") and "def get_pose_landmarks" in src
    assert src.count("detect_for_video") == 1
    for f in (ROOT / "services").rglob("*.py"):
        if f.name not in ("pose_track.py", "track.py"):
            assert "mediapipe" not in f.read_text(encoding="utf-8"), f.name


# --------------------------------------------------------------------------- body_point on the cue

def cue_dict(**changes) -> dict:
    return cue_to_dict(make_cue(**changes))


def test_an_old_cue_has_no_body_point_and_loads_as_the_palm_of_its_hand():
    c = make_cue()
    assert c.body_point is None and body_point_of(c) == BodyPoint("palm", "right")
    assert "body_point" not in cue_to_dict(c)
    left = replace(make_cue(), hand="left")
    assert body_point_of(left) == BodyPoint("palm", "left")
    for name in ("throw_71319.cue.json",):
        g = json.loads((ROOT / "contract" / "golden" / name).read_text(encoding="utf-8"))
        assert "body_point" not in g and body_point_of(cue_from_dict(g)) == BodyPoint("palm", g["hand"])
        assert "body_point" not in cue_to_dict(cue_from_dict(g))
    assert body_point_of(load_cue(ROOT / "tests" / "fixtures" / "throw_71319.cue.json")).part == "palm"


def test_body_point_round_trips_and_a_head_side_is_null():
    for bp in (BodyPoint("fist", "left"), BodyPoint("head", None), BodyPoint("palm", "right")):
        d = cue_dict(body_point=bp)
        assert d["body_point"] == {"part": bp.part, "side": bp.side}
        assert api.schema_errors("cue", d) == []
        assert cue_to_dict(cue_from_dict(d)) == d and cue_from_dict(d).body_point == bp


def test_validate_accepts_every_good_body_point_and_names_the_bad_ones():
    good = [BodyPoint(p, s) for p, s in [("palm", "right"), ("fist", "left"), ("fist", "right"), ("foot", "left"),
                                         ("knee", "right"), ("shoulder", "left"), ("head", None), ("head", "left")]]
    for bp in good:
        assert "body_point_invalid" not in {i.code for i in check(make_cue(body_point=bp))}, bp
    bad = [BodyPoint("elbow", "left"), BodyPoint("fist", None), BodyPoint("foot", "middle"),
           BodyPoint("palm", "left"),    # the cue's hand is right: the palm's side must equal it
           BodyPoint("palm", None)]
    for bp in bad:
        issues = [i for i in check(make_cue(body_point=bp)) if i.code == "body_point_invalid"]
        assert len(issues) == 1 and issues[0].severity == "error" and issues[0].message.startswith("body_point."), bp
        w = [w for w in api.validate(cue_dict(body_point=bp)) if w["code"] == "body_point_invalid"]
        assert len(w) == 1 and w[0]["severity"] == "error" and w[0]["cause"]["path"].startswith("body_point")
        assert api.schema_errors("warning", w[0]) == []
    assert BodyPoint("palm", "left") and body_point_of(make_cue(hand="left", body_point=BodyPoint("palm", "left")))


def test_a_body_point_with_a_wrong_type_is_a_schema_error_not_an_unknown_field():
    d = cue_dict()
    d["body_point"] = {"part": 3}
    [w] = api.validate(d)
    assert w["code"] == "invalid_field" and w["cause"]["path"] == "body_point.part"
    d["body_point"] = {"part": "fist", "side": "left", "colour": 1}
    assert [w["code"] for w in api.validate(d)] == ["invalid_field"]
    d = cue_dict(body_point=BodyPoint("fist", "left"))
    assert "unknown_field" not in {w["code"] for w in api.validate(d)}


def test_parse_body_point():
    assert api.parse_body_point(None) is None
    assert api.parse_body_point({"part": "fist", "side": "left"}) == BodyPoint("fist", "left")
    assert api.parse_body_point({"part": "head"}) == BodyPoint("head", None)
    assert api.parse_body_point({"part": "head", "side": "right"}) == BodyPoint("head", None)
    assert api.parse_body_point({"part": "palm", "side": "right"}, "right") == BodyPoint("palm", "right")
    for raw, hand in [({"part": "palm", "side": "left"}, "right"), ({"part": "fist"}, "right"), ({"part": "x"}, "right"),
                      ({"side": "left"}, "right"), ({"part": "fist", "side": 1}, "right"),
                      ({"part": "fist", "side": "left", "z": 1}, "right")]:
        with pytest.raises(api.CueFXError) as e:
            api.parse_body_point(raw, hand)
        assert e.value.code == "body_point_invalid", raw


def test_normalize_versions():
    d = cue_dict(version="2.2")
    assert api.normalize(d)["version"] == "2.2" and "body_point" not in api.normalize(d)   # a v2.2 file stays v2.2
    assert api.normalize(cue_dict())["version"] == "2.3"
    for v in ("2.2", "2.3"):
        out = api.normalize(cue_dict(version=v, body_point=BodyPoint("fist", "right")))
        assert out["version"] == "2.4" and out["body_point"] == {"part": "fist", "side": "right"}
    assert api.normalize(cue_dict(version="2.4"))["version"] == "2.4"
    assert api.schema_errors("cue", api.normalize(cue_dict(body_point=BodyPoint("head", None)))) == []


# --------------------------------------------------------------------------- cue/new: default size and offset per part

def api_track(size: float, n: int = 150, throw_at: int | None = None) -> list[dict]:
    """A point that stays still, then (if `throw_at`) moves 2.5 sizes per frame; palm_w = `size`."""
    x, out = 800.0, []
    for f in range(n):
        x += 2.5 * size if throw_at is not None and f >= throw_at else 0.0
        out.append({"frame": f, "x": x, "y": 500.0, "confidence": 0.9, "uncertain": False, "source": "tracked",
                    "palm_w": size})
    return out


def new_cue(part=None, side=None, size=100.0, **kw):
    bp = None if part is None else {"part": part, "side": side}
    return api.new_cue_with_warnings("c.mp4", 24, 150, [1920, 1080], "right", {"f": 40, "x": 800, "y": 500},
                                     api_track(size, **kw), body_point=bp)


DEFAULTS = [  # part, side, diameter, anchor_offset for a size measure of 100
    ("fist", "right", 160, [0, 0]),
    ("foot", "left", 100, [0, -50]),  # task 36: up 0.5 d, so the effect sits on the shoe (was [0, 0])
    ("knee", "right", 100, [0, 0]),
    ("shoulder", "left", 100, [0, -50]),
    ("head", None, 100, [0, -110]),
]
STRIKE = ("fist", "foot", "knee")  # task 36: these start at their fastest frame; the others at the end of the steady run


@pytest.mark.parametrize("part,side,diameter,offset", DEFAULTS, ids=[x[0] for x in DEFAULTS])
def test_default_size_and_offset_per_part(part, side, diameter, offset):
    cue, warnings = new_cue(part, side, throw_at=100)
    assert cue["diameter_px"] == diameter and cue["anchor_offset"] == offset
    assert cue["body_point"] == {"part": part, "side": side} and cue["version"] == "2.4"
    # the kick or punch is the throw: 2.5 sizes per frame from frame 100, the smoothed speed first peaks at 101
    assert cue["behaviour"] == "hold_throw" and cue["release_frame"] == (101 if part in STRIKE else 100)
    assert api.schema_errors("cue", cue) == [] and [w["code"] for w in warnings] == []
    assert [i for i in check(cue_from_dict(cue)) if i.severity == "error"] == []


def test_the_palm_defaults_are_the_old_ones():
    cue, _ = new_cue(throw_at=100)
    assert cue["diameter_px"] == 115 and cue["anchor_offset"] == [0, -75] and cue["version"] == "2.3"
    assert "body_point" not in cue
    explicit, _ = new_cue("palm", "right", throw_at=100)
    assert {k: v for k, v in explicit.items() if k not in ("body_point", "version")} == \
           {k: v for k, v in cue.items() if k not in ("body_point", "version")}
    assert explicit["body_point"] == {"part": "palm", "side": "right"} and explicit["version"] == "2.4"
    assert PART_SIZING["palm"][:2] == (1.15, 0.65)


@pytest.mark.parametrize("size", [40.0, 133.0, 250.0])
def test_start_timings_for_the_palm_are_the_same_with_and_without_the_part_argument(size):
    pts = api.points_from_track(api_track(size, throw_at=90))
    assert start_timings(pts, 24, 150) == start_timings(pts, 24, 150, part="palm")


def test_a_still_body_point_without_a_throw_starts_as_a_hold():
    cue, warnings = new_cue("knee", "left")
    assert cue["behaviour"] == "hold" and cue["release_frame"] is None and cue["route"] == []
    assert [w["code"] for w in warnings] == ["started_as_hold"] and cue["version"] == "2.4"


def test_the_head_offset_is_above_the_head_in_size_measures():
    for size in (60.0, 200.0):
        cue, _ = new_cue("head", None, size=size, throw_at=100)
        assert cue["anchor_offset"] == [0, -round(1.1 * size)] and cue["diameter_px"] == round(size)


def test_cue_new_with_a_bad_body_point_is_body_point_invalid():
    for bp in ({"part": "palm", "side": "left"}, {"part": "fist"}, {"part": "ear", "side": "left"}):
        with pytest.raises(api.CueFXError) as e:
            api.new_cue_with_warnings("c.mp4", 24, 150, [1920, 1080], "right", {"f": 40, "x": 800, "y": 500},
                                      api_track(100.0), body_point=bp)
        assert e.value.code == "body_point_invalid"


# --------------------------------------------------------------------------- the palm path is untouched

def test_a_palm_analyze_never_asks_for_pose(monkeypatch):
    import services.analysis.track as track

    def boom(*a, **k):
        raise AssertionError("pose was used for the palm")

    monkeypatch.setattr(pose_track, "get_pose_landmarks", boom)
    monkeypatch.setattr(api, "_probe_frames", lambda clip: 10)
    seen = []

    def fake_track(video, hand, seed, cfg=None, model=None):
        seen.append(seed)
        raise TrackError("stop here", "no_hand")

    monkeypatch.setattr(track, "track_video", fake_track)
    for bp in (None, {"part": "palm", "side": "right"}):
        with pytest.raises(api.CueFXError) as e:
            api.analyze(Path("x.mp4"), {"frame": 3, "x": 1, "y": 2}, body_point=bp, hand="right")
        assert e.value.code == "no_hand"
    assert len(seen) == 2


def test_is_palm():
    assert is_palm(None) and is_palm(BodyPoint("palm", "left")) and not is_palm(BodyPoint("head", None))
