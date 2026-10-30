"""Task 19 routes: cue/new, cue/normalize, the stand-in render and its files."""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server_helpers import (uploaded_template,  # noqa: F401
                            FRAMES, SYNTHETIC_TRACK, assert_error, assert_fits, clip_file, client,  # noqa: F401
                            golden, golden_length_clip, settings, synthetic_cue, uploaded)
from services.cue import api
from services.cue.cue import cue_from_dict
from services.delivery.video import read_frames
from services import config
from app import create_app

SEED = {"frame": 60, "x": 1160, "y": 550}
SEED_SYNTHETIC = {"frame": 5, "x": 660, "y": 660}  # the hand_seed of synthetic_cue()
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake_analyze(monkeypatch):
    """api.analyze replaced by a counter that returns the golden track."""
    calls = []

    def fake(clip_path, seed, model=None, **kw):
        calls.append((clip_path, seed, kw))
        return api.analyze_result(api.points_from_track(golden("throw_71319.track.json")))

    monkeypatch.setattr(api, "analyze", fake)
    return calls


def errors_of(warnings: list[dict]) -> list[dict]:
    return [w for w in warnings if w["severity"] == "error"]


# --------------------------------------------------------------------------- cue/new

def test_cue_new_on_the_golden_track_is_a_valid_cue(client, golden_length_clip, fake_analyze):
    up = client.post("/api/clips", files={"file": (golden_length_clip.name, golden_length_clip.read_bytes())}).json()
    r = client.post("/api/cue/new", json={"clip_id": up["clip_id"], "hand": "right", "hand_seed": SEED})
    assert r.status_code == 200, r.text
    cue = r.json()
    assert_fits("cue", cue)
    assert errors_of(api.validate(cue)) == []
    assert errors_of(client.post("/api/validate", json={"cue": cue}).json()) == []
    assert (cue["version"], cue["clip"], cue["frames"], cue["fps"], cue["plate_size"]) == (
        "2.3", "synthetic193.mp4", 193, 24, [1920, 1080])
    assert cue["transform"] == {"scale": 6.0, "offset": [0, 0]}  # 320x180 -> 1920x1080
    assert cue["hand"] == "right" and cue["hand_seed"] == {"f": 60, "x": 1160, "y": 550}
    assert api.track_from_points(cue_from_dict(cue).track) == golden("throw_71319.track.json")
    assert cue["route"] == [{"f": cue["release_frame"], "x": None, "y": None, "diameter_px": cue["diameter_px"],
                             "easing_in": "linear"}]
    assert cue["behind"] == [] and cue["corrections"] == []
    for k in ("effect_range", "canvas_size", "canvas_rect", "transform", "anchor_offset", "appear_frame", "vanish",
              "grade", "entrance"):
        assert k in cue
    assert cue["canvas_rect"] == [0, 0, 1920, 1080]  # the whole (conformed) plate
    assert client.post("/api/cue/normalize", json={"cue": cue}).json() == cue  # already normalized


def test_cue_new_uses_the_analyze_cache(client, uploaded, fake_analyze):
    body = {"clip_id": uploaded["clip_id"], "hand": "right", "hand_seed": SEED}
    client.post("/api/analyze", json=body)
    client.post("/api/cue/new", json=body)
    client.post("/api/cue/new", json={**body, "hand_seed": {"f": 60, "x": 1160, "y": 550}})
    assert len(fake_analyze) == 1
    assert fake_analyze[0][2]["landmarks_out"].name.endswith(".part.npz")  # the landmarks are kept for Merge


def test_cue_new_errors(client, uploaded, fake_analyze):
    assert_error(client.post("/api/cue/new", json={"clip_id": "000000000000", "hand": "right", "hand_seed": SEED}),
                 404, "not_found")
    assert_error(client.post("/api/cue/new", json={"clip_id": uploaded["clip_id"], "hand": "up", "hand_seed": SEED}),
                 400, "input_invalid")


# --------------------------------------------------------------------------- cue/normalize

def test_normalize_returns_the_golden_cue_unchanged(client):
    cue = golden("throw_71319.cue.json")
    r = client.post("/api/cue/normalize", json={"cue": cue})
    assert r.status_code == 200 and r.json() == cue


def test_normalize_recomputes_the_derived_fields(client):
    cue = golden("throw_71319.cue.json")
    cue["version"] = "2.1"
    cue["effect_range"] = {"start": 0, "end": 5}
    del cue["route"][1]["easing_in"]
    del cue["canvas_rect"]
    out = client.post("/api/cue/normalize", json={"cue": cue}).json()
    assert out["version"] == "2.2"
    assert out["effect_range"] == {"start": 34, "end": 115}   # propose_effect_range(appear 40, vanish.end 113)
    assert out["route"][1]["easing_in"] == "linear"
    assert out["canvas_rect"] == [0, 0, 1920, 1080]
    cue["vanish"] = {"start": 111, "end": 140}                 # the artist moves vanish: the window follows
    assert client.post("/api/cue/normalize", json={"cue": cue}).json()["effect_range"] == {"start": 36, "end": 141}
    assert cue["effect_range"] == {"start": 0, "end": 5}        # the input is not changed


def test_normalize_keeps_a_cue_it_cannot_derive_from(client):
    cue = golden("throw_71319.cue.json")
    cue["appear_frame"] = 500  # outside the clip: nothing to cover; validate reports it
    out = client.post("/api/cue/normalize", json={"cue": cue}).json()
    assert out["effect_range"] == {"start": 34, "end": 115}


@pytest.mark.parametrize("body", [{}, {"cue": 5}, {"cue": {"clip": "x"}}, {"cue": {}, "track": []}])
def test_normalize_bad_bodies_are_input_invalid(client, body):
    assert_error(client.post("/api/cue/normalize", json=body), 400, "input_invalid")


# --------------------------------------------------------------------------- stand-in on a small synthetic clip

@pytest.mark.slow  # a full render (task 27: kept out of the default run)
def test_standin_renders_serves_and_caches(client, uploaded, settings, tmp_path):
    cue = synthetic_cue()
    assert errors_of(api.validate(cue)) == []
    r = client.post("/api/standin?wait=1", json={"cue": cue, "clip_id": uploaded["clip_id"]})
    assert r.status_code == 200, r.text
    res = r.json()
    assert_fits("standin_result", res)
    assert res["cached"] is False and res["clip_id"] == uploaded["clip_id"]
    assert (res["frames"], res["fps"], res["plate_size"]) == (FRAMES, 24, [1920, 1080])
    assert res["effect_range"] == {"start": 4, "end": 21}
    key = res["make_key"]
    assert res["standin_id"].startswith(key + "-")
    make_dir = settings.standin_dir / key
    for f in ("layout.mp4", "effect.mp4", "matte.mp4", "layout_sheet.png", "standin_sheet.png", "make.json"):
        assert (make_dir / f).is_file(), f
    assert set(res["files"]) == {"hardcomp_preview", "hardcomp_download", "element_download", "layout_download"}
    merge_dir = make_dir / "merge" / res["standin_id"].split("-")[1]
    assert not (merge_dir / "element.mp4").exists() and not (merge_dir / "layers").exists()  # not built for the preview
    assert res["timings_s"]["make"] > 0 and res["timings_s"]["merge"] > 0 and res["timings_s"]["element"] == 0
    assert res["timings_s"]["total"] >= res["timings_s"]["make"] + res["timings_s"]["merge"]

    preview = client.get(res["files"]["hardcomp_preview"])
    assert preview.status_code == 200 and preview.headers["content-type"] == "video/mp4"
    assert "attachment" not in preview.headers.get("content-disposition", "")
    p = tmp_path / "preview.mp4"
    p.write_bytes(preview.content)
    frames = list(read_frames(p))
    assert len(frames) == FRAMES and frames[0].shape == (1080, 1920, 3)
    assert frames[12].max() > frames[0].max()  # the orb is on the plate inside the window, not outside it
    names = {"hardcomp_download": "synthetic_{}_hardcomp.mp4", "element_download": "synthetic_{}_element.mp4",
             "layout_download": "synthetic_{}_layout.mp4"}
    for k, name in names.items():
        d = client.get(res["files"][k] + "?wait=1")
        assert d.status_code == 200 and len(d.content) > 0
        assert d.headers["content-disposition"].startswith("attachment") and name.format(key) in d.headers[
            "content-disposition"]
        if k == "element_download":  # made by the first GET, kept by the second
            element = merge_dir / "element.mp4"
            assert element.is_file() and not (merge_dir / "element_work").exists() and not (merge_dir / "layers").exists()
            first = element.stat().st_mtime_ns
            assert client.get(res["files"][k] + "?wait=1").content == d.content and element.stat().st_mtime_ns == first
            assert len(list(read_frames(element))) == FRAMES
    assert client.get(res["files"]["layout_download"]).content == (make_dir / "layout.mp4").read_bytes()
    assert client.get(res["files"]["hardcomp_preview"], headers={"Range": "bytes=0-99"}).status_code == 206

    # the same cue again: from the cache, nothing re-rendered
    final = make_dir / "merge" / res["standin_id"].split("-")[1] / "final.mp4"
    stamp = final.stat().st_mtime_ns, (make_dir / "layout.mp4").stat().st_mtime_ns
    t0 = time.perf_counter()
    again = client.post("/api/standin?wait=1", json={"cue": cue, "clip_id": uploaded["clip_id"]}).json()
    assert time.perf_counter() - t0 < 2
    assert again["cached"] is True and again["standin_id"] == res["standin_id"]
    assert again["timings_s"]["merge"] == res["timings_s"]["merge"] and again["timings_s"]["element"] > 0
    assert_fits("standin_result", again)
    assert (final.stat().st_mtime_ns, (make_dir / "layout.mp4").stat().st_mtime_ns) == stamp

    # a Merge-only change (vanish, grade) keeps the Make folder and makes a new Merge
    cue2 = {**cue, "grade": {**cue["grade"], "glow": 0.2}}
    third = client.post("/api/standin?wait=1", json={"cue": cue2, "clip_id": uploaded["clip_id"]}).json()
    assert third["make_key"] == key and third["standin_id"] != res["standin_id"] and third["cached"] is False
    assert third["timings_s"]["make"] == 0 and third["timings_s"]["merge"] > 0  # Make was reused
    assert (make_dir / "layout.mp4").stat().st_mtime_ns == stamp[1]


@pytest.mark.slow  # a full render (task 27: kept out of the default run)
def test_standin_renders_a_hold_on_the_hand(client, uploaded, tmp_path):
    throw = synthetic_cue()
    hold = api.normalize({**throw, "behaviour": "hold", "release_frame": None, "route": []})
    assert errors_of(api.validate(hold)) == [] and "hold_path_unused" not in {w["code"] for w in api.validate(hold)}
    res = client.post("/api/standin?wait=1", json={"cue": hold, "clip_id": uploaded["clip_id"]}).json()
    other = client.post("/api/standin?wait=1", json={"cue": throw, "clip_id": uploaded["clip_id"]}).json()
    assert_fits("standin_result", res)
    assert res["make_key"] != other["make_key"] and res["effect_range"] == other["effect_range"] == {"start": 4, "end": 21}
    p = tmp_path / "hold.mp4"
    p.write_bytes(client.get(res["files"]["hardcomp_preview"]).content)
    frames = list(read_frames(p))
    assert len(frames) == FRAMES and frames[12].max() > frames[0].max()   # the orb is on the plate in the window
    # held, not thrown: in frame 18 (after the throw's release) the Hold orb is still where the hand is
    q = tmp_path / "throw.mp4"
    q.write_bytes(client.get(other["files"]["hardcomp_preview"]).content)
    assert not np.array_equal(frames[16], list(read_frames(q))[16])


@pytest.mark.slow  # a full render (task 27: kept out of the default run)
def test_standin_with_a_behind_range_reuses_the_landmarks_from_analyze(client, uploaded, monkeypatch):
    calls = []
    cue = synthetic_cue()

    def fake(clip_path, seed, model=None, landmarks_out=None, **kw):
        calls.append(seed)
        nan = np.full((FRAMES, 21, 3), np.nan, np.float32)
        np.savez_compressed(landmarks_out, chosen=nan, other=nan)
        return api.analyze_result(api.points_from_track(SYNTHETIC_TRACK))

    monkeypatch.setattr(api, "analyze", fake)
    seed = {"clip_id": uploaded["clip_id"], "hand": "right", "hand_seed": SEED_SYNTHETIC}
    client.post("/api/cue/new", json=seed)
    cue["behind"] = [{"start": 10, "end": 13, "occluder": "left_hand"}]
    assert errors_of(api.validate(cue)) == []
    res = client.post("/api/standin?wait=1", json={"cue": cue, "clip_id": uploaded["clip_id"]}).json()
    assert len(calls) == 1  # the stand-in did not analyze again
    assert res["timings_s"]["landmarks"] >= 0 and "landmarks" in res["timings_s"]
    cue["behind"][0]["end"] = 14  # a Merge-only change: still no analyze
    client.post("/api/standin?wait=1", json={"cue": cue, "clip_id": uploaded["clip_id"]})
    assert len(calls) == 1


@pytest.mark.slow  # a full render (task 27: kept out of the default run)
def test_standin_finds_the_clip_by_the_cues_clip_name(client, uploaded):
    r = client.post("/api/standin?wait=1", json={"cue": synthetic_cue()})
    assert r.status_code == 200 and r.json()["clip_id"] == uploaded["clip_id"]
    assert_error(client.post("/api/standin?wait=1", json={"cue": synthetic_cue("other.mp4")}), 404, "not_found")


def test_standin_with_an_invalid_cue_is_input_invalid(client, uploaded, settings):
    bad = synthetic_cue()
    bad["release_frame"] = 3  # before appear_frame
    body = assert_error(client.post("/api/standin?wait=1", json={"cue": bad, "clip_id": uploaded["clip_id"]}), 400,
                        "input_invalid")
    assert "release_frame" in body["message"]
    assert_error(client.post("/api/standin?wait=1", json={"cue": {"clip": 1}}), 400, "input_invalid")
    assert_error(client.post("/api/standin?wait=1", json={"cue": golden("throw_71319.cue.json")}), 400,
                 "input_invalid")  # no track: no_track is an error
    assert not settings.standin_dir.exists()  # nothing was rendered


def test_standin_cue_for_another_plate_is_input_invalid(client, uploaded):
    cue = synthetic_cue()
    cue["frames"] = FRAMES + 1
    cue["track"].append({**cue["track"][-1], "f": FRAMES})
    body = assert_error(client.post("/api/standin?wait=1", json={"cue": cue, "clip_id": uploaded["clip_id"]}), 400,
                        "input_invalid")
    assert "frames" in body["message"]


@pytest.mark.slow  # a full render (task 27: kept out of the default run)
def test_standin_unknown_ids_and_names_are_404(client, uploaded):
    assert_error(client.get("/api/standin/000000000000-000000000000/hardcomp_preview.mp4"), 404, "not_found")
    assert_error(client.get("/api/standin/nonsense/layout.mp4"), 404, "not_found")
    assert_error(client.post("/api/standin?wait=1", json={"cue": synthetic_cue(), "clip_id": "000000000000"}), 404,
                 "not_found")
    res = client.post("/api/standin?wait=1", json={"cue": synthetic_cue(), "clip_id": uploaded["clip_id"]}).json()
    assert_error(client.get(f"/api/standin/{res['standin_id']}/final.mp4"), 404, "not_found")
    assert_error(client.get(f"/api/standin/{res['standin_id']}/..%2Fmake.json"), 404, "not_found")


def test_standin_failure_is_render_failed(client, uploaded, monkeypatch):
    from services.make.layout import LayoutError
    from services.delivery import standin

    def boom(*a, **k):
        raise LayoutError("ffmpeg exploded at C:\\private")

    monkeypatch.setattr(standin, "make_layout", boom)
    r = client.post("/api/standin?wait=1", json={"cue": synthetic_cue(), "clip_id": uploaded["clip_id"]})
    body = assert_error(r, 500, "render_failed")
    assert "Building reference frame" in body["message"] and "private" not in body["message"]


@pytest.mark.slow  # a full render (task 27: kept out of the default run)
def test_standin_draws_the_default_look_when_none_is_there(client, uploaded, settings):
    assert not settings.look_path.exists()
    client.post("/api/standin?wait=1", json={"cue": synthetic_cue(), "clip_id": uploaded["clip_id"]})
    assert settings.look_path.is_file()


def test_look_image_setting():
    assert config.from_env({"CUEFX_LOOK_IMAGE": "x/look.png"}).look_path.name == "look.png"
    d = config.from_env({"CUEFX_DATA_DIR": "somewhere"})
    assert d.look_path == d.data_dir / "look" / "standin_orb.png" and d.standin_dir == d.data_dir / "standin"


# --------------------------------------------------------------------------- slow: the golden cue on clip 71319

def ffprobe(path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames",
                          "-of", "json", str(path)], capture_output=True, check=True)
    return json.loads(out.stdout)["streams"][0]


@pytest.mark.slow
def test_golden_standin_render(tmp_path):
    clip = ROOT / "data" / "input" / "throw_71319.mov"
    look = ROOT / "data" / "look" / "standin_orb.png"
    if not clip.is_file() or not look.is_file():
        pytest.skip("data/input/throw_71319.mov or data/look/standin_orb.png is not here")
    s = config.Settings(data_dir=tmp_path / "data", look_image=look)
    c = TestClient(create_app(s))
    up = c.post("/api/clips", files={"file": (clip.name, clip.read_bytes(), "video/quicktime")})
    assert up.status_code == 201, up.text
    cue = golden("throw_71319.cue.json")
    track = golden("throw_71319.track.json")

    t0 = time.perf_counter()
    r = c.post("/api/standin?wait=1", json={"cue": cue, "track": track, "clip_id": up.json()["clip_id"]})
    cold = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    res = r.json()
    assert_fits("standin_result", res)
    t1 = time.perf_counter()
    again = c.post("/api/standin?wait=1", json={"cue": cue, "track": track}).json()
    cached = time.perf_counter() - t1
    assert again["cached"] is True and again["standin_id"] == res["standin_id"]

    files = {}
    for k, url in res["files"].items():
        d = c.get(url + "?wait=1")
        assert d.status_code == 200
        files[k] = tmp_path / f"{k}.mp4"
        files[k].write_bytes(d.content)
    p = ffprobe(files["hardcomp_preview"])
    assert (p["codec_name"], p["pix_fmt"], p["width"], p["height"], p["r_frame_rate"], p["nb_read_frames"]) == (
        "h264", "yuv420p", 1920, 1080, "24/1", "193")
    e = ffprobe(files["element_download"])
    assert (e["codec_name"], e["pix_fmt"], e["width"], e["height"], e["nb_read_frames"]) == (
        "h264", "yuv444p", 1920, 1080, "193")
    sizes = {k: f"{v.stat().st_size / 1e6:.1f} MB" for k, v in files.items()}
    print(f"\nstand-in 71319: cold {cold:.1f} s, cached {cached * 1000:.0f} ms; "
          f"sizes {sizes}; standin dir {s.standin_dir / res['make_key']}")


def test_cue_new_warnings_header_is_a_list_of_warnings(client, golden_length_clip, fake_analyze):
    up = client.post("/api/clips", files={"file": (golden_length_clip.name, golden_length_clip.read_bytes())}).json()
    r = client.post("/api/cue/new", json={"clip_id": up["clip_id"], "hand": "right", "hand_seed": SEED})
    assert json.loads(r.headers["X-CueFX-Warnings"]) == []  # the golden track has a steady hold


def test_cue_new_warnings_header_is_readable_cross_origin(client, golden_length_clip, fake_analyze):
    up = client.post("/api/clips", files={"file": (golden_length_clip.name, golden_length_clip.read_bytes())}).json()
    r = client.post("/api/cue/new", json={"clip_id": up["clip_id"], "hand": "right", "hand_seed": SEED},
                    headers={"Origin": "http://localhost:5173"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    exposed = [h.strip().lower() for h in r.headers["access-control-expose-headers"].split(",")]
    assert "x-cuefx-warnings" in exposed
