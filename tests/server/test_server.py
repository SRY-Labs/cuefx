"""The HTTP server (server/): every route, its errors, and equality with the contract's golden files."""
import json
import subprocess
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from server_helpers import (uploaded_template,  # noqa: F401
                            FRAMES, assert_error, assert_fits, clip_file, client, golden, grey, settings,  # noqa: F401
                            uploaded)
from services.cue import api
from services.cue.cue import cue_from_dict, cue_to_dict
from services import config
from app import create_app

SEED = {"frame": 60, "x": 1160, "y": 550}


# --------------------------------------------------------------------------- health, config, CORS

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": "0.1.0", "contract_version": "3.0"}


def test_config_defaults_and_overrides(tmp_path):
    d = config.from_env({})
    assert (d.host, d.port, d.cors_origins, d.max_upload_mb) == (
        "127.0.0.1", 8000, ("http://localhost:5173",), 500)
    assert d.data_dir.name == "data" and d.data_dir.is_absolute()
    o = config.from_env({"CUEFX_DATA_DIR": str(tmp_path), "CUEFX_HOST": "0.0.0.0", "CUEFX_PORT": "9001",
                         "CUEFX_CORS_ORIGINS": "http://a.test, http://b.test", "CUEFX_MAX_UPLOAD_MB": "7"})
    assert (o.host, o.port, o.cors_origins, o.max_upload_mb) == ("0.0.0.0", 9001, ("http://a.test", "http://b.test"), 7)
    assert o.data_dir == tmp_path.resolve()


def test_cors_allows_the_ui_origin(client):
    r = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    pre = client.options("/api/validate", headers={"Origin": "http://localhost:5173",
                                                   "Access-Control-Request-Method": "POST",
                                                   "Access-Control-Request-Headers": "content-type"})
    assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_does_not_allow_other_origins(client):
    r = client.get("/api/health", headers={"Origin": "http://evil.test"})
    assert "access-control-allow-origin" not in r.headers


# --------------------------------------------------------------------------- clips

def test_clip_list_starts_empty(client):
    assert client.get("/api/clips").json() == []


def test_upload_stores_the_clip_and_lists_it(client, uploaded, settings):
    assert set(uploaded) == {"clip_id", "clip", "fps", "frames", "plate_size"}
    assert len(uploaded["clip_id"]) == 12
    assert (uploaded["clip"], uploaded["fps"], uploaded["frames"], uploaded["plate_size"]) == (
        "synthetic.mp4", 24.0, FRAMES, [1920, 1080])  # the 640x360 upload, conformed
    assert (settings.uploads_dir / uploaded["clip_id"] / "synthetic.mp4").is_file()
    assert client.get("/api/clips").json() == [uploaded]


def test_the_same_file_twice_gives_the_same_id_and_is_stored_once(client, uploaded, clip_file, settings):
    r = client.post("/api/clips", files={"file": ("renamed.mp4", clip_file.read_bytes(), "video/mp4")})
    assert r.status_code == 200 and r.json()["clip_id"] == uploaded["clip_id"]
    assert [p.name for p in (settings.uploads_dir / uploaded["clip_id"]).iterdir()] == ["synthetic.mp4"]
    assert len(client.get("/api/clips").json()) == 1
    assert [p.name for p in settings.uploads_dir.iterdir()] == [uploaded["clip_id"]]  # no temp folders left behind


def test_upload_wrong_type_is_refused(client, settings):
    r = client.post("/api/clips", files={"file": ("photo.png", b"\x89PNG", "image/png")})
    assert_error(r, 400, "input_invalid")
    assert ".png" in r.json()["message"]


def test_upload_that_is_not_a_video_is_unreadable(client):
    r = client.post("/api/clips", files={"file": ("fake.mp4", b"this is not a video" * 100, "video/mp4")})
    assert_error(r, 400, "clip_unreadable")
    assert client.get("/api/clips").json() == []


def test_upload_empty_file_is_refused(client):
    assert_error(client.post("/api/clips", files={"file": ("empty.mp4", b"", "video/mp4")}), 400, "input_invalid")


def test_upload_too_large_is_413(settings, clip_file):
    c = TestClient(create_app(config.Settings(data_dir=settings.data_dir, max_upload_mb=1)))
    r = c.post("/api/clips", files={"file": ("big.mp4", b"\0" * (2 * 1024 * 1024), "video/mp4")})
    assert_error(r, 413, "input_invalid")
    assert "1 MB" in r.json()["message"]
    assert c.get("/api/clips").json() == []


def test_upload_over_the_analyze_frame_cap_is_refused(client, clip_file, monkeypatch):
    monkeypatch.setattr(api, "MAX_CLIP_FRAMES", FRAMES - 1)
    r = client.post("/api/clips", files={"file": ("long.mp4", clip_file.read_bytes(), "video/mp4")})
    assert_error(r, 400, "clip_too_long")
    assert client.get("/api/clips").json() == []


def test_upload_without_a_file_is_a_contract_error(client):
    assert_error(client.post("/api/clips"), 400, "input_invalid")


# --------------------------------------------------------------------------- proxy video and exact frames

def ffprobe(path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames,profile",
                          "-of", "json", str(path)], capture_output=True, check=True)
    return json.loads(out.stdout)["streams"][0]


def test_video_is_a_browser_playable_proxy_of_the_plate(client, uploaded, tmp_path):
    r = client.get(f"/api/clips/{uploaded['clip_id']}/video")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    f = tmp_path / "proxy.mp4"
    f.write_bytes(r.content)
    s = ffprobe(f)
    assert (s["codec_name"], s["pix_fmt"], s["width"], s["height"], s["r_frame_rate"], s["nb_read_frames"]) == (
        "h264", "yuv420p", 1920, 1080, "24/1", str(FRAMES))


def test_video_is_made_once_and_supports_range_requests(client, uploaded, settings):
    url = f"/api/clips/{uploaded['clip_id']}/video"
    client.get(url)
    proxy = settings.cache_dir / uploaded["clip_id"] / "proxy.mp4"
    made = proxy.stat().st_mtime_ns
    r = client.get(url, headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert proxy.stat().st_mtime_ns == made


def test_frame_is_the_exact_frame(client, uploaded):
    import cv2
    import numpy as np

    for f in (0, 1, 7, FRAMES - 1):
        r = client.get(f"/api/clips/{uploaded['clip_id']}/frames/{f}.jpg")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (1080, 1920, 3)  # plate size, whatever the clip's own size
        assert abs(float(img.mean()) - grey(f)) < 6, f


def test_frame_out_of_range_and_unknown_clip(client, uploaded):
    assert_error(client.get(f"/api/clips/{uploaded['clip_id']}/frames/{FRAMES}.jpg"), 400, "input_invalid")
    assert_error(client.get("/api/clips/000000000000/frames/0.jpg"), 404, "not_found")
    assert_error(client.get("/api/clips/000000000000/video"), 404, "not_found")
    assert_error(client.get("/api/clips/not-an-id/video"), 404, "not_found")


# --------------------------------------------------------------------------- analyze

@pytest.fixture
def fake_analyze(monkeypatch):
    """api.analyze replaced by a counter that returns the golden track as a finished analyze result."""
    calls = []

    def fake(clip_path, seed, model=None, **kw):
        calls.append((clip_path, seed))
        return api.analyze_result(api.points_from_track(golden("throw_71319.track.json")))

    monkeypatch.setattr(api, "analyze", fake)
    return calls


def analyze_body(clip_id, **changes):
    return {"clip_id": clip_id, "hand": "right", "hand_seed": SEED, **changes}


def test_analyze_returns_the_result_and_fits_its_schema(client, uploaded, fake_analyze):
    r = client.post("/api/analyze", json=analyze_body(uploaded["clip_id"]))
    assert r.status_code == 200
    assert_fits("analyze_result", r.json())
    assert len(fake_analyze) == 1 and fake_analyze[0][1] == SEED
    assert fake_analyze[0][0].name == "plate.mp4"  # analyze reads the conformed plate, not the upload


def test_analyze_is_cached_by_clip_hand_and_seed(client, uploaded, fake_analyze, settings):
    body = analyze_body(uploaded["clip_id"])
    first = client.post("/api/analyze", json=body).json()
    assert client.post("/api/analyze", json=body).json() == first
    assert len(fake_analyze) == 1
    client.post("/api/analyze", json=analyze_body(uploaded["clip_id"], hand="left"))
    client.post("/api/analyze", json=analyze_body(uploaded["clip_id"], hand_seed={**SEED, "x": 1000}))
    assert len(fake_analyze) == 3
    # the cache is on disk: a new app on the same data folder answers without analysing
    c2 = TestClient(create_app(settings))
    assert c2.post("/api/analyze", json=body).json() == first and len(fake_analyze) == 3


def test_analyze_cache_follows_the_code_version(client, uploaded, fake_analyze, monkeypatch):
    from services.analysis import clips

    client.post("/api/analyze", json=analyze_body(uploaded["clip_id"]))
    monkeypatch.setattr(clips, "_code_version", "another-version")
    client.post("/api/analyze", json=analyze_body(uploaded["clip_id"]))
    assert len(fake_analyze) == 2


def test_analyze_accepts_the_cue_seed_name(client, uploaded, fake_analyze):
    r = client.post("/api/analyze", json=analyze_body(uploaded["clip_id"], hand_seed={"f": 60, "x": 1160, "y": 550}))
    assert r.status_code == 200 and fake_analyze[0][1] == SEED


@pytest.mark.parametrize("code", ["no_hand", "track_lost", "clip_unreadable", "clip_too_long", "input_invalid"])
def test_analyze_errors_keep_their_contract_code(client, uploaded, monkeypatch, code):
    def fail(clip_path, seed, model=None, **kw):
        raise api.CueFXError(code, "plain words")

    monkeypatch.setattr(api, "analyze", fail)
    r = client.post("/api/analyze", json=analyze_body(uploaded["clip_id"]))
    assert assert_error(r, 400, code)["message"] == "plain words"


def test_analyze_unknown_clip_is_404(client, fake_analyze):
    assert_error(client.post("/api/analyze", json=analyze_body("000000000000")), 404, "not_found")
    assert fake_analyze == []


@pytest.mark.parametrize("body", [
    {"clip_id": "x", "hand": "up", "hand_seed": SEED},
    {"clip_id": "x", "hand": "right"},
    {"hand": "right", "hand_seed": SEED},
    {"clip_id": "x", "hand": "right", "hand_seed": {"x": 1, "y": 2}},
    {"clip_id": "x", "hand": "right", "hand_seed": {"frame": 1, "f": 1, "x": 1, "y": 2}},
    {"clip_id": "x", "hand": "right", "hand_seed": {"frame": "a", "x": 1, "y": 2}},
])
def test_analyze_bad_request_is_400(client, body):
    assert_error(client.post("/api/analyze", json=body), 400, "input_invalid")


def test_analyze_seed_frame_outside_the_clip_is_a_contract_error(client, uploaded):
    """The real api.analyze runs here (no fake): the synthetic clip has no hand, so it ends in a contract error."""
    r = client.post("/api/analyze", json=analyze_body(uploaded["clip_id"], hand_seed={"frame": 9999, "x": 1, "y": 1}))
    assert r.status_code == 400 and r.json()["code"] in api.codes()


# --------------------------------------------------------------------------- validate, preview, propose_timing

def golden_body() -> dict:
    return {"cue": golden("throw_71319.cue.json"), "track": golden("throw_71319.track.json")}


def test_validate_equals_the_golden_file(client):
    r = client.post("/api/validate", json=golden_body())
    assert r.status_code == 200
    assert r.json() == golden("throw_71319.validate.json")
    assert all(api.schema_errors("warning", w) == [] for w in r.json())


def without_held(preview: list[dict]) -> list[dict]:
    return [{k: v for k, v in e.items() if k != "held"} for e in preview]


def test_preview_equals_the_golden_file(client):
    r = client.post("/api/preview", json=golden_body())
    assert r.status_code == 200
    assert without_held(r.json()) == golden("throw_71319.preview.json")  # every v2.2 field unchanged
    assert_fits("preview", r.json())
    assert api.preview_path(golden("throw_71319.cue.json"), golden("throw_71319.track.json")) == golden(
        "throw_71319.preview.json")  # the Python call without with_held is the golden file itself


def test_preview_held_is_the_corrected_track_plus_the_anchor(client):
    from services.cue.route import held_xy

    body = golden_body()
    body["cue"]["corrections"] = [{"f": 60, "x": 1180, "y": 540}]  # the corrections move the marker too
    preview = client.post("/api/preview", json=body).json()
    c = replace(cue_from_dict(body["cue"]), track=api.points_from_track(body["track"]))
    want = held_xy(c)
    assert len(preview) == len(want) == 193
    for e, (x, y) in zip(preview, want):
        assert e["held"] == {"x": round(float(x), 3), "y": round(float(y), 3)}
    plain = client.post("/api/preview", json=golden_body()).json()
    assert preview[60]["held"] != plain[60]["held"] and preview[0]["held"] == plain[0]["held"]


def test_preview_uses_the_cues_own_track_when_none_is_sent(client):
    track = golden("throw_71319.track.json")
    cue = cue_to_dict(replace(cue_from_dict(golden("throw_71319.cue.json")), track=api.points_from_track(track)))
    assert len(cue["track"]) == 193
    without = client.post("/api/preview", json={"cue": cue})
    assert without.status_code == 200 and without_held(without.json()) == golden("throw_71319.preview.json")


def test_propose_timing_equals_the_backend_and_fits_the_cue_schema(client):
    r = client.post("/api/propose_timing", json=golden_body())
    assert r.status_code == 200
    assert r.json() == api.propose_timing(golden("throw_71319.cue.json"), golden("throw_71319.track.json"))
    assert_fits("cue", r.json())


def test_validate_reports_a_bad_cue_as_warnings_not_an_error(client):
    cue = golden("throw_71319.cue.json")
    cue["route"][1]["x"] = "left"
    r = client.post("/api/validate", json={"cue": cue})
    assert r.status_code == 200 and r.json()[0]["code"] == "invalid_field"
    assert all(api.schema_errors("warning", w) == [] for w in r.json())


def test_validate_without_any_track_says_no_track(client):
    r = client.post("/api/validate", json={"cue": golden("throw_71319.cue.json")})
    assert "no_track" in {w["code"] for w in r.json()}


def test_preview_and_timing_of_a_bad_cue_are_input_invalid(client):
    bad = golden("throw_71319.cue.json")
    bad["hand"] = 3
    for route in ("/api/preview", "/api/propose_timing"):
        assert_error(client.post(route, json={"cue": bad, "track": golden("throw_71319.track.json")}), 400, "input_invalid")


@pytest.mark.parametrize("route", ["/api/validate", "/api/preview", "/api/propose_timing"])
@pytest.mark.parametrize("body", [{}, {"cue": 5}, {"cue": {}, "track": "x"}, {"cue": {}, "extra": 1}, [1, 2]])
def test_bad_bodies_are_400(client, route, body):
    assert_error(client.post(route, json=body), 400, "input_invalid")


def test_body_that_is_not_json_is_400(client):
    r = client.post("/api/validate", content=b"{nope", headers={"content-type": "application/json"})
    assert_error(r, 400, "input_invalid")


# --------------------------------------------------------------------------- errors in general

def test_unknown_route_and_wrong_method_use_the_error_shape(client):
    assert_error(client.get("/api/nothing"), 404, "not_found")
    assert_error(client.get("/api/validate"), 405, "input_invalid")


def test_unexpected_exception_is_a_safe_500_and_logged(client, monkeypatch, caplog):
    def boom(cue, track=None):
        raise RuntimeError("secret path C:\\private\\thing")

    monkeypatch.setattr(api, "validate", boom)
    with caplog.at_level("ERROR", logger="cuefx.server"):
        r = client.post("/api/validate", json={"cue": {}})
    body = assert_error(r, 500)
    assert "secret" not in r.text and "Traceback" not in r.text and "RuntimeError" not in r.text
    assert any("secret path" in str(rec.exc_info[1]) for rec in caplog.records if rec.exc_info)
    assert body["code"] == "internal"


def test_every_error_status_maps_to_a_code_in_error_codes():
    from services.errors import STATUS

    assert set(STATUS) <= set(api.codes())


# --------------------------------------------------------------------------- slow: the real clip

@pytest.mark.slow
def test_real_clip_analyze_matches_the_golden_track(tmp_path):
    clip = api.CONTRACT_DIR.parent / "data" / "input" / "throw_71319.mov"
    if not clip.is_file():
        pytest.skip("data/input/throw_71319.mov is not here")
    import time

    c = TestClient(create_app(config.Settings(data_dir=tmp_path / "data")))
    up = c.post("/api/clips", files={"file": (clip.name, clip.read_bytes(), "video/quicktime")})
    assert up.status_code == 201, up.text
    info = up.json()
    assert (info["frames"], info["fps"], info["plate_size"]) == (193, 24.0, [1920, 1080])
    t0 = time.perf_counter()
    r = c.post("/api/analyze", json=analyze_body(info["clip_id"]))
    cold = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    t1 = time.perf_counter()
    again = c.post("/api/analyze", json=analyze_body(info["clip_id"]))
    cached = time.perf_counter() - t1
    assert again.json() == r.json()
    assert_fits("analyze_result", r.json())
    got, want = r.json()["track"], golden("throw_71319.track.json")
    assert len(got) == len(want) == 193
    worst = 0.0
    for g, w in zip(got, want):
        assert (g["frame"], g["uncertain"], g["source"]) == (w["frame"], w["uncertain"], w["source"])
        worst = max(worst, abs(g["x"] - w["x"]), abs(g["y"] - w["y"]), abs(g["palm_w"] - w["palm_w"]),
                    abs(g["confidence"] - w["confidence"]))
    print(f"\nanalyze cold {cold:.1f} s, cached {cached * 1000:.0f} ms, worst deviation from golden {worst:.2e}")
    assert worst < 1e-6
