"""Task 27: light requests never wait on renders (render queue, ?wait=0, status), the frame cache, error codes."""
import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server_helpers import (uploaded_template,  # noqa: F401
                            assert_error, assert_fits, clip_file, client, golden_length_clip,  # noqa: F401
                            settings, synthetic_cue, uploaded)
from services.cue import api
from services.delivery import standin
from app import create_app
from services.analysis.clips import ClipStore, _encode

LIGHT_S = 1.0  # every light request answers within this while a render runs


@pytest.fixture
def slow_render(monkeypatch):
    """StandinStore._render replaced by a render that keeps a core busy the way Make and Merge do (numpy work on
    plate-size frames between short Python steps) until `release` is set, then writes a finished render's files
    (no video). On a real 71319 render the light routes answered in at most 0.24 s (task 27 report)."""
    release, started = threading.Event(), threading.Event()
    calls = []

    def fake(self, plate, clip_id, c, cue, look, key, merge_id, t0):
        calls.append(merge_id)
        started.set()
        frame = np.full((1080, 1920, 3), 0.5, np.float32)
        while not release.is_set():
            frame = np.clip(frame * 1.001 + 0.0001, 0, 1)
            sum(i * i for i in range(2000))
        merge_dir = self.s.standin_dir / key / "merge" / merge_id
        merge_dir.mkdir(parents=True, exist_ok=True)
        standin._write_json(merge_dir / "cue.json", cue)
        standin._write_json(merge_dir / "standin.json", {"clip": c.clip, "clip_id": clip_id, "make_key": key,
                                                         "merge_id": merge_id, "warnings": [], "seconds": 0,
                                                         "timings_s": {"make": 0.1, "merge": 0.1, "total": 0.2}})
        return False

    monkeypatch.setattr(standin.StandinStore, "_render", fake)
    yield release, started, calls
    release.set()


def timed(call) -> float:
    t = time.perf_counter()
    r = call()
    assert r.status_code == 200, r.text
    return time.perf_counter() - t


def test_light_routes_answer_while_a_standin_renders(client, uploaded, slow_render):
    release, started, _ = slow_render
    cue, clip_id = synthetic_cue(), uploaded["clip_id"]
    body = {"cue": cue, "clip_id": clip_id}
    blocking: dict = {}

    def old_style():  # a script's blocking POST (the default) on its own connection
        with TestClient(client.app) as c2:
            blocking["r"] = c2.post("/api/standin?wait=1", json=body)

    t = threading.Thread(target=old_style)
    t.start()
    assert started.wait(10)

    queued = client.post("/api/standin?wait=0", json=body)  # the same cue: joins the render on its way
    assert queued.status_code == 202
    st = queued.json()
    assert_fits("standin_status", st)
    assert (st["status"], st["result"], st["error"], st["element"]) == ("running", None, None, "none")
    sid = st["standin_id"]

    light = {
        "health": lambda: client.get("/api/health"),
        "validate": lambda: client.post("/api/validate", json={"cue": cue}),
        "preview": lambda: client.post("/api/preview", json={"cue": cue}),
        "frame": lambda: client.get(f"/api/clips/{clip_id}/frames/{7}.jpg"),
        "clips": lambda: client.get("/api/clips"),
        "status": lambda: client.get(f"/api/standin/{sid}"),
    }
    worst = {name: max(timed(call) for _ in range(3)) for name, call in light.items()}
    assert all(s < LIGHT_S for s in worst.values()), worst
    assert client.get(f"/api/standin/{sid}").json()["status"] == "running"

    release.set()
    t.join(30)
    r = blocking["r"]
    assert r.status_code == 200, r.text  # the default still answers the finished standin_result, as before
    assert_fits("standin_result", r.json())
    assert r.json()["standin_id"] == sid and r.json()["cached"] is False
    done = client.get(f"/api/standin/{sid}").json()
    assert_fits("standin_status", done)
    assert done["status"] == "done" and done["result"]["standin_id"] == sid
    again = client.post("/api/standin?wait=0", json=body)
    assert again.status_code == 200 and again.json()["status"] == "done" and again.json()["result"]["cached"] is True


def test_renders_queue_one_at_a_time(client, uploaded, slow_render):
    release, started, calls = slow_render
    cue = synthetic_cue()
    first = client.post("/api/standin?wait=0", json={"cue": cue, "clip_id": uploaded["clip_id"]}).json()
    assert started.wait(10)
    other = {**cue, "grade": {**cue["grade"], "glow": 0.3}}
    second = client.post("/api/standin?wait=0", json={"cue": other, "clip_id": uploaded["clip_id"]})
    assert second.status_code == 202 and second.json()["status"] == "queued"
    assert first["status"] in ("queued", "running") and len(calls) == 1
    release.set()
    queue = client.app.state.queue
    for sid in (first["standin_id"], second.json()["standin_id"]):
        queue.get(sid).wait()
        assert client.get(f"/api/standin/{sid}").json()["status"] == "done"
    assert len(calls) == 2


def test_a_failed_background_render_reports_its_error(client, uploaded, monkeypatch):
    def boom(*a, **k):
        raise standin._failed(standin.MAKE_STAGE)

    monkeypatch.setattr(standin.StandinStore, "_render", boom)
    st = client.post("/api/standin?wait=0", json={"cue": synthetic_cue(), "clip_id": uploaded["clip_id"]}).json()
    client.app.state.queue.get(st["standin_id"]).future.exception()
    failed = client.get(f"/api/standin/{st['standin_id']}").json()
    assert_fits("standin_status", failed)
    assert failed["status"] == "failed" and failed["error"]["code"] == "render_failed"
    assert "Building reference frame" in failed["error"]["message"]


def test_an_unexpected_render_error_is_internal(client, uploaded, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret C:\\private")

    monkeypatch.setattr(standin.StandinStore, "_render", boom)
    r = client.post("/api/standin?wait=1", json={"cue": synthetic_cue(), "clip_id": uploaded["clip_id"]})
    assert assert_error(r, 500, "internal") and "private" not in r.text


def test_standin_status_of_an_unknown_id_is_not_found(client):
    assert_error(client.get("/api/standin/000000000000-000000000000"), 404, "not_found")
    assert_error(client.get("/api/standin/nonsense"), 404, "not_found")


def test_element_with_wait_0_answers_the_status(client, uploaded, slow_render, monkeypatch):
    release, _, _ = slow_render
    release.set()
    sid = client.post("/api/standin?wait=1", json={"cue": synthetic_cue(), "clip_id": uploaded["clip_id"]}).json()[
        "standin_id"]
    gate = threading.Event()

    def element(self, key, merge_id, done):
        gate.wait(10)
        out = done.parent / "element.mp4"
        out.write_bytes(b"element")
        return out

    monkeypatch.setattr(standin.StandinStore, "element", element)
    r = client.get(f"/api/standin/{sid}/element.mp4?wait=0")
    assert r.status_code == 202 and r.json()["element"] in ("queued", "running")
    assert_fits("standin_status", r.json())
    assert timed(lambda: client.get("/api/health")) < LIGHT_S
    gate.set()
    assert client.get(f"/api/standin/{sid}/element.mp4?wait=1").content == b"element"  # ?wait=1 waits, as before v2.4
    assert client.get(f"/api/standin/{sid}").json()["element"] == "done"


def test_not_found_and_internal_are_contract_codes():
    c = api.codes()
    assert c["not_found"]["call"] == c["internal"]["call"] == "any"
    assert "{detail}" in c["not_found"]["message"] and api.message("internal")


# --------------------------------------------------------------------------- frames

def test_frames_are_cached_and_late_frames_are_as_fast_as_early_ones(client, golden_length_clip, settings):
    up = client.post("/api/clips", files={"file": (golden_length_clip.name, golden_length_clip.read_bytes())}).json()
    clip_id = up["clip_id"]
    store: ClipStore = client.app.state.store
    url = f"/api/clips/{clip_id}/frames/{{}}.jpg"
    t = time.perf_counter()
    first_190 = client.get(url.format(190)).content  # cold: decodes from the plate's keyframe
    cold = time.perf_counter() - t
    store.wait_frames(clip_id, 60)
    cached = sorted(p.name for p in store.frames_dir(clip_id).glob("*.jpg"))
    assert cached == [f"{f:05d}.jpg" for f in range(193)]  # the background pass wrote every frame
    assert not list(store.frames_dir(clip_id).glob("*.part"))

    cold_app = TestClient(create_app(settings))  # a new server on the same data: its memory LRU is empty
    times = {}
    for f in (5, 190):
        t = time.perf_counter()
        body = cold_app.get(url.format(f)).content  # from the disk cache
        times[f] = time.perf_counter() - t
        assert body == (store.frames_dir(clip_id) / f"{f:05d}.jpg").read_bytes()
    assert times[190] < max(3 * times[5], 0.05), (times, cold)
    assert client.get(url.format(190)).content == first_190  # from the memory LRU
    plate = store.plate(clip_id)
    assert first_190 == _encode(ClipStore._decode(plate, 190))  # the cache serves the exact frame
    assert (store.frames_dir(clip_id) / "00005.jpg").read_bytes() == _encode(ClipStore._decode(plate, 5))


def test_frame_lru_is_bounded(client, uploaded, monkeypatch):
    from services.analysis import clips

    monkeypatch.setattr(clips, "FRAME_LRU", 4)
    for f in range(10):
        client.get(f"/api/clips/{uploaded['clip_id']}/frames/{f}.jpg")
    assert len(client.app.state.store._jpegs) == 4


def test_an_unknown_job_and_route_are_not_found(client):
    assert_error(client.get("/api/jobs/job-0000000000"), 404, "not_found")
    assert_error(client.get("/api/nothing/here"), 404, "not_found")
    assert_error(client.get("/api/validate"), 405, "input_invalid")


def test_contract_version_is_30(client):
    assert client.get("/api/health").json()["contract_version"] == "3.0"
    assert json.loads((api.SCHEMA_DIR / "error_codes.json").read_text(encoding="utf-8"))["contract_version"] == "3.0"


def test_heavy_routes_answer_at_once_by_default(client, uploaded, slow_render, monkeypatch):
    """v2.4: no `wait` means ?wait=0; ?wait=1 is the old blocking answer."""
    release, started, _ = slow_render
    body = {"cue": synthetic_cue(), "clip_id": uploaded["clip_id"]}
    t = time.perf_counter()
    r = client.post("/api/standin", json=body)  # no wait: 202 while the render is held
    assert r.status_code == 202 and time.perf_counter() - t < LIGHT_S, r.text
    assert_fits("standin_status", r.json())
    assert r.json()["status"] in ("queued", "running")
    sid = r.json()["standin_id"]
    release.set()
    client.app.state.queue.get(sid).wait()
    old = client.post("/api/standin?wait=1", json=body)  # the old body: the standin_result
    assert old.status_code == 200
    assert_fits("standin_result", old.json())
    assert old.json()["standin_id"] == sid

    gate = threading.Event()

    def element(self, key, merge_id, done):
        gate.wait(10)
        out = done.parent / "element.mp4"
        out.write_bytes(b"element")
        return out

    monkeypatch.setattr(standin.StandinStore, "element", element)
    r = client.get(f"/api/standin/{sid}/element.mp4")  # no wait: the status, not the file
    assert r.status_code == 202 and r.json()["element"] in ("queued", "running")
    gate.set()
    assert client.get(f"/api/standin/{sid}/element.mp4?wait=1").content == b"element"
