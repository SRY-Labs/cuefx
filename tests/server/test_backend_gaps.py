"""Task 30 (contract v2.4): GET /api/looks, GET /api/jobs/{id}/result and its files, POST /api/manual/export?wait=0.
The default flip of the heavy routes is tested in test_responsiveness.py; a result of a real manual job is in
test_manual.py (slow)."""
import io
import json
import threading
import time
import zipfile

import pytest

from server_helpers import (uploaded_template,  # noqa: F401
                            assert_error, assert_fits, clip_file, client, make_clip, settings, synthetic_cue, uploaded)
from services.cue import api
from services.make.ingest import MATTE_WARNING
from services.delivery.manual import ManualStore

LOOK = "energy_orb"
JOB = "job-0123456789"


def put_looks(settings, *names: str) -> None:
    settings.looks.mkdir(parents=True, exist_ok=True)
    for n in names:
        (settings.looks / n).write_bytes(b"not read by the list")


# --------------------------------------------------------------------------- looks

def test_looks_of_a_missing_or_empty_folder_is_an_empty_list(client, settings):
    assert client.get("/api/looks").json() == []  # no folder yet
    settings.looks.mkdir(parents=True)
    assert client.get("/api/looks").json() == []


def test_looks_lists_one_entry_per_png(client, settings):
    put_looks(settings, "energy_orb.png")
    assert client.get("/api/looks").json() == [{"id": "energy_orb", "name": "Energy Orb", "file": "energy_orb.png"}]
    put_looks(settings, "blue_plasma.png", "Fire-ball.png", "notes.txt", "photo.jpg", "Energy Orb copy.png", "x.PNG")
    got = client.get("/api/looks").json()
    assert got == [{"id": "Fire-ball", "name": "Fire-Ball", "file": "Fire-ball.png"},
                   {"id": "blue_plasma", "name": "Blue Plasma", "file": "blue_plasma.png"},
                   {"id": "energy_orb", "name": "Energy Orb", "file": "energy_orb.png"}]  # sorted by id; no txt/jpg/.PNG
    assert [x["id"] for x in got] == sorted(x["id"] for x in got)  # a name with a space cannot be a look id: not listed


def test_an_unknown_look_on_export_lists_the_valid_ids(client, uploaded, settings):
    body = {"cue": synthetic_cue(), "clip_id": uploaded["clip_id"], "look": "fireball"}
    none = assert_error(client.post("/api/manual/export", json=body), 400, "input_invalid")["message"]
    assert "fireball.png" in none and "no look images" in none
    put_looks(settings, "energy_orb.png", "blue_plasma.png")
    msg = assert_error(client.post("/api/manual/export", json=body), 400, "input_invalid")["message"]
    assert "Valid look ids: blue_plasma, energy_orb." in msg
    assert not settings.jobs_dir.exists()  # no job is made for a request that cannot start


def test_the_standin_does_not_read_the_looks_folder(client, uploaded, settings):
    """The stand-in draws its own orb (CUEFX_LOOK_IMAGE); the cue's `look` is only an id for generate and the manual
    render, so an id that is not in the looks folder must not stop it (the UI's cue carries its mock look ids)."""
    r = client.post("/api/standin?wait=1", json={"cue": {**synthetic_cue(), "look": "fireball"},
                                                 "clip_id": uploaded["clip_id"]})
    assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- result

def fake_done_job(settings, status: str = "done", matte: bool = False, warnings=None) -> str:
    """A job folder as a finished manual job leaves it (tiny videos, the real shapes), without the render."""
    d = settings.jobs_dir / JOB
    for sub in ("merge", "ltx", "upload"):
        (d / sub).mkdir(parents=True)
    cue = {**synthetic_cue(), "look": LOOK}
    for p in ("merge/final.mp4", "merge/compare.mp4", "ltx/effect.mp4", "ltx/matte.mp4"):
        make_clip(d / p, frames=4, size=(64, 36))
    if matte:
        make_clip(d / "upload" / "matte.mp4", frames=4, size=(64, 36))
    (d / "merge" / "cue.json").write_text(json.dumps(cue), encoding="utf-8")
    (d / "export.json").write_text(json.dumps({"created": 1.0, "clip": "synthetic.mp4", "clip_id": "c" * 12,
                                               "look": LOOK, "cue": cue}), encoding="utf-8")
    notes = ["no prompt: type one in the graph's Prompt node before you queue it", MATTE_WARNING] \
        if warnings is None else warnings
    job = {"id": JOB, "look": LOOK, "status": status, "stage": 5 if status == "done" else 2,
           "stage_name": "Done" if status == "done" else "Rendering (pass 1 of 2)", "progress": 1, "elapsed_s": 1.5,
           "eta_s": 0, "error": None, "manual": True, "clip_id": "c" * 12, "make_key": "k" * 12, "files": {},
           "warnings": notes}
    (d / "job.json").write_text(json.dumps(job), encoding="utf-8")
    return JOB


def test_result_of_a_done_job_fits_the_schema_and_every_url_works(client, settings):
    job_id = fake_done_job(settings)
    r = client.get(f"/api/jobs/{job_id}/result")
    assert r.status_code == 200, r.text
    res = r.json()
    assert_fits("result", res)
    assert (res["job_id"], res["look"], res["pipeline"]) == (job_id, LOOK, "composite")
    urls = [res["videos"]["final"], res["videos"]["compare"], res["layers"]["effect_on_black"], res["layers"]["matte"]]
    assert urls == [f"/api/jobs/{job_id}/{n}" for n in ("final.mp4", "compare.mp4", "effect_on_black.mp4", "matte.mp4")]
    assert res["layers"]["occluder_matte"] is None  # no step writes one
    for url in urls:
        got = client.get(url)
        assert got.status_code == 200 and got.headers["content-type"] == "video/mp4" and len(got.content) > 0
        assert "attachment" not in got.headers.get("content-disposition", "")
        assert client.get(url, headers={"Range": "bytes=0-99"}).status_code == 206  # range requests, like the others
        dl = client.get(url + "?download=1")
        assert dl.headers["content-disposition"].startswith("attachment") and f"synthetic_{job_id}_" in dl.headers[
            "content-disposition"]
    assert client.get(urls[0]).content == (settings.jobs_dir / job_id / "merge" / "final.mp4").read_bytes()


def test_result_checks_say_what_is_not_measured(client, settings):
    res = client.get(f"/api/jobs/{fake_done_job(settings)}/result").json()
    checks = {c["id"]: c for c in res["checks"]}
    assert list(checks) == ["plate_unchanged", "black_clamp", "matte_source", "face_shift_px"]
    for name in ("plate_unchanged", "black_clamp", "face_shift_px"):  # nobody computes these yet
        assert (checks[name]["status"], checks[name]["value"], checks[name]["message"]) == ("warn", None,
                                                                                            "not measured yet")
    assert (checks["matte_source"]["status"], checks["matte_source"]["value"], checks["matte_source"]["message"]) == (
        "warn", "temporary", MATTE_WARNING)
    assert checks["matte_source"]["frames"] == checks["black_clamp"]["frames"] == [4, 21]  # the cue's window, [start, end)
    notes = [w for w in res["warnings"] if w["code"] == "render_note"]
    assert [w["message"] for w in notes] == ["no prompt: type one in the graph's Prompt node before you queue it"]
    assert all(w["code"] in api.codes() for w in res["warnings"])


def test_result_with_an_uploaded_matte_passes_matte_source(client, settings):
    res = client.get(f"/api/jobs/{fake_done_job(settings, matte=True, warnings=[])}/result").json()
    assert_fits("result", res)
    matte = next(c for c in res["checks"] if c["id"] == "matte_source")
    assert (matte["status"], matte["value"]) == ("pass", "uploaded")
    assert not [w for w in res["warnings"] if w["code"] == "render_note"]


def test_result_of_a_job_that_is_not_done_is_409_and_unknown_is_404(client, settings):
    job_id = fake_done_job(settings, status="waiting")
    body = assert_error(client.get(f"/api/jobs/{job_id}/result"), 409, "input_invalid")
    assert "not done yet" in body["message"] and "waiting" in body["message"]
    assert_error(client.get(f"/api/jobs/{job_id}/final.mp4"), 404, "not_found")  # files only once it is done
    assert_error(client.get("/api/jobs/job-0000000000/result"), 404, "not_found")
    assert_error(client.get("/api/jobs/nonsense/result"), 404, "not_found")
    assert_error(client.get(f"/api/jobs/{job_id}/package.zip"), 404, "not_found")  # not a result file


# --------------------------------------------------------------------------- export?wait=0

@pytest.fixture
def look(settings):
    put_looks(settings, f"{LOOK}.png")
    import cv2

    from services.make.look import standin_look
    cv2.imwrite(str(settings.looks / f"{LOOK}.png"), standin_look())


def test_export_with_wait_0_answers_at_once_and_the_job_reaches_waiting(client, uploaded, settings, look, monkeypatch):
    gate = threading.Event()
    make = ManualStore._make

    def slow_make(self, *a):
        gate.wait(30)
        return make(self, *a)

    monkeypatch.setattr(ManualStore, "_make", slow_make)
    body = {"cue": synthetic_cue(), "clip_id": uploaded["clip_id"], "look": LOOK}
    t = time.perf_counter()
    r = client.post("/api/manual/export?wait=0", json=body)
    assert r.status_code == 202 and time.perf_counter() - t < 1, r.text
    job = r.json()
    assert_fits("job", job)
    job_id = job["id"]
    assert r.headers["x-cuefx-job-id"] == job_id
    assert (job["status"], job["stage"], job["stage_name"], job["manual"]) == ("running", 0, "Preparing shot", True)
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "running"
    assert [j["id"] for j in client.get("/api/jobs").json()] == [job_id]  # listed while it builds
    assert_error(client.get(f"/api/manual/{job_id}/package.zip"), 404, "not_found")  # no zip yet
    upload = client.post(f"/api/manual/{job_id}/result?wait=1", files={"effect_on_black": ("e.mp4", b"x")})
    assert "no package yet" in assert_error(upload, 409, "input_invalid")["message"]  # nothing to check an upload against
    gate.set()
    waiting = client.app.state.queue.get(f"{job_id}/export").wait()
    assert_fits("job", waiting)
    assert (waiting["status"], waiting["stage"], waiting["files"]) == (
        "waiting", 2, {"package_download": f"/api/manual/{job_id}/package.zip"})
    assert client.get(f"/api/jobs/{job_id}").json() == waiting

    # the same package as a blocking export of the same cue: equal but for the job id inside it
    zip0 = client.get(waiting["files"]["package_download"])
    assert zip0.status_code == 200 and zip0.headers["content-type"] == "application/zip"
    old = client.post("/api/manual/export?wait=1", json=body)
    assert old.status_code == 200 and old.headers["content-type"] == "application/zip"
    z0, z1 = zipfile.ZipFile(io.BytesIO(zip0.content)), zipfile.ZipFile(io.BytesIO(old.content))
    id1 = old.headers["x-cuefx-job-id"]
    assert id1 != job_id
    assert sorted(n.replace(id1, "JOB") for n in z1.namelist()) == sorted(n.replace(job_id, "JOB") for n in z0.namelist())
    for n in z0.namelist():
        a, b = z0.read(n), z1.read(n.replace(job_id, id1))
        if n.endswith((".json", ".txt")):  # the job id is in the manifest, the graph and the three lines
            a, b = a.replace(job_id.encode(), b"JOB"), b.replace(id1.encode(), b"JOB")
        assert a == b, n


def test_a_failed_export_leaves_a_failed_job(client, uploaded, settings, look, monkeypatch):
    from services.make.layout import LayoutError
    from services.delivery import manual

    def boom(*a, **k):
        raise LayoutError("secret C:\\private")

    monkeypatch.setattr(manual, "make_layout", boom)
    body = {"cue": synthetic_cue(), "clip_id": uploaded["clip_id"], "look": LOOK}
    r = client.post("/api/manual/export?wait=0", json=body)
    job_id = r.json()["id"]
    with pytest.raises(Exception):
        client.app.state.queue.get(f"{job_id}/export").wait()
    job = client.get(f"/api/jobs/{job_id}").json()
    assert_fits("job", job)
    assert job["status"] == "failed" and job["error"]["code"] == "render_failed"
    assert "Building reference frame" in job["error"]["message"] and "private" not in json.dumps(job)
    upload = client.post(f"/api/manual/{job_id}/result?wait=1", files={"effect_on_black": ("e.mp4", b"x")})
    assert_error(upload, 409, "input_invalid")
    again = client.post("/api/manual/export?wait=1", json=body)  # the blocking form raises the error too
    assert_error(again, 500, "render_failed")
    assert sorted(j["status"] for j in client.get("/api/jobs").json()) == ["failed", "failed"]
