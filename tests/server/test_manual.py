"""Task 21: the manual render bridge (export a ComfyUI package, import the LTX result, ingest + Merge, jobs)."""
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server_helpers import (uploaded_template,  # noqa: F401
                            assert_error, assert_fits, clip_file, client, golden, golden_length_clip,  # noqa: F401
                            settings, synthetic_cue, uploaded)
from services.cue import api
from services.make.comfy_graph import load_nodes
from services.cue.cue import cue_from_dict, cue_to_dict
from services.make.ingest import MATTE_WARNING
from services.make.look import standin_look
from services.cue.make_key import make_key
from services.delivery.video import VideoWriter, read_frames
from services import config
from app import create_app

LOOK = "energy_orb"
WINDOW = 17  # frames of synthetic_cue()'s effect window, 4..20
CANVAS = (512, 256)


@pytest.fixture
def look(settings) -> Path:
    p = settings.looks / f"{LOOK}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), standin_look())
    return p


def export(client, cue: dict, clip_id: str, **kw):
    return client.post("/api/manual/export?wait=1", json={"cue": cue, "clip_id": clip_id, "look": LOOK, **kw})


def package(r) -> tuple[str, zipfile.ZipFile]:
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"].startswith("attachment")
    return r.headers["x-cuefx-job-id"], zipfile.ZipFile(io.BytesIO(r.content))


def effect_video(path: Path, frames: int = WINDOW, size=CANVAS, fps: float = 24) -> Path:
    """A synthetic LTX result: a bright disc on near-black, moving right."""
    with VideoWriter(path, size, fps, crf=12) as w:
        for i in range(frames):
            img = np.full((size[1], size[0], 3), 6, np.uint8)
            cv2.circle(img, (60 + 20 * i, size[1] // 2), 30, (255, 220, 160), -1, cv2.LINE_AA)
            w.write(img)
    return path


def upload(client, job_id: str, effect: Path, matte: Path | None = None, cue: dict | None = None):
    files = {"effect_on_black": ("effect.mp4", effect.read_bytes(), "video/mp4")}
    if matte is not None:
        files["matte"] = ("matte.mp4", matte.read_bytes(), "video/mp4")
    data = {"cue": json.dumps(cue)} if cue is not None else None
    return client.post(f"/api/manual/{job_id}/result?wait=1", files=files, data=data)


# --------------------------------------------------------------------------- export

@pytest.mark.slow  # a full render (task 27: kept out of the default run)
def test_export_on_the_golden_cue(client, golden_length_clip, settings, look):
    up = client.post("/api/clips", files={"file": (golden_length_clip.name, golden_length_clip.read_bytes())}).json()
    cue = cue_to_dict(replace(cue_from_dict(golden("throw_71319.cue.json")),
                              track=api.points_from_track(golden("throw_71319.track.json"))))
    job_id, z = package(export(client, cue, up["clip_id"], prompt="A blue-white energy orb on his palm."))
    graph_name = f"CueFX_Phase1_Quality_{job_id}.json"
    inputs = {f"comfy_input/{job_id}_layout.mp4", f"comfy_input/{job_id}_look.png"}
    assert set(z.namelist()) == inputs | {graph_name, "manifest.json", "HOW_TO_RUN.txt"}

    graph = json.loads(z.read(graph_name))  # loads as JSON, UI format
    assert {"nodes", "links", "definitions"} <= set(graph)
    loaded = sorted(n["widgets_values"][0] for n in load_nodes(graph))
    assert loaded == sorted(Path(n).name for n in inputs)  # every load node reads a file of the package
    prompt = next(n for n in graph["nodes"] if (n.get("title") or "").startswith("Prompt · describe"))
    assert prompt["widgets_values"] == ["A blue-white energy orb on his palm."]

    m = json.loads(z.read("manifest.json"))
    key = make_key(cue_from_dict(cue), look)
    assert (m["job_id"], m["make_key"], m["clip_id"], m["look"]) == (job_id, key, up["clip_id"], LOOK)
    assert m["effect_range"] == {"start": 26, "end": 115}  # grown over the entrance, as Make draws it
    out = m["output"]
    assert (out["file"], out["width"], out["height"], out["frames"], out["fps"]) == ("effect.mp4", 1024, 576, 89, 24)
    assert m["comfy_input"] == {"layout": f"{job_id}_layout.mp4", "look": f"{job_id}_look.png"}

    make_dir = settings.manual_dir / key
    assert z.read(f"comfy_input/{job_id}_layout.mp4") == (make_dir / "layout.mp4").read_bytes()
    assert z.read(f"comfy_input/{job_id}_look.png") == (make_dir / "look_placed.png").read_bytes()
    look_png = cv2.imdecode(np.frombuffer(z.read(f"comfy_input/{job_id}_look.png"), np.uint8), cv2.IMREAD_COLOR)
    assert look_png.shape == (576, 1024, 3)  # the canvas: the graph fits nothing
    for debug in ("layout_sheet.png", "look_check.png", "layout_overlay.mp4"):
        assert (make_dir / debug).is_file()
    steps = z.read("HOW_TO_RUN.txt").decode().splitlines()
    assert len(steps) == 3 and "ComfyUI\\input" in steps[0] and graph_name in steps[1]
    assert f"cuefx\\{job_id}_" in steps[2] and "effect.mp4" in steps[2] and "1024x576, 89 frames" in steps[2]

    job = client.get(f"/api/jobs/{job_id}").json()
    assert_fits("job", job)
    assert (job["status"], job["manual"], job["stage"], job["make_key"], job["look"]) == ("waiting", True, 2, key, LOOK)
    assert job["files"] == {"package_download": f"/api/manual/{job_id}/package.zip"}
    assert client.get(job["files"]["package_download"]).content == (settings.jobs_dir / job_id / "package.zip").read_bytes()
    assert_error(client.get(f"/api/manual/{job_id}/final.mp4"), 404, "not_found")  # not done yet


def test_export_without_the_look_image_is_input_invalid(client, uploaded, settings):
    body = assert_error(export(client, synthetic_cue(), uploaded["clip_id"]), 400, "input_invalid")
    assert f"{LOOK}.png" in body["message"] and "CUEFX_LOOKS_DIR" in body["message"]
    assert_error(client.post("/api/manual/export?wait=1", json={"cue": synthetic_cue(), "clip_id": uploaded["clip_id"],
                                                         "look": "../x"}), 400, "input_invalid")
    assert not settings.jobs_dir.exists() and not settings.manual_dir.exists()  # nothing made, no job


def test_export_takes_the_cues_prompt_and_warns_when_there_is_none(client, uploaded, look):
    cue = synthetic_cue()
    _, z = package(export(client, cue, uploaded["clip_id"]))
    assert json.loads(z.read("manifest.json"))["prompt"] == ""
    assert "no prompt" in " ".join(json.loads(z.read("manifest.json"))["warnings"])
    assert "it is empty" in z.read("HOW_TO_RUN.txt").decode()
    _, z = package(export(client, {**cue, "prompt": "from the cue"}, uploaded["clip_id"]))
    assert json.loads(z.read("manifest.json"))["prompt"] == "from the cue"


def test_export_of_a_short_window_keeps_the_graph_at_121_and_says_so_in_the_manifest(client, uploaded, look):
    job_id, z = package(export(client, synthetic_cue(), uploaded["clip_id"]))
    m = json.loads(z.read("manifest.json"))
    assert m["graph_frames"] == 121 and m["output"]["frames"] == WINDOW   # a 17-frame window: today's graph
    graph = json.loads(z.read(f"CueFX_Phase1_Quality_{job_id}.json"))
    assert next(n for n in graph["nodes"] if n["type"] == "CueFXOfficialPrepare")["widgets_values"][2] == 121


@pytest.mark.slow  # a 193-frame render
def test_export_of_an_8_second_hold_raises_the_graph_to_193_frames(client, golden_length_clip, settings, look):
    frames = 193
    up = client.post("/api/clips", files={"file": (golden_length_clip.name, golden_length_clip.read_bytes())}).json()
    clip_id = up["clip_id"]
    track = [{"frame": f, "x": 600.0 + 3 * f, "y": 660.0, "confidence": 0.9, "uncertain": False, "source": "tracked",
              "palm_w": 120.0} for f in range(frames)]
    cue = api.new_cue(up["clip"], 24, frames, [1920, 1080], "right", {"f": 5, "x": 660, "y": 660}, track,
                      {"scale": 6.0, "offset": [0, 0]})
    cue.update(behaviour="hold", release_frame=None, route=[], canvas_size=[512, 256], canvas_rect=[0, 0, 1920, 960],
               appear_frame=12, diameter_px=120, vanish={"start": 186, "end": 192}, entrance={"seconds": 0.5},
               effect_range={"start": 0, "end": 193})
    cue = api.normalize(cue)
    assert [w for w in api.validate(cue) if w["severity"] == "error"] == []
    job_id, z = package(export(client, cue, clip_id))
    m = json.loads(z.read("manifest.json"))
    assert m["graph_frames"] == 193 and m["effect_range"] == {"start": 0, "end": 193}
    assert (m["output"]["frames"], m["warnings"]) == (193, ["no prompt: type one in the graph's Prompt node before you queue it"])
    graph = json.loads(z.read(f"CueFX_Phase1_Quality_{job_id}.json"))
    assert next(n for n in graph["nodes"] if n["type"] == "CueFXOfficialPrepare")["widgets_values"][2] == 193
    # a Hold: Make drew the orb on the hand in every frame, the same key as the cue gives
    assert m["make_key"] == make_key(cue_from_dict(cue), look)
    assert make_key(cue_from_dict({**cue, "behaviour": "hold_throw"}), look) != m["make_key"]


def test_export_with_an_invalid_cue_is_input_invalid(client, uploaded, look):
    bad = {**synthetic_cue(), "release_frame": 3}
    assert_error(export(client, bad, uploaded["clip_id"]), 400, "input_invalid")
    assert_error(export(client, synthetic_cue(), "000000000000"), 404, "not_found")


# --------------------------------------------------------------------------- import

@pytest.mark.slow  # a full render (task 27: kept out of the default run)
def test_import_runs_ingest_and_merge(client, uploaded, settings, look, tmp_path):
    job_id, _ = package(export(client, synthetic_cue(), uploaded["clip_id"]))
    effect = effect_video(tmp_path / "effect.mp4")
    r = upload(client, job_id, effect)
    assert r.status_code == 200, r.text
    job = r.json()
    assert_fits("job", job)
    assert (job["status"], job["stage"], job["stage_name"], job["progress"], job["error"]) == (
        "done", 5, "Done", 1.0, None)
    assert MATTE_WARNING in job["warnings"]  # no matte uploaded: the temporary one
    assert set(job["files"]) == {"package_download", "final_preview", "final_download"}
    assert client.get(f"/api/jobs/{job_id}").json() == job

    preview = client.get(job["files"]["final_preview"])
    assert preview.status_code == 200 and preview.headers["content-type"] == "video/mp4"
    assert "attachment" not in preview.headers.get("content-disposition", "")
    download = client.get(job["files"]["final_download"])
    assert download.headers["content-disposition"].startswith("attachment")
    assert f"synthetic_{job_id}_final.mp4" in download.headers["content-disposition"]
    final = tmp_path / "final.mp4"
    final.write_bytes(download.content)
    assert download.content == preview.content
    d = settings.jobs_dir / job_id
    for f in ("upload/effect_on_black.mp4", "ltx/effect.mp4", "ltx/matte.mp4", "ltx/ingest_sheet.png",
              "merge/compare.mp4", "merge/merge_sheet.png", "merge/cue.json"):
        assert (d / f).is_file(), f
    assert len(list(read_frames(final))) == 24 and next(read_frames(final)).shape == (1080, 1920, 3)


@pytest.mark.slow  # a full render (task 27: kept out of the default run)
def test_result_of_a_real_job_fits_the_schema_and_every_url_works(client, uploaded, settings, look, tmp_path):
    job_id, _ = package(export(client, synthetic_cue(), uploaded["clip_id"]))
    files = {"effect_on_black": ("effect.mp4", effect_video(tmp_path / "effect.mp4").read_bytes())}
    r = client.post(f"/api/manual/{job_id}/result", files=files)  # no `wait`: v2.4 answers at once
    assert r.status_code == 202 and r.json()["status"] == "running"
    assert_error(client.get(f"/api/jobs/{job_id}/result"), 409, "input_invalid")  # still blending
    client.app.state.queue.get(f"{job_id}/import").wait()
    res = client.get(f"/api/jobs/{job_id}/result").json()
    assert_fits("result", res)
    (tmp_path / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    urls = [res["videos"]["final"], res["videos"]["compare"], res["layers"]["effect_on_black"], res["layers"]["matte"]]
    assert all(u and client.get(u).status_code == 200 for u in urls)
    d = settings.jobs_dir / job_id
    assert client.get(urls[0]).content == (d / "merge" / "final.mp4").read_bytes()
    assert client.get(urls[1]).content == (d / "merge" / "compare.mp4").read_bytes()
    assert client.get(urls[2]).content == (d / "ltx" / "effect.mp4").read_bytes()
    assert client.get(urls[3]).content == (d / "ltx" / "matte.mp4").read_bytes()
    assert next(c for c in res["checks"] if c["id"] == "matte_source")["value"] == "temporary"


@pytest.mark.slow  # a full render (task 27: kept out of the default run)
def test_import_with_a_matte_uses_it(client, uploaded, settings, look, tmp_path):
    job_id, _ = package(export(client, synthetic_cue(), uploaded["clip_id"]))
    matte = tmp_path / "matte.mp4"
    with VideoWriter(matte, CANVAS, 24, crf=12) as w:
        for _ in range(WINDOW):
            w.write(np.full((CANVAS[1], CANVAS[0], 3), 200, np.uint8))
    job = upload(client, job_id, effect_video(tmp_path / "effect.mp4"), matte).json()
    assert job["status"] == "done" and MATTE_WARNING not in job["warnings"]
    ingested = next(read_frames(settings.jobs_dir / job_id / "ltx" / "matte.mp4"))
    assert abs(int(ingested[..., 1].mean()) - 200) <= 2  # the uploaded matte, not one made from the effect


@pytest.mark.slow  # a full render (task 27: kept out of the default run)
@pytest.mark.parametrize("make,expect", [
    (lambda p: effect_video(p, size=(640, 256)), "is 640x256, the export expects 512x256"),
    (lambda p: effect_video(p, frames=WINDOW - 2), f"has {WINDOW - 2} frames, the export expects {WINDOW}"),
    (lambda p: effect_video(p, fps=30), "is 30 fps, the export expects 24"),
])
def test_import_of_the_wrong_video_is_input_invalid(client, uploaded, settings, look, tmp_path, make, expect):
    job_id, _ = package(export(client, synthetic_cue(), uploaded["clip_id"]))
    body = assert_error(upload(client, job_id, make(tmp_path / "bad.mp4")), 400, "input_invalid")
    assert expect in body["message"] and "manifest.json" in body["message"]
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "waiting" and job["error"] is None
    assert not list((settings.jobs_dir / job_id / "upload").iterdir())  # the bad file is not kept
    assert upload(client, job_id, effect_video(tmp_path / "good.mp4")).json()["status"] == "done"  # still takes one


def test_import_after_the_cue_changed_is_input_invalid(client, uploaded, look, tmp_path):
    cue = synthetic_cue()
    job_id, _ = package(export(client, cue, uploaded["clip_id"]))
    effect = effect_video(tmp_path / "effect.mp4")
    moved = {**cue, "diameter_px": 100, "route": [{**cue["route"][0], "diameter_px": 100}, cue["route"][1]]}
    body = assert_error(upload(client, job_id, effect, cue=moved), 400, "input_invalid")
    assert "the cue changed since export" in body["message"]
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "waiting"
    look.write_bytes(look.read_bytes() + b"\0")  # the look image file changed: the make key did too
    body = assert_error(upload(client, job_id, effect), 400, "input_invalid")
    assert "look image" in body["message"]


@pytest.mark.slow  # a full render (task 27: kept out of the default run)
def test_import_with_a_merge_only_change_merges_that_cue(client, uploaded, settings, look, tmp_path):
    cue = synthetic_cue()
    job_id, _ = package(export(client, cue, uploaded["clip_id"]))
    glow = {**cue, "grade": {**cue["grade"], "glow": 0.2}}
    job = upload(client, job_id, effect_video(tmp_path / "effect.mp4"), cue=glow).json()
    assert job["status"] == "done"
    used = json.loads((settings.jobs_dir / job_id / "merge" / "cue.json").read_text(encoding="utf-8"))
    assert used["grade"]["glow"] == 0.2


def test_import_errors(client, uploaded, look, tmp_path):
    effect = effect_video(tmp_path / "effect.mp4")
    assert_error(upload(client, "job-0000000000", effect), 404, "not_found")
    assert_error(upload(client, "nonsense", effect), 404, "not_found")
    job_id, _ = package(export(client, synthetic_cue(), uploaded["clip_id"]))
    not_video = tmp_path / "x.mp4"
    not_video.write_bytes(b"not a video")
    assert "not a video" in assert_error(upload(client, job_id, not_video), 400, "input_invalid")["message"]
    r = client.post(f"/api/manual/{job_id}/result?wait=1", files={"effect_on_black": ("e.mp4", effect.read_bytes())},
                    data={"cue": "{nope"})
    assert_error(r, 400, "input_invalid")
    assert_error(client.post(f"/api/manual/{job_id}/result?wait=1"), 400, "input_invalid")  # no file


@pytest.mark.slow  # a full render (task 27: kept out of the default run)
def test_merge_failure_fails_the_job(client, uploaded, look, tmp_path, monkeypatch):
    from services.delivery.video import VideoError
    from services.delivery import manual

    def boom(*a, **k):
        raise VideoError("ffmpeg exploded at C:\\private")

    job_id, _ = package(export(client, synthetic_cue(), uploaded["clip_id"]))
    monkeypatch.setattr(manual, "run_merge", boom)
    body = assert_error(upload(client, job_id, effect_video(tmp_path / "effect.mp4")), 500, "render_failed")
    assert "Blending into your footage" in body["message"] and "private" not in body["message"]
    job = client.get(f"/api/jobs/{job_id}").json()
    assert_fits("job", job)
    assert job["status"] == "failed" and job["error"]["code"] == "render_failed"
    monkeypatch.undo()
    assert upload(client, job_id, effect_video(tmp_path / "effect.mp4")).json()["status"] == "done"


@pytest.mark.slow  # a full render (task 27: kept out of the default run)
def test_import_with_wait_0_blends_in_the_background(client, uploaded, settings, look, tmp_path):
    job_id, _ = package(export(client, synthetic_cue(), uploaded["clip_id"]))
    r = client.post(f"/api/manual/{job_id}/result?wait=0",
                    files={"effect_on_black": ("effect.mp4", effect_video(tmp_path / "effect.mp4").read_bytes())})
    assert r.status_code == 202, r.text
    job = r.json()
    assert_fits("job", job)
    assert (job["status"], job["stage"], job["stage_name"]) == ("running", 4, "Blending into your footage")
    client.app.state.queue.get(f"{job_id}/import").wait()
    done = client.get(f"/api/jobs/{job_id}").json()
    assert_fits("job", done)
    assert done["status"] == "done" and client.get(done["files"]["final_preview"]).status_code == 200


@pytest.mark.slow  # a full render (task 27: kept out of the default run)
def test_an_upload_while_the_job_blends_is_409_and_light_routes_answer(client, uploaded, look, tmp_path,
                                                                        monkeypatch):
    import threading
    import time

    from services.delivery.manual import ManualStore

    job_id, _ = package(export(client, synthetic_cue(), uploaded["clip_id"]))
    gate = threading.Event()
    blend = ManualStore._blend

    def slow_blend(self, *a):
        gate.wait(20)
        return blend(self, *a)

    monkeypatch.setattr(ManualStore, "_blend", slow_blend)
    effect = effect_video(tmp_path / "effect.mp4")
    files = {"effect_on_black": ("effect.mp4", effect.read_bytes())}
    assert client.post(f"/api/manual/{job_id}/result?wait=0", files=files).status_code == 202
    try:
        assert_error(client.post(f"/api/manual/{job_id}/result?wait=0", files=files), 409, "input_invalid")
        for route in ("/api/health", f"/api/jobs/{job_id}", "/api/jobs"):
            t = time.perf_counter()
            assert client.get(route).status_code == 200 and time.perf_counter() - t < 1
        assert client.get(f"/api/jobs/{job_id}").json()["status"] == "running"
        assert_error(client.get(f"/api/manual/{job_id}/final.mp4"), 404, "not_found")
    finally:
        gate.set()
    assert client.app.state.queue.get(f"{job_id}/import").wait()["status"] == "done"


def test_an_unexpected_import_error_fails_the_job(client, uploaded, look, tmp_path, monkeypatch):
    from services.delivery import manual

    def boom(*a, **k):
        raise RuntimeError("secret C:\\private")

    job_id, _ = package(export(client, synthetic_cue(), uploaded["clip_id"]))
    monkeypatch.setattr(manual, "ingest_ltx", boom)
    r = upload(client, job_id, effect_video(tmp_path / "effect.mp4"))
    assert_error(r, 500, "internal")
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "failed" and job["error"]["code"] == "render_failed"  # never left running


# --------------------------------------------------------------------------- jobs

@pytest.mark.slow  # a full render (task 27: kept out of the default run)
def test_jobs_survive_a_server_restart(client, uploaded, settings, look, tmp_path):
    first, _ = package(export(client, synthetic_cue(), uploaded["clip_id"]))
    second, _ = package(export(client, synthetic_cue(), uploaded["clip_id"]))
    done = upload(client, first, effect_video(tmp_path / "effect.mp4")).json()
    before = client.get("/api/jobs").json()
    assert [j["id"] for j in before] == [second, first]  # newest first
    for j in before:
        assert_fits("job", j)

    again = TestClient(create_app(settings), raise_server_exceptions=False)  # a new server on the same data
    assert again.get("/api/jobs").json() == before
    assert again.get(f"/api/jobs/{first}").json() == done
    assert again.get(done["files"]["final_preview"]).content == client.get(done["files"]["final_preview"]).content
    assert upload(again, second, effect_video(tmp_path / "effect.mp4")).json()["status"] == "done"
    assert_error(again.get("/api/jobs/job-0000000000"), 404, "not_found")


@pytest.mark.slow  # a full render (task 27: kept out of the default run)
def test_a_job_left_running_comes_back_failed(client, uploaded, settings, look, tmp_path):
    job_id, _ = package(export(client, synthetic_cue(), uploaded["clip_id"]))
    path = settings.jobs_dir / job_id / "job.json"
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), "status": "running", "stage": 4,
                                "stage_name": "Blending into your footage"}), encoding="utf-8")
    again = TestClient(create_app(settings), raise_server_exceptions=False)
    job = again.get(f"/api/jobs/{job_id}").json()
    assert_fits("job", job)
    assert job["status"] == "failed" and "Blending into your footage" in job["error"]["message"]
    assert upload(again, job_id, effect_video(tmp_path / "effect.mp4")).json()["status"] == "done"


def test_looks_dir_setting():
    assert config.from_env({"CUEFX_LOOKS_DIR": "x/looks"}).looks.name == "looks"
    d = config.from_env({"CUEFX_DATA_DIR": "somewhere"})
    assert d.looks == d.data_dir / "looks" and d.jobs_dir == d.data_dir / "jobs"
    assert d.quality_graph.is_file()
