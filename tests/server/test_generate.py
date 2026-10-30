"""Focused local tests for the experimental one-click LTX orchestration."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from server_helpers import (assert_error, assert_fits, client, clip_file, settings, synthetic_cue, uploaded,
                            uploaded_template)  # noqa: F401
from services.make.comfy_worker import WorkerError, WorkerResult
from services.delivery.video import read_frames


class FakeAdapters:
    def __init__(self, fail: str | None = None):
        self.fail = fail
        self.calls = []

    @staticmethod
    def result(path: Path, kind: str) -> WorkerResult:
        return WorkerResult(path, f"{kind}-worker-id", {}, {"filename": path.name, "type": "output"})

    def edit_look(self, reference, prompt, seed, output_dir, cancelled):
        self.calls.append(("edit", reference, prompt, seed))
        if self.fail == "edit":
            raise WorkerError("image edit failed")
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "edited.png"
        shutil.copy2(reference, out)
        return self.result(out, "edit")

    def generate_main(self, video, look, prompt, output_dir, cancelled):
        self.calls.append(("main", video, look, prompt))
        if self.fail == "main":
            raise WorkerError("main failed")
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "main.mp4"
        shutil.copy2(video, out)
        return self.result(out, "main")

    def generate_reverse(self, guide, matched_look, prefix_dir, main_prompt, plan, output_dir, cancelled):
        self.calls.append(("reverse", guide, matched_look, prefix_dir, plan))
        if self.fail == "reverse":
            raise WorkerError("reverse failed")
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "reverse.mp4"
        shutil.copy2(guide, out)
        return self.result(out, "reverse")


def approved_standin(client, settings, uploaded):
    cue = synthetic_cue()
    key, merge = "a" * 12, "b" * 12
    sid = f"{key}-{merge}"
    d = settings.standin_dir / key / "merge" / merge
    d.mkdir(parents=True)
    shutil.copy2(settings.cache_dir / uploaded["clip_id"] / "plate.mp4", d / "final.mp4")
    (d / "cue.json").write_text(json.dumps(cue), encoding="utf-8")
    (d / "standin.json").write_text(json.dumps({
        "clip": cue["clip"], "clip_id": uploaded["clip_id"], "make_key": key, "merge_id": merge,
        "warnings": [], "seconds": 0, "timings_s": {},
    }), encoding="utf-8")
    return cue, sid


def enable(client, settings, adapters):
    object.__setattr__(settings, "ltx_overlap_enabled", True)
    client.app.state.generate.adapters = adapters
    return client.app.state.generate


def start(client, cue, uploaded, sid, mode="user", prompt="blue plasma"):
    body = {"cue": cue, "clip_id": uploaded["clip_id"], "standin_id": sid,
            "prompt_mode": mode, "experimental_ltx_overlap": True}
    if mode == "user":
        body["prompt_text"] = prompt
    response = client.post("/api/generate", json=body)
    assert response.status_code == 202, response.text
    return response.json()


def finish(client, job_id):
    task = client.app.state.queue.get(f"{job_id}/generate")
    try:
        task.future.result(timeout=30)
    except Exception:
        pass
    return client.get(f"/api/jobs/{job_id}").json()


def test_end_to_end_uses_reference_edit_main_reverse_assembly_and_result(client, settings, uploaded):
    cue, sid = approved_standin(client, settings, uploaded)
    adapters = FakeAdapters()
    enable(client, settings, adapters)
    job = start(client, cue, uploaded, sid)
    done = finish(client, job["id"])
    assert_fits("job", done)
    assert done["status"] == "done"
    assert [call[0] for call in adapters.calls] == ["edit", "main", "reverse"]
    assert adapters.calls[1][2].name == "edited.png"
    d = settings.jobs_dir / job["id"]
    provenance = json.loads((d / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["prompt_mode"] == "user"
    assert provenance["stages"]["PREPARING"]["look_source_plate_frame"] == cue["appear_frame"]
    assert provenance["stages"]["EDITING_LOOK"]["source_frame"] == cue["appear_frame"]
    assert provenance["stages"]["GENERATING_MAIN"]["edited_look_sha256"] == \
        provenance["stages"]["EDITING_LOOK"]["sha256"]
    assert provenance["stages"]["ASSEMBLING"]["frames"] == cue["frames"]
    assert len(list(read_frames(d / "merge" / "final.mp4"))) == cue["frames"]
    result = client.get(f"/api/jobs/{job['id']}/result").json()
    assert_fits("result", result)
    assert result["videos"]["final"] == f"/api/jobs/{job['id']}/final.mp4"
    assert result["layers"]["effect_on_black"] is None


def test_surprise_and_user_prompts_share_the_same_pipeline(client, settings, uploaded):
    cue, sid = approved_standin(client, settings, uploaded)
    adapters = FakeAdapters()
    enable(client, settings, adapters)
    jobs = [start(client, cue, uploaded, sid, "user", "an icy orb"),
            start(client, cue, uploaded, sid, "surprise")]
    for job in jobs:
        assert finish(client, job["id"])["status"] == "done"
    assert [c[0] for c in adapters.calls] == ["edit", "main", "reverse"] * 2
    modes = [json.loads((settings.jobs_dir / j["id"] / "provenance.json").read_text(encoding="utf-8"))["prompt_mode"]
             for j in jobs]
    assert modes == ["user", "surprise"]


def test_retry_reuses_completed_expensive_stages(client, settings, uploaded):
    cue, sid = approved_standin(client, settings, uploaded)
    adapters = FakeAdapters("reverse")
    enable(client, settings, adapters)
    job = start(client, cue, uploaded, sid)
    assert finish(client, job["id"])["status"] == "failed"
    assert [c[0] for c in adapters.calls] == ["edit", "main", "reverse"]
    adapters.fail = None
    response = client.post(f"/api/generate/{job['id']}/retry", json={})
    assert response.status_code == 202
    assert finish(client, job["id"])["status"] == "done"
    assert [c[0] for c in adapters.calls] == ["edit", "main", "reverse", "reverse"]


def test_each_remote_stage_failure_is_recorded(client, settings, uploaded):
    cue, sid = approved_standin(client, settings, uploaded)
    for stage in ("edit", "main", "reverse"):
        adapters = FakeAdapters(stage)
        enable(client, settings, adapters)
        job = start(client, cue, uploaded, sid, prompt=f"prompt {stage}")
        failed = finish(client, job["id"])
        assert failed["status"] == "failed" and failed["error"]["code"] == "render_failed"
        provenance = json.loads((settings.jobs_dir / job["id"] / "provenance.json").read_text(encoding="utf-8"))
        assert provenance["failure"]["stage"]


def test_feature_flag_off_preserves_absent_live_generate(client, uploaded):
    cue = synthetic_cue()
    body = {"cue": cue, "clip_id": uploaded["clip_id"], "standin_id": "a" * 12 + "-" + "b" * 12,
            "prompt_mode": "user", "prompt_text": "blue orb", "experimental_ltx_overlap": True}
    assert "features" not in client.get("/api/health").json()
    assert_error(client.post("/api/generate", json=body), 404, "not_found")
