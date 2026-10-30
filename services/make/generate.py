"""Experimental one-click image-edit + main/reverse LTX orchestration."""
from __future__ import annotations

import hashlib
import json
import logging
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import cv2

from services.cue.cue import cue_from_dict
from services.delivery.video import VideoWriter, add_audio, probe, read_all
from services.analysis.clips import ClipStore
from services.make.comfy_worker import ComfyAdapters, ComfyWorker, WorkerError, WorkerResult
from services.config import Settings
from services.errors import ApiError, invalid, not_found
from services.make.ltx_overlap import (assemble_full_scene, plan_overlap, prepare_full_scene_reference_record,
                                prepare_reverse_entrance)
from services.make.ltx_overlap_graph import fill_main_api
from services.delivery.manual import DONE_STAGE, JOB_RE, ManualStore, STAGES
from services.delivery.standin import StandinStore, checked_cue
from services.make.worker import RenderQueue

log = logging.getLogger("cuefx.server")

SURPRISE_PROMPTS = (
    "Transform the existing stand-in sphere into a vivid electric-blue plasma orb with a bright layered core, fine surface currents, restrained sparks, and believable blue light on the nearby hand and clothing. Preserve the person, room, pose, framing, orb center, and orb diameter.",
    "Transform the existing stand-in sphere into a dense amber fire orb with a white-hot center, curling flame filaments, a few detached embers, and warm interactive light on the nearby hand and clothing. Preserve the person, room, pose, framing, orb center, and orb diameter.",
    "Transform the existing stand-in sphere into a crystalline violet energy orb with a luminous inner lattice, thin irregular arcs, subtle particles, and restrained violet light on the nearby hand and clothing. Preserve the person, room, pose, framing, orb center, and orb diameter.",
)

STATE_STAGE = {
    "QUEUED": 0, "PREPARING": 0, "EDITING_LOOK": 1, "GENERATING_MAIN": 2,
    "PREPARING_ENTRANCE": 3, "GENERATING_ENTRANCE": 3,
    "ASSEMBLING": 4, "INGESTING": 4, "COMPLETED": 5,
}


def _write_json(path: Path, doc: dict) -> None:
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    part.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _remote_record(job_dir: Path, result: WorkerResult) -> dict:
    return {"path": str(result.path.relative_to(job_dir)), "sha256": result.sha256,
            "worker_job_id": result.prompt_id, "worker_output": result.output}


class GenerateStore:
    """Generated jobs use the existing on-disk job/result contract and queue."""

    def __init__(self, settings: Settings, clips: ClipStore, standins: StandinStore,
                 manual: ManualStore, queue: RenderQueue, adapters: Any | None = None) -> None:
        self.s, self.clips, self.standins, self.manual, self.queue = settings, clips, standins, manual, queue
        self.adapters = adapters
        if adapters is None and settings.ltx_overlap_enabled and settings.comfy_url:
            self.adapters = ComfyAdapters(ComfyWorker(settings.comfy_url, settings.comfy_token,
                                                       settings.comfy_timeout_s))
        self._cancelled: set[str] = set()
        self._guard = threading.Lock()

    @property
    def available(self) -> bool:
        return self.s.ltx_overlap_enabled and self.adapters is not None

    def _save_job(self, job: dict, state: str | None = None) -> dict:
        if state is not None:
            stage = STATE_STAGE[state]
            job = {**job, "stage": stage, "stage_name": STAGES[stage],
                   "progress": round(stage / DONE_STAGE, 3)}
        _write_json(self.s.jobs_dir / job["id"] / "job.json", job)
        return job

    def _job(self, job_id: str) -> tuple[Path, dict, dict]:
        if not JOB_RE.match(job_id):
            raise not_found(f"unknown job {job_id!r}")
        d = self.s.jobs_dir / job_id
        if not (d / "job.json").is_file() or not (d / "export.json").is_file():
            raise not_found(f"unknown job {job_id!r}")
        job = json.loads((d / "job.json").read_text(encoding="utf-8"))
        exp = json.loads((d / "export.json").read_text(encoding="utf-8"))
        if exp.get("kind") != "ltx_overlap":
            raise invalid(f"job {job_id} is not an automatic LTX job", 409)
        return d, job, exp

    @staticmethod
    def _prompt(mode: str, text: str | None, cue: dict) -> str:
        if mode == "user":
            prompt = (text or cue.get("prompt") or "").strip()
            if not prompt:
                raise invalid("prompt_text is required unless prompt_mode is surprise")
            return prompt
        if mode != "surprise":
            raise invalid("prompt_mode must be user or surprise")
        identity = json.dumps([cue, cue.get("seed", 0)], sort_keys=True).encode()
        return SURPRISE_PROMPTS[int(hashlib.sha256(identity).hexdigest()[:8], 16) % len(SURPRISE_PROMPTS)]

    def start(self, cue: dict, clip_id: str, standin_id: str, prompt_mode: str,
              prompt_text: str | None, look: str | None = None) -> dict:
        if not self.s.ltx_overlap_enabled:
            raise not_found("experimental LTX overlap generation is disabled")
        if self.adapters is None:
            raise invalid("experimental LTX overlap is enabled but CUEFX_COMFY_URL is not configured", 503)
        cue, c, clip_id, _ = checked_cue(self.clips, cue, None, clip_id)
        status = self.standins.status(standin_id)
        if status["status"] != "done" or status["result"]["clip_id"] != clip_id:
            raise invalid("the approved hard composite is not ready for this clip")
        _, _, merge_dir = self.standins._merge_dir(standin_id)
        approved_cue = json.loads((merge_dir / "cue.json").read_text(encoding="utf-8"))
        if approved_cue != cue:
            raise invalid("the cue changed after the hard composite was approved; render and approve it again")
        prompt = self._prompt(prompt_mode, prompt_text, cue)
        plan = plan_overlap(c)
        if plan is None:
            raise invalid("this experimental path requires an active entrance after frame zero")
        job_id = f"job-{secrets.token_hex(5)}"
        d = self.s.jobs_dir / job_id
        d.mkdir(parents=True)
        request_hash = hashlib.sha256(json.dumps(
            {"cue": cue, "clip_id": clip_id, "standin_id": standin_id, "prompt_mode": prompt_mode,
             "prompt": prompt, "look": look}, sort_keys=True).encode()).hexdigest()
        exp = {"kind": "ltx_overlap", "created": time.time(), "clip": c.clip, "clip_id": clip_id,
               "cue": cue, "standin_id": standin_id, "look": look or cue.get("look") or "generated",
               "prompt_mode": prompt_mode, "prompt": prompt, "request_sha256": request_hash}
        _write_json(d / "export.json", exp)
        _write_json(d / "provenance.json", {"request_sha256": request_hash, "prompt": prompt,
                                             "prompt_mode": prompt_mode, "plan": plan.as_dict(),
                                             "completed_stage": None, "stages": {}})
        job = self._save_job({"id": job_id, "look": exp["look"], "status": "queued", "stage": 0,
                              "stage_name": STAGES[0], "progress": 0, "elapsed_s": 0, "eta_s": None,
                              "error": None, "manual": False, "clip_id": clip_id,
                              "make_key": status["result"]["make_key"], "files": {}, "warnings": []}, "QUEUED")
        self.queue.submit(f"{job_id}/generate", lambda: self._run(job_id))
        return job

    def retry(self, job_id: str) -> dict:
        _, job, _ = self._job(job_id)
        if job["status"] not in ("failed", "cancelled"):
            raise invalid(f"job {job_id} cannot be retried while it is {job['status']}", 409)
        with self._guard:
            self._cancelled.discard(job_id)
        job = self._save_job({**job, "status": "queued", "error": None, "eta_s": None}, "QUEUED")
        self.queue.submit(f"{job_id}/generate", lambda: self._run(job_id))
        return job

    def cancel(self, job_id: str) -> dict:
        _, job, _ = self._job(job_id)
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        with self._guard:
            self._cancelled.add(job_id)
        self.queue.cancel(f"{job_id}/generate")
        worker = getattr(self.adapters, "worker", None)
        if worker is not None:
            worker.interrupt()
        return self._save_job({**job, "status": "cancelled", "eta_s": None,
                               "error": {"code": "cancelled", "message": "The render was cancelled."}})

    def _is_cancelled(self, job_id: str) -> bool:
        with self._guard:
            return job_id in self._cancelled

    def _check_cancelled(self, job_id: str) -> None:
        if self._is_cancelled(job_id):
            raise WorkerError("cancelled")

    @staticmethod
    def _valid_artifact(d: Path, record: dict | None) -> Path | None:
        if not record or "path" not in record or "sha256" not in record:
            return None
        path = d / record["path"]
        return path if path.is_file() and _sha256(path) == record["sha256"] else None

    def _record_stage(self, d: Path, provenance: dict, name: str, record: dict) -> dict:
        provenance = {**provenance, "completed_stage": name,
                      "stages": {**provenance.get("stages", {}), name: record}}
        _write_json(d / "provenance.json", provenance)
        return provenance

    @staticmethod
    def _write_video(path: Path, frames: list, size: tuple[int, int], fps: float) -> None:
        with VideoWriter(path, size, fps, crf=18) as writer:
            for frame in frames:
                writer.write(frame)

    @staticmethod
    def _fit(frame, size: tuple[int, int]):
        return frame if (frame.shape[1], frame.shape[0]) == size else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    def _run(self, job_id: str) -> dict:
        d, job, exp = self._job(job_id)
        t0 = time.perf_counter()
        provenance = json.loads((d / "provenance.json").read_text(encoding="utf-8"))
        try:
            cue = cue_from_dict(exp["cue"])
            plan = plan_overlap(cue)
            if plan is None:
                raise ValueError("overlap plan became inactive")
            source, _ = self.standins.file(exp["standin_id"], "hardcomp.mp4")
            assert source is not None
            self._check_cancelled(job_id)
            job = self._save_job({**job, "status": "running", "error": None}, "PREPARING")
            prep = d / "preparation"
            prep.mkdir(exist_ok=True)
            reference = prep / "reference_R.png"
            reference_record = provenance.get("stages", {}).get("PREPARING")
            if self._valid_artifact(d, reference_record) is None:
                record = prepare_full_scene_reference_record(cue, source, reference)
                record.update(path=str(reference.relative_to(d)), sha256=_sha256(reference),
                              input_video_sha256=_sha256(source))
                provenance = self._record_stage(d, provenance, "PREPARING", record)
            else:
                reference = d / reference_record["path"]

            self._check_cancelled(job_id)
            job = self._save_job(job, "EDITING_LOOK")
            look_record = provenance.get("stages", {}).get("EDITING_LOOK")
            edited = self._valid_artifact(d, look_record)
            if edited is None:
                result = self.adapters.edit_look(reference, exp["prompt"], cue.seed, d / "image_edit",
                                                 lambda: self._is_cancelled(job_id))
                edited = result.path
                if cv2.imread(str(edited), cv2.IMREAD_COLOR) is None:
                    raise ValueError("image-edit output is not a readable image")
                look_record = _remote_record(d, result)
                look_record.update(source_frame=plan.look_source_plate_frame,
                                   source_frame_sha256=provenance["stages"]["PREPARING"]["sha256"])
                provenance = self._record_stage(d, provenance, "EDITING_LOOK", look_record)

            self._check_cancelled(job_id)
            source_frames = read_all(source)
            if len(source_frames) != cue.frames:
                raise ValueError(f"approved hard composite has {len(source_frames)} frames; expected {cue.frames}")
            main_input = prep / "main_input.mp4"
            main_frames = list(source_frames[plan.look_source_plate_frame:plan.window_end])
            main_frames += [main_frames[-1]] * (plan.main_padded_count - len(main_frames))
            if not main_input.is_file():
                self._write_video(main_input, main_frames, cue.plate_size, cue.fps)
            job = self._save_job(job, "GENERATING_MAIN")
            main_record = provenance.get("stages", {}).get("GENERATING_MAIN")
            main_output = self._valid_artifact(d, main_record)
            if main_output is None:
                result = self.adapters.generate_main(main_input, edited, exp["prompt"], d / "main",
                                                     lambda: self._is_cancelled(job_id))
                main_output = result.path
                main_record = _remote_record(d, result)
                main_record.update(input_path=str(main_input.relative_to(d)), input_sha256=_sha256(main_input),
                                   edited_look_sha256=_sha256(edited))
                provenance = self._record_stage(d, provenance, "GENERATING_MAIN", main_record)
            main_generated = read_all(main_output)
            if len(main_generated) < plan.main_real_count:
                raise ValueError("main LTX output is shorter than its real interval")

            self._check_cancelled(job_id)
            job = self._save_job(job, "PREPARING_ENTRANCE")
            reverse_dir = d / "reverse_preparation"
            reverse_dir.mkdir(exist_ok=True)
            prepared = prepare_reverse_entrance(source_frames[plan.window_start:plan.window_end],
                                                main_generated, plan)
            guide = reverse_dir / "reverse_guide.mp4"
            prefix = reverse_dir / "generated_prefix"
            matched = reverse_dir / "generated_anchor.png"
            prefix.mkdir(exist_ok=True)
            if not guide.is_file():
                self._write_video(guide, list(prepared.guide_frames), cue.plate_size, cue.fps)
            if not matched.is_file() and not cv2.imwrite(str(matched), prepared.matched_look):
                raise OSError(f"could not write {matched}")
            for index, frame in enumerate(prepared.fixed_generated_prefix):
                path = prefix / f"frame_{index:03d}.png"
                if not path.is_file() and not cv2.imwrite(str(path), frame):
                    raise OSError(f"could not write {path}")
            prep_record = {"path": str(guide.relative_to(d)), "sha256": _sha256(guide),
                           "matched_look": str(matched.relative_to(d)), "matched_look_sha256": _sha256(matched),
                           "prefix_dir": str(prefix.relative_to(d)), "prefix_frames": plan.context_count,
                           "guide_plate_frames": list(prepared.guide_plate_frames),
                           "prefix_plate_frames": list(prepared.prefix_plate_frames)}
            provenance = self._record_stage(d, provenance, "PREPARING_ENTRANCE", prep_record)

            self._check_cancelled(job_id)
            job = self._save_job(job, "GENERATING_ENTRANCE")
            reverse_record = provenance.get("stages", {}).get("GENERATING_ENTRANCE")
            reverse_output = self._valid_artifact(d, reverse_record)
            if reverse_output is None:
                main_prompt = fill_main_api("main.mp4", "look.png", exp["prompt"], "cuefx/main")
                result = self.adapters.generate_reverse(guide, matched, prefix, main_prompt, plan, d / "reverse",
                                                        lambda: self._is_cancelled(job_id))
                reverse_output = result.path
                reverse_record = _remote_record(d, result)
                reverse_record.update(main_result_sha256=_sha256(main_output), guide_sha256=_sha256(guide))
                provenance = self._record_stage(d, provenance, "GENERATING_ENTRANCE", reverse_record)
            reverse_generated = read_all(reverse_output)
            if len(reverse_generated) < plan.reverse_real_count:
                raise ValueError("reverse LTX output is shorter than its real interval")

            self._check_cancelled(job_id)
            job = self._save_job(job, "ASSEMBLING")
            main_fitted = [self._fit(frame, cue.plate_size) for frame in main_generated]
            reverse_fitted = [self._fit(frame, cue.plate_size) for frame in reverse_generated]
            final_frames = assemble_full_scene(source_frames, main_fitted, reverse_fitted, plan)
            merge_dir = d / "merge"
            merge_dir.mkdir(exist_ok=True)
            final = merge_dir / "final.mp4"
            if probe(source).has_audio:
                video_only = d / "assembly" / "video_only.mp4"
                self._write_video(video_only, final_frames, cue.plate_size, cue.fps)
                warnings = add_audio(video_only, source, final)
            else:
                self._write_video(final, final_frames, cue.plate_size, cue.fps)
                warnings = []
            _write_json(merge_dir / "cue.json", exp["cue"])
            assembly = {"path": str(final.relative_to(d)), "sha256": _sha256(final),
                        "frames": len(final_frames), "fps": cue.fps, "window_ownership": {
                            "clean": [plan.window_start, plan.zero_frame + 1],
                            "reverse": [plan.zero_frame + 1, plan.look_source_plate_frame],
                            "main": [plan.look_source_plate_frame, plan.window_end]}}
            provenance = self._record_stage(d, provenance, "ASSEMBLING", assembly)
            job = self._save_job(job, "INGESTING")
            files = {"final_preview": f"/api/manual/{job_id}/final_preview.mp4",
                     "final_download": f"/api/manual/{job_id}/final.mp4"}
            provenance = self._record_stage(d, provenance, "INGESTING", {
                "result_contract": f"/api/jobs/{job_id}/result", "full_scene": True})
            took = round(time.perf_counter() - t0, 2)
            job = self._save_job({**job, "status": "done", "eta_s": 0, "files": files,
                                  "warnings": list(dict.fromkeys(job.get("warnings", []) + warnings)),
                                  "elapsed_s": round(job.get("elapsed_s", 0) + took, 2), "error": None},
                                 "COMPLETED")
            self._record_stage(d, provenance, "COMPLETED", {"final_sha256": _sha256(final)})
            return job
        except Exception as e:
            d, current, _ = self._job(job_id)
            if self._is_cancelled(job_id) or isinstance(e, WorkerError) and str(e) == "cancelled":
                return self._save_job({**current, "status": "cancelled", "eta_s": None,
                                       "error": {"code": "cancelled", "message": "The render was cancelled."}})
            log.error("automatic LTX job %s failed", job_id, exc_info=e)
            provenance["failure"] = {"stage": current["stage_name"], "message": str(e)}
            _write_json(d / "provenance.json", provenance)
            self._save_job({**current, "status": "failed", "eta_s": None,
                            "elapsed_s": round(current.get("elapsed_s", 0) + time.perf_counter() - t0, 2),
                            "error": {"code": "render_failed", "message":
                                      f"{current['stage_name']} failed. Retry keeps completed stages."}})
            if isinstance(e, ApiError):
                raise
            raise
