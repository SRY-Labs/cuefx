"""Manual render bridge (task 21): export a ComfyUI package for a cue, import the LTX result, then ingest + Merge.

ComfyUI runs on another machine. Export makes the layout and the placed look (Make 1-2, `services.make.pipeline.make_layout`),
fills the quality graph with their file names and the prompt, and zips them with a manifest; the job then waits. The
owner runs the graph by hand and uploads its effect.mp4 (and, optionally, a matte). Import checks the upload against
the manifest and the cue, then runs the CLI's own ingest (`services.make.pipeline.ingest_ltx`) and Merge
(`services.delivery.merge.run_merge`). No maths here.

    <data>/manual/<make_key>/   Make 1-2: layout.mp4, look_placed.png, layout_sheet.png, look_check.png,
                                layout_overlay.mp4; make.json is written last (the folder is complete)
    <data>/jobs/<job_id>/       job.json (the job, contract shape), export.json (cue, clip, manifest), package.zip,
                                upload/ (what the owner brought back), ltx/ (ingest: effect.mp4, matte.mp4,
                                ingest_sheet.png), merge/ (final.mp4, compare.mp4, merge_sheet.png, cue.json)

Jobs are files, so they survive a restart. A job that was blending when the server stopped comes back `failed`
and takes a new upload.
"""
from __future__ import annotations

import json
import logging
import re
import secrets
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO

from services.cue import api
from services.make.comfy_graph import SAVE_FILE, fill_quality_graph, graph_frames
from services.cue.cue import Cue, cue_from_dict, ending_of
from services.make.ingest import MATTE_WARNING
from services.make.layout import LayoutError, count_frames
from services.make.look import LookError
from services.cue.make_key import make_key
from services.delivery.merge import run_merge
from services.make.pipeline import ingest_ltx, make_layout
from services.cue.route import orb_path
from services.analysis.track import TrackError
from services.delivery.video import VideoError, probe
from services.cue.window import effect_window

from services.analysis.clips import CHUNK, ClipStore, safe_name
from services.config import Settings
from services.errors import ApiError, invalid, not_found
from services.delivery.standin import MAKE_STAGE, MERGE_STAGE, checked_cue, cue_landmarks
from services.make.worker import RenderQueue

log = logging.getLogger("cuefx.server")

JOB_RE = re.compile(r"^job-[0-9a-f]{10}$")
LOOK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")  # a look id is a file stem: no folders
STAGES = ("Preparing shot", "Building reference frame", "Rendering (pass 1 of 2)", "Refining (pass 2 of 2)",
          "Blending into your footage", "Done")  # CONTRACT.md section 9
WAIT_STAGE, BLEND_STAGE, DONE_STAGE = 2, 4, 5
FPS_TOLERANCE = 0.01
UPLOADS = ("effect_on_black", "matte")
# published name -> (file in the job folder, download name suffix; None = shown inline)
FILES = {"final_preview.mp4": ("merge/final.mp4", None), "final.mp4": ("merge/final.mp4", "final.mp4"),
         "package.zip": ("package.zip", "comfyui.zip")}
# GET /api/jobs/{id}/<name>: the result's videos and layers -> file in the job folder
RESULT_FILES = {"final.mp4": "merge/final.mp4", "compare.mp4": "merge/compare.mp4",
                "effect_on_black.mp4": "ltx/effect.mp4", "matte.mp4": "ltx/matte.mp4"}
NOT_MEASURED = "not measured yet"  # CONTRACT.md section 10: the face_shift_px placeholder's message


def _write_json(path: Path, doc: dict) -> None:
    part = path.with_suffix(".part")
    part.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    part.replace(path)


def _failed(stage: str) -> ApiError:
    return ApiError("render_failed", api.message("render_failed", stage_name=stage), 500)


def _stage(job: dict, stage: int) -> dict:
    return {**job, "stage": stage, "stage_name": STAGES[stage], "progress": round(stage / DONE_STAGE, 3)}


ENDING_HINTS = {"burst": "At the end the effect bursts in a bright flash.",
                "shatter": "At the end the effect shatters into glowing shards."}


def ending_hint(prompt: str, cue: Cue) -> str:
    """`prompt` with one more line for a burst or shatter ending (v2.6), so LTX is told what the layout shows. An empty
    prompt stays empty (the export warns that it needs one); a fade adds nothing."""
    hint = ENDING_HINTS.get(ending_of(cue))
    return prompt if hint is None or not prompt.strip() else f"{prompt.rstrip()}\n{hint}"


def how_to_run(job_id: str, names: dict[str, str], graph: str, out: dict, prompt: str) -> str:
    """HOW_TO_RUN.txt: three lines for the owner at the ComfyUI machine."""
    check = "type the prompt in the Prompt node (it is empty)" if not prompt.strip() else "check the Prompt node"
    return (f"1. Copy everything inside comfy_input\\ ({names['layout']}, {names['look']}) into ComfyUI\\input\\.\n"
            f"2. Drag {graph} onto ComfyUI, {check}, press Queue.\n"
            f"3. Bring back ComfyUI\\output\\cuefx\\{job_id}_<date>_<time>_<id>\\{SAVE_FILE} ({out['width']}x"
            f"{out['height']}, {out['frames']} frames, {out['fps']:g} fps) and upload it to CueFX job {job_id} as "
            f"effect_on_black; it is kept as data\\jobs\\{job_id}\\upload\\effect_on_black.mp4.\n")


class ManualStore:
    """Manual render jobs under settings.jobs_dir; Make, ingest and Merge run on the render queue (one at a time:
    they use the whole CPU)."""

    def __init__(self, settings: Settings, clips: ClipStore, queue: RenderQueue) -> None:
        self.s, self.clips, self.queue = settings, clips, queue
        self._recover()

    # ----------------------------------------------------------------- jobs on disk

    def _dir(self, job_id: str) -> Path:
        d = self.s.jobs_dir / job_id
        if not JOB_RE.match(job_id) or not (d / "job.json").is_file():
            raise not_found(f"unknown job {job_id!r}")
        return d

    def _read(self, job_id: str, name: str = "job.json") -> dict:
        return json.loads((self.s.jobs_dir / job_id / name).read_text(encoding="utf-8"))

    def _save(self, job: dict) -> dict:
        _write_json(self.s.jobs_dir / job["id"] / "job.json", job)
        return job

    def _recover(self) -> None:
        """A job left `running` by a stopped server failed: say so, and let it take a new upload."""
        if not self.s.jobs_dir.is_dir():
            return
        for d in self.s.jobs_dir.iterdir():
            if JOB_RE.match(d.name) and (d / "job.json").is_file():
                job = self._read(d.name)
                if job["status"] == "running":
                    log.warning("job %s was running when the server stopped: marked failed", d.name)
                    self._save({**job, "status": "failed", "eta_s": None, "error": {
                        "code": "render_failed", "message": api.message("render_failed", stage_name=job["stage_name"])}})

    def list(self) -> list[dict]:
        """Every job, newest first."""
        if not self.s.jobs_dir.is_dir():
            return []
        ids = [d.name for d in self.s.jobs_dir.iterdir() if JOB_RE.match(d.name) and (d / "job.json").is_file()]
        created = {i: self._read(i, "export.json")["created"] for i in ids}
        return [self._read(i) for i in sorted(ids, key=lambda i: (created[i], i), reverse=True)]

    def get(self, job_id: str) -> dict:
        self._dir(job_id)
        return self._read(job_id)

    def file(self, job_id: str, name: str) -> tuple[Path, str | None]:
        """(file, download name or None for inline) of a job; 404 input_invalid if there is none yet."""
        d = self._dir(job_id)
        if name not in FILES:
            raise not_found(f"unknown job file {job_id}/{name}")
        rel, suffix = FILES[name]
        path = d / rel
        if not path.is_file() or (rel.startswith("merge/") and self._read(job_id)["status"] != "done"):
            raise not_found(f"job {job_id} has no {name} yet")
        if suffix is None:
            return path, None
        clip = self._read(job_id, "export.json")["clip"]
        return path, safe_name(f"{Path(clip).stem}_{job_id}_{suffix}")

    def result_file(self, job_id: str, name: str) -> tuple[Path, str]:
        """(file, download name) of a done job's video or layer (RESULT_FILES); 404 not_found if there is none."""
        d = self._dir(job_id)
        if name not in RESULT_FILES:
            raise not_found(f"unknown job file {job_id}/{name}")
        path = d / RESULT_FILES[name]
        if not path.is_file() or self._read(job_id)["status"] != "done":
            raise not_found(f"job {job_id} has no {name} yet")
        return path, safe_name(f"{Path(self._read(job_id, 'export.json')['clip']).stem}_{job_id}_{name}")

    def result(self, job_id: str) -> dict:
        """The result.schema.json object of a done job, from the files the job already has and the checks the steps
        already made; 409 input_invalid until the job is done. Nothing is measured here: a check nobody computes is
        `warn` with a null value and the message "not measured yet" (the schema's own way, CONTRACT.md section 10)."""
        d = self._dir(job_id)
        job, exp = self._read(job_id), self._read(job_id, "export.json")
        if job["status"] != "done":
            raise invalid(f"job {job_id} is not done yet (it is {job['status']}); its result is ready when it is done", 409)
        if exp.get("kind") == "ltx_overlap":
            not_measured = {"status": "warn", "value": None, "message": NOT_MEASURED}
            checks = [
                {"id": "plate_unchanged", "label": "Footage unchanged outside the generated window",
                 "unit": "8-bit levels", "frames": None, **not_measured},
                {"id": "black_clamp", "label": "Black level of the effect", "unit": "linear",
                 "frames": None, **not_measured},
                {"id": "matte_source", "label": "Result mode", "unit": None, "frames": None,
                 "status": "pass", "value": "full_scene",
                 "message": "The assembled full-scene LTX result is used directly; it is not treated as an effect layer."},
                {"id": "face_shift_px", "label": "Face moved", "unit": "px", "frames": None, **not_measured},
            ]
            warnings = [x for x in api.validate(exp["cue"]) if x["severity"] != "error"]
            warnings += [{"id": f"render_note_{i}", "severity": "warning", "code": "render_note",
                          "message": note, "frames": None, "jump": None, "cause": {"kind": "render"}}
                         for i, note in enumerate(job.get("warnings", []), 1)]
            return {"job_id": job_id, "look": job["look"], "pipeline": "composite",
                    "videos": {"final": f"/api/jobs/{job_id}/final.mp4", "compare": None},
                    "layers": {"effect_on_black": None, "matte": None, "occluder_matte": None},
                    "checks": checks, "warnings": warnings}
        cue = json.loads((d / "merge" / "cue.json").read_text(encoding="utf-8"))
        w = effect_window(cue_from_dict(cue))
        window = [w.start, w.end]

        def url(name: str) -> str | None:  # only for files that exist
            return f"/api/jobs/{job_id}/{name}" if (d / RESULT_FILES[name]).is_file() else None

        temporary = not (d / "upload" / "matte.mp4").is_file()
        not_measured = {"status": "warn", "value": None, "message": NOT_MEASURED}
        checks = [
            {"id": "plate_unchanged", "label": "Footage untouched outside the effect", "unit": "8-bit levels",
             "frames": None, **not_measured},
            {"id": "black_clamp", "label": "Black level of the effect", "unit": "linear", "frames": window,
             **not_measured},
            {"id": "matte_source", "label": "Matte", "unit": None, "frames": window,
             **({"status": "warn", "value": "temporary", "message": MATTE_WARNING} if temporary else
                {"status": "pass", "value": "uploaded", "message": "The matte you uploaded was used."})},
            {"id": "face_shift_px", "label": "Face moved", "unit": "px", "frames": None, **not_measured},
        ]
        notes = [n for n in job.get("warnings", []) if n != MATTE_WARNING]  # the matte_source check says that one
        warnings = [x for x in api.validate(cue) if x["severity"] != "error"]
        warnings += [{"id": f"render_note_{i}", "severity": "warning", "code": "render_note", "message": n,
                      "frames": None, "jump": None, "cause": {"kind": "render"}} for i, n in enumerate(notes, 1)]
        return {"job_id": job_id, "look": job["look"], "pipeline": "composite",
                "videos": {"final": url("final.mp4"), "compare": url("compare.mp4")},
                "layers": {"effect_on_black": url("effect_on_black.mp4"), "matte": url("matte.mp4"),
                           "occluder_matte": None},
                "checks": checks, "warnings": warnings}

    # ----------------------------------------------------------------- export

    def looks(self) -> list[dict]:
        """[{id, name, file}] of the look images in CUEFX_LOOKS_DIR, sorted by id: one per `<id>.png` whose id the
        export accepts (a file with another name could never be used, so it is not offered)."""
        d = self.s.looks
        found = []
        for p in d.iterdir() if d.is_dir() else ():
            if p.suffix != ".png" or not p.is_file():
                continue
            if not LOOK_RE.match(p.stem):
                log.warning("look image %s is not listed: its name is not a look id", p)
                continue
            found.append({"id": p.stem, "name": p.stem.replace("_", " ").title(), "file": p.name})
        return sorted(found, key=lambda look: look["id"])

    def _look(self, look: str) -> Path:
        """The look image of a look id: CUEFX_LOOKS_DIR/<look>.png. 400 input_invalid if it is not there; the
        message lists the ids that are."""
        if not LOOK_RE.match(look):
            raise invalid(f"look {look!r} is not a look id (letters, digits, '_', '-', '.'; no folders)")
        p = self.s.looks / f"{look}.png"
        if not p.is_file():
            ids = [x["id"] for x in self.looks()]
            valid = f"Valid look ids: {', '.join(ids)}." if ids else "There are no look images there yet."
            raise invalid(f"there is no look image for look {look!r}: put {look}.png (the look on pure black) in "
                          f"{self.s.looks} (CUEFX_LOOKS_DIR). {valid}")
        return p

    def _make(self, plate: Path, c: Cue, look: Path, key: str) -> tuple[Path, list[str]]:
        """Make 1-2 (layout + placed look) into <manual>/<key>/, or the folder already there for that key."""
        make_dir = self.s.manual_dir / key
        done = make_dir / "make.json"
        if done.is_file():
            return make_dir, json.loads(done.read_text(encoding="utf-8"))["warnings"]
        make_dir.mkdir(parents=True, exist_ok=True)
        try:
            run = make_layout(plate, c, look, make_dir)
        except (LayoutError, LookError, TrackError) as e:
            log.error("manual Make failed in %s", make_dir, exc_info=e)
            raise _failed(MAKE_STAGE) from e
        _write_json(done, {"warnings": run.warnings, "look_frame": run.look_frame, "frames": run.frames})
        return make_dir, run.warnings

    def export(self, cue: dict, clip_id: str, look: str, prompt: str | None,
               wait: bool = True) -> tuple[dict, Path | None]:
        """A new job for `cue` with `look`. Make 1-2 and its package.zip (comfy_input/ with the layout and the placed
        look, named as the graph loads them, the filled graph, manifest.json and HOW_TO_RUN.txt) run on the render
        queue; the job is `running` ("Preparing shot") until the zip is ready, then `waiting`.

        wait: (the `waiting` job, package.zip). Not wait: (the `running` job, None) at once; poll GET /api/jobs/{id}.
        400 input_invalid for cue errors, a wrong clip or a missing look image (no job is made); 500 render_failed
        if Make fails (the job stays, `failed`)."""
        cue, c, clip_id, plate = checked_cue(self.clips, cue, None, clip_id)
        look_path = self._look(look)
        c = replace(c, look=look)  # one job per look (generate's shape): the job's look is in its make key
        cue = {**cue, "look": look}
        prompt = ending_hint(prompt if prompt is not None else (cue.get("prompt") or ""), c)
        t0 = time.perf_counter()
        key = make_key(c, look_path)
        job_id = f"job-{secrets.token_hex(5)}"
        d = self.s.jobs_dir / job_id
        d.mkdir(parents=True)
        # export.json before job.json: a job is listed (by its job.json) only once it can be sorted by `created`
        _write_json(d / "export.json", {"created": time.time(), "clip": c.clip, "clip_id": clip_id, "look": look,
                                        "cue": cue})
        job = self._save(_stage({"id": job_id, "look": look, "status": "running", "stage": 0, "stage_name": "",
                                 "progress": 0, "elapsed_s": 0, "eta_s": None, "error": None, "manual": True,
                                 "clip_id": clip_id, "make_key": key, "files": {}, "warnings": []}, 0))
        task = self.queue.submit(f"{job_id}/export", lambda: self._export(job, plate, c, look_path, prompt, t0))
        if not wait:
            return job, None
        return task.wait(), d / "package.zip"

    def _export(self, job: dict, plate: Path, c: Cue, look_path: Path, prompt: str, t0: float) -> dict:
        """Make 1-2 and the package of an exported job, on the render queue: the job saved `waiting`, or `failed`
        (and the error raised) if a step fails."""
        try:
            return self._build_package(job, plate, c, look_path, prompt, t0)
        except Exception as e:  # ApiError from Make, or anything unexpected: the job must not stay `running`
            log.error("manual job %s: export failed", job["id"], exc_info=e)
            self._save({**job, "status": "failed", "elapsed_s": round(time.perf_counter() - t0, 2),
                        "error": {"code": "render_failed", "message": api.message("render_failed",
                                                                                  stage_name=MAKE_STAGE)}})
            raise

    def _build_package(self, job: dict, plate: Path, c: Cue, look_path: Path, prompt: str, t0: float) -> dict:
        job_id, key, look, clip_id = job["id"], job["make_key"], job["look"], job["clip_id"]
        d = self.s.jobs_dir / job_id
        make_dir, warnings = self._make(plate, c, look_path, key)
        layout = make_dir / "layout.mp4"
        info = probe(layout)
        out = {"file": SAVE_FILE, "width": info.width, "height": info.height, "frames": count_frames(layout),
               "fps": info.fps}
        w = effect_window(c)
        if out["frames"] != w.length:
            warnings = warnings + [f"layout.mp4 decodes to {out['frames']} frames, the effect window has {w.length}"]
        if not prompt.strip():
            warnings = warnings + ["no prompt: type one in the graph's Prompt node before you queue it"]

        names = {"layout": f"{job_id}_layout.mp4", "look": f"{job_id}_look.png"}
        graph_name = f"CueFX_Phase1_Quality_{job_id}.json"
        try:
            generic = json.loads(self.s.quality_graph.read_text(encoding="utf-8"))
            graph = fill_quality_graph(generic, names, prompt, f"cuefx/{job_id}", w.length)
        except (OSError, ValueError) as e:  # the graph file is part of the install: not the artist's input
            log.error("cannot fill the quality graph %s", self.s.quality_graph, exc_info=e)
            raise _failed(MAKE_STAGE) from e
        manifest = {
            "job_id": job_id, "make_key": key, "clip_id": clip_id, "clip": c.clip, "look": look, "prompt": prompt,
            "effect_range": {"start": w.start, "end": w.end}, "graph_frames": graph_frames(w.length),
            "comfy_input": names, "graph": graph_name,
            "output": {**out, "comfy_folder": f"ComfyUI/output/cuefx/{job_id}_<UTC date>_<UTC time>_<6 hex>/"},
            "upload": {"effect_on_black": f"required: that folder's {SAVE_FILE}",
                       "matte": "optional: same size, frames and fps; without it Merge uses the temporary "
                                "luminance matte"},
            "warnings": warnings,
        }
        package = d / "package.zip"
        part = d / "package.part"
        with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(layout, f"comfy_input/{names['layout']}", compress_type=zipfile.ZIP_STORED)
            z.write(make_dir / "look_placed.png", f"comfy_input/{names['look']}", compress_type=zipfile.ZIP_STORED)
            z.writestr(graph_name, json.dumps(graph, indent=1, ensure_ascii=False))
            z.writestr("manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False))
            z.writestr("HOW_TO_RUN.txt", how_to_run(job_id, names, graph_name, out, prompt))
        part.replace(package)
        _write_json(d / "export.json", {**self._read(job_id, "export.json"), "manifest": manifest})
        waiting = self._save(_stage({**job, "status": "waiting", "elapsed_s": round(time.perf_counter() - t0, 2),
                                     "files": {"package_download": f"/api/manual/{job_id}/package.zip"},
                                     "warnings": warnings}, WAIT_STAGE))
        log.info("manual job %s exported (make key %s) in %.1f s", job_id, key, waiting["elapsed_s"])
        return waiting

    # ----------------------------------------------------------------- import

    def _receive(self, source: BinaryIO, path: Path) -> None:
        """Copy an upload to `path`, at most CUEFX_MAX_UPLOAD_MB (413 beyond)."""
        size = 0
        with path.open("wb") as f:
            while chunk := source.read(CHUNK):
                size += len(chunk)
                if size > self.s.max_upload_bytes:
                    f.close()
                    path.unlink()
                    raise invalid(f"the file is larger than {self.s.max_upload_mb} MB", 413)
                f.write(chunk)
        if size == 0:
            path.unlink()
            raise invalid(f"{path.stem.removeprefix('incoming_')} is empty")

    @staticmethod
    def _check_video(path: Path, name: str, out: dict, job_id: str) -> None:
        """400 input_invalid unless `path` has the size, frame count and fps the manifest expects."""
        try:
            info = probe(path)
        except VideoError:
            raise invalid(f"{name} is not a video CueFX can read") from None
        n = count_frames(path)
        problems = []
        if (info.width, info.height) != (out["width"], out["height"]):
            problems.append(f"is {info.width}x{info.height}, the export expects {out['width']}x{out['height']}")
        if n != out["frames"]:
            problems.append(f"has {n} frames, the export expects {out['frames']}")
        if abs(info.fps - out["fps"]) > FPS_TOLERANCE:
            problems.append(f"is {info.fps:g} fps, the export expects {out['fps']:g}")
        if problems:
            raise invalid(f"{name} " + "; ".join(problems) + f" (manifest.json of {job_id}). Upload the "
                          f"{SAVE_FILE} that this job's graph wrote, unchanged")

    def _merge_cue(self, job: dict, exp: dict, cue: dict | None) -> tuple[dict, Cue, Path]:
        """The cue Merge uses (the one sent with the upload, else the exported one), its Cue and the plate.
        400 input_invalid if its make key is no longer the job's: the cue's Make fields or the look image changed."""
        look_path = self._look(job["look"])
        if make_key(cue_from_dict(exp["cue"]), look_path) != job["make_key"]:
            raise invalid(f"the look image {look_path.name} changed since export; export again")
        if cue is None:
            cue = exp["cue"]
        else:
            api.check_schema("cue", cue)
            cue = {**cue, "look": job["look"]}
        cue, c, _, plate = checked_cue(self.clips, cue, None, job["clip_id"])
        if make_key(c, look_path) != job["make_key"]:
            raise invalid("the cue changed since export (its layout would differ); export again")
        return cue, c, plate

    def import_result(self, job_id: str, effect: BinaryIO, matte: BinaryIO | None, cue: dict | None,
                      wait: bool = True) -> dict:
        """Check the LTX result against the job's manifest and cue (here, at once), then ingest + Merge on the
        render queue: the job ends `done` with final.mp4, or `failed` (render_failed) if a step fails. A waiting,
        done or failed job takes a new upload; one still blending is 409 input_invalid.

        wait: the finished job (500 render_failed if a step failed). Not wait: the `running` job at once; poll
        GET /api/jobs/{job_id}."""
        d = self._dir(job_id)
        task_id = f"{job_id}/import"
        with self.clips.lock(f"job-{job_id}"):  # one upload of a job at a time; never held during ingest + Merge
            if self.queue.busy(task_id):
                raise invalid(f"job {job_id} is still blending the previous upload; wait for it to finish", 409)
            job, exp = self._read(job_id), self._read(job_id, "export.json")
            if "manifest" not in exp:  # the export is still building its package, or it failed
                raise invalid(f"job {job_id} has no package yet (its export is not finished, or it failed); "
                              f"wait for it, or export again", 409)
            out = exp["manifest"]["output"]
            up = d / "upload"
            up.mkdir(exist_ok=True)
            given = {"effect_on_black": effect, "matte": matte}
            incoming = {}
            try:
                for name in UPLOADS:
                    if given[name] is not None:
                        incoming[name] = up / f"incoming_{name}.mp4"
                        self._receive(given[name], incoming[name])
                        self._check_video(incoming[name], name, out, job_id)
                cue, c, plate = self._merge_cue(job, exp, cue)
            except BaseException:
                for p in incoming.values():
                    p.unlink(missing_ok=True)
                raise
            for name in UPLOADS:  # the upload is accepted: it replaces the previous one
                (up / f"{name}.mp4").unlink(missing_ok=True)
                if name in incoming:
                    incoming[name].replace(up / f"{name}.mp4")
            matte_path = up / "matte.mp4" if "matte" in incoming else None
            job = self._save(_stage({**job, "status": "running", "error": None, "eta_s": None}, BLEND_STAGE))
            task = self.queue.submit(task_id, lambda: self._blend(job, exp, cue, c, plate, matte_path))
        return task.wait() if wait else job

    def _blend(self, job: dict, exp: dict, cue: dict, c: Cue, plate: Path, matte_path: Path | None) -> dict:
        """Ingest + Merge of an accepted upload, on the render queue; the job saved `done`, or `failed` (and the
        error raised) if a step fails."""
        job_id = job["id"]
        d = self.s.jobs_dir / job_id
        t0 = time.perf_counter()
        files = {"package_download": f"/api/manual/{job_id}/package.zip"}
        try:
            res = ingest_ltx(c, d / "upload" / "effect_on_black.mp4", d / "ltx", matte_path)
            landmarks, warnings = cue_landmarks(self.clips, job["clip_id"], c)
            rep = run_merge(plate, c, orb_path(c), d / "ltx", d / "merge", landmarks, [], export_layers=False)
            _write_json(d / "merge" / "cue.json", cue)
        except Exception as e:  # LayoutError/VideoError/IngestError, or anything unexpected: the job must not hang
            log.error("manual job %s: ingest/Merge failed", job_id, exc_info=e)
            self._save({**job, "status": "failed", "files": files,
                        "elapsed_s": round(job["elapsed_s"] + time.perf_counter() - t0, 2),
                        "error": {"code": "render_failed", "message": api.message("render_failed",
                                                                                  stage_name=MERGE_STAGE)}})
            if isinstance(e, LayoutError):  # VideoError is a LayoutError; IngestError a VideoError
                raise _failed(MERGE_STAGE) from e
            raise
        warnings = list(dict.fromkeys(exp["manifest"]["warnings"] + res.warnings + warnings + rep.warnings))
        files.update(final_preview=f"/api/manual/{job_id}/final_preview.mp4",
                     final_download=f"/api/manual/{job_id}/final.mp4")
        took = time.perf_counter() - t0
        log.info("manual job %s: ingest + Merge in %.1f s", job_id, took)
        return self._save(_stage({**job, "status": "done", "eta_s": 0, "files": files, "warnings": warnings,
                                  "elapsed_s": round(job["elapsed_s"] + took, 2)}, DONE_STAGE))
