"""The FastAPI app: routes under /api that only call cuefx.api and the clip store (contract/HTTP.md)."""
from __future__ import annotations

import json
import logging
from functools import cache
from typing import Any, Literal

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

import uvicorn

from services import __version__, config
from services.analysis.clips import ClipStore
from services.cue import api
from services.cue.cue import cue_from_dict
from services.delivery.manual import ManualStore
from services.delivery.standin import StandinStore
from services.errors import ApiError, from_cuefx, internal, invalid, not_found
from services.make.generate import GenerateStore
from services.make.worker import RenderQueue

log = logging.getLogger("cuefx.server")


class SeedIn(BaseModel):
    """The artist's click. `frame` (analyze's name) or `f` (the cue's hand_seed name) both work."""
    model_config = ConfigDict(extra="forbid")
    frame: int | None = None
    f: int | None = None
    x: float
    y: float

    @model_validator(mode="after")
    def _one_frame(self) -> "SeedIn":
        if (self.frame is None) == (self.f is None):
            raise ValueError("give the seed frame as `frame` (or `f`)")
        return self


class BodyPointIn(BaseModel):
    """What the effect follows (v2.6); the values are checked by the backend (body_point_invalid)."""
    model_config = ConfigDict(extra="forbid")
    part: str
    side: str | None = None


class AnalyzeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clip_id: str
    hand: Literal["right", "left"]
    hand_seed: SeedIn | None = None
    body_point: BodyPointIn | None = None  # v2.6; missing = the palm of `hand`, today's behaviour

    @model_validator(mode="after")
    def _seed_for_palm(self) -> "AnalyzeIn":
        """A click stays required for the hand tracker; Pose body points can use their explicit part and side."""
        if self.hand_seed is None and (self.body_point is None or self.body_point.part == "palm"):
            raise ValueError("hand_seed is required for the palm")
        return self


class CueIn(BaseModel):
    """`cue` as in contract/schemas/cue.schema.json; `track` (API form) replaces the cue's own track when given."""
    model_config = ConfigDict(extra="forbid")
    cue: dict[str, Any]
    track: list[dict[str, Any]] | None = None


class NormalizeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cue: dict[str, Any]


class StandinIn(CueIn):
    """CueIn plus the clip to render on; without `clip_id` the stored clip named like the cue's `clip` is used."""
    clip_id: str | None = None


class ManualExportIn(BaseModel):
    """The cue to render by hand in ComfyUI, its clip, the look id (CUEFX_LOOKS_DIR/<look>.png) and the prompt
    (default: the cue's `prompt`)."""
    model_config = ConfigDict(extra="forbid")
    cue: dict[str, Any]
    clip_id: str
    look: str  # an id from GET /api/looks
    prompt: str | None = None


class GenerateIn(BaseModel):
    """One logical experimental image-edit + main/reverse LTX request."""
    model_config = ConfigDict(extra="forbid")
    cue: dict[str, Any]
    clip_id: str
    standin_id: str
    prompt_mode: Literal["user", "surprise"]
    prompt_text: str | None = None
    saved_look_identity: str | None = None
    experimental_ltx_overlap: bool = True


def create_app(settings: config.Settings | None = None) -> FastAPI:
    """The app for `settings` (default: from the environment)."""
    s = settings or config.from_env()
    store = ClipStore(s)
    queue = RenderQueue()  # the heavy work, one at a time; light routes never touch it
    standins = StandinStore(s, store, queue)
    manual = ManualStore(s, store, queue)
    generate = GenerateStore(s, store, standins, manual, queue)
    app = FastAPI(title="CueFX local API", version=__version__, docs_url="/api/docs", redoc_url=None,
                  openapi_url="/api/openapi.json")
    app.state.store = store
    app.state.queue = queue
    app.state.standins = standins
    app.state.manual = manual
    app.state.generate = generate
    app.add_middleware(CORSMiddleware, allow_origins=list(s.cors_origins), allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["Content-Disposition", "X-CueFX-Job-Id", "X-CueFX-Warnings", "X-CueFX-Effect-Range"])

    # ------------------------------------------------------------- errors: every failure is {code, message}

    def reply(e: ApiError) -> JSONResponse:
        return JSONResponse(e.to_dict(), status_code=e.status)

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError) -> JSONResponse:
        return reply(e)

    @app.exception_handler(api.CueFXError)
    async def _cuefx_error(_: Request, e: api.CueFXError) -> JSONResponse:
        return reply(from_cuefx(e))

    @app.exception_handler(RequestValidationError)
    async def _bad_request(_: Request, e: RequestValidationError) -> JSONResponse:
        detail = "; ".join(f"{'.'.join(str(p) for p in err['loc'][1:])}: {err['msg']}" for err in e.errors()[:5])
        return reply(invalid(detail or "the request body is not valid"))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, e: StarletteHTTPException) -> JSONResponse:
        if e.status_code == 404:
            return reply(not_found("no such route"))
        return reply(invalid(str(e.detail), e.status_code))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, e: Exception) -> JSONResponse:
        log.error("unhandled error on %s %s", request.method, request.url.path, exc_info=e)
        return reply(internal())

    # ------------------------------------------------------------- routes

    @app.get("/api/health")
    def health() -> dict:
        doc = {"status": "ok", "version": __version__, "contract_version": _contract_version()}
        return {**doc, "features": {"ltx_overlap_generate": True}} if generate.available else doc

    @app.get("/api/clips")
    def list_clips() -> list[dict]:
        return store.list()

    @app.post("/api/clips")
    def upload_clip(file: UploadFile, request: Request) -> JSONResponse:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > s.max_upload_bytes + (1 << 20):
            raise invalid(f"the file is larger than {s.max_upload_mb} MB", 413)
        info, created = store.upload(file.filename or "", file.file)
        store.warm_frames(info["clip_id"])
        return JSONResponse(info, status_code=201 if created else 200)

    @app.get("/api/clips/{clip_id}/video")
    def clip_video(clip_id: str) -> FileResponse:
        proxy = store.proxy(clip_id)
        store.warm_frames(clip_id)
        return FileResponse(proxy, media_type="video/mp4",
                            headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.get("/api/clips/{clip_id}/frames/{f:int}.jpg")
    def clip_frame(clip_id: str, f: int) -> Response:
        return Response(store.frame_jpeg(clip_id, f), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.post("/api/analyze")
    def analyze(body: AnalyzeIn) -> dict:
        return store.analyze(body.clip_id, body.hand, _seed(body), _body_point(body))

    @app.post("/api/cue/new")
    def cue_new(body: AnalyzeIn) -> JSONResponse:
        seed, bp = _seed(body), _body_point(body)
        result, info = store.analyze(body.clip_id, body.hand, seed, bp), store.info(body.clip_id)
        cue_seed = seed or result["body_point_seed"]
        cue, warnings = api.new_cue_with_warnings(
            info["clip"], info["fps"], info["frames"], info["plate_size"], body.hand,
            {"f": cue_seed["frame"], "x": cue_seed["x"], "y": cue_seed["y"]}, result["track"], store.transform(body.clip_id),
            result.get("body_point", None if body.body_point is None else body.body_point.model_dump()))
        # v2.7: the side the click chose is the cue's body_point (analyze's `body_point`); its side_from_click goes with it
        warnings = [w for w in result["warnings"] if w["code"] == "side_from_click"] + warnings
        # the body stays a cue; its warnings (none, or placeholder_timings) go in a header, as a JSON list
        return JSONResponse(cue, headers={"X-CueFX-Warnings": json.dumps(warnings)})

    @app.post("/api/cue/normalize")
    def cue_normalize(body: NormalizeIn) -> JSONResponse:
        # the body stays a cue; what normalize changed that the artist should hear about (v2.8: `vanish_extended`,
        # `window_trimmed` for an ending) goes in a header, as cue/new's warnings do
        cue, warnings = api.normalize_with_warnings(body.cue)
        return JSONResponse(cue, headers={"X-CueFX-Warnings": json.dumps(warnings)})

    @app.post("/api/validate")
    def validate(body: CueIn) -> JSONResponse:
        # the body stays a warning list; the effect_range that normalize would compute goes in a header (v2.8)
        headers = {}
        er = api.computed_effect_range(body.cue, body.track)
        if er is not None:
            headers["X-CueFX-Effect-Range"] = json.dumps(er)
        return JSONResponse(api.validate(body.cue, body.track), headers=headers)

    @app.get("/api/defaults")
    def defaults() -> dict:
        return api.defaults()

    @app.post("/api/preview")
    def preview(body: CueIn) -> list[dict]:
        return api.preview_path(body.cue, _track_of(body), with_held=True)

    @app.post("/api/propose_timing")
    def propose_timing(body: CueIn) -> dict:
        return api.propose_timing(body.cue, _track_of(body))

    @app.get("/api/looks")
    def list_looks() -> list[dict]:
        return manual.looks()

    # Heavy routes: the work runs on the render queue. By default (`?wait=0`, v2.4) they answer at once (202 while
    # queued or running) and the client polls the status route; `?wait=1` answers when the work is done.

    @app.post("/api/standin")
    def standin(body: StandinIn, wait: bool = False) -> JSONResponse:
        doc, finished = standins.render(body.cue, body.track, body.clip_id, wait)
        return JSONResponse(doc, status_code=200 if finished else 202)

    @app.get("/api/standin/{standin_id}")
    def standin_status(standin_id: str) -> dict:
        return standins.status(standin_id)

    @app.get("/api/standin/{standin_id}/{name}")
    def standin_file(standin_id: str, name: str, wait: bool = False) -> Response:
        path, download = standins.file(standin_id, name, wait)
        if path is None:  # element.mp4 is being built: ?wait=0 answers with the status instead
            return JSONResponse(standins.status(standin_id), status_code=202)
        return FileResponse(path, media_type="video/mp4", filename=download,
                            content_disposition_type="attachment" if download else "inline")

    @app.post("/api/manual/export")
    def manual_export(body: ManualExportIn, wait: bool = False) -> Response:
        job, package = manual.export(body.cue, body.clip_id, body.look, body.prompt, wait)
        headers = {"X-CueFX-Job-Id": job["id"]}
        if package is None:  # ?wait=0: the running job; poll GET /api/jobs/{id} until it is `waiting`
            return JSONResponse(job, status_code=202, headers=headers)
        _, download = manual.file(job["id"], "package.zip")
        return FileResponse(package, media_type="application/zip", filename=download, headers=headers)

    @app.post("/api/generate")
    def generate_job(body: GenerateIn) -> JSONResponse:
        if not body.experimental_ltx_overlap:
            raise invalid("experimental_ltx_overlap must be true for this endpoint")
        job = generate.start(body.cue, body.clip_id, body.standin_id, body.prompt_mode,
                             body.prompt_text, body.saved_look_identity)
        return JSONResponse(job, status_code=202)

    @app.post("/api/generate/{job_id}/retry")
    def retry_generate(job_id: str) -> JSONResponse:
        return JSONResponse(generate.retry(job_id), status_code=202)

    @app.post("/api/generate/{job_id}/cancel")
    def cancel_generate(job_id: str) -> dict:
        return generate.cancel(job_id)

    @app.post("/api/manual/{job_id}/result")
    def manual_result(job_id: str, effect_on_black: UploadFile, matte: UploadFile | None = None,
                      cue: str | None = Form(None), wait: bool = False) -> JSONResponse:
        current = None
        if cue is not None:
            try:
                current = json.loads(cue)
            except json.JSONDecodeError as e:
                raise invalid(f"cue: not JSON ({e})") from None
            if not isinstance(current, dict):
                raise invalid("cue: must be a JSON object")
        job = manual.import_result(job_id, effect_on_black.file, matte.file if matte else None, current, wait)
        return JSONResponse(job, status_code=200 if job["status"] == "done" else 202)

    @app.get("/api/manual/{job_id}/{name}")
    def manual_file(job_id: str, name: str) -> FileResponse:
        path, download = manual.file(job_id, name)
        media = "application/zip" if name.endswith(".zip") else "video/mp4"
        return FileResponse(path, media_type=media, filename=download,
                            content_disposition_type="attachment" if download else "inline")

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return manual.list()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        return manual.get(job_id)

    @app.get("/api/jobs/{job_id}/result")
    def job_result(job_id: str) -> dict:
        return manual.result(job_id)

    @app.get("/api/jobs/{job_id}/{name}")
    def job_file(job_id: str, name: str, download: bool = False) -> FileResponse:
        path, filename = manual.result_file(job_id, name)
        return FileResponse(path, media_type="video/mp4", filename=filename if download else None,
                            content_disposition_type="attachment" if download else "inline")

    return app


def _seed(body: AnalyzeIn) -> dict | None:
    """The analyze seed {frame, x, y} of the request (x, y as floats, so the cache key does not depend on 60 vs 60.0)."""
    sd = body.hand_seed
    if sd is None:
        return None
    return {"frame": sd.frame if sd.frame is not None else sd.f, "x": sd.x, "y": sd.y}


def _body_point(body: AnalyzeIn):
    """The request's body point as a BodyPoint (None if it has none); body_point_invalid if it cannot be used."""
    return api.parse_body_point(None if body.body_point is None else body.body_point.model_dump(), body.hand)


def _track_of(body: CueIn) -> list[dict]:
    """The API track to use: the request's, else the cue's own (cue form -> API form)."""
    if body.track is not None:
        return body.track
    api.check_schema("cue", body.cue)
    return api.track_from_points(cue_from_dict(body.cue).track)


@cache
def _contract_version() -> str:
    return json.loads((api.SCHEMA_DIR / "error_codes.json").read_text(encoding="utf-8"))["contract_version"]


def main() -> None:
    """Start the API with settings from CUEFX_* environment variables."""
    s = config.from_env()
    uvicorn.run(create_app(s), host=s.host, port=s.port)


if __name__ == "__main__":
    main()
