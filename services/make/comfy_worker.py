"""Narrow ComfyUI adapters for image edit, main LTX and reverse LTX jobs."""
from __future__ import annotations

import hashlib
import json
import mimetypes
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from services.make.ltx_overlap import OverlapPlan
from services.make.ltx_overlap_graph import fill_image_edit_api, fill_main_api, fill_reverse_api


class WorkerError(RuntimeError):
    """The configured ComfyUI worker could not complete a job."""


@dataclass(frozen=True)
class WorkerResult:
    path: Path
    prompt_id: str
    history: dict
    output: dict

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


class ComfyWorker:
    """Small standard-library ComfyUI HTTP client; no endpoint is hardcoded."""

    def __init__(self, base_url: str, token: str | None = None, timeout_s: int = 7200) -> None:
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise ValueError("CUEFX_COMFY_URL must be an http(s) URL")
        self.base = base_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(extra or {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _open(self, request: Request, timeout: int = 60):
        try:
            return urlopen(request, timeout=timeout)
        except (HTTPError, URLError, TimeoutError) as e:
            raise WorkerError(f"ComfyUI request failed: {e}") from e

    def json(self, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = self._headers({"Content-Type": "application/json"} if data is not None else None)
        with self._open(Request(self.base + path, data=data, headers=headers)) as response:
            try:
                return json.load(response)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise WorkerError(f"ComfyUI returned invalid JSON for {path}") from e

    def upload(self, path: Path, remote_name: str, subfolder: str = "") -> str:
        """Upload one input through ComfyUI's input endpoint and return its graph name."""
        boundary = "----CueFX" + uuid.uuid4().hex
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        fields = {"overwrite": "true", "type": "input", "subfolder": subfolder}
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                           f"{value}\r\n").encode())
        chunks.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
                       f"filename=\"{remote_name}\"\r\nContent-Type: {mime}\r\n\r\n").encode())
        chunks.extend((path.read_bytes(), b"\r\n", f"--{boundary}--\r\n".encode()))
        body = b"".join(chunks)
        request = Request(self.base + "/upload/image", data=body,
                          headers=self._headers({"Content-Type": f"multipart/form-data; boundary={boundary}"}))
        with self._open(request, timeout=300) as response:
            result = json.load(response)
        name = result.get("name", remote_name)
        folder = result.get("subfolder", subfolder)
        return f"{folder}/{name}" if folder else name

    def interrupt(self) -> None:
        try:
            self.json("/interrupt", {})
        except WorkerError:
            pass

    def run(self, graph: dict, output_dir: Path, extension: str,
            cancelled: Callable[[], bool] = lambda: False) -> WorkerResult:
        queued = self.json("/prompt", {"prompt": graph})
        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            raise WorkerError("ComfyUI did not return a prompt_id")
        deadline = time.monotonic() + self.timeout_s
        record = None
        while time.monotonic() < deadline:
            if cancelled():
                self.interrupt()
                raise WorkerError("cancelled")
            history = self.json("/history/" + prompt_id)
            record = history.get(prompt_id)
            if record is None:
                time.sleep(2)
                continue
            status = record.get("status", {})
            if status.get("completed") and status.get("status_str") == "success":
                break
            if status.get("status_str") == "error":
                raise WorkerError("ComfyUI prompt failed")
            time.sleep(2)
        else:
            raise WorkerError(f"ComfyUI prompt {prompt_id} timed out")
        candidates = [item for output in record.get("outputs", {}).values()
                      for item in output.get("images", [])
                      if str(item.get("filename", "")).lower().endswith(extension.lower())]
        if len(candidates) != 1:
            raise WorkerError(f"ComfyUI returned {len(candidates)} {extension} outputs; expected one")
        item = candidates[0]
        query = urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                           "type": item.get("type", "output")})
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / Path(item["filename"]).name
        request = Request(self.base + "/view?" + query, headers=self._headers())
        with self._open(request, timeout=300) as response, destination.open("wb") as target:
            while block := response.read(8 * 1024 * 1024):
                target.write(block)
        return WorkerResult(destination, prompt_id, record, item)


class ComfyAdapters:
    """Three stage-specific adapters sharing one configured worker."""

    def __init__(self, worker: ComfyWorker) -> None:
        self.worker = worker

    def edit_look(self, reference: Path, prompt: str, seed: int, output_dir: Path,
                  cancelled: Callable[[], bool]) -> WorkerResult:
        remote = self.worker.upload(reference, reference.name)
        return self.worker.run(fill_image_edit_api(remote, prompt, seed), output_dir, ".png", cancelled)

    def generate_main(self, video: Path, look: Path, prompt: str, output_dir: Path,
                      cancelled: Callable[[], bool]) -> WorkerResult:
        remote_video = self.worker.upload(video, video.name)
        remote_look = self.worker.upload(look, look.name)
        graph = fill_main_api(remote_video, remote_look, prompt, "cuefx/main")
        return self.worker.run(graph, output_dir, ".mp4", cancelled)

    def generate_reverse(self, guide: Path, matched_look: Path, prefix_dir: Path,
                         main_prompt: dict, plan: OverlapPlan, output_dir: Path,
                         cancelled: Callable[[], bool]) -> WorkerResult:
        remote_guide = self.worker.upload(guide, guide.name)
        remote_look = self.worker.upload(matched_look, matched_look.name)
        remote_prefix = "cuefx_prefix_" + uuid.uuid4().hex[:12]
        for frame in sorted(prefix_dir.glob("*.png")):
            self.worker.upload(frame, frame.name, remote_prefix)
        graph = fill_reverse_api(main_prompt, plan, remote_guide, remote_look, remote_prefix, "cuefx/reverse")
        return self.worker.run(graph, output_dir, ".mp4", cancelled)
