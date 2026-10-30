"""Clip storage: uploaded clips by content-hash id, the browser proxy, exact frames and the analyze cache."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import BinaryIO

from services.cue import api
from services.analysis import conform
from services.cue.cue import BodyPoint, is_palm
from services.make.layout import find_ffmpeg
from services.analysis.pose_track import POSE_SOURCE
from services.delivery.video import probe

from services.config import ALLOWED_EXTENSIONS, Settings
from services.errors import ApiError, invalid, not_found

log = logging.getLogger("cuefx.server")

ID_RE = re.compile(r"^[0-9a-f]{12}$")
PLATE_SIZE = (1920, 1080)
PLATE_NAME = "plate.mp4"  # the conformed plate in <cache>/<id>/
CHUNK = 1 << 20
FRAME_LRU = 48  # frames kept in memory (about 0.3 MB each)


def safe_name(name: str) -> str:
    """A file name that is safe to store: no directories, only letters, digits and ._- ."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name.replace("\\", "/")).name).strip("._")
    return stem or "clip"


class ClipStore:
    """Where clips live. Uploads: <data>/uploads/<id>/<name>. Cache: <data>/cache/<id>/.

    `path` is the original file, never changed. Everything downstream (analyze, frames, proxy, stand-in) uses
    `plate`: the original itself if it is 1920x1080 at 24 fps, else its conformed copy in the cache."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._infos: dict[str, dict] = {}
        self._transforms: dict[str, dict] = {}
        self._plates: dict[str, Path] = {}
        self._jpegs: OrderedDict[tuple[str, int], bytes] = OrderedDict()  # frame LRU, FRAME_LRU entries
        self._fills: dict[str, threading.Thread] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self.analyze_lock = threading.Lock()  # one analyze at a time: the tracker uses the whole CPU

    def lock(self, name: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(name, threading.Lock())

    # ----------------------------------------------------------------- lookup

    def _uploads(self) -> list[tuple[str, Path]]:
        d = self.s.uploads_dir
        if not d.is_dir():
            return []
        return [(c.name, f) for c in sorted(d.iterdir()) if ID_RE.match(c.name) and c.is_dir()
                for f in sorted(c.iterdir())[:1] if f.is_file() and not f.name.startswith(".")]

    def path(self, clip_id: str) -> Path:
        """The file of clip `clip_id`; 404 (input_invalid) if there is none."""
        if ID_RE.match(clip_id):
            for cid, p in self._uploads():
                if cid == clip_id:
                    return p
        raise not_found(f"unknown clip {clip_id!r}")

    def by_name(self, name: str) -> str:
        """The id of the one stored clip whose file is called `name` (a cue's `clip`); 404 if none, 400 if several
        different clips have that name (the request must then give clip_id)."""
        ids = {cid for cid, p in self._uploads() if p.name == name}
        if not ids:
            raise not_found(f"no stored clip is called {name!r}; upload it or send clip_id")
        if len(ids) > 1:
            raise invalid(f"several clips are called {name!r}; send clip_id")
        return ids.pop()

    def info(self, clip_id: str) -> dict:
        """{clip_id, clip, fps, frames, plate_size} of the conformed plate of a stored clip."""
        if clip_id not in self._infos:
            self._infos[clip_id] = self._info(clip_id, self.path(clip_id).name, self.plate(clip_id))
        return self._infos[clip_id]

    @staticmethod
    def _info(clip_id: str, name: str, plate: Path) -> dict:
        frames = api._probe_frames(plate)  # same count and same CueFXError(clip_unreadable) as analyze
        p = probe(plate)
        return {"clip_id": clip_id, "clip": name, "fps": p.fps, "frames": frames, "plate_size": [p.width, p.height]}

    def transform(self, clip_id: str) -> dict:
        """The cue's `transform` {scale, offset} that maps the original (after rotation) onto the plate."""
        if clip_id not in self._transforms:
            self._transforms[clip_id] = conform.transform_for(self._source(self.path(clip_id)))
        return self._transforms[clip_id]

    @staticmethod
    def _source(path: Path) -> conform.SourceInfo:
        try:
            return conform.probe_source(path)
        except conform.ConformError as e:
            log.error("%s", e)
            raise api.error("clip_unreadable") from e

    def plate(self, clip_id: str) -> Path:
        """The conformed plate of clip `clip_id` (1920x1080, 24 fps), made once and cached by content hash. A clip
        that already is a plate is returned as it is. Remembered in memory: later calls cost no ffprobe."""
        src = self.path(clip_id)
        known = self._plates.get(clip_id)
        if known is not None and known.is_file():
            return known
        with self.lock(f"plate-{clip_id}"):
            out = self.s.cache_dir / clip_id / PLATE_NAME
            if not out.is_file():
                info = self._source(src)
                if conform.is_plate(info):
                    out = src
                else:
                    try:
                        conform.conform(src, out, info)
                    except conform.ConformError as e:
                        log.error("%s", e)
                        raise api.error("clip_unreadable") from e
            self._plates[clip_id] = out
            return out

    def list(self) -> list[dict]:
        """Uploaded clips, one entry per content hash."""
        out, seen = [], set()
        for cid, p in self._uploads():
            if cid in seen:
                continue
            seen.add(cid)
            try:
                out.append(self.info(cid))
            except Exception as e:  # an unreadable file in the folder must not break the list
                log.warning("skipping %s in the clip list: %s", p, e)
        return out

    # ----------------------------------------------------------------- upload

    def upload(self, filename: str, source: BinaryIO) -> tuple[dict, bool]:
        """Store an uploaded clip; (info, created). The same bytes give the same id and are stored once."""
        ext = Path(filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise invalid(f"{ext or 'this file'} is not a supported clip format; use " + ", ".join(ALLOWED_EXTENSIONS))
        tmp_dir = self.s.uploads_dir / f".tmp-{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True)
        try:
            tmp = tmp_dir / ("clip" + ext)
            h, size = hashlib.sha256(), 0
            with tmp.open("wb") as out:
                while chunk := source.read(CHUNK):
                    size += len(chunk)
                    if size > self.s.max_upload_bytes:
                        raise invalid(f"the file is larger than {self.s.max_upload_mb} MB", 413)
                    h.update(chunk)
                    out.write(chunk)
            if size == 0:
                raise invalid("the file is empty")
            clip_id = h.hexdigest()[:12]
            dest = self.s.uploads_dir / clip_id / safe_name(filename)
            if dest.parent.is_dir() and any(dest.parent.iterdir()):
                return self.info(clip_id), False
            try:
                src = self._source(tmp)  # unreadable / too long are refused before anything is kept
                if (est := conform.estimated_frames(src)) > api.MAX_CLIP_FRAMES:
                    raise ApiError(*_too_long(est))
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp.replace(dest)
                try:
                    info = self._info(clip_id, dest.name, self.plate(clip_id))  # conforms here, once
                    if info["frames"] > api.MAX_CLIP_FRAMES:  # the limit again, on the conformed frame count
                        raise ApiError(*_too_long(info["frames"]))
                except Exception:
                    shutil.rmtree(dest.parent, ignore_errors=True)
                    shutil.rmtree(self.s.cache_dir / clip_id, ignore_errors=True)
                    raise
            except api.CueFXError as e:
                raise ApiError(e.code, e.message, 400) from e
            self._infos[clip_id] = info
            return info, True
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ----------------------------------------------------------------- proxy video

    def proxy(self, clip_id: str) -> Path:
        """Browser-playable H.264 MP4 of the conformed plate (1920x1080, 24 fps, its frame count), made once."""
        src, info = self.plate(clip_id), self.info(clip_id)
        out = self.s.cache_dir / clip_id / "proxy.mp4"
        with self.lock(f"proxy-{clip_id}"):
            if out.is_file():
                return out
            out.parent.mkdir(parents=True, exist_ok=True)
            part = out.with_name("proxy.part.mp4")
            w, h = PLATE_SIZE
            cmd = [find_ffmpeg(), "-y", "-loglevel", "error", "-i", str(src), "-map", "0:v:0", "-an",
                   "-vf", f"scale={w}:{h}:flags=accurate_rnd+full_chroma_int:out_color_matrix=bt709:out_range=tv,"
                          "format=yuv420p",
                   "-fps_mode", "cfr", "-r", f"{info['fps']:g}",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-profile:v", "high", "-g", "12",
                   "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                   "-movflags", "+faststart", str(part)]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode:
                part.unlink(missing_ok=True)
                log.error("ffmpeg proxy failed for %s: %s", src, proc.stderr.decode(errors="replace").strip())
                raise api.error("clip_unreadable")
            part.replace(out)
            return out

    # ----------------------------------------------------------------- exact frame

    def frame_jpeg(self, clip_id: str, f: int) -> bytes:
        """Frame `f` (0-based) of the conformed plate, decoded exactly with PyAV, at plate size, as JPEG bytes.

        Served from a small in-memory LRU, else from <cache>/<id>/frames/<f>.jpg. The first request of a clip starts
        one background pass that decodes the plate once and writes every frame there: a plate with few keyframes
        (71319 has one) would otherwise decode from the start for every late frame. A frame not written yet is
        decoded directly (seek to the keyframe before it), so no request waits for that pass."""
        src, info = self.plate(clip_id), self.info(clip_id)
        if not 0 <= f < info["frames"]:
            raise invalid(f"frame {f} is outside the clip (0..{info['frames'] - 1})")
        with self._guard:
            jpeg = self._jpegs.get((clip_id, f))
            if jpeg is not None:
                self._jpegs.move_to_end((clip_id, f))
                return jpeg
        path = self.frames_dir(clip_id) / f"{f:05d}.jpg"
        if path.is_file():
            jpeg = path.read_bytes()
        else:
            self._start_fill(clip_id, src, info["frames"])
            jpeg = _encode(self._decode(src, f))
            _write_atomic(path, jpeg)
        with self._guard:
            self._jpegs[(clip_id, f)] = jpeg
            while len(self._jpegs) > FRAME_LRU:
                self._jpegs.popitem(last=False)
        return jpeg

    def frames_dir(self, clip_id: str) -> Path:
        return self.s.cache_dir / clip_id / "frames"

    def warm_frames(self, clip_id: str) -> None:
        """Start the background frame pass of a clip now (the UI opens its video first, then scrubs)."""
        self._start_fill(clip_id, self.plate(clip_id), self.info(clip_id)["frames"])

    def _start_fill(self, clip_id: str, src: Path, frames: int) -> None:
        with self._guard:
            if clip_id in self._fills:
                return
            t = threading.Thread(target=self._fill, args=(clip_id, src, frames), name=f"cuefx-frames-{clip_id}",
                                 daemon=True)
            self._fills[clip_id] = t
        t.start()

    def wait_frames(self, clip_id: str, timeout: float | None = None) -> None:
        """Wait for the background frame pass of `clip_id` (if one was started)."""
        t = self._fills.get(clip_id)
        if t is not None:
            t.join(timeout)

    def _fill(self, clip_id: str, src: Path, frames: int) -> None:
        """Decode the plate once from the start and write every frame not on disk yet as JPEG."""
        import av

        out = self.frames_dir(clip_id)
        t0 = time.perf_counter()
        try:
            with av.open(str(src)) as c:
                s = c.streams.video[0]
                s.thread_type = "AUTO"  # frame threads: the same frames, decoded faster
                start, rate = s.start_time or 0, s.average_rate
                for fr in c.decode(s):
                    idx = round((fr.pts - start) * s.time_base * rate)
                    if idx >= frames:
                        break
                    path = out / f"{idx:05d}.jpg"
                    if idx >= 0 and not path.is_file():
                        _write_atomic(path, _encode(_rgb(fr)))
            log.info("frames of clip %s cached on disk in %.1f s", clip_id, time.perf_counter() - t0)
        except Exception as e:  # the cache is an optimisation: requests still decode directly
            log.warning("frame cache of clip %s stopped: %s", clip_id, e)

    @staticmethod
    def _decode(src: Path, f: int):
        """Frame `f` of `src` as plate-size RGB: seek to the keyframe at or before it, decode forward."""
        import av

        with av.open(str(src)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"  # frame threads: the same frames, decoded faster
            start = s.start_time or 0
            rate = s.average_rate
            c.seek(start + round(f / rate / s.time_base), stream=s, backward=True)
            for fr in c.decode(s):
                idx = round((fr.pts - start) * s.time_base * rate)
                if idx < f:
                    continue
                if idx > f:
                    break
                return _rgb(fr)
        raise not_found(f"frame {f} cannot be decoded from this clip")

    # ----------------------------------------------------------------- analyze cache

    def _analyze_files(self, clip_id: str, hand: str, seed: dict | None, body_point: BodyPoint | None = None
                       ) -> tuple[Path, Path]:
        """(analyze result .json, landmarks .npz) in the cache, keyed by (clip content, hand, seed, code version) and,
        for a body point other than the palm, by the body point (so a palm key is what it always was)."""
        parts = [clip_id, hand, seed, code_version()]
        if not is_palm(body_point):
            parts.append([body_point.part, body_point.side])
        key = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]
        d = self.s.cache_dir / clip_id
        return d / f"analyze_{key}.json", d / f"landmarks_{key}.npz"

    def pose_file(self, clip_id: str) -> Path:
        """The cached Pose landmarks of a clip (all 33 per frame): one file per clip and pose source, whatever the body
        point, so switching the body point never runs Pose again."""
        key = hashlib.sha256(json.dumps([clip_id, POSE_SOURCE], sort_keys=True).encode()).hexdigest()[:16]
        return self.s.cache_dir / clip_id / f"pose_{key}.npz"

    def _run_analyze(self, path: Path, seed: dict, landmarks: Path) -> dict:
        """api.analyze, writing the landmarks next to the result (atomically: a .part file renamed at the end)."""
        landmarks.parent.mkdir(parents=True, exist_ok=True)
        part = landmarks.with_name(landmarks.stem + ".part.npz")
        result = api.analyze(path, seed, landmarks_out=part)
        if part.is_file():
            part.replace(landmarks)
        return result

    def analyze(self, clip_id: str, hand: str, seed: dict | None, body_point: BodyPoint | None = None) -> dict:
        """api.analyze, cached on disk by (clip content, hand, seed, analyze code version); the hand landmarks
        Merge needs are cached beside it. A body point other than the palm is tracked from the clip's cached Pose
        landmarks (`pose_file`) and cached by its part and side too."""
        path = self.plate(clip_id)
        cached, landmarks = self._analyze_files(clip_id, hand, seed, body_point)
        with self.analyze_lock:
            if cached.is_file():
                try:
                    return json.loads(cached.read_text(encoding="utf-8"))
                except ValueError:
                    log.warning("discarding unreadable analyze cache %s", cached)
            if is_palm(body_point):
                result = self._run_analyze(path, seed, landmarks)
            else:
                result = api.analyze(path, seed, body_point=body_point, hand=hand, pose_cache=self.pose_file(clip_id))
            part = cached.with_suffix(".part")
            part.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            part.replace(cached)
            return result

    def landmarks(self, clip_id: str, hand: str, seed: dict) -> Path:
        """The cached hand landmarks of this analyze; analyzes again (about 12 s) if they are not there."""
        path = self.plate(clip_id)
        _, landmarks = self._analyze_files(clip_id, hand, seed)
        with self.analyze_lock:
            if not landmarks.is_file():
                self._run_analyze(path, seed, landmarks)
            return landmarks


def _rgb(frame):
    """A decoded PyAV frame as plate-size RGB (the reformat honours the file's colour tags, as Merge does)."""
    import cv2

    rgb = frame.reformat(format="rgb24").to_ndarray()
    if (rgb.shape[1], rgb.shape[0]) != PLATE_SIZE:
        rgb = cv2.resize(rgb, PLATE_SIZE, interpolation=cv2.INTER_AREA)
    return rgb


def _encode(rgb) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("jpeg encoding failed")
    return buf.tobytes()


def _write_atomic(path: Path, data: bytes) -> None:
    """Write through a unique .part file renamed at the end: readers never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.stem}.{uuid.uuid4().hex[:8]}.part")
    part.write_bytes(data)
    try:
        part.replace(path)
    except OSError:  # another writer got there first (Windows refuses to replace an open file): theirs is the same
        part.unlink(missing_ok=True)


def _too_long(frames: int) -> tuple[str, str, int]:
    return ("clip_too_long", api.message("clip_too_long", frames=frames, max_frames=api.MAX_CLIP_FRAMES,
                                         max_seconds=api.MAX_CLIP_FRAMES // 24), 400)


_code_version: str | None = None


def code_version() -> str:
    """Hash of every services/**/*.py: any change to the library invalidates the analyze cache."""
    global _code_version
    if _code_version is None:
        root = Path(__file__).resolve().parents[1]
        h = hashlib.sha256()
        for p in sorted(root.rglob("*.py")):
            h.update(p.relative_to(root).as_posix().encode() + p.read_bytes())
        _code_version = h.hexdigest()[:12]
    return _code_version
