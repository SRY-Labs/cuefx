"""Streaming video read and write through ffmpeg, as 8-bit BGR frames.

Reading honours the file's colour tags (the plate is tagged Rec.709); an untagged file is read as BT.601, which
is how ffmpeg writes untagged files (`services.make.layout.write_video`), so our own effect files round-trip.
OpenCV's reader always uses BT.601, which shifts a Rec.709 plate by up to ~10 codes; Merge uses this reader.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np

from services.make.layout import SWS_ACCURATE, LayoutError, find_ffmpeg

def bt709_out(pix_fmt: str = "yuv420p") -> list[str]:
    """ffmpeg arguments for a Rec.709-matrix, tv-range, Rec.709-tagged file in `pix_fmt`, so players show it like
    the plate. Use with `VideoWriter(..., pix_fmt=pix_fmt)`."""
    return ["-vf", f"scale=out_color_matrix=bt709:out_range=tv,format={pix_fmt}", "-colorspace", "bt709",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


# Rec.709 matrix and tags for the files Merge writes, so players show them like the plate
BT709_OUT = bt709_out()


class VideoError(LayoutError):
    """A video cannot be read or written."""


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    has_audio: bool


def _ffprobe() -> str:
    exe = find_ffmpeg()
    probe = Path(exe).with_name("ffprobe" + Path(exe).suffix)
    return str(probe) if probe.exists() else "ffprobe"


def probe(video: Path) -> VideoProbe:
    """Size and frame rate of the first video stream, and whether there is an audio stream."""
    proc = subprocess.run([_ffprobe(), "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate",
                           "-of", "json", str(video)], capture_output=True)
    if proc.returncode:
        raise VideoError(f"cannot probe {video}: {proc.stderr.decode(errors='replace').strip()}")
    streams = json.loads(proc.stdout)["streams"]
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise VideoError(f"{video} has no video stream")
    return VideoProbe(int(v["width"]), int(v["height"]), float(Fraction(v["r_frame_rate"])),
                      any(s.get("codec_type") == "audio" for s in streams))


def read_frames(video: Path) -> Iterator[np.ndarray]:
    """Every frame of `video` as (h, w, 3) uint8 BGR, in order."""
    info = probe(video)
    size = info.width * info.height * 3
    proc = subprocess.Popen([find_ffmpeg(), "-v", "error", "-i", str(video), "-map", "0:v:0", "-f", "rawvideo",
                             "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def read_all(video: Path) -> list[np.ndarray]:
    """Every frame of `video`, as a list."""
    return list(read_frames(video))


class VideoWriter:
    """Encode 8-bit BGR frames as H.264 (yuv420p unless `pix_fmt` says otherwise), one `write` per frame; use as a
    context manager."""

    def __init__(self, out: Path, size: tuple[int, int], fps: float, crf: int, extra: list[str] | None = None,
                 pix_fmt: str = "yuv420p"):
        out.parent.mkdir(parents=True, exist_ok=True)
        self.out, self.size, self.frames = out, size, 0
        w, h = size
        self.proc = subprocess.Popen(
            [find_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", f"{fps:g}", "-i", "-", *SWS_ACCURATE, *(extra or []), "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", pix_fmt,
             "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.size[1], self.size[0], 3) or frame.dtype != np.uint8:
            raise VideoError(f"{self.out.name}: frame {frame.shape} {frame.dtype} is not {self.size} uint8 BGR")
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace").strip()
        if self.proc.wait():
            raise VideoError(f"ffmpeg failed writing {self.out.name}: {err}")

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.proc.kill()
            self.proc.wait()


def add_audio(video: Path, source: Path, out: Path) -> list[str]:
    """Write `out`: the video of `video` with the audio of `source`, copied. Returns warnings.

    If the audio codec cannot be copied into the container it is re-encoded as AAC, with a warning;
    if even that fails, `out` is the video without audio, with a warning.
    """
    base = [find_ffmpeg(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(source),
            "-map", "0:v:0", "-map", "1:a", "-c:v", "copy"]
    proc = subprocess.run(base + ["-c:a", "copy", "-movflags", "+faststart", str(out)], capture_output=True)
    if not proc.returncode:
        return []
    proc = subprocess.run(base + ["-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(out)],
                          capture_output=True)
    if not proc.returncode:
        return [f"audio: the plate's audio cannot be copied into {out.name}; re-encoded as AAC"]
    subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-i", str(video), "-c", "copy", str(out)],
                   capture_output=True)
    return [f"audio: could not add the plate's audio to {out.name} "
            f"({proc.stderr.decode(errors='replace').strip()}); written without audio"]
