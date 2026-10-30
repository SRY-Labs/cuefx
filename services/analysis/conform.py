"""Conform an upload to the plate: 1920x1080, 24 fps (BRIEF section 2.1).

Scale to fit inside the plate keeping the aspect ratio, centre, pad with black, resample to 24 fps with ffmpeg's `fps`
filter, and apply the rotation tag first (phone clips). A clip that already is 1920x1080 at 24 fps with no rotation is
used as it is. `transform` is the cue's field: plate_px = source_px * scale + offset (source after rotation).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from services.make.layout import SWS_ACCURATE, LayoutError, find_ffmpeg
from services.delivery.video import _ffprobe

PLATE_W, PLATE_H, PLATE_FPS = 1920, 1080, 24
FPS_TOLERANCE = 0.01
CRF = 6  # near-lossless: the conformed plate is an intermediate, Merge composites onto it


class ConformError(LayoutError):
    """A clip cannot be probed or conformed."""


@dataclass(frozen=True)
class SourceInfo:
    """What conforming needs to know about a clip. `width`/`height` are after the rotation tag is applied."""
    width: int
    height: int
    fps: float          # r_frame_rate
    avg_fps: float      # avg_frame_rate (differs from fps for a variable-frame-rate clip)
    rotation: int       # degrees, as ffmpeg applies it (0, 90, 180, 270)
    duration: float     # seconds, container or stream
    color_space: str    # ffprobe's name, "" if untagged


def _fraction(text: str | None) -> float:
    try:
        return float(Fraction(text)) if text else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_source(video: Path) -> SourceInfo:
    """Size (after rotation), frame rates, rotation and duration of the first video stream."""
    proc = subprocess.run([_ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format",
                           "-of", "json", str(video)], capture_output=True)
    if proc.returncode:
        raise ConformError(f"cannot probe {video}: {proc.stderr.decode(errors='replace').strip()}")
    doc = json.loads(proc.stdout)
    if not doc.get("streams"):
        raise ConformError(f"{video} has no video stream")
    s = doc["streams"][0]
    rotation = 0
    for sd in s.get("side_data_list", []):
        if "rotation" in sd:
            rotation = round(float(sd["rotation"]))
    if not rotation and "rotate" in s.get("tags", {}):
        rotation = round(float(s["tags"]["rotate"]))
    rotation %= 360
    w, h = int(s["width"]), int(s["height"])
    if rotation % 180 == 90:
        w, h = h, w
    duration = _fraction(s.get("duration")) or _fraction(doc.get("format", {}).get("duration"))
    return SourceInfo(w, h, _fraction(s.get("r_frame_rate")), _fraction(s.get("avg_frame_rate")), rotation, duration,
                      s.get("color_space", ""))


def is_plate(src: SourceInfo) -> bool:
    """True if the clip already is a plate: 1920x1080, 24 fps (constant), no rotation."""
    return ((src.width, src.height) == (PLATE_W, PLATE_H) and src.rotation == 0
            and abs(src.fps - PLATE_FPS) < FPS_TOLERANCE and abs(src.avg_fps - PLATE_FPS) < FPS_TOLERANCE)


def fit(width: int, height: int) -> tuple[float, tuple[int, int], tuple[int, int]]:
    """(scale, scaled size, offset) that fit a width x height picture inside the plate, centred. The scaled size and
    the offset are even, as yuv420p needs; `scale` is the exact ratio (the scaled size is that, rounded to even)."""
    scale = min(PLATE_W / width, PLATE_H / height)
    sw, sh = (min(PLATE_W, max(2, round(width * scale / 2) * 2)), min(PLATE_H, max(2, round(height * scale / 2) * 2)))
    return scale, (sw, sh), ((PLATE_W - sw) // 2 // 2 * 2, (PLATE_H - sh) // 2 // 2 * 2)


def transform_for(src: SourceInfo) -> dict:
    """The cue's `transform` {scale, offset} for this clip: identity for a clip that already is a plate."""
    if is_plate(src):
        return {"scale": 1.0, "offset": [0, 0]}
    scale, _, (ox, oy) = fit(src.width, src.height)
    return {"scale": round(scale, 6), "offset": [ox, oy]}


def estimated_frames(src: SourceInfo) -> int:
    """round(duration x 24): what the conformed plate will have (0 if the duration is unknown)."""
    return round(src.duration * PLATE_FPS)


def _in_matrix(src: SourceInfo) -> str:
    """The YUV matrix to read the clip with. An untagged HD clip is Rec.709 (ffmpeg would assume BT.601)."""
    if src.color_space in ("bt709", "bt2020nc", "smpte170m", "bt470bg"):
        return {"bt709": "bt709", "bt2020nc": "bt2020", "smpte170m": "bt601", "bt470bg": "bt601"}[src.color_space]
    return "bt709" if src.height >= 720 else "bt601"


def conform_command(src_path: Path, src: SourceInfo, out: Path) -> list[str]:
    """The ffmpeg command that writes the conformed plate `out` from `src_path`. ffmpeg applies the rotation tag
    itself, before the filters. Audio is kept (AAC) so Merge can copy it to its output."""
    _, (sw, sh), (ox, oy) = fit(src.width, src.height)
    vf = (f"setpts=PTS-STARTPTS,fps={PLATE_FPS},"
          f"scale={sw}:{sh}:flags=accurate_rnd+full_chroma_int:in_color_matrix={_in_matrix(src)}:out_color_matrix=bt709,"
          f"pad={PLATE_W}:{PLATE_H}:{ox}:{oy}:color=black,setsar=1,format=yuv420p")
    return [find_ffmpeg(), "-y", "-loglevel", "error", "-i", str(src_path), "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", vf, *SWS_ACCURATE, "-fps_mode", "cfr", "-r", str(PLATE_FPS),
            "-c:v", "libx264", "-preset", "fast", "-crf", str(CRF), "-profile:v", "high", "-g", "12",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(out)]


def conform(src_path: Path, out: Path, src: SourceInfo | None = None) -> None:
    """Write the conformed plate of `src_path` to `out` (atomically: a .part file renamed at the end).
    ConformError if ffmpeg fails."""
    src = src or probe_source(src_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.stem + ".part" + out.suffix)
    proc = subprocess.run(conform_command(src_path, src, part), capture_output=True)
    if proc.returncode:
        part.unlink(missing_ok=True)
        raise ConformError(f"ffmpeg failed conforming {src_path.name}: {proc.stderr.decode(errors='replace').strip()}")
    part.replace(out)
