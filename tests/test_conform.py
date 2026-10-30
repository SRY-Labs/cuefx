"""services.analysis.conform: the fit maths and the plate test (no ffmpeg; the ffmpeg cases are in tests/server/test_conform_upload.py)."""
import pytest

from services.analysis import conform
from services.analysis.conform import SourceInfo


def src(w, h, fps=24.0, avg=None, rotation=0, duration=2.0, color="bt709") -> SourceInfo:
    return SourceInfo(w, h, fps, fps if avg is None else avg, rotation, duration, color)


@pytest.mark.parametrize("w, h, scale, size, offset", [
    (1280, 720, 1.5, (1920, 1080), (0, 0)),
    (3840, 2160, 0.5, (1920, 1080), (0, 0)),
    (1080, 1920, 0.5625, (608, 1080), (656, 0)),     # portrait: pillarbox, 656 px of black each side
    (1920, 1440, 0.75, (1440, 1080), (240, 0)),      # 4:3
    (1920, 800, 1.0, (1920, 800), (0, 140)),         # letterbox
    (1921, 1081, 1080 / 1081, (1920, 1080), (0, 0)),  # odd size: the scaled size is rounded to even
])
def test_fit_scales_to_fit_and_centres(w, h, scale, size, offset):
    s, got_size, got_offset = conform.fit(w, h)
    assert s == pytest.approx(scale) and got_size == size and got_offset == offset
    assert all(v % 2 == 0 for v in (*got_size, *got_offset))  # yuv420p needs even sizes and offsets
    assert got_offset[0] + got_size[0] <= 1920 and got_offset[1] + got_size[1] <= 1080


def test_a_plate_is_used_as_is_with_the_identity_transform():
    assert conform.is_plate(src(1920, 1080))
    assert conform.transform_for(src(1920, 1080)) == {"scale": 1.0, "offset": [0, 0]}


@pytest.mark.parametrize("info", [
    src(1920, 1080, fps=30), src(1920, 1080, fps=23.976), src(1920, 1080, avg=23.5),   # other or variable rate
    src(1920, 1080, rotation=180), src(1280, 720), src(1080, 1920),
])
def test_anything_else_is_conformed(info):
    assert not conform.is_plate(info)


def test_transform_for_a_conformed_clip():
    assert conform.transform_for(src(1280, 720, fps=30)) == {"scale": 1.5, "offset": [0, 0]}
    assert conform.transform_for(src(1080, 1920, fps=30)) == {"scale": 0.5625, "offset": [656, 0]}


def test_estimated_frames_is_the_duration_at_24_fps():
    assert conform.estimated_frames(src(1280, 720, duration=2.0)) == 48
    assert conform.estimated_frames(src(1280, 720, duration=2.51)) == 60
    assert conform.estimated_frames(src(1280, 720, duration=0)) == 0


def test_an_untagged_hd_clip_is_read_as_rec709():
    assert conform._in_matrix(src(1280, 720, color="")) == "bt709"
    assert conform._in_matrix(src(640, 360, color="")) == "bt601"
    assert conform._in_matrix(src(1280, 720, color="smpte170m")) == "bt601"
