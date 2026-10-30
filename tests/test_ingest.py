"""Ingest of LTX output: size and count checks, the temporary luminance matte, and --effect-dir leaving files alone."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from services.common.colour import decode8, encode8, luminance
from services.make.ingest import (MATTE_WARNING, IngestError, MatteConfig, bright_level, fit_to_range, ingest, luminance_matte,
                          read_ltx, write_ingest)
from services.make.layout import count_frames
from services.delivery.merge import merge_dirs
from services.delivery.video import read_all

CANVAS = (64, 32)


def blob_frame(level: float = 1.0) -> np.ndarray:
    """Black canvas with a soft bright disc of linear brightness `level` (8-bit sRGB BGR)."""
    yy, xx = np.mgrid[0:CANVAS[1], 0:CANVAS[0]].astype(np.float32)
    lum = level * np.clip(1.5 - np.hypot(xx - 32, yy - 16) / 5, 0, 1)
    return encode8(np.dstack([lum] * 3))


def test_size_mismatch_is_a_clear_error():
    with pytest.raises(IngestError, match="64x32"):
        ingest([np.zeros((30, 64, 3), np.uint8)] * 4, 4, CANVAS)


def test_too_few_frames_warns_and_leaves_the_end_empty():
    res = ingest([blob_frame()] * 3, 5, CANVAS)
    assert len(res.effect) == len(res.matte) == 5
    assert any("3 frames" in w and "5" in w for w in res.warnings)
    assert MATTE_WARNING in res.warnings
    assert not res.effect[3].any() and not res.effect[4].any() and not res.matte[4].any()
    assert res.matte[0].any()


def test_too_many_frames_warns_and_drops_the_extra():
    res = ingest([blob_frame()] * 7, 5, CANVAS)
    assert len(res.effect) == 5
    assert any("7 frames" in w for w in res.warnings)


def test_exact_count_only_warns_about_the_temporary_matte():
    assert ingest([blob_frame()] * 5, 5, CANVAS).warnings == [MATTE_WARNING]


def test_frame_count_mismatch_still_writes_both_files(tmp_path):
    res = ingest([blob_frame()] * 3, 5, CANVAS)
    write_ingest(res, 24, tmp_path)
    assert count_frames(tmp_path / "effect.mp4") == count_frames(tmp_path / "matte.mp4") == 5


def test_png_folder_is_read_in_name_order(tmp_path):
    for name, v in (("b.png", 200), ("a.png", 50)):
        cv2.imwrite(str(tmp_path / name), np.full((8, 8, 3), v, np.uint8))
    assert [int(f[0, 0, 0]) for f in read_ltx(tmp_path)] == [50, 200]


def test_matte_is_0_on_black_and_rises_with_brightness():
    lum = np.array([[0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5]], np.float32)
    m = luminance_matte(lum, 1.0)
    assert m[0, 0] == 0 and m[0, -1] == 1
    assert (np.diff(m[0]) >= 0).all() and 0 < m[0, 3] < 1
    assert not luminance_matte(np.zeros((4, 4), np.float32), 0.0).any()   # an all-black clip


def test_matte_thresholds_scale_with_the_clip():
    full = [blob_frame(1.0)] * 3
    half = [blob_frame(0.5)] * 3
    lum_full, lum_half = luminance(decode8(full[0])), luminance(decode8(half[0]))
    b_full, b_half = bright_level(full, MatteConfig(stride=1)), bright_level(half, MatteConfig(stride=1))
    assert b_half == pytest.approx(0.5 * b_full, rel=0.02)
    # same shape: the half-brightness clip's matte equals the full one (up to 8-bit quantisation of the input)
    np.testing.assert_allclose(luminance_matte(lum_half, b_half), luminance_matte(lum_full, b_full), atol=0.05)


def test_effect_dir_leaves_the_standin_files_untouched(tmp_path):
    """Ingest writes only under ltx/<name>/; the stand-in effect.mp4 and matte.mp4 beside it are not touched."""
    make = tmp_path / "make"
    make.mkdir()
    standin = {n: bytes([i]) * 100 for i, n in enumerate(("effect.mp4", "matte.mp4"))}
    for n, b in standin.items():
        (make / n).write_bytes(b)
    write_ingest(ingest([blob_frame()] * 3, 3, CANVAS), 24, make / "ltx" / "run")
    assert {n: (make / n).read_bytes() for n in standin} == standin
    assert all((make / "ltx" / "run" / n).exists() for n in ("effect.mp4", "matte.mp4"))
    assert len(read_all(make / "ltx" / "run" / "effect.mp4")) == 3


def test_effect_dir_gets_its_own_merge_folder():
    make, out = Path("make"), Path("merge")
    assert merge_dirs(make, out, "k", None) == (make / "k", out / "k")
    assert merge_dirs(make, out, "k", Path("x") / "ltx" / "test_01") == (Path("x") / "ltx" / "test_01",
                                                                      out / "k" / "test_01")
