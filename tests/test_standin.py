"""Stand-in effect and matte: placed on the ball, black floor, solid core, one orb per frame like the layout;
and the ffmpeg reader / writer they go through."""
import shutil
import subprocess

import numpy as np
import pytest

from services.common.canvas import plate_to_canvas
from services.common.colour import decode8, luminance
from services.make.layout import count_frames, render_layout, write_video
from services.make.look import find_orb, standin_look
from services.make.standin_effect import StandinConfig, prepare_look, render_standin, write_standin
from services.delivery.video import BT709_OUT, VideoWriter, add_audio, probe, read_all

CANVAS = (256, 128)
RECT = (0, 0, 512, 256)  # 0.5 canvas px per plate px
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def samples(*pos_d):
    return [(100 + i, float(x), float(y), float(d), 1.0) for i, (x, y, d) in enumerate(pos_d)]


def centroid(img8):
    lum = luminance(decode8(img8)) if img8.ndim == 3 else img8.astype(np.float32)
    lum = np.maximum(lum - np.percentile(lum, 5), 0)
    ys, xs = np.mgrid[0:lum.shape[0], 0:lum.shape[1]] + 0.5
    return (lum * xs).sum() / lum.sum(), (lum * ys).sum() / lum.sum()


@pytest.fixture(scope="module")
def look():
    return prepare_look(standin_look())[0]


def test_effect_sits_on_the_ball_with_its_size(look):
    path = samples((200, 120, 30), (204, 121, 30), (208, 124, 20))  # small and slow: no clipping, no blur
    effect, matte = render_standin(look, path, RECT, CANVAS)
    assert len(effect) == len(matte) == 3
    for s, e in zip(path, effect):
        (cx, cy), cd = plate_to_canvas([s[1], s[2]], s[3], RECT, CANVAS)
        assert e.shape == (128, 256, 3) and e.dtype == np.uint8
        assert centroid(e) == pytest.approx((cx, cy), abs=0.5)
    # the look's orb (Otsu blob) is scaled to the ball's canvas diameter, as Place Look does
    assert find_orb(effect[2]).diameter == pytest.approx(float(plate_to_canvas([0, 0], 20, RECT, CANVAS)[1]), abs=1.5)


def test_black_floor_is_002_linear_everywhere(look):
    effect, _ = render_standin(look, samples((200, 120, 60)), RECT, CANVAS)
    corner = decode8(effect[0][:5, :5])
    assert corner == pytest.approx(0.02, abs=0.002)                     # 8-bit: the nearest code to 0.02
    assert decode8(effect[0]).min() >= 0.018


def test_matte_is_the_solid_core_only(look):
    path = samples((200, 120, 40))
    effect, matte = render_standin(look, path, RECT, CANVAS)
    m = matte[0][..., 1].astype(np.float32) / 255
    assert (matte[0][..., 0] == matte[0][..., 2]).all()                  # grey
    assert m.max() == pytest.approx(1.0) and m[:10, :10].max() == 0
    (cx, cy), cd = plate_to_canvas([200, 120], 40, RECT, CANVAS)
    assert centroid(matte[0][..., 1]) == pytest.approx((cx, cy), abs=0.5)
    core_d = 2 * np.sqrt(m.sum() / np.pi)
    lum = luminance(look.light)
    look_core_d = 2 * np.sqrt((lum >= 0.5 * lum.max()).sum() / np.pi)  # the look's 50 %-of-peak core, scaled
    assert core_d == pytest.approx(look_core_d * float(cd) / look.orb.diameter, rel=0.05)
    edge = m[(m > 0.02) & (m < 0.98)]
    assert len(edge) > 0                                                # feathered, not a hard edge
    glow_only = luminance(decode8(effect[0])) > 0.03       # above the 0.02 floor
    assert (glow_only & (m == 0)).any()                                  # glow outside the core is not in the matte


def test_fast_frames_place_one_orb_like_the_layout_s_one_ball(look):
    """No motion-blur copies: a frame moving 4 diameters to the next is the look placed once at its own sample,
    exactly as if it were drawn alone, on the same canvas centre as the layout's ball."""
    path = samples((100, 120, 40), (102, 120, 40), (180, 120, 40), (260, 120, 40), (340, 140, 40))  # slow, then fast
    effect, matte = render_standin(look, path, RECT, CANVAS)
    layout = render_layout(path, RECT, CANVAS)
    for i, s in enumerate(path):
        single_e, single_m = render_standin(look, [s], RECT, CANVAS)
        np.testing.assert_array_equal(effect[i], single_e[0])
        np.testing.assert_array_equal(matte[i], single_m[0])
        assert matte[i][..., 1].max() == 255                             # the core is solid, not smeared
        (cx, cy), _ = plate_to_canvas([s[1], s[2]], s[3], RECT, CANVAS)
        assert centroid(effect[i]) == pytest.approx((cx, cy), abs=0.5)
        ys, xs = np.nonzero(layout[i][..., 1] > 0)
        assert (xs.mean() + 0.5, ys.mean() + 0.5) == pytest.approx((cx, cy), abs=0.5)  # where the layout's ball is


def test_the_whole_image_is_used_when_the_look_has_no_orb():
    img = np.full((64, 64, 3), 200, np.uint8)
    look, warnings = prepare_look(img)
    assert len(warnings) == 1 and warnings[0].startswith("look:")
    effect, matte = render_standin(look, samples((200, 120, 60)), RECT, CANVAS)
    assert effect[0].shape == (128, 256, 3)


@needs_ffmpeg
def test_written_files_have_one_frame_per_sample_at_canvas_size(look, tmp_path):
    path = samples(*[(150 + 6 * i, 120, 80) for i in range(9)])
    effect, matte = write_standin(look, path, RECT, CANVAS, 24.0, tmp_path)
    for name, frames in (("effect.mp4", effect), ("matte.mp4", matte)):
        assert count_frames(tmp_path / name) == 9
        info = probe(tmp_path / name)
        assert (info.width, info.height, info.fps) == (256, 128, 24.0)
        decoded = read_all(tmp_path / name)
        assert len(decoded) == 9
        assert np.abs(decoded[4].astype(int) - frames[4]).mean() < 1.0  # CRF 12: close to the frames written


@needs_ffmpeg
def test_reader_round_trips_untagged_and_rec709_files(tmp_path):
    rng = np.random.default_rng(3)
    flat = [np.full((64, 96, 3), c, np.uint8) for c in ((30, 120, 200), (200, 60, 90), (128, 128, 128))]
    noise = [np.clip(f.astype(int) + rng.integers(-3, 4, f.shape), 0, 255).astype(np.uint8) for f in flat]
    write_video(noise, tmp_path / "untagged.mp4", 24.0, 12)
    with VideoWriter(tmp_path / "bt709.mp4", (96, 64), 24.0, 12, BT709_OUT) as w:
        for f in noise:
            w.write(f)
    for name in ("untagged.mp4", "bt709.mp4"):
        back = read_all(tmp_path / name)
        assert len(back) == 3
        for a, b in zip(back, flat):
            err = np.abs(a.reshape(-1, 3).mean(0) - b[0, 0]).max()
            assert err <= (1.0 if b[0, 0, 0] == b[0, 0, 1] else 3.5), name  # grey exact; colour: 8-bit YUV steps


@needs_ffmpeg
def test_audio_is_carried_over_when_the_plate_has_some(tmp_path):
    src = tmp_path / "plate.mov"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=64x32:r=24:d=0.5",
                    "-f", "lavfi", "-i", "sine=f=440:d=0.5", "-c:v", "libx264", "-c:a", "pcm_s16le", "-shortest",
                    str(src)], check=True)
    assert probe(src).has_audio
    write_video([np.zeros((32, 64, 3), np.uint8)] * 12, tmp_path / "v.mp4", 24.0, 16)
    warnings = add_audio(tmp_path / "v.mp4", src, tmp_path / "final.mp4")
    assert probe(tmp_path / "final.mp4").has_audio
    assert all(w.startswith("audio:") for w in warnings)                # pcm cannot be copied into mp4: re-encoded
    assert not probe(tmp_path / "v.mp4").has_audio
