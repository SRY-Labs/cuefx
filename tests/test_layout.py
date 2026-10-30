"""Layout video: one clean ball per frame where the path says (no motion-blur copies), and the right frame count."""
import shutil
from dataclasses import replace

import cv2
import numpy as np
import pytest

from services.cue import api
from services.common.canvas import plate_to_canvas
from services.cue.cue import cue_from_dict
from services.make.layout import LayoutConfig, ball_layer, count_frames, render_frame, render_layout, to_bgr8, write_layout
from services.cue.route import orb_gains, orb_path
from test_contract import golden

CANVAS = (256, 128)
RECT = (0, 0, 512, 256)      # 0.5 canvas px per plate px
CFG = LayoutConfig()


def measure(img, threshold=10):
    """Mask centroid and area-equivalent diameter of the ball in an 8-bit or float image (pixel-edge coordinates)."""
    g = img if img.ndim == 2 else img[..., 0]
    mask = g > (threshold if g.dtype == np.uint8 else threshold / 255)
    ys, xs = np.nonzero(mask)
    return (xs.mean() + 0.5, ys.mean() + 0.5), 2 * np.sqrt(mask.sum() / np.pi)


def samples(*pos_d):
    return [(100 + i, float(x), float(y), float(d), 1.0) for i, (x, y, d) in enumerate(pos_d)]


@pytest.mark.parametrize("cx,cy,d", [(100.0, 64.0, 60.0), (100.37, 64.81, 33.3), (41.5, 30.2, 20.0), (200.9, 90.1, 80.0)])
def test_ball_centroid_and_diameter_match_the_request_within_a_pixel(cx, cy, d):
    (mx, my), md = measure(to_bgr8(ball_layer((128, 256), cx, cy, d, 0.0)))
    assert (mx, my) == pytest.approx((cx, cy), abs=1.0)
    assert md == pytest.approx(d, abs=1.0)


def test_rendered_layout_frames_match_the_path_after_mapping():
    rect = (100, 50, 512, 256)
    path = samples((300, 150, 100), (305, 152, 98), (310, 155, 96))
    frames = render_layout(path, rect, CANVAS, CFG)
    assert len(frames) == 3
    for s, frame in zip(path, frames):
        (cx, cy), cd = plate_to_canvas([s[1], s[2]], s[3], rect, CANVAS)
        (mx, my), md = measure(frame)
        assert (mx, my) == pytest.approx((cx, cy), abs=1.0)
        assert md == pytest.approx(float(cd), abs=1.0)


def test_ball_is_grey_matte_lit_from_the_upper_left_with_a_soft_edge():
    img = ball_layer((128, 256), 128.0, 64.0, 80.0, 0.0)
    assert 0.4 < img.max() <= 0.6 * 1.1                                  # albedo about 0.6, stripes at most +8 %
    assert img[64 - 20, 128 - 20] > img[64 + 20, 128 + 20]               # upper left is brighter than lower right
    assert img[64 + 20, 128 + 20] > 0                                    # the shadow side is still visible
    assert img[0, 0] == 0 and img[64, 128 + 41] == 0                     # black outside
    yy, xx = np.mgrid[0:128, 0:256]
    dist = np.hypot(yy + 0.5 - 64.0, xx + 0.5 - 128.0)
    edge = img[(dist > 39.0) & (dist < 41.0)]
    assert ((edge > 0) & (edge < 0.05)).any()                            # partial coverage pixels: an anti-aliased edge
    bgr = to_bgr8(img)
    assert (bgr[..., 0] == bgr[..., 1]).all() and (bgr[..., 1] == bgr[..., 2]).all()  # grey


def test_stripes_rotate_and_are_faint():
    a = ball_layer((128, 256), 128.0, 64.0, 80.0, 0.0)
    b = ball_layer((128, 256), 128.0, 64.0, 80.0, 25.0)
    inside = (a > 0.05) & (b > 0.05)
    rel = np.abs(a[inside] - b[inside]) / a[inside]
    assert 0.04 < rel.max() < 0.2                                        # visible, but faint (+- 8 %)
    flat = LayoutConfig(stripe_amount=0.0)
    np.testing.assert_array_equal(ball_layer((128, 256), 128.0, 64.0, 80.0, 0.0, flat),
                                  ball_layer((128, 256), 128.0, 64.0, 80.0, 25.0, flat))
    cpath = np.array([[128.0, 64.0, 80.0]] * 3)
    f0, f1 = (render_frame(cpath, i, (256, 128), CFG) for i in (0, 1))
    assert not np.array_equal(f0, f1)                                    # the spin advances per frame


def register(img8):
    """The brief's registration on a layout frame: threshold (any light) -> largest 8-connected blob -> centroid and
    diameter, in canvas pixel-edge coordinates; plus the number of blobs and the blob's bounding box (w, h).

    The centroid is of the blob's pixels, not light-weighted: the shading would pull a weighted one towards the light.
    The ball's light reaches r + 0.5 (its 1 px anti-aliased edge), so the diameter is the blob's area-equivalent
    diameter minus that 1 px.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats((img8[..., 1] > 0).astype(np.uint8), connectivity=8)
    if n == 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == k)
    return ((xs.mean() + 0.5, ys.mean() + 0.5), 2 * np.sqrt(len(xs) / np.pi) - 1.0, n - 1,
            (int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT])))


def blob_counts(img8, fractions=(0.0, 0.1, 0.25, 0.5, 0.75)):
    """8-connected blob count at each threshold (a fraction of the frame's peak): ghost copies split apart at some."""
    g = img8[..., 1]
    return [cv2.connectedComponents((g > f * g.max()).astype(np.uint8), connectivity=8)[0] - 1 for f in fractions]


def assert_one_clean_ball(img8, d):
    """One blob at every threshold, and no longer than the ball is wide: no motion-blur copies, no trail."""
    assert blob_counts(img8) == [1] * 5
    w, h = register(img8)[3]
    assert w <= np.ceil(d) + 2 and h <= np.ceil(d) + 2


def test_fast_motion_draws_one_sharp_ball_per_frame():
    """However far the ball moves to the next frame (here up to 3 diameters), each frame is the single ball at its own
    sample: what ball_layer draws there, nothing smeared towards the next frame."""
    d = 30.0
    cpath = np.array([[30.0 + 3 * d * i, 40.0 + 20 * (i % 3) + 0.3 * i, d] for i in range(6)])
    path = samples(*(2 * cpath))                                          # 0.5 canvas px per plate px
    frames = render_layout(path, (0, 0, 1024, 256), (512, 128), CFG)
    for i, frame in enumerate(frames):
        x, y, _ = cpath[i]
        np.testing.assert_array_equal(frame, to_bgr8(ball_layer((128, 512), x, y, d, CFG.spin_deg_per_frame * i, CFG)))
        assert_one_clean_ball(frame, d)
        (mx, my), md, _, _ = register(frame)
        assert (mx, my) == pytest.approx((x, y), abs=0.5) and md == pytest.approx(d, abs=0.5)


@pytest.fixture(scope="module")
def golden_layout():
    """The golden cue (with the golden track) through the calls make_layout makes, and its preview."""
    cue_d, track = golden("throw_71319.cue.json"), golden("throw_71319.track.json")
    cue = replace(cue_from_dict(cue_d), track=api.points_from_track(track))
    path = orb_path(cue)
    frames = render_layout(path, cue.canvas_rect, cue.canvas_size, gains=orb_gains(cue, path))
    return cue, path, frames, api.preview_path(cue_d, track)


def test_golden_layout_ball_is_on_the_preview_path_within_half_a_pixel(golden_layout):
    """Every frame the preview calls visible: the registered blob's centre and diameter are the preview's (mapped
    to the canvas) within 0.5 canvas px. Frames whose ball is not wholly on the canvas cannot be measured whole:
    there the blob must touch the edge, or the frame is black."""
    cue, path, frames, preview = golden_layout
    cw, ch = cue.canvas_size
    measured = 0
    for s, frame in zip(path, frames):
        p = preview[s[0]]
        if not p["visible"]:
            assert not frame.any()
            continue
        (cx, cy), cd = plate_to_canvas([p["x"], p["y"]], p["diameter"], cue.canvas_rect, cue.canvas_size)
        r = float(cd) / 2
        reg = register(frame)
        if cx - r < 0 or cy - r < 0 or cx + r > cw or cy + r > ch:              # leaving the canvas (frames 111, 112)
            assert reg is None if cx + r < 0 else reg[2] == 1 and reg[3][0] < 2 * r
            continue
        (mx, my), md, n, _ = reg
        assert n == 1, s[0]
        assert (mx, my) == pytest.approx((cx, cy), abs=0.5), s[0]
        assert md == pytest.approx(2 * r, abs=0.5 if 2 * r >= 2 else 1.0), s[0]  # under 2 px (frame 29, gain 0.02) the grid is too coarse
        measured += 1
    assert measured >= 80


def test_golden_layout_has_one_ball_and_no_ghosts_on_the_throw(golden_layout):
    cue, path, frames, _ = golden_layout
    at = {s[0]: i for i, s in enumerate(path)}
    for f in range(100, 116):
        if f not in at or not frames[at[f]].any():                               # 112 on: off the canvas; 115: past the window
            assert f >= 112, f
            continue
        cd = float(plate_to_canvas([0, 0], path[at[f]][3], cue.canvas_rect, cue.canvas_size)[1])
        if f == 111:                                                             # cut by the left edge: one blob, no trail
            assert blob_counts(frames[at[f]]) == [1] * 5 and register(frames[at[f]])[3][0] < cd
        else:
            assert_one_clean_ball(frames[at[f]], cd)


def test_every_frame_has_a_ball_whatever_its_visibility():
    path = [(i, 128.0 * 2, 64.0 * 2, 40.0, 0.0) for i in range(4)]       # visibility 0 everywhere
    frames = render_layout(path, RECT, CANVAS, CFG)
    assert all(f.max() > 100 for f in frames)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("n", [1, 9, 25])
def test_video_has_exactly_one_frame_per_sample_and_pngs_match(tmp_path, n):
    path = samples(*[(100 + 8 * i, 100 + 3 * i, 60) for i in range(n)])
    frames = write_layout(path, RECT, CANVAS, 24.0, tmp_path, CFG)
    assert len(frames) == n and count_frames(tmp_path / "layout.mp4") == n
    assert len(list((tmp_path / "layout_png").glob("*.png"))) == n
    cap = cv2.VideoCapture(str(tmp_path / "layout.mp4"))
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(24.0)
    ok, first = cap.read()
    cap.release()
    assert first.shape[:2] == (128, 256)
    (cx, cy), cd = plate_to_canvas([path[0][1], path[0][2]], path[0][3], RECT, CANVAS)
    assert measure(first, threshold=10)[0] == pytest.approx((cx, cy), abs=1.0)   # the decoded frame still has it in place


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_rewriting_removes_stale_pngs(tmp_path):
    write_layout(samples(*[(100, 100, 60)] * 5), RECT, CANVAS, 24.0, tmp_path, CFG)
    write_layout(samples(*[(100, 100, 60)] * 3), RECT, CANVAS, 24.0, tmp_path, CFG)
    assert len(list((tmp_path / "layout_png").glob("*.png"))) == 3
