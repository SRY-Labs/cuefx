"""Merge maths on synthetic arrays: composite identities, black clamp, behind, grain, and the untouched plate."""
import numpy as np
import pytest

from conftest import make_cue
from services.common.canvas import canvas_to_plate
from services.common.colour import decode8, encode8, linear_to_srgb, luminance, srgb_to_linear
from services.cue.cue import Behind, Grade
from services.analysis.masks import OccluderTrack
from services.delivery.merge import (MergeConfig, black_clamp, black_floor, canvas_to_plate_image, composite, grain,
                         merge_frame, merge_frames, noise_std, restore_behind, sheet_frames, spill_light)

RNG = np.random.default_rng(7)
PLATE = (320, 180)                 # w, h
CANVAS = (128, 64)
RECT = (40, 30, 192, 96)           # 1.5 plate px per canvas px
GRADE = Grade("#9fd4ff", 1.0, 0.6, 0.4, 1.0, 3)


def rand(*shape):
    return RNG.random(shape, dtype=np.float32)


def orb_canvas(cx=64.0, cy=32.0, d=20.0, floor=0.02):
    """A linear canvas effect (blue-white disc with a glow) on a black floor, and its core matte."""
    h, w = CANVAS[1], CANVAS[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    r = np.hypot(xx - cx, yy - cy)
    core = np.clip(d / 2 + 0.5 - r, 0, 1)
    glow = 0.3 * np.exp(-0.5 * (r / (0.5 * d)) ** 2)
    light = np.maximum(core, glow)[..., None] * np.array([1.0, 0.9, 0.7], np.float32)
    return light + floor, core


# --------------------------------------------------------------------------- colour

def test_8bit_srgb_round_trips_exactly_through_linear():
    v = np.arange(256, dtype=np.uint8)
    np.testing.assert_array_equal(encode8(decode8(v)), v)
    assert linear_to_srgb(srgb_to_linear(np.linspace(0, 1, 101))) == pytest.approx(np.linspace(0, 1, 101), abs=1e-9)
    assert luminance(np.ones((1, 1, 3), np.float32))[0, 0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- composite identities

def test_no_effect_gives_the_plate_exactly():
    plate = rand(50, 60, 3)
    zero3, zero = np.zeros_like(plate), np.zeros(plate.shape[:2], np.float32)
    for g, o in ((0.6, 1.0), (1.0, 0.3), (2.0, 0.0)):
        np.testing.assert_array_equal(composite(plate, zero3, zero, g, o), plate)
    plate8 = (rand(PLATE[1], PLATE[0], 3) * 255).astype(np.uint8)
    out = merge_frame(100, plate8, np.zeros((64, 128, 3), np.uint8), np.zeros((64, 128, 3), np.uint8), 1.0, GRADE,
                      RECT, CANVAS, [rand(PLATE[1], PLATE[0])], 20.0)
    np.testing.assert_array_equal(out.out8, plate8)                     # through every step, behind mask included


def test_no_matte_and_full_glow_adds_the_effect():
    plate, E = rand(50, 60, 3), rand(50, 60, 3)
    np.testing.assert_array_equal(composite(plate, E, np.zeros((50, 60), np.float32), 1.0, 1.0), plate + E)


def test_full_matte_and_full_opacity_gives_the_effect():
    plate, E = rand(50, 60, 3), rand(50, 60, 3)
    for g in (0.0, 0.6, 1.0):
        np.testing.assert_array_equal(composite(plate, E, np.ones((50, 60), np.float32), g, 1.0), E)


def test_full_glow_never_darkens_an_edge_twice():
    plate, E, m = rand(80, 90, 3), rand(80, 90, 3), rand(80, 90)
    out = composite(plate, E, m, 1.0, 1.0)
    assert (out >= np.maximum(plate * (1 - m[..., None]), E) - 1e-6).all()


# --------------------------------------------------------------------------- black clamp

def test_black_clamp_turns_a_002_floor_into_0():
    E, _ = orb_canvas(floor=0.02)
    floor = black_floor(E)
    assert floor == pytest.approx(0.02, abs=1e-6)
    clamped = black_clamp(E, floor)
    assert clamped[0, 0].tolist() == [0, 0, 0] and clamped.min() == 0
    assert clamped[32, 64, 0] == pytest.approx(1.0, abs=1e-6)
    e8 = encode8(E)                                                     # and after an 8-bit round trip
    assert black_clamp(decode8(e8), black_floor(decode8(e8)))[0, 0].tolist() == [0, 0, 0]


# --------------------------------------------------------------------------- canvas to plate

def test_canvas_maps_onto_canvas_rect_and_nothing_outside():
    E, _ = orb_canvas(50.0, 20.0, 12.0, floor=0.0)
    lum = luminance(canvas_to_plate_image(E, RECT, CANVAS, PLATE))
    (px, py), _ = canvas_to_plate([50.0, 20.0], 0, RECT, CANVAS)
    ys, xs = np.mgrid[0:PLATE[1], 0:PLATE[0]] + 0.5
    cx, cy = (lum * xs).sum() / lum.sum(), (lum * ys).sum() / lum.sum()
    assert (cx, cy) == pytest.approx((px, py), abs=0.1)
    x, y, w, h = RECT
    inside = np.zeros(lum.shape, bool)
    inside[y:y + h, x:x + w] = True
    ones = canvas_to_plate_image(np.ones((64, 128), np.float32), RECT, CANVAS, PLATE)
    assert (ones[~inside] == 0).all() and ones[y + 1:y + h - 1, x + 1:x + w - 1].min() == pytest.approx(1.0)


# --------------------------------------------------------------------------- behind

def test_behind_restores_the_plate_with_a_faint_spill():
    plate, E, m = rand(40, 40, 3), rand(40, 40, 3), rand(40, 40)
    out = composite(plate, E, m, 0.6, 1.0)
    spill = spill_light(E, 10.0, 0.6, 0.15)
    np.testing.assert_allclose(restore_behind(out, plate, spill, np.ones((40, 40), np.float32)), plate + spill, atol=1e-6)
    np.testing.assert_array_equal(restore_behind(out, plate, spill, np.zeros((40, 40), np.float32)), out)


def test_spill_is_soft_light_with_no_ring():
    E, _ = orb_canvas(64.0, 32.0, 24.0, floor=0.0)                      # a crisp disc: a ring would show its edge
    d, g, s = 24.0, 0.6, 0.15
    spill = spill_light(E, d, g, s)
    sigma = 0.3 * d
    steepest = s * g * float(E.max()) / (sigma * np.sqrt(2 * np.pi))   # a Gaussian-blurred step is never steeper
    grad = max(np.abs(np.diff(spill, axis=a)).max() for a in (0, 1))
    assert grad <= steepest * 1.05
    assert np.abs(np.diff(E, axis=1)).max() > 10 * steepest              # the effect itself has a crisp edge
    assert spill.sum() == pytest.approx(s * g * E.sum(), rel=0.02)       # blur moves light, it does not lose it
    lum = luminance(spill)
    assert lum[32, 64] == lum.max()                                      # brightest at the orb's centre, not a ring
    assert (spill_light(np.zeros_like(E), d, g, s) == 0).all()


# --------------------------------------------------------------------------- grain

def test_noise_std_measures_the_plate_noise():
    yy, xx = np.mgrid[0:400, 0:600].astype(np.float32)
    smooth = (0.2 + 0.5 * xx / 600 + 0.1 * np.sin(yy / 40))[..., None].repeat(3, axis=2)
    noisy = smooth + np.random.default_rng(1).normal(0, [0.01, 0.02, 0.005], smooth.shape).astype(np.float32)
    assert noise_std(noisy) == pytest.approx([0.01, 0.02, 0.005], rel=0.08)
    assert (noise_std(smooth) < 1e-4).all()


def test_grain_is_deterministic_per_frame_and_zero_where_there_is_no_effect():
    weight = np.zeros((60, 80), np.float32)
    weight[20:40, 30:50] = 0.5
    std = np.array([0.01, 0.02, 0.03], np.float32)
    a, b = grain(std, weight, 0, 12), grain(std, weight, 0, 12)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, grain(std, weight, 0, 13))           # another frame, other grain
    assert (a[weight == 0] == 0).all() and np.abs(a[weight > 0]).max() > 0
    assert (grain(std, np.zeros((60, 80), np.float32), 0, 12) == 0).all()


def test_merging_a_frame_twice_gives_the_same_bytes():
    E, core = orb_canvas()
    plate8 = (rand(PLATE[1], PLATE[0], 3) * 255).astype(np.uint8)
    e8, m8 = encode8(E), np.rint(core * 255).astype(np.uint8)
    a = merge_frame(90, plate8, e8, m8, 1.0, GRADE, RECT, CANVAS, [rand(PLATE[1], PLATE[0])], 30.0)
    b = merge_frame(90, plate8, e8, m8, 1.0, GRADE, RECT, CANVAS, [a.masks[0]], 30.0)
    np.testing.assert_array_equal(a.out8, b.out8)
    assert not np.array_equal(a.out8, plate8)


# --------------------------------------------------------------------------- the whole clip, on arrays

def test_plate_is_untouched_outside_the_effect_range_and_outside_canvas_rect():
    cue = make_cue(plate_size=PLATE, canvas_size=CANVAS, canvas_rect=RECT, frames=12, grade=GRADE,
                   behind=[Behind(5, 7, "left_hand")])
    path = [(f, 0.0, 0.0, 0.0, 1.0) for f in range(3, 9)]              # effect range 3..8, fully visible
    plates = [(rand(PLATE[1], PLATE[0], 3) * 255).astype(np.uint8) for _ in range(12)]
    E, core = orb_canvas(cx=10.0, cy=8.0, d=30.0)                       # the glow reaches the canvas edge
    e8, m8 = encode8(E), np.rint(core * 255).astype(np.uint8)
    # a big hand right over the rect's corner, so its mask spills outside canvas_rect
    hand = np.array([[40 + 30 * np.cos(a), 30 + 30 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 21)])
    tracks = [OccluderTrack(0, cue.behind[0], {5: [hand], 6: [hand]})]
    out = list(merge_frames(cue, path, plates, iter([e8] * 6), iter([m8] * 6), tracks))
    assert [fr.f for fr in out] == list(range(12))
    x, y, w, h = RECT
    outside = np.ones((PLATE[1], PLATE[0]), bool)
    outside[y:y + h, x:x + w] = False
    for fr, plate8 in zip(out, plates):
        if not 3 <= fr.f < 9:
            assert fr.out8 is plate8                                    # copied, not recomputed
        else:
            assert fr.masks if fr.f in (5, 6) else not fr.masks
            np.testing.assert_array_equal(fr.out8[outside], plate8[outside])
            assert not np.array_equal(fr.out8[~outside], plate8[~outside])


def test_sheet_frames_fall_back_to_12_evenly_spaced_frames():
    path = [(f, 0.0, 0.0, 0.0, 1.0) for f in range(33, 122)]
    assert sheet_frames([60, 70], 193, path) == [60, 70]
    fallback = sheet_frames([60, 300], 193, path)
    assert len(fallback) == 12 and fallback[0] == 33 and fallback[-1] == 121
