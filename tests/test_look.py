"""Place Look: find the orb, scale and move it onto the first layout ball."""
import cv2
import numpy as np
import pytest

from services.common.canvas import plate_to_canvas
from services.make.look import LookError, find_orb, place_look, standin_look

CANVAS = (1024, 576)
RECT = (200, 100, 1024, 576)


def disc_image(cx, cy, d, size=(300, 400), glow=True):
    """An orb on black at (cx, cy) (pixel-edge coordinates): bright core of diameter d with a faint glow."""
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    r = np.hypot(xx - cx, yy - cy)
    img = 255 * np.clip(d / 2 + 0.5 - r, 0, 1)
    if glow:
        img = np.maximum(img, 70 * np.exp(-0.5 * (r / (d * 0.8)) ** 2))
    return np.dstack([img, img * 0.9, img * 0.7]).clip(0, 255).astype(np.uint8)


def first_ball(x, y, d):
    return [(100, x, y, d, 1.0), (101, x + 10, y, d, 1.0)]


def test_find_orb_measures_centre_and_area_equivalent_diameter():
    orb = find_orb(disc_image(150.3, 120.6, 60.0))
    assert (orb.x, orb.y) == pytest.approx((150.3, 120.6), abs=0.3)
    assert orb.diameter == pytest.approx(60.0, abs=1.0)


def test_find_orb_takes_the_largest_blob():
    img = disc_image(150, 120, 60.0, glow=False) + disc_image(300, 60, 14.0, glow=False)
    assert find_orb(img).x == pytest.approx(150, abs=0.5)


@pytest.mark.parametrize("d_plate,look_d", [(110.0, 60.0), (40.0, 90.0), (25.0, 25.0)])
def test_placed_look_has_the_first_ball_centre_and_diameter_within_a_pixel(d_plate, look_d):
    look = disc_image(171.7, 133.2, look_d)
    path = first_ball(803.4, 391.9, d_plate)
    placed, warnings = place_look(look, path, RECT, CANVAS)
    assert warnings == []
    assert placed.shape == (576, 1024, 3) and placed.dtype == np.uint8
    (cx, cy), cd = plate_to_canvas([803.4, 391.9], d_plate, RECT, CANVAS)
    orb = find_orb(placed)
    assert (orb.x, orb.y) == pytest.approx((cx, cy), abs=1.0)
    assert orb.diameter == pytest.approx(float(cd), abs=1.0)
    assert placed[:5].sum() == 0 and placed[:, :5].sum() == 0              # black elsewhere


def test_standin_look_is_an_orb_on_black_that_places_correctly():
    look = standin_look()
    assert look.shape == (512, 512, 3) and look[0, 0].tolist() == [0, 0, 0]
    orb = find_orb(look)
    assert orb.x == pytest.approx(256, abs=0.5) and 100 < orb.diameter < 200
    placed, warnings = place_look(look, first_ball(900, 500, 110.0), RECT, CANVAS)
    assert warnings == []
    (cx, cy), cd = plate_to_canvas([900, 500], 110.0, RECT, CANVAS)
    p = find_orb(placed)
    assert (p.x, p.y, p.diameter) == pytest.approx((cx, cy, float(cd)), abs=1.0)
    glow = look[256, 256 + 130]                                           # in the glow: blue-ish tint (BGR order)
    assert glow[0] > glow[2] > 0


def centred_image_at(placed, x0, y0, w, h):
    """True if `placed` is black except a w x h block whose top left is (x0, y0)."""
    outside = placed.copy()
    outside[y0:y0 + h, x0:x0 + w] = 0
    return outside.sum() == 0


def test_empty_look_image_is_centred_whole_with_a_warning():
    look = np.full((256, 300, 3), 3, np.uint8)  # a faint floor, nothing like an orb
    placed, warnings = place_look(look, first_ball(900, 500, 110.0), RECT, CANVAS)
    assert placed.shape == (576, 1024, 3)
    assert len(warnings) == 1 and warnings[0].startswith("look:") and "empty" in warnings[0] and "centred" in warnings[0]
    assert (placed[(576 - 256) // 2:(576 + 256) // 2, (1024 - 300) // 2:(1024 + 300) // 2] == 3).all()
    assert centred_image_at(placed, (1024 - 300) // 2, (576 - 256) // 2, 300, 256)
    placed, warnings = place_look(np.zeros((256, 256, 3), np.uint8), first_ball(900, 500, 110.0), RECT, CANVAS)
    assert placed.shape == (576, 1024, 3) and warnings and placed.sum() == 0   # all black in, all black out


def test_orb_touching_the_border_is_centred_whole_with_a_warning():
    look = disc_image(20, 150, 60.0)  # cut off on the left
    placed, warnings = place_look(look, first_ball(900, 500, 110.0), RECT, CANVAS)
    assert len(warnings) == 1 and "border" in warnings[0]
    assert centred_image_at(placed, (1024 - 400) // 2, (576 - 300) // 2, 400, 300)   # unscaled, centred, whole
    assert (placed[(576 - 300) // 2:(576 + 300) // 2, (1024 - 400) // 2:(1024 + 400) // 2] == look).all()
    _, warnings = place_look(disc_image(200, 150, 400.0, size=(300, 400)), first_ball(900, 500, 110.0), RECT, CANVAS)
    assert warnings and "border" in warnings[0]


def test_find_orb_itself_still_raises_so_callers_can_decide():
    with pytest.raises(LookError, match="empty"):
        find_orb(np.zeros((64, 64, 3), np.uint8))
    with pytest.raises(LookError, match="border"):
        find_orb(disc_image(20, 150, 60.0))


def test_a_look_larger_than_the_canvas_is_shrunk_to_fit_when_centred():
    placed, warnings = place_look(np.full((1200, 1600, 3), 3, np.uint8), first_ball(900, 500, 110.0), RECT, CANVAS)
    assert warnings and placed.shape == (576, 1024, 3)
    assert (placed[:, 128:896] == 3).all() and placed[:, :128].sum() == 0 and placed[:, 896:].sum() == 0  # 768 x 576, centred


def test_grey_and_bgra_looks_are_accepted():
    grey = cv2.cvtColor(disc_image(150, 120, 60.0), cv2.COLOR_BGR2GRAY)
    bgra = cv2.cvtColor(disc_image(150, 120, 60.0), cv2.COLOR_BGR2BGRA)
    for img in (grey, bgra):
        assert place_look(img, first_ball(900, 500, 110.0), RECT, CANVAS)[0].shape == (576, 1024, 3)
