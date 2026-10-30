"""Canvas rect proposal and the plate <-> canvas maps, on synthetic orb paths."""
import numpy as np
import pytest

from services.common.canvas import CanvasConfig, canvas_scale, canvas_to_plate, plate_to_canvas, propose_canvas_rect

CANVAS, PLATE = (1024, 576), (1920, 1080)


def path_of(*discs):
    """Orb samples from (x, y, d) discs."""
    return [(i, float(x), float(y), float(d), 1.0) for i, (x, y, d) in enumerate(discs)]


def check_rect(rect, canvas=CANVAS, plate=PLATE):
    x, y, w, h = rect
    assert all(isinstance(v, int) for v in rect)
    assert 0 <= x and 0 <= y and x + w <= plate[0] and y + h <= plate[1]
    assert abs((w / h) / (canvas[0] / canvas[1]) - 1) <= 0.005


def contains(rect, disc, margin=0.0):
    x, y, w, h = rect
    cx, cy, d = disc
    r = d / 2 + margin
    return x <= cx - r and cx + r <= x + w and y <= cy - r and cy + r <= y + h


def test_rect_contains_every_disc_plus_the_margin_and_keeps_the_aspect():
    discs = [(800, 500, 110), (900, 450, 80), (1100, 420, 60)]
    rect = propose_canvas_rect(path_of(*discs), CANVAS, PLATE, margin=1.0)
    check_rect(rect)
    for d in discs:
        assert contains(rect, d, margin=1.0 * 110)
    assert rect[2] < PLATE[0]  # and it is a real crop


def test_margin_scales_with_the_largest_diameter():
    discs = [(900, 500, 100), (1000, 500, 40)]
    small = propose_canvas_rect(path_of(*discs), CANVAS, PLATE, margin=0.5)
    large = propose_canvas_rect(path_of(*discs), CANVAS, PLATE, margin=2.0)
    assert small[2] < large[2]
    for d in discs:
        assert contains(small, d, margin=0.5 * 100) and contains(large, d, margin=2.0 * 100)


def test_tall_box_is_grown_in_width_to_the_canvas_aspect_and_stays_centred():
    rect = propose_canvas_rect(path_of((900, 300, 50), (900, 800, 50)), CANVAS, PLATE, margin=0.0)
    check_rect(rect)
    assert rect[3] >= 550 and abs(rect[0] + rect[2] / 2 - 900) <= 1 and abs(rect[1] + rect[3] / 2 - 550) <= 1


def test_rect_is_never_smaller_than_half_the_canvas():
    rect = propose_canvas_rect(path_of((900, 500, 20)), CANVAS, PLATE, margin=1.0)  # a tiny orb
    check_rect(rect)
    assert rect[2] >= 512 and rect[3] >= 288
    assert abs(rect[0] + rect[2] / 2 - 900) <= 1 and abs(rect[1] + rect[3] / 2 - 500) <= 1
    cfg = CanvasConfig(max_magnify=4.0)  # the cap is a config
    assert propose_canvas_rect(path_of((900, 500, 20)), CANVAS, PLATE, 1.0, cfg)[2] >= 256


def test_rect_is_shifted_inside_the_plate_not_shrunk():
    rect = propose_canvas_rect(path_of((60, 60, 100)), CANVAS, PLATE, margin=1.0)  # near the top-left corner
    check_rect(rect)
    assert rect[:2] == (0, 0) and rect[2] >= 512
    assert contains(rect, (60, 60, 100))  # the disc itself is inside, the margin is lost at the edge
    right = propose_canvas_rect(path_of((1900, 1050, 100)), CANVAS, PLATE, margin=1.0)
    check_rect(right)
    assert right[0] + right[2] == PLATE[0] and right[1] + right[3] == PLATE[1]
    assert right[2] == propose_canvas_rect(path_of((60, 60, 100)), CANVAS, PLATE, margin=1.0)[2]  # same size, only moved


def test_rect_is_shrunk_only_when_larger_than_the_plate():
    rect = propose_canvas_rect(path_of((100, 100, 100), (1800, 980, 100)), CANVAS, PLATE, margin=2.0)
    check_rect(rect)
    assert rect == (0, 0, 1920, 1080)
    odd = propose_canvas_rect(path_of((100, 100, 100), (1800, 980, 100)), (1024, 640), PLATE, margin=2.0)
    check_rect(odd, canvas=(1024, 640))  # a different canvas aspect still fits the plate
    assert odd[3] == 1080 and odd[2] < 1920


@pytest.mark.parametrize("canvas", [(1024, 576), (1024, 640), (768, 576), (960, 640)])
def test_rect_is_whole_pixels_with_the_aspect_within_half_a_percent(canvas):
    rng = np.random.default_rng(3)
    for _ in range(40):
        cx, cy = rng.uniform(200, 1700), rng.uniform(150, 900)
        discs = [(cx + rng.uniform(-300, 300), cy + rng.uniform(-200, 200), rng.uniform(20, 140)) for _ in range(4)]
        rect = propose_canvas_rect(path_of(*discs), canvas, PLATE, margin=rng.uniform(0, 2))
        check_rect(rect, canvas)


def test_empty_path_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        propose_canvas_rect([], CANVAS, PLATE)


def test_mapping_round_trips_exactly():
    rect = (310, 120, 800, 450)
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 1920, size=(20, 2))
    d = rng.uniform(10, 150, size=20)
    cxy, cd = plate_to_canvas(xy, d, rect, CANVAS)
    pxy, pd = canvas_to_plate(cxy, cd, rect, CANVAS)
    np.testing.assert_allclose(pxy, xy, atol=1e-9)
    np.testing.assert_allclose(pd, d, atol=1e-9)
    one, od = plate_to_canvas((310, 120), 110, rect, CANVAS)  # also a single point; the rect corner is the origin
    np.testing.assert_allclose(one, (0, 0), atol=1e-12)
    assert od == pytest.approx(110 * 1024 / 800)
    np.testing.assert_allclose(canvas_to_plate((1024, 576), 0, rect, CANVAS)[0], (1110, 570), atol=1e-9)
    assert canvas_scale(rect, CANVAS) == pytest.approx(1.28)


def test_a_route_point_off_the_plate_is_clamped_inside_it():
    path = [(0, 1000.0, 500.0, 110.0, 1.0), (1, 170.0, 345.0, 100.0, 1.0), (2, -120.0, 300.0, 90.0, 1.0)]
    x, y, w, h = propose_canvas_rect(path, (1024, 576), (1920, 1080))
    assert x == 0 and y >= 0 and x + w <= 1920 and y + h <= 1080
