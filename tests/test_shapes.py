"""Effect shapes (task 31): the ring, crystal and cube drawn by the layout's renderer and the stand-in, the cue's `shape`
and `spin_deg_per_s`, the make key, and that the sphere is exactly what it was."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient  # noqa: F401  (server tests import it the same way)

from conftest import make_cue
from services.cue import api
from services.common.canvas import plate_to_canvas
from services.cue.cue import SHAPES, check, cue_from_dict, cue_to_dict, min_version, shape_of
from services.make.layout import SHAPE_SPECS, render_layout, render_shape, shape_layer, spin_per_frame
from services.make.look import standin_look
from services.cue.make_key import make_key
from services.cue.route import entrance_gain, orb_gains, orb_path
from services.make.standin_effect import look_tint, prepare_look, render_standin, render_standin_shape

ROOT = Path(__file__).resolve().parents[1]
NEW_SHAPES = ("ring", "crystal", "cube")
SHEET = (576, 1024)


def mask_stats(mask: np.ndarray) -> tuple[float, float, float]:
    """Diameter of the smallest circle round the mask (> 0.5), and its coverage-weighted centre, pixel-edge coordinates."""
    ys, xs = np.nonzero(mask > 0.5)
    (_, _), r = cv2.minEnclosingCircle(np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float32))
    m = mask.astype(np.float64)
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]] + 0.5
    return 2 * r, float((m * xx).sum() / m.sum()), float((m * yy).sum() / m.sum())


# --------------------------------------------------------------------------- the sphere is untouched

def _sphere_snapshot() -> str:
    """Layout and stand-in frames of the sphere (with and without an entrance), hashed. Taken on the code before task 31."""
    path = [(f, 600 + 7 * f, 400 - 2 * f, 120 - 0.5 * f, 1.0) for f in range(30)]
    gains = [min(1.0, f / 10) for f in range(30)]
    h = hashlib.sha256()
    for g in (None, gains):
        for fr in render_layout(path, (0, 0, 1920, 1080), (1024, 576), gains=g):
            h.update(fr.tobytes())
    look, _ = prepare_look(standin_look())
    for g in (None, gains):
        effect, matte = render_standin(look, path, (0, 0, 1920, 1080), (1024, 576), gains=g)
        for a in effect + matte:
            h.update(a.tobytes())
    return h.hexdigest()


def test_the_sphere_layout_and_stand_in_are_byte_identical_to_before():
    assert _sphere_snapshot() == "7eebdea05704aba7a471a5bb335b686bdbfc0952517adb9b0137e77c07398636"


def test_a_sphere_cue_draws_through_the_shape_arguments_exactly_as_without(cue):
    path = orb_path(cue)[:6]
    plain = render_layout(path, cue.canvas_rect, cue.canvas_size)
    named = render_layout(path, cue.canvas_rect, cue.canvas_size, kind=shape_of(cue), spin=spin_per_frame(cue))
    assert all((a == b).all() for a, b in zip(plain, named)) and spin_per_frame(cue) is None


def test_the_sphere_make_keys_do_not_change(tmp_path):
    look = tmp_path / "look.png"
    look.write_bytes(b"look image bytes")
    key = make_key(make_cue(), look)
    assert key == "d5fa87152707"
    assert make_key(make_cue(shape="sphere"), look) == key                 # the default, written out
    assert make_key(make_cue(spin_deg_per_s=SHAPE_SPECS["sphere"].spin), look) == key  # the default spin, written out


def test_the_make_key_differs_per_shape_and_spin(tmp_path):
    look = tmp_path / "look.png"
    look.write_bytes(b"look image bytes")
    keys = {s: make_key(make_cue(shape=s), look) for s in SHAPES}
    assert len(set(keys.values())) == 4
    assert make_key(make_cue(shape="cube", spin_deg_per_s=SHAPE_SPECS["cube"].spin), look) == keys["cube"]
    assert make_key(make_cue(shape="cube", spin_deg_per_s=0), look) != keys["cube"]
    assert make_key(make_cue(spin_deg_per_s=0), look) != keys["sphere"]


# --------------------------------------------------------------------------- the cue

def test_old_cues_load_as_sphere_and_keep_their_version():
    d = cue_to_dict(make_cue(version="2.2"))
    assert "shape" not in d and "spin_deg_per_s" not in d
    c = cue_from_dict(d)
    assert shape_of(c) == "sphere" and c.version == "2.2" and cue_to_dict(c) == d
    assert api.normalize(d)["version"] == "2.2" and "shape" not in api.normalize(d)


def test_a_cue_that_sets_a_shape_is_written_as_2_5_and_round_trips():
    d = cue_to_dict(make_cue(version="2.3", shape="ring", spin_deg_per_s=45))
    out = api.normalize(d)
    assert out["version"] == "2.5" and out["shape"] == "ring" and out["spin_deg_per_s"] == 45
    assert api.schema_errors("cue", out) == [] and cue_to_dict(cue_from_dict(out)) == out
    assert min_version(make_cue(shape="cube", body_point=None)) == "2.5"
    both = api.normalize(cue_to_dict(make_cue(version="2.3", shape="cube", behaviour="hold", release_frame=None, route=[])))
    assert both["version"] == "2.5"


def test_a_bad_shape_is_shape_invalid():
    issues = [i for i in check(make_cue(shape="pyramid")) if i.code == "shape_invalid"]
    assert len(issues) == 1 and issues[0].severity == "error" and issues[0].message.startswith("shape:")
    assert [w for w in api.validate(cue_to_dict(make_cue(shape="pyramid"))) if w["code"] == "shape_invalid"]
    for s in SHAPES:
        assert not [i for i in check(make_cue(shape=s)) if i.code == "shape_invalid"]


# --------------------------------------------------------------------------- the renderer

@pytest.mark.parametrize("kind", NEW_SHAPES)
@pytest.mark.parametrize("spin", [0, 37, 90, 200])
def test_a_shape_is_where_the_cue_says_and_as_big(kind, spin):
    cx, cy, d = 400.3, 300.7, 150.0
    image, mask = render_shape(kind, SHEET, cx, cy, d, spin)
    assert image.max() > 0.1 and mask.max() == 1.0
    size, mx, my = mask_stats(mask)
    assert abs(mx - cx) < 0.5 and abs(my - cy) < 0.5, (mx - cx, my - cy)
    assert abs(size / d - 1) < 0.05, size
    assert image[mask == 0].max() == 0  # nothing is drawn outside the mask


@pytest.mark.parametrize("kind", NEW_SHAPES)
def test_a_shape_is_deterministic_and_spin_changes_it(kind):
    a = shape_layer(kind, SHEET, 300.0, 200.0, 120.0, 20.0)
    assert (a == shape_layer(kind, SHEET, 300.0, 200.0, 120.0, 20.0)).all()
    assert not (a == shape_layer(kind, SHEET, 300.0, 200.0, 120.0, 35.0)).all()
    assert a.dtype == np.float32 and a.min() >= 0 and a.max() <= 1


@pytest.mark.parametrize("kind", NEW_SHAPES)
def test_the_layout_turns_with_spin_and_stands_still_without(kind, cue):
    path = orb_path(cue)[:3]
    static = [(f, 600.0, 300.0, 110.0, 1.0) for f, *_ in path]
    spun = render_layout(static, cue.canvas_rect, cue.canvas_size, kind=kind, spin=4.0)
    still = render_layout(static, cue.canvas_rect, cue.canvas_size, kind=kind, spin=0.0)
    assert not (spun[0] == spun[1]).all()
    assert all((still[0] == fr).all() for fr in still)


def test_default_spins_come_from_the_table():
    assert {k: v.spin for k, v in SHAPE_SPECS.items()} == {"sphere": 288.0, "ring": 90.0, "crystal": 60.0, "cube": 60.0}
    assert spin_per_frame(make_cue(shape="ring")) == pytest.approx(90 / 24)
    assert spin_per_frame(make_cue(shape="cube", spin_deg_per_s=120)) == 5.0
    assert spin_per_frame(make_cue(spin_deg_per_s=0)) == 0.0   # a sphere with a spin of its own
    assert spin_per_frame(make_cue()) is None                  # a plain sphere: the layout's own 12 degrees per frame


def test_the_ring_is_a_tilted_torus_with_a_hole():
    image, mask = render_shape("ring", SHEET, 300.0, 200.0, 200.0, 0.0)
    assert mask[200, 300] == 0 and mask[200, 215] > 0.9          # open in the middle, solid in the tube
    ys, xs = np.nonzero(mask > 0.5)
    assert (ys.max() - ys.min()) < 0.65 * (xs.max() - xs.min())  # turned 60 degrees: about half as high as wide


def test_the_crystal_is_a_tall_octahedron():
    _, mask = render_shape("crystal", SHEET, 300.0, 200.0, 200.0, 0.0)
    ys, xs = np.nonzero(mask > 0.5)
    assert ys.max() - ys.min() + 1 == pytest.approx(200, abs=3) and xs.max() - xs.min() + 1 <= 0.6 * 200 + 2


@pytest.mark.parametrize("kind", NEW_SHAPES)
def test_entrance_growth_works_for_every_shape(kind):
    cue = make_cue(shape=kind, entrance=make_cue().entrance.__class__(0.5))
    path = orb_path(cue)
    gains = orb_gains(cue, path)
    frames = render_layout(path, cue.canvas_rect, cue.canvas_size, gains=gains, kind=kind, spin=spin_per_frame(cue))
    first = next(i for i, g in enumerate(gains) if g > 0)
    whole = next(i for i, g in enumerate(gains) if g >= 1)
    area = [int((fr[..., 0] > 0).sum()) for fr in frames]
    assert all(a == 0 for a in area[:first]) or gains[0] == 0
    grow = area[first:whole + 1]
    assert grow[-1] > 3 * max(grow[0], 1) and all(b >= a * 0.9 for a, b in zip(grow, grow[1:]))
    assert gains[whole] == 1.0 and entrance_gain(cue, path[whole][0]) == 1.0


def test_the_whole_shape_takes_the_size_of_every_route_point(cue):
    cue = replace(cue, shape="crystal")
    path = orb_path(cue)
    for s in (path[0], path[60], path[-1]):
        (x, y), d = plate_to_canvas([s[1], s[2]], s[3], cue.canvas_rect, cue.canvas_size)
        _, mask = render_shape("crystal", (cue.canvas_size[1], cue.canvas_size[0]), float(x), float(y), float(d), 10.0)
        size, mx, my = mask_stats(mask)
        assert abs(size - d) < max(0.05 * d, 2.5) and abs(mx - x) < 0.5 and abs(my - y) < 0.5  # a 13 px crystal loses its two tips (under half covered)


# --------------------------------------------------------------------------- the stand-in

@pytest.mark.parametrize("kind", NEW_SHAPES)
def test_the_shape_stand_in_shows_the_shape_in_the_layouts_place(kind, cue):
    look, _ = prepare_look(standin_look())
    tint = look_tint(look.light)
    assert tint.max() == 1.0 and tint[0] >= tint[2]            # the stand-in look is blue: B (first) >= R (last)
    path = [(f, 600.0 + f, 400.0, 160.0, 1.0) for f in range(3)]
    spin = 5.0
    effect, matte = render_standin_shape(kind, tint, path, cue.canvas_rect, cue.canvas_size, spin)
    layout = render_layout(path, cue.canvas_rect, cue.canvas_size, kind=kind, spin=spin)
    for i in range(3):
        lay, m = layout[i][..., 0] > 0, matte[i][..., 1] > 0
        assert (lay & m).sum() > 0.97 * (lay | m).sum()  # the matte is the layout's shape (they differ at the faint edge)
        assert effect[i].shape == (cue.canvas_size[1], cue.canvas_size[0], 3)
    assert not (matte[0] == matte[1]).all()


def test_the_sphere_stand_in_is_still_the_look_image_and_a_shape_is_not(cue):
    look, _ = prepare_look(standin_look())
    path = [(0, 600.0, 400.0, 160.0, 1.0)]
    sphere, _ = render_standin(look, path, cue.canvas_rect, cue.canvas_size)
    cube, _ = render_standin_shape("cube", look_tint(look.light), path, cue.canvas_rect, cue.canvas_size, 0.0)
    assert not (sphere[0] == cube[0]).all()


def test_the_stand_in_grows_from_nothing_with_the_entrance(cue):
    look, _ = prepare_look(standin_look())
    path = [(f, 600.0, 400.0, 160.0, 1.0) for f in range(3)]
    effect, matte = render_standin_shape("ring", look_tint(look.light), path, cue.canvas_rect, cue.canvas_size, 3.0,
                                         gains=[0.0, 0.5, 1.0])
    area = [int((m[..., 1] > 0).sum()) for m in matte]
    assert area[0] == 0 and 0 < area[1] < area[2]


# --------------------------------------------------------------------------- one real render (slow)

@pytest.mark.slow
def test_a_cube_stand_in_on_clip_71319(tmp_path):
    """hold_throw on the real clip: layout, stand-in and their sheets for a cube; the sheet goes to data/debug."""
    import time

    from services.make.pipeline import make_layout, make_standin_effect

    clip, look = ROOT / "data" / "input" / "throw_71319.mov", ROOT / "data" / "look" / "standin_orb.png"
    if not clip.is_file() or not look.is_file():
        pytest.skip("data/input/throw_71319.mov or data/look/standin_orb.png is not here")
    golden = json.loads((ROOT / "contract" / "golden" / "throw_71319.cue.json").read_text(encoding="utf-8"))
    track = json.loads((ROOT / "contract" / "golden" / "throw_71319.track.json").read_text(encoding="utf-8"))
    cue = api._cue(api.normalize({**golden, "shape": "cube", "version": "2.5"}), track)
    assert cue.shape == "cube" and cue.behaviour is None and not [i for i in check(cue) if i.severity == "error"]
    t0 = time.perf_counter()
    lay = make_layout(clip, cue, look, tmp_path)
    t1 = time.perf_counter()
    run = make_standin_effect(cue, look, tmp_path)
    t2 = time.perf_counter()
    print(f"\ncube on 71319: layout {t1 - t0:.0f} s, stand-in {t2 - t1:.0f} s, {run.frames} frames")
    assert lay.frames == run.frames == len(orb_path(cue))
    for name in ("layout.mp4", "effect.mp4", "matte.mp4", "layout_sheet.png", "standin_sheet.png"):
        assert (tmp_path / name).stat().st_size > 1000
    out = ROOT / "data" / "debug"
    out.mkdir(parents=True, exist_ok=True)
    sheet = cv2.imread(str(tmp_path / "standin_sheet.png"))
    cv2.imwrite(str(out / "task31_cube_standin_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(out / "task31_cube_layout_sheet.jpg"), cv2.imread(str(tmp_path / "layout_sheet.png")),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
