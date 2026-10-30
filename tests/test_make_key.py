"""make_key: which cue changes make Make run again, and which do not."""
from dataclasses import replace

import pytest

from conftest import make_cue
from services.cue.cue import Correction, Grade, RoutePoint, Vanish
from services.cue.make_key import make_key


@pytest.fixture
def look(tmp_path):
    p = tmp_path / "look.png"
    p.write_bytes(b"look image bytes")
    return p


def test_key_is_12_hex_characters_and_stable(look):
    k = make_key(make_cue(), look)
    assert len(k) == 12 and int(k, 16) >= 0
    assert make_key(make_cue(), look) == k


@pytest.mark.parametrize("changes", [
    dict(grade=Grade("#ff0000", 2.0, 0.1, 0.9, 0.5, 9)),   # every grade value
    dict(behind=[]),
    dict(vanish=Vanish(120, 125)),
    dict(appear_frame=41),
], ids=["grade", "behind", "vanish", "appear_frame"])
def test_merge_only_changes_keep_the_key(look, changes):
    assert make_key(make_cue(**changes), look) == make_key(make_cue(), look)


@pytest.mark.parametrize("changes", [
    dict(route=[RoutePoint(107, None, None, 110), RoutePoint(116, 701, 420, 60), RoutePoint(130, 420, 360, 25)]),
    dict(canvas_rect=(0, 0, 1536, 864)),
    dict(seed=43),
    dict(canvas_size=(1280, 704)),
    dict(release_frame=108),
    dict(diameter_px=111),
    dict(anchor_offset=(0, -71)),
    dict(corrections=[Correction(60, 1001, 600)]),
    dict(correction_falloff=6),
    dict(look="other_orb"),
], ids=["route", "canvas_rect", "seed", "canvas_size", "release_frame", "diameter_px", "anchor_offset",
        "corrections", "correction_falloff", "look"])
def test_make_inputs_change_the_key(look, changes):
    assert make_key(make_cue(**changes), look) != make_key(make_cue(), look)


def test_track_and_effect_range_change_the_key(look):
    cue = make_cue()
    moved = replace(cue.track[10], x=cue.track[10].x + 1)
    assert make_key(make_cue(track=[*cue.track[:10], moved, *cue.track[11:]]), look) != make_key(cue, look)
    assert make_key(make_cue(effect_range=type(cue.effect_range)(40, 153)), look) != make_key(cue, look)


def test_look_image_bytes_change_the_key(look, tmp_path):
    k = make_key(make_cue(), look)
    same = tmp_path / "copy.png"
    same.write_bytes(look.read_bytes())
    assert make_key(make_cue(), same) == k          # the name does not matter, the bytes do
    look.write_bytes(b"look image bytes!")
    assert make_key(make_cue(), look) != k
