"""Entrance (task 15a): the cue blocks, the ramp in the layout / stand-in / Merge, the effect window, the make key,
and the segment maths."""
import json
from dataclasses import replace

import numpy as np
import pytest

from conftest import make_cue, with_lost_gap
from services.cue.cue import (EffectRange, Entrance, Interaction, TrackPoint, cue_from_dict, cue_to_dict, entrance_frames,
                       entrance_start, load_cue, save_cue, validate)
from services.make.interaction import (SETTLE_PREROLL, SegmentError, last_visible, padded_length,
                               plan_for_cue, plan_segment)
from services.make.layout import render_layout
from services.make.look import standin_look
from services.cue.make_key import make_key
from services.delivery.merge import merge_frame
from services.cue.route import entrance_gain, first_full, orb_gains, orb_path, smoothstep, visibility
from services.make.standin_effect import prepare_look, render_standin
from services.cue.window import cover_from, effect_window
from golden_seconds0 import EXAMPLE, FIXTURE, example_cue, fingerprint


def entrance_cue(seconds=0.5, fps=24, **changes):
    return make_cue(entrance=Entrance(seconds), fps=fps, **changes)


# --------------------------------------------------------------------------- schema

def test_missing_blocks_get_the_defaults_and_old_cues_stay_valid():
    d = cue_to_dict(make_cue())
    del d["entrance"]
    cue = cue_from_dict(d)
    assert cue.entrance is None and cue.interaction is None
    assert entrance_frames(cue) == 12 and entrance_start(cue) == 28            # 0.5 s x 24 fps
    plan = plan_for_cue(cue, last=120, frames=200)
    assert (plan.K, plan.T, plan.P) == (6, 6, SETTLE_PREROLL)                 # 0.25 s each
    assert validate(cue)[0] == []
    assert "entrance" not in cue_to_dict(cue) and "interaction" not in cue_to_dict(cue)  # saved as it came


def test_blocks_round_trip(tmp_path):
    cue = make_cue(entrance=Entrance(0.75), interaction=Interaction(0.5, 0.125))
    save_cue(cue, tmp_path / "c.json")
    back = load_cue(tmp_path / "c.json")
    assert back.entrance == Entrance(0.75) and back.interaction == Interaction(0.5, 0.125)
    d = json.loads((tmp_path / "c.json").read_text())
    assert d["entrance"] == {"seconds": 0.75}
    assert d["interaction"] == {"crossfade_seconds": 0.5, "exit_tail_seconds": 0.125}
    assert cue_from_dict({**d, "interaction": {"crossfade_seconds": 0.1}}).interaction == Interaction(0.1, 0.25)


@pytest.mark.parametrize("changes,path", [
    (dict(entrance=Entrance(-0.1)), "entrance.seconds"),
    (dict(entrance=Entrance("half")), "entrance.seconds"),
    (dict(interaction=Interaction(-1, 0.25)), "interaction.crossfade_seconds"),
    (dict(interaction=Interaction(0.25, -0.5)), "interaction.exit_tail_seconds"),
])
def test_bad_values_are_errors_with_their_field_path(changes, path):
    errors, _ = validate(make_cue(**changes))
    assert any(e.startswith(path + ":") for e in errors), errors


@pytest.mark.parametrize("block,msg", [({"entrance": {"secs": 1}}, "entrance.secs"),
                                       ({"interaction": {"crossfade": 1}}, "interaction.crossfade"),
                                       ({"entrance": 0.5}, "entrance")])
def test_unknown_fields_name_their_path(block, msg):
    with pytest.raises(ValueError, match=msg.replace(".", r"\.")):
        cue_from_dict({**cue_to_dict(make_cue()), **block})


def test_the_example_cue_has_a_half_second_entrance():
    assert json.loads(EXAMPLE.read_text())["entrance"] == {"seconds": 0.5}
    assert example_cue().entrance == Entrance(0.5)


def test_entrance_longer_than_appear_frame_is_cut_with_a_warning():
    cue = entrance_cue(2.0)                                                   # 48 frames, appear_frame 40
    assert entrance_frames(cue) == 40 and entrance_start(cue) == 0
    assert any(w.startswith("entrance.seconds:") for w in validate(cue)[1])


def test_seconds_0_renders_byte_identical_to_before_the_entrance():
    """The example cue with entrance.seconds = 0, through the same calls the scripts make, against fingerprints
    taken on the code before the entrance existed (tests/golden_seconds0.py; the throw frames re-recorded in 05a),
    and against the same calls without gains."""
    cue = replace(example_cue(), entrance=Entrance(0.0))
    path = orb_path(cue)
    gains = orb_gains(cue, path)
    assert set(gains) == {1.0} and first_full(gains) == 0
    layout = render_layout(path, cue.canvas_rect, cue.canvas_size, gains=gains)
    look, _ = prepare_look(standin_look())
    effect, matte = render_standin(look, path, cue.canvas_rect, cue.canvas_size, gains=gains)
    no_gains = (render_layout(path, cue.canvas_rect, cue.canvas_size),
                *render_standin(look, path, cue.canvas_rect, cue.canvas_size))
    for got, want in zip((layout, effect, matte), no_gains):
        assert all(np.array_equal(a, b) for a, b in zip(got, want))

    def mf(f, plate8, e8, m8, vis, d):
        return merge_frame(f, plate8, e8, m8, vis, cue.grade, cue.canvas_rect, cue.canvas_size, [], d).out8

    assert fingerprint(cue, layout, effect, matte, path, mf) == json.loads(FIXTURE.read_text())


# --------------------------------------------------------------------------- ramp

@pytest.mark.parametrize("fps", [24, 25, 30])
@pytest.mark.parametrize("seconds", [0.25, 0.5, 1.0])
def test_gain_rises_smoothly_from_0_at_c_to_1_at_a(fps, seconds):
    cue = entrance_cue(seconds, fps)
    a, n = cue.appear_frame, entrance_frames(cue)
    c = a - n
    assert n == round(seconds * fps) and c == entrance_start(cue)
    g = [entrance_gain(cue, f) for f in range(c - 3, a + 3)]
    assert g[:4] == [0.0] * 4                                                 # nothing before C, and 0 at C
    assert g[-3:] == [1.0] * 3 and entrance_gain(cue, a) == 1.0
    steps = np.diff(g)
    assert (steps >= 0).all() and steps.max() <= 1.5 / n + 1e-9               # smoothstep's steepest slope is 1.5
    assert all(visibility(cue, f) == 0 for f in range(c))                     # Merge shows nothing before C
    assert visibility(cue, c) > 0 and visibility(cue, cue.appear_frame - 1) == 1


def _measure(img):
    g = img[..., 1] if img.ndim == 3 else img
    on = g > 6
    return 2 * np.sqrt(on.sum() / np.pi), int(g.max())


@pytest.mark.parametrize("fps", [24, 25, 30])
@pytest.mark.parametrize("seconds", [0.25, 0.5, 1.0])
def test_layout_and_standin_grow_in_size_and_brightness_over_the_entrance(fps, seconds):
    cue = entrance_cue(seconds, fps)
    path = orb_path(cue)
    gains = orb_gains(cue, path)
    at = {s[0]: i for i, s in enumerate(path)}
    c, a = entrance_start(cue), cue.appear_frame
    assert path[0][0] <= c                                                    # the window covers the entrance
    layout = render_layout(path, cue.canvas_rect, cue.canvas_size, gains=gains)
    full_d = _measure(layout[at[a]])[0]
    scale = cue.canvas_size[0] / cue.canvas_rect[2]
    assert full_d == pytest.approx(cue.diameter_px * scale, abs=1.5)
    assert all(not layout[i].any() for i in range(at[c] + 1))                 # black up to and including C
    ramp = range(c + 1, a + 1)
    ds = [_measure(layout[at[f]])[0] for f in ramp]
    light = [float(layout[at[f]].astype(float).sum()) for f in ramp]
    assert all(d1 >= d0 - 1.0 for d0, d1 in zip(ds, ds[1:]))                  # grows (pixel rounding aside)
    assert all(l1 > l0 for l0, l1 in zip(light, light[1:]))                   # and its light rises every frame
    # brightness: the same frames drawn at the grown size but without the brightness gain carry 1 / g as much light
    sized = [(s[0], s[1], s[2], s[3] * g, s[4]) for s, g in zip(path, gains)]
    unlit = render_layout(sized, cue.canvas_rect, cue.canvas_size)
    for f in ramp:
        g = entrance_gain(cue, f)
        if g * full_d >= 8:                                                   # big enough to measure
            assert _measure(layout[at[f]])[0] == pytest.approx(g * full_d, abs=2.0)
            ratio = layout[at[f]].astype(float).sum() / unlit[at[f]].astype(float).sum()
            assert ratio == pytest.approx(g, abs=0.02)
    look, _ = prepare_look(standin_look())
    pick = [at[c - 1] if c - 1 in at else at[c], at[c], at[c + entrance_frames(cue) // 2], at[a]]
    sub = [path[i] for i in pick]
    effect, matte = render_standin(look, sub, cue.canvas_rect, cue.canvas_size, gains=[gains[i] for i in pick])
    floor = effect[0].max()
    assert not matte[0].any() and not matte[1].any() and effect[1].max() == floor  # only the black floor before / at C
    mid, full = matte[2].astype(float).sum(), matte[3].astype(float).sum()
    g = gains[pick[2]]
    assert 0 < mid < full and mid / full == pytest.approx(g ** 3, rel=0.25)  # area x g^2, opacity x g
    assert effect[2].max() < effect[3].max()


def test_entrance_off_draws_the_ball_in_every_frame(cue):
    path = orb_path(cue)
    assert set(orb_gains(cue, path)) == {1.0} and visibility(cue, 39) == 0 and visibility(cue, 40) > 0
    assert path[0][0] == cue.effect_range.start                               # the window is not grown


def test_lost_track_in_the_entrance_holds_the_appear_frame_position_and_warns():
    track = with_lost_gap(make_cue().track, 30, 36, jump=-300.0)
    cue = entrance_cue(0.5, track=track)
    path = {s[0]: s for s in orb_path(cue)}
    for f in range(30, 36):
        assert path[f][1:3] == path[40][1:3]
    assert path[29][1:3] != path[40][1:3]                                     # tracked frames still follow the hand
    assert any("lost during the entrance" in w for w in validate(cue)[1])


# --------------------------------------------------------------------------- window and make key

def test_window_grows_by_whole_8_frame_steps_to_cover_the_entrance():
    er = EffectRange(32, 153)
    assert cover_from(er, 40, 200) == er
    assert cover_from(er, 28, 200) == EffectRange(24, 153)
    assert cover_from(er, 24, 200) == EffectRange(24, 153)
    assert cover_from(EffectRange(10, 59), 2, 200) == EffectRange(2, 59)
    assert cover_from(EffectRange(10, 59), 1, 200) == EffectRange(0, 65)     # would start at -6: start 0, end later
    assert cover_from(EffectRange(10, 59), 1, 60) == EffectRange(0, 57)      # and cut to the clip, still 8k+1
    w = effect_window(entrance_cue(0.5))
    assert w == EffectRange(24, 153) and (w.length - 1) % 8 == 0              # 129 frames: under the 193 limit, not trimmed
    assert any("starts after the entrance start" in m for m in validate(entrance_cue(0.5))[1])
    assert not any("is more than" in m for m in validate(entrance_cue(0.5))[1])


def test_make_key_follows_appear_frame_and_entrance_only_while_the_entrance_is_on(tmp_path):
    look = tmp_path / "look.png"
    look.write_bytes(b"look")
    on = entrance_cue(0.5)
    assert make_key(replace(on, appear_frame=41), look) != make_key(on, look)
    assert make_key(replace(on, entrance=Entrance(0.25)), look) != make_key(on, look)
    off = make_cue()
    assert make_key(replace(off, appear_frame=41), look) == make_key(off, look)
    assert make_key(on, look) != make_key(off, look)
    d = cue_to_dict(on)
    del d["entrance"]                                                         # the 0.5 s default draws the same
    assert make_key(cue_from_dict(d), look) == make_key(on, look)


# --------------------------------------------------------------------------- segment maths

def test_segment_values():
    p = plan_segment(appear=40, n_in=12, crossfade=6, preroll=8, last=111, tail=6, frames=193)
    assert (p.C, p.S, p.End, p.length, p.pin_frame_in_segment) == (28, 14, 123, 110, 26)
    assert p.padded_length == 113 and p.warnings == []
    assert padded_length(110) == 113 and padded_length(113) == 113 and padded_length(1) == 1
    assert last_visible([0, 0, 5, 3, 0, 0]) == 3 and last_visible([0, 0]) == -1


@pytest.mark.parametrize("appear,want", [
    (24, dict(P=6, K=6, N_in=12, S=0)),     # 2 short: P shrinks
    (15, dict(P=0, K=3, N_in=12, S=0)),     # 11 short: P to 0, then K
    (9, dict(P=0, K=0, N_in=9, S=0)),       # 17 short: P and K to 0, then N_in
])
def test_start_clamps_shrink_p_then_k_then_n_in_each_with_a_warning(appear, want):
    p = plan_segment(appear, 12, 6, 8, last=appear + 30, tail=6, frames=193)
    assert {k: getattr(p, k) for k in want} == want
    shrunk = [k for k, v in (("P", 8), ("K", 6), ("N_in", 12)) if want[k] < v]
    assert [w.split(":")[0] for w in p.warnings] == shrunk
    assert p.pin_frame_in_segment == appear - p.S == appear


def test_end_past_the_clip_is_clamped_with_a_warning():
    p = plan_segment(40, 12, 6, 8, last=185, tail=6, frames=193)
    assert p.End == 192 and any(w.startswith("End:") for w in p.warnings)


def test_an_over_long_segment_is_an_error_not_a_trim():
    with pytest.raises(SegmentError, match="more than max_segment_frames 100"):
        plan_segment(40, 12, 6, 8, last=111, tail=6, frames=193, max_len=100)
    assert plan_segment(40, 12, 6, 8, last=111, tail=6, frames=193, max_len=110).length == 110


def test_plan_for_cue_reads_the_cue_and_fps():
    cue = entrance_cue(0.5, fps=30, interaction=Interaction(0.5, 0.2))
    p = plan_for_cue(cue, last=120, frames=200)
    assert (p.N_in, p.K, p.T, p.P) == (15, 15, 6, 8)
    assert (p.C, p.S, p.End) == (25, 2, 141)
