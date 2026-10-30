"""Task 36 part C: cue v2.6 `ending` (fade | burst | shatter): the cue, normalize, Merge's visibility, the layout, the
stand-in, the Make key and the manual export hint."""
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import make_cue
from services.cue import api
from services.cue.cue import (ENDING_MIN_SECONDS, Transform, Vanish, cue_from_dict, cue_to_dict, ending_frames, ending_of, min_version,
                       vanish_of)
from services.make.effects import (SHARD_COUNT, SHARD_MIN_SIZE, burst_factors, confine, ending_plan, render_shards, shard_polygon, shards,
                           shatter_alpha)
from services.make.layout import render_layout, spin_per_frame
from services.make.look import standin_look
from services.cue.make_key import make_key
from services.cue.route import orb_gains, orb_path, visibility
from services.make.standin_effect import look_tint, prepare_look, render_standin, render_standin_shape
from services.delivery.manual import ending_hint

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "contract" / "golden"


def ended(kind, **changes):
    """The synthetic cue with an ending and a vanish long enough for it."""
    need = round(ENDING_MIN_SECONDS[kind] * 24 + 0.49)
    return make_cue(ending=kind, vanish=Vanish(128, 128 + need), **changes)


def layout_of(cue, kind="sphere"):
    path = orb_path(cue)
    frames = render_layout(path, cue.canvas_rect, cue.canvas_size, gains=orb_gains(cue, path), kind=kind,
                           spin=spin_per_frame(cue), ending=ending_plan(cue, path))
    return path, {s[0]: f[..., 0].astype(np.float32) for s, f in zip(path, frames)}


# --------------------------------------------------------------------------- the cue

def test_ending_is_fade_when_missing_and_the_file_is_unchanged():
    d = json.loads((GOLDEN / "throw_71319.cue.json").read_text(encoding="utf-8"))
    cue = cue_from_dict(d)
    assert cue.ending is None and ending_of(cue) == "fade" and ending_frames(cue) == 0 and vanish_of(cue) == cue.vanish
    assert cue_to_dict(cue) == d and "ending" not in cue_to_dict(cue)


def test_a_non_fade_ending_makes_the_cue_2_6_and_a_fade_does_not():
    d = api.normalize(cue_to_dict(make_cue()))
    assert d["version"] == "2.3"
    assert api.normalize({**d, "ending": "fade"})["version"] == "2.3"                 # setting the default changes nothing
    for kind in ("burst", "shatter"):
        out = api.normalize({**d, "ending": kind})
        assert out["version"] == "2.6" and out["ending"] == kind
        assert api.schema_errors("cue", out) == []
    old = api.normalize({**d, "version": "2.2", "ending": "burst"})
    assert old["version"] == "2.6"
    assert min_version(make_cue(ending="shatter")) == "2.6" and min_version(make_cue(ending="fade")) is None


def test_a_bad_ending_is_an_error():
    w = [w for w in api.validate(cue_to_dict(make_cue(ending="explode"))) if w["code"] == "ending_invalid"]
    assert len(w) == 1 and w[0]["severity"] == "error" and w[0]["cause"] == {"kind": "field", "path": "ending"}
    assert api.schema_errors("warning", w[0]) == []


def test_the_ending_needs_its_seconds_in_frames():
    assert ending_frames(make_cue(ending="burst")) == 8      # ceil(0.3 s x 24)
    assert ending_frames(make_cue(ending="shatter")) == 12   # 0.5 s x 24
    assert ending_frames(make_cue(ending="burst", fps=30)) == 9
    assert vanish_of(make_cue(ending="shatter")) == Vanish(128, 140)
    assert vanish_of(make_cue(ending="shatter", vanish=Vanish(128, 150))) == Vanish(128, 150)  # long enough: as stored


# --------------------------------------------------------------------------- normalize and validate

def starter(**changes):
    """A normalized cue dict whose vanish is 3 frames long, with `changes` put on it afterwards (not normalized)."""
    return {**api.normalize(cue_to_dict(make_cue())), **changes}


def test_normalize_lengthens_a_short_vanish_and_the_window_grows_with_it():
    d = starter(ending="shatter")
    out, warnings = api.normalize_with_warnings(d)
    assert out["vanish"] == {"start": 128, "end": 140}
    assert out["effect_range"]["end"] >= 140 and out["effect_range"]["start"] <= 40 - 4
    assert [w["code"] for w in warnings] == ["vanish_extended"]
    assert warnings[0]["frames"] == [131, 140] and warnings[0]["severity"] == "warning"
    assert "A shatter needs at least 0.5 s" in warnings[0]["message"] and api.schema_errors("warning", warnings[0]) == []
    assert not [w for w in api.validate(out) if w["severity"] == "error"]
    assert "vanish_extended" not in {w["code"] for w in api.validate(out)}      # it has been done
    assert api.normalize(out) == out                                            # and normalize is stable


def test_validate_warns_before_normalize_has_run():
    d = starter(ending="burst")
    d["vanish"] = {"start": 128, "end": 131}
    got = [w for w in api.validate(d) if w["code"] == "vanish_extended"]
    assert len(got) == 1 and got[0]["frames"] == [131, 136] and got[0]["cause"] == {"kind": "timing"}


def test_a_long_enough_vanish_and_a_fade_are_left_alone():
    d = starter()
    assert api.normalize_with_warnings(d) == (d, [])
    long = starter(ending="burst", vanish={"start": 128, "end": 140})
    assert api.normalize_with_warnings(long)[0]["vanish"] == {"start": 128, "end": 140}
    assert api.normalize_with_warnings(long)[1] == []
    fade = starter(vanish={"start": 128, "end": 129})
    assert api.normalize(fade)["vanish"] == {"start": 128, "end": 129}           # a fade has no minimum


def test_the_8_second_cap_cuts_the_ending_and_says_window_trimmed():
    d = cue_to_dict(make_cue(appear_frame=10, frames=400))
    d["track"] = [{**p, "f": f} for f, p in zip(range(400), d["track"] * 2)]
    d.update(release_frame=190, ending="shatter", vanish={"start": 190, "end": 191},
             route=[{"f": 190, "x": None, "y": None, "diameter_px": 110, "easing_in": "linear"}], behind=[])
    out, warnings = api.normalize_with_warnings(d)
    assert out["vanish"]["end"] == out["effect_range"]["end"] and out["vanish"]["end"] < 190 + 12
    assert out["effect_range"]["end"] - out["effect_range"]["start"] <= 193
    assert [w["code"] for w in warnings] == ["window_trimmed"]
    assert not [w for w in api.validate(out) if w["severity"] == "error"], api.validate(out)
    assert "window_trimmed" in {w["code"] for w in api.validate(out)}                            # and it stays said


# --------------------------------------------------------------------------- Merge's fade

def test_merge_applies_no_fade_out_for_a_burst_or_shatter():
    fade = make_cue()
    assert [round(visibility(fade, f), 3) for f in (127, 128, 129, 130, 131)] == [1.0, 1.0, round(2 / 3, 3), round(1 / 3, 3), 0.0]
    for kind in ("burst", "shatter"):
        cue = ended(kind)
        end = vanish_of(cue).end
        assert [visibility(cue, f) for f in range(128, end)] == [1.0] * (end - 128)
        assert visibility(cue, end) == 0.0 and visibility(cue, end + 3) == 0.0
        assert visibility(cue, 5) == 0.0  # before the entrance: as before
    assert [s[4] for s in orb_path(fade)] == [visibility(fade, s[0]) for s in orb_path(fade)]


# --------------------------------------------------------------------------- the effect itself

def test_a_burst_grows_to_1_8_flashes_and_fades():
    assert burst_factors(0.0) == (1.0, 1.0, 1.0)
    ts = np.linspace(0, 0.999, 200)
    scale, light, alpha = (np.array(v) for v in zip(*[burst_factors(t) for t in ts]))
    assert scale.max() == pytest.approx(1.8, abs=0.01) and np.all(np.diff(scale) >= -1e-9)
    assert burst_factors(0.4)[0] == pytest.approx(1.8) and burst_factors(0.39)[0] > 1.79          # 1.8 x over the first 40 %
    assert burst_factors(0.2)[0] < 1.8 and burst_factors(0.6)[0] == pytest.approx(1.8)            # (and stays there while it fades)
    assert light.max() == pytest.approx(1.6, abs=0.01)                                            # 1.6 x the orb: near white
    assert np.all(np.diff(light[ts <= 0.4]) >= -1e-9) and burst_factors(0.4)[1] == pytest.approx(1.6)
    assert all(burst_factors(t)[1] == pytest.approx(1.6) and burst_factors(t)[2] == 1.0 for t in (0.4, 0.5, 0.6))  # held bright to 60 %
    assert np.all(np.diff(alpha) <= 1e-9) and alpha[-1] < 0.01 and alpha[0] == 1.0
    assert burst_factors(0.7)[2] < 1.0 and burst_factors(0.7)[1] < 1.6 and np.all(np.diff(light[ts >= 0.6]) < 0)  # then it fades
    assert 0.6 * burst_factors(0.5)[1] == pytest.approx(0.96)                                       # the layout's grey at the peak


def test_shards_are_deterministic_from_the_seed_and_big_enough():
    a, b, c = shards(42), shards(42), shards(7)
    assert a == b and a != c
    assert SHARD_COUNT[0] <= len(a) <= SHARD_COUNT[1]
    counts = {len(shards(s)) for s in range(60)}
    assert min(counts) >= 12 and max(counts) <= 20 and len(counts) > 3       # the count varies with the seed
    for seed in range(40):
        for sh in shards(seed):
            assert sh.size >= SHARD_MIN_SIZE
            assert max(np.hypot(*c) for c in sh.corners) == pytest.approx(1.0)       # the polygon touches its circle
    assert shards(None) == shards(0)
    angles = sorted(sh.angle % (2 * np.pi) for sh in a)
    assert max(np.diff(angles + [angles[0] + 2 * np.pi])) < 2.5 * 2 * np.pi / len(a)    # spread all round


def test_rendered_shards_are_not_specks():
    d = 100.0
    for k in (6, 8, 11):  # (in the first frames the shards still overlap in a cluster)
        grey, cover = render_shards((400, 400), 200, 150, d, k, 24, 42, 0.6, 0.2, (-0.5, -0.5, 0.7))
        n, _, stats, _ = cv2.connectedComponentsWithStats((cover > 0.5).astype(np.uint8))
        areas = stats[1:, cv2.CC_STAT_AREA]
        assert len(areas) >= 8 and cover.max() <= 1.0 and grey.max() <= 0.6 + 1e-6
        assert np.median(areas) >= 0.3 * (SHARD_MIN_SIZE * d / 2) ** 2   # a shard is a real polygon, not a pixel


def test_shards_fly_outward_fall_and_fade():
    cue = ended("shatter")
    path, frames = layout_of(cue)
    start, end = 128, vanish_of(cue).end
    assert frames[start].max() > 0 and frames[end].max() == 0 and frames[end + 3].max() == 0     # black once it is over
    spread = []
    for f in (start + 1, start + 5, end - 1):
        ys, xs = np.nonzero(frames[f] > 8)
        spread.append(np.hypot(xs - xs.mean(), ys - ys.mean()).mean() if len(xs) else 0.0)
    assert spread[0] < spread[1] < spread[2] or frames[end - 1].max() < 20                       # it opens up
    assert frames[end - 1].max() < frames[start + 2].max() * 0.5                                  # and fades


def test_the_layout_of_a_fade_is_what_it_always_was():
    cue = make_cue()
    path = orb_path(cue)
    gains = orb_gains(cue, path)
    plain = render_layout(path, cue.canvas_rect, cue.canvas_size, gains=gains)
    assert ending_plan(cue, path) is None
    again = render_layout(path, cue.canvas_rect, cue.canvas_size, gains=gains, ending=ending_plan(cue, path))
    assert all(np.array_equal(a, b) for a, b in zip(plain, again))


def test_a_burst_layout_grows_flashes_then_is_black():
    cue = ended("burst")
    path, frames = layout_of(cue)
    start, end = 128, vanish_of(cue).end
    plain = {s[0]: f[..., 0].astype(np.float32) for s, f in zip(path, render_layout(
        path, cue.canvas_rect, cue.canvas_size, gains=orb_gains(cue, path)))}
    assert np.array_equal(frames[start], plain[start]) and np.array_equal(frames[start - 5], plain[start - 5])
    area = {f: int((frames[f] > 10).sum()) for f in range(start, end + 2)}
    assert area[start + 3] > 1.8 * area[start]                                    # (1.8 x the diameter is 3.2 x the area)
    assert frames[start + 1].max() > plain[start + 1].max()                       # the flash
    assert frames[end].max() == 0 and frames[end + 1].max() == 0


def test_the_standin_draws_the_endings():
    look = prepare_look(standin_look())[0]
    for kind in ("burst", "shatter"):
        cue = ended(kind)
        path = orb_path(cue)
        gains = orb_gains(cue, path)
        start, end = 128 - path[0][0], vanish_of(cue).end - path[0][0]
        for shape in ("sphere", "crystal"):
            if shape == "sphere":
                e, m = render_standin(look, path, cue.canvas_rect, cue.canvas_size, gains=gains, ending=ending_plan(cue, path))
            else:
                e, m = render_standin_shape(shape, look_tint(look.light), path, cue.canvas_rect, cue.canvas_size,
                                            spin_per_frame(cue) or 2.5, gains=gains, ending=ending_plan(cue, path))
            assert len(e) == len(m) == len(path)
            cover = [int((x[..., 1] > 128).sum()) for x in m]
            assert cover[start] > 0 and cover[end] == 0 and cover[end + 1] == 0, (kind, shape)
            assert e[end].mean() <= e[0].mean() + 3                              # nothing left but the black floor
            if kind == "burst":
                assert cover[start + 3] > 1.5 * cover[start], shape
                assert e[start + 1].mean() > e[start].mean(), shape
            else:
                assert cover[start + 4] > 0, shape


# --------------------------------------------------------------------------- Make key, export hint

def test_the_make_key_changes_only_for_a_burst_or_shatter(tmp_path):
    look = tmp_path / "look.png"
    look.write_bytes(b"look image bytes")
    assert make_key(make_cue(), look) == make_key(make_cue(ending="fade"), look) == "d5fa87152707"
    burst, shatter = make_key(ended("burst"), look), make_key(ended("shatter"), look)
    assert len({"d5fa87152707", burst, shatter}) == 3
    assert make_key(ended("burst"), look) == burst                                   # stable
    assert make_key(replace(ended("shatter"), seed=7), look) != shatter              # the shards come from the seed
    assert make_key(replace(ended("burst"), vanish=Vanish(128, 140)), look) != burst  # the ending runs over the vanish
    assert make_key(make_cue(vanish=Vanish(128, 135)), look) == "d5fa87152707"       # a fade's vanish is still Merge's alone
    assert make_key(make_cue(ending="burst", vanish=Vanish(128, 130)), look) == burst  # a short vanish is the same as the minimum


def test_the_export_prompt_gets_one_hint_line_for_a_non_fade_ending():
    assert ending_hint("a glowing orb", make_cue()) == "a glowing orb"
    assert ending_hint("a glowing orb", make_cue(ending="fade")) == "a glowing orb"
    assert ending_hint("a glowing orb", make_cue(ending="burst")) == "a glowing orb\nAt the end the effect bursts in a bright flash."
    assert ending_hint("a glowing orb\n", make_cue(ending="shatter")).splitlines() == [
        "a glowing orb", "At the end the effect shatters into glowing shards."]
    assert ending_hint("  ", make_cue(ending="burst")) == "  "                      # no prompt stays no prompt


def test_preview_phase_follows_the_longer_vanish():
    out = api.normalize(starter(ending="shatter"))
    phases = {p["frame"]: p["phase"] for p in api.preview_path(out, api.track_from_points(cue_from_dict(out).track))}
    assert phases[128] == "vanish" and phases[139] == "vanish" and phases[140] == "after"


def test_a_burst_or_shatter_at_the_end_of_the_route_does_not_hold_still():
    """The orb sits where its path ends, but a burst or shatter starts there: only the frames before vanish.start count."""
    last = make_cue().route[-1].f  # 130
    assert "orb_holds_after_route" in {w["code"] for w in api.validate(cue_to_dict(make_cue(vanish=Vanish(last, last + 6))))}
    for kind in ("burst", "shatter"):
        at_end = make_cue(ending=kind, vanish=Vanish(last, last + 12))
        assert "orb_holds_after_route" not in {w["code"] for w in api.validate(cue_to_dict(at_end))}
        early = make_cue(ending=kind, vanish=Vanish(last + 4, last + 16))
        w = [w for w in api.validate(cue_to_dict(early)) if w["code"] == "orb_holds_after_route"]
        assert len(w) == 1 and w[0]["frames"] == [last + 1, last + 4]                        # holds still for 3 frames, then ends


# --------------------------------------------------------------------------- task 38: the ending is the whole vanish

def ended_for(kind, seconds):
    """The synthetic cue with a `seconds` long vanish (normalized: the window grows to hold it)."""
    n = round(seconds * 24)
    return cue_from_dict(api.normalize(cue_to_dict(make_cue(ending=kind, vanish=Vanish(128, 128 + n)))))


@pytest.mark.parametrize("kind", ("burst", "shatter"))
@pytest.mark.parametrize("seconds", (0.5, 1.0, 2.0))
def test_the_animation_covers_the_whole_vanish_whatever_its_length(kind, seconds):
    cue = ended_for(kind, seconds)
    n = round(seconds * 24)
    assert vanish_of(cue) == Vanish(128, 128 + n)                                  # long enough as it is: not touched
    path, frames = layout_of(cue)
    plan = ending_plan(cue, path)
    states = {s[0]: st for s, st in zip(path, plan.states)}
    assert all(states[f].kind == kind and states[f].length == n and states[f].k == f - 128 for f in range(128, 128 + n))
    assert states[127] is None and all(states[f].kind == "gone" for f in states if f >= 128 + n)   # (the window may end with the vanish)
    # the last frame of the ending is at the end of its progress, not early ...
    assert states[128 + n - 1].t == pytest.approx((n - 1) / n) and states[128 + n - 1].t > 0.9
    # ... and nothing is held empty before it: every frame up to the last couple has something drawn, then black
    for f in range(128, 128 + n - 2):
        assert frames[f].max() > 0, (kind, seconds, f)
    assert all(frames[f].max() == 0 for f in (128 + n, 128 + n + 2) if f in frames)      # (the window may end with the vanish)
    # the same progress looks the same at any length: the brightness (burst) and the fade (shatter) are stretched
    mid = frames[128 + n // 2]
    assert mid.max() > 0 and frames[128 + n - 2].max() < mid.max()


def test_a_longer_vanish_is_the_same_animation_stretched():
    for kind in ("burst", "shatter"):
        a, b = ended_for(kind, 1.0), ended_for(kind, 2.0)
        pa, pb = orb_path(a), orb_path(b)
        ta = [st.t for st in ending_plan(a, pa).states if st is not None and st.kind == kind]
        tb = [st.t for st in ending_plan(b, pb).states if st is not None and st.kind == kind]
        assert len(ta) == 24 and len(tb) == 48 and ta[0] == tb[0] == 0.0
        assert tb[2 * 12] == pytest.approx(ta[12]) and tb[-1] == pytest.approx(47 / 48)


def test_the_defaults_say_how_long_the_ending_lasts_when_picked():
    d = api.defaults()
    assert d["ending_default_seconds"] == {"burst": 0.6, "shatter": 1.0}
    assert d["ending_seconds"] == {"fade": 0.0, "burst": 0.3, "shatter": 0.5}      # the minimums, as before
    for kind, seconds in d["ending_default_seconds"].items():                      # a default is never shorter than the minimum
        assert seconds >= d["ending_seconds"][kind]


# --------------------------------------------------------------------------- task 38 part B: a stronger shatter and burst

def test_shards_are_at_least_a_quarter_of_the_effect():
    assert SHARD_MIN_SIZE == 0.25
    for seed in list(range(80)) + [None]:
        for sh in shards(seed):
            assert sh.size >= 0.25 * 0.999999
    d = 100.0
    for k in (8, 14, 20):
        grey, cover = render_shards((600, 600), 300, 250, d, k, 24, 42, 0.6, 0.2, (-0.5, -0.5, 0.7))
        _, _, stats, _ = cv2.connectedComponentsWithStats((cover > 0.5).astype(np.uint8))
        wide = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
        assert np.median(wide) >= 0.25 * d * 0.5                      # (a four-cornered shard is thinner than its bounding circle)


def test_shards_stay_fully_bright_for_half_the_vanish_then_fade():
    assert shatter_alpha(0.0) == shatter_alpha(0.25) == shatter_alpha(0.5) == 1.0
    ts = np.linspace(0.5, 0.999, 50)
    a = np.array([shatter_alpha(t) for t in ts])
    assert np.all(np.diff(a) < 0) and a[-1] < 0.01
    look = prepare_look(standin_look())[0]
    for seconds in (0.5, 1.0, 2.0):
        cue = ended_for("shatter", seconds)
        n = round(seconds * 24)
        path = orb_path(cue)
        _, matte = render_standin(look, path, cue.canvas_rect, cue.canvas_size, gains=orb_gains(cue, path),
                                  ending=ending_plan(cue, path))
        peak = {s[0]: int(m[..., 1].max()) for s, m in zip(path, matte)}
        for f in range(129, 128 + n // 2 + 1):                          # up to 50 % of the vanish: full brightness
            assert peak[f] == 255, (seconds, f)
        assert peak[128 + n // 2 + 2] < 255 and peak[128 + n - 2] < 0.5 * 255


def test_shards_stay_inside_the_picture_at_any_length():
    # the synthetic cue, and the same cue with 100 px of conform padding left and right
    cues = [ended_for("shatter", 1.0), ended_for("shatter", 2.0)]
    padded = ended_for("shatter", 2.0)
    padded.transform = Transform(1.0, (100, 0))
    cues.append(padded)
    for cue in cues:
        path = orb_path(cue)
        plan = ending_plan(cue, path)
        x0, y0, x1, y1 = plan.picture
        assert (x0, y0) == pytest.approx((0, 0) if cue.transform.offset[0] == 0 else (100 * 1024 / 1920, 0))
        assert x1 <= 1024 and y1 <= 576
        for s, st in zip(path, plan.states):
            if st is None or st.kind != "shatter" or st.k == 0:
                continue
            cx, cy, d = s[1] * 1024 / 1920, s[2] * 1024 / 1920, path[st.ref][3] * 1024 / 1920
            for sh in shards(cue.seed):
                poly = shard_polygon(sh, st.k / cue.fps, cx, cy, d, plan.picture)
                assert poly[:, 0].min() >= x0 - 1e-6 and poly[:, 0].max() <= x1 + 1e-6, (st.k, sh.angle)
                assert poly[:, 1].min() >= y0 - 1e-6 and poly[:, 1].max() <= y1 + 1e-6, (st.k, sh.angle)
    # and without bounds the same shards of a long shatter really would leave it (so the test above proves something)
    cue = cues[1]
    path = orb_path(cue)
    far = shard_polygon(max(shards(cue.seed), key=lambda sh: sh.speed), 2.0, 500.0, 300.0, 58.0)
    assert far[:, 0].max() > 1024 or far[:, 1].max() > 576 or far[:, 0].min() < 0 or far[:, 1].min() < 0


def test_confine_slows_a_shard_smoothly_and_never_passes_the_edge():
    b = (0.0, 0.0, 100.0, 100.0)
    assert confine(50, 50, 10, -5, 5, b) == (60, 45)                               # well inside: untouched
    assert confine(50, 50, 10, -5, 5, None) == (60, 45)
    xs = [confine(50, 50, t, 0, 5, b)[0] for t in np.linspace(0, 400, 81)]
    assert np.all(np.diff(xs) >= -1e-9) and max(xs) <= 95 and xs[-1] > 94.9 and xs[30] < xs[50]   # keeps creeping, never past the edge
    assert confine(95, 50, 30, 0, 5, b) == (95, 50)                                # no room on that side
    assert confine(2, 50, 0, 0, 5, b) == (5, 50)                                   # a centre too near the edge is moved in
    assert confine(50, 50, 20, 0, 60, b) == (50.0, 50.0)                           # a picture too small for it: its centre


def test_a_longer_shatter_drifts_further():
    reach = {}
    for seconds in (0.5, 1.0, 2.0):
        cue = ended_for("shatter", seconds)
        path, frames = layout_of(cue)
        n = round(seconds * 24)
        last = 128 + int(0.6 * n)                                                  # a frame in the fade
        ys, xs = np.nonzero(frames[last] > 4)
        reach[seconds] = float(np.hypot(xs - xs.mean(), ys - ys.mean()).mean())
    assert reach[0.5] < reach[1.0] < reach[2.0]
