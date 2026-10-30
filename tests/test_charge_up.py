"""Task 36 part D: behaviour "charge_up" (cue v2.6): grows 0.6 -> 1.0 and pulses 3.5 Hz +-8 % between appear and release,
then is a hold_throw."""
import math
from dataclasses import replace

import numpy as np
import pytest

from conftest import make_cue
from services.cue import api
from services.cue.cue import CHARGE_UP, Entrance, RoutePoint, cue_to_dict, no_flight
from services.make.layout import render_layout
from services.make.look import standin_look
from services.cue.make_key import make_key
from services.make.pipeline import _lights
from services.cue.route import charge_growth, charge_pulse, orb_gains, orb_lights, orb_path, smoothstep
from services.make.standin_effect import prepare_look, render_standin


def charge(**changes):
    """The synthetic cue as a charge-up: appear 40, release 107 (67 frames, 2.8 s)."""
    return make_cue(behaviour="charge_up", **changes)


def codes(cue) -> set[str]:
    return {w["code"] for w in api.validate(cue_to_dict(cue))}


# --------------------------------------------------------------------------- the cue

def test_charge_up_is_a_version_2_6_behaviour_with_a_release_and_a_route():
    cue = charge()
    assert not no_flight(cue)
    assert not [w for w in api.validate(cue_to_dict(cue)) if w["severity"] == "error"]
    d = api.normalize(cue_to_dict(cue))
    assert d["behaviour"] == "charge_up" and d["version"] == "2.6" and d["release_frame"] == 107 and len(d["route"]) == 3
    assert api.schema_errors("cue", d) == []
    assert "no_path" in codes(charge(route=[RoutePoint(107, None, None, 110)]))        # like hold_throw
    assert "hold_path_unused" not in codes(cue)
    broken = [w for w in api.validate(cue_to_dict(charge(release_frame=None))) if w["code"] == "missing_field"]
    assert broken and broken[0]["cause"]["path"] == "release_frame"                  # a charge_up needs its release
    assert "behaviour_invalid" not in codes(cue) and "behaviour_invalid" in codes(make_cue(behaviour="swing"))


def test_the_other_behaviours_are_not_charge_up():
    assert api.defaults()["charge_up"] == CHARGE_UP == {"start_scale": 0.6, "end_scale": 1.0, "pulse_hz": 3.5, "pulse_amount": 0.08}
    assert charge_growth(make_cue(), 60) == 1.0 and charge_pulse(make_cue(), 60) == 1.0
    assert orb_lights(make_cue(), orb_path(make_cue())) == [1.0] * len(orb_path(make_cue()))
    assert _lights(make_cue(), orb_path(make_cue())) is None


# --------------------------------------------------------------------------- the numbers

def test_the_size_grows_from_0_6_to_1_0_by_smoothstep():
    cue = charge()
    assert charge_growth(cue, 40) == pytest.approx(0.6) and charge_growth(cue, 20) == pytest.approx(0.6)   # also before appear
    assert charge_growth(cue, 107) == pytest.approx(1.0)
    mid = 40 + (107 - 40) / 2
    assert charge_growth(cue, round(mid)) == pytest.approx(0.6 + 0.4 * smoothstep((round(mid) - 40) / 67))
    g = [charge_growth(cue, f) for f in range(40, 108)]
    assert all(b >= a for a, b in zip(g, g[1:])) and g[1] - g[0] < 0.01        # flat at the start (smoothstep)


def test_the_pulse_is_3_5_hz_of_plus_minus_8_percent_between_appear_and_release():
    cue = charge()
    p = [charge_pulse(cue, f) for f in range(40, 107)]
    assert p[0] == 1.0 and max(p) == pytest.approx(1.08, abs=0.02) and min(p) == pytest.approx(0.92, abs=0.02)
    assert max(p) <= 1.08 and min(p) >= 0.92
    assert charge_pulse(cue, 39) == 1.0 and charge_pulse(cue, 107) == 1.0 and charge_pulse(cue, 130) == 1.0
    for f in (45, 60, 90):
        assert charge_pulse(cue, f) == pytest.approx(1 + 0.08 * math.sin(2 * math.pi * 3.5 * (f - 40) / 24))
    crossings = sum(1 for a, b in zip(p, p[1:]) if (a - 1) * (b - 1) < 0)
    assert 17 <= crossings <= 21                                                    # 3.5 Hz over 67 frames: 9.8 cycles


def test_held_frames_have_the_charged_size_and_the_flight_is_a_hold_throws():
    cue, plain = charge(), make_cue()
    path, ref = {s[0]: s for s in orb_path(cue)}, {s[0]: s for s in orb_path(plain)}
    assert path.keys() == ref.keys()
    for f in (50, 70, 100):
        assert path[f][3] == pytest.approx(110 * charge_growth(cue, f) * charge_pulse(cue, f))
        assert path[f][1:3] == ref[f][1:3] and path[f][4] == ref[f][4]       # same place, same visibility
    for f in range(107, 140):
        assert path[f] == ref[f]                                              # after the release: identical to hold_throw
    assert path[106][3] < 110 * 1.081


def test_lights_follow_the_pulse_and_the_entrance_is_left_alone():
    cue = charge(entrance=Entrance(0.5))
    path = orb_path(cue)
    assert orb_gains(cue, path) == orb_gains(make_cue(entrance=Entrance(0.5)), path)       # the entrance gain is the same
    assert orb_lights(cue, path) == [charge_pulse(cue, s[0]) for s in path]
    assert _lights(cue, path) is not None


# --------------------------------------------------------------------------- the layout and the stand-in

def test_the_layout_ball_grows_and_pulses_brightness():
    cue = charge()
    path = orb_path(cue)
    lights = orb_lights(cue, path)
    frames = render_layout(path, cue.canvas_rect, cue.canvas_size, gains=orb_gains(cue, path), lights=lights)
    f0 = path[0][0]
    area = {f: int((frames[f - f0][..., 0] > 5).sum()) for f in (41, 70, 100)}
    assert area[41] < area[70] < area[100]
    assert area[100] / area[41] == pytest.approx((charge_growth(cue, 100) * charge_pulse(cue, 100)
                                                 / (charge_growth(cue, 41) * charge_pulse(cue, 41))) ** 2, rel=0.1)
    peak = [float(frames[f - f0][..., 0].max()) for f in range(60, 80)]
    assert max(peak) > min(peak) * 1.05                                                   # the brightness pulses
    plain = render_layout(path, cue.canvas_rect, cue.canvas_size, gains=orb_gains(cue, path))
    assert all(np.array_equal(a, b) for a, b in zip(plain[:f0 and 40 - f0], frames[:40 - f0]))   # before appear: lights are 1
    assert not all(np.array_equal(a, b) for a, b in zip(plain[45 - f0:60 - f0], frames[45 - f0:60 - f0]))


def test_the_standin_light_pulses_and_its_matte_does_not():
    look = prepare_look(standin_look())[0]
    cue = charge()
    path = orb_path(cue)
    lights = orb_lights(cue, path)
    e, m = render_standin(look, path, cue.canvas_rect, cue.canvas_size, gains=orb_gains(cue, path), lights=lights)
    e0, m0 = render_standin(look, path, cue.canvas_rect, cue.canvas_size, gains=orb_gains(cue, path))
    f0 = path[0][0]
    pulsing = [i for i in range(len(path)) if abs(lights[i] - 1) > 0.03]
    assert len(pulsing) > 20
    for i in pulsing:
        assert np.array_equal(m[i], m0[i])
        got, base = float(e[i].astype(np.float32).mean()), float(e0[i].astype(np.float32).mean())
        assert (got > base) == (lights[i] > 1.0), (path[i][0], lights[i], got, base)


def test_the_make_key_has_the_behaviour_and_appear_frame(tmp_path):
    look = tmp_path / "look.png"
    look.write_bytes(b"look image bytes")
    plain = make_key(make_cue(), look)
    assert plain == "d5fa87152707"
    k = make_key(charge(), look)
    assert k != plain and k == make_key(charge(), look)
    assert make_key(charge(appear_frame=41), look) != k                                  # the growth starts at appear
    assert make_key(make_cue(appear_frame=41), look) == plain                           # hold_throw: as before (entrance off)
    assert k != make_key(make_cue(behaviour="hold", release_frame=None, route=[]), look)
