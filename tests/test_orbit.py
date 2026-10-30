"""Task 36 part E: behaviour "orbit" (cue v2.6): the effect circles its body point; front bigger and brighter, back smaller
and dimmer; no release and no route."""
import math
from dataclasses import replace

import numpy as np
import pytest

from conftest import PALM_W, make_cue
from services.cue import api
from services.cue.cue import ORBIT_DEFAULTS, Orbit, RoutePoint, cue_from_dict, cue_to_dict, min_version, no_flight, orbit_of
from services.make.layout import render_layout
from services.cue.make_key import make_key
from services.make.pipeline import _lights
from services.cue.route import ORBIT_BACK_LIGHT, ORBIT_SIZE_SWING, held_xy, orb_gains, orb_lights, orb_path, orbit_state

HELD = (1000.0, 600.0 - 70.0)  # the synthetic hand (1000, 600) plus its anchor_offset (0, -70)


def orbiting(**changes):
    """The synthetic cue as an orbit: appear 40, a 3-frame vanish at 128; no release, no route."""
    return make_cue(**{"behaviour": "orbit", "release_frame": None, "route": [], **changes})


def codes(cue) -> set[str]:
    return {w["code"] for w in api.validate(cue_to_dict(cue))}


def errors(cue) -> list[dict]:
    return [w for w in api.validate(cue_to_dict(cue)) if w["severity"] == "error"]


# --------------------------------------------------------------------------- the cue

def test_orbit_is_a_version_2_6_behaviour_without_release_and_route():
    cue = orbiting()
    assert no_flight(cue) and errors(cue) == [] and "no_path" not in codes(cue)
    d = api.normalize(cue_to_dict(replace(cue, release_frame=None, route=None)))
    assert d["behaviour"] == "orbit" and d["version"] == "2.6" and d["release_frame"] is None and d["route"] == []
    assert api.schema_errors("cue", d) == [] and api.normalize(d) == d
    assert min_version(cue) == "2.6" and min_version(make_cue(orbit=Orbit(2.0))) == "2.6"
    assert not [w for w in api.validate(d) if w["severity"] == "error"]


def test_missing_orbit_fields_take_the_defaults_and_the_block_round_trips():
    cue = cue_from_dict({**cue_to_dict(orbiting()), "orbit": {"radius_sizes": 2}})
    assert cue.orbit == Orbit(radius_sizes=2)                               # as written: a missing field stays missing
    assert orbit_of(cue) == Orbit(2, ORBIT_DEFAULTS["speed_deg_per_s"], ORBIT_DEFAULTS["tilt_deg"])
    assert orbit_of(orbiting()) == Orbit(1.5, 180, 60)                      # no block at all: the defaults
    assert cue_to_dict(cue)["orbit"] == {"radius_sizes": 2}
    assert "orbit" not in cue_to_dict(orbiting())
    assert api.schema_errors("cue", {**api.normalize(cue_to_dict(orbiting())), "orbit": {"radius_sizes": 1, "colour": 1}})


@pytest.mark.parametrize("block", [Orbit(0.0), Orbit(-1.0), Orbit(11.0), Orbit(speed_deg_per_s=4000), Orbit(speed_deg_per_s=-4000),
                                   Orbit(tilt_deg=-1), Orbit(tilt_deg=91), Orbit(speed_deg_per_s=float("nan"))])
def test_bad_orbit_values_are_orbit_invalid(block):
    w = [w for w in api.validate(cue_to_dict(orbiting(orbit=block))) if w["code"] == "orbit_invalid"]
    assert w and all(x["severity"] == "error" for x in w) and api.schema_errors("warning", w[0]) == []
    assert w[0]["cause"]["kind"] == "field" and w[0]["cause"]["path"].startswith("orbit.")


def test_good_orbit_values_pass():
    for block in (Orbit(), Orbit(0.1, -720, 0), Orbit(10, 3600, 90), Orbit(1.5, 0, 45)):
        assert "orbit_invalid" not in codes(orbiting(orbit=block))


def test_a_route_on_an_orbit_is_hold_path_unused_like_a_hold():
    got = [w for w in api.validate(cue_to_dict(orbiting(release_frame=107, route=[RoutePoint(107, None, None, 110)])))
           if w["code"] == "hold_path_unused"]
    assert len(got) == 1 and got[0]["severity"] == "warning" and "Orbit" in got[0]["message"]
    hold = [w for w in api.validate(cue_to_dict(make_cue(behaviour="hold"))) if w["code"] == "hold_path_unused"]
    assert hold[0]["message"] == "This effect is set to Hold, so the flight path is not used."


# --------------------------------------------------------------------------- the circle

def test_the_orbit_state_follows_the_angle():
    cue = orbiting(entrance=None)
    R = 1.5 * PALM_W
    dx, dy, size, light = orbit_state(cue, 40)                              # angle 0: right of the body point
    assert (dx, dy) == pytest.approx((R, 0.0)) and size == 1.0 and light == pytest.approx(0.75)
    dx, dy, size, light = orbit_state(cue, 52)                              # 0.5 s x 180 deg/s = 90 deg: the front
    assert (dx, dy) == pytest.approx((0.0, R * math.cos(math.radians(60))), abs=1e-9)
    assert dy == pytest.approx(R / 2) and size == pytest.approx(1 + ORBIT_SIZE_SWING) and light == pytest.approx(1.0)
    dx, dy, size, light = orbit_state(cue, 76)                              # 270 deg: the very back
    assert dx == pytest.approx(0.0, abs=1e-9) and dy == pytest.approx(-R / 2)
    assert size == pytest.approx(1 - ORBIT_SIZE_SWING) and light == pytest.approx(ORBIT_BACK_LIGHT)
    assert orbit_state(cue, 88)[:2] == pytest.approx(orbit_state(cue, 40)[:2])        # a turn takes 2 s = 48 frames


def test_speed_sign_radius_and_tilt():
    base = orbiting()
    backwards = orbiting(orbit=Orbit(speed_deg_per_s=-180))
    assert orbit_state(backwards, 52)[1] == pytest.approx(-orbit_state(base, 52)[1])
    assert orbit_state(orbiting(orbit=Orbit(radius_sizes=3.0)), 40)[0] == pytest.approx(2 * orbit_state(base, 40)[0])
    flat = orbiting(orbit=Orbit(tilt_deg=0))
    assert orbit_state(flat, 52)[1] == pytest.approx(1.5 * PALM_W)               # tilt 0: a full circle
    edge = orbiting(orbit=Orbit(tilt_deg=90))
    assert abs(orbit_state(edge, 52)[1]) < 1e-9                                  # tilt 90: a line
    still = orbiting(orbit=Orbit(speed_deg_per_s=0))
    assert orbit_state(still, 40) == orbit_state(still, 100)


def test_the_radius_is_in_size_measures_of_the_body_point():
    wide = orbiting(track=[replace(p, palm_w=2 * PALM_W) for p in make_cue().track])
    assert orbit_state(wide, 40)[0] == pytest.approx(2 * orbit_state(orbiting(), 40)[0])


def test_the_path_circles_the_held_position_at_full_size():
    cue = orbiting()
    path = orb_path(cue)
    xs, ys = np.array([s[1] for s in path]), np.array([s[2] for s in path])
    for s in path:
        dx, dy, size, _ = orbit_state(cue, s[0])
        assert (s[1], s[2], s[3]) == pytest.approx((HELD[0] + dx, HELD[1] + dy, 110 * size))
    full = [s for s in path if 40 <= s[0] < 88]                                  # one turn
    assert np.mean([s[1] for s in full]) == pytest.approx(HELD[0], abs=12) and np.mean([s[2] for s in full]) == pytest.approx(HELD[1], abs=12)
    assert max(s[3] for s in full) <= 110 * 1.1501 and min(s[3] for s in full) >= 110 * 0.8499
    assert (xs.max() - xs.min()) / 2 == pytest.approx(150, rel=0.02)               # radius 1.5 x 100
    assert held_xy(cue)[100] == pytest.approx(HELD)                                # the held marker stays on the body point
    assert [s[4] for s in path] == [s[4] for s in orb_path(make_cue(behaviour="hold", release_frame=None, route=[]))]


def test_the_anchor_offset_moves_the_whole_circle():
    a, b = orb_path(orbiting()), orb_path(orbiting(anchor_offset=(30, -10)))
    assert np.array([(y[1] - x[1], y[2] - x[2]) for x, y in zip(a, b)]) == pytest.approx(np.array([(30.0, 60.0)] * len(a)))


def test_the_lights_dim_the_back_and_every_other_behaviour_has_none():
    cue = orbiting()
    path = orb_path(cue)
    lights = orb_lights(cue, path)
    assert min(lights) == pytest.approx(0.5, abs=0.01) and max(lights) == pytest.approx(1.0, abs=0.01)
    assert lights == [orbit_state(cue, s[0])[3] for s in path] and _lights(cue, path) is not None
    assert _lights(make_cue(), orb_path(make_cue())) is None
    assert api.defaults()["orbit"] == {"radius_sizes": 1.5, "speed_deg_per_s": 180, "tilt_deg": 60}


def test_the_layout_ball_is_dimmer_and_smaller_at_the_back():
    cue = orbiting(diameter_px=200)
    path = orb_path(cue)
    frames = render_layout(path, cue.canvas_rect, cue.canvas_size, gains=orb_gains(cue, path), lights=orb_lights(cue, path))
    f0 = path[0][0]
    front, back = frames[52 - f0][..., 0].astype(float), frames[76 - f0][..., 0].astype(float)
    assert front.max() > 1.5 * back.max()                                              # 1.0 against 0.5 of the light
    assert (back > 5).sum() < (front > 5).sum()
    ys, xs = np.nonzero(front > 5)
    cx, cy = xs.mean() / 1024 * 1920, ys.mean() / 576 * 1080                           # canvas -> plate
    assert (cx, cy) == pytest.approx((HELD[0], HELD[1] + 75), abs=6)                   # R/2 = 75 below the held position


def test_the_make_key_has_the_circle_only_for_an_orbit(tmp_path):
    look = tmp_path / "look.png"
    look.write_bytes(b"look image bytes")
    k = make_key(orbiting(), look)
    assert k == make_key(orbiting(orbit=Orbit(1.5, 180, 60)), look) == make_key(orbiting(orbit=Orbit()), look)  # a default spelled out
    assert make_key(orbiting(orbit=Orbit(radius_sizes=2.0)), look) != k
    assert make_key(orbiting(orbit=Orbit(speed_deg_per_s=90)), look) != k
    assert make_key(orbiting(orbit=Orbit(tilt_deg=30)), look) != k
    assert make_key(orbiting(appear_frame=41), look) != k                              # the angle starts at appear_frame
    hold = make_cue(behaviour="hold", release_frame=None, route=[])
    assert k != make_key(hold, look) and make_key(replace(hold, orbit=Orbit(5.0)), look) == make_key(hold, look)
    assert make_key(make_cue(), look) == "d5fa87152707"


def test_preview_and_phase_of_an_orbit():
    d = api.normalize(cue_to_dict(replace(orbiting(), release_frame=None, route=None)))
    track = api.track_from_points(cue_from_dict(d).track)
    rows = api.preview_path(d, track, with_held=True)
    phases = {r["frame"]: r["phase"] for r in rows}
    assert phases[10] == "before" and phases[39] == "before"  # (the synthetic cue has no entrance)
    assert phases[60] == "held" and phases[d["vanish"]["start"]] == "vanish"
    assert all(r["held"]["x"] == pytest.approx(HELD[0]) for r in rows[40:100])
    assert rows[52]["y"] > rows[76]["y"]                                              # the front is the lower side


# --------------------------------------------------------------------------- the head's tilt (task 36 follow-up)

def test_the_default_tilt_is_80_for_the_head_and_60_for_every_other_part():
    from services.cue.cue import BodyPoint
    head = BodyPoint("head", None)
    assert orbit_of(orbiting(body_point=head)) == Orbit(1.5, 180, 80)
    for part, side in (("palm", "right"), ("fist", "left"), ("foot", "right"), ("knee", "left"), ("shoulder", "right")):
        assert orbit_of(orbiting(body_point=BodyPoint(part, side))).tilt_deg == 60
    assert orbit_of(orbiting()).tilt_deg == 60                                       # no body point: the palm
    assert orbit_of(orbiting(body_point=head, orbit=Orbit(tilt_deg=30))).tilt_deg == 30   # an explicit tilt wins
    assert orbit_of(orbiting(body_point=head, orbit=Orbit(radius_sizes=2.0))) == Orbit(2.0, 180, 80)
    assert api.defaults()["orbit_tilt_deg_by_part"] == {"head": 80} and api.defaults()["orbit"]["tilt_deg"] == 60


def test_the_head_circle_is_nearly_edge_on():
    from services.cue.cue import BodyPoint
    head, palm = orbiting(body_point=BodyPoint("head", None)), orbiting()
    assert orbit_state(head, 52)[1] == pytest.approx(1.5 * PALM_W * math.cos(math.radians(80)))
    assert orbit_state(palm, 52)[1] == pytest.approx(1.5 * PALM_W * 0.5)
    assert errors(head) == []


def test_the_head_tilt_is_in_the_make_key_as_its_resolved_value(tmp_path):
    from services.cue.cue import BodyPoint
    look = tmp_path / "look.png"
    look.write_bytes(b"look image bytes")
    head = BodyPoint("head", None)
    k = make_key(orbiting(body_point=head), look)
    assert k == make_key(orbiting(body_point=head, orbit=Orbit(1.5, 180, 80)), look)    # spelled out: the same
    assert k != make_key(orbiting(body_point=head, orbit=Orbit(1.5, 180, 60)), look)
    assert make_key(orbiting(), look) == make_key(orbiting(orbit=Orbit(1.5, 180, 60)), look)  # other parts: unchanged


def test_a_partial_orbit_block_is_written_back_as_it_was_given():
    d = api.normalize(cue_to_dict(orbiting(orbit=Orbit(tilt_deg=45))))
    assert d["orbit"] == {"tilt_deg": 45} and api.schema_errors("cue", d) == []
    assert not [w for w in api.validate(d) if w["severity"] == "error"]
