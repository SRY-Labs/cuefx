"""Task 36 part A (contract v2.8): GET /api/defaults, and the effect_range that validate and normalize give back."""
import json

from server_helpers import client, golden, settings, synthetic_cue  # noqa: F401
from services.cue import api
from services.cue.cue import MAX_EFFECT_FRAMES
from services.make.layout import SHAPE_SPECS


def test_defaults_route_has_every_default_and_reads_the_backend_constants(client):
    r = client.get("/api/defaults")
    assert r.status_code == 200
    d = r.json()
    assert d == api.defaults()
    assert d["spin_deg_per_s"] == {"sphere": 288, "ring": 90, "crystal": 60, "cube": 60}
    assert d["spin_deg_per_s"] == {k: v.spin for k, v in SHAPE_SPECS.items()}
    assert set(d) == {"spin_deg_per_s", "ending_seconds", "ending_default_seconds", "charge_up", "orbit", "max_window_frames", "orbit_tilt_deg_by_part"}
    assert d["orbit_tilt_deg_by_part"] == {"head": 80}
    assert d["max_window_frames"] == MAX_EFFECT_FRAMES == 193
    assert d["ending_seconds"] == {"fade": 0.0, "burst": 0.3, "shatter": 0.5}
    assert d["ending_default_seconds"] == {"burst": 0.6, "shatter": 1.0}   # task 38: what the UI sets on picking that ending
    assert d["charge_up"] == {"start_scale": 0.6, "end_scale": 1.0, "pulse_hz": 3.5, "pulse_amount": 0.08}
    assert d["orbit"] == {"radius_sizes": 1.5, "speed_deg_per_s": 180, "tilt_deg": 60}


def test_defaults_follow_the_constants_not_a_copy(monkeypatch):
    monkeypatch.setattr(api, "MAX_EFFECT_FRAMES", 97)
    monkeypatch.setitem(SHAPE_SPECS, "ring", type(SHAPE_SPECS["ring"])(45.0))
    d = api.defaults()
    assert d["max_window_frames"] == 97 and d["spin_deg_per_s"]["ring"] == 45.0


def test_normalize_already_returns_the_computed_effect_range(client):
    cue = synthetic_cue()
    moved = {**cue, "vanish": {"start": 18, "end": 22}, "effect_range": {"start": 0, "end": 9}}
    out = client.post("/api/cue/normalize", json={"cue": moved}).json()
    er = api.computed_effect_range(moved)
    assert out["effect_range"] == er and er != moved["effect_range"]


def test_validate_gives_the_effect_range_normalize_would_compute_in_a_header(client):
    cue = {**synthetic_cue(), "vanish": {"start": 18, "end": 22}}
    r = client.post("/api/validate", json={"cue": cue})
    assert r.status_code == 200 and isinstance(r.json(), list)
    header = json.loads(r.headers["x-cuefx-effect-range"])
    assert header == client.post("/api/cue/normalize", json={"cue": cue}).json()["effect_range"]


def test_the_effect_range_header_is_readable_cross_origin(client):
    r = client.post("/api/validate", json={"cue": synthetic_cue()}, headers={"Origin": "http://localhost:5173"})
    exposed = [h.strip().lower() for h in r.headers["access-control-expose-headers"].split(",")]
    assert "x-cuefx-effect-range" in exposed


def test_validate_has_no_header_for_a_cue_with_schema_errors(client):
    r = client.post("/api/validate", json={"cue": {**synthetic_cue(), "frames": "many"}})
    assert r.status_code == 200 and "x-cuefx-effect-range" not in r.headers
    assert r.json()[0]["code"] == "invalid_field"


def test_the_golden_validate_answer_is_unchanged(client):
    cue, track = golden("throw_71319.cue.json"), golden("throw_71319.track.json")
    r = client.post("/api/validate", json={"cue": cue, "track": track})
    assert r.json() == golden("throw_71319.validate.json")
    assert json.loads(r.headers["x-cuefx-effect-range"]) == cue["effect_range"]


def test_normalize_gives_what_it_changed_in_the_warnings_header(client):
    cue = {**synthetic_cue(), "ending": "shatter", "vanish": {"start": 9, "end": 10}}  # a 24-frame clip
    r = client.post("/api/cue/normalize", json={"cue": cue})
    assert r.status_code == 200 and r.json()["ending"] == "shatter" and r.json()["version"] == "2.6"
    warnings = json.loads(r.headers["x-cuefx-warnings"])
    assert [w["code"] for w in warnings] == ["vanish_extended"]
    assert r.json()["vanish"] == {"start": 9, "end": 21}
    again = client.post("/api/cue/normalize", json={"cue": r.json()})
    assert again.json() == r.json() and json.loads(again.headers["x-cuefx-warnings"]) == []
    plain = client.post("/api/cue/normalize", json={"cue": synthetic_cue()})
    assert json.loads(plain.headers["x-cuefx-warnings"]) == []


def test_the_end_of_the_clip_cuts_an_ending_and_normalize_says_window_trimmed(client):
    cue = {**synthetic_cue(), "ending": "shatter"}  # vanish 18..: a shatter needs 30 frames, the clip has 24
    r = client.post("/api/cue/normalize", json={"cue": cue})
    assert [w["code"] for w in json.loads(r.headers["x-cuefx-warnings"])] == ["window_trimmed"]
    assert r.json()["vanish"]["end"] == r.json()["effect_range"]["end"] <= 24
