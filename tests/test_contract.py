"""The app contract (contract/): golden files fit their schemas, golden outputs are reproduced, v2.1 cues still load and
route as before, and easing_in shapes the flight as the UI's EASE table does."""
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from conftest import make_cue
from services.cue import api
from services.cue.cue import VERSION, Entrance, RoutePoint, Vanish, cue_from_dict, cue_to_dict, load_cue
from services.cue.make_key import make_key
from services.cue.route import EASE, entrance_gain, orb_path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "contract" / "golden"
EXAMPLE = Path(__file__).parent / "fixtures" / "throw_71319.cue.json"
FIXTURES = Path(__file__).parent / "fixtures"


def golden(name: str):
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


def example_with_golden_track():
    return replace(load_cue(EXAMPLE), track=api.points_from_track(golden("throw_71319.track.json")))


# --------------------------------------------------------------------------- golden files are valid

@pytest.mark.parametrize("file,schema", [
    ("throw_71319.cue.json", "cue"),
    ("throw_71319.track.json", "track"),
    ("throw_71319.preview.json", "preview"),
    ("sample_job_running.json", "job"),
    ("sample_job_failed.json", "job"),
    ("sample_job_waiting.json", "job"),
    ("sample_result.json", "result"),
])
def test_golden_file_fits_its_schema(file, schema):
    assert api.schema_errors(schema, golden(file)) == []


def test_golden_validate_output_is_a_list_of_warnings():
    warnings = golden("throw_71319.validate.json")
    assert warnings and all(api.schema_errors("warning", w) == [] for w in warnings)
    assert all(w["code"] in api.codes() for w in warnings)


def test_samples_are_marked_illustrative():
    for name in ("sample_job_running.json", "sample_job_failed.json", "sample_result.json"):
        assert golden(name)["$comment"].startswith("ILLUSTRATIVE SAMPLE")


def test_sample_result_has_the_required_checks():
    checks = {c["id"]: c for c in golden("sample_result.json")["checks"]}
    assert {"plate_unchanged", "black_clamp", "matte_source", "face_shift_px"} <= set(checks)
    assert checks["face_shift_px"]["status"] == "warn" and checks["face_shift_px"]["message"] == "not measured yet"


def test_the_example_fits_the_cue_schema_after_loading():
    d = cue_to_dict(load_cue(EXAMPLE))
    assert d["version"] == "2.2" and VERSION == "2.3"  # a v2.1 file loads as 2.2; new cues are written as 2.3
    assert api.schema_errors("cue", d) == []


def test_an_analyze_result_of_the_golden_track_fits_its_schema():
    res = api.analyze_result(api.points_from_track(golden("throw_71319.track.json")))
    assert api.schema_errors("analyze_result", res) == []
    assert res["track"] == golden("throw_71319.track.json")  # lossless both ways


def test_the_golden_cue_is_the_example_shot():
    ex, g = cue_to_dict(load_cue(EXAMPLE)), golden("throw_71319.cue.json")
    for k in ("clip", "frames", "effect_range", "hand_seed", "corrections", "anchor_offset", "appear_frame",
              "release_frame", "diameter_px", "vanish", "behind", "seed", "entrance"):
        assert g[k] == ex[k], k
    assert [{k: p[k] for k in ("f", "x", "y", "diameter_px")} for p in g["route"]] == \
           [{k: p[k] for k in ("f", "x", "y", "diameter_px")} for p in ex["route"]]


# --------------------------------------------------------------------------- golden outputs are reproduced

def test_preview_path_reproduces_the_golden_preview_exactly():
    assert api.preview_path(golden("throw_71319.cue.json"), golden("throw_71319.track.json")) == \
           golden("throw_71319.preview.json")


def test_validate_reproduces_the_golden_validate_output():
    assert api.validate(golden("throw_71319.cue.json"), golden("throw_71319.track.json")) == \
           golden("throw_71319.validate.json")


def test_the_golden_preview_is_the_render_path():
    cue = replace(cue_from_dict(golden("throw_71319.cue.json")),
                  track=api.points_from_track(golden("throw_71319.track.json")))
    preview = golden("throw_71319.preview.json")
    assert len(preview) == cue.frames == 193
    for f, x, y, d, vis in orb_path(cue):
        p = preview[f]
        assert (p["x"], p["y"], p["diameter"], p["visibility"]) == \
               (round(x, 3), round(y, 3), round(d * entrance_gain(cue, f), 3), round(vis, 4))


# --------------------------------------------------------------------------- ranges: v2.1 -> v2.2

def test_v21_loads_as_v22_with_the_same_ranges_and_round_trips():
    d21 = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert d21["version"] == "2.1"
    cue = cue_from_dict(d21)
    assert cue.version == "2.2"
    assert (cue.effect_range.start, cue.effect_range.end) == (d21["effect_range"]["start"], d21["effect_range"]["end"])
    assert (cue.vanish.start, cue.vanish.end) == (d21["vanish"]["start"], d21["vanish"]["end"])
    assert [(b.start, b.end) for b in cue.behind] == [(b["start"], b["end"]) for b in d21["behind"]]
    assert all(p.easing_in == "linear" and p.note is None for p in cue.route)
    d22 = cue_to_dict(cue)
    assert cue_from_dict(d22) == cue and cue_to_dict(cue_from_dict(d22)) == d22
    assert "note" not in d22["route"][0] and "spill" not in d22["grade"]  # unset v2.2 extras are not written


def test_the_old_example_renders_the_same_orb_path_as_before():
    old = json.loads((FIXTURES / "orb_path_throw_71319_v21.json").read_text(encoding="utf-8"))
    assert [list(s) for s in orb_path(example_with_golden_track())] == old


def test_v21_make_keys_are_unchanged(tmp_path):
    look = tmp_path / "look.png"
    look.write_bytes(b"look image bytes")
    assert make_key(make_cue(), look) == "d5fa87152707"  # the v2.2 schema adds nothing (was 32a6eef7fb99 before
    noted = make_cue(route=[replace(p, note="hi") for p in make_cue().route])  # MAKE_VERSION 3, task 05a)
    assert make_key(noted, look) == "d5fa87152707"       # a note never reruns Make
    eased = make_cue(route=[*make_cue().route[:2], replace(make_cue().route[2], easing_in="ease_out")])
    assert make_key(eased, look) != "d5fa87152707"       # easing changes the path, so it does


# --------------------------------------------------------------------------- easing

@pytest.mark.parametrize("case", ["synthetic", "synthetic_entrance", "synthetic_4pt"])
def test_linear_easing_is_byte_identical_to_the_v21_orb_path(case):
    cues = {
        "synthetic": make_cue(),
        "synthetic_entrance": make_cue(entrance=Entrance(0.5)),
        "synthetic_4pt": make_cue(route=[RoutePoint(107, None, None, 110), RoutePoint(112, 900, 500, 90),
                                         RoutePoint(120, 700, 420, 60), RoutePoint(130, 420, 360, 25)],
                                  vanish=Vanish(128, 131)),
    }
    old = json.loads((FIXTURES / "orb_path_v21.json").read_text(encoding="utf-8"))[case]
    cue = cues[case]
    assert all(p.easing_in == "linear" for p in cue.route)
    path = [list(s) for s in orb_path(cue)]
    assert path[:len(old)] == old  # the window limit is 193 now (task 33): a 129-frame window is no longer trimmed to 121


def test_the_ease_curves_are_the_ui_table():
    s = np.linspace(0, 1, 11)
    np.testing.assert_allclose([EASE["ease_in"](t) for t in s], s * s)
    np.testing.assert_allclose([EASE["ease_out"](t) for t in s], 1 - (1 - s) ** 2)
    np.testing.assert_allclose([EASE["ease_in_out"](t) for t in s], s * s * (3 - 2 * s))
    assert all(EASE["linear"](t) is t for t in s.tolist())


def straight(easing: str):
    """3 collinear, equally spaced points, level with the release point (1000, 530): the spline is the straight line,
    so x tells how far along the first segment the orb is."""
    return make_cue(route=[RoutePoint(107, None, None, 110), RoutePoint(117, 700, 530, 60, easing_in=easing),
                           RoutePoint(127, 400, 530, 10)], vanish=Vanish(126, 128))


@pytest.mark.parametrize("easing", ["linear", "ease_in", "ease_out", "ease_in_out"])
def test_each_easing_shapes_the_segment_into_its_point(easing):
    path = {s[0]: s for s in orb_path(straight(easing))}
    u = [(1000 - path[f][1]) / 300 for f in range(107, 118)]          # fraction of the segment done
    np.testing.assert_allclose(u, [EASE[easing](k / 10) for k in range(11)], atol=1e-9)
    d = [path[f][3] for f in range(107, 118)]
    np.testing.assert_allclose(d, [110 - 50 * EASE[easing](k / 10) for k in range(11)], atol=1e-9)  # size too
    steps = np.diff(u)
    assert all(steps > 0)                                              # monotonic: never stops or goes back
    if easing == "ease_in":
        assert all(np.diff(steps) > 0)                                 # speeds up into the point
    elif easing == "ease_out":
        assert all(np.diff(steps) < 0)                                 # slows down into the point
    elif easing == "ease_in_out":
        assert steps[0] < steps[4] and steps[-1] < steps[5]            # slow, fast, slow
    after = [path[f][1] for f in range(117, 128)]                      # the next segment stays linear
    np.testing.assert_allclose(after, np.linspace(700, 400, 11), atol=1e-9)


# --------------------------------------------------------------------------- error codes

def test_every_code_the_backend_emits_is_in_error_codes():
    src = "".join(p.read_text(encoding="utf-8") for p in (
        ROOT / "services" / "cue" / "cue.py",
        ROOT / "services" / "cue" / "api.py",
        ROOT / "services" / "analysis" / "track.py",
    ))
    used = set(re.findall(r'\bout\.(?:error|warning)\("([a-z_]+)"', src))     # cue.check
    used |= set(re.findall(r'(?<![\w.])(?:_warning|error)\("([a-z_]+)"', src))  # api
    used |= set(re.findall(r'TrackError\([^)]*,\s*"([a-z_]+)"\)', src))
    assert used and used <= set(api.codes()), used - set(api.codes())


def test_the_contract_error_codes_are_all_there():
    c = api.codes()
    for code in ("no_hand", "track_lost", "clip_unreadable", "clip_too_long"):
        assert c[code]["call"] == "analyze" and c[code]["severity"] == "error"
    for code in ("gpu_memory", "server_lost", "render_failed", "cancelled"):
        assert c[code]["call"] == "generate"
    assert c["input_invalid"]["call"] == "any"
    for code in ("no_track", "track_uncertain", "check_pass_edge", "entrance_shortened", "no_path", "setting_not_used"):
        assert c[code]["call"] == "validate"
    assert {v["severity"] for v in c.values()} <= {"info", "warning", "error"}
    job_codes = set(json.loads((ROOT / "contract" / "schemas" / "job.schema.json").read_text(encoding="utf-8"))
                    ["properties"]["error"]["oneOf"][1]["properties"]["code"]["enum"])
    assert job_codes <= set(c)
