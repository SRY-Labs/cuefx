"""Fingerprints of what the example cue (with a synthetic track) renders, taken before the entrance existed.

    python tests/golden_seconds0.py     writes tests/fixtures/golden_seconds0.json (only ever run on the old code)

`fingerprint` is what test_entrance.py compares against that file with `entrance.seconds = 0`.
Task 05a (no motion-blur copies) changed only the frames the old code blurred (107..111): their layout / effect / matte
and merge_107 / 109 / 111 were re-recorded; the path and every other merge hash are still the pre-entrance ones.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.cue.cue import Cue, TrackPoint, cue_from_dict  # noqa: E402

EXAMPLE = Path(__file__).parent / "fixtures" / "throw_71319.cue.json"
FIXTURE = Path(__file__).parent / "fixtures" / "golden_seconds0.json"
MERGE_FRAMES = (34, 39, 40, 41, 42, 60, 85, 107, 109, 111, 112, 114)  # inside the example's effect_range


def example_cue() -> Cue:
    """The example cue with a synthetic track: a palm drifting slowly around (1160, 550)."""
    cue = cue_from_dict(json.loads(EXAMPLE.read_text(encoding="utf-8")))
    track = [TrackPoint(f, 1160.0 + 12.0 * math.sin(f / 9.0), 550.0 + 6.0 * math.cos(f / 13.0), "tracked", 95.0, 0.9)
             for f in range(cue.frames)]
    return replace(cue, track=track)


def _h(*arrays) -> str:
    d = hashlib.sha256()
    for a in arrays:
        d.update(np.ascontiguousarray(a).tobytes())
    return d.hexdigest()


def synthetic_plate(f: int, size: tuple[int, int]) -> np.ndarray:
    return np.random.default_rng([7, f]).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)


def fingerprint(cue: Cue, layout_frames, effect_frames, matte_frames, path, merge_frame_fn) -> dict[str, str]:
    """Hashes of the path, the layout, the stand-in effect and matte, and a few merged frames."""
    out = {"path": hashlib.sha256(repr([tuple(float(v) for v in s) for s in path]).encode()).hexdigest(),
           "layout": _h(*layout_frames), "effect": _h(*effect_frames), "matte": _h(*matte_frames)}
    first = path[0][0]
    for f in MERGE_FRAMES:
        i = f - first
        out[f"merge_{f}"] = _h(merge_frame_fn(f, synthetic_plate(f, cue.plate_size), effect_frames[i], matte_frames[i],
                                              path[i][4], path[i][3]))
    return out


def _old_code() -> dict[str, str]:
    from services.make.layout import render_layout
    from services.make.look import standin_look
    from services.delivery.merge import merge_frame
    from services.cue.route import orb_path
    from services.make.standin_effect import prepare_look, render_standin

    cue = example_cue()
    path = orb_path(cue)
    layout = render_layout(path, cue.canvas_rect, cue.canvas_size)
    look, _ = prepare_look(standin_look())
    effect, matte = render_standin(look, path, cue.canvas_rect, cue.canvas_size)

    def mf(f, plate8, e8, m8, vis, d):
        return merge_frame(f, plate8, e8, m8, vis, cue.grade, cue.canvas_rect, cue.canvas_size, [], d).out8

    return fingerprint(cue, layout, effect, matte, path, mf)


if __name__ == "__main__":
    if FIXTURE.exists():  # taken on commit 53a2504, before the entrance; rerunning now would record the new code
        raise SystemExit(f"{FIXTURE} exists; it must stay the pre-entrance reference. Delete it only on purpose.")
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps(_old_code(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {FIXTURE}")
