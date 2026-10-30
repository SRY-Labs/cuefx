"""Make (layout + stand-in effect) and the element video as functions, so the command-line scripts and the server run
exactly the same steps. No maths here: each function only calls the library in the order the scripts always did.

    make_layout(video, cue, look_path, out)          scripts/make_layout.py
    make_standin_effect(cue, look_path, out)         scripts/make_standin_effect.py
    ingest_ltx(cue, ltx, out, matte=None)            scripts/ingest_ltx.py
    write_element_video(cue, layers, out)            scripts/make_element_video.py

Merge is already one function (`services.delivery.merge.run_merge`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from services.cue.cue import Cue, shape_of
from services.delivery.debug import (write_ingest_sheet, write_layout_overlay, write_layout_sheet, write_look_check,
                         write_standin_sheet)
from services.make.ingest import IngestResult, ingest, ingest_with_matte, read_ltx, write_ingest
from services.make.interaction import element_frames, element_from_layers, layer_pair
from services.make.layout import LayoutError, spin_per_frame, write_layout
from services.make.effects import ending_plan
from services.make.look import place_look
from services.cue.route import OrbSample, first_full, orb_gains, orb_lights, orb_path
from services.make.standin_effect import look_tint, prepare_look, render_standin_shape, write_standin, write_standin_files
from services.delivery.video import VideoWriter, bt709_out, read_frames
from services.cue.window import effect_window

ELEMENT_CRF = 0  # lossless YUV: only the RGB -> YUV rounding is left (CRF 12 is up to ~12/255 off at the orb's edge)
ELEMENT_CHECK_FRAMES = 6
ELEMENT_CHECK_LIMIT = 2  # of 255


def read_look(look_path: Path) -> np.ndarray:
    """The look image (any channels, as stored); LayoutError if it cannot be read."""
    image = cv2.imread(str(look_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise LayoutError(f"cannot read look image {look_path}")
    return image


@dataclass
class LayoutRun:
    path: list[OrbSample]
    frames: int        # layout frames written
    look_frame: int    # layout frame the look is placed at (the first whole-orb frame)
    warnings: list[str] = field(default_factory=list)


def _lights(cue: Cue, path: list[OrbSample]) -> list[float] | None:
    """`orb_lights` (the charge-up pulse, orbit's front and back), or None where every sample is 1: the layout and the
    stand-in then draw exactly as they did before those behaviours."""
    lights = orb_lights(cue, path)
    return None if all(x == 1.0 for x in lights) else lights


def make_layout(video: Path, cue: Cue, look_path: Path, out: Path) -> LayoutRun:
    """Make 1-2 into `out`: layout.mp4, layout_png/, look_placed.png, look_check.png, layout_sheet.png and
    layout_overlay.mp4 (over the plate `video`). The cue must be free of errors. Raises LayoutError / TrackError."""
    look = read_look(look_path)
    path = orb_path(cue)
    gains = orb_gains(cue, path)
    full = first_full(gains)  # the look goes where the whole orb first is (frame 0 without an entrance)
    layout = write_layout(path, cue.canvas_rect, cue.canvas_size, cue.fps, out, gains=gains, kind=shape_of(cue),
                          spin=spin_per_frame(cue), ending=ending_plan(cue, path), lights=_lights(cue, path))
    placed, warnings = place_look(look, path, cue.canvas_rect, cue.canvas_size, at=full)
    cv2.imwrite(str(out / "look_placed.png"), placed)
    write_look_check(layout[full], placed, out / "look_check.png")
    write_layout_sheet(cue, layout, out / "layout_sheet.png")
    write_layout_overlay(video, cue, layout, out / "layout_overlay.mp4")
    return LayoutRun(path, len(layout), full, list(warnings))


@dataclass
class StandinRun:
    path: list[OrbSample]
    frames: int
    warnings: list[str] = field(default_factory=list)


def make_standin_effect(cue: Cue, look_path: Path, out: Path) -> StandinRun:
    """Stand-in Make into `out`: effect.mp4, matte.mp4 (canvas size, one frame per window frame) and
    standin_sheet.png. The cue must be free of errors. Raises LayoutError."""
    look, warnings = prepare_look(read_look(look_path))
    path = orb_path(cue)
    gains = orb_gains(cue, path)
    ending = ending_plan(cue, path)  # task 36: None for a fade, which draws as it always did
    lights = _lights(cue, path)
    if shape_of(cue) == "sphere":
        effect, matte = write_standin(look, path, cue.canvas_rect, cue.canvas_size, cue.fps, out, gains=gains,
                                      ending=ending, lights=lights)
    else:  # task 31: the shape is drawn by the layout's renderer, tinted from the look; the sphere is the look placed
        effect, matte = render_standin_shape(shape_of(cue), look_tint(look.light), path, cue.canvas_rect, cue.canvas_size,
                                             spin_per_frame(cue), gains=gains, ending=ending, lights=lights)
        write_standin_files(effect, matte, len(path), cue.fps, out)
    write_standin_sheet(path, effect, matte, out / "standin_sheet.png")
    return StandinRun(path, len(effect), list(warnings))


def ingest_ltx(cue: Cue, ltx: Path, out: Path, matte: Path | None = None) -> IngestResult:
    """Ingest an LTX effect (mp4 or PNG folder) into `out`: effect.mp4, matte.mp4 (canvas size, one frame per
    window frame) and ingest_sheet.png. Without `matte` the temporary luminance matte is made; with it, that matte
    is used. Raises VideoError (IngestError) if a file cannot be read or is not the canvas size."""
    w = effect_window(cue)
    if matte is None:
        res = ingest(read_ltx(ltx), w.length, cue.canvas_size)
    else:
        res = ingest_with_matte(read_ltx(ltx), read_ltx(matte), w.length, cue.canvas_size)
    write_ingest(res, cue.fps, out)
    write_ingest_sheet(w.start, res.effect, res.matte, out / "ingest_sheet.png")
    return res


def element_check_frames(layer_frames: list[int], n: int = ELEMENT_CHECK_FRAMES) -> list[int]:
    """`n` evenly spaced frames among those that have layers."""
    if not layer_frames:
        return []
    idx = np.linspace(0, len(layer_frames) - 1, min(n, len(layer_frames))).round().astype(int)
    return sorted({layer_frames[i] for i in idx})


@dataclass
class ElementRun:
    frames: int                  # frames the written file decodes to
    have: list[int]              # plate frames that had layers
    diffs: dict[int, int]        # check frame -> max difference from the layers (of 255)
    warnings: list[str] = field(default_factory=list)

    @property
    def worst(self) -> int:
        return max(self.diffs.values(), default=0)


def element_layer_frames(cue: Cue, layers: Path) -> list[int]:
    """Plate frames Merge exported a core layer for."""
    return [f for f in range(cue.frames) if (layers / "core" / f"core_{f:04d}.png").exists()]


def write_element_video(cue: Cue, layers: Path, out: Path) -> ElementRun:
    """The element video (the effect alone on black, plate size, one frame per plate frame) from Merge's exported
    layers: H.264 4:4:4 CRF 0, Rec.709 tagged like the plate, checked against the layers on a few frames (at most
    ELEMENT_CHECK_LIMIT / 255 apart, else a warning). Raises LayoutError if there are no layers; VideoError."""
    have = element_layer_frames(cue, layers)
    if not have:
        raise LayoutError(f"no core layers in {layers}; run Merge with export_layers first")
    check = element_check_frames(have)
    with VideoWriter(out, tuple(cue.plate_size), cue.fps, ELEMENT_CRF, bt709_out("yuv444p"), "yuv444p") as w:
        for _, frame in element_frames(layers, cue.frames, cue.plate_size):
            w.write(frame)
    n, decoded = 0, {}
    for f, fr in enumerate(read_frames(out)):
        n += 1
        if f in check:
            decoded[f] = fr
    warnings = []
    if n != cue.frames:
        warnings.append(f"the element video decodes to {n} frames, the cue has {cue.frames}")
    diffs = {f: int(np.abs(decoded[f].astype(int) - element_from_layers(*layer_pair(layers, f)).astype(int)).max())
             for f in check if f in decoded}
    run = ElementRun(n, have, diffs, warnings)
    if run.worst > ELEMENT_CHECK_LIMIT:
        warnings.append(f"the element video differs from the layers by up to {run.worst}/255")
    return run
