"""Experimental LTX overlap planning and exact frame assembly.

All indices in a plan are plate frames unless explicitly named ``index``.
The render service must supply generated frames; this module never synthesizes
an entrance or substitutes a timing fade for generation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generic, Sequence, TypeVar

import cv2

from services.cue.cue import Cue, entrance_frames, entrance_start, vanish_of
from services.cue.route import entrance_gain, orb_path
from services.delivery.video import read_frames
from services.cue.window import effect_window

T = TypeVar("T")


def padded_count(count: int) -> int:
    """Smallest valid LTX frame count at least ``count``."""
    if count < 1:
        raise ValueError("frame count must be positive")
    return 1 + 8 * ((count - 1 + 7) // 8)


@dataclass(frozen=True)
class OverlapPlan:
    window_start: int
    window_end: int
    appear_frame: int
    zero_frame: int
    look_source_plate_frame: int
    look_source_window_index: int
    anchor_frame: int
    reverse_pre_start: int
    main_real_count: int
    main_padded_count: int
    context_count: int
    reverse_real_count: int
    reverse_padded_count: int
    first_empty_cell: int
    selection_reason: str

    @property
    def final_count(self) -> int:
        return self.window_end - self.window_start

    def as_dict(self) -> dict:
        return {"method": "ltx_overlap_reverse_entrance_v1", "plan_version": 1,
                **asdict(self), "final_real_count": self.final_count}


def _usable(cue: Cue, sample: tuple, frame: int) -> bool:
    """Known geometry and cue timing, with no new semantic detector."""
    x, y, d = sample[1:4]
    cx, cy, cw, ch = cue.canvas_rect
    return (entrance_gain(cue, frame) >= 1 and sample[4] > 0 and d > 0
            and cx + d / 2 <= x <= cx + cw - d / 2
            and cy + d / 2 <= y <= cy + ch - d / 2
            and frame < vanish_of(cue).start
            and (cue.release_frame is None or frame < cue.release_frame))


def select_reference_frame(cue: Cue) -> tuple[int, str]:
    """Earliest usable full state in a bounded nine-frame range from A.

    A remains the fallback when geometry cannot justify choosing a later frame;
    callers must still validate that the actual rendered reference is usable.
    """
    if cue.appear_frame is None:
        raise ValueError("cue has no appear_frame")
    w = effect_window(cue)
    samples = {s[0]: s for s in orb_path(cue)}
    a = cue.appear_frame
    for f in range(a, min(a + 9, w.end)):
        if f in samples and _usable(cue, samples[f], f):
            return f, "earliest full visible unclipped frame from cue geometry"
    if a not in samples:
        raise ValueError("appear_frame is outside the processing window")
    return a, "first_full fallback; no reliable later geometry candidate"


def plan_overlap(cue: Cue, look_source_plate_frame: int | None = None,
                 max_frames: int = 193) -> OverlapPlan | None:
    """Plan the main-first reverse entrance without changing cue timing.

    ``None`` means the existing frame-zero/disabled-entrance path. A successful
    saved look must pass its recorded source frame rather than reselecting one.
    """
    w = effect_window(cue)
    a, c = cue.appear_frame, entrance_start(cue)
    if entrance_frames(cue) == 0 or a == 0:
        return None
    r, reason = select_reference_frame(cue) if look_source_plate_frame is None else (
        look_source_plate_frame, "recorded look provenance")
    if not w.start <= c < a <= r < w.end:
        raise ValueError("window must cover zero, full state, and matching look frame")
    samples = {s[0]: s for s in orb_path(cue)}
    if not _usable(cue, samples[r], r):
        raise ValueError("look source is not a visible, unclipped full state before release/ending")
    # Context count is 1, 9 or 17. Stop before geometry or cue action becomes unsuitable.
    last = r
    for f in range(r + 1, min(r + 17, w.end)):
        if not _usable(cue, samples[f], f):
            break
        last = f
    h = r + 8 * ((last - r) // 8)
    b = max(w.start, c - 8)
    q = h - c
    first_empty_cell = 1 + (q - 1 + 7) // 8
    reverse_real = h - b + 1
    reverse_padded = padded_count(max(reverse_real, 8 * first_empty_cell + 1))
    if reverse_padded > max_frames:
        # Reducing overlap first is the smallest valid single-window fallback.
        for candidate in range(h - 8, r - 1, -8):
            q2 = candidate - c
            cell2 = 1 + (q2 - 1 + 7) // 8
            count2 = padded_count(max(candidate - b + 1, 8 * cell2 + 1))
            if count2 <= max_frames:
                h, q, first_empty_cell, reverse_padded = candidate, q2, cell2, count2
                reverse_real = h - b + 1
                break
        else:
            raise ValueError("reverse entrance exceeds single-window budget; requires chained windows")
    main_real = w.end - r
    if padded_count(main_real) > max_frames:
        raise ValueError("main render exceeds single-window budget")
    return OverlapPlan(w.start, w.end, a, c, r, r - w.start, h, b,
                       main_real, padded_count(main_real), h - r + 1,
                       reverse_real, reverse_padded, first_empty_cell, reason)


def extract_layout_reference(layout_video: Path, plan: OverlapPlan, output: Path,
                             canvas_size: tuple[int, int]) -> str:
    """Write the exact decoded layout frame R and return its SHA-256.

    The caller keeps this next to the generated look's provenance. No scaling
    occurs here; the frame is the same encoded canvas that LTX will ingest.
    """
    frame = None
    count = 0
    for index, decoded in enumerate(read_frames(layout_video)):
        if index == plan.look_source_window_index:
            frame = decoded.copy()
        count = index + 1
    if count != plan.final_count:
        raise ValueError(f"layout has {count} frames; expected {plan.final_count}")
    if frame is None:
        raise ValueError("look source frame is missing from layout")
    if (frame.shape[1], frame.shape[0]) != canvas_size:
        raise ValueError("layout canvas does not match cue canvas")
    if not cv2.imwrite(str(output), frame):
        raise OSError(f"could not write {output}")
    return hashlib.sha256(output.read_bytes()).hexdigest()


def prepare_reference_record(cue: Cue, layout_video: Path, output_png: Path,
                             look_source_plate_frame: int | None = None) -> dict:
    """Prepare the exact image-model input and durable mapping provenance.

    The image-generation service can consume ``output_png`` and must attach its
    own job/result identity to this record before a saved look is reused.
    """
    plan = plan_overlap(cue, look_source_plate_frame)
    if plan is None:
        raise ValueError("overlap is inactive; use the existing render path")
    frame_hash = extract_layout_reference(layout_video, plan, output_png, cue.canvas_size)
    layout_hash = hashlib.sha256(layout_video.read_bytes()).hexdigest()
    mapping = {"canvas_size": cue.canvas_size, "canvas_rect": cue.canvas_rect,
               "plate_size": cue.plate_size, "window_start": plan.window_start,
               "window_end": plan.window_end}
    mapping_hash = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    return {"look_source_plate_frame": plan.look_source_plate_frame,
            "look_source_window_index": plan.look_source_window_index,
            "layout_sha256": layout_hash, "extracted_frame_sha256": frame_hash,
            "canvas_mapping_sha256": mapping_hash, "canvas_mapping": mapping,
            "selection_reason": plan.selection_reason,
            "look_generation_job_id": None, "look_generation_result_id": None,
            "plan": plan.as_dict()}


def prepare_full_scene_reference_record(cue: Cue, source_video: Path, output_png: Path,
                                        look_source_plate_frame: int | None = None) -> dict:
    """Extract exact full-scene plate frame ``R`` and record its provenance.

    Unlike :func:`prepare_reference_record`, ``source_video`` is the approved
    full-length hard composite, not the effect-window layout.
    """
    plan = plan_overlap(cue, look_source_plate_frame)
    if plan is None:
        raise ValueError("overlap is inactive; use the existing render path")
    selected = None
    count = 0
    for index, decoded in enumerate(read_frames(source_video)):
        if index == plan.look_source_plate_frame:
            selected = decoded.copy()
        count = index + 1
    if count != cue.frames:
        raise ValueError(f"hard composite has {count} frames; expected {cue.frames}")
    if selected is None:
        raise ValueError("look source frame is missing from hard composite")
    if (selected.shape[1], selected.shape[0]) != cue.plate_size:
        raise ValueError("hard-composite canvas does not match cue plate")
    if not cv2.imwrite(str(output_png), selected):
        raise OSError(f"could not write {output_png}")
    source_hash = hashlib.sha256(source_video.read_bytes()).hexdigest()
    frame_hash = hashlib.sha256(output_png.read_bytes()).hexdigest()
    mapping = {"plate_size": cue.plate_size, "canvas_size": cue.canvas_size,
               "canvas_rect": cue.canvas_rect, "window_start": plan.window_start,
               "window_end": plan.window_end}
    mapping_hash = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    return {"look_source_plate_frame": plan.look_source_plate_frame,
            "look_source_window_index": plan.look_source_window_index,
            "source_sha256": source_hash, "extracted_frame_sha256": frame_hash,
            "canvas_mapping_sha256": mapping_hash, "canvas_mapping": mapping,
            "selection_reason": plan.selection_reason,
            "look_generation_job_id": None, "look_generation_result_id": None,
            "plan": plan.as_dict()}


def reverse_source(layout: Sequence[T], plan: OverlapPlan) -> list[T]:
    """Reverse actual layout pixels, repeating clean B at the end only."""
    if len(layout) != plan.final_count:
        raise ValueError("layout count differs from plan")
    frames = [layout[f - plan.window_start] for f in range(plan.anchor_frame,
                                                            plan.reverse_pre_start - 1, -1)]
    return frames + [frames[-1]] * (plan.reverse_padded_count - len(frames))


@dataclass(frozen=True)
class ReversePreparation(Generic[T]):
    """Pixel-frame inputs for a fresh reverse encode; no latent slicing."""

    guide_frames: tuple[T, ...]
    matched_look: T
    fixed_generated_prefix: tuple[T, ...]
    clean_source_frames: tuple[T, ...]
    guide_plate_frames: tuple[int | None, ...]
    prefix_plate_frames: tuple[int, ...]


def prepare_reverse_entrance(layout: Sequence[T], main: Sequence[T],
                             plan: OverlapPlan) -> ReversePreparation[T]:
    """Match reverse pixels and look to the actual main frames at H..R.

    The whole reversed guide is also the clean-source encode input; the
    downstream fixed-content helper selects only complete clean ending cells.
    Padded repeats have no plate-time owner.
    """
    if len(main) < plan.main_real_count:
        raise ValueError("main output is missing real frames")
    guide = tuple(reverse_source(layout, plan))
    prefix_indices = tuple(range(plan.anchor_frame, plan.look_source_plate_frame - 1, -1))
    prefix = tuple(main[f - plan.look_source_plate_frame] for f in prefix_indices)
    if len(prefix) != plan.context_count or len(guide) != plan.reverse_padded_count:
        raise AssertionError("reverse preparation count differs from plan")
    real_indices = tuple(range(plan.anchor_frame, plan.reverse_pre_start - 1, -1))
    guide_indices = real_indices + (None,) * (len(guide) - len(real_indices))
    return ReversePreparation(guide, prefix[0], prefix, guide, guide_indices, prefix_indices)


def assemble(clean: Sequence[T], main: Sequence[T], reverse: Sequence[T],
             plan: OverlapPlan) -> list[T]:
    """Own every plate-time index exactly once; retain main frames R onward."""
    if len(clean) != plan.final_count or len(main) < plan.main_real_count or len(reverse) < plan.reverse_real_count:
        raise ValueError("one or more real output intervals are incomplete")
    first = list(clean[:plan.zero_frame - plan.window_start + 1])
    middle = [reverse[plan.anchor_frame - f] for f in range(plan.zero_frame + 1,
                                                              plan.look_source_plate_frame)]
    result = first + middle + list(main[:plan.main_real_count])
    if len(result) != plan.final_count:
        raise AssertionError("assembled frame count changed")
    return result


def assemble_full_scene(clean: Sequence[T], main: Sequence[T], reverse: Sequence[T],
                        plan: OverlapPlan) -> list[T]:
    """Insert the exact assembled window into the complete approved scene."""
    if len(clean) < plan.window_end:
        raise ValueError("full-scene source is shorter than the planned window")
    window = assemble(clean[plan.window_start:plan.window_end], main, reverse, plan)
    return list(clean[:plan.window_start]) + window + list(clean[plan.window_end:])
