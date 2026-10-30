"""The interaction pass inputs: the element video and the segment of the shot LTX re-renders.

Element video: the effect alone on black at plate size, one frame per plate frame, rebuilt from the layers Merge
exports (`merge_clip.py --export-layers`): element = encode(linear(core) + linear(glow)), i.e. the light the effect adds
to a black plate. Frames without layers are black.

Segment (task 15a; all frames, from the cue, the fps and the element video):
    A    appear_frame (the orb is whole)            N_in  entrance frames (`services.cue.cue.entrance_frames`)
    C    A - N_in, entrance start                   K     round(interaction.crossfade_seconds x fps)
    P    SETTLE_PREROLL                             S     C - K - P, segment start
    Z    last frame the element is visible          T     round(interaction.exit_tail_seconds x fps)
    End  Z + T + K, segment end (inclusive)         pin   A - S, the matched reference's index in the segment
If S < 0, P then K then N_in shrink until S = 0, each with a warning; End past the clip is clamped with a warning;
a segment longer than `max_len` is an error (SegmentError), never trimmed silently.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from services.common.colour import encode8, srgb_to_linear
from services.cue.cue import Cue, entrance_frames, interaction_of, seconds_to_frames

SETTLE_PREROLL = 8      # LTX's latent step (8 frames per latent): the first frames of an LTX run are unstable
MAX_SEGMENT_FRAMES = 193


class SegmentError(ValueError):
    """The segment cannot be made as asked (longer than the limit)."""


# --------------------------------------------------------------------------- element video

def element_from_layers(core16: np.ndarray, glow16: np.ndarray) -> np.ndarray:
    """One element frame (8-bit BGR, sRGB) from Merge's 16-bit BGRA core and glow layers: the sum of their colours in
    linear light. The colours are the light itself (core = E m, glow = g E (1 - m), `services.delivery.merge.layer_images`), so
    alpha is not used."""
    lin = srgb_to_linear(core16[..., :3] / 65535.0) + srgb_to_linear(glow16[..., :3] / 65535.0)
    return encode8(lin)


def layer_pair(layers_dir: Path, f: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(core, glow) 16-bit BGRA layers of plate frame f, or None if Merge exported none for it."""
    core = layers_dir / "core" / f"core_{f:04d}.png"
    glow = layers_dir / "glow" / f"glow_{f:04d}.png"
    if not core.exists() or not glow.exists():
        return None
    return cv2.imread(str(core), cv2.IMREAD_UNCHANGED), cv2.imread(str(glow), cv2.IMREAD_UNCHANGED)


def element_frames(layers_dir: Path, frames: int, plate_size: tuple[int, int]) -> Iterator[tuple[int, np.ndarray]]:
    """(f, element frame) for every plate frame 0..frames-1; black where there are no layers."""
    w, h = plate_size
    black = np.zeros((h, w, 3), np.uint8)
    for f in range(frames):
        pair = layer_pair(layers_dir, f)
        yield f, (black if pair is None else element_from_layers(*pair))


# --------------------------------------------------------------------------- segment

def padded_length(n: int) -> int:
    """The smallest 8k+1 >= n: what LTX pads a clip of n frames to."""
    return 8 * math.ceil((n - 1) / 8) + 1 if n > 1 else 1


def last_visible(areas: list[int]) -> int:
    """Z: the last frame with any "on" pixel (areas as `find_entrance` returns them); -1 if none."""
    return max((i for i, a in enumerate(areas) if a > 0), default=-1)


@dataclass
class SegmentPlan:
    """Every value of the definitions table (frames), after the clamps, and their warnings."""
    A: int
    N_in: int
    C: int
    K: int
    P: int
    S: int
    Z: int
    T: int
    End: int
    frames: int                       # in the shot
    length: int                       # End - S + 1
    padded_length: int                # the 8k+1 LTX pads it to
    pin_frame_in_segment: int         # A - S
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def plan_segment(appear: int, n_in: int, crossfade: int, preroll: int, last: int, tail: int, frames: int,
                 max_len: int = MAX_SEGMENT_FRAMES) -> SegmentPlan:
    """The segment for an orb whole from `appear`, with an entrance of `n_in` frames, a cross-fade of `crossfade`,
    a settle pre-roll of `preroll`, the element last visible at `last` and an exit tail of `tail`, in a shot of
    `frames` frames. Clamps as the module docstring says. Raises SegmentError if End - S + 1 > max_len.

    N_in only shrinks if appear < n_in; `services.cue.cue.entrance_frames` already cuts it to appear_frame, so the segment
    and what Make drew agree.
    """
    warnings: list[str] = []
    p, k, n = preroll, crossfade, n_in
    short = max(0, n + k + p - appear)
    for name, why in (("P", "settle pre-roll"), ("K", "cross-fade"), ("N_in", "entrance")):
        if short == 0:
            break
        have = {"P": p, "K": k, "N_in": n}[name]
        cut = min(have, short)
        if cut:
            warnings.append(f"{name}: the {why} is cut from {have} to {have - cut} frames so the segment does not start "
                            f"before frame 0")
            short -= cut
            if name == "P":
                p -= cut
            elif name == "K":
                k -= cut
            else:
                n -= cut
    c = appear - n
    s = c - k - p
    if last < appear:
        warnings.append(f"Z: the element is last visible at frame {last}, before appear_frame {appear}; check the "
                        f"element video")
    end = last + tail + k
    if end > frames - 1:
        warnings.append(f"End: {last} + {tail} + {k} = {end} is past the last frame {frames - 1}; clamped to it, so "
                        f"the exit tail and cross-fade are {end - (frames - 1)} frames short")
        end = frames - 1
    length = end - s + 1
    if length > max_len:
        raise SegmentError(f"the segment {s}..{end} is {length} frames, more than max_segment_frames {max_len}. "
                           f"Shorten the effect (appear_frame to the element's last frame), the entrance, the "
                           f"cross-fade or the exit tail, or raise the limit if the graph can take it.")
    return SegmentPlan(appear, n, c, k, p, s, last, tail, end, frames, length, padded_length(length), appear - s,
                       warnings)


def plan_for_cue(cue: Cue, last: int, frames: int, max_len: int = MAX_SEGMENT_FRAMES) -> SegmentPlan:
    """`plan_segment` with A, N_in, K and T from the cue (and its fps) and P = SETTLE_PREROLL."""
    inter = interaction_of(cue)
    return plan_segment(cue.appear_frame, entrance_frames(cue), seconds_to_frames(inter.crossfade_seconds, cue.fps),
                        SETTLE_PREROLL, last, seconds_to_frames(inter.exit_tail_seconds, cue.fps), frames, max_len)
