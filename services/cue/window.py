"""Effect window: which plate frames the effect is made for."""
from __future__ import annotations

import math

from services.cue.cue import MAX_EFFECT_FRAMES, Cue, EffectRange, entrance_start

LEAD_IN = 4  # frames before the orb appears that the window should still cover


def _longest(limit: int) -> int:
    """Largest 8k+1 that is <= limit (at least 1)."""
    return max(1, (limit - 1) // 8 * 8 + 1)


def make_range(effect_range: EffectRange, max_len: int = MAX_EFFECT_FRAMES) -> EffectRange:
    """The frames Make and Merge actually use: `effect_range`, trimmed like `propose_effect_range` trims (keep the
    start, lose the end) when it is longer than `max_len`. `validate` warns when that happens."""
    if effect_range.length <= max_len:
        return effect_range
    return EffectRange(effect_range.start, effect_range.start + _longest(max_len))


def cover_from(effect_range: EffectRange, first: int, frames: int) -> EffectRange:
    """`effect_range` grown earlier by whole 8-frame steps until it starts at or before `first`, so it stays 8k+1.

    Unchanged if it already does. If the grown window would start before frame 0 it starts at 0 and keeps the grown
    length, so it ends later instead; if that runs past the clip it is cut to the longest 8k+1 that fits (`validate`
    warns, both when the window grows and when it is cut).
    """
    if first >= effect_range.start:
        return effect_range
    grow = 8 * math.ceil((effect_range.start - first) / 8)
    if effect_range.start - grow >= 0:
        return EffectRange(effect_range.start - grow, effect_range.end)
    length = effect_range.length + grow
    return EffectRange(0, length if length <= frames else _longest(frames))


def effect_window(cue: Cue, max_len: int = MAX_EFFECT_FRAMES) -> EffectRange:
    """The frames Make and Merge use for this cue: `effect_range` grown to cover the entrance (`cover_from`
    entrance_start), then trimmed by `make_range`. With the entrance off this is `make_range(effect_range)`."""
    er = cue.effect_range
    if cue.appear_frame is not None:
        er = cover_from(er, entrance_start(cue), cue.frames)
    return make_range(er, max_len)


def propose_effect_range(appear_frame: int, end_frame: int, frames: int,
                         max_len: int = MAX_EFFECT_FRAMES) -> tuple[EffectRange, list[str]]:
    """The shortest 8k+1 window that covers appear_frame - 4 .. end_frame (end_frame excluded, like all ranges),
    and warnings.

    `end_frame` is usually `vanish.end`. The window is centred on that span and moved, not shrunk,
    where it would leave the clip 0..frames. If the span needs more than `max_len` frames, or the clip is
    shorter than the window, the window is trimmed to the limit instead (it keeps the start of the span and
    loses the end) and a warning says how many frames are cut. Raises ValueError only when there is
    nothing to cover.
    """
    start, end = max(0, appear_frame - LEAD_IN), min(frames, end_frame)
    span = end - start
    if span < 1:
        raise ValueError(f"nothing to cover: appear_frame {appear_frame}, end_frame {end_frame}, clip has {frames} frames")
    warnings: list[str] = []
    length = (span + 6) // 8 * 8 + 1  # smallest 8k+1 that is >= span
    limit = _longest(min(max_len, frames))
    if length > limit:
        cut = span - limit
        why = (f"the span {start}..{end} is {span} frames and the window can be at most {max_len}" if length > _longest(max_len)
               else f"the clip has {frames} frames, fewer than the {length} the window needs")
        warnings.append(f"effect_range: {why}; trimmed to {limit} frames, so the last {cut} frames of the span "
                        f"({start + limit}..{end}) are outside it. Move appear_frame or vanish.end earlier, or shorten the shot.")
        length, span, end = limit, limit, start + limit
    first = start - (length - span) // 2
    first = min(max(first, 0), frames - length)
    return EffectRange(first, first + length), warnings
