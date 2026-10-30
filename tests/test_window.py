"""Effect window proposal."""
import pytest

from conftest import make_cue
from services.cue.cue import EffectRange, RoutePoint, TrackPoint, Vanish, validate
from services.cue.route import orb_path
from services.cue.window import make_range, propose_effect_range


@pytest.mark.parametrize("appear,end", [(40, 136), (60, 61), (50, 100), (10, 120)])
def test_window_is_8k_plus_1_and_covers_the_span(appear, end):
    r, warnings = propose_effect_range(appear, end, 400)
    assert warnings == []
    assert (r.length - 1) % 8 == 0 and r.length <= 193
    assert r.start <= appear - 4 and r.end >= end
    assert r.length - 8 < end - (appear - 4)  # the shortest one: the next smaller 8k+1 would not cover


def test_window_is_centred_on_the_span():
    r, _ = propose_effect_range(100, 200, 400)  # span 96..200 = 104 frames, window 105
    assert (r.start, r.end) == (96, 201)
    r, _ = propose_effect_range(100, 197, 400)  # span 96..197 = 101 frames, window 105: 4 spare, 2 each side
    assert (r.start, r.end) == (94, 199)


def test_window_is_clamped_to_the_clip_not_shrunk():
    r, _ = propose_effect_range(6, 60, 193)  # span would start at 2; centring wants to start before 0
    assert r.start == 0 and r.length == 65 and r.end >= 60
    r, _ = propose_effect_range(103, 193, 193)  # span 99..193 = 94 frames; centring wants to run past the end
    assert (r.start, r.end) == (96, 193)
    r, _ = propose_effect_range(1, 30, 40)  # appear - 4 is before the clip: the span is clipped to 0..30
    assert (r.start, r.end) == (0, 33)


def test_over_long_window_is_trimmed_to_the_limit_and_warns():
    r, warnings = propose_effect_range(40, 40 - 4 + 207, 400)  # span 207 > 193: 14 frames too long
    assert (r.start, r.end) == (36, 229) and r.length == 193    # keeps the start of the span
    assert len(warnings) == 1 and warnings[0].startswith("effect_range:") and "last 14 frames" in warnings[0]
    assert "229..243" in warnings[0]
    r, warnings = propose_effect_range(40, 40 - 4 + 121 + 8, 400, max_len=121)
    assert r.length == 121 and "last 8 frames" in warnings[0]
    r, warnings = propose_effect_range(40, 90, 400, max_len=57)  # span 54 -> window 57: fits, no warning
    assert r.length == 57 and warnings == []
    r, warnings = propose_effect_range(40, 40 - 4 + 58, 400, max_len=57)
    assert r.length == 57 and "last 1 frames" in warnings[0]
    r, warnings = propose_effect_range(40, 40 - 4 + 193, 400)   # exactly the maximum fits
    assert r.length == 193 and warnings == []


def test_a_max_len_that_is_not_8k_plus_1_trims_to_the_largest_8k_plus_1_below_it():
    r, warnings = propose_effect_range(40, 40 - 4 + 60, 400, max_len=60)
    assert r.length == 57 and warnings


def test_window_longer_than_the_clip_is_trimmed_to_the_clip_and_warns():
    r, warnings = propose_effect_range(5, 30, 30)  # span 1..30 = 29, window 33 > 30: the largest 8k+1 in 30 is 25
    assert (r.start, r.end) == (1, 26) and r.end <= 30
    assert len(warnings) == 1 and "clip has 30 frames" in warnings[0] and "last 4 frames" in warnings[0]


def test_make_range_trims_an_over_long_range_like_the_proposal_does():
    assert make_range(EffectRange(32, 241)) == EffectRange(32, 225)       # 209 -> 193, the start kept
    assert make_range(EffectRange(32, 225)) == EffectRange(32, 225)       # 193 fits
    assert make_range(EffectRange(32, 161), max_len=121) == EffectRange(32, 153)  # the old limit, still a parameter
    assert make_range(EffectRange(10, 59), max_len=40) == EffectRange(10, 43)


def test_orb_path_of_an_over_long_range_covers_the_trimmed_frames():
    cue = make_cue(effect_range=EffectRange(32, 241), vanish=Vanish(150, 160), frames=300,
                   track=[TrackPoint(f, 1000.0, 600.0, "tracked", 80.0, 0.9) for f in range(300)],
                   route=[RoutePoint(107, None, None, 110), RoutePoint(116, 700, 420, 60), RoutePoint(158, 420, 360, 25)])
    errors, warnings = validate(cue)
    assert errors == [] and any(w.startswith("effect_range:") for w in warnings)
    assert [s[0] for s in orb_path(cue)] == list(range(32, 225))          # degrades: no route point is an error


def test_nothing_to_cover_is_still_an_error():
    with pytest.raises(ValueError, match="nothing to cover"):
        propose_effect_range(50, 40, 400)
