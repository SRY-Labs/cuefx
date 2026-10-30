"""Tracking tests on synthetic data only: no video, no model, no network."""
from dataclasses import replace

import numpy as np
import pytest

from services.cue.cue import Correction, Cue, HandSeed, TrackPoint, load_cue, save_cue
from services.analysis.track import (Hand, OneEuroConfig, TrackConfig, TrackError, VideoInfo, apply_corrections, build_track,
                         fill_gaps, missing_runs, select_hands, smooth_track)

PW = 100.0  # palm width of the synthetic hands


def hand(cx: float, cy: float = 500.0, label: str = "Right", w: float = PW, score: float = 0.9) -> Hand:
    """A hand whose palm centre is (cx, cy) and palm width w."""
    xyz = np.zeros((21, 3))
    xyz[:] = (cx, cy, 0)
    xyz[5] = (cx - w / 2, cy, 0)
    xyz[17] = (cx + w / 2, cy, 0)
    return Hand(xyz, label, score)


def pts(xs, source="tracked") -> list[TrackPoint]:
    return [TrackPoint(f, float(x), 100.0, source, PW, 0.9) for f, x in enumerate(xs)]


def lost_pts(xs, a: int, b: int) -> list[TrackPoint]:
    """pts(xs) with frames [a, b) interpolated and flagged lost."""
    return [replace(p, source="interpolated", lost=True) if a <= p.f < b else p for p in pts(xs)]


def chosen_x(hands, sel) -> list[float | None]:
    return [None if c is None else float(hands[f][c].palm_center[0]) for f, c in enumerate(sel.chosen)]


def track(hands, seed_x, seed_f=0):
    sel = select_hands(hands, HandSeed(seed_f, seed_x, 500))
    info = VideoInfo(1920, 1080, 24.0, len(hands), len(hands))
    return sel, build_track(info, hands, sel)


def test_every_candidate_is_kept_for_the_masks():
    hands = [[hand(500), hand(900, label="Left")] for _ in range(10)] + [[hand(500)] for _ in range(5)]
    sel, res = track(hands, 500)
    assert res.cand_lm.shape == (15, 2, 21, 3) and res.cand_is_chosen.shape == (15, 2)
    assert res.cand_is_chosen[:, 0].all() and not res.cand_is_chosen[:, 1].any()
    np.testing.assert_array_equal(res.cand_lm[3, 1], hands[3][1].xyz)
    assert np.isnan(res.cand_lm[12, 1]).all()                           # one hand only: the second slot is empty


# --------------------------------------------------------------------------- gaps and smoothing

def test_missing_runs_and_interpolation_fill_gap():
    valid = np.array([1, 1, 0, 0, 0, 1, 0], dtype=bool)
    assert missing_runs(valid) == [(2, 5), (6, 7)]
    xy = np.array([[0, 10], [10, 20], [0, 0], [0, 0], [0, 0], [50, 60], [0, 0]], dtype=float)
    filled = fill_gaps(xy, valid)
    np.testing.assert_allclose(filled[2:5], [[20, 30], [30, 40], [40, 50]])
    np.testing.assert_allclose(filled[6], [50, 60])  # after the last accepted frame: held
    leading = fill_gaps(np.array([0, 0, 5.0, 7.0]), np.array([0, 0, 1, 1], dtype=bool))
    np.testing.assert_allclose(leading, [5, 5, 5, 7])  # before the first accepted frame: held


def test_smoothing_reduces_jitter_without_lag():
    rng = np.random.default_rng(0)
    n, speed = 120, 2.0  # px per frame along x
    line = np.stack([np.arange(n) * speed, np.full(n, 300.0)], axis=1)
    noisy = line + rng.normal(0, 3.0, line.shape)
    smooth = smooth_track(noisy, np.ones(n, dtype=bool), 24.0, PW, OneEuroConfig())
    jitter = lambda a: float(np.std(np.diff(a, axis=0)))  # noqa: E731
    assert jitter(smooth) < 0.5 * jitter(noisy)
    assert abs(float(np.mean(smooth[:, 0] - line[:, 0]))) < 0.5  # signed error along the motion: no lag


def test_smoothing_never_crosses_a_gap():
    xy = np.array([[0.0, 0.0]] * 10 + [[0.0, 0.0]] * 5 + [[500.0, 0.0]] * 10)
    tracked = np.array([True] * 10 + [False] * 5 + [True] * 10)
    out = smooth_track(xy, tracked, 24.0, PW, OneEuroConfig())
    np.testing.assert_allclose(out[15:, 0], 500.0)   # the second run is not dragged towards the first
    np.testing.assert_allclose(out[10:15, 0], np.linspace(0, 500, 7)[1:-1])  # the gap stays linear


# --------------------------------------------------------------------------- corrections

def test_corrections_between_keyframes_and_falloff():
    cfg = TrackConfig(correction_falloff=2)
    out = apply_corrections(pts([100] * 12), [Correction(4, 120, 100), Correction(8, 100, 100)], cfg)
    # offset +20 at frame 4 falling linearly to 0 at frame 8, and fading over 2 frames before frame 4
    np.testing.assert_allclose([p.x for p in out], [100, 100, 100, 110, 120, 115, 110, 105, 100, 100, 100, 100])
    # only frames with a non-zero offset are marked (the keyframe at 8 has offset 0)
    assert [p.source for p in out] == ["tracked"] * 3 + ["corrected"] * 5 + ["tracked"] * 4


def test_single_correction_moves_a_window_with_linear_fade():
    track_ = pts([100] * 60)
    out = apply_corrections(track_, [Correction(30, 110, 100)])  # default falloff 12, offset +10
    for f, p in enumerate(out):
        expected = 10 * max(0.0, 1 - abs(f - 30) / 12)
        assert p.x == pytest.approx(100 + expected), f
        assert p.source == ("corrected" if expected else "tracked"), f
    assert out[30].x == 110 and out[18].x == 100 and out[42].x == 100
    assert apply_corrections(track_, []) == track_


def test_correction_in_a_lost_gap_is_an_anchor_for_that_gap_only():
    track_ = lost_pts([100] * 60, 20, 30)
    out = apply_corrections(track_, [Correction(25, 200, 120)])
    assert out[25].x == 200 and out[25].y == 120
    # the gap runs straight from the last tracked frame (19) through the anchor to the first one after it (30)
    np.testing.assert_allclose([out[f].x for f in range(20, 30)],
                               np.interp(range(20, 30), [19, 25, 30], [100, 200, 100]))
    assert all(p.source == "corrected" for p in out[20:30])
    assert out[:20] == track_[:20] and out[30:] == track_[30:]   # no falloff: nothing outside the gap moves


def test_in_gap_anchors_interpolate_through_every_anchor():
    track_ = lost_pts([0] * 40, 10, 20)
    out = apply_corrections(track_, [Correction(12, 50, 100), Correction(16, 90, 100)])
    np.testing.assert_allclose([out[f].x for f in range(9, 21)],
                               np.interp(range(9, 21), [9, 12, 16, 20], [0, 50, 90, 0]))


def test_in_gap_correction_at_the_clip_edge_holds_the_nearest_side():
    track_ = lost_pts([0] * 20, 15, 20)  # the gap runs to the end of the clip
    out = apply_corrections(track_, [Correction(17, 30, 100)])
    assert [out[f].x for f in range(15, 20)] == pytest.approx(np.interp(range(15, 20), [14, 17], [0, 30]))


def test_falloff_from_a_tracked_correction_stops_at_a_lost_gap():
    track_ = lost_pts([100] * 60, 30, 40)
    out = apply_corrections(track_, [Correction(27, 130, 100)])  # +30, three frames before the gap
    assert [out[f].x for f in range(24, 30)] == pytest.approx([100 + 30 * (1 - abs(f - 27) / 12) for f in range(24, 30)])
    assert out[29].source == "corrected"
    assert out[30:] == track_[30:]                    # the gap and everything after it: untouched
    assert out[15].x == 100 and out[16].x > 100       # the fade is 12 frames long, but cut at the gap


def test_falloff_stops_at_the_far_edge_of_a_gap_too():
    track_ = lost_pts([100] * 60, 20, 30)
    out = apply_corrections(track_, [Correction(32, 130, 100)])
    assert out[:30] == track_[:30]                    # does not reach back across the gap
    assert out[31].x > 100 and out[44].x == 100


def test_corrections_either_side_of_a_gap_do_not_interpolate_across_it():
    track_ = lost_pts([100] * 60, 20, 30)
    out = apply_corrections(track_, [Correction(15, 120, 100), Correction(35, 80, 100)])
    assert out[15].x == 120 and out[35].x == 80
    assert out[19].x == pytest.approx(100 + 20 * (1 - 4 / 12))   # fading after 15, as if it were the last keyframe
    assert out[20:30] == track_[20:30]


# --------------------------------------------------------------------------- identity

def overlap_scene(blob_frame=None, same_label=True):
    """A (chosen) comes from the left, B from the right; for 10 frames they overlap and only B is detected."""
    n = 50
    ax = [200 + 10 * min(f, 9) if f < 30 else 290 - 10 * (f - 29) for f in range(n)]  # 200 -> 290 -> back left
    bx = [600 - 30 * min(f, 9) if f < 30 else 330 + 10 * (f - 29) for f in range(n)]  # 600 -> 330 -> back right
    a_label, b_label = ("Right", "Right") if same_label else ("Right", "Left")
    hands = []
    for f in range(n):
        if f == blob_frame:  # the two palms merge into one blob, much narrower than either
            hands.append([hand(305, w=0.6 * PW, label=b_label)])
        elif 15 <= f < 25:   # B covers A: only B is detected
            hands.append([hand(bx[f] - 10, label=b_label)])
        else:
            hands.append([hand(ax[f], label=a_label), hand(bx[f], label=b_label)])
    return hands, ax


@pytest.mark.parametrize("same_label", [True, False])
def test_identity_does_not_switch_when_hands_overlap(same_label):
    hands, ax = overlap_scene(same_label=same_label)
    sel, res = track(hands, 200)
    xs = chosen_x(hands, sel)
    assert all(x is None for x in xs[15:25])  # B's detection was not taken as A
    for f in list(range(15)) + list(range(25, len(hands))):
        assert xs[f] == ax[f], f"chosen identity switched at frame {f}"
    assert res.ambiguous[9:30].all()  # hands within one palm width: flagged
    assert not res.ambiguous[:8].any()


def test_merged_blob_in_the_overlap_is_not_taken():
    hands, ax = overlap_scene(blob_frame=20)
    sel, res = track(hands, 200)
    xs = chosen_x(hands, sel)
    assert all(x is None for x in xs[15:25])
    assert sel.tracklets[sel.cand_tracklet[20][0]].chain == "dropped"
    for f in list(range(15)) + list(range(25, len(hands))):
        assert xs[f] == ax[f], f"chosen identity switched at frame {f}"


def test_other_hand_appearing_after_the_seed_frame():
    """The case that failed on the clip: one hand in the seed frame, the other arrives, covers the first, leaves."""
    n, hands = 60, []
    for f in range(n):
        bx = 700 - 20 * (f - 10) if 10 <= f < 30 else 320 + 20 * (f - 39)  # B: approaches A, covers it, leaves
        cands = []
        if not 30 <= f < 40:           # A (static, label Right) is hidden while B covers it
            cands.append(hand(300, label="Right"))
        if 10 <= f < 30 or f >= 30:    # B (label Left) appears at frame 10; frames 30-39 it hovers over A
            cands.append(hand(bx if f < 30 or f >= 40 else 305, label="Left"))
        hands.append(cands)
    sel, res = track(hands, 300)
    for f, c in enumerate(sel.chosen):
        if c is not None:
            assert hands[f][c].label == "Right", f"chosen took the other hand at frame {f}"
    assert all(c is None for c in sel.chosen[30:40])
    assert sel.reference_label == "Right"
    assert res.ambiguous[30:40].all()  # A is missing and B sits on its interpolated position


def test_short_spurious_detection_is_dropped():
    hands = [[hand(100 + 5 * f)] for f in range(10)]
    hands[6] = [hand(900)]  # a lone detection far from the hand
    sel, _ = track(hands, 100)
    xs = chosen_x(hands, sel)
    assert xs[6] is None
    assert xs[5] == 125 and xs[7] == 135


def test_one_frame_false_detection_does_not_start_the_other_chain():
    """A lone 1-frame detection beside the chosen hand is noise, not the other hand."""
    hands = [[hand(100 + 5 * f)] for f in range(12)]
    hands[6].append(hand(900))   # overlaps the chosen hand in time, far away
    sel, res = track(hands, 100)
    false_tl = sel.tracklets[sel.cand_tracklet[6][1]]
    assert false_tl.n == 1 and false_tl.chain == "dropped"
    assert all(o is None for o in sel.other)
    assert not np.isnan(res.chosen).any()  # the chosen hand is untouched


def reacquire_scene(label):
    """A hand that vanishes for 4 frames and comes back 900 px away."""
    return [[hand(300)] for _ in range(20)] + [[] for _ in range(4)] + [[hand(1200, label=label)] for _ in range(12)]


def test_reacquisition_needs_a_matching_label_and_marks_the_gap_lost():
    hands = reacquire_scene("Right")
    sel, res = track(hands, 300)
    assert all(c is not None for c in sel.chosen[24:])
    assert res.gaps == [(20, 24)] and res.lost == [(20, 24)]  # short, but a jump: lost
    assert all(p.source == "interpolated" for p in res.points[20:24])
    assert [p.lost for p in res.points] == [20 <= f < 24 for f in range(len(res.points))]  # the flag marks the gap

    hands = reacquire_scene("Left")
    sel, res = track(hands, 300)
    assert all(c is None for c in sel.chosen[20:])
    assert res.lost == [(20, 36)]
    assert [p.f for p in res.points if p.lost] == list(range(20, 36))


def test_short_ordinary_gap_is_interpolated_but_not_lost():
    hands = [[hand(300)] for _ in range(10)] + [[] for _ in range(3)] + [[hand(300)] for _ in range(10)]
    _, res = track(hands, 300)
    assert res.gaps == [(10, 13)] and res.lost == []
    assert all(p.source == "interpolated" and not p.lost for p in res.points[10:13])


def test_long_gap_is_lost_by_length():
    hands = [[hand(300)] for _ in range(10)] + [[] for _ in range(7)] + [[hand(300)] for _ in range(10)]
    _, res = track(hands, 300)
    assert res.lost == [(10, 17)] and all(p.lost for p in res.points[10:17])
    assert not any(p.lost for p in res.points[:10] + res.points[17:])


def small_wrong_label_scene():
    """A (Right, palm width PW) is hidden for frames 20-29; a 3-frame blip 30 px away, 0.35 x the width and labelled Left, shows up."""
    return ([[hand(300)] for _ in range(20)] + [[], []]
            + [[hand(330, w=0.35 * PW, label="Left")] for _ in range(3)] + [[] for _ in range(5)]
            + [[hand(300)] for _ in range(10)])


def test_short_tracklet_with_the_wrong_width_and_label_is_rejected_by_the_label_gate():
    hands = small_wrong_label_scene()
    sel, _ = track(hands, 300)
    assert all(c is None for c in sel.chosen[20:30])                # nothing joins the chosen chain in the gap
    assert [c is not None for c in sel.chosen[:20] + sel.chosen[30:]] == [True] * 30
    # without the label gate the blip is inside the distance gate and is the cheapest (only) fit
    off = TrackConfig(vote_gate=2.0)
    assert any(c is not None for c in select_hands(hands, HandSeed(0, 300, 500), off).chosen[22:25])


def test_a_short_tracklet_with_the_wrong_label_is_rejected_even_at_the_right_width():
    blip = [[hand(300)] for _ in range(20)] + [[], []] + [[hand(330, label="Left")] for _ in range(3)] + [[] for _ in range(5)] + [[hand(300)] for _ in range(10)]
    sel, _ = track(blip, 300)
    assert all(c is None for c in sel.chosen[20:30])


def test_palm_width_is_no_hard_gate_the_hand_may_shrink_across_a_gap():
    """The throw: the palm goes from PW to 0.3 PW while the hand is lost for 4 frames, and it is still the hand."""
    hands = [[hand(300)] for _ in range(20)] + [[] for _ in range(4)] + [[hand(330, w=0.3 * PW)] for _ in range(10)]
    sel, _ = track(hands, 300)
    assert all(c is not None for c in sel.chosen[24:])


def test_a_long_tracklet_outvotes_wrong_labels():
    hands = [[hand(300)] for _ in range(20)] + [[hand(310, label="Left")] for _ in range(10)] + [[hand(320)] for _ in range(10)]
    sel, _ = track(hands, 300)
    assert all(c is not None for c in sel.chosen)                   # 10 frames >= 8: the label gate does not apply


# --------------------------------------------------------------------------- seed and cue file

def test_seed_that_misses_every_palm_uses_the_nearest_hand_and_warns():
    hands = [[hand(100), hand(900)] for _ in range(5)]
    sel = select_hands(hands, HandSeed(0, 100 + 2.5 * PW, 500))   # 2.5 palm widths from hand 100: too far
    assert len(sel.warnings) == 1 and sel.warnings[0].startswith("hand_seed:") and "using that hand at (100, 500)" in sel.warnings[0]
    assert [c for c in sel.chosen] == [0] * 5                      # followed hand 100, not 900
    sel = select_hands(hands, HandSeed(0, 5000, 5000))              # nowhere near either: the nearest is still used
    assert sel.warnings and sel.chosen == [1] * 5
    assert select_hands(hands, HandSeed(0, 100 + 1.9 * PW, 500)).warnings == []   # inside the radius: no warning


def test_seed_errors_only_when_there_is_nothing_to_follow():
    hands = [[hand(100)] for _ in range(3)]
    with pytest.raises(TrackError, match="no hand"):
        select_hands([[], [hand(100)]], HandSeed(0, 100, 500))
    with pytest.raises(TrackError, match="outside"):
        select_hands(hands, HandSeed(7, 100, 500))


def test_cue_roundtrip_and_validation(tmp_path):
    cue = Cue(clip="a.mov", fps=24, frames=3, plate_size=(1920, 1080), hand="right",
              hand_seed=HandSeed(1, 10, 20), track=pts([1, 2, 3]), corrections=[Correction(2, 5, 6)],
              correction_falloff=8, extra={"ui_note": "keep me"})
    save_cue(cue, tmp_path / "c.json")
    assert load_cue(tmp_path / "c.json") == cue
    bad = [
        Cue("a", 24, 3, (1, 1), "right", HandSeed(1, 0, 0), pts([1, 2, 3], source="guessed")),
        Cue("a", 24, 3, (1, 1), "right", HandSeed(3, 0, 0)),
        Cue("a", 24, 3, (1, 1), "right", HandSeed(0, 0, 0), corrections=[Correction(3, 0, 0)]),
        Cue("a", 24, 3, (1, 1), "middle", HandSeed(0, 0, 0)),
        Cue("a", 24, 3, (1, 1), "right", HandSeed(0, 0, 0), track=pts([1, 2])),
        Cue("a", 24, 3, (1, 1), "right", HandSeed(0, 0, 0), correction_falloff=-1),
    ]
    for c in bad:
        with pytest.raises(ValueError):
            save_cue(c, tmp_path / "bad.json")
