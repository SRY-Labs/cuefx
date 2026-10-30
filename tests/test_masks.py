"""Occluder masks from hand landmarks: which chain, hull, dilation and feather, gap interpolation and warnings."""
import cv2
import numpy as np
import pytest

from conftest import make_cue
from services.cue.cue import Behind
from services.analysis.masks import (MaskConfig, fill_landmark_gaps, hand_mask, hand_size, hull, masks_for_frame, occluder_chain,
                         occluder_tracks, palm_width, smoothed_sizes)

SHAPE = (400, 600)  # h, w


def hand(cx=300.0, cy=200.0, pw=80.0):
    """21 hand-like landmarks: wrist, then five fingers fanning upwards from a palm `pw` wide (5 to 17)."""
    pts = [(cx, cy + 0.9 * pw)]
    for k, ang in enumerate(np.linspace(-0.9, 0.9, 5)):
        base = np.array([cx + (k - 2) / 4 * pw * (1.3 if k == 0 else 1.0), cy])
        d = np.array([np.sin(ang), -np.cos(ang)])
        pts += [tuple(base + d * pw * s) for s in (0.0, 0.35, 0.65, 0.9)]
    xy = np.array(pts)
    xy[[5, 17]] = [(cx - pw / 2, cy), (cx + pw / 2, cy)]  # index and pinky knuckles: the palm width
    return xy


def test_chain_is_picked_from_the_cue_hand_not_a_label():
    assert occluder_chain("left_hand", "right") == "other"
    assert occluder_chain("right_hand", "right") == "chosen"
    assert occluder_chain("left_hand", "left") == "chosen"
    assert occluder_chain("person", "right") is None
    assert occluder_chain({"point": [80, 1, 2]}, "right") is None


def test_hull_contains_every_landmark_and_the_mask_covers_them():
    xy = hand()
    poly = hull(xy).astype(np.float32)
    assert all(cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in xy)
    m = hand_mask(xy, SHAPE)
    assert m.dtype == np.float32 and m.shape == SHAPE
    assert all(m[int(y), int(x)] > 0.9 for x, y in xy)                 # fingertips are sharp hull corners: ~0.92


def test_mask_is_dilated_by_025_and_feathered_by_006_hand_sizes():
    xy, cfg = hand(pw=80.0), MaskConfig()
    assert palm_width(xy) == pytest.approx(80.0) and hand_size(xy) == pytest.approx(80.0)  # wrist to 9 is 72
    m = hand_mask(xy, SHAPE, cfg)
    wrist_y = int(xy[0, 1])                                             # the hull's lowest point, straight below it
    r, sigma = 0.25 * 80, 0.06 * 80
    assert m[wrist_y + int(r), 300] == pytest.approx(0.5, abs=0.15)     # the 50 % edge sits about r outside the hull
    assert m[wrist_y + int(r + 4 * sigma), 300] < 0.01                  # gone a few sigma further out
    assert np.abs(np.diff(m, axis=0)).max() < 0.1                       # feathered: no hard edge anywhere
    assert np.abs(np.diff(m, axis=1)).max() < 0.1
    assert m[0, 0] == 0


def side_on(xy, squeeze=0.4):
    """The same hand turned side-on: everything squeezed towards its centre line in x, so 5 -> 17 is 0.4x."""
    out = xy.copy()
    out[:, 0] = xy[0, 0] + squeeze * (xy[:, 0] - xy[0, 0])
    return out


def mask_width(m, row):
    """Width in px of the mask's >= 50 % part on one row."""
    return int((m[row] >= 0.5).sum())


def test_a_side_on_hand_gets_a_mask_at_least_as_thick_around_it_as_face_on():
    face = hand(pw=80.0)
    face[0] = (300.0, 200.0 + 80.0)                                     # wrist to middle knuckle = palm width
    side = side_on(face)
    assert palm_width(side) == pytest.approx(32.0) and hand_size(side) == pytest.approx(hand_size(face))
    row = 200 + 40                                                      # across the palm
    for xy in (face, side):
        hull_w = mask_width(hand_mask(xy, SHAPE, MaskConfig(dilate=0, feather=0)), row)  # the bare hull on that row
        margin = (mask_width(hand_mask(xy, SHAPE), row) - hull_w) / 2  # how far the mask reaches past the joints
        assert margin >= 0.25 * 80 - 2, (margin, hull_w)
    old = MaskConfig(dilate=0.15, feather=0.08)                          # sized by palm width, as before: collapses
    assert mask_width(hand_mask(side, SHAPE, old, size=palm_width(side)), row) < mask_width(hand_mask(side, SHAPE), row) - 20


def test_hand_size_is_the_median_of_the_last_5_frames_with_landmarks():
    face, side = hand(pw=80.0), side_on(hand(pw=80.0), 0.2)
    sets = [[face]] * 5 + [[]] + [[side]] + [[face]]                    # one collapsed frame after a gap
    sizes = smoothed_sizes(sets, 5)
    assert sizes[5] is None
    assert sizes[6] == pytest.approx(hand_size(face))                   # one odd frame does not shrink the mask
    assert hand_size(side) < hand_size(face)


def test_mask_scales_with_the_palm():
    small, big = hand_mask(hand(pw=40.0), SHAPE), hand_mask(hand(pw=80.0), SHAPE)
    assert big.sum() / small.sum() == pytest.approx(4.0, rel=0.1)       # area grows with palm width squared


def test_mask_near_the_frame_edge_is_cut_not_an_error():
    m = hand_mask(hand(cx=5.0, cy=5.0), SHAPE)
    assert m.shape == SHAPE and m[0, 0] > 0.9


def lms_with_gap(a, b, n=200):
    """(n, 21, 3) landmarks of a hand moving right 2 px per frame, NaN in frames [a, b)."""
    lms = np.stack([np.column_stack([hand(cx=200 + 2.0 * f), np.zeros(21)]) for f in range(n)])
    lms[a:b] = np.nan
    return lms


def test_gaps_of_up_to_6_frames_are_interpolated():
    filled, left = fill_landmark_gaps(lms_with_gap(82, 88), 6)          # 6 frames missing
    assert left == []
    np.testing.assert_allclose(filled[85], lms_with_gap(0, 0)[85], atol=1e-9)  # linear motion comes back exactly
    cue = make_cue(behind=[Behind(80, 91, "left_hand")])
    tracks, warnings = occluder_tracks(cue, {"chosen": lms_with_gap(0, 0), "other": lms_with_gap(82, 88)})
    assert warnings == []
    assert all(tracks[0].points[f] for f in range(80, 91))
    assert len(masks_for_frame(tracks, 85, SHAPE)) == 1 and masks_for_frame(tracks, 95, SHAPE) == []


def test_a_longer_gap_gets_no_mask_and_a_warning_naming_the_frames():
    cue = make_cue(behind=[Behind(80, 91, "left_hand")])
    tracks, warnings = occluder_tracks(cue, {"chosen": lms_with_gap(0, 0), "other": lms_with_gap(83, 90)})  # 7
    assert [f for f, p in tracks[0].points.items() if not p] == list(range(83, 90))
    assert len(warnings) == 1 and warnings[0].startswith("behind[0]") and "83..89" in warnings[0]
    assert masks_for_frame(tracks, 85, SHAPE) == []


def test_a_gap_at_the_clip_edge_cannot_be_interpolated():
    filled, left = fill_landmark_gaps(lms_with_gap(0, 3), 6)
    assert left == [(0, 3)] and np.isnan(filled[0]).all()


def test_person_and_point_occluders_warn_and_have_no_mask():
    cue = make_cue(behind=[Behind(80, 85, "person"), Behind(86, 91, {"point": [86, 10, 20]})])
    tracks, warnings = occluder_tracks(cue, {"chosen": lms_with_gap(0, 0), "other": lms_with_gap(0, 0)})
    assert len(warnings) == 2 and all("SAM" in w for w in warnings)
    assert all(not p for t in tracks for p in t.points.values())


def test_missing_landmarks_file_warns():
    tracks, warnings = occluder_tracks(make_cue(), None)
    assert len(warnings) == 1 and "no landmarks" in warnings[0]
    assert all(not p for p in tracks[0].points.values())


# --------------------------------------------------------------------------- non-chosen candidates (task 04b)

def candidates(n=200, other_hand_at=(500.0, 150.0), on_palm=False):
    """landmarks.npz arrays: the chosen hand at the synthetic track's palm (1000, 600), the other chain empty,
    and a dropped second candidate: another hand at `other_hand_at`, or (on_palm) a duplicate on the chosen palm."""
    chosen = np.stack([np.column_stack([hand(cx=1000.0, cy=600.0, pw=100.0), np.zeros(21)])] * n)
    second = hand(cx=1000.0 + 10, cy=600.0 + 5, pw=100.0) if on_palm else hand(*other_hand_at, pw=80.0)
    cand = np.full((n, 2, 21, 3), np.nan)
    cand[:, 0] = chosen
    cand[:, 1, :, :2] = second
    cand[:, 1, :, 2] = 0
    is_chosen = np.zeros((n, 2), bool)
    is_chosen[:, 0] = True
    return {"chosen": chosen, "other": np.full((n, 21, 3), np.nan), "cand_lm": cand, "cand_is_chosen": is_chosen}


def test_a_dropped_non_chosen_candidate_gives_a_mask_when_the_other_chain_is_empty():
    cue = make_cue(behind=[Behind(80, 91, "left_hand")])               # the cue's hand is right: left = non-chosen
    tracks, warnings = occluder_tracks(cue, candidates())
    assert warnings == []
    assert all(len(tracks[0].points[f]) == 1 for f in range(80, 91))
    m = masks_for_frame(tracks, 85, (1080, 1920))[0]
    assert m[150, 500] > 0.9 and m[600, 1000] == 0                     # the other hand, not the chosen palm


def test_a_candidate_on_top_of_the_chosen_palm_is_ignored():
    cue = make_cue(behind=[Behind(80, 91, "left_hand")])
    tracks, warnings = occluder_tracks(cue, candidates(on_palm=True))
    assert all(not p for p in tracks[0].points.values())
    assert len(warnings) == 1 and "80..90" in warnings[0]


def test_the_other_chain_wins_where_it_has_the_frame():
    lms = candidates()
    lms["other"][85] = np.column_stack([hand(cx=300.0, cy=300.0, pw=80.0), np.zeros(21)])
    tracks, _ = occluder_tracks(make_cue(behind=[Behind(80, 91, "left_hand")]), lms)
    np.testing.assert_allclose(tracks[0].points[85][0], hand(cx=300.0, cy=300.0, pw=80.0))
    np.testing.assert_allclose(tracks[0].points[84][0], hand(500.0, 150.0, pw=80.0))


def test_the_chosen_hand_can_occlude_its_own_orb():
    tracks, warnings = occluder_tracks(make_cue(behind=[Behind(80, 91, "right_hand")]), candidates())
    assert warnings == []
    np.testing.assert_allclose(tracks[0].points[85][0], hand(cx=1000.0, cy=600.0, pw=100.0))


def test_two_non_chosen_candidates_make_a_union_of_hulls():
    lms = candidates()
    lms["cand_is_chosen"][:, 0] = False                                 # chosen hand not detected: both are others
    lms["cand_lm"][:, 0, :, :2] = hand(cx=200.0, cy=800.0, pw=80.0)
    tracks, _ = occluder_tracks(make_cue(behind=[Behind(80, 91, "left_hand")]), lms)
    m = masks_for_frame(tracks, 85, (1080, 1920))[0]
    assert m[150, 500] > 0.9 and m[800, 200] > 0.9
