"""Exact timing and ownership checks for the experimental LTX overlap path."""
from dataclasses import replace
import hashlib

import cv2
import numpy as np
import pytest

from conftest import make_cue
from services.cue.cue import EffectRange, Entrance, RoutePoint, TrackPoint, Vanish
from services.delivery.video import VideoWriter, read_frames
from services.make.ltx_overlap import (assemble, extract_layout_reference, padded_count, plan_overlap,
                                prepare_full_scene_reference_record, prepare_reference_record,
                                prepare_reverse_entrance, reverse_source,
                                select_reference_frame)
from services.make.ltx_overlap_graph import (GRAPH_DIR, OFFICIAL_GRAPH, fill_image_edit_api, fill_main_api,
                                      fill_main_graph, fill_reverse_api, official_graph)


def test_delayed_example_and_ownership():
    cue = make_cue(entrance=Entrance(.5), effect_range=EffectRange(34, 115),
                   vanish=Vanish(111, 113), release_frame=108,
                   route=[RoutePoint(108, None, None, 110), RoutePoint(110, 700, 420, 60),
                          RoutePoint(112, 420, 360, 25)])
    plan = plan_overlap(cue)
    assert (plan.window_start, plan.window_end, plan.zero_frame, plan.appear_frame) == (26, 115, 28, 40)
    assert (plan.look_source_plate_frame, plan.look_source_window_index) == (40, 14)
    assert (plan.anchor_frame, plan.context_count, plan.reverse_pre_start) == (56, 17, 26)
    assert (plan.main_real_count, plan.main_padded_count) == (75, 81)
    assert (plan.reverse_real_count, plan.reverse_padded_count, plan.first_empty_cell) == (31, 41, 5)
    layout = list(range(26, 115))
    reverse_layout = reverse_source(layout, plan)
    assert reverse_layout[:31] == list(range(56, 25, -1))
    assert reverse_layout[31:] == [26] * 10
    # Encode ownership in the values; the generated middle is intentionally distinct.
    main = [("main", f) for f in range(40, 115)] + [("pad", 114)] * 6
    reverse = [("reverse", f) for f in range(56, 25, -1)] + [("pad", 26)] * 10
    clean = [("clean", f) for f in range(26, 115)]
    out = assemble(clean, main, reverse, plan)
    assert len(out) == 89
    assert out == [("clean" if f <= 28 else "reverse" if f < 40 else "main", f)
                   for f in range(26, 115)]


def test_later_saved_reference_keeps_original_appear():
    cue = make_cue(entrance=Entrance(.5))
    p = plan_overlap(cue, 43)
    assert (p.appear_frame, p.look_source_plate_frame, p.look_source_window_index) == (40, 43, 19)
    assert assemble(list(range(p.window_start, p.window_end)),
                    list(range(43, p.window_end)),
                    list(range(p.anchor_frame, p.reverse_pre_start - 1, -1)), p) == list(range(p.window_start, p.window_end))
    assert p.selection_reason == "recorded look provenance"


def test_reverse_preparation_uses_actual_main_anchor_and_no_plate_padding():
    cue = make_cue(entrance=Entrance(.5))
    plan = plan_overlap(cue, 43)
    layout = [("layout", f) for f in range(plan.window_start, plan.window_end)]
    main = [("generated", f) for f in range(43, plan.window_end)]
    original_layout, original_main = list(layout), list(main)

    prepared = prepare_reverse_entrance(layout, main, plan)

    assert prepared.matched_look == ("generated", plan.anchor_frame)
    assert prepared.fixed_generated_prefix == tuple(
        ("generated", f) for f in range(plan.anchor_frame, 42, -1))
    assert prepared.prefix_plate_frames == tuple(range(plan.anchor_frame, 42, -1))
    assert prepared.guide_frames[:plan.reverse_real_count] == tuple(
        ("layout", f) for f in range(plan.anchor_frame, plan.reverse_pre_start - 1, -1))
    assert prepared.guide_frames[plan.reverse_real_count:] == (
        ("layout", plan.reverse_pre_start),) * (plan.reverse_padded_count - plan.reverse_real_count)
    assert prepared.guide_plate_frames == (
        tuple(range(plan.anchor_frame, plan.reverse_pre_start - 1, -1))
        + (None,) * (plan.reverse_padded_count - plan.reverse_real_count))
    assert prepared.clean_source_frames == prepared.guide_frames
    assert layout == original_layout and main == original_main

    with pytest.raises(ValueError, match="missing real frames"):
        prepare_reverse_entrance(layout, main[:-1], plan)


def test_bypass_and_invalid_provenance():
    assert plan_overlap(make_cue()) is None
    cue = make_cue(appear_frame=0, entrance=Entrance(.5), effect_range=EffectRange(0, 153))
    assert plan_overlap(cue) is None
    cue = make_cue(entrance=Entrance(.5))
    with pytest.raises(ValueError, match="matching look frame"):
        plan_overlap(cue, 25)
    with pytest.raises(ValueError, match="visible, unclipped"):
        plan_overlap(cue, 108)


def test_short_context_and_budget_failure():
    cue = make_cue(entrance=Entrance(.5), release_frame=41)
    cue.route[0] = replace(cue.route[0], f=41)
    cue.route[1] = replace(cue.route[1], f=42)
    cue.route[2] = replace(cue.route[2], f=43)
    p = plan_overlap(cue)
    assert p.context_count == 1 and p.anchor_frame == 40
    assert p.reverse_padded_count >= p.reverse_real_count
    with pytest.raises(ValueError, match="budget"):
        plan_overlap(cue, max_frames=9)


def test_missing_entrance_coverage_and_exact_counts():
    cue = make_cue(entrance=Entrance(.5), effect_range=EffectRange(40, 48))
    # Invalid 8k+1 range/insufficient coverage is rejected by the existing cue validator.
    with pytest.raises(ValueError):
        plan_overlap(cue)
    assert [padded_count(x) for x in (1, 2, 8, 9, 10, 17, 31)] == [1, 9, 9, 9, 17, 17, 33]
    with pytest.raises(ValueError):
        padded_count(0)


def test_selection_prefers_first_full():
    cue = make_cue(entrance=Entrance(.5))
    assert select_reference_frame(cue)[0] == cue.appear_frame


def test_extracted_reference_is_exact_encoded_frame_with_provenance(tmp_path):
    cue = replace(make_cue(entrance=Entrance(.5)), canvas_size=(256, 128),
                  canvas_rect=(0, 0, 1920, 960))
    plan = plan_overlap(cue)
    video = tmp_path / "layout.mp4"
    with VideoWriter(video, cue.canvas_size, cue.fps, crf=0) as writer:
        for index in range(plan.final_count):
            frame = np.zeros((128, 256, 3), dtype=np.uint8)
            frame[..., 0] = index
            frame[..., 1] = 255 - index
            writer.write(frame)
    png = tmp_path / "reference.png"
    record = prepare_reference_record(cue, video, png)
    decoded = list(read_frames(video))
    assert np.array_equal(cv2.imread(str(png)), decoded[plan.look_source_window_index])
    assert record["look_source_plate_frame"] == cue.appear_frame
    assert record["look_source_window_index"] == cue.appear_frame - plan.window_start
    assert record["extracted_frame_sha256"] == hashlib.sha256(png.read_bytes()).hexdigest()
    assert record["layout_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert record["canvas_mapping"]["canvas_size"] == cue.canvas_size
    assert record["look_generation_result_id"] is None
    with pytest.raises(ValueError, match="canvas"):
        extract_layout_reference(video, plan, tmp_path / "wrong.png", (64, 64))


def test_official_main_graph_keeps_two_asset_stack():
    source = official_graph()
    derived = fill_main_graph("main.mp4", "look.png", "A bright orb")
    top = {n["id"]: n for n in derived["nodes"]}
    assert top[5001]["widgets_values"][0] == "main.mp4"
    assert top[9009]["widgets_values"][0] == "look.png"
    assert top[5508]["widgets_values"] == ["A bright orb"]
    assert source == official_graph()  # no mutation of the source
    assert OFFICIAL_GRAPH.read_bytes()  # official file retained
    import json
    assert json.loads((GRAPH_DIR / "CueFX_Overlap_Main_Experimental.json").read_text(encoding="utf-8")) == \
           fill_main_graph("cuefx_main_layout.mp4", "cuefx_generated_look.png", "")


def test_image_edit_and_executable_ltx_graphs_fill_only_runtime_inputs():
    edit = fill_image_edit_api("reference.png", "electric orb", 123)
    assert edit["470"]["inputs"]["image"] == "reference.png"
    assert edit["459:500"]["inputs"]["prompt"] == "electric orb"
    assert edit["459:458"]["inputs"]["seed"] == edit["459:500"]["inputs"]["sampling_mode.seed"] == 123
    main = fill_main_api("hardcomp.mp4", "edited.png", "electric orb", "cuefx/main")
    assert main["5001"]["inputs"]["file"] == "hardcomp.mp4"
    assert main["9009"]["inputs"]["image"] == "edited.png"
    plan = plan_overlap(make_cue(entrance=Entrance(.5)))
    reverse = fill_reverse_api(main, plan, "guide.mp4", "anchor.png", "prefix", "cuefx/reverse")
    assert reverse["5516:4829"]["inputs"]["latent_image"] == ["80002", 0]
    assert reverse["9034:9033"]["inputs"]["latent_image"] == ["80003", 0]


def test_full_scene_reference_uses_exact_plate_frame(tmp_path):
    cue = replace(make_cue(entrance=Entrance(.5)), frames=81, effect_range=EffectRange(0, 81),
                  vanish=Vanish(78, 80), release_frame=70, plate_size=(256, 128), canvas_size=(256, 128),
                  canvas_rect=(0, 0, 256, 128), anchor_offset=(0, 0), diameter_px=20,
                  behind=[],
                  track=[TrackPoint(f, 128, 64, "tracked", 20, .9) for f in range(81)],
                  route=[RoutePoint(70, None, None, 20), RoutePoint(75, 180, 64, 16)])
    video = tmp_path / "hardcomp.mp4"
    with VideoWriter(video, cue.plate_size, cue.fps, crf=0) as writer:
        for index in range(cue.frames):
            frame = np.full((cue.plate_size[1], cue.plate_size[0], 3), index, np.uint8)
            writer.write(frame)
    output = tmp_path / "reference.png"
    record = prepare_full_scene_reference_record(cue, video, output)
    decoded = list(read_frames(video))
    assert np.array_equal(cv2.imread(str(output)), decoded[record["look_source_plate_frame"]])
