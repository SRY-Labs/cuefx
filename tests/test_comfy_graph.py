"""services.make.comfy_graph: filling the generic interaction graph, on the committed graph and synthetic files."""
import json
from pathlib import Path

import pytest

from services.make.comfy_graph import (PREPARE_TYPE, SAVE_TYPE, SEGMENT_FILES, comfy_name, copy_comfy_inputs, fill_graph,
                               fill_quality_graph, graph_frames, load_nodes)

GENERIC = Path(__file__).resolve().parents[1] / "comfyui" / "CueFX_Interaction.json"


@pytest.fixture(scope="module")
def generic() -> dict:
    return json.loads(GENERIC.read_text(encoding="utf-8"))


def test_fill_sets_only_shot_values(generic):
    g = fill_graph(generic, "c9", "look.png", "a prompt", 26, 185)
    top = {n["id"]: n for n in g["nodes"]}
    by_title = {n.get("title"): n["widgets_values"][0] for n in g["nodes"] if n["type"] in ("LoadImage", "LoadVideo")}
    assert sorted(by_title.values()) == sorted(["c9_comp_segment.mp4", "c9_element_segment.mp4", "c9_comp_frame_F.png",
                                                "c9_edit_mask_F.png", "c9_qwen_raw.png", "look.png"])
    assert next(n for n in g["nodes"] if n["type"] == "CueFXEntrance")["widgets_values"][0] == 26
    assert next(n for n in g["nodes"] if n["type"] == "CueFXOfficialPrepare")["widgets_values"][2] == 185
    assert top[5508]["widgets_values"] == ["a prompt"] and top[5509]["widgets_values"] == [""]
    assert top[5001]["widgets_values_named"]["file"] == "c9_comp_segment.mp4"
    assert g["links"] == generic["links"] and g["definitions"] == generic["definitions"]
    assert next(n for n in generic["nodes"] if n["type"] == "CueFXEntrance")["widgets_values"][0] == -1  # not mutated


@pytest.mark.parametrize("look,pin", [("sub/look.png", 3), ("look.png", -1)])
def test_fill_rejects_paths_and_automatic_pin(generic, look, pin):
    with pytest.raises(ValueError):
        fill_graph(generic, "c9", look, "p", pin)


QUALITY = GENERIC.parent / "CueFX_Phase1_Quality.json"


@pytest.fixture(scope="module")
def quality() -> dict:
    return json.loads(QUALITY.read_text(encoding="utf-8"))


def test_quality_graph_loads_only_the_layout_and_the_look(quality):
    loads = {(n["type"], n.get("title")) for n in load_nodes(quality)}
    assert loads == {("LoadVideo", "Load layout video (Clay Video)"), ("LoadImage", "Load reference image (Look Still)")}
    assert [n["type"] for n in quality["nodes"] if n["type"] == SAVE_TYPE] == [SAVE_TYPE]


def test_fill_quality_graph_sets_names_prompt_and_prefix_only(quality):
    g = fill_quality_graph(quality, {"layout": "j_layout.mp4", "look": "j_look.png"}, "an orb", "cuefx/j")
    top = {n["id"]: n for n in g["nodes"]}
    assert sorted(n["widgets_values"][0] for n in load_nodes(g)) == ["j_layout.mp4", "j_look.png"]
    assert top[5508]["widgets_values"] == ["an orb"] and top[5509]["widgets_values"] == [""]
    assert next(n for n in g["nodes"] if n["type"] == SAVE_TYPE)["widgets_values"] == ["cuefx/j", True]
    assert g["links"] == quality["links"] and g["definitions"] == quality["definitions"]
    changed = {i for i, n in top.items() if n != next(m for m in quality["nodes"] if m["id"] == i)}
    assert changed == {5001, 9009, 5508, 9047}  # the two loaders, the prompt, the save prefix: no LTX setting
    assert top[5001]["widgets_values"][1:] == ["image"]


FILES = {"layout": "j_layout.mp4", "look": "j_look.png"}


def prepare_limit(graph: dict) -> int:
    return next(n for n in graph["nodes"] if n["type"] == PREPARE_TYPE)["widgets_values"][2]


def test_the_graph_length_is_121_for_windows_up_to_121_and_the_window_length_above(quality):
    assert [graph_frames(n) for n in (17, 89, 121, 129, 193)] == [121, 121, 121, 129, 193]
    assert prepare_limit(quality) == 121  # the committed graph
    base = json.dumps(fill_quality_graph(quality, FILES, "an orb", "cuefx/j"))
    for n in (17, 89, 121):  # task 33: today's graph, byte for byte (json.dumps keeps node and key order)
        assert json.dumps(fill_quality_graph(quality, FILES, "an orb", "cuefx/j", n)) == base, n
    long = fill_quality_graph(quality, FILES, "an orb", "cuefx/j", 193)
    assert prepare_limit(long) == 193
    top, was = {n["id"]: n for n in long["nodes"]}, {n["id"]: n for n in quality["nodes"]}
    assert {i for i in top if top[i] != was[i]} == {5001, 9009, 5508, 9047, 9045}  # the four fills and the Prepare limit
    assert long["links"] == quality["links"] and long["definitions"] == quality["definitions"]
    assert quality["nodes"] == json.loads(QUALITY.read_text(encoding="utf-8"))["nodes"]  # the input is not changed


def test_the_prepare_node_is_the_only_live_control_of_the_length(quality):
    """The latent length (node 9002 -> the Preprocess subgraph's EmptyLTXVLatentVideo) is linked to the Prepare node's
    padded_frames output, so the 121 typed in those two widgets is overridden by the link."""
    prepare = next(n for n in quality["nodes"] if n["type"] == PREPARE_TYPE)
    link = next(o for o in prepare["outputs"] if o["name"] == "padded_frames")["links"]
    assert len(link) == 1
    l = next(l for l in quality["links"] if l[0] == link[0])
    holder = next(n for n in quality["nodes"] if n["id"] == l[3])
    assert holder["inputs"][l[4]]["name"] == "length" and holder["inputs"][l[4]]["link"] == link[0]
    inner = next(s for s in quality["definitions"]["subgraphs"] if s["id"] == holder["type"])
    sub = next(n for n in inner["nodes"] if n["type"] == "EmptyLTXVLatentVideo")
    assert next(i for i in sub["inputs"] if i["name"] == "length")["link"] is not None


def test_fill_quality_graph_refuses_a_graph_the_package_does_not_cover(quality):
    extra = json.loads(json.dumps(quality))
    extra["definitions"]["subgraphs"][0]["nodes"].append({"id": 1, "type": "LoadImage", "title": "Load mask",
                                                          "widgets_values": ["mask.png", "image"]})
    with pytest.raises(ValueError, match="Load mask"):
        fill_quality_graph(extra, {"layout": "a.mp4", "look": "b.png"}, "p", "cuefx/j")
    with pytest.raises(ValueError):
        fill_quality_graph(quality, {"layout": "sub/a.mp4", "look": "b.png"}, "p", "cuefx/j")


def test_copy_comfy_inputs(tmp_path):
    seg = tmp_path / "seg"
    seg.mkdir()
    for i, n in enumerate(SEGMENT_FILES):
        (seg / n).write_bytes(bytes([i]) * 4)
    out = copy_comfy_inputs(seg, tmp_path / "comfy_input", "c9")
    assert [p.name for p in out] == [comfy_name("c9", n) for n in SEGMENT_FILES]
    assert [p.read_bytes() for p in out] == [(seg / n).read_bytes() for n in SEGMENT_FILES]
    (seg / "edit_mask_F.png").unlink()
    with pytest.raises(FileNotFoundError, match="edit_mask_F.png"):
        copy_comfy_inputs(seg, tmp_path / "other", "c9")
