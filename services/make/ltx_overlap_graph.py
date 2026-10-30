"""Fill the preserved image-edit and LTX graphs for overlap generation."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from services.make.ltx_overlap import OverlapPlan

GRAPH_DIR = Path(__file__).resolve().parents[2] / "comfyui"
OFFICIAL_GRAPH = GRAPH_DIR / "LTX-2.5_ICLoRA_Layout_To_Render_Two_Stage_Distilled (3).json"
OFFICIAL_SHA256 = "1b475b5e7a33b2f8072b92cd09cbaff67db1c66fcdde780f579723b526995a01"
IMAGE_EDIT_API = GRAPH_DIR / "CueFX_Image_Edit_API.json"
EXECUTED_REVERSE_API = GRAPH_DIR / "CueFX_71320_Reverse_Experimental_API.json"


def official_graph() -> dict:
    raw = OFFICIAL_GRAPH.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != OFFICIAL_SHA256:
        raise ValueError(f"official source graph changed: {actual}")
    return json.loads(raw)


def fill_main_graph(layout_name: str, look_name: str, prompt: str,
                    graph: dict | None = None) -> dict:
    """Keep official loaders, model stack, schedules and two sampling stages.

    ``layout_name`` is the automatically prepared main segment beginning at R,
    padded to 8k+1. ``look_name`` is the matching generated look. The returned
    graph has precisely those two external media inputs.
    """
    for value in (layout_name, look_name):
        if Path(value).name != value or value in ("", ".", ".."):
            raise ValueError("asset names must be bare nonempty file names")
    g = copy.deepcopy(official_graph() if graph is None else graph)
    nodes = {n["id"]: n for n in g["nodes"]}
    expected = {5001: "LoadVideo", 9009: "LoadImage", 5508: "PrimitiveStringMultiline"}
    for node_id, kind in expected.items():
        if nodes[node_id]["type"] != kind:
            raise ValueError(f"official node {node_id} changed type")
    nodes[5001]["widgets_values"][0] = layout_name
    nodes[9009]["widgets_values"][0] = look_name
    nodes[5508]["widgets_values"] = [prompt]
    # The optional API key on the official input subgraph stays empty. The
    # local encoder branch is retained; no remote prompt service is selected.
    api = nodes[5014]["widgets_values"]
    if any(api[:3]):
        raise ValueError("official graph unexpectedly selects an API prompt branch")
    loaders = [n for n in g["nodes"] if n["type"] in ("LoadImage", "LoadVideo")]
    if {n["id"] for n in loaders} != {5001, 9009}:
        raise ValueError("official external media loader set changed")
    return g


def _bare(value: str) -> str:
    if Path(value).name != value or value in ("", ".", ".."):
        raise ValueError("asset names must be bare nonempty file names")
    return value


def fill_image_edit_api(reference_name: str, prompt: str, seed: int,
                        graph: dict | None = None) -> dict:
    """Fill the API prompt recovered from the successful edited PNG metadata."""
    g = copy.deepcopy(json.loads(IMAGE_EDIT_API.read_text(encoding="utf-8")) if graph is None else graph)
    expected = {"470": "LoadImage", "501": "PreviewImage", "459:458": "KSampler",
                "459:500": "TextGenerate", "459:484": "ComfySwitchNode"}
    for node_id, kind in expected.items():
        if g.get(node_id, {}).get("class_type") != kind:
            raise ValueError(f"image-edit node {node_id} is not {kind}")
    g["470"]["inputs"]["image"] = _bare(reference_name)
    g["459:458"]["inputs"]["seed"] = seed
    g["459:500"]["inputs"]["prompt"] = prompt
    g["459:500"]["inputs"]["sampling_mode.seed"] = seed
    g["459:484"]["inputs"]["on_false"] = prompt
    return g


def main_api_graph() -> dict:
    """Recover the successful main API graph from the preserved executed reverse graph.

    The reverse artifact was built from this exact main prompt. Removing the
    three CueFX context nodes and restoring the two official guide links yields
    the API form of the unchanged two-stage main graph.
    """
    g = json.loads(EXECUTED_REVERSE_API.read_text(encoding="utf-8"))
    for node_id in ("80001", "80002", "80003"):
        if node_id not in g:
            raise ValueError(f"executed reverse graph has no {node_id}")
        del g[node_id]
    g["5516:4829"]["inputs"]["latent_image"] = ["9002:9042", 2]
    g["9034:9033"]["inputs"]["latent_image"] = ["9034:9043", 2]
    return g


def fill_main_api(video_name: str, look_name: str, prompt: str, output_prefix: str,
                  graph: dict | None = None) -> dict:
    """Fill the executable main graph while retaining its proven model stack."""
    g = copy.deepcopy(main_api_graph() if graph is None else graph)
    expected = {"5001": "LoadVideo", "9009": "LoadImage", "5508": "PrimitiveStringMultiline",
                "4852": "SaveVideo", "5516:4829": "SamplerCustomAdvanced",
                "9034:9033": "SamplerCustomAdvanced"}
    for node_id, kind in expected.items():
        if g.get(node_id, {}).get("class_type") != kind:
            raise ValueError(f"main API node {node_id} is not {kind}")
    g["5001"]["inputs"]["file"] = _bare(video_name)
    g["9009"]["inputs"]["image"] = _bare(look_name)
    g["5508"]["inputs"]["value"] = prompt
    g["4852"]["inputs"]["filename_prefix"] = output_prefix
    return g


def fill_reverse_api(main_prompt: dict, plan: OverlapPlan, guide_name: str,
                     matched_look_name: str, prefix_dir: str, output_prefix: str) -> dict:
    """Add the proven fixed generated/clean context at both sampling stages."""
    g = copy.deepcopy(main_prompt)
    for value in (guide_name, matched_look_name, prefix_dir):
        _bare(value)
    required = {"5001": "LoadVideo", "9009": "LoadImage",
                "5516:4829": "SamplerCustomAdvanced", "9034:9033": "SamplerCustomAdvanced",
                "9002:9042": "LTXAddVideoICLoRAGuide", "9034:9043": "LTXAddVideoICLoRAGuide"}
    for node_id, kind in required.items():
        if g.get(node_id, {}).get("class_type") != kind:
            raise ValueError(f"main graph node {node_id} is not {kind}")
    if g["5516:4829"]["inputs"]["latent_image"] != ["9002:9042", 2] or \
            g["9034:9033"]["inputs"]["latent_image"] != ["9034:9043", 2]:
        raise ValueError("main guide-to-sampler links changed")
    if any(node_id in g for node_id in ("80001", "80002", "80003")):
        raise ValueError("reserved CueFX node ID already used")
    g["5001"]["inputs"]["file"] = guide_name
    g["9009"]["inputs"]["image"] = matched_look_name
    g["5508"]["inputs"]["value"] = (
        str(g["5508"]["inputs"]["value"]).rstrip()
        + " In reversed time the orb contracts and dims toward clean source frames."
    )
    g["4852"]["inputs"]["filename_prefix"] = output_prefix
    g["80001"] = {"class_type": "CueFXLoadFrameBatch",
                    "inputs": {"relative_dir": prefix_dir, "expected_frames": plan.context_count}}
    common = {"vae": ["5004:5601", 0], "generated_prefix": ["80001", 0],
              "reverse_source": ["5548:9006", 0], "first_empty_cell": plan.first_empty_cell}
    g["80002"] = {"class_type": "CueFXFixTemporalContext",
                    "inputs": {"latent": ["9002:9042", 2], **common}}
    g["80003"] = {"class_type": "CueFXFixTemporalContext",
                    "inputs": {"latent": ["9034:9043", 2], **common}}
    g["5516:4829"]["inputs"]["latent_image"] = ["80002", 0]
    g["9034:9033"]["inputs"]["latent_image"] = ["80003", 0]
    return g
