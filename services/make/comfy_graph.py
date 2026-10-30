"""Fill the ComfyUI graphs with one shot's values (plain JSON; no ComfyUI imports): the interaction graph
(`fill_graph`) and the quality graph of the manual render (`fill_quality_graph`).

ComfyUI's Load nodes read file names from its own `input` folder, so every value written here is a bare
file name, and the segment script's media are copied into one folder as `<clip>_<name>`.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

SEGMENT_FILES = ("comp_segment.mp4", "element_segment.mp4", "comp_frame_F.png", "edit_mask_F.png")
QWEN_RAW = "qwen_raw.png"  # saved by the owner as <clip>_qwen_raw.png

# node title prefix -> file it loads, as `<clip>_<file>`; None = the look image, used as given
LOADERS = {
    "Load hard comp": "comp_segment.mp4",
    "Load element video": "element_segment.mp4",
    "Load comp frame F": "comp_frame_F.png",
    "Load edit mask F": "edit_mask_F.png",
    "Load raw Qwen output": QWEN_RAW,
    "Load reference image": None,
}
PROMPT_TITLE = "Prompt · describe"


def comfy_name(clip_id: str, name: str) -> str:
    """The file name ComfyUI loads for `name` of this clip."""
    return f"{clip_id}_{name}"


def _one(nodes: list[dict], match) -> dict:
    found = [n for n in nodes if match(n)]
    if len(found) != 1:
        raise ValueError(f"expected exactly one matching node, found {len(found)}")
    return found[0]


def _titled(prefix: str):
    return lambda n: (n.get("title") or "").startswith(prefix)


def _set_file(node: dict, name: str) -> None:
    node["widgets_values"][0] = name
    named = node.get("widgets_values_named")
    if named:  # keep the owner's named copy in step with the positional values
        named[next(iter(named))] = name


def fill_graph(generic: dict, clip_id: str, look: str, prompt: str, pin_frame: int,
               max_padded_frames: int = 193) -> dict:
    """A copy of the generic interaction graph with this shot's file names, prompt and entrance override.

    `look` must be a bare file name; `pin_frame` is `pin_frame_in_segment` from segment.json.
    """
    if Path(look).name != look:
        raise ValueError(f"look must be a file name without a folder: {look!r}")
    if pin_frame < 0:
        raise ValueError(f"pin_frame must be 0 or more, got {pin_frame}")
    g = copy.deepcopy(generic)
    top = g["nodes"]
    for prefix, name in LOADERS.items():
        _set_file(_one(top, _titled(prefix)), look if name is None else comfy_name(clip_id, name))
    _one(top, lambda n: n["type"] == "CueFXEntrance")["widgets_values"][0] = int(pin_frame)
    _one(top, lambda n: n["type"] == "CueFXOfficialPrepare")["widgets_values"][2] = int(max_padded_frames)
    _one(top, _titled(PROMPT_TITLE))["widgets_values"] = [prompt]
    return g


LOAD_TYPES = ("LoadImage", "LoadVideo")
# the quality graph (CueFX_Phase1_Quality.json, manual render): node title prefix -> package input it loads
QUALITY_LOADERS = {"Load layout video": "layout", "Load reference image": "look"}
SAVE_TYPE = "CueFXGenericSave"  # widgets: filename_prefix, save_raw_frames; writes <prefix>_<UTC>_<hex>/effect.mp4
SAVE_FILE = "effect.mp4"
PREPARE_TYPE = "CueFXOfficialPrepare"  # widgets: resolution_mode, look_fit, max_padded_frames
GRAPH_MIN_FRAMES = 121  # the graph's own padded-frame limit; longer windows raise it (`graph_frames`)


def graph_frames(window_length: int) -> int:
    """The padded-frame limit of the quality graph for an effect window of `window_length` frames: 121 (what the
    graph ships with) or the window length if that is more. The window is 8k+1, so it is also its padded length."""
    return max(GRAPH_MIN_FRAMES, int(window_length))


def load_nodes(graph: dict) -> list[dict]:
    """Every file-loading node of a UI-format graph, top level and inside subgraphs."""
    nodes = list(graph["nodes"])
    for sub in graph.get("definitions", {}).get("subgraphs", []):
        nodes += sub.get("nodes", [])
    return [n for n in nodes if n["type"] in LOAD_TYPES]


def fill_quality_graph(generic: dict, files: dict[str, str], prompt: str, output_prefix: str,
                       frames: int = GRAPH_MIN_FRAMES) -> dict:
    """A copy of the quality graph with the package's file names, the prompt and the save prefix filled in, and the
    padded-frame limit of the Prepare node (the one place that sets the length: the latent's `length` follows the
    node's `padded_frames` output) raised to `graph_frames(frames)`; nothing else (models, sizes, steps, seeds)
    changes, and a window of 121 frames or less leaves the graph exactly as it is.

    `files` maps each QUALITY_LOADERS value ("layout", "look") to a bare file name in ComfyUI's input folder.
    `output_prefix` is the save node's relative prefix (e.g. "cuefx/job-1a2b"). Raises ValueError if the graph
    loads a file the package does not make, or a node is missing: the package would not run as it is.
    """
    if set(files) != set(QUALITY_LOADERS.values()) or any(Path(v).name != v for v in files.values()):
        raise ValueError(f"files must be bare names for {sorted(QUALITY_LOADERS.values())}: {files!r}")
    g = copy.deepcopy(generic)
    loads = load_nodes(g)
    known = [n for n in loads if any((n.get("title") or "").startswith(p) for p in QUALITY_LOADERS)]
    if len(known) != len(loads):
        extra = [n.get("title") or n["type"] for n in loads if n not in known]
        raise ValueError(f"the graph loads files the package does not make: {extra}")
    top = g["nodes"]
    for prefix, role in QUALITY_LOADERS.items():
        _set_file(_one(top, _titled(prefix)), files[role])
    _one(top, _titled(PROMPT_TITLE))["widgets_values"] = [prompt]
    _one(top, lambda n: n["type"] == SAVE_TYPE)["widgets_values"][0] = output_prefix
    _one(top, lambda n: n["type"] == PREPARE_TYPE)["widgets_values"][2] = graph_frames(frames)
    return g


def copy_comfy_inputs(segment_dir: Path, out_dir: Path, clip_id: str) -> list[Path]:
    """Copy the segment script's media into `out_dir` as `<clip>_<name>`, the names the shot graph loads."""
    missing = [n for n in SEGMENT_FILES if not (segment_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"missing in {segment_dir}: {', '.join(missing)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return [Path(shutil.copy2(segment_dir / n, out_dir / comfy_name(clip_id, n))) for n in SEGMENT_FILES]
