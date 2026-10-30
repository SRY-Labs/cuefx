"""Cache key of the Make step: a hash of exactly the cue fields that make Make run again (BRIEF section 4)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from services.cue.cue import Cue, behaviour_of, cue_to_dict, ending_of, entrance_frames, orbit_of, shape_of, vanish_of
from services.make.layout import SHAPE_SPECS
from services.cue.window import effect_window

# Bump when Make draws differently from the same inputs, so old make folders are not reused.
# 2: the entrance (the stand-in grows over entrance.seconds before appear_frame; the window covers it)
# 3: one clean ball per layout frame, no motion-blur copies (task 05a); the stand-in places one orb per frame too
MAKE_VERSION = 3
ENDING_LOOK = 3  # which drawing of burst and shatter a key stands for: 1 = task 36, 2 = task 38, 3 = task 38 brighter burst (only a non-fade ending has it in its key)

# the first row of "What reruns what" in BRIEF section 4, plus canvas_size (the canvas LTX works on)
# correction_falloff is added to that row: it changes the corrected track, so the path
MAKE_FIELDS = ("look", "track", "corrections", "correction_falloff", "anchor_offset", "release_frame", "route", "diameter_px",
               "effect_range", "canvas_rect", "seed", "canvas_size")


def file_hash(path: Path) -> str:
    """sha256 of the bytes of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_key(cue: Cue, look_image: Path) -> str:
    """12 hex characters (sha256) over MAKE_VERSION, the Make fields of the cue and the bytes of the look image file.

    A "hold" (or "charge_up", v2.6) cue adds its behaviour to the key (a "hold_throw" cue does not, so its key is what it
    always was); a "charge_up" cue also adds appear_frame, an "orbit" cue appear_frame and its circle (only then).
    Fields that only Merge reads (vanish, behind, grade; appear_frame while the entrance is off) are not in it. A burst
    or shatter `ending` (v2.6) is drawn by Make: the ending and the vanish it runs over are then in the key.
    With the entrance on, Make draws the build-up over appear_frame - N_in .. appear_frame - 1, so appear_frame and
    N_in (from entrance.seconds, or its default, and fps) are in it.
    """
    d = cue_to_dict(cue)
    payload = {k: d.get(k) for k in MAKE_FIELDS}
    if payload["route"] is not None:  # v2.2 fields: a note never changes Make, and linear easing draws as v2.1 did,
        payload["route"] = [{k: v for k, v in p.items()  # so v2.1 cues keep their make keys
                             if k != "note" and not (k == "easing_in" and v == "linear")} for p in payload["route"]]
    if entrance_frames(cue) > 0:
        payload["appear_frame"] = cue.appear_frame
        payload["entrance_frames"] = entrance_frames(cue)
    if effect_window(cue).length > 121:  # task 33: windows of 121 or less keep their key; a longer one used to be trimmed
        payload["window_frames"] = effect_window(cue).length
    if behaviour_of(cue) != "hold_throw":  # v2.3: only the key of a "hold" cue changes; every older key stays
        payload["behaviour"] = behaviour_of(cue)
    if behaviour_of(cue) == "orbit":  # v2.6: the circle (its missing fields filled, so spelling a default changes nothing) and
        o = orbit_of(cue)             # where it starts (appear_frame) are what Make draws
        payload["appear_frame"] = cue.appear_frame
        payload["orbit"] = {"radius_sizes": o.radius_sizes, "speed_deg_per_s": o.speed_deg_per_s, "tilt_deg": o.tilt_deg}
    if behaviour_of(cue) == "charge_up":  # v2.6: the growth and the pulse start at appear_frame, even with the entrance off
        payload["appear_frame"] = cue.appear_frame
    kind = shape_of(cue)  # v2.5: a sphere at its default spin adds nothing, so every older key stays
    spin = SHAPE_SPECS[kind].spin if cue.spin_deg_per_s is None else cue.spin_deg_per_s
    if kind != "sphere":
        payload["shape"] = kind
    if kind != "sphere" or spin != SHAPE_SPECS["sphere"].spin:
        payload["spin_deg_per_s"] = spin
    if ending_of(cue) != "fade":  # v2.6: a burst or shatter is drawn by Make from the vanish on, so it and the vanish are in
        payload["ending"] = ending_of(cue)  # the key; a fade is Merge's, and every older key stays
        payload["vanish"] = {"start": vanish_of(cue).start, "end": vanish_of(cue).end}
        payload["ending_look"] = ENDING_LOOK  # task 38 drew both endings again: a render of the older look is not reused
    payload["make_version"] = MAKE_VERSION
    payload["look_image"] = file_hash(look_image)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
