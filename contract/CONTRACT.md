# CueFX app contract (UI ↔ backend), v3.0

This folder is the one contract both sides test against. **The render is the truth.** Where the UI mock's maths
differs from the backend, this file describes the backend, and the UI adapts (section 11 lists every place).

- Schemas: `schemas/*.schema.json` (JSON Schema 2020-12). Codes and messages: `schemas/error_codes.json`.
- Golden files: `golden/` (clip 71319). `golden/sample_*.json` are **illustrative**: generate is not built yet.
- Backend: `cuefx/api.py` (pure Python functions) and `scripts/cuefx_api.py` (command line). No server yet (task 17).

## 1. Conventions

| | |
|---|---|
| Ranges | Stored **half-open `[start, end)`**: start included, end excluded. The UI shows them inclusive (`end − 1`), so `{start: 80, end: 91}` reads "80–90". Applies to `effect_range`, `vanish`, `behind`, warning `frames`, `uncertain_ranges`, result `checks[].frames`. |
| Frames | Plate frames, starting at **0**; fps 24 after conform. |
| Coordinates | Conformed **plate pixels** (1920×1080), origin top-left, **y down**. Values outside the frame are allowed (x = −120 is a throw leaving the left edge). |
| Hand | `hand` is the **person's** hand (`right` \| `left`), never MediaPipe's mirrored label. |
| Field names | snake_case. Where the backend already had a field, the contract uses the backend name; fields the UI introduced keep the UI name (section 3). |
| Unknown fields | Top-level fields the schema does not know are kept in the file and reported (`unknown_field`, info). Unknown keys inside a block are schema errors. |

### Versioning
- `version` of the cue = the contract version that last changed the cue. Contract: **3.0**; cue: **2.6** (2.5 adds
  `behaviour`, 2.6 adds `body_point`, 2.7 adds `shape` and `spin_deg_per_s`, 2.8 adds `ending`; section 13). New cues (`cue/new`) are written as 2.3, or as 2.4 when the request
  or the cue sets `body_point`, or as 2.5 when the cue sets `shape` or `spin_deg_per_s`. **A v2.2 file is read as it is** and written back unchanged (it keeps `"2.2"`; a `hold`
  cue is written as `"2.3"`, a cue that sets `body_point` as `"2.4"`, one that sets `shape` or `spin_deg_per_s` as `"2.5"`, one that sets a non-fade `ending` as `"2.6"`), so golden files do not change.
- **v2.1 files load.** Their ranges were already `[start, end)`, so loading converts nothing but the version; the
  v2.2 fields get their defaults (`easing_in: "linear"`, no `note`, no `spill`). Linear easing routes byte for byte as
  v2.1 did, and v2.1 cues keep their Make cache keys.
- Adding an optional field or a new code: minor bump (2.3), both sides must accept files without it.
  Renaming, removing or changing the meaning of a field or code: needs both teams to agree, and a loader that still
  reads the previous version. A code is never reused with a different meaning.
- Golden outputs change only through `scripts/make_contract_golden.py`, and the diff is reviewed.

## 2. The cue, v2.6

What v2.6 adds to v2.5 (task 36, contract 2.8):

| Field | Type, default | Meaning |
|---|---|---|
| `behaviour` (adds) | `"charge_up"` | **charge_up (v2.6):** like `hold_throw` (it needs `release_frame` and `route`, and flies the same way after the release), but between `appear_frame` and `release_frame` the effect grows from 0.6 × to 1.0 × its `diameter_px` (smoothstep) and pulses at 3.5 Hz by ±8 % of both its size and its brightness (the defaults are in `GET /api/defaults` → `charge_up`). Setting it makes the cue version **2.6**. `behaviour_invalid` now lists `hold_throw`, `hold`, `charge_up`, `orbit`. |
| `behaviour` (adds) | `"orbit"` | **orbit (v2.6):** no release and no route, like `hold` (`release_frame: null`, `route: []`; `hold_path_unused` warns if either is set). The effect travels on a circle round the held position (body point + `anchor_offset`): radius `orbit.radius_sizes` × the body point's size measure (the track's median `palm_w`), at angle `orbit.speed_deg_per_s` × t (t in seconds from `appear_frame`, angle 0 = to the right of the point; the sign is the direction), the vertical axis squashed by cos(`orbit.tilt_deg`). The lower side of the circle is the front: the effect is up to 15 % bigger there and at full brightness; the upper side is the back: 15 % smaller and at 50 % brightness (smooth with the angle), so the back half reads as behind. **There is no real occlusion by the body (known limit).** Setting it, or an `orbit` block, makes the cue version **2.6**. |
| `orbit` | `{radius_sizes, speed_deg_per_s, tilt_deg}`; each missing field = its default (`GET /api/defaults` → `orbit`: 1.5, 180, 60; the tilt of a **head** is 80, `orbit_tilt_deg_by_part`) | The circle of an `orbit` behaviour; ignored by the other behaviours. `radius_sizes` above 0 and up to 10; `speed_deg_per_s` within ±3600; `tilt_deg` from 0 (a full circle) to 90 (a line). Anything else: `orbit_invalid` (error). It is in the Make key only when the behaviour is `orbit` (with `appear_frame`; the defaults filled in, so writing a default out changes nothing). |
| `ending` | `"fade"` \| `"burst"` \| `"shatter"`; missing = `"fade"` | How the effect ends. It starts at `vanish.start` (the last route point of a throw, the end of the hold for a hold) and runs to `vanish.end`. **fade:** Merge fades the effect out linearly over `vanish`, exactly as before. **burst:** the effect grows to 1.8 × its size over the first 40 % of the vanish while its light rises to 1.6 × the orb's (near white), holds that bright until 60 %, then fades; at least 0.3 s. **shatter:** 12 to 20 fragments (their number and directions come from `seed`) in the same grey shading fly outward, spin and fall under slight gravity; they stay **fully bright for the first 50 % of the vanish**, then fade out; at least 0.5 s; every fragment is at least **0.25** × the effect's diameter (v2.9; was 0.15), and shards are simple polygons for every shape. Their outward speed is fixed, so a longer shatter drifts further; they are kept inside the picture (the conformed picture, without the black padding, where it meets the canvas). For a burst or shatter **Merge applies no fade-out of its own** (the effect fades itself in the layout and the stand-in), and nothing is drawn after `vanish.end`. If the vanish is shorter than the ending needs (burst 8 frames, shatter 12 at 24 fps; `GET /api/defaults` → `ending_seconds`: these are the **minimums**), `cue/normalize` makes `vanish.end` later and gives the warning `vanish_extended` (also in `validate` until it has). The window grows with it; the 8 s cap still applies, and where it (or the end of the clip) cuts the ending, `vanish.end` stops at the window end and `window_trimmed` is given. **The ending lasts for the whole vanish** (`vanish.start` → `vanish.end`; there is no separate length field): a longer vanish stretches the animation, it never stops early and holds empty frames. When the artist picks an ending the UI sets the vanish to `GET /api/defaults` → `ending_default_seconds` (burst 0.6 s, shatter 1.0 s). Any other value: `ending_invalid` (error). Setting a non-fade `ending` makes the cue version **2.6**; `"fade"` changes nothing. |

A `charge_up` cue adds its behaviour and `appear_frame` to the Make key (the growth and pulse start there); every other key is unchanged.

`ending` is in the Make key only when it is not `fade` (with the vanish it runs over and, since v2.9, a look number so a render of the older burst or shatter is not reused): every existing Make key is unchanged. The manual export adds one line to the prompt when `ending` is not `fade` and the prompt is not empty ("At the end the effect bursts in a bright flash." / "... shatters into glowing shards.").

## 2b. The cue, v2.5

What v2.5 adds to v2.4 (task 31):

| Field | Type, default | Meaning |
|---|---|---|
| `shape` | `"sphere"` \| `"ring"` \| `"crystal"` \| `"cube"`; missing = `"sphere"` | What the layout and the stand-in draw, and what LTX is shown through the layout. `diameter_px` (the top-level one and each route point's) is the shape's **outer size on screen**: the diameter of the smallest circle that holds it. Ring: a torus tilted 60° towards the camera, tube 0.18 × the diameter. Crystal: an elongated octahedron, 1.0 × high and 0.6 × wide. Cube: a cube turning about an axis tilted 20° from vertical (its side follows its pose so the outer size stays put). Any other value: `shape_invalid` (error). The look image is placed as before, at the first whole-shape frame and the shape's size; one look image per shape is the artist's job. |
| `spin_deg_per_s` | number; missing = the shape's default | Spin in degrees per second (negative turns the other way, 0 = no spin). Defaults: sphere 288 (as before), ring 90, crystal 60, cube 60. A sphere with no `spin_deg_per_s` draws exactly as it always did. |

`shape` and `spin_deg_per_s` are in the Make key only when the shape is not `sphere` or the spin differs from the sphere's default, so every existing Make key is unchanged. Merge treats every shape alike (no registration is built yet).

## 2a. The cue, v2.4

Full shape: `schemas/cue.schema.json`. The schema checks **shape and types**; the value rules (8k+1 windows, frames
inside the clip, enums such as `hand` or `easing_in`, opacity 0..1, …) are checked by `validate`, which returns each
with its own code and a plain message. Canonical example: `golden/throw_71319.cue.json`.

What v2.4 adds to v2.3 (task 32):

| Field | Type, default | Meaning |
|---|---|---|
| `body_point` | `{part, side}`; missing = `{part: "palm", side: <hand>}` | What the effect follows. `part`: `palm` \| `fist` \| `foot` \| `knee` \| `shoulder` \| `head`. `side`: the **person's** `left` \| `right`; `null` only for `head` (a side given for the head is ignored). For `palm`, `side` must equal `hand`. Anything else: `body_point_invalid` (error). `hand_seed` is the artist's click, or for a seedless non-palm `cue/new` request, the real tracked sample returned as `analyze_result.body_point_seed`. Everything after tracking (cue, start timings, Make, stand-in, Merge, manual export) is the same for every part: a body-point track has the same format as the palm track. |

`palm` is tracked by the Hand Landmarker (unchanged); the other parts by the MediaPipe Pose Landmarker (heavy, float16):
**in the track, `palm_w` is the body point's size measure** (palm: palm width; fist: wrist to index; foot: heel to foot
index; knee: 0.4 × hip to knee; shoulder: 0.35 × shoulder width; head: ear to ear), `conf` is the mean landmark
visibility, and `analyze_result.hand_size_px` is its median. A frame is missing (interpolated, `lost` after more than 6)
when no pose is found or `conf` is below 0.5. `cue/new` gives each part a default `diameter_px` and `anchor_offset`
(palm 1.15 × size, [0, −0.65 d]; fist 1.6 × size, [0, 0]; foot 1.0 × size, [0, −0.5 d] (v2.8: was [0, 0]); knee 1.0 × size, [0, 0]; shoulder 1.0 × size,
[0, −0.5 d]; head 1.0 × size, [0, −1.1 × size]); starting values the artist edits.

**Starting timings of a fist, foot or knee (v2.8, the strike rule).** These parts are steady only while the person
stands, so `cue/new` first looks for a strike: the point's speed in size measures per frame (its `palm_w`, the track's
median), averaged over 3 frames. FIST: if the fastest frame is above 1.0 it is the `release_frame`; FOOT and KNEE (slower swing, bigger size measure, and a swing out and back): the peak of the first run of at least 3 frames above 0.4. `appear_frame` is 1 s
before it (not below 0), the cue is `hold_throw`, and the route has two points: the release (derived) and, 0.5 s later,
the point 6 size measures along the mean velocity of the 3 frames before the release, kept inside the conformed picture (the plate without the black padding of the conform) by 0.9 × the diameter (1.8 × the radius, so a burst or shatter there is fully in frame). The vanish
is as for any throw (release + 0.25 s, 0.25 s long). A track with no frame above 1.0 is read as before. Palm, shoulder
and head are unchanged.

What v2.3 adds to v2.2 (task 33):

| Field | Type, default | Meaning |
|---|---|---|
| `behaviour` | `hold_throw`, `hold` or (v2.6) `charge_up`; missing = `hold_throw` | `hold_throw`: held on the hand, released at `release_frame`, flies along `route` (everything up to v2.2). `hold`: the effect follows the tracked hand + `anchor_offset` (+ corrections) in **every frame of the window**, at `diameter_px`; no release, no flight path. A `hold` cue has `release_frame: null` and `route: []`; entrance, `appear_frame`, `vanish` (the fade-out), `behind`, `grade` and `look` work as before. Any other value: `behaviour_invalid` (error). |
| `release_frame` | integer, **null for `hold`** | Was always an integer. |

Also in v2.3: the effect window may be up to **193 frames** (8 s at 24 fps; was 121), see 7.6.

What v2.2 adds to v2.1 (BRIEF section 4):

| Field | Type, default | Meaning |
|---|---|---|
| `route[].easing_in` | `linear` \| `ease_in` \| `ease_out` \| `ease_in_out`, default `linear` | Shapes the time of the segment that **ends** at this point (section 7.3). |
| `route[].note` | string or null | Artist's label. Stored, never rendered. |
| `grade.spill` | number ≥ 0, optional | UI "spill on occluders". Accepted, not used yet. |
| `prompt` | string or null | Accepted for generate. |
| `reference_image` | string (file name) or null | Accepted for generate. |
| `advanced.track_smoothing` | 0..1 | Accepted, not used yet. |
| `advanced.export_layers` | `{effect_on_black, matte, occluder_matte}` booleans | Accepted, not used yet. |

The cue file's `track` keeps its backend form (`{f, x, y, source, palm_w, conf, lost}`); the API passes the track
separately in its own form (section 8). A cue sent to the API may have `track: []`.

## 3. Field mapping: UI mock → contract

| UI mock (step2_flow.html) | Contract | Note |
|---|---|---|
| `clip.width`, `clip.height` | `plate_size: [w, h]` | backend name |
| `clip.fps` | `fps` | backend name |
| `clip.frames` | `frames` | backend name |
| (clip file name) | `clip` (string) | backend: `clip` is the file name, not an object |
| `hand` | `hand` | same |
| `look` | `look` | same name; any look id string |
| `prompt` | `prompt` | UI name kept |
| `reference_image` | `reference_image` | UI name kept |
| `entrance.seconds` | `entrance.seconds` | same |
| `appear_frame`, `release_frame` | same | same |
| `vanish.start`, `vanish.end` | same | same, `[start, end)` |
| `orb.diameter_px` | `diameter_px` | backend name (top level) |
| `orb.offset_from_palm_px: [x, y]` | `anchor_offset: [x, y]` | backend name |
| `orb.tint` | `grade.tint` | backend name |
| `orb.glow` | `grade.glow` | backend name |
| `orb.spill` | `grade.spill` | UI name kept, moved into `grade` (the UI's `orb` block has no backend counterpart) |
| `route[].frame` | `route[].f` | backend name |
| `route[].x`, `route[].y` | same | `null` only on `route[0]` |
| `route[].diameter` | `route[].diameter_px` | backend name |
| `route[].easing_in` | `route[].easing_in` | UI name kept (new in v2.2) |
| `route[].note` | `route[].note` | UI name kept (new in v2.2) |
| `passes_in_front[]` | `behind[]` | backend name |
| `passes_in_front[].start/end` | `behind[].start/end` | same, `[start, end)` |
| `passes_in_front[].who: "left hand"` | `behind[].occluder: "left_hand"` | backend name and values: `left_hand` \| `right_hand` \| `person` \| `{"point": [f, x, y]}` |
| `advanced.corrections[]{frame, x, y}` | `corrections[]{f, x, y}` (top level) | backend name; also `correction_falloff` (frames, default 12) |
| `advanced.track_smoothing` | `advanced.track_smoothing` | UI name kept |
| `advanced.export_layers` | `advanced.export_layers` | UI name kept |
| — | `version`, `transform`, `effect_range`, `canvas_size`, `canvas_rect`, `hand_seed`, `track`, `correction_falloff`, `seed`, `grade.brightness`, `grade.opacity`, `grade.person_light`, `grade.fade_in_frames`, `interaction` | backend fields the UI does not edit yet: keep them and send them back unchanged |
| track entry `{frame, x, y, confidence, uncertain}` | same (API track) | maps to the cue file's `{f, x, y, conf, lost}` |
| `handSize` | `hand_size_px` | |
| `summary.uncertain` | `summary.uncertain_ranges` | |
| preview `{frame, x, y, diameter, visible, phase}` | same + `visibility` | |
| job `stageName`, `elapsedMin`, `etaMin` | `stage_name`, `elapsed_s`, `eta_s` | seconds, snake_case |

## 4. Status of the UI-added fields

| Field | Status | Maps to / what happens |
|---|---|---|
| `look` | **accepted for generate** | Stored (`Cue.look`, part of the Make cache key); generate (task 18) picks the look image and prompt from it. |
| `prompt` | **accepted for generate** | Stored (kept in the cue file); passed to LTX by generate (task 18). |
| `reference_image` | **accepted for generate** | Stored; generate uses it as the look image (task 18). |
| `route[].easing_in` | **implemented** | `cuefx.route.EASE`, applied in `orb_path` to every segment. On `route[0]` it has no segment and is reported (`setting_not_used`). |
| `route[].note` | stored, label only | Never rendered; never reported. |
| `advanced.corrections` | **implemented** | It is the top-level `corrections` (keyframes with falloff, `cuefx.track.apply_corrections`). |
| `orb.spill` → `grade.spill` | **accepted, not used yet** | `setting_not_used`. Merge uses its fixed `MergeConfig.spill = 0.15` (share of the glow). Wiring it is one line in a later task; the UI's 0.4 would then mean 40 % of the glow. |
| `advanced.track_smoothing` | **accepted, not used yet** | `setting_not_used`. The track is already smoothed once, by the tracker (zero-lag One Euro); there is no second smoothing step. |
| `advanced.export_layers` | **accepted, not used yet** | `setting_not_used`. Every render writes all layers (section 10). |
| `grade.person_light` (backend field) | **accepted, not used yet** | `setting_not_used` when > 0. Light on the person is not built (docs/OPEN_ISSUES.md). |

No field is ignored silently: anything stored but not used gives an `info` warning, anything unknown gives `unknown_field`.

## 5. The calls

All inputs and outputs are JSON. Python: `cuefx.api`; command line: `scripts/cuefx_api.py`. A failed call raises
`CueFXError(code, message)`; on the command line it prints `{"code", "message"}` to stdout and exits **2**.

### analyze(clip_path, seed) → analyze_result
- `seed`: `{frame, x, y}` in plate pixels, the artist's click on the palm.
- Returns `schemas/analyze_result.schema.json`: `{track[], summary{tracked, total, uncertain_ranges[[s, e)]}, hand_size_px, warnings[]}`.
  - `summary.tracked`: frames where the hand was **detected**; `uncertain_ranges`: the lost gaps.
  - `hand_size_px`: median palm width (index to little-finger knuckle) over the detected frames.
  - `warnings`: `seed_far` when the click missed every hand (the nearest hand is used: degrade, don't fail).
- Errors: `clip_unreadable`, `clip_too_long` (more than 1440 frames = 60 s), `no_hand` (no hand at all in the seed
  frame), `track_lost` (more than half of the frames inside lost gaps), `input_invalid` (bad seed, seed frame outside the clip).
- CLI: `cuefx_api.py analyze CLIP --seed FRAME X Y [-o FILE]`. Synchronous, about 15 s for 193 frames on a laptop CPU;
  progress steps come with the server (task 17).

### preview_path(cue, track) → preview[]
- One entry per frame `0 … frames−1`: `{frame, x, y, diameter, visibility, visible, phase}` (`schemas/preview.schema.json`).
- Made by the render's own code (`cuefx.route.orb_path` and `entrance_gain`), never a copy. x, y, diameter: 3 decimals; visibility: 4.
- Errors: `input_invalid` (cue or track does not fit its schema, or the cue has severity-error problems: run validate).
- CLI: `cuefx_api.py preview CUE TRACK [-o FILE]`.

### validate(cue, track=None) → warning[]
- Never raises for a bad cue: every problem is a warning (`schemas/warning.schema.json`), sorted errors → warnings → info.
- Schema problems come back alone, as `invalid_field` (severity error) with the field path in `cause.path`
  (e.g. `route[1].x`). Otherwise: the backend's own checks (`cuefx.cue.check`, the rules the render applies), then
  `no_path`, `check_pass_edge`, `setting_not_used`, `unknown_field`.
- `track` (API form) replaces the cue's own track when given. No track anywhere → `no_track` (error).
- Only `severity: "error"` blocks preview and generate.
- CLI: `cuefx_api.py validate CUE [--track TRACK] [-o FILE]`; exit 0 even when it reports errors.

### propose_timing(cue, track) → cue
- A copy of the cue with `route[1:-1].f` re-timed by `cuefx.route.propose_route_timing` (constant speed in the room,
  pinhole camera: on-screen speed ∝ on-screen size). `route[0]` stays on `release_frame`, the last point keeps its
  frame. The input is not changed. A 1-point route comes back unchanged.
- Errors: `input_invalid` (cue errors, or more points than frames between release and the last point).
- CLI: `cuefx_api.py timing CUE TRACK [-o FILE]`.

TRACK files on the command line may be an API track (list), an analyze result, or a tracker `track.json` (cue form).

### generate(cue, looks) → job[]
The normal path remains disabled. With `CUEFX_LTX_OVERLAP_ENABLED=1` and a configured private worker,
`POST /api/generate` creates one experimental job from an approved hard composite and either a user or Surprise Me
prompt. Both modes enter the same image-edit → main LTX → reverse LTX → assembly path. The manual render remains
available. Shapes: `schemas/job.schema.json`, `schemas/result.schema.json`, samples in `golden/`. Errors: `gpu_memory`,
`server_lost`, `render_failed`, `cancelled`, `input_invalid`. `cancelJob(id)` → the job ends with `status: "cancelled"`.

### uploadClip, uploadReference, checkPrompt — UI mock only (task 17)
Not in the backend yet. Their mock codes (`unsupported_format`, `too_large`, `not_image`, `too_big`, `unreadable`,
`fps_conformed`, `bright_background`, `small_image`) are reserved for them and will be added to `error_codes.json`
when they are built; until then they are not part of the frozen contract.

## 6. Error and warning codes

`schemas/error_codes.json` holds every code with its call, severity and plain-language message template
(`{first}`–`{last}` are the inclusive frames for display). A warning also carries `detail`, the backend's technical
text (field path first), for logs and tooltips.

| Call | Codes |
|---|---|
| any HTTP route (v2.3) | `input_invalid`, `not_found`, `internal` |
| analyze (errors) | `no_hand`, `track_lost`, `clip_unreadable`, `clip_too_long`, `input_invalid` |
| analyze (warning) | `seed_far`; `seed_far_from_body_point` (v2.6, only when `body_point` is not the palm); `side_from_click` (v2.7, info, only for a fist, foot, knee or shoulder whose click is nearer the other side) |
| generate (errors) | `gpu_memory`, `server_lost`, `render_failed`, `cancelled`, `input_invalid` |
| validate, from the UI mock | `no_track` (error), `track_uncertain` (warning), `check_pass_edge` (info), `entrance_shortened` (info, with `fix`), `no_path` (warning) |
| validate, contract | `invalid_field` (error), `unknown_field` (info), `setting_not_used` (info) |
| validate, backend checks (errors) | `behaviour_invalid` (v2.5), `shape_invalid` (v2.7), `ending_invalid` (v2.8), `orbit_invalid` (v2.8), `body_point_invalid` (v2.6; also the 400 of `analyze` and `cue/new`), `missing_field`, `version_unsupported`, `hand_invalid`, `frame_outside_clip`, `track_invalid`, `correction_duplicate`, `value_out_of_range`, `effect_range_length`, `effect_range_outside`, `canvas_invalid`, `frame_outside_effect_range`, `release_not_after_appear`, `route_empty`, `route_start`, `route_order`, `route_position_missing`, `easing_invalid`, `range_reversed`, `range_outside_effect`, `occluder_invalid`, `tint_invalid` |
| validate, backend checks (warnings) | `vanish_extended` (v2.8: a burst or shatter needs a longer vanish; `cue/normalize` also gives it), `release_in_lost_gap`, `orb_holds_after_route`, `window_cut_short`, `window_trimmed` (warning; message now "The effect can run up to 8 seconds (193 frames). Move the vanish earlier or remove late path points."), `hold_path_unused` (v2.5, warning: a `hold` cue still has a `route` or `release_frame`, which are not used); `window_grown` (info) |
| cue/new (warnings, in the `X-CueFX-Warnings` header) | `placeholder_timings` (warning), `started_as_hold` (v2.5, info), `hold_shortened` (v2.5, info), `side_from_click` (v2.7, info: analyze's warning, passed on) |

**`hold`: validate skips every release and route check and warning** (`no_path`, `release_in_lost_gap`,
`orb_holds_after_route`, `release_not_after_appear`, the route errors). `track_uncertain` still reports a lost track
while the effect is held (appear .. vanish.end). Window length, 8k+1, the entrance, `behind` and `vanish` ranges are
checked as for any cue.

`cause.kind`: `field {path}` · `setting {path}` · `track` · `pass {index}` (into `behind`) · `timing` · `tool-draw`.
`fix` (only `entrance_shortened` so far): `{label, op: "set", path, value}`, path dotted (`entrance.seconds`).
`id`: code plus the first frame or the field path, unique within one response; do not parse it.

## 7. The maths (what the render does; the UI copies it)

Notation: `A` = appear_frame, `R` = release_frame, `fps`, `N_in` = `min(round(entrance.seconds × fps), A)`,
`C = A − N_in` (entrance start), `V0, V1` = vanish start/end.

### 7.1 Held position (frames < R)
1. Corrected track (`cuefx.track.apply_corrections`): a correction on a tracked frame shifts the track by an offset
   that is linear between such keyframes and fades linearly to 0 over `correction_falloff` frames beyond the first and
   last; the fade stops at the edge of a lost gap. A correction inside a lost gap re-interpolates that gap linearly
   through its corrections and its two neighbours, and moves nothing else.
2. Held = corrected track + `anchor_offset`.
3. Entrance frames (`C ≤ f < A`) inside an **open** lost gap (no correction in it) hold the position of frame A.

### 7.2 Release point and flight (frames ≥ R)
- `route[0]` x, y null → the held position at R, except when R is inside an open lost gap: the held position at the
  last frame before the gap (the first after it if the gap starts the clip). Only `route[0]` may be null.
- Segment i runs from `route[i]` to `route[i+1]`; `s = (f − f_i) / (f_{i+1} − f_i)`; `u = EASE[route[i+1].easing_in](s)`.
- Position: centripetal Catmull-Rom (Barry-Goldman, α = 0.5) through p0, p1 = route[i], p2 = route[i+1], p3 at `u`.
  Knot step `max(|b − a|^0.5, 1e-9)`. End phantoms mirrored: `p0 = 2·route[0] − route[1]`,
  `p3 = 2·route[last] − route[last−1]`.
- Size: `diameter_px_i + u · (diameter_px_{i+1} − diameter_px_i)`.
- From the last point's frame on, the orb stays at the last point with its diameter.

### 7.3 Easing (applied to the segment's time fraction s, for position and size)
`linear: s` · `ease_in: s²` · `ease_out: 1 − (1 − s)²` · `ease_in_out: s²(3 − 2s)` — the UI's `EASE` table.

### 7.4 Entrance growth (drawn size)
`gain(f) = smoothstep((f − C) / N_in)` with `smoothstep(t) = t²(3 − 2t)`, t clipped to 0..1: 0 at C, 1 from A.
With `N_in = 0` the gain is 1 everywhere. The layout draws the ball with **diameter × gain** and brightness × gain.
`preview.diameter` is that drawn diameter.

### 7.5 Visibility (Merge's opacity, `preview.visibility`)
0 before C · `min(1, (f − C + 1) / fade_in_frames)` from C (1 if `fade_in_frames` is 0) · 1 until V0 ·
`(V1 − f) / (V1 − V0)` for V0 < f < V1 · 0 from V1. The diameter does **not** shrink during vanish; the orb fades.

### 7.6 Effect window
Nothing is drawn outside the effect window: `effect_range` grown earlier in whole 8-frame steps until it covers C
(`window_grown`), then trimmed to at most 193 frames (`MAX_EFFECT_FRAMES`, 8k+1), keeping the start
(`window_trimmed`; a window of 201 frames loses its last 8). Outside it the preview has
`diameter 0`, `visibility 0`, `visible false`, and x, y copied from the nearest window frame.

### 7.7 Phase (first rule that fits)
`before` f < C · `entrance` f < A · `held` f < R · `flight` f < V0 · `vanish` f < V1 · `after`.
For `behaviour: hold` there is no R: `held` runs to V0 and there is no `flight`.
`visible = diameter > 0 and visibility > 0`.

## 8. Track

API track entry (`schemas/track.schema.json`), one per frame `0 … frames−1`:

| API | Cue file | Meaning |
|---|---|---|
| `frame` | `f` | |
| `x`, `y` | `x`, `y` | palm centre (mean of landmarks 0, 5, 9, 13, 17), plate px, **already smoothed** (zero-lag One Euro, per tracked run) |
| `confidence` | `conf` | MediaPipe's per-hand score (it has no presence score), 0..1; **0** on frames where the hand was not detected |
| `uncertain` | `lost` | inside a lost gap: a run of more than 6 undetected frames, or a gap the hand was found again across by a jump |
| `source` (extra) | `source` | `tracked` \| `interpolated` \| `corrected` |
| `palm_w` (extra) | `palm_w` | palm width, plate px |

Short gaps (≤ 6 frames, no jump) are interpolated and not uncertain. Smoothing is fixed in the tracker; there is no
smoothing parameter (see `advanced.track_smoothing`). Send the track back exactly as analyze returned it.

## 9. Render stages (composite pipeline)

| # | `stage_name` | Composite-pipeline step |
|---|---|---|
| 0 | Preparing shot | Validate the cue, decode the plate, compute `orb_path` and the effect window, compute the Make key (cache lookup). |
| 1 | Building reference frame | Make 1–2: the grey-ball layout video (`cuefx.layout.write_layout`) and the look image placed at the first whole-orb frame (`cuefx.look.place_look` → `look_placed.png`, LTX's reference frame). |
| 2 | Rendering (pass 1 of 2) | LTX Layout-to-Render, stage 1 (half resolution) in ComfyUI. |
| 3 | Refining (pass 2 of 2) | LTX Layout-to-Render, stage 2 (full canvas resolution), then the matte: Alpha Gen when it is wired; today `cuefx.ingest`'s temporary luminance matte. |
| 4 | Blending into your footage | Ingest (`scripts/ingest_ltx.py`: fit to the window, matte) and Merge (`cuefx.merge.run_merge`: black clamp, composite, behind ranges, grain; final and compare videos). |
| 5 | Done | Result written. |

When the Make key is already cached, stages 1–3 are skipped (the job jumps from 0 to 4).

## 10. Render result (Review screen)

`schemas/result.schema.json`, sample `golden/sample_result.json` (illustrative):
`{job_id, look, pipeline: "composite", videos{final, compare}, layers{effect_on_black, matte, occluder_matte|null},
checks[], warnings[]}`. Paths are relative to the job's output folder; `occluder_matte` is null without behind ranges.
`checks[]`: `{id, label, status pass|warn|fail, value, unit, frames[s, e)|null, message}`. At least:

| id | What | status |
|---|---|---|
| `plate_unchanged` | max difference of the output from the plate outside the effect window (8-bit levels) | pass at 0 |
| `black_clamp` | the effect's black floor that was subtracted (linear) | pass; warn when high |
| `matte_source` | `temporary` (luminance matte) or `alpha_gen` | warn while temporary |
| `face_shift_px` | placeholder | always `warn`, message "not measured yet" |

**Manual jobs (v2.4):** `GET /api/jobs/{id}/result` (HTTP.md) fills this from files and numbers the steps already have. Paths are download URLs (`/api/jobs/{id}/final.mp4`, `compare.mp4`, `effect_on_black.mp4`, `matte.mp4`: Merge's final and compare, and the ingested effect and matte Merge used); `occluder_matte` is always null (no step writes an occluder matte video). A check nobody computes yet uses the placeholder's way: `status: warn`, `value: null`, message "not measured yet" (`plate_unchanged`, `black_clamp`, `face_shift_px`). `matte_source` is known from the upload: `temporary` (warn) without a matte, `uploaded` (pass) with one. `warnings[]` are the cue's validate warnings (not errors) plus the job's plain-text notes as `render_note` warnings (cause kind `render`).

## 11. Differences the UI must adopt

**Cue fields**
1. `clip` is the plate's file name (string). `clip.width/height` → `plate_size: [w, h]`; `clip.fps` → `fps`; `clip.frames` → `frames` (top level).
2. The `orb` block goes away: `orb.diameter_px` → `diameter_px`; `orb.offset_from_palm_px` → `anchor_offset`; `orb.tint` → `grade.tint`; `orb.glow` → `grade.glow`; `orb.spill` → `grade.spill`.
3. Route points: `frame` → `f`; `diameter` → `diameter_px`. Write `easing_in` on every point (default `linear`); easing on `route[0]` does nothing (reported as `setting_not_used`).
4. Only `route[0]` may have `x, y = null`. A null on any later point is an error (`route_position_missing`); the UI resolves nulls anywhere.
5. `passes_in_front[] {start, end, who: "left hand"}` → `behind[] {start, end, occluder: "left_hand"}`. Occluder values: `left_hand`, `right_hand` (also `person` and `{"point": [f, x, y]}`, not built yet).
6. `advanced.corrections[] {frame, x, y}` → top-level `corrections[] {f, x, y}`; `advanced` keeps only `track_smoothing` and `export_layers`. Corrections are keyframes with a falloff (`correction_falloff`, default 12 frames), not single-frame overrides (see 15).
7. Keep and send back unchanged the fields the UI does not edit: `version` (write `"2.2"`), `transform`, `effect_range`, `canvas_size`, `canvas_rect`, `hand_seed`, `track`, `correction_falloff`, `seed`, `grade.brightness`, `grade.opacity`, `grade.person_light`, `grade.fade_in_frames`, `interaction`.
8. `hand_seed {f, x, y}` is the analyze seed and belongs in the cue (the UI keeps the seed point only in memory).
9. Sample cue: the UI's `passes_in_front` is `[80, 92)` (frames 80–91); the backend's example and the golden cue have `behind [80, 91)` (frames 80–90). The UI brief says the pass is 80–91, BRIEF.md section 5 says ~78–92. The owner should decide which is right for clip 71319; the golden keeps the example's 91.

**Analyze and track**
10. analyze returns `{track, summary{tracked, total, uncertain_ranges}, hand_size_px, warnings}`: `handSize` → `hand_size_px`, `summary.uncertain` → `summary.uncertain_ranges`, new `warnings[]`.
11. `summary.tracked` counts the frames where the hand was **detected**, not `total − uncertain` (short interpolated gaps are neither tracked nor uncertain).
12. Track entries carry two extras, `source` and `palm_w`; send them back unchanged. `confidence` is 0 on undetected frames even where `uncertain` is false; use `uncertain`, not a confidence threshold, to mark bad frames.
13. A click that misses the hand is not an error: analyze uses the nearest hand and returns a `seed_far` warning. `no_hand` only when the seed frame has no hand at all.
14. `track_lost` is an analyze error only when more than half of the frames are inside lost gaps; otherwise lost gaps come back as `uncertain_ranges` and as validate warnings.

**Preview maths (section 7)**
15. Held position: corrected track + `anchor_offset` with the falloff rules of 7.1 (the UI replaces just the corrected frame). There is no `track_smoothing` box filter: the track is already smoothed; drop the UI's `smoothedTrack`.
16. Entrance frames inside an open lost gap hold the appear_frame position (7.1.3).
17. Release point: from the held position at `release_frame`, or from the last frame before the gap when `release_frame` is in an open lost gap (7.2). The UI uses the palm at `route[0].frame` plus the offset.
18. Catmull-Rom knot step floor is `1e-9`, not `1e-3`.
19. Easing: the UI's `EASE` table and its use (destination point's `easing_in`, applied to `t` for both position and size) are the contract as they are; no change.
20. Entrance growth: drawn diameter = `diameter_px × smoothstep((f − C) / N_in)` (smoothstep `t²(3 − 2t)`), not `ease_out(t)`; with `N_in = 0` the orb is full size from the start of the window. The stand-in's brightness grows the same way.
21. Vanish: the orb **fades** (`visibility`, 7.5) and keeps its size; the UI shrinks the diameter instead.
22. Fade-in: visibility ramps over `grade.fade_in_frames` from the entrance start C (7.5); the UI ignores it.
23. New preview field `visibility` (0..1, the opacity Merge applies). `visible = diameter > 0 and visibility > 0`.
24. Nothing is drawn outside the effect window (7.6): there `diameter 0`, `visibility 0`, `visible false`, x and y copied from the nearest window frame.
25. x, y and diameter are floats with 3 decimals; do not round to whole pixels before comparing with the backend.
26. Phases: the same six names and boundaries as the UI's `phaseAt` (UI `es` = backend C). No change.

**Validate**
27. Warnings may carry `detail` (backend text). New cause kinds `field {path}` and `setting {path}`. Many more codes (section 6); show any unknown code with its `message` and `severity`.
28. `check_pass_edge` is **info** (the backend cannot measure the mask edge without the video) and covers the last two frames of the range, `[max(start, end − 2), end)`; the UI has a one-frame warning.
29. `track_uncertain` is only reported where the track drives the orb: open lost gaps (no correction in them) overlapping the entrance or the hold; `frames` is that overlap, not the whole gap. Gaps elsewhere (before the entrance, after the release) are only in `summary.uncertain_ranges`. A release inside a gap is `release_in_lost_gap`.
30. `no_path` is for a route with exactly one point; an empty route is the error `route_empty`.
31. Warnings come sorted errors → warnings → info. Do not parse `id` (format `<code>_<first frame>` or `<code>_<path>`, `_2`, `_3` on repeats); the UI's ids (`pass_edge_0`, `entrance_short`, …) change.
32. `setting_not_used` (info) appears for `advanced.track_smoothing`, `advanced.export_layers`, `grade.spill`, `grade.person_light` (when > 0) and a non-linear `route[0].easing_in`. `unknown_field` (info) for top-level fields the schema does not know.
33. Severity `error` blocks preview and generate: `preview_path` returns `input_invalid` until validate is clean of errors.

**Jobs and results (shapes only until task 18)**
34. Job fields are snake_case and in **seconds**: `stage_name`, `elapsed_s`, `eta_s` (null when unknown); no `elapsedMin`, `etaMin`, `totalMin`, `startedAt`.
35. `job.look` is the look **id** (string), not the look object; the UI takes `tint`, `name` and `minutes` from its own look list. Jobs have no `tint`, `frame`, `summary`, `warning`, `failCause`.
36. `job.error` is `{code, message}` with codes `gpu_memory` (UI: `out_of_memory`), `server_lost`, `render_failed`, `cancelled`, `input_invalid`; there is no `canChange` (offer "change settings" for `gpu_memory` only).
37. Per-render warnings (the UI's `colour_gain`) move to `result.warnings`; the Review screen reads `result.checks` (section 10).
38. Cancelling ends a job with `status: "cancelled"` (the UI removes it from the list); the UI may still hide it.
39. Analyze has no progress callback yet; the progress steps (`Detecting hands`, …) come with the server (task 17).

## 12. Manual render

ComfyUI runs on a separate GPU machine and there is no deployment yet. The manual render is the demo path: the
app **exports a ready package**, the owner runs it in ComfyUI by hand, **uploads the result**, and the backend runs
ingest + Merge and shows the real LTX final next to the stand-in. Nobody wires nodes by hand. HTTP routes:
`HTTP.md`, "Manual render".

### The graph: `comfyui/CueFX_Phase1_Quality/CueFX_Phase1_Quality.json`
Read from the committed graph (UI format). It loads exactly two files from `ComfyUI\input\` and has one prompt:

| Node | Type | Reads / does | The package sets |
|---|---|---|---|
| 5001 "Load layout video (Clay Video)" | `LoadVideo` | the layout: Make's `layout.mp4` (grey ball on black, canvas size, one frame per effect-window frame, 24 fps) | `<job_id>_layout.mp4` |
| 9009 "Load reference image (Look Still)" | `LoadImage` | the look: Make's `look_placed.png` (the look on black at the canvas size, placed on the first whole-orb ball); the graph uses it as the reference at frame −1 in both stages | `<job_id>_look.png` |
| 5508 "Prompt · describe YOUR finished shot" | `PrimitiveStringMultiline` | the positive prompt | the prompt |
| 5509 "Prompt (negative)" | `PrimitiveStringMultiline` | negative prompt | unchanged (empty) |
| 9047 "Generic export" | `CueFXGenericSave` | writes `ComfyUI\output\<prefix>_<UTC date>_<UTC time>_<6 hex>\effect.mp4` (plus PNG frames and `run.json`): the layout's own size, frame count and fps, verified by decoding; RGB on black, H.264 CRF 18 yuv420p | prefix `cuefx/<job_id>` |

Nothing else changes: models, LoRA, sizes, steps, sigmas, seed 42, CFG 1. **The padded-frame limit (v2.5):** the graph
ships with a limit of 121 frames (node 9045 `CueFXOfficialPrepare`, widget `max_padded_frames`, the third value). The
effect window is at most 193 frames (8k+1), so the package sets that one value to `max(121, window length)`: a window
of 121 frames or less gets exactly the graph above (byte for byte), a longer one gets its own length. The latent length
(node 9002 and the `EmptyLTXVLatentVideo` inside the Preprocess subgraph) is linked to that node's `padded_frames`
output, so the 121 typed in those two widgets is not used. The value is in `manifest.json` as `graph_frames`. The prompt goes into node 5508, which feeds both the local
Gemma encoder and the "Gemma API Text Encode" nodes; the switch uses the API nodes only when an LTX API key is
typed in, so with none the local (open-weights) encoder is used. Task 06b removes the API nodes.

Corrections to the expected package: the file names are **per job** (`<job_id>_layout.mp4`, `<job_id>_look.png`),
so a stale file from an earlier export can never be picked up silently (ComfyUI says "file not found" instead), and
the output file is **`effect.mp4`** (the save node's name), uploaded as `effect_on_black`. There is no matte output
in this graph: `matte` is optional, and without it Merge uses today's temporary luminance matte (`matte_source`
`temporary`). `reference.png` is not needed (that was the parked interaction pass).

### The package (zip)
```
comfy_input/<job_id>_layout.mp4     copy into ComfyUI\input\
comfy_input/<job_id>_look.png       copy into ComfyUI\input\
CueFX_Phase1_Quality_<job_id>.json  drag onto ComfyUI: file names, prompt and output prefix already filled in
manifest.json                       what the upload must match
HOW_TO_RUN.txt                      the three steps
```
`manifest.json`: `{job_id, make_key, clip_id, clip, look, prompt, effect_range{start, end}, comfy_input{layout,
look}, graph, graph_frames (v2.5), output{file: "effect.mp4", width, height, frames, fps, comfy_folder},
upload{effect_on_black, matte}, warnings[]}`. `output` is measured on the exported `layout.mp4` (for the golden cue: 1024×576, 89 frames, 24 fps,
window 26–114).

Look images: `CUEFX_LOOKS_DIR/<look>.png` (default `<CUEFX_DATA_DIR>/looks`), the look on pure black. A missing
file is `input_invalid` with the folder in the message; none is ever made up. The look id is part of the make key.

### The job
A manual job is a normal job (`schemas/job.schema.json`) with `manual: true`:

| When | `status` | `stage` / `stage_name` | `progress` | `files` |
|---|---|---|---|---|
| export asked (v2.4, `?wait=0`) | `running` | 0 Preparing shot | 0 | – |
| exported | `waiting` | 2 Rendering (pass 1 of 2): the LTX render is being run by hand | 0.4 | `package_download` |
| result uploaded, ingest + Merge | `running` | 4 Blending into your footage | 0.8 | `package_download` |
| finished | `done` | 5 Done | 1 | + `final_preview` (inline, browser H.264), `final_download` |
| ingest or Merge failed, or the server stopped while blending | `failed`, `error.code` `render_failed` | the stage it was in | | `package_download` |

`elapsed_s` is the backend's own time (export + ingest + Merge), not the time at the GPU. `eta_s` is null.
`warnings[]` are plain-text notes (Make, the prompt, ingest, Merge; the temporary matte note while no matte was
uploaded). A `waiting`, `done` or `failed` job takes a new upload (it replaces the previous one).

### Upload checks (all `input_invalid`, the job stays as it was)
- `effect_on_black` (and `matte`, if sent) must have the manifest's width × height, frame count and fps (±0.01).
  The message names what is wrong, e.g. "effect_on_black is 640x256, the export expects 512x256".
- The make key must still be the job's: the cue sent with the upload (optional form field `cue`; the UI should send
  its current cue) and the look image on disk. A Make field changed: "the cue changed since export"; the look file
  changed: "the look image … changed since export". A Merge-only change (grade, vanish, behind) is fine: Merge uses
  the cue sent. Without `cue`, the exported cue is used.

Ingest and Merge are the CLI's own code (`cuefx.pipeline.ingest_ltx` = `scripts/ingest_ltx.py`, then
`cuefx.merge.run_merge` = `scripts/merge_clip.py --effect-dir`); a test checks the decoded frames are identical.

## 13. Changelog

| Version | Change |
|---|---|
| 3.0 | Additive (request 39). `POST /api/analyze` and `POST /api/cue/new` may omit `hand_seed` when `body_point` is not the palm. The explicitly requested part and side are tracked without click-based side selection or click warnings. `analyze_result` then gains `body_point_seed {frame, x, y}`, a real highest-confidence tracked sample; `cue/new` stores it as the cue's required `hand_seed`. Seeded callers and palm tracking are unchanged. `contract_version` 3.0. |
| 2.9 | Additive (task 38). `GET /api/defaults` gains `ending_default_seconds {burst: 0.6, shatter: 1.0}` (what the UI sets as the vanish length when the artist picks that ending; `ending_seconds` stays the minimum each ending needs). The ending lasts for the whole vanish, so the vanish length is the ending length (no new cue field). The look of burst and shatter is stronger (section 2, `ending`: burst 1.8 × over the first 40 % with the light at 1.6 × (near white), held to 60 %, then fading; shatter fragments ≥ 0.25 d, fully bright for the first 50 %, kept inside the picture). `contract_version` 2.9. |
| 2.8 | Additive (task 36). **Cue 2.6, `ending`** (burst, shatter), **`behaviour: charge_up`** and **`behaviour: orbit`** with the `orbit` block (section 2), error codes `ending_invalid` and `orbit_invalid`, warning code `vanish_extended` (`hold_path_unused` is also given for an `orbit` cue, with its own message); `POST /api/cue/normalize` keeps its cue body and gains the header `X-CueFX-Warnings` (a JSON list of warnings, as `cue/new` has: `vanish_extended`, or `window_trimmed` where the cap cuts the ending). **`GET /api/defaults`:** every default the UI needs, read from the backend's own constants: `{spin_deg_per_s {sphere, ring, crystal, cube}, ending_seconds {fade, burst, shatter}, charge_up {start_scale, end_scale, pulse_hz, pulse_amount}, orbit {radius_sizes, speed_deg_per_s, tilt_deg}, max_window_frames, orbit_tilt_deg_by_part {head}}` (`orbit_tilt_deg_by_part` was added after the first 2.8 text: the head's default tilt is 80). **`POST /api/validate`** keeps its warning-list body and gains the response header `X-CueFX-Effect-Range` (JSON `{"start", "end"}`): the `effect_range` that `cue/normalize` would compute for the same cue (absent when the cue does not fit its schema or its timings cover nothing). `cue/normalize` already returns it in the cue. **Starting timings:** a fist, foot or knee track with a movement faster than 1.0 size/frame starts as a throw at that frame (section 2a); the foot's default `anchor_offset` is now `[0, −0.5 d]`. `contract_version` 2.8. |
| 2.7 | Additive (task 31). **Cue 2.5:** top-level `shape` (`sphere` default, `ring`, `crystal`, `cube`) and `spin_deg_per_s` (section 2); a cue without them keeps its version and draws exactly as before (the sphere's layout, stand-in, golden files and Make keys are byte for byte unchanged). Error code `shape_invalid`. **Side by click:** for a fist, foot, knee or shoulder the click in the seed frame picks the side whose point is nearer (when both sides are visible there); `analyze_result` gains an optional `body_point {part, side}` (only for a body point other than the palm) with the side that was tracked, `cue/new` writes it into the cue, and when it differs from the side asked for the info warning `side_from_click` is in `analyze_result.warnings` (and in `cue/new`'s `X-CueFX-Warnings`). The palm and the head are unchanged. **Body-point `track_lost`** now names the point in its message ("CueFX lost the right foot on …"); the palm's text is unchanged. A pose landmark outside the plate counts as missing for that frame, whatever its visibility. `contract_version` 2.7. |
| 2.6 | Additive (task 32). **Cue 2.4:** top-level `body_point {part, side}` (section 2); missing = the palm of `hand`, so older cues load and save unchanged (a v2.2 or v2.3 file keeps its version unless it sets `body_point`, then it is written as 2.4). `POST /api/analyze` and `POST /api/cue/new` take an optional `body_point` (missing or `palm` = today's hand tracker, same cache key, same result). Other parts use the Pose Landmarker; its 33 landmarks per frame are cached once per clip. In a body-point track `palm_w` is the size measure of the body point. Error code `body_point_invalid` (validate; 400 on analyze and cue/new), warning code `seed_far_from_body_point` (analyze). `contract_version` 2.6. |
| 2.5 | Additive (task 33). **Cue 2.3:** top-level `behaviour` (`hold_throw` default, or `hold`); `release_frame` may be null (a `hold` has no release and `route: []`). Windows up to **193 frames** (8 s; was 121): `window_trimmed` now means more than 193 and has a new message. Error code `behaviour_invalid`; warning codes `hold_path_unused` (validate), `started_as_hold` and `hold_shortened` (cue/new, info, in `X-CueFX-Warnings`). `cue/new` starts as `hold` when the clip has a steady palm and no throw after it. The manual export sets the graph's padded-frame limit to `max(121, window length)` and writes `graph_frames` in `manifest.json`. v2.2 cue files load and are written back unchanged; no existing field renamed or removed; `contract_version` 2.5. |
| 2.4 | Additive (task 30, backend gaps the UI hit). `GET /api/looks`; `GET /api/jobs/{id}/result` and the files it links (`GET /api/jobs/{id}/{name}`); `POST /api/manual/export?wait=0` (the job is `running`, stage 0 "Preparing shot", until the package is ready); **the heavy routes (`/api/standin`, `…/element.mp4`, `/api/manual/export`, `/api/manual/{id}/result`) now default to `wait=0`** (`?wait=1` is the old blocking answer). Schemas: `result.videos.compare`, `result.layers.effect_on_black` and `.matte` may be null (the file does not exist); `warning.cause.kind` gains `render`; error code `render_note` (`call: generate`, severity warning, message `{detail}`); `contract_version` 2.4. Nothing renamed or removed. |
| 2.3 | Additive (task 27, server responsiveness). Preview entries gain optional `held {x, y}` (corrected track + anchor; `/api/preview` always sends it, `api.preview_path` only with `with_held=True`, so the golden file is unchanged). Error codes `not_found` (404, was `input_invalid`) and `internal` (unexpected 500, was `render_failed`). New `schemas/standin_status.schema.json` (`GET /api/standin/{id}`, `?wait=0` on the heavy routes; HTTP.md). Nothing renamed or removed; default responses keep their shape. |
| 2.3 | Additive (task 21, manual render). Job: `status` gains `waiting`; new optional fields `manual` (bool), `clip_id`, `make_key`, `files {package_download, final_preview, final_download}`, `warnings[]` (strings). Section 12. Nothing renamed; the cue is unchanged (still 2.2). |
| 2.2 | Cue: `route[].easing_in`, `route[].note`, `grade.spill`, `prompt`, `reference_image`, `advanced` (section 2). |
