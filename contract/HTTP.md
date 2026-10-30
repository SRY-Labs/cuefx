# CueFX local HTTP API (v0.1), for the UI's `live` mode

The server (`server/`) **only wraps** `cuefx/api.py`: it adds clip storage, a browser video proxy and the HTTP shape.
Bodies and results are exactly the schemas in `schemas/` (cue v2.3, snake_case, plate pixels, frames from 0, ranges
`[start, end)`); nothing in this file changes `CONTRACT.md`. Examples below come from `golden/` (clip 71319).

Start: `python scripts\run_server.py` (or `python -m server`). Settings: environment variables, see the end.
Reference upload and prompt check are **not** served. Generate/cancel remain on `mock` unless `/api/health` reports
`features.ltx_overlap_generate: true`; the experimental path is disabled by default. The manual render is served.

## Routes

| Route | Body → result | Wraps |
|---|---|---|
| `GET /api/health` | → `{status, version, contract_version}` | – |
| `GET /api/clips` | → `[{clip_id, clip, fps, frames, plate_size}]` (uploads) | – |
| `POST /api/clips` | multipart, field `file` → one clip entry (of the conformed plate); **201** new, **200** same bytes seen before | `cuefx.conform` |
| `GET /api/clips/{clip_id}/video` | → `video/mp4` (H.264 High, yuv420p, Rec.709, 1920×1080, 24 fps, the conformed frame count, range requests) | – |
| `GET /api/clips/{clip_id}/frames/{f}.jpg` | → `image/jpeg`, exactly frame `f` (0-based) at 1920×1080; cached (task 27, see Clips) | – |
| `POST /api/analyze` | `{clip_id, hand, hand_seed?, body_point?}` → `analyze_result`; `hand_seed` may be omitted only for a non-palm `body_point` (v3.0) | `api.analyze` |
| `POST /api/validate` | `{cue, track?}` → `warning[]`; header `X-CueFX-Effect-Range` (v2.8): JSON `{start, end}`, the `effect_range` `cue/normalize` would compute for this cue (absent when the cue has schema errors or its timings cover nothing) | `api.validate`, `api.computed_effect_range` |
| `GET /api/defaults` | → `{spin_deg_per_s, ending_seconds, ending_default_seconds, charge_up, orbit, max_window_frames, orbit_tilt_deg_by_part}` (v2.8; `ending_default_seconds` v2.9): the defaults the UI needs, read from the backend's constants. `ending_seconds` = the minimum vanish each ending needs; `ending_default_seconds` = the vanish the UI sets when the artist picks burst (0.6) or shatter (1.0) | `api.defaults` |
| `POST /api/preview` | `{cue, track?}` → `preview[]`, each entry with `held` (v2.3) | `api.preview_path(..., with_held=True)` |
| `POST /api/propose_timing` | `{cue, track?}` → `cue` | `api.propose_timing` |
| `POST /api/cue/new` | `{clip_id, hand, hand_seed?, body_point?}` → `cue` (full, valid v2.3, or v2.4 with `body_point`; with the track); `hand_seed` may be omitted only for a non-palm `body_point` (v3.0) | `analyze` + `api.new_cue` |
| `POST /api/cue/normalize` | `{cue}` → `cue` (derived fields recomputed); header `X-CueFX-Warnings` (v2.8): a JSON list of warnings about what it changed (`vanish_extended`, `window_trimmed`; `[]` if nothing) | `api.normalize_with_warnings` |
| `POST /api/standin` | `{cue, track?, clip_id?}` → `standin_status` (202 while queued/running, 200 once done; **default since v2.4**); `?wait=1` → `standin_result` when finished | Make + Merge (`cuefx.pipeline`, `cuefx.merge`) |
| `GET /api/standin/{standin_id}` | → `standin_status` (v2.3) | – |
| `GET /api/standin/{standin_id}/{name}` | → `video/mp4`; `name` = `hardcomp_preview.mp4` · `hardcomp.mp4` · `element.mp4` · `layout.mp4`; `element.mp4` → `standin_status` (202) while it is built (default since v2.4); `?wait=1` waits for the file | – |
| `GET /api/looks` | → `[{id, name, file}]` (v2.4), see "Looks" |
| `POST /api/manual/export` | `{cue, clip_id, look, prompt?}` → **202** `job` (`running`, stage 0 "Preparing shot"; default since v2.4); `?wait=1` → `application/zip` (the ComfyUI package). Header `X-CueFX-Job-Id` either way | Make 1-2 (`cuefx.pipeline.make_layout`), `cuefx.comfy_graph.fill_quality_graph` |
| `POST /api/manual/{job_id}/result` | multipart: `effect_on_black` (file), `matte` (file, optional), `cue` (JSON text, optional) → `job` (`running`, 202; default since v2.4); `?wait=1` → `job` (done) | ingest + Merge (`cuefx.pipeline.ingest_ltx`, `cuefx.merge`) |
| `GET /api/manual/{job_id}/{name}` | → `name` = `package.zip` · `final_preview.mp4` (inline) · `final.mp4` (attachment) | – |
| `GET /api/jobs` | → `job[]`, newest first (manual jobs; generate's later) | – |
| `GET /api/jobs/{job_id}` | → `job` | – |
| `GET /api/jobs/{job_id}/result` | → `result` of a done job (v2.4), see "Manual render" | – |
| `GET /api/jobs/{job_id}/{name}` | → `video/mp4` (range requests); `name` = `final.mp4` · `compare.mp4` · `effect_on_black.mp4` · `matte.mp4`; `?download=1` → attachment | – |

### Experimental automatic LTX routes

Disabled by default. With `CUEFX_LTX_OVERLAP_ENABLED=1` and `CUEFX_COMFY_URL` configured:

- `POST /api/generate` takes `{cue, clip_id, standin_id, prompt_mode, prompt_text?, saved_look_identity?, experimental_ltx_overlap:true}` and returns one job (202).
- `POST /api/generate/{job_id}/retry` resumes the same job and reuses hash-verified completed stages.
- `POST /api/generate/{job_id}/cancel` cancels the job and interrupts the configured worker when it is running.
- `GET /api/health` reports `features.ltx_overlap_generate`; the UI keeps its existing mock Generate behavior when false.

Both `user` and `surprise` prompt modes enter the same image-edit, main LTX, reverse LTX, assembly and existing result path.

### Renders run on a queue (task 27)
Heavy work (stand-in Make + Merge, the element video, the manual export's Make, the manual import's ingest + Merge)
runs on one **render queue**: one worker thread, one render at a time, the rest waiting in order. The light routes
(health, clips, frames, video, validate, preview, propose_timing, cue/normalize, jobs, the status routes) never wait
on it. Measured on the laptop during a cold 71319 stand-in: health at most 31 ms, validate 147 ms, preview 235 ms.

- **`?wait=1`:** the heavy routes answer when the work is done, with the body they had before task 27. Scripts and
  tests that want the finished answer pass it. The request waits on the queue; it no longer blocks anything else.
- **Default (no query), since v2.4:** the same as `?wait=0`.
- **`?wait=0`** (the default): answers at once. `POST /api/standin?wait=0` → `standin_status` (`schemas/standin_status.schema.json`):
  **202** while `queued` or `running`, **200** when `done` (a cached render answers `done` straight away, with the
  `standin_result` in `result`). Poll `GET /api/standin/{standin_id}` (about once a second) until `status` is `done`
  or `failed` (`error` is then the contract error). `GET …/element.mp4?wait=0` answers the same status (202; see
  `element`) until the video exists, then the file. `POST /api/manual/{job_id}/result?wait=0` → the `running` job
  (202); poll `GET /api/jobs/{job_id}`. `POST /api/manual/export?wait=0` → the `running` job (202) while the package is
  built; poll `GET /api/jobs/{job_id}` until it is `waiting`, then `GET /api/manual/{job_id}/package.zip`.
- The same stand-in requested twice joins the render already on its way. A manual job that is still blending
  refuses another upload: **409** `input_invalid`.
- v2.4 made `wait=0` the default (the UI polls). A client that sends no `wait` and expects the finished body must
  now send `?wait=1`.

`track` is the API track (`track.schema.json`). When it is left out, `preview` and `propose_timing` use the cue's own
`track` (cue-file form, converted); `validate` then reports `no_track` if that is empty too, as the Python call does.
Unknown keys in a request body are refused (400).

### Clips
`clip_id` is the first 12 hex digits of the SHA-256 of the file: the same bytes always give the same id, whatever the
name, and are stored once (`<data>/uploads/<clip_id>/<name>`).
Accepted: `.mov .mp4 .mkv .webm`, up to `CUEFX_MAX_UPLOAD_MB`, up to 1440 frames (the analyze limit, checked on the
conformed frame count).

**Conform (task 23, BRIEF section 2.1).** The upload is kept unchanged. On upload the server also makes the **plate**:
the clip scaled to fit inside 1920×1080 keeping its aspect ratio, centred, padded with black, resampled to 24 fps
(ffmpeg `fps` filter), with the rotation tag (phone clips) applied first; H.264 near-lossless (CRF 6), Rec.709,
audio kept as AAC; cached by clip id in `<data>/cache/<clip_id>/plate.mp4`. A clip that already is 1920×1080 at a
constant 24 fps with no rotation is used as it is (no copy, no re-encode). **Everything downstream reads the plate**:
analyze, the frames route, the proxy video, the stand-in. So a clip entry always says `plate_size` `[1920, 1080]`,
`fps` `24.0` and the conformed `frames` (= round(duration × 24)), and the analyze track and cue coordinates are in
plate pixels. The upload can take a while for a long or 4K clip (the conform runs before the answer); the same bytes
later answer at once. A frame count over the limit is refused (`clip_too_long`, 400) and nothing is kept.
`cue/new` writes the clip's `transform` into the cue: `plate_px = source_px * scale + offset`, the source taken after
rotation; `{"scale": 1.0, "offset": [0, 0]}` for a clip that already is a plate. Examples: 1280×720 → scale 1.5,
offset [0, 0]; 1080×1920 portrait → scale 0.5625, offset [656, 0]; 3840×2160 → scale 0.5. (Scaled size and offset are
rounded to even pixels, as yuv420p needs.) No schema change: `transform` was already in the cue.
The proxy video is made on the first request (about 3 s for 193 frames) and cached; later requests are served from disk.
**Frames (task 27):** served from a small memory cache, else from `<data>/cache/<clip_id>/frames/<f>.jpg`. Uploading a
clip or opening its video starts one background pass that decodes the plate once and writes every frame there; a
frame not written yet is decoded directly (seek to the keyframe before it). Clip 71319 has a single keyframe, so a
late frame used to decode from the start (frame 190: about 2 s); cached, every frame takes a few ms.

```
GET /api/clips
[{"clip_id": "8861297b65b9", "clip": "throw_71319.mov", "fps": 24.0, "frames": 193, "plate_size": [1920, 1080]}]
```

### Analyze
`hand` is `right` | `left` (the person's hand). `hand_seed` is the click, `{frame, x, y}`; the cue's own name
`{f, x, y}` is accepted too (give one of `frame` / `f`). It remains required for a missing or explicit palm
`body_point`. Synchronous: about 12 s cold for 71319, then the same
request is answered from a cache (about 20 ms) keyed by clip content, hand, seed and a hash of the `cuefx/` code.
`hand` only names the person's hand for the cue: the tracker follows the clicked hand either way.

**`body_point` (v2.6, optional):** `{"part": "palm"|"fist"|"foot"|"knee"|"shoulder"|"head", "side": "left"|"right"|null}`,
on `analyze` and `cue/new`. Missing or `palm` is the hand tracker above, unchanged. Another part is tracked from the
MediaPipe Pose Landmarker (about 0.1 s per frame on a laptop CPU, about 20 s for a clip, once per clip: the 33 landmarks
of every frame are cached, so asking for another body point of the same clip does not run Pose again; the result of each
part and side is cached like the palm's). `hand_seed` is the click on that body point; if no pose is near it the result
carries a `seed_far_from_body_point` warning and the track is made anyway. `palm_w` and `hand_size_px` are the body
point's size measure (CONTRACT.md section 2). `400 body_point_invalid`: unknown part, missing side (except the head), or
a palm whose side is not `hand`. A body point that is missing in more than half of the clip (a foot below the frame) is
`400 track_lost`; its message names the body point ("CueFX lost the right foot on 120 of 158 frames. …").

**Side by click (v2.7).** For a fist, foot, knee or shoulder the click in the seed frame picks the side: the one whose
point is nearer the click wins, even if `body_point.side` says the other (when either side is not visible in the seed
frame the side you gave is kept). The result then carries `"body_point": {"part", "side"}` (only for a body point
other than the palm: the side that was tracked) and, when it differs from the one asked for, an info warning
`side_from_click` in `warnings`. `cue/new` writes that `body_point` into the cue (and passes the warning on in
`X-CueFX-Warnings`). The palm and the head are unchanged.

**No-click body-point switch (v3.0, request 39).** For a non-palm `body_point`, `hand_seed` may be omitted. The
explicitly requested part and side are then tracked; there is no click-based side override and no
`seed_far_from_body_point` or `side_from_click` warning. `analyze_result.body_point_seed {frame, x, y}` is a real
highest-confidence sample from that track, not a made-up click. `cue/new` records that sample in the cue's required
`hand_seed {f, x, y}`. Existing requests with a click behave exactly as before. A missing seed for the palm is
`400 input_invalid`.

```
POST /api/analyze   {"clip_id": "8861297b65b9", "hand": "right", "hand_seed": {"frame": 60, "x": 1160, "y": 550}}
200  {"track": [... 193 entries as golden/throw_71319.track.json ...],
      "summary": {"tracked": 115, "total": 193, "uncertain_ranges": [[0, 10], [107, 111], [137, 193]]},
      "hand_size_px": 85.3, "warnings": []}
```

### Validate, preview, propose_timing
```
POST /api/validate   {"cue": <golden/throw_71319.cue.json>, "track": <golden/throw_71319.track.json>}
200  [{"id": "window_grown_26", "severity": "info", "code": "window_grown", "frames": [26, 115], ...}, ...]
     = golden/throw_71319.validate.json

POST /api/preview    same body
200  [{"frame": 38, "x": 1209.723, "y": 496.812, "diameter": 101.852, "visibility": 1.0, "visible": true,
       "phase": "entrance"}, ... 193 entries]
     = golden/throw_71319.preview.json, plus "held" on every entry (v2.3, below)

POST /api/propose_timing   same body
200  the cue with route[1:-1].f re-timed
```
**`held` (v2.3, additive).** Every preview entry also carries `held: {x, y}`: the corrected track plus `anchor_offset`
at that frame (`cuefx.route.held_xy`, CONTRACT.md 7.1 steps 1-2), plate px, 3 decimals: where the hand marker goes,
on every frame (also outside the effect window and after the release). The other fields are unchanged: without
`held` the answer is still `golden/throw_71319.preview.json`, which is also what `api.preview_path` returns by default.

`validate` answers **200 with warnings** for a bad cue (it never fails for a cue; schema problems come back as
`invalid_field`). `preview` and `propose_timing` answer **400 `input_invalid`** while the cue has errors.

### New cue, normalize (task 19)
`cue/new` takes the same body as analyze (with the optional `body_point`, which the cue then carries as version 2.4), runs it (or answers from the analyze cache) and returns a full cue v2.3 the
UI edits from then on. The UI never makes up a backend field: it sends the cue back to `cue/normalize` after every
edit, then `validate`, then `preview`.

- From the clip: `clip` (file name), `fps`, `frames`, `plate_size`; `hand`, `hand_seed {f, x, y}` (the request click,
  or the real `body_point_seed` sample for a seedless non-palm request); `track` (the analyze track, in cue form).
- Starting timings and size come from **the clip's own track** (`cuefx.start_timings`, task 28; the artist edits them):
  the longest steady run of the palm (tracked, speed < 0.06 palm widths per frame, gaps of up to 2 frames allowed)
  gives `appear_frame` = run start + the entrance frames (0.5 s) and `release_frame` = the throw after the run (the first frame within 1.5 s whose palm speed is above 1.0 palm widths per frame, or the first frame of a lost gap; else run end + 1; task 29); `vanish` is
  0.25 s after the release and 0.25 s long; `diameter_px` = 1.15 x the median palm width over the run and
  `anchor_offset` = [0, -0.65 x diameter]. If appear would make the effect window longer than the limit (193
  frames, 8 s) it moves later; the release never moves.
  **No throw found (task 33, v2.5):** when there is a steady run but neither a fast frame nor a lost gap within 1.5 s
  after it, the cue starts as `"behaviour": "hold"`: `release_frame` null, `route` `[]`, `appear_frame` as above,
  `vanish` right after the run (`vanish.start` = run end + 1, clipped to the last frame; 0.25 s long). A hold longer
  than 193 frames keeps `appear_frame` and the vanish moves earlier so the window is 193 frames. A throw found is
  `"behaviour": "hold_throw"`, exactly as before; the fallback with no steady run stays `hold_throw`. Every new cue
  carries `behaviour`. With no steady run of at least entrance + 8 frames (a short or
  shaky clip), appear is at 25 % and release at 60 % of the clip and the response carries the `placeholder_timings`
  warning (below). `look`, `seed`, `grade`, `entrance`, `canvas_size` are the starter values copied from the canonical
  example `golden/throw_71319.cue.json` (`api.STARTER_FIELDS`); `behind` stays empty.
- **Warnings:** the body stays a cue (no schema change). Its warnings, in `warning.schema.json` form, are the
  `X-CueFX-Warnings` response header (exposed to CORS, so a page on another origin can read it), a JSON list: `[]`, or `placeholder_timings` (severity warning, `cause.kind`
  `timing`, `frames` appear..release), or, for a hold, `started_as_hold` (severity info, `timing`, `frames`
  appear..vanish.start) and, if it was cut to the limit, `hold_shortened` (info, `timing`, `frames` the vanish).
- Fixed: `transform` the clip's conform transform (identity if it already was a plate; see Clips), `canvas_rect` the whole plate (BRIEF section 4, "full frame
  now"), `corrections: []`, `behind: []`, `route` = only `route[0]` (`f` = `release_frame`, `x, y` null).
- `effect_range`: derived, as in normalize.

`cue/normalize` recomputes the **derived** fields and changes nothing else:

| Field | Rule (existing backend code) |
|---|---|
| `version` | a v2.1 cue is upgraded to `"2.2"` as on load; v2.2 and v2.3 keep theirs, except a `hold` cue, which is `"2.3"` (`cuefx.cue.cue_from_dict`) |
| `release_frame`, `route` (`hold` only) | missing: `null` and `[]`; for `hold_throw` they are filled as above |
| `route[].easing_in` | `"linear"` where missing |
| missing fields | filled with the `cue/new` values above; present fields are kept, an explicit `null` too |
| `effect_range` | `cuefx.window.propose_effect_range(appear_frame, vanish.end, frames)`: the shortest 8k+1 window over `appear_frame − 4 … vanish.end`, inside the clip. The entrance is not added here: Make grows the window over it (`window_grown`, info), as for the golden cue. Kept as it is when appear/vanish are missing or cover nothing. |

`route[0].f` is **not** derived: the UI moves it with `release_frame` (as it does now); validate reports
`route_start` otherwise. Example: normalize of `golden/throw_71319.cue.json` returns it unchanged (key order aside).

```
POST /api/cue/new   {"clip_id": "8861297b65b9", "hand": "right", "hand_seed": {"frame": 60, "x": 1160, "y": 550}}
200  {"version": "2.3", "clip": "throw_71319.mov", "fps": 24, "frames": 193, "plate_size": [1920, 1080],
      "transform": {"scale": 1.0, "offset": [0, 0]}, "effect_range": {"start": 34, "end": 115},
      "canvas_size": [1024, 576], "canvas_rect": [0, 0, 1920, 1080], "hand": "right",
      "hand_seed": {"f": 60, "x": 1160.0, "y": 550.0}, "track": [... 193 × {f, x, y, source, palm_w, conf, lost}],
      "corrections": [], "correction_falloff": 12, "anchor_offset": [0, -70], "behaviour": "hold_throw", "appear_frame": 40,
      "release_frame": 108, "diameter_px": 110,
      "route": [{"f": 108, "x": null, "y": null, "diameter_px": 110, "easing_in": "linear"}],
      "vanish": {"start": 111, "end": 113}, "behind": [], "look": "energy_orb", "seed": 42,
      "grade": {"tint": "#9fd4ff", "brightness": 1.0, "glow": 0.6, "person_light": 0.4, "opacity": 1.0,
                "fade_in_frames": 3, "spill": 0.4}, "entrance": {"seconds": 0.5}}

POST /api/cue/normalize   {"cue": <that cue, appear_frame moved to 50>}
200  the cue with "effect_range": {"start": 43, "end": 116}
```
Errors: as analyze for `cue/new`; `400 input_invalid` when the cue does not fit its schema (`cue/normalize`).

### Looks (v2.4)
`GET /api/looks` → `[{id, name, file}]`, one entry per `<id>.png` in `CUEFX_LOOKS_DIR`, sorted by `id`; `id` is the file
stem (what `look` takes on `POST /api/manual/export`), `name` the stem with `_` as spaces, title case, `file` the file
name. An empty or missing folder is `[]`. A file whose stem is not a valid look id (letters, digits, `_`, `-`, `.`) is
not listed. An unknown `look` on the export is `400 input_invalid` and the message lists the valid ids. The stand-in
does not take a look from this folder (it draws its own orb, `CUEFX_LOOK_IMAGE`), so the cue's `look` is not checked there.

### Stand-in render (task 19)
`POST /api/standin` renders the cue with the **stand-in** effect (code-drawn ball, no LTX) and composites it onto the
untouched plate: Make (layout + stand-in effect) and Merge, the same functions the scripts run. Synchronous. It makes
only what the preview needs: **no layer PNG export and no element video** (task 24). `element_download` is built by
the first `GET …/element.mp4` (Merge again with the layer export, then the video; about as long as Make+Merge) and
kept; later GETs are plain file reads, and the layer PNGs are deleted afterwards. Timings for 71319 on the laptop are
in `timings_s` (and in the server log). The same cue again answers in milliseconds. Use a long client timeout for the
POST and for the first element GET. `track` (API form) replaces the cue's track when given;
the cue from `cue/new` already has it. `clip_id`: the clip to render on; without it, the stored clip whose file name is
the cue's `clip` (400 if two different clips have that name). **The UI should send `clip_id`.**

- The cue must be free of errors (`400 input_invalid` with the first problems; run validate).
- The clip must have the cue's `plate_size` and `frames` (`400 input_invalid`).
- Cache: Make by the normal Make key (`make_key`, as the CLI prints it), Merge by the whole cue. The same cue again
  answers in milliseconds with `cached: true`; a Merge-only change (vanish, behind, grade) reuses Make.
- Behind ranges need the hand landmarks: `analyze` / `cue/new` keep them in the analyze cache (same key: clip content,
  hand, seed, cuefx code version). A stand-in for a clip analyzed with the cue's `hand` and `hand_seed` never analyzes
  again; only a different seed (or hand, or a cuefx code change) does (about 12 s, reported as `timings_s.landmarks`).
- `timings_s` (optional): seconds per stage of the render that made the files: `landmarks` (behind ranges only),
  `make` (0 when an earlier Make was reused), `merge`, `total`; `element` is 0 until `element.mp4` has been requested,
  then it and `element_layers` are filled in (a `cached` answer repeats the original numbers).
- Result: `schemas/standin_result.schema.json`. `effect_range` is the window Make and Merge **used** (grown over the
  entrance, CONTRACT.md 7.6). `warnings` are plain-text notes from Make and Merge (degrade, don't fail).

```
POST /api/standin   {"cue": <the cue>, "clip_id": "8861297b65b9"}      (golden cue + track)
200  {"standin_id": "d61ef2ba70a3-fea2a9016b05", "make_key": "d61ef2ba70a3", "clip_id": "8861297b65b9",
      "frames": 193, "fps": 24, "plate_size": [1920, 1080], "effect_range": {"start": 26, "end": 115},
      "files": {"hardcomp_preview":  "/api/standin/d61ef2ba70a3-fea2a9016b05/hardcomp_preview.mp4",
                "hardcomp_download": "/api/standin/d61ef2ba70a3-fea2a9016b05/hardcomp.mp4",
                "element_download":  "/api/standin/d61ef2ba70a3-fea2a9016b05/element.mp4",
                "layout_download":   "/api/standin/d61ef2ba70a3-fea2a9016b05/layout.mp4"},
      "cached": false, "warnings": [], "timings_s": {"landmarks": 0, "make": 0, "merge": 0, "element": 0, "total": 0}}
```

| `name` | What | Disposition |
|---|---|---|
| `hardcomp_preview.mp4` | the hard comp (footage + stand-in): Merge's `final.mp4`, H.264 yuv420p, plate size, the plate's fps and frame count; range requests | inline (play it) |
| `hardcomp.mp4` | the same file, for ComfyUI's "Load hard comp" | `attachment; filename="<clip>_<make_key>_hardcomp.mp4"` |
| `element.mp4` | **built on the first request, then cached.** The stand-in alone on black at plate size, exactly as `scripts/make_element_video.py` writes it (H.264 4:4:4 CRF 0), for "Load element video" | `attachment; filename="<clip>_<make_key>_element.mp4"` |
| `layout.mp4` | the grey ball on black for Layout to Render, exactly as Make writes it (canvas size, the window's frames) | `attachment; filename="<clip>_<make_key>_layout.mp4"` |

Errors: `400 input_invalid` (cue errors, wrong clip), `404 not_found` (unknown clip, unknown `standin_id` or
`name`), `500 render_failed` with the stage (`Building reference frame` = Make, `Blending into your footage` = Merge).
Files on disk: `<CUEFX_DATA_DIR>/standin/<make_key>/` (Make) and `…/<make_key>/merge/<merge id>/` (Merge, with
`compare.mp4`, `cue.json`, `standin.json`, and `element.mp4` once requested).

### Manual render (task 21, CONTRACT.md section 12)
The export runs Make 1-2 for the cue with `look` (the look image `CUEFX_LOOKS_DIR/<look>.png`; the job's look is the
cue's `look` from then on), creates a job with `status: "waiting"`, `manual: true`, and answers with the package zip
(`attachment; filename="<clip>_<job_id>_comfyui.zip"`). The job id is in the `X-CueFX-Job-Id` header (exposed to
CORS) and in the zip's `manifest.json`; `GET /api/manual/{job_id}/package.zip` downloads the same zip again.
`prompt` defaults to the cue's `prompt` (empty: a warning, and HOW_TO_RUN says to type one). Make runs on the render
queue, about as long as the stand-in's Make. By default (`?wait=0`) the answer is **202** and the `running` job (stage 0
"Preparing shot"); when the zip is ready the job goes to `waiting` exactly as before. `?wait=1` answers with the zip.
The package's graph has its padded-frame limit set to `max(121, window length)` (v2.5; `manifest.json` `graph_frames`),
so windows of up to 193 frames (8 s) fit; CONTRACT.md section 12.
If Make fails the job stays, `failed` (`render_failed`), and takes no upload (`409`: it has no package); export again.

The result upload is checked against the manifest (size, frame count, fps) and the make key (the `cue` form field,
else the exported cue; and the look file), then ingest + Merge run on the render queue; the answer is the `running` job (202, default) or, with `?wait=1`, the
`done` job (use a long client timeout). Merge uses the `cue` sent, so Merge-only edits after export are applied.

```
POST /api/manual/export   {"cue": <cue>, "clip_id": "8861297b65b9", "look": "energy_orb", "prompt": "..."}
200  application/zip                          X-CueFX-Job-Id: job-3f9c0a1b2d
GET  /api/jobs/job-3f9c0a1b2d
200  {"id": "job-3f9c0a1b2d", "look": "energy_orb", "status": "waiting", "stage": 2,
      "stage_name": "Rendering (pass 1 of 2)", "progress": 0.4, "elapsed_s": 21.4, "eta_s": null, "error": null,
      "manual": true, "clip_id": "8861297b65b9", "make_key": "5b1e0c7d9a24",
      "files": {"package_download": "/api/manual/job-3f9c0a1b2d/package.zip"}, "warnings": []}
POST /api/manual/job-3f9c0a1b2d/result   multipart effect_on_black=@effect.mp4, cue=<cue as JSON text>
200  {... "status": "done", "stage": 5, "stage_name": "Done", "progress": 1.0,
      "files": {"package_download": "...", "final_preview": "/api/manual/job-3f9c0a1b2d/final_preview.mp4",
                "final_download": "/api/manual/job-3f9c0a1b2d/final.mp4"},
      "warnings": ["temporary luminance matte — replace with Alpha Gen"]}
```

`GET /api/jobs/{job_id}/result` (v2.4) is the `result` of a `done` job (`schemas/result.schema.json`, CONTRACT.md
section 10): the videos and layers are download URLs under `/api/jobs/{job_id}/`; `409 input_invalid` until the job is
done, `404 not_found` for an unknown job. It reads files and numbers the steps already made and measures nothing: a
check nobody computes yet is `warn`, `value: null`, "not measured yet".

Errors: `400 input_invalid` (cue errors, wrong clip, missing look image, upload of the wrong size / frame count /
fps, "the cue changed since export", not a video), `404 not_found` (unknown job or file, or `final.mp4` before
the job is done), `409 input_invalid` (an upload while the job is still blending), `413` (upload larger than `CUEFX_MAX_UPLOAD_MB`), `500 render_failed` (Make: "Building reference
frame"; ingest/Merge: "Blending into your footage"; the job is then `failed` and takes a new upload).
Files on disk: `<CUEFX_DATA_DIR>/manual/<make_key>/` (Make 1-2 and its debug sheets) and
`<CUEFX_DATA_DIR>/jobs/<job_id>/` (`job.json`, `export.json`, `package.zip`, `upload/`, `ltx/`, `merge/`). Jobs
are read from disk, so they survive a restart; a job that was `running` when the server stopped comes back `failed`.

## Errors

Every failure, on every route (also unknown routes, wrong methods and unreadable bodies), is the contract shape
`{"code": ..., "message": ...}` with a code from `schemas/error_codes.json` and the plain-language message:

```
404  {"code": "not_found", "message": "CueFX cannot find what this request asks for: unknown clip '000000000000'"}
```

| Status | When | `code` |
|---|---|---|
| 400 | bad body or field (also a seed outside the clip) | `input_invalid` |
| 400 | upload is not a supported extension, is empty | `input_invalid` |
| 400 | file cannot be read as video; analyze cannot open it | `clip_unreadable` |
| 400 | more than 1440 frames (upload or analyze) | `clip_too_long` |
| 400 | analyze: no hand in the seed frame / more than half the frames lost | `no_hand` / `track_lost` |
| 404 | unknown `clip_id`, stand-in, job or file, or an unknown route | `not_found` (v2.3; was `input_invalid`) |
| 405 | wrong method | `input_invalid` |
| 409 | a manual result upload while that job is still blending | `input_invalid` |
| 413 | upload larger than `CUEFX_MAX_UPLOAD_MB` | `input_invalid` |
| 500 | a render step failed (the stage is in the message) | `render_failed` |
| 500 | anything unexpected; the message is fixed and safe, the traceback is only in the server log | `internal` (v2.3; was `render_failed`) |

v2.3 adds `not_found` and `internal` (both `call: "any"`). "Unsupported format" and "too large" still use
`input_invalid` (the reason is in the message); their codes are reserved in `CONTRACT.md` section 5. A manual job
that fails unexpectedly still records `render_failed` in `job.error` (the job schema's codes are unchanged).

## Environment variables

| Variable | Default | |
|---|---|---|
| `CUEFX_DATA_DIR` | `data` | uploads in `uploads/`, proxies, analyze results and landmarks in `cache/`, stand-in renders in `standin/`; relative to the working directory |
| `CUEFX_HOST` | `127.0.0.1` | |
| `CUEFX_PORT` | `8000` | |
| `CUEFX_CORS_ORIGINS` | `http://localhost:5173` | comma-separated list of allowed origins |
| `CUEFX_MAX_UPLOAD_MB` | `500` | |
| `CUEFX_LOOK_IMAGE` | `<CUEFX_DATA_DIR>/look/standin_orb.png` | the stand-in look (part of the make key); drawn with the default look if missing |
| `CUEFX_LOOKS_DIR` | `<CUEFX_DATA_DIR>/looks` | the manual render's look images, `<look id>.png`, the look on pure black (part of the make key); never drawn: a missing one is `input_invalid` |

Interactive docs of the same routes: `/api/docs`.
