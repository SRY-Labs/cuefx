# CueFX backend

Python FastAPI server for CueFX. `app.py` holds the HTTP routes; the work lives in `services/`.

```
app.py                 FastAPI routes (`create_app()`, `main()`)
services/
  config.py            settings from CUEFX_* environment variables
  errors.py            contract error shape
  analysis/            clip store, track, pose, conform, masks
  cue/                 cue contract, route, window, timings, public api
  make/                generate, ComfyUI worker, layout, look, pipeline, LTX
  delivery/            stand-in and manual stores, merge, video, debug
contract/              schemas, golden files, HTTP.md
comfyui/               ComfyUI graph JSON files the services fill in
tests/
CueFX-frontend/        Next.js app; its own Dockerfile, package.json, and src/app
```

## Run

Needs Python 3.11+, `ffmpeg` and `ffprobe` on `PATH`.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python app.py
```

Or: `uvicorn app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1`

Then `GET http://127.0.0.1:8000/api/health` and `GET /api/docs`. Routes, bodies and errors are in `contract/HTTP.md`.

## Settings

| Variable | Default | |
|---|---|---|
| `CUEFX_DATA_DIR` | `data` | uploads, cache, stand-in, manual and jobs. In Azure: `/data` on the container disk (ephemeral; one replica) |
| `CUEFX_HOST` | `127.0.0.1` | used by `python app.py` |
| `CUEFX_PORT` | `8000` | used by `python app.py` |
| `CUEFX_CORS_ORIGINS` | `http://localhost:5173` | comma-separated |
| `CUEFX_MAX_UPLOAD_MB` | `500` | |
| `CUEFX_LOOK_IMAGE` | `<data>/look/standin_orb.png` | stand-in look image |
| `CUEFX_LOOKS_DIR` | `<data>/looks` | look images for manual render, `<look id>.png` |
| `CUEFX_LTX_OVERLAP_ENABLED` | `0` | `1` enables the experimental generate route |
| `CUEFX_COMFY_URL` | unset | private ComfyUI worker; required when generate is enabled |
| `CUEFX_COMFY_TOKEN` | unset | optional bearer token; never commit it |
| `CUEFX_COMFY_TIMEOUT_S` | `7200` | per-workflow completion timeout |

## Frontend

The UI stays in `CueFX-frontend/` and deploys on its own. From that directory:

```
npm ci
npm run dev
```

Copy `CueFX-frontend/.env.example` to `.env.local` and set `CUEFX_BACKEND_URL` when the app should call this API. Next.js routes live in `CueFX-frontend/src/app`.

## Tests

```
pytest -q
pytest -q -m slow tests -s
```
