# CLAUDE.md — PicturaAI

## What the app does
PicturaAI ("pictura" = Latin for "a painting") is a single-page neural style transfer
studio. A user uploads a photo, picks one of 13 built-in art styles (or uploads a
painting, or types a text description that is matched to the closest preset), and
gets a stylised JPEG back in a few seconds. Extras: blending two styles, painting a
regional mask, palette-only colour transfer, a 0→100% style animation GIF,
a before/after slider, and a session-only history strip of the last 10 results.

There are no user accounts, no database and no payments. All state is in-memory on
the server (job metadata) or in the browser tab (history).

## Stack
- **Backend:** Python 3.10, FastAPI, Uvicorn, slowapi (rate limits)
- **ML:** TensorFlow 2.20 + TF Hub — Google Magenta
  `arbitrary-image-stylization-v1-256/2` (single forward pass, no training)
- **Image pipeline:** Pillow, NumPy
- **Frontend:** plain HTML/CSS/JS, no build step (`frontend/`)
- **Hosting:** Docker on Render free tier (512 MB RAM, sleeps after idle)

## Architecture
`backend/main.py` is the FastAPI app. It serves the frontend as static files at
`/app` (`/` redirects there) and exposes a JSON API under `/api`. Uploads are read
with a size cap and verified with Pillow (`read_image_upload`). Heavy work runs on a
single-thread `ThreadPoolExecutor`, so only one TF job runs at a time; a counter
(`MAX_PENDING_JOBS`) returns 429 when too many requests are queued, and slowapi adds
per-IP rate limits.

`/api/transfer` and `/api/palette-transfer` are **synchronous**: the request waits
for the result and the response carries the image as base64. That was a deliberate
change because Render's free tier OOM-kills the process, which wiped the in-memory
job store mid-job. `/api/interpolate` (GIF) is the only async job: it returns a
`job_id`, and the client polls `/api/jobs/{id}` (a WebSocket at `/ws/{id}` also
exists). Finished jobs and files in `backend/outputs/` are cleaned up after
`JOB_TTL_SECONDS`.

`backend/nst_engine.py` holds the model singleton (`StyleTransferModel`) and the
pipelines: `run_nst` (preprocess → Magenta forward pass → luminance-preserving
blend → high-frequency detail re-injection → optional mask → unsharp mask → JPEG),
`run_interpolation_gif`, and `color_palette_transfer` (LAB mean/std matching, no
neural net). Resolution is capped by `NST_MAX_DIM` because of the 512 MB limit.

`frontend/assets/app.js` is a single script that uses global state and DOM ids.
`frontend/index.html` uses inline `onclick` handlers. Style thumbnails arrive as
base64 inside the `/api/styles` JSON, which is also Render's health check.

## Run locally
```bash
python -m venv venv
venv\Scripts\pip install -r backend\requirements.txt
python run.py                      # uvicorn with reload on :8000
# or: cd backend && ..\venv\Scripts\python -m uvicorn main:app --port 8000
```
Open http://localhost:8000/app (API docs at /docs). The first run downloads the TF
Hub model (~90 MB) into `.tfhub_cache/` (gitignored).

## Environment variables (all optional, defaults in code)
| Name | Purpose |
|---|---|
| `PORT` | Port when started via `python backend/main.py` (Render sets it) |
| `NST_MAX_DIM` | Max side (px) of the content image fed to the model; memory-critical (320 on Render) |
| `NST_OUTPUT_DIM` | Max side of the composited output; keep equal to `NST_MAX_DIM` on 512 MB |
| `NST_PALETTE_MAX_DIM` | Working resolution cap for palette transfer (1024) |
| `MAX_UPLOAD_MB` | Per-file upload cap (10) |
| `MAX_IMAGE_PIXELS` | Pillow decompression-bomb limit (64,000,000) |
| `MAX_PENDING_JOBS` | Queue size before 429 (3) |
| `JOB_TTL_SECONDS` | How long finished jobs/outputs are kept (3600) |
| `RATE_LIMIT_TRANSFER` / `RATE_LIMIT_INTERPOLATE` / `RATE_LIMIT_PALETTE` | slowapi limit strings |
| `CLIENT_IP_HEADER` | Header holding the real visitor IP for rate limiting (`cf-connecting-ip` on Render; leave empty elsewhere) |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (default `*`; no credentials) |
| `TFHUB_CACHE_DIR` | Model cache location |
| `TF_CPP_MIN_LOG_LEVEL`, `TF_ENABLE_ONEDNN_OPTS`, `TF_NUM_INTRAOP_THREADS`, `TF_NUM_INTEROP_THREADS`, `OMP_NUM_THREADS`, `MALLOC_ARENA_MAX` | TF logging and memory tuning for the free tier |

No secrets are needed to run the app.

## Useful commands
```bash
venv\Scripts\python -m pytest backend/tests          # backend tests
venv\Scripts\python -m pip_audit -r backend/requirements.txt   # dependency CVEs (if installed)
node docs/revamp/screenshots.mjs before|after        # scripted Playwright screenshots
docker build -t picturaai . && docker run -p 8000:8000 picturaai
```

## Conventions
- Backend: type-hinted functions, section banners (`# ── Name ──`), env-driven limits
  with safe defaults, and generic error messages to clients (details go to the log).
- Never put TF work on the event loop; use `run_in_executor(executor, …)`.
- Every memory-relevant change must respect the 512 MB Render limit.
- Frontend: no framework and no build step. Insert user or server text with
  `textContent`, never `innerHTML`.
- Small commits, one concern each.

## Deployment
Render reads `render.yaml` (Docker runtime, free plan, health check `/api/styles`)
and builds the `Dockerfile`, which pre-downloads the model at build time and runs
`python backend/main.py` as a non-root user. Pushing to `main` auto-deploys.
Live: https://picturaai-k7q6.onrender.com/app. There is no Vercel deployment.
