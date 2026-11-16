# CLAUDE.md — PicturaAI

## What the app does
PicturaAI ("pictura" = Latin for "a painting") is a single-page neural style transfer
studio. A user uploads a photo, picks one of the built-in public-domain paintings
listed in `data/styles.json` (or uploads a painting, or describes a style in words:
generated with Cloudflare Workers AI when configured, otherwise matched to the
closest preset), and the photo is repainted in a few seconds. By default the model
runs **on the device** (TF.js in a Web Worker, when WebGL2 is available); otherwise,
or after an engine failure, the browser asks the server for the raw stylisation.
Both paths finish in the same browser compositor. Extras: a live intensity slider (no
new request), mixing two styles, painting a regional mask or one-tap "Style the
subject/background" (on-device segmentation), "Colours only" (palette transfer, done
in the browser), a before/after handle, downloads (full size, Instagram, A4 print),
Share, "Animate" (a 0→100% reveal video recorded in the browser with MediaRecorder,
MP4 or WebM), and "My gallery" (saved paintings in IndexedDB, this device only).

There are no user accounts, no database and no payments. Server state is in memory
(job metadata); browser state is IndexedDB (gallery) and `localStorage` (engine hints).

## Stack
- **Backend:** Python 3.12, FastAPI, Uvicorn, slowapi (rate limits); optional
  `sentry-sdk` (only imported when `SENTRY_DSN` is set)
- **ML (server):** TensorFlow 2.20 — Google Magenta `arbitrary-image-stylization-v1-256/2`,
  loaded as a plain SavedModel by `backend/model_store.py` (single forward pass,
  no training)
- **ML (browser):** the same model converted to TF.js (uint8 weights) in
  `frontend/models/`; TensorFlow.js 4.22.0 self-hosted in `frontend/vendor/tfjs/4.22.0/`;
  MediaPipe Tasks Vision 0.10.35 (DeepLab-v3 segmentation) loaded from jsDelivr
- **Image pipeline:** Pillow, NumPy
- **Frontend:** plain HTML/CSS/JS (native ES modules), no build step (`frontend/`)
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
`JOB_TTL_SECONDS`. The new UI calls only `/api/styles`, `/api/describe-style` and
`/api/transfer` (`output=raw`, server engine); `/api/palette-transfer`,
`/api/interpolate` and the job/result routes stay for API users.

**Static caching** (`static_cache_control` in `main.py`): everything under `/app` is
`no-cache` (revalidate with ETag, usually a 304) because HTML, ES modules, CSS, icons
and `config.js` keep their names across deploys; `/app/models/**` is
`public, max-age=31536000, immutable` (the engine requests model files with
`?v=<manifest version>`) except `manifest.json` and `style-vectors.json`, which are
`no-cache`; `/app/vendor/**` is immutable (versioned folder); `/app/assets/styles/**`
and `/app/assets/samples/**` get one day. A changed model file needs a new
`version` in `frontend/models/arbitrary-stylization-v1-256/manifest.json`; a TF.js
upgrade needs a new version folder. `GZipMiddleware` compresses text responses of
1 KB or more (HTML, JS, CSS, JSON such as the ~1 MB `model.json` files), not images
or `application/octet-stream` weights.

Style presets live in `data/styles.json` (schema: `docs/revamp/BUILD-PLAN.md` §4).
`backend/style_catalog.py` loads and validates it at import time (path from
`STYLES_FILE`); a missing or invalid file stops the app with
`PicturaAI cannot start: …`. Each entry's `file` must exist in
`images/style_image/`. The same module holds the keyword matcher (tags, then
title/artist/description) used by `text_prompt` and `/api/describe-style`.

`backend/site_config.py` builds everything derived from public env settings: the
browser config at `/app/config.js`, the Content-Security-Policy (third parties are
allowed only when configured), `robots.txt` and `sitemap.xml`.
`backend/describe_style.py` is the Workers AI client (urllib, no SDK) with the
in-memory daily cap.

`backend/nst_engine.py` holds the model singleton (`StyleTransferModel`) and the
pipelines: `run_nst` (preprocess → Magenta forward pass → luminance-preserving
blend → high-frequency detail re-injection → optional mask → unsharp mask → JPEG),
`run_interpolation_gif`, and `color_palette_transfer` (LAB mean/std matching, no
neural net). `run_nst(..., raw=True)` returns only the stylized image at output
size (no blend/detail/mask/sharpen) so a browser can composite intensity live;
the default path must stay byte-identical. Resolution is capped by `NST_MAX_DIM` because of the 512 MB limit.

**Frontend** ("Salon Wall", see `docs/revamp/BUILD-PLAN.md`): `frontend/index.html`
loads `config.js` and `assets/js/main.js`; code is split into `core/`, `services/`
(API, engine adapter, export, `gallery-db.js`, `video.js`) and `ui/`. Static pages use
`assets/js/page.js`. `manifest.webmanifest` makes the site installable (gilt-frame
icons in `assets/`). CSP-ready: no inline scripts, `on*` handlers or `style=""`
attributes (set CSS custom properties from JS instead).

**Engine** (`frontend/assets/js/engine/`, details and measurements in
`docs/revamp/engine/ENGINE.md`): the UI reaches it only through
`services/engine-adapter.js`. `index.js` `createEngine()` picks the browser engine
when WebGL2 works (and it has not failed here in 7 days), else `server-engine.js`
(`POST /api/transfer` with `output=raw`). The browser engine runs in `worker.js`
(module worker, TF.js on an `OffscreenCanvas`) behind `worker-engine.js`
(`engine.offMainThread`); it falls back to `browser-engine.js` on the page when the
worker cannot run. `composite.js` is the only compositor (a port of `run_nst`'s
post-processing and palette transfer; parity-tested). `segment.js` runs MediaPipe on
the CPU delegate. URL switches for testing: `?engine=browser|server`, `?worker=0`.

**Generated files** (rerun the script and commit its output when the input changes):
- `scripts/build_style_assets.py` (`--check`, `--force`): after editing
  `data/styles.json` or a file in `images/style_image/`; writes
  `frontend/assets/styles/<key>-{512.jpg,480.webp,160.webp}`.
- `scripts/precompute_style_vectors.py`: after the presets or the model change; writes
  `frontend/models/style-vectors/<key>.bin` and `style-vectors.json` (needs TF).
- `scripts/build_pages.py` (`--check`): after changing `data/styles.json`,
  `scripts/style_notes.json`, `frontend/assets/samples/credits.json` or the script's
  page template (head, header, footer); writes
  `frontend/styles/**`, `credits.html`, `404.html`, and the head/header/footer of
  `privacy.html` and `terms.html`. Their text between the `page-content:start` /
  `page-content:end` markers is hand-written and kept as is.
- `.gitattributes` marks `frontend/vendor/**` and `frontend/models/**` as `-text`:
  the TF.js files are pinned by SRI hashes and model files must stay byte-exact, so
  Git must not convert line endings on Windows.

## API
| Method & path | What it does |
|---|---|
| `POST /api/transfer` | Stylize (sync, base64 JPEG). Form: `content_image`, one of `style_image` / `style_preset` / `text_prompt`, `intensity` 0–1 (alias of legacy `style_weight`; wins if both are sent), `output` = `final` (default) or `raw` (stylized image only; mask ignored), optional `style_image_2` / `style_preset_2` + `style_mix_ratio`, `mask_image` (white = styled). `content_weight`, `tv_weight`, `num_steps`, `learning_rate` are accepted and ignored. |
| `POST /api/palette-transfer` | Colour-only transfer (sync; the UI does this in the browser) |
| `POST /api/interpolate`, then `GET /api/jobs/{id}` or `WS /ws/{id}` | 0→100% style GIF (async job; not used by the UI, which records a video) |
| `POST /api/describe-style` | `prompt` (form field or JSON, 1–200 chars after control characters are stripped) → `{source:"generated", image, media_type, prompt}` or `{source:"matched", key, title, artist, message, reason}`; 404 when nothing matches or when `DESCRIBE_ENABLED=false` |
| `GET /api/styles` | Presets: `key, slug, title, name` (legacy), `artist, year, medium, description, image, card, thumb` (`/app/…` URLs), `page, aspect, credit, source_url, license`. `Cache-Control: public, max-age=3600`. `?include=thumbnail` adds the old base64 `thumbnail`. |
| `GET` / `HEAD /api/health` | `{"status":"ok"}` (Render health check) |
| `GET /app/config.js` | `window.PICTURA_CONFIG = {siteUrl, sentryDsn, cfWebAnalyticsToken, describeEnabled, describeGenerates}` (public values only) |
| `GET /robots.txt`, `GET /sitemap.xml` | Built from `PUBLIC_SITE_URL`. The sitemap lists `/app/`, every `/app/styles/<slug>/`, privacy/terms/credits, and `/app/styles/` once `frontend/styles/index.html` exists. |
| `GET /api/result/{id}`, `DELETE /api/jobs/{id}` | Download or remove a result |

Every response carries `X-Content-Type-Options`, `X-Frame-Options: DENY`,
`Referrer-Policy`, `Permissions-Policy: camera=(), microphone=(), geolocation=()`
and a CSP (`script-src 'self' https://cdn.jsdelivr.net 'wasm-unsafe-eval'`, no
`unsafe-inline` or `unsafe-eval`; `/docs` and `/redoc` get a relaxed policy). HSTS
is added only when the request is HTTPS (`X-Forwarded-Proto: https`). jsDelivr is
needed only for MediaPipe (`segment.js`: ES module and wasm). TF.js is self-hosted as
the `es2017` builds (`tf-core`, `tf-backend-webgl`, `tf-converter`), loaded with SRI on
the page and by `import()` in the worker; never use `dist/tf.min.js`, which calls
`Function()` at load time (regenerator-runtime) and this CSP blocks it. To upgrade,
add a new `frontend/vendor/tfjs/<version>/` folder and update `TF_VERSION` and the
hashes in `browser-engine.js` (see that folder's `README.md`).

## Run locally
```bash
python -m venv venv
venv\Scripts\pip install -r backend\requirements.txt
python run.py                      # uvicorn with reload on :8000
# or: cd backend && ..\venv\Scripts\python -m uvicorn main:app --port 8000
```
Open http://localhost:8000/app (API docs at /docs). The first run downloads the
model (~90 MB) into `.tfhub_cache/` (gitignored).

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
| `RATE_LIMIT_TRANSFER` / `RATE_LIMIT_INTERPOLATE` / `RATE_LIMIT_PALETTE` / `RATE_LIMIT_DESCRIBE` | slowapi limit strings (describe default `5/minute`) |
| `CLIENT_IP_HEADER` | Header holding the real visitor IP for rate limiting (`cf-connecting-ip` on Render; leave empty elsewhere) |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (default `*`; no credentials) |
| `TFHUB_CACHE_DIR` | Model cache location (`NST_MODEL_DIR` overrides the exact model directory) |
| `STYLES_FILE` | Style catalogue JSON (default `<repo>/data/styles.json`); the app refuses to start if it is missing or invalid |
| `PUBLIC_SITE_URL` | Absolute site URL for `sitemap.xml`, `robots.txt` and `config.js` (default `https://picturaai-k7q6.onrender.com`) |
| `ENABLE_API_DOCS` | Serve `/docs`, `/redoc` and `/openapi.json` (default `true`) |
| `DESCRIBE_ENABLED` | Turn "Describe it" on or off entirely (default `true`) |
| `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` | **Secret.** Enable Workers AI generation (`@cf/black-forest-labs/flux-1-schnell`) in `/api/describe-style`; unset means keyword matching only. The token needs only the Workers AI permission and is never logged. |
| `DESCRIBE_DAILY_LIMIT` | Generations per UTC day, counted in memory (default 50). A provider quota error also stops calls until the next day. |
| `DESCRIBE_TIMEOUT_SECONDS` | Workers AI request timeout (default 30) |
| `SENTRY_DSN` | **Secret.** Backend error reporting (no PII, request bodies or local variables are sent). `SENTRY_ENVIRONMENT` (default `production`), `SENTRY_TRACES_SAMPLE_RATE` (default 0). |
| `SENTRY_DSN_FRONTEND` | Public DSN for the browser; also opens the CSP to Sentry's CDN and ingest hosts |
| `CF_WEB_ANALYTICS_TOKEN` | Cloudflare Web Analytics site token; also opens the CSP to Cloudflare Insights |
| `CSP_REPORT_ONLY` | Send the CSP as `Content-Security-Policy-Report-Only` (default `false`) for a cautious rollout |
| `TF_CPP_MIN_LOG_LEVEL`, `TF_ENABLE_ONEDNN_OPTS`, `TF_NUM_INTRAOP_THREADS`, `TF_NUM_INTEROP_THREADS`, `OMP_NUM_THREADS`, `MALLOC_ARENA_MAX` | TF logging and memory tuning for the free tier |

No secrets are needed to run the app. The optional ones (Cloudflare token, Sentry
DSN) go in the host's dashboard, never in the repo.

## Useful commands
```bash
venv\Scripts\pip install -r requirements-dev.txt               # pytest, httpx, Playwright, ruff, pip-audit
venv\Scripts\python -m playwright install chromium             # once, for tests/e2e
venv\Scripts\python -m pytest backend/tests                    # backend tests
venv\Scripts\python -m pytest tests/e2e                        # browser e2e (own server on :8230, ?engine=server, ~80 s)
venv\Scripts\python -m pytest tests/e2e -m slow                # on-device engine smoke test (minutes)
venv\Scripts\ruff check --select F,E9 backend tests/e2e        # lint (errors only)
venv\Scripts\pip-audit -r backend/requirements.txt             # dependency CVEs
venv\Scripts\python docs/revamp/engine/parity/composite_parity.py   # composite.js vs run_nst (Node 18+)
venv\Scripts\python scripts/build_style_assets.py --check      # generated style images current
venv\Scripts\python scripts/build_pages.py --check             # generated static pages current
docker build -t picturaai . && docker run -p 8000:8000 --memory 512m picturaai
```
Details, timings and how to write e2e tests: `docs/revamp/TESTING.md`. CI
(`.github/workflows/ci.yml`) runs all of the above except the slow test on pushes to
`main` and `revamp/**` and on pull requests.

## Conventions
- Backend: type-hinted functions, section banners (`# ── Name ──`), env-driven limits
  with safe defaults, and generic error messages to clients (details go to the log).
- Never put TF work on the event loop; use `run_in_executor(executor, …)`.
- Every memory-relevant change must respect the 512 MB Render limit.
- Frontend: no framework and no build step. Insert user or server text with
  `textContent`, never `innerHTML`. Respect `prefers-reduced-motion`.
- Small commits, one concern each.

## Deployment
Render reads `render.yaml` (Docker runtime, free plan, health check `/api/health`;
optional secrets are declared with `sync: false`) and builds the `Dockerfile`, which pre-downloads the model at build time and runs
`python backend/main.py` as a non-root user. The image must contain `data/`
(`data/styles.json`) as well as `images/style_image/`, or the app refuses to start. Pushing to `main` auto-deploys.
Live: https://picturaai-k7q6.onrender.com/app. There is no Vercel deployment.
