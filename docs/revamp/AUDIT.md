# PicturaAI — Audit (revamp/2026-11-16)

Scope: local run on Windows 11 (Python 3.10.11, venv with `backend/requirements.txt`),
Chromium via Playwright MCP, read-only browsing of https://picturaai-k7q6.onrender.com/app.
Production serves exactly the code on `main` (live `app.js` is byte-identical to `HEAD`).

Status legend: **OPEN** = not yet fixed · **FIXED ✔** = fixed and verified by running it ·
**FIXED (unverified)** = fixed but can't be verified locally · **DECISION** = waiting on you.

## Summary

| # | Severity | Issue | Status |
|---|---|---|---|
| C1 | Critical | Render API token in public git history | FIXED ✔ (owner revoked the key; history purged and force-pushed, `main` → `3d40053`) |
| H1 | High | Large photos decoded at full resolution → OOM on 512 MB | FIXED ✔ (on Render: unverified) |
| H2 | High | EXIF orientation ignored → phone photos come out sideways | FIXED ✔ |
| H3 | High | Toggling mask mode wipes the mask → silently unstyled output | FIXED ✔ |
| H4 | High | Mobile (≤ 900 px) studio overflows horizontally after a result | FIXED ✔ |
| H5 | High | Upload zones and style cards unreachable by keyboard | FIXED ✔ |
| H6 | High | *(found in Phase 4)* Every paint stroke opened the file picker | FIXED ✔ |
| M1 | Medium | Rate limits bypassable with a spoofed `X-Forwarded-For` | FIXED (unverified on Render) |
| M2 | Medium | Non-JSON errors (Render 502/503) shown as "Unexpected token '<'…" | FIXED ✔ |
| M3 | Medium | Download of history/palette results saves a JSON 404 as `.jpg` | FIXED ✔ |
| M4 | Medium | Animation fails after a text-prompt-only result | FIXED ✔ |
| M5 | Medium | Known CVEs in Pillow, python-multipart, keras | FIXED ✔ (Python 3.12, Pillow 13, keras 3.15.1; pip-audit clean) |
| M6 | Medium | Style list never retries after a cold-start failure | FIXED ✔ |
| M7 | Medium | Low-contrast text (`--text-3`) fails WCAG AA | FIXED ✔ (one more case left for the redesign, see notes) |
| M8 | Medium | Unlabelled sliders and icon-only buttons | FIXED ✔ |
| M9 | Medium | *(found in Phase 4)* "Try Again" wiped the user's photo and settings | FIXED ✔ |
| M10 | Medium | *(found in Phase 5)* Painting a mask on a touch screen scrolls the page (no `touch-action`, passive listeners) | FIXED (unverified on a real touch device) |
| L10 | Low | *(found in Phase 5)* `prefers-reduced-motion` ignored (spinning star, floating orbs, shimmer loops) | FIXED ✔ |
| L1 | Low | Animation GIF base64 kept in the in-memory job store | FIXED ✔ |
| L2 | Low | 174 KB logo PNG rendered at 32 px (twice) | FIXED ✔ |
| L3 | Low | `/favicon.ico` 404 at the site root | FIXED ✔ |
| L4 | Low | No tests, lint config or CI | Tests added ✔; CI in the proposal |
| L5 | Low | Dead code: hidden hero visual with a broken SVG, unused params | FIXED ✔ (old frontend replaced in Phase 6; legacy params documented as ignored) |
| L6 | Low | No CSP / HSTS headers; `/docs` public; CORS `*` | FIXED ✔ (CSP, HTTPS-only HSTS, Permissions-Policy, `ENABLE_API_DOCS`; CORS still `*` by default, set `ALLOWED_ORIGINS` to restrict) |
| L7 | Low | Style thumbnails embedded as base64 in `/api/styles` (154 KB, uncacheable) | FIXED ✔ (static WebP files; response 10 KB, cached 1 h) |
| L8 | Low | Cold start of ~56 s on the live site with no in-app explanation | Partly FIXED ✔ (messaging); real fix in the proposal |
| L9 | Low | *(found in Phase 4)* HTML cached heuristically, so deploys showed stale pages | FIXED ✔ |

### Phase 4 verification evidence

| # | How it was verified | Commit |
|---|---|---|
| H1 | Peak memory to decode a 49 MP JPEG twice through `load_and_preprocess`: **383 MB → 14 MB** (psutil). A 49 MP upload through `/api/transfer` now returns 200 in 6.9 s. pytest checks the reduced decode. Not run on a 512 MB Render instance. | `d4fe181` |
| H2 | An EXIF Orientation=6 upload now returns 160×320 (portrait), and palette returns 200×400. pytest covers both. | `d4fe181` |
| H3 / H6 | Playwright: paint a stroke (8,335 px), toggle Paint Mask off and on, and 8,335 px remain. Painting fires 0 file-picker opens (the old handler fired 1 per stroke). | `a3b804a` |
| H4 | Playwright: page scroll width minus viewport in the result state is 0 at 360, 390, 768 and 1440 px (was +99 / +69 at 360 / 390). Screenshot `after/phase4-mobile-08-studio-result.png`. | `da75e36` |
| H5 / M8 | Playwright: Tab reaches the upload zone and each preset; Enter opens the picker, and Space selects a preset (`aria-pressed=true`). Unlabelled controls: 0. Screenshot `after/phase4-keyboard-focus.png`. | `a3b804a` |
| M1 | Real uvicorn (`python backend/main.py`, `forwarded_allow_ips="*"`), limit 2/min, 5 requests with rotating spoofed XFF. Old behaviour: `200 200 200 200 200`. Fixed: `200 200 429 429 429`. pytest covers it. On Render this relies on Cloudflare sending `CF-Connecting-IP` and on the `CLIENT_IP_HEADER` env var (added to `render.yaml`). | `9d2207a` |
| M2 | Playwright with mocked responses: 502 HTML → "The server is waking up or restarting…"; network failure → "Could not reach the server…"; 429 → "The server is busy…"; a 400 still shows the server's own message. Screenshot `after/phase4-mobile-09-studio-error.png`. | `a3b804a` |
| M3 | Playwright: palette result → history item → Download gives `pictura_<timestamp>.jpg` starting with JPEG magic `ffd8ff`. | `a3b804a` |
| M4 | Playwright: text prompt "watercolor portrait" → Generate → Animation downloads a real `GIF89a` file. | `a3b804a` |
| M5 | `pillow 12.1.1 → 12.3.0`, `python-multipart 0.0.22 → 0.0.32`. pip-audit now flags only fixes that need Python ≥ 3.11 (see decisions). Full API smoke test re-run. | `1b8d0a7` |
| M6 | Playwright: `/api/styles` fails twice with an HTML 503. The panel shows "Waking up the server…" and then 13 presets load. | `a3b804a` |
| M7 | New `--text-3: #808ba0` measures 4.68–5.73:1 on all four theme backgrounds. Still failing: `--accent` text (#7c3aed) used on section tags is 3.1–3.5:1; I left it for the redesign rather than change the brand colour now. | `eb48eaf` |
| M9 | Playwright: first request 503 → Try Again → second request succeeds, and the photo and preset are kept. | `4e4b5b1` |
| L1 | pytest: job results are read from disk, and `result_path` (a server filesystem path) is no longer returned to clients. A live interpolation job returns a `GIF89a`. | `78a8c5a` |
| L2 | `logo-96.png` is 12 KB (the original is 174 KB). | `49dc027` |
| L3 | pytest: `/favicon.ico` → 200 `image/x-icon`. | `a995d47` |
| M10 / L10 | Playwright: the active mask canvas computes `touch-action: none`. With `reducedMotion: 'reduce'` emulated, looping animations run once (and still loop normally without it). I couldn't test on a real touch device. | `af0a6ec` |
| L9 | pytest: `/app/` sends `Cache-Control: no-cache`, and assets don't. I hit this live in Chromium, which kept serving the old `?v=5` HTML. | `bf8c8f2` |

### Waiting on your decision

1. **C1: revoke the Render API key** (Dashboard → Account Settings → API Keys). Then decide
   whether to purge it from history (`git filter-repo` plus a force-push to `main`; destructive).
2. **Python 3.10 → 3.12 runtime** (Dockerfile `python:3.12-slim` and the local venv). Python
   3.10 reached end-of-life in Oct 2026, and the remaining Pillow (13.0.0) and keras (≥ 3.15.0)
   security fixes are published only for Python ≥ 3.11. TF 2.20 supports 3.12. That is a
   runtime change, so it needs a full rebuild and a test on Render's 512 MB.
3. **Pillow 13** (major version), which only makes sense together with item 2.
4. **API clean-up (L5):** drop the ignored `content_weight`, `tv_weight`, `num_steps` and
   `learning_rate` form fields and rename `style_weight` → `intensity`. That breaks anyone
   calling the public API directly.

### Not verifiable locally
- Memory behaviour on Render's 512 MB instance (H1); measured locally only.
- Whether Render actually forwards `CF-Connecting-IP` (M1). After deploying, check that two
  different networks get separate rate-limit buckets. If the header is missing, the limiter
  falls back to the socket address, which is no worse than before.
- Real phone browsers (tested with Chromium device sizes only); Safari/WebKit not tested.

---

## Critical

### C1 — Render API token committed to a public repo
- **Where:** commit `aefcb26` added `.kiro/settings/mcp.json`, which contains an
  `Authorization` header for Render's MCP endpoint (with `create_web_service` and
  `update_web_service` auto-approved). `29839d9` deleted the file, but the commit is still
  in history on `origin/main`, and the GitHub repo is **public**.
- **Reproduce:** `git show 5a7d6b8:.kiro/settings/mcp.json` (I only inspected it with values redacted).
- **Impact:** Anyone can read the token and control your Render account (create or modify
  services, read logs and env vars).
- **Fix (you must do this; I can't):**
  1. In the Render dashboard → Account Settings → API Keys, **revoke that key now**
     and create a new one if you still need it. Revoking is the actual fix.
  2. Optional: rewrite history (`git filter-repo --path .kiro --invert-paths`) and
     force-push. That is destructive and rewrites `main`, so it needs your go-ahead.
     After revoking, it is cosmetic.
- No other secrets were found in tracked files or in client code. The app needs none.

## High

### H1 — Full-resolution decode of large uploads exhausts memory
- **Where:** `nst_engine.load_and_preprocess`, `color_palette_transfer._open_capped`.
  Each opens the upload, `convert("RGB")`s it at full size, and then resizes. `run_nst`
  decodes the content image twice.
- **Reproduce:** decode a 7000×7000 JPEG the current way and the process's peak memory
  rises by **375 MB**. With `Image.draft()` (JPEG DCT scaling) it rises by **7 MB**. Render's
  free tier has 512 MB, and TF plus the model already take most of it, so a modern phone
  photo (12–50 MP) is very likely what OOM-kills the live service (git history shows
  repeated OOM firefighting).
- **Fix:** one shared loader that uses `draft()` and `thumbnail()` (reduced-scale decode),
  and decode the content once when inference and output sizes are equal.

### H2 — EXIF orientation ignored
- **Where:** every `Image.open` in `nst_engine.py`.
- **Reproduce:** upload a JPEG with EXIF Orientation=6 (shows as 200×400 portrait in the
  browser). The result is a 320×160 sideways landscape. A painted mask is also misaligned,
  because the browser preview is upright while the server's pixels are rotated.
- **Fix:** apply `ImageOps.exif_transpose` in the shared loader.

### H3 — Re-toggling "Paint Mask" erases the mask but still sends it
- **Where:** `app.js` `toggleMaskMode()` resets `canvas.width`, which clears it, while
  `maskHasContent` stays `true`.
- **Reproduce:** upload → Paint Mask → Fill All → Paint Mask (off) → Paint Mask (on):
  0 painted pixels, `maskHasContent: true`. Generate sends an all-black mask and the
  "artwork" is the original photo, unchanged.
- **Fix:** size the canvas once when the content image loads, and don't touch it on toggle.

### H4 — Mobile studio overflows horizontally after a result
- **Where:** `style.css` `.studio-layout { grid-template-columns: 1fr }` at ≤ 900 px. A
  `1fr` track can't shrink below its min-content width, and the nowrap "Download /
  Animation / Reset" buttons make that 420 px.
- **Reproduce:** at 360 px or 390 px, generate an image. The page scroll width becomes
  444 px and the panels, buttons and before/after slider are cut off on the right (see
  `before/mobile-08-studio-result.png`).
- **Fix:** `minmax(0, 1fr)` and let the output action row wrap.

### H5 — Core flow not keyboard accessible
- **Where:** `index.html` upload zones are `<div onclick>`, and style preset cards are `<div>`s.
- **Reproduce:** Tab through the page. Focus goes from the nav to the style tabs and never
  reaches the photo upload zone or any preset card, so a keyboard user can't create anything.
- **Fix:** give the zones and cards `role="button"`, `tabindex="0"` and Enter/Space
  handlers, plus `aria-pressed` on the selected style.

## Medium

### M1 — Per-IP rate limits can be bypassed
- **Where:** `main.py` runs uvicorn with `forwarded_allow_ips="*"`. In uvicorn 0.41 that
  takes the **leftmost** `X-Forwarded-For` entry, which the client controls, and Render
  appends to that header rather than resetting it.
- **Reproduce:** send `X-Forwarded-For: <random>` on each request; every request gets a
  fresh limit bucket.
- **Fix:** key the limiter on `CF-Connecting-IP` (Render sits behind Cloudflare, which
  overwrites that header), falling back to the socket address. The global queue cap
  (`MAX_PENDING_JOBS`) still protects the server either way.

### M2 — Raw JSON parse errors shown to users
- **Where:** `app.js` calls `await res.json()` on every non-OK response.
- **Reproduce:** make `/api/transfer` return Render's HTML 502. The UI shows "Unexpected token
  '<', "<html><bod"… is not valid JSON" (`before/desktop-09-studio-error.png`). On the live
  site this is what users see when the service is waking up or was OOM-killed.
- **Fix:** one response parser that falls back to friendly messages per status
  (413 too large, 429 busy, 502/503/504 waking up or overloaded, network offline).

### M3 — Some downloads save an error page as `pictura_*.jpg`
- **Where:** `downloadResult()` links to `/api/result/{currentJobId}`. Opening a palette
  result from history sets the id to `palette_…`, which the server doesn't know. Every job
  also disappears when Render restarts the free instance or after `JOB_TTL_SECONDS`.
- **Reproduce:** run a palette transfer, click its history thumbnail, then click Download.
  The file `pictura_palette_.jpg` contains a JSON 404.
- **Fix:** the browser already holds the image, so always download from the in-memory bytes.

### M4 — Animation after a text-prompt result says "Pick a style first"
- **Where:** `startInterpolation()` checks only `selectedPreset` and `styleFile`.
- **Reproduce:** type "watercolor portrait" with no preset selected → Generate → Animation.
- **Fix:** remember the server's `resolved_style_key` and use it for the animation.

### M5 — Vulnerable dependencies (`pip-audit`)
- `pillow 12.1.1`: about 60 advisories. Many are fixed in **12.2.0/12.3.0**; the rest need **13.0.0**.
- `python-multipart 0.0.22`: 5 advisories, fixed in **0.0.31** (patch line).
- `keras 3.12.4` (pulled in by TensorFlow): 3 advisories, fixed in **3.15.0**.
- **Fix:** bump within the current major (Pillow 12.3.x, python-multipart 0.0.31, pin
  keras ≥ 3.15) and retest. **Pillow 13 is a major upgrade, so it's listed for your decision.**

### M6 — Style list never retries
- **Where:** `loadStyles()` makes one attempt and doesn't check `res.ok`.
- **Reproduce:** load the page while the API is down or still waking. The preset grid stays
  "Could not load styles" and the gallery stays empty until a manual reload.
- **Fix:** retry with backoff and a "waking up the server" message.

### M7 — Low-contrast text
- `--text-3: #475569` is used for hints, stat labels and footer text. Contrast is
  **2.60:1** on the page background and **2.34:1** on panels (AA needs 4.5:1).
- **Fix:** raise the token to a value that clears 4.5:1 on both backgrounds.

### M8 — Missing accessible names
- Range inputs `#styleWeight`, `#mixRatio` and `#brushSize`, the icon-only download button
  and the "✕" clear buttons have no label. **Fix:** add `aria-label`s and `for=` attributes.

## Low

- **L1** `_interpolation_worker` stores the GIF base64 in the `jobs` dict for up to an hour
  (the transfer path avoids this on purpose for memory). **Fix:** read it from disk on request.
- **L2** `logo.png` (492×499, 174 KB) is shown at 32–34 px. **Fix:** ship a 96 px version.
- **L3** Browsers request `/favicon.ico` and get a 404 (seen on the live site). **Fix:** route it to the asset.
- **L4** No tests, lint config or CI. **Fix:** add pytest tests for the fixes in this phase
  (CI goes in the proposal).
- **L5** `.hero-visual` is `display:none` at every width, and its inline SVG is malformed
  (`<definS>`). The progress UI still shows "Loss"/"Step" from the old iterative engine,
  and `/api/transfer` accepts `content_weight`, `tv_weight`, `num_steps` and `learning_rate`
  but ignores them. Left for the redesign, so this phase doesn't change the API.
- **L6** No `Content-Security-Policy` (inline `onclick` handlers would force
  `'unsafe-inline'` for now), no HSTS from the app, `/docs` is public and CORS defaults to `*`
  (no credentials, so the risk is low). Proposed for the redesign.
- **L7** `/api/styles` returns 154 KB of base64 thumbnails inside JSON. The response can't
  be cached as images, and the preset grid stays empty until the API is awake. Proposed:
  serve static thumbnails.
- **L8** The live site took **~56 s** to show styles from a cold start (Render free tier
  sleeps after 15 minutes idle). Partly addressed by M2/M6 messaging. A real fix is in the proposal.

## Checks run

| Check | Result |
|---|---|
| `node --check frontend/assets/app.js` | pass |
| `python -m compileall backend` | pass |
| Existing tests / lint / build | **none exist** in the repo |
| `ruff --select F,E9,B` (ad hoc) | 15 findings, all style-level (B008 is FastAPI's normal `File()` default pattern, B904); 1 unused import |
| `pip-audit -r backend/requirements.txt` | 3 vulnerable packages (M5) |
| API smoke test: transfer (preset, text, mix), palette, interpolate, download, validation errors | all behave as designed. First transfer 26 s (model load), then ~5 s |
| Browser console (local, every state) | no errors except the 502 I injected on purpose |
| Live site | 503 Render wake page, then a working app after ~56 s; `/favicon.ico` 404 |
| Responsive 360/768/1440 | 768 and 1440 OK; 360/390 overflow in the result state (H4) |

## Phase 6 (Salon Wall): security and robustness fixes

Found while reviewing the rebuilt app (wave 3) and fixed on this branch. Each item has
backend tests (`backend/tests/test_limits.py`, `test_site.py`) unless noted.

| Fix | What changed | Commit |
|---|---|---|
| Upload pixel budgets | Pixel counts are read from the image header before anything is decoded: 413 above `MAX_UPLOAD_PIXELS` (12 MP) or, for JPEGs (decoded at a reduced DCT scale), `MAX_UPLOAD_PIXELS_JPEG` (40 MP). A few KB of PNG could declare 100+ MP and OOM-kill the 512 MB instance. JPEG masks are also decoded at reduced scale. | `dc3ef08` |
| Request body cap | ASGI `BodySizeLimitMiddleware`: POST `/api/*` bodies over `MAX_REQUEST_BYTES` (4 × `MAX_UPLOAD_MB` + 64 KB) get 413, by declared `Content-Length` or by counting streamed bytes. Multipart parsing allows at most 4 files and 24 fields (400). | `dc3ef08` |
| Queue slot before reading | Handlers take a `MAX_PENDING_JOBS` slot before reading any upload, so rejected requests (429) never hold image bytes, and only the job about to run holds them in memory. The slot is always released. | `dc3ef08` |
| `text_prompt` capped | Same 1–200 character limit and control-character cleaning as `/api/describe-style`, checked before the matcher runs on the event loop. | `dc3ef08` |
| Results not stored | `/api/transfer` and `/api/palette-transfer` return the image inline and write nothing to `backend/outputs/` or the job store (data minimisation). Only the GIF job keeps a file, for `JOB_TTL_SECONDS`. | `dc3ef08` |
| Proxy trust | uvicorn trusts `X-Forwarded-*` only from `FORWARDED_ALLOW_IPS` (default `127.0.0.1`, never `*`); rate limits keep using `CLIENT_IP_HEADER`. | `dc3ef08` |
| CSP narrowed | `script-src` / `connect-src` allow only `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/`, not the whole jsDelivr host; a test keeps the version equal to `segment.js`. | `4c846a7` |
| CI hardening | Docker check runs with `--memory 512m --memory-swap 512m` (no swap, as on Render) with diagnostics on failure; actions pinned to commit SHAs with Dependabot; `persist-credentials: false`; `permissions: contents: read`. | `dda8703` |
| Engine fallbacks | The studio paints on the server when the on-device engine's code cannot load (`ENGINE_UNAVAILABLE`); idle prefetch never falls back to the in-page engine. e2e tests in `tests/e2e/test_fallbacks.py`. | `9cf9b84`, `90292ad`, `63dc190` |
| https downgrade on redirects (wave 4) | With the proxy not a trusted peer, Starlette's trailing-slash redirects (`/app`, `/app/styles/<slug>`, `/docs/`) sent absolute `http://` `Location` headers to https visitors. Same-origin `Location` headers are now path-only (leading slashes collapsed, so `//host` cannot become an open redirect), and `/` goes straight to `/app/` in one hop. | this wave |
| CI peak memory (wave 4) | The Docker job prints the cgroup's `memory.peak` after one real stylization and warns above 460 MB. | this wave |
