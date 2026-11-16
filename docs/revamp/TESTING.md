# Testing PicturaAI

There are five kinds of check. Each runs on a laptop with the commands below and in
GitHub Actions (`.github/workflows/ci.yml`). All commands run from the repo root on
Windows. On Linux or macOS, use `venv/bin/python` instead of `venv\Scripts\python`.

| Suite | Command | Time here* | CI job |
|---|---|---|---|
| Backend unit and API tests | `venv\Scripts\python -m pytest backend/tests -q` | ~75 s, 167 tests | `backend` |
| Lint (errors only) | `venv\Scripts\ruff check --select F,E9 backend tests/e2e` | < 1 s | `backend` |
| Dependency CVEs | `venv\Scripts\pip-audit -r backend/requirements.txt` | ~50 s | `backend` |
| Browser end-to-end | `venv\Scripts\python -m pytest tests/e2e -q` | ~170 s, 48 tests (+2 slow, skipped) | `e2e` |
| On-device engine (slow) | `venv\Scripts\python -m pytest tests/e2e -m slow -q` | several minutes per test (2 tests) | not in CI |
| Compositor parity (JS vs Python) | `venv\Scripts\python docs/revamp/engine/parity/composite_parity.py` | ~12 s | `parity` |
| Generated files are current | `venv\Scripts\python scripts/build_style_assets.py --check` and `venv\Scripts\python scripts/build_pages.py --check` | < 5 s | `pages` |
| Docker image starts | `docker build -t picturaai .` then `docker run -p 8000:8000 --memory 512m --memory-swap 512m picturaai` | — | `docker` |

\* Timed on the development laptop (Windows 11, CPU only) on 2026-11-16. Times go up a
lot while other heavy processes run.

## One-time setup

```bash
python -m venv venv
venv\Scripts\pip install -r backend\requirements.txt -r requirements-dev.txt
venv\Scripts\python -m playwright install chromium
```

`requirements-dev.txt` pins pytest, httpx (needed by FastAPI's `TestClient`), Playwright,
pip-audit and ruff. The first test run downloads the style model (~90 MB) into
`.tfhub_cache/`.

## Backend tests (`backend/tests`)
These test the FastAPI app in-process with `TestClient` and a small fixed style catalogue,
so they don't depend on the real `data/styles.json`. They cover upload validation,
`/api/transfer` (final/raw, intensity, masks, mixing), palette transfer, the GIF job,
`/api/styles`, `/api/describe-style` (with the Workers AI client mocked), `config.js`,
CSP and the other security headers, cache headers, `robots.txt` and `sitemap.xml`, the
style catalogue validator, and the image pipeline.

| File | What it checks |
|---|---|
| `test_api.py` | HTTP behaviour without the model: palette transfer, a non-image upload (400), per-IP rate limits (a spoofed `X-Forwarded-For` does not reset them), a GIF job's result read from disk without leaking its path, `/favicon.ico`, HTML revalidation. |
| `test_transfer.py` | `/api/transfer`: `intensity` as an alias of `style_weight`, `output=raw` (mask ignored), legacy fields, preset and `text_prompt` resolution (an unmatched prompt gets the plain-language 400 on both sync routes), a second preset for mixing, and `run_nst(raw=True)`. |
| `test_limits.py` | Memory and abuse limits: per-format pixel budgets checked from the header (413 before decoding), the `MAX_REQUEST_BYTES` body cap for declared and chunked bodies, too many form files or fields (400), the queue slot taken before uploads are read (429) and always released, the `text_prompt` length cap, transfer and palette results never stored (no `job_id`, nothing in `backend/outputs/`), and `FORWARDED_ALLOW_IPS` (default `127.0.0.1`). |
| `test_site.py` | `/api/styles`, health, `config.js` (no secrets, escaping), the app and docs CSPs (MediaPipe path pinned and equal to `segment.js`), the other security headers, HSTS only over HTTPS, path-only `Location` on same-origin redirects behind an untrusted proxy (`/`, `/app`, `/app/styles/<slug>`, `/docs/`), `robots.txt`, `sitemap.xml`, the docs toggle, optional Sentry. |
| `test_static_cache.py` | `Cache-Control`, gzip and content types for files under `/app`. |
| `test_describe.py` | `/api/describe-style` and the Workers AI client (HTTP mocked): generation, fallback to matching, daily cap, provider quota, timeouts, prompt validation, oversized bodies, rate limit, `DESCRIBE_ENABLED=false`. |
| `test_style_catalog.py` | Loading and validating `data/styles.json`, and the keyword matcher. |
| `test_image_loading.py` | EXIF orientation and reduced-scale decoding of large uploads. |

## End-to-end tests (`tests/e2e`)
These use pytest and Python Playwright (sync API) against a real server in headless Chromium.

- **Server.** A session fixture starts `uvicorn main:app` from `backend/` on port
  **8230** with `NST_MAX_DIM=320` and very high rate limits. It removes every Cloudflare,
  Sentry and analytics variable from the environment, waits for `/api/health` and stops
  the server at the end. To use a server that is already running, set `E2E_BASE_URL`, for
  example `E2E_BASE_URL=http://127.0.0.1:8000`. `E2E_PORT` changes the port, and
  `E2E_HEADED=1` shows the browser.
- **Engine.** The flows use `?engine=server`, because the on-device TF.js engine is very
  slow in headless Chromium, which renders WebGL in software (SwiftShader). Both engines
  share one compositor, and the parity script tests that compositor.
- **Fonts.** The fonts are self-hosted under `frontend/assets/fonts/`, so the layout
  checks measure the fonts users get and the suite needs no internet for them.
  `test_assets.py` checks they load from `/app/assets/fonts/` and that no page asks
  Google Fonts for anything.
- **Watchers on every page.** The suite records console errors, uncaught page errors and
  CSP violations through a `securitypolicyviolation` listener that is injected before the
  page's own scripts. Any of these fails the test at teardown unless the test expected
  it with `watch.allow(...)`, as the mocked 503 test does.

| File | What it checks |
|---|---|
| `test_shell.py` | `/api/health`. Security headers on `/app/`: nosniff, `X-Frame-Options: DENY`, Referrer-Policy, Permissions-Policy, a CSP without `unsafe-inline`/`unsafe-eval` in `script-src`, and no HSTS on plain HTTP. `config.js` is JavaScript and has no token in it. The studio loads with the sample photo, exactly one painting card per preset returned by `/api/styles` (14 today), one tab stop, and Paint disabled with its reason shown. At 1440×900 and 1024×768 the gallery shows "14 paintings. Scroll for more." until it is scrolled to the end or another tab is chosen; at 390 and 768 px there is no such line. There is no horizontal overflow and no clipped control group at 360, 390, 768, 1024 and 1440 px, at the top and bottom of the page. The menu sheet at 360 px keeps focus inside with Tab and Shift+Tab, Esc closes it and focus returns to the button. A 503 from `/api/styles` shows a status. The CSP really blocks an injected inline script, and the listener catches it. |
| `test_studio.py` | Choosing a painting with the arrow keys, then Paint it. The wall label reads "After The Starry Night". The compare slider responds to ←/→/Home/End. Moving the live intensity slider changes the canvas pixels with **no** network request. The download menu shows sizes and Esc returns focus to the button. The Instagram download is 1080 px wide, the print download is 1754 px on its long side, and the full-size download is a JPEG. Colours only runs locally with no `/api/transfer` call. Mixing two styles. A region mask painted with the mouse styles only the brushed area. Invert changes the mask overlay and swaps the result (the brushed stroke shows the photo, an unbrushed corner is styled), and Clear empties the mask and disables Paint again. My gallery: save, then "Saved", delete with the empty state, Undo, and open back into the frame. Describe it in matched mode (no Cloudflare): the button says "Find a style" and it picks The Starry Night. Describe it falls back to a local match when the endpoint returns 502. A mocked 503 HTML response becomes a readable error in the frame, then Dismiss, then a successful retry. Cancel while painting hides the veil in under 1 s (measured in the page) and shows a toast. A PDF upload is rejected on the frame. A 900×1200 PNG updates the label and the frame's aspect ratio. A custom painting gives "After your painting". Animate downloads a WebM/MP4; the test is skipped (with the reason) only when the browser has no MediaRecorder video type, otherwise a hidden Animate button fails it. |
| `test_phone.py` | At 390 px with touch and reduced motion: the bottom bar's Paint it, then Download (opens the size menu). Changing the style brings Paint it back. With reduced motion there is no brush-wipe animation and nothing overflows sideways. Painting a region mask with real touch events (CDP) enables Paint and does not scroll the page. |
| `test_fallbacks.py` | Engine fallbacks, made to fail early with routes so no TF.js inference runs: `engine/worker.js` blocked moves the browser engine onto the page (`offMainThread` false, `browser-engine.js` loaded by the page); TF.js blocked makes the server paint with the "server is painting it" toast; a WebGL backend that fails only in the worker (`ENGINE_WORKER`) moves the engine to the page; `engine/index.js` blocked uses the server engine with no on-device claim; `?worker=0` with `browser-engine.js` blocked falls back to the server with the toast; no WebGL at all picks the server up front without downloading TF.js. `config.js`: `describeEnabled:false` hides the Describe it tab, `describeGenerates:true` labels the button "Create style", and a missing `config.js` still gives a working studio. The browser-engine tests skip when Chromium has no WebGL2 (the engine would never be chosen). One `slow` test lets the in-page engine really paint with the worker blocked and checks there is no `/api/transfer` call. |
| `test_assets.py` | The `modulepreload` links in `index.html` match exactly the modules `main.js` imports statically, and no engine module is among them. The studio, a style page and the privacy page load their `.woff2` fonts only from `/app/assets/fonts/` and nothing from Google Fonts. A style page's hero picks the 512 px JPEG at DPR 1 and the 1024 px WebP at DPR 2 at the same on-screen size. |
| `test_short_screens.py` | Short laptop screens and phone toasts. At 1536×690 (1920×1080 at 125 % Windows scaling) and 1366×625, Paint it and the Mix toggle are fully visible inside the style panel and above its stuck call to action in every tab (Gallery, Your painting, Describe it after a matched answer), and with Mix open its ratio slider and Paint it are too. At 1536×690, arrow keys through the gallery (six right, six left) keep the focused card fully visible. At 360×640 with touch, after painting and saving, the "Saved to my gallery." toast does not cover the intensity slider on the frame's ledge (`elementFromPoint` at the slider's centre is the slider). |
| `test_browser_engine.py` (`slow`) | `?engine=browser`: paints on the device with **no** `/api/transfer` request, and model files come only from `/app/models/`. It is skipped unless you pass `-m slow`. In headless Chromium with SwiftShader, the first paint (model download, shader compile, inference) takes minutes. |

### Writing new e2e tests
- Find elements by role, label or text: `page.get_by_role("button", name="Paint it")`,
  `style_radio(page, "The Kiss")`. Use ids only for canvases and output regions that have
  no role. Helpers are in `tests/e2e/e2e_helpers.py`.
- Use `open_page(width, path=..., **context_options)` for a fresh browser context, which
  also gives a fresh IndexedDB. It returns a `PageWatch`, with the page at `.page`.
- Keep the default run under about 4 minutes. Anything longer gets `@pytest.mark.slow`.

## Parity and pages checks
`composite_parity.py` runs the backend `run_nst` post-processing and palette transfer in
Python, runs `frontend/assets/js/engine/composite.js` in Node on the same inputs, and fails
if any mean difference is above 0.5 (on a 0–255 scale). It needs Node 18 or newer and the backend
venv. CI uses Node 22 (Node 20 is end-of-life). `build_style_assets.py --check` validates `data/styles.json` and
checks that every generated image exists. `build_pages.py --check` fails when a generated
static page (per-style pages, credits) is out of date.

## Continuous integration
`.github/workflows/ci.yml` runs on pushes to `main` and `revamp/**` and on every pull
request. It has `permissions: contents: read`, and a newer push cancels the running
workflow for the same ref. Each action is pinned to a full commit SHA, with the tag in
a trailing comment (`checkout` v7.0.1, `setup-python` v7.1.0, `setup-node` v7.1.0,
`cache` v6.1.0), so the owner moving a tag cannot change what runs.
`.github/dependabot.yml` (`github-actions`, weekly, one grouped PR) bumps the SHA and
the comment together. To check a pin by hand, run
`git ls-remote https://github.com/actions/<name> refs/tags/<tag>`. These are lightweight
tags, so the SHA printed is the commit. Every `actions/checkout` step sets
`persist-credentials: false`, so `GITHUB_TOKEN` is not left in `.git/config` while
the jobs pip-install and run the branch's code. No job pushes or calls git with
credentials after checkout.

| Job | Runs | Notes |
|---|---|---|
| `backend` | ruff (F, E9), backend pytest, `pip-audit -r backend/requirements.txt` | Caches pip and `.tfhub_cache`. The cache key is the model handle (`MODEL_CACHE_KEY`), so change it when `MODEL_HANDLE` changes. |
| `e2e` | `playwright install --with-deps chromium`, then `pytest tests/e2e` | Runs after `backend` passes and reuses the model cache. |
| `docker` | `docker build` (no push), runs the container with `--memory 512m --memory-swap 512m` (Render's limit, with no swap), polls `/api/health` for up to 180 s, sends one real `/api/transfer`, checks `State.OOMKilled`, prints `docker stats` and the cgroup's peak memory | The model is downloaded during the image build. Without `--memory-swap` the container could swap on GitHub's runners (they have a swapfile) and pass where Render would OOM-kill it. The step prints the cgroup's `memory.swap.max` and warns if it is not 0. After the stylization it reads `/sys/fs/cgroup/memory.peak` (model load plus one request), prints it in MB and emits a `::warning::` above 460 MB, so creeping memory use shows up before Render starts OOM-killing; on cgroup v1 it says the value is not available. A failed curl (exit 52/28 when the server dies) does not abort the step, and on any failure a diagnostics step prints the container state, `OOMKilled` and the last 300 log lines. |
| `parity` | Node 22 and Python 3.12, `composite_parity.py` | |
| `pages` | Pillow only, then the two `--check` scripts | Skips `build_pages.py` if the file is missing. |

### Cost and free-tier limits
From GitHub's billing page
(<https://docs.github.com/en/billing/managing-billing-for-your-products/managing-billing-for-github-actions/about-billing-for-github-actions>,
checked 2026-11-16):

- **Public repositories:** standard GitHub-hosted runners are **free**, so this workflow
  costs nothing while the repo is public.
- **Private repositories:** each plan includes a monthly amount of compute in US dollars,
  not in minutes. GitHub Free includes **$12 of compute and 500 MB of artifact storage**,
  Pro $18 and 1 GB, Team $18 and 2 GB, and Enterprise Cloud $300 and 50 GB. A standard
  Linux 2-core runner costs **$0.006 per minute** (Windows $0.010, macOS $0.062), so $12
  is about **2,000 Linux minutes**. Self-hosted runners are free and don't use the
  included amount.
- **Estimate for this workflow (not yet measured on GitHub).** About 20–30 runner-minutes
  per push across the five parallel jobs, about 12 minutes of wall-clock time. Installing
  TensorFlow and building the Docker image take most of it. That is roughly 70–100
  pushes a month on a private Free plan. To save minutes on a private repo, limit `docker`
  and `parity` to `main` (add `if: github.ref == 'refs/heads/main'`), or run them only
  when `Dockerfile`, `backend/**` or `frontend/assets/js/engine/**` change.
- The workflow uploads no artifacts, so it uses none of the artifact storage.

## What is not covered
- **Real devices and other browsers.** The e2e suite runs only in desktop Chromium, with
  touch and phone viewports emulated. Safari (iOS and macOS), Firefox, Android Chrome on
  real GPUs, Web Share sheets, the native file picker, and video recording on phones are
  untested. Check them by hand before a release.
- **The on-device engine in CI.** The `slow` tests are not in CI because SwiftShader makes
  it take minutes. Real-GPU speed, WebGPU and memory pressure on low-end phones are not
  measured.
- **Render's memory limit under load.** The `docker` job starts the image at 512 MB and
  sends one request. It does not test concurrent requests, the GIF job, long uptimes, or
  Render's cold start and sleep. Check those with Render's metrics after a deploy.
- **Third-party services.** Cloudflare Workers AI generation, Sentry and Cloudflare Web
  Analytics are mocked (backend tests) or switched off (e2e). Their live behaviour, and
  the CSP once their variables are set, are checked only by the backend's header tests.
- **Visual appearance.** No pixel or screenshot comparison. Screenshots for a person to
  review are in `docs/revamp/after/final/`. The short-screen tests measure two
  laptop sizes (1536×690, 1366×625) and one phone; the shorter sizes where the style
  panel scrolls with "More below" (1280×585, 1024×600, 175 % zoom), the frame and
  intensity ledge being in view after painting, the compare slider's peek and the
  download menu following its button were checked by hand only.
- **Accessibility audit.** Keyboard paths, roles and focus are tested. Colour contrast
  and screen-reader output are not checked automatically.
- **actionlint.** The workflow YAML was checked by parsing it and running `bash -n` on
  every `run:` block. actionlint has not been run on it yet.
