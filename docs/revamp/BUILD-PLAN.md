# Phase 6 build plan (shared spec for every build agent)

Approved by the owner on 2026-11-16: build everything in `PROPOSAL.md` with design direction
**A "Salon Wall"**, fix the copyrighted presets, and make mobile/tablet flawless.
Not in scope (needs accounts/business decisions): payments, user accounts, print-on-demand.

Read `docs/CLAUDE.md` (stack, conventions) and `docs/revamp/PROPOSAL.md` (why) first.

## 0. Hard constraints
- **Stack stays:** FastAPI backend (Python 3.12, `venv/`), plain HTML/CSS/JS frontend with
  **native ES modules and no build step**. No React, no bundler, no npm runtime deps.
- **Free only.** Third-party scripts only from `cdn.jsdelivr.net` (exact pinned versions,
  with `integrity` SRI where the file is static). Fonts self-hosted under frontend/assets/fonts (SIL OFL; originally Google Fonts). Model weights are
  **self-hosted** under `frontend/models/` (licence files alongside).
- **CSP-ready:** no inline `<script>`, no `on*=""` attributes, no `style=""` attributes in
  HTML. Set styles from JS via `el.style.x` / classes (CSSOM is allowed).
- Never print or commit secrets. Optional third-party keys come from env vars (names only).
- Don't push, don't deploy. Commit small, one concern per commit, with the trailer
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Only edit the files your task owns** (see §7). If you need a change elsewhere, write it
  in your final report instead.
- Run the app: `cd backend && ../venv/Scripts/python.exe -m uvicorn main:app --port <your port>`
  (pick a port 8100–8199 given in your task so parallel agents don't collide). App URL:
  `http://127.0.0.1:<port>/app/`. Tests: `venv/Scripts/python.exe -m pytest backend/tests -q`.

## 1. Design system — "Salon Wall"
The app is a gallery: the user's photo hangs in a gilt frame on a deep green museum wall;
each style is presented like a museum piece with a wall label (title, artist, year).
Spend boldness in ONE place: the frame + the "unveiling" reveal. Everything else is quiet.

### Colour tokens (contrast measured)
| Token | Hex | Use |
|---|---|---|
| `--wall` | `#1D3530` | page background |
| `--wall-deep` | `#142622` | panels, header, bottom bar |
| `--wall-raised` | `#24423B` | hover/selected surfaces, inputs |
| `--linen` | `#ECE5D6` | primary text (≥ 8.7:1 on all walls) |
| `--linen-muted` | `#B8B2A2` | secondary text (≥ 5.2:1 on all walls) |
| `--gilt` | `#C9A44C` | primary button fill, selection, focus ring, frame (≥ 4.6:1 on walls) |
| `--gilt-bright` | `#DDBB66` | primary hover |
| `--on-gilt` | `#26221D` | text on gilt (6.7:1) |
| `--label` | `#F3EEE3` | wall-label cards (museum labels) |
| `--ink` | `#26221D` | text on label (13.7:1) |
| `--ink-muted` | `#6B6357` | secondary text on label (5.1:1) |
| `--rust` | `#8A3B22` | accent text on label (6.7:1) |
| `--vermilion` | `#E0613F` | error icons/markers only |
| `--vermilion-text` | `#F29478` | error text on the walls (5.8:1 on `--wall`, 7.0:1 on `--wall-deep`, 4.84:1 on `--wall-raised`; the earlier `#F08A6C` was only 4.46:1 on `--wall-raised`) |
| `--hairline` | `rgba(236,229,214,.14)` | dividers on wall |
| `--gilt-shade` / `--gilt-light` | `#A9863A` / `#D8B865` | frame bevel tones (with `--gilt`); decorative, never text |
| `--frame-edge` | `#0F1C19` | the frame's dark outer edge |

Wall tones by name: `--wall` (page), `--wall-deep` (panels, header, bottom bar),
`--wall-raised` (hover, selected, inputs). The values above are the ones in
`frontend/assets/css/tokens.css`, which is the source of truth.

No purple, no gradients-as-decoration, no glassmorphism, no emoji icons (use inline SVG,
1.5px stroke, `currentColor`). Permitted light effect: a soft **gallery spotlight** above the
frame (radial gradient from the top, very subtle).

### Type
- **Display:** `Libre Caslon Display` (400 only) — page title, style names in labels,
  result title. Never for UI controls. Never all caps.
- **UI/body:** `Schibsted Grotesk` 400/500/600/700.
- Scale (px): 13 (meta), 14 (small/controls), 16 (body), 20 (h3 / label title), 28 (h2),
  44 (h1 desktop; 34 mobile), 64 (hero display on ≥1100 only if used).
- Line-height: 1.55 body, 1.15 display. Max line length ~70ch. Sentence case everywhere.
  **No all-caps tracked eyebrows, no "A · B · C" meta strings, no "→" appended to buttons,
  no one-word-accented headlines.**

### Spacing, radius, shadow, motion
- Spacing scale 4px base: `--s-1:4 --s-2:8 --s-3:12 --s-4:16 --s-5:24 --s-6:32 --s-7:48 --s-8:64 --s-9:96`.
- Radius by hierarchy: frame **0** (real frames are square), wall labels **2px** (paper),
  buttons/inputs **8px**, chips/segmented control **999px**, panels **12px**.
- Shadows: frame `0 40px 70px -30px rgba(0,0,0,.65), 0 2px 0 rgba(0,0,0,.4)`; label
  `0 10px 24px -14px rgba(0,0,0,.6)`; nothing else gets a shadow.
- Motion: `--t-fast:120ms`, `--t-base:200ms`, `--t-reveal:700ms`,
  `--ease: cubic-bezier(.2,.7,.2,1)`. Motion only answers user actions, except the single
  signature **unveiling**: when a result arrives, it is revealed across the frame by one
  brush-edged wipe (CSS `mask-image` with an SVG brush edge, left→right, ~700ms) and the wall
  label title fades in. `prefers-reduced-motion: reduce` → 150ms cross-fade, no wipe.
- Focus: `outline: 2px solid var(--gilt); outline-offset: 3px` on `:focus-visible`.
- Touch targets ≥ 44×44px. Hover effects only on `(hover: hover)` devices.

### The frame (signature element)
Gilt frame built in CSS: outer dark edge → gilt band with subtle bevel
(`linear-gradient` with 3 close gilt tones, NOT a rainbow) → linen passe-partout (mat,
`--label` colour, ~4% of width) → image. Square corners. Frame scales with the image's
aspect
ratio; max height `min(70vh, 760px)` on desktop. The before/after compare lives **on the
frame** (draggable gilt handle; keyboard ←/→ moves it 5%; `role="slider"` with aria values).

## 2. Screens & states (every screen needs loading, empty, error, success)
Single page `frontend/index.html`, studio first (no marketing hero above it).

1. **Header** (`--wall-deep`): wordmark "PicturaAI" (Caslon) + nav: Studio · Styles ·
   My gallery · How it works (anchor links). Mobile (<768): wordmark + menu button →
   accessible sheet (focus trap, Esc closes).
2. **Studio** (`#studio`):
   - Intro line above frame: h1 "Turn a photo into a painting" + one sentence:
     "Pick a photo and a painting. It runs on your device, so your photo stays private."
     (only claim on-device if the engine is the browser engine; otherwise say "in seconds").
   - **Frame column**: empty state shows a sample photo in the frame with
     "Try a sample" chips (Dog / City / Lake) + "Upload your photo" button + drag-and-drop
     anywhere on the frame (visible drop state). Accepts JPG/PNG/WEBP up to 25 MB; check
     type/size/megapixels before processing and show the reason on the frame.
   - **Wall label** under the frame (museum label card): before result: "Your photo" +
     filename + size; after result: Caslon title "After *The Starry Night*" (italic title
     allowed here because it's a real artwork title, not an accent), line 2 "Vincent van Gogh,
     1889 · intensity 75%" → write it as two lines, not a dotted string.
   - **Controls (plinth)** under the label:
     - Mode segmented control: **Full style · Colours only · Paint a region**.
     - Intensity slider (0–100%), **live**: updates the frame instantly (no new request).
     - Region tools (only in "Paint a region"): **Style the subject** / **Style the
       background** (one tap, on-device segmentation), brush draw/erase, brush size, clear,
       invert. Mask overlay shown as translucent gilt hatch, not green.
     - Result actions: **Download** (menu: Full size · Instagram 1080 · Print, with pixel
       sizes shown), **Share** (Web Share API when available, else hidden), **Animate**
       (records a 0→100% style reveal video, WebM/MP4), **Save to my gallery**.
   - **Style column** (`--wall-deep` panel): tabs **Gallery · Your painting · Describe it**.
     - Gallery: "salon hang" grid of style cards with varied spans based on each painting's
       aspect ratio; card = image + small label (title, artist, year). Selected = gilt
       outline + check; `role="radio"` within a radiogroup, arrow keys move.
     - Your painting: upload zone for a custom style image.
     - Describe it: text field + "Create style" → generated swatch (server, Workers AI) or
       closest classic style fallback with a clear message.
     - Disclosure "Mix with a second style" → second picker (compact) + ratio slider.
     - Primary button **Paint it** (gilt). Disabled until a photo + a style exist, with the
       reason shown ("Add a photo first").
   - **Generating**: frame keeps the photo, linen veil + thin gilt progress line along the
     frame's bottom edge, status text with elapsed seconds; first-time model load shows
     real download progress ("Preparing the studio — 4.2 of 11 MB"). Cancel button.
   - **Error**: inside the frame area: vermilion marker + plain message + actions
     (Try again keeps inputs; "Use a smaller photo" when relevant).
3. **Styles** (`#styles`): the full collection with wall labels + short descriptions and a
   "Use this style" button each (selects it and scrolls to studio). Links to the per-style
   SEO pages.
4. **My gallery** (`#gallery`): saved paintings (IndexedDB, this device only). Empty state:
   "Paintings you save appear here. They stay on this device." Items: open in frame,
   download, delete (with undo toast). Shows storage note.
5. **How it works** (`#how-it-works`): 3 numbered steps (it IS a sequence) + short FAQ
   (privacy/on-device, image sizes, where styles come from → credits page, offline?).
6. **Footer**: Credits (artwork sources & licences), Privacy, Terms, GitHub link.
7. **Static pages** (`frontend/privacy.html`, `terms.html`, `credits.html`,
   `frontend/styles/<slug>/index.html`): same header/footer, `pages.css`.

### Responsive targets (all must be perfect)
- 360 & 390 (phone): one column; frame full-bleed inside 16px gutter; style gallery becomes
  a horizontal scroll-snap rail (2.3 cards visible) with the tabs above it; controls stacked;
  **sticky bottom action bar** (`--wall-deep`) holding "Paint it" (or Download/Share after a
  result), respecting `env(safe-area-inset-bottom)`. No horizontal page scroll ever.
- 768 (tablet portrait): one column, frame max-width 640 centred, style grid 4 columns,
  controls in two columns.
- 1024 (tablet landscape) & ≥1100: two columns (frame ~58% / styles ~42%), styles panel
  sticky within the viewport.
- 1440: max content width 1320px centred.

## 3. Engine contract (owned by the engine agent; UI calls only this)
`frontend/assets/js/engine/index.js` exports:
```js
// Picks the browser engine (TF.js, WebGL) when supported, else the server engine.
export async function createEngine(options = {}) // -> Engine
// Engine:
// {
//   kind: 'browser' | 'server',
//   label: string,                // "on this device" | "on our server"
//   maxInferSide: number,         // long side of the raw stylization
//   warmup(onProgress?): Promise<void>  // loads models; onProgress({loaded,total,phase})
//   stylize({ content, style, style2, mix, signal, onProgress }): Promise<{ stylized: ImageData }>
//     content: ImageBitmap | HTMLCanvasElement (any size; engine downsizes for inference)
//     style/style2: { key: string } (preset) | { image: ImageBitmap|HTMLCanvasElement } (custom)
//     mix: 0..1 weight of style2 (ignored when style2 is null)
//     returns the RAW stylized image (no intensity blend), long side = maxInferSide
// }
```
`frontend/assets/js/engine/composite.js` (pure functions, no DOM except OffscreenCanvas/canvas):
```js
// Port of backend run_nst post-processing so browser + server engines look identical.
export function createCompositor(contentImageData, stylizedImageData) // same size required; caller resizes
// -> { render({ intensity, mask /* ImageData|null, white = styled */, sharpen = true }) : ImageData }
//    must be fast enough for live slider at ≤1200px long side (target < 40ms on desktop)
export function paletteTransfer(contentImageData, styleImageData, { strength, style2ImageData, mix }) // -> ImageData
export function resizeImageData(imageData, width, height) // high quality downscale/upscale helper
```
`frontend/assets/js/engine/segment.js`:
```js
export async function createSegmenter() // -> { segment(image): Promise<{ subjectMask: ImageData /* white=subject */, foundLabels: string[] }> }
```
Server engine: `POST /api/transfer` with `output=raw` returns the raw stylized image (see §5),
so the same compositor runs for both engines.

## 4. Data contract — `data/styles.json` (owned by the presets agent)
```json
{
  "version": 1,
  "styles": [
    {
      "key": "starry_night",                 // stable id, snake_case
      "slug": "van-gogh-starry-night",       // for /app/styles/<slug>/
      "title": "The Starry Night",
      "artist": "Vincent van Gogh",
      "year": "1889",
      "medium": "Oil on canvas",
      "description": "Swirling night sky in deep blues and yellows.",  // ≤ 90 chars, plain
      "tags": ["starry", "night", "swirl", "van", "gogh", "impressionist", "sky"],
      "file": "starry_night.jpg",            // in images/style_image/ (server)
      "image": "assets/styles/starry_night-512.jpg",   // web, ≤512px long side, for the browser engine
      "card": "assets/styles/starry_night-480.webp",   // gallery card
      "thumb": "assets/styles/starry_night-160.webp",  // small picker
      "aspect": 1.26,                        // width / height of the artwork
      "credit": "Vincent van Gogh, The Starry Night, 1889. Museum of Modern Art, New York.",
      "source_url": "https://commons.wikimedia.org/wiki/File:...",
      "license": "Public domain"
    }
  ]
}
```
All paths in `image/card/thumb` are relative to `frontend/` (served under `/app/`).
Sample photos: `frontend/assets/samples/{dog,city,lake}.jpg` (≤1600px, CC0/PD) +
`frontend/assets/samples/credits.json`.

## 5. Backend changes (owned by the backend agent)
1. Load presets from `data/styles.json` (replaces `STYLE_PRESETS` / `STYLE_TEXT_TAGS`).
2. `GET /api/styles` → list with key, slug, title (+ legacy `name`), artist, year,
   description, image/card/thumb URLs (prefixed `/app/`), aspect, credit. Base64 `thumbnail`
   only when `?include=thumbnail` (back-compat). `Cache-Control: public, max-age=3600`.
3. `GET /api/health` → `{"status":"ok"}` (cheap; keep `/api/styles` working for Render).
4. `POST /api/transfer`: add `intensity` (0–1, alias of `style_weight`) and
   `output` = `final` (default, current behaviour) | `raw` (stylized image at content size,
   no blend/detail/sharpen, JPEG q95). Legacy fields stay accepted (documented as ignored).
5. `POST /api/describe-style` (`prompt`, ≤ 200 chars): if `CLOUDFLARE_ACCOUNT_ID` and
   `CLOUDFLARE_API_TOKEN` are set, call Workers AI (`@cf/black-forest-labs/flux-1-schnell`,
   prompt wrapped as "a painterly texture study in the style of: …", small size) and return
   `{source:"generated", image:<base64 jpeg>}`; enforce `DESCRIBE_DAILY_LIMIT` (default 50/day,
   in-memory counter) and a per-IP rate limit; on any failure/quota/not-configured return
   `{source:"matched", key, title, message}` using the keyword matcher, or 404-style message
   if nothing matches. Never log the token.
6. `GET /app/config.js` → `window.PICTURA_CONFIG = {...}` from env: `PUBLIC_SITE_URL`,
   `SENTRY_DSN_FRONTEND`, `CF_WEB_ANALYTICS_TOKEN`, `DESCRIBE_ENABLED` (bool). No secrets.
7. Security headers: CSP built from config (self + jsdelivr + Google Fonts + optional
   Cloudflare analytics / Sentry; `'wasm-unsafe-eval'` for TF.js/MediaPipe; `worker-src 'self' blob:`;
   `img-src 'self' data: blob:`), HSTS when the request is HTTPS (`X-Forwarded-Proto`),
   `Permissions-Policy`, keep existing headers. Must not break `/docs` (relax CSP there).
8. `GET /robots.txt`, `GET /sitemap.xml` (app URL, styles section, every style page,
   privacy/terms/credits) using `PUBLIC_SITE_URL` (default `https://picturaai-k7q6.onrender.com`).
9. Optional Sentry for the backend when `SENTRY_DSN` is set (`sentry-sdk`, import guarded).
10. `ENABLE_API_DOCS` (default `true`).
11. Update `render.yaml` env list + `CLAUDE.md` env table. Tests for everything.

## 6. Quality bar
- WCAG AA contrast, keyboard for everything, visible focus, labels/aria for all controls,
  `aria-live="polite"` status region, reduced motion respected.
- No console errors. No layout shift on load (reserve frame space). No horizontal scroll
  at 360/390/768/1024/1440.
- Real copy only (see §2). No lorem ipsum, no placeholder images.
- Performance: first paint must not wait for TF.js/MediaPipe; load them lazily (on first
  "Paint it" or idle prefetch after load). Our own JS < 120 KB uncompressed total.

## 7. File ownership (parallel agents MUST stay inside their area)
| Agent | Owns |
|---|---|
| Presets & samples | `images/style_image/**`, `data/styles.json`, `frontend/assets/styles/**`, `frontend/assets/samples/**`, `scripts/build_style_assets.py`, `docs/revamp/presets/**` |
| Engine | `frontend/models/**`, `frontend/assets/js/engine/**`, `scripts/precompute_*`, `docs/revamp/engine/**` |
| Backend | `backend/**` (incl. tests), `render.yaml`, `CLAUDE.md` |
| UI shell | `frontend/index.html`, `frontend/assets/css/**`, `frontend/assets/js/**` except `engine/`, `frontend/assets/icons/**`, removal of legacy `frontend/assets/app.js` + `style.css` |
| Content & SEO (later) | `frontend/privacy.html`, `terms.html`, `credits.html`, `frontend/styles/**`, `scripts/build_pages.py`, `README.md` |
| Tests & CI (later) | `.github/**`, `tests/e2e/**`, `pytest.ini`/`pyproject.toml` (test config only) |

Progress is logged in `docs/revamp/BUILD-LOG.md` (append-only, by the orchestrator).
