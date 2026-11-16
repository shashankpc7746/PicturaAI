# Phase 6 build log (append-only)

## 2026-11-16 ~03:20 — kickoff
- Owner approvals: history purge (done, force-pushed main → 3d40053), Python 3.12 (done,
  byte-identical outputs, tensorflow-hub dropped), build everything in PROPOSAL.md with
  design A "Salon Wall", fix copyrighted presets.
- Plan: `docs/revamp/BUILD-PLAN.md`.
- Wave 1 (parallel): presets & samples · engine (TF.js on-device + compositor + segmentation)
  · backend API changes · UI shell (Salon Wall design system + all screens).
- Wave 2: integration (wire engine + data into UI), content & SEO pages, tests & CI.
- Wave 3: QA fan-out (viewports 360/390/768/1024/1440, a11y, e2e, perf, security,
  code review) → fix loop → after screenshots → final report.

## ~03:35 — wave 1 launched
- Workflow run `wf_2c7c9c61-e29` (task wdb046ghv): presets (port 8140), engine (8110),
  backend (8120), ui-shell (8130). Agents do not commit; orchestrator commits their
  commit_groups after review.
- If this session resumes from a scheduled trigger and the workflow is gone: check
  `git status` for uncommitted agent work and commit it by area, then continue with wave 2.

## ~06:05 — wave 1 done, committed
- Presets (14 public-domain works + 3 CC0 samples), backend API (111 tests pass), on-device
  engine (server's own model as TF.js, compositor parity 0.0000), Salon Wall UI shell.
- Orchestrator fixes: Dockerfile now copies `data/`; style vectors re-generated (unchanged);
  audit statuses updated (C1, M5, L5, L6, L7).
- Carried into wave 2: shader-compile freezes the main thread in headless (move engine off
  the main thread or use parallel compile), config.js script tag, remove adapter fallback,
  no-cache for unversioned ES modules + long cache for /app/models, static pages (privacy,
  terms, credits, per-style), OG image, README, CI + e2e tests.

## ~06:10 — wave 2 launched
- Workflow `wf_1016e4e3-e30` (task w013as0s5): integration (port 8210: config.js, remove
  adapter fallback, engine off the main thread, cache/gzip headers, icons/manifest, JS trim),
  content & SEO (8220: privacy/terms/credits, per-style pages via scripts/build_pages.py, OG
  image, README, capture.js), tests & CI (8230: tests/e2e, .github/workflows/ci.yml,
  TESTING.md), then an independent reviewer per area. UI-shell PNG screenshots in
  docs/revamp/after/ui-shell/ are intentionally uncommitted (19 MB); final ones come in wave 3.

## ~07:35 — wave 2 done, committed; wave 3 launched
- Integration: engine runs in a module Web Worker (OffscreenCanvas); main-thread long tasks on
  first warm-up 9.0 s → 0.17 s on the real GPU, Cancel < 150 ms. TF.js self-hosted (same SRI),
  cache/gzip rules, gilt-frame icons + manifest. 127 backend tests pass.
- Content: 19 static pages (collection, 14 style pages, privacy, terms, credits, 404), OG image,
  README. Tests & CI: tests/e2e (25 pass + 1 slow), .github/workflows/ci.yml, TESTING.md.
- Added .gitattributes (-text for vendor/models) so SRI hashes survive Windows checkouts.
- Wave 3: fix reviewer findings (engine lazy-import fallback, prefetch guard, privacy wording,
  CI memory-swap, test assertions, collection CSS / sticky footer, docs), then QA fan-out →
  verify → fix by area → final screenshots in docs/revamp/after/final/.

## ~10:50 — wave 3 done, committed; wave 4 launched
- QA fan-out: 8 lenses (phone, tablet, a11y, edge-case flows, perf, security, code, polish),
  79 findings → 78 unique → 76 verified and fixed by area; 2 rejected.
- Highlights: upload pixel budgets + body cap + queue slot before reading (bomb went from
  +567 MB to 413 / +0 MB), CSP narrowed to the pinned MediaPipe path, transfer results no
  longer stored, CI actions SHA-pinned, 44 px touch targets, in-flow veil/error cards,
  mid-paint race fixes, focus management, friendly storage errors.
- Final run: backend 153 passed; e2e 34 passed (+2 slow, skipped by default); pages/assets
  --check ok; ruff ok; parity 0.0000; on-device engine painted at 1440 in 19.7 s (real GPU).
- Orchestrator caught a regression risk: uvicorn no longer trusts proxy headers from any
  peer, so Starlette's slash redirects would send https visitors to http:// for one hop.
- Wave 4 (`wf_2fc442d8-8ff`): relative redirects, privacy text (results not stored), style
  page layout, panel scroll hint, self-hosted fonts, 1024 px HiDPI style images, docs.

## ~11:45 — wave 4 done; Phase 6 build complete
- Redirects are path-only (no https→http hop behind Render's proxy); `/` → `/app/` in one
  hop. Plain-language no-match error. Dev harness kept out of the Docker image. CI reports
  container peak memory.
- Fonts self-hosted (OFL); app CSP is now `'self'` for styles and fonts. Style pages
  composed in two columns; HiDPI 1024 px style images; gallery "Scroll for more" hint;
  privacy text matches the no-storage server.
- Final: backend 167 passed; e2e 44 passed (+2 slow); pages/assets --check ok; ruff ok;
  parity 0.0000. Final screenshots in docs/revamp/after/final/.
- Not verified here: Docker build and CI on GitHub, Render memory, Safari/Firefox/real
  phones, a real Workers AI call.

## Day 2 (owner testing), 2026-11-16 ~12:00-16:05
The owner tested on a 1920×1080 laptop at 125 % Windows scaling (1536×690 viewport) and
on a phone. Issues reported and the fixes, by commit subject:
- Mix overlapped the "Your painting" and "Describe it" tabs on a short screen:
  `c74df03` Stop Your painting and Describe it panels overlapping Mix on short screens.
- The frame sat right of the intro text and wall label: `2513b77` Align the studio frame
  with the intro text and wall label.
- A new logo: `45c64a5` New brand mark: moonlit brush stroke in a gilt frame; `160f710`
  Richer brand mark, P monogram favicon and a Pictura/AI lockup (`assets/brand/mark.svg`,
  `monogram.svg`, favicon, apple-touch, manifest and maskable icons, OG image);
  `4df53a5` Tidy the brand SVG header comments.
- The before/after slider was not obvious, then should not cover half the painting:
  `6e21ecd` Make the before/after slider obvious; `c5553bd` Compare slider: keep the
  painting whole; edge grip, hint chip and a one-time peek; `5281e05` Hide the compare
  rule at rest so it doesn't read as a stray white border.
- Dragging the intensity slider scrolled half the picture out of view: `62d5e51` Put the
  intensity slider on the frame's ledge, in view with the picture.
- The intensity slider froze after release and jumped on a second grab; the download
  menu stayed put while the page scrolled: `c6c48ae` Keep the intensity slider
  responsive; make the download menu follow its button (sharpened render deferred
  350 ms, display-size live compositor, `render({ into })`).
- A maker credit: `d60e5d5` Footer: Made by Shashank Gupta.
- Bug hunt (several agents on desktop and phone sizes), layout fixes: `5d604bf` Roving
  radios: scroll the newly focused item fully into view; `ec1b84a` Deep links to
  #gallery and #how-it-works land on their section; `3def180` Short-screen layout fixes
  in the studio; `4a2b97c` Style pages: keep the notes inside the artwork section on
  short wide screens.
- The style panel did not fit short laptop screens, and phone toasts covered the
  intensity slider: `f0cdd72` Fit the style panel on short laptop screens; move phone
  toasts off the slider (compact three-column gallery under 820 px tall, sticky Paint
  it / Download, "More below" pill; toasts under the header on phones, in the header
  bar on desktop); `f6e4d5e` Download menu: treat the sticky header as out of view.
- Tests: `61783f0` Add short-screen and phone toast regression tests
  (`tests/e2e/test_short_screens.py`, 4 tests).
- At the owner's request, commit messages on this branch carry no tool attribution
  lines and are committed under the owner's name only. Three older commits already on
  GitHub's `main` (`3d40053`, `fc2267f`, `f2ef7ba`) still mention the tool; rewriting
  `main` again is waiting for the owner's decision.
- Final: backend 167 passed (80 s); e2e 48 passed (+2 slow, skipped) in 172 s on port
  8951. Nothing pushed.
