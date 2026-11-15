# PicturaAI: Revamp Proposal

Written 2026-11-16 on branch `revamp/2026-11-16`, after the audit and fixes in `AUDIT.md`.
Nothing here is built yet. Pick what you want and I'll build it in Phase 6.

Free-tier figures were looked up today. Where sources disagreed or I couldn't reach the
official page, I say so. **Verify those numbers on the vendor's pricing page before relying on them.**

---

## 1. Product

**Who it's for.** People who want a photo to look like a painting and don't want to learn an
editor. That means people making a gift or a print of a pet, a partner or a trip, and social
posters who want an "art version" of a photo. A smaller group is small shops and creators who
want a consistent painted look across product or event photos.

**The job.** "Turn this photo into a painting I'd be happy to print or post, in under a
minute, without signing up."

**The one thing worth paying for: print-quality output.** Today every result is capped at
**320 px** on its longest side (`NST_MAX_DIM=320`) because Render's free tier has 512 MB of
RAM. That's fine for a thumbnail and useless for a print, a phone wallpaper or even an
Instagram post (1080 px). Every competitor below charges for "HD". Nothing else on the list
matters as much as fixing this. The good news is that it can be fixed for free (see
feature F1).

### Comparable products

| Product | What it does better | Where PicturaAI can beat it |
|---|---|---|
| **Prisma** (iOS/Android) | 500+ styles; polished mobile app; portrait segmentation and background replacement. HD saving is a premium feature, reportedly via weekly, monthly or yearly plans (prices I found conflict: about $7.99/mo or $24.99–29.99/yr, not verified on Prisma's site). | No install and no signup on the web. **Bring your own style** (upload any painting). **Region control** (mask, mix two styles), which Prisma doesn't offer as far as I found. |
| **NightCafe** (web) | Big community, accounts and galleries; style transfer is one of several generators. Credits system: about 5 free credits/day, paid from about $6/mo (sources conflict). | No credits to manage; results in seconds rather than a queue; a focused single-purpose flow instead of a generator buffet. |
| **Fotor** (web editor) | A full photo editor around the effects (crop, retouch, templates). Free exports are watermarked and 72 dpi; Pro is about $8.99/mo or $39.99/yr (third-party sources). | No watermark on free results; **private by design** if inference moves into the browser (F1): "your photo never leaves your device" is a claim none of the three make. |

**Where the gap is:** none of them combine *your own style image + painting only part of the
photo + private, no-signup, no-watermark*. That's a defensible niche for a small product.

---

## 2. Design: two directions

The current UI is the default "AI app" look: near-black background, purple-to-indigo
gradients, glowing orbs, a gradient-text headline, emoji icons and identical rounded cards.
Nothing in it says "painting". Both directions below take their look from where paintings
actually live.

### Direction A: "Salon Wall" (recommended)

The app is a gallery. Your photo hangs on a museum wall in a gilt frame, and each style is
presented with a real wall label: title, artist and year (we already have this data).

| Token | Value | Use |
|---|---|---|
| `wall` | `#1D3530` | Page background: deep gallery green, the colour of old-master rooms |
| `wall-deep` | `#142622` | Panels, studio surface |
| `linen` | `#ECE5D6` | Primary text on wall (10.4:1) |
| `gilt` | `#C9A44C` | Selection, focus ring, primary button (5.6:1 on wall; ink on gilt 6.7:1) |
| `label` | `#F3EEE3` | Wall-label cards with `ink #26221D` text (13.7:1) |
| `vermilion` | `#E0613F` | Error marker and icons only (3.7:1, so never for body text) |

- **Type:** *Libre Caslon Display* for titles and the result's "wall label" (a catalogue
  serif, used large and only for display), with *Schibsted Grotesk* for all UI text and
  controls. Scale: 14 / 16 / 20 / 28 / 44 / 68, line-height 1.5 for body and 1.1 for display.
  Sentence case everywhere; no all-caps eyebrows.
- **Layout:** the studio *is* the first screen, not a marketing hero. On desktop the photo sits
  framed on the left wall, styles hang salon-style on the right (mixed sizes, each with its
  label), and controls sit beneath the frame like a plinth. On mobile it's one column: frame,
  then a horizontally scrolling rail of styles, then a sticky "Paint it" bar.

  ```
  ┌──────────────────────────────────────────────────────────┐
  │ PicturaAI                          How it works  Gallery │
  ├──────────────────────────────┬───────────────────────────┤
  │  ╔════════════════════════╗  │  ┌────┐ ┌──────┐ ┌───┐    │
  │  ║                        ║  │  │    │ │      │ │   │    │
  │  ║   your photo / result  ║  │  └────┘ └──────┘ └───┘    │
  │  ║   (before/after drag)  ║  │  The Starry Night         │
  │  ╚════════════════════════╝  │  Vincent van Gogh, 1889   │
  │  ┌ label ───────────────┐    │  ┌──────┐ ┌────┐ ┌────┐   │
  │  │ After The Great Wave │    │  └──────┘ └────┘ └────┘   │
  │  │ intensity 75%        │    │  Upload a painting · Mix  │
  │  └──────────────────────┘    │                           │
  │  [ intensity ───●── ]  [ Paint it ]                      │
  └──────────────────────────────┴───────────────────────────┘
  ```
- **Motion:** one signature moment, the *unveiling*. When a result arrives it is revealed by
  a single brush-stroke wipe across the frame (about 600 ms) and the wall label types in its
  title. Everything else is instant feedback only (selection, slider). With reduced motion
  it becomes a 150 ms cross-fade.
- **Why it suits the audience:** gift-makers and casual creators are choosing *a famous
  painting's look*. Presenting styles the way a museum does (title, artist, date) makes the
  choice feel meaningful and trustworthy, and no competitor looks like this.

### Direction B: "Pigment Bench"

The app is a painter's workbench: a light, tool-first workspace where each mode is a
pigment.

| Token | Value | Use |
|---|---|---|
| `plaster` | `#DCDDD8` | Background: cool studio plaster |
| `ink` | `#1F2328` | Text (11.6:1) |
| `graphite` | `#565B63` | Secondary text (5.0:1) |
| `ultramarine` | `#2A3D9B` | "Full style" mode, primary actions (6.9:1; white on it 9.4:1) |
| `cadmium-yellow` | `#F2B705` | "Colours only" mode (ink text on it 8.7:1) |
| `cadmium-red` | `#B3261E` | "Paint a region" mode and errors (4.8:1) |

- **Type:** *Archivo*, a single family used across its width axis: expanded and heavy for
  the few headings, normal for UI, condensed for dense control labels. The width change is
  the typographic personality.
- **Layout:** a full-viewport app with no scrolling page. A left rail holds the mode chips
  (pigment swatches), the centre is a large canvas, the right panel holds the style library,
  and a bottom filmstrip holds history (Lightroom-like). On mobile: canvas, then a bottom
  sheet with tabs for Style, Adjust and History.
- **Motion:** a pigment "drop" fills the canvas edge in the active mode's colour while
  generating; the mode chips morph when switched.
- **Fits best if** you see the audience as repeat users and small shops doing batches,
  rather than one-off gift-makers.

**Recommendation: A.** The product's distinct value is famous painterly styles plus your own
paintings, and the audience is mostly one-off, emotional uses (gifts, prints, posts). Salon
Wall turns the style choice into the experience and immediately separates PicturaAI from
the dark-purple AI-app crowd. I'd borrow one idea from B: an explicit **mode switch**
(Full style / Colours only / Paint a region) instead of today's scattered toggle.

---

## 3. UX improvements by screen

| Screen | Problem today | Change |
|---|---|---|
| **Landing / hero** | A marketing hero with contradictory copy ("Real-time AI art, in minutes"), filler stats ("13+ · AI · Real-time") and a demo visual that is hidden at every width. The studio starts two screens down. | Open straight into the studio, with a **sample photo already loaded** and a live before/after slider of a pre-rendered result. One button: "Use your own photo". |
| **Onboarding** | An empty upload box with no hint of what a good input looks like; a first visit after idle meets a ~1 minute silent wait. | Three sample photos (pet, portrait, city) one tap away. Warm the server on page load and show a small "Warming up the studio…" chip on the Generate button until `/api/styles` answers. |
| **Style picker** | Four competing inputs (preset grid, custom upload, text prompt, second style). The text prompt is silently ignored if a preset is selected. Every click fires a toast. | One picker with tabs **Gallery / Your painting / Describe it**. The active source is always visible; switching tabs replaces it explicitly. No toast on selection, since the selected state is the feedback. Show title, artist and year under the selected style. |
| **Mode & controls** | "Color Palette Only" is a toggle buried below the intensity slider; mask painting hides under the photo. | A segmented control: *Full style · Colours only · Paint a region*. The intensity slider updates the preview **live** (F2), so it no longer needs a re-generate. |
| **Generating** | A fake 4-phase progress bar with leftover "Loss / Step" labels from an old engine, for a request that has no real progress. | An honest indeterminate state: the frame dims, shows elapsed seconds, and after 8 s adds "The server was asleep. First run takes up to a minute." |
| **Result** | Result, before/after slider and history are stacked vertically, so on phones the result sits far below the button that made it. | The result replaces the photo *in the same frame* (before/after drag on the frame itself). Auto-scroll the frame into view on mobile. Actions: Download, Share (Web Share API on mobile), Animate. |
| **Empty states** | "Your Canvas Awaits" with decorative circles. | The sample photo plus a one-line instruction: "Pick a style on the right, then Paint it." |
| **Errors** | Fixed in Phase 4 (friendly messages, retry keeps inputs). | Add pre-upload checks (type, size, megapixels) with the reason shown on the upload box itself, not only in a toast. |
| **Mobile** | Long single column; the Generate button scrolls away; the nav hides all links. | A sticky bottom bar with the primary action; styles as a horizontal rail; larger touch targets (min 44 px) for the mask tools. |

---

## 4. Features

Ordered by user value. "Free tier" lists the service and its limits as found today.

### F1. Run style transfer in the browser (high-res, private, no cold start)
- **User benefit:** results at **1024–2048 px** instead of 320 px; no waiting for a
  sleeping server; "your photo never leaves your device".
- **Build:** use Magenta's arbitrary-stylization model in TensorFlow.js (the
  `@magenta/image` port, the same Ghiasi et al. architecture the server uses). Its published
  sizes are ~9.6 MB for the distilled style network and 2.4–7.9 MB for the transformer,
  cached after the first visit. Run it in a Web Worker with the WebGL/WebGPU backend, and keep
  the server as a fallback for devices without WebGL.
- **Free tier:** none needed, since it runs on the user's device. Hosting the model files on
  Cloudflare Pages (free: 500 builds/mo, unlimited bandwidth per Cloudflare's docs) or from
  the existing server.
- **When the quota runs out:** there is no quota. On weak or old devices it falls back to the
  current server path at 320 px, with a note.
- **Risk / to verify:** the browser model is an older, distilled checkpoint, so quality may
  differ from the TF Hub model the server uses. I'd prototype it side by side on 10 photos
  before committing. **This changes the architecture, so it needs your approval.**

### F2. Live intensity slider
- **User benefit:** drag the slider and see the result change instantly instead of
  re-generating.
- **Build:** the blend, detail re-injection and sharpening in `run_nst` are cheap pixel maths.
  Return (or compute, with F1) the raw stylised image once and do the blend on a `<canvas>`
  in the browser.
- **Free tier:** none needed. **Quota:** none.

### F3. One-tap "style the subject" / "style the background" (AI)
- **User benefit:** today regional styling means painting a mask by hand. Most people want
  "just the background" or "just the person or pet".
- **Build:** the MediaPipe Image Segmenter (`@mediapipe/tasks-vision`) runs in the browser
  (selfie or multiclass and DeepLab-v3 models) and produces the mask the server already
  accepts. Keep the brush to refine edges. One known issue: wrong labels on iOS Safari with
  the GPU delegate for the multiclass model, so use the CPU delegate there.
- **Free tier:** none needed (runs on the device). **Quota:** none.

### F4. "Describe it" that actually works (AI)
- **User benefit:** today "Text-to-Style" is keyword matching against 13 presets: "neon
  cyberpunk city" either fails or picks something unrelated. Generating a style image from
  the description makes the text box do what it promises.
- **Build:** a small Cloudflare Worker calls Workers AI (a Flux model) to generate a
  256–512 px swatch from the prompt, then uses it as the style image. Rate-limit per visitor.
- **Free tier:** Workers AI includes **10,000 Neurons/day** free on the Workers Free plan.
  Image cost depends on the model. Cloudflare's docs list `flux-2-klein-9b` at ~1,364
  neurons per first megapixel (≈7 images/day). A third-party guide quotes `flux-1-schnell`
  at ~58 neurons per image (≈170/day), which I could not confirm officially.
- **When the quota runs out:** fall back to today's keyword matcher and say so: "Today's
  describe limit is used up. We matched the closest classic style instead."
- **Honest caveat:** at 7–170 generations a day across *all* users, this is a taste, not a
  core feature, unless you pay ($0.011 per 1,000 neurons above the free amount on Workers Paid).

### F5. Saved gallery without accounts
- **User benefit:** history survives a page refresh; today it's lost on reload.
- **Build:** store results in IndexedDB (in the browser) with a "Clear my gallery" button.
- **Free tier:** none needed.

### F6. Share and print sizes
- **User benefit:** a one-tap share on phones, plus downloads at Instagram (1080 px square or
  4:5) and print (A4 at 150 dpi ≈ 1754×1240) sizes.
- **Build:** Web Share API, and canvas export at preset crops. Needs F1 for real resolution.
- **Free tier:** none needed.

### F7. Style animation as video
- **User benefit:** today's GIF is 320 px and banded; MP4/WebM is smaller and shares better.
- **Build:** encode frames with `MediaRecorder` from a canvas in the browser (with F1/F2).
- **Free tier:** none needed.

**Considered and rejected:** "AI suggests styles for your photo" via an LLM vision API. The
Gemini free tier exists, but its limits are no longer published per model (sources range from
20 to 1,500 requests/day), and a simple colour-similarity ranking gives 80% of the value
with no quota. Build the non-AI version if you want suggestions.

---

## 5. Product readiness

| Area | Today | Needed before real users rely on it | Free option (verify limits) |
|---|---|---|---|
| **Security** | A Render API key sits in public git history (C1). | **Revoke it** first. Then add a CSP once inline `onclick`s are gone (Phase 6 redesign removes them). | — |
| **Hosting** | Render free: 512 MB RAM, sleeps after 15 min idle, ~1 min wake, 750 free instance-hours/month per workspace. One source claims the free tier was dropped in March 2026, but newer ones say it still exists. | With F1 the server is optional. Otherwise keep it warm with a 5-minute ping (one always-on service uses ~720–744 h, inside 750 h). | UptimeRobot free: 50 monitors, 5-minute checks; its help centre says commercial use is allowed. |
| **Static frontend** | Served by FastAPI. | Optional split: frontend on a CDN. **Note:** Vercel's Hobby plan is for non-commercial use as far as I know, so for a product you sell, use Cloudflare Pages. | Cloudflare Pages: 500 builds/mo, unlimited bandwidth. |
| **Error monitoring** | Server logs only. | Frontend and backend error reporting. | Sentry Developer plan: 5,000 errors/month (most sources agree). |
| **Analytics** | None. | Page views and generate/download counts, cookie-free. | Cloudflare Web Analytics, which I believe is free but couldn't confirm today. |
| **SEO / sharing** | `og:image` is a relative path (`assets/logo.png`), which social sites can't resolve; no sitemap; one generic title. | Absolute OG image (a before/after sample), `robots.txt`, `sitemap.xml`, per-style pages ("Photo to Van Gogh painting") that can rank. | — |
| **Landing page** | Merged into the app. | Keep it in the app (studio first), with a short "How it works" and an FAQ below. | — |
| **Legal** | None. | **Privacy policy** (uploads kept up to 1 h on the server today, or never with F1) and **terms**. **Style licensing:** Van Gogh, Munch, Hokusai and Turner are public domain. **Picasso's *La Muse* is still under copyright** (Picasso died 1973). Afremov's *Rain Princess* (artist died 2019) is very likely protected, and the "Hampson", "Space", "Mountain" and "Paris" images have unknown sources. Before charging money, replace or license those presets. | — |
| **Tests** | 13 backend tests (added in Phase 4); none for the frontend. | Playwright smoke test of the main flow (upload → style → result → download). | — |
| **CI** | None. | GitHub Actions running pytest and the Playwright smoke test on every PR. | GitHub Actions is free for public repositories (standard runners). |
| **Backups** | Nothing to back up (no database). | Only needed if accounts arrive; Supabase Free (500 MB DB, 1 GB files, 50k MAU, pauses after 7 days idle) doesn't include backups as far as I know, so verify that. | — |
| **Payments** | None. | Only after F1 + licensing are done. Lemon Squeezy (5% + $0.50 per sale, merchant of record, so it handles VAT/sales tax) or Stripe (2.9% + $0.30, but you handle tax). Neither has a monthly fee. | — |
| **Runtime** | Python 3.10 (end-of-life Oct 2026). | Python 3.12 to receive remaining Pillow/keras security fixes. | — |

---

## 6. Roadmap (impact ÷ effort)

| # | Item | Impact | Effort | Size |
|---|---|---|---|---|
| 1 | **Revoke the leaked Render key** (you) | Critical | Minutes | S |
| 2 | Replace or license copyrighted presets (*La Muse*, *Rain Princess*, unknown-source images) | High (legal) | Low | S |
| 3 | Fix SEO basics: absolute OG image, `robots.txt`, sitemap, real title/description | Medium | Low | S |
| 4 | UX quick wins without a redesign: remove selection toasts, honest progress, auto-scroll to the result, sample photos | High | Low | S |
| 5 | F2 live intensity slider | High | Low–medium | S |
| 6 | F5 gallery that survives refresh (IndexedDB) | Medium | Low | S |
| 7 | CI with GitHub Actions + a Playwright smoke test; Sentry | Medium | Low | S |
| 8 | Python 3.12 runtime + Pillow 13 (needs your approval; test on 512 MB) | Medium (security) | Medium | M |
| 9 | **Design direction A (Salon Wall)**: tokens first, then studio, picker, result, errors and mobile | High | Medium–high | L |
| 10 | F3 one-tap subject/background mask (MediaPipe, on-device) | High | Medium | M |
| 11 | **F1 on-device style transfer** (prototype quality first; architecture change) | Very high | High | L |
| 12 | F6 share and print sizes (after F1) | High | Low | S |
| 13 | F7 animation as video | Low–medium | Medium | M |
| 14 | F4 "Describe it" with Workers AI (small daily quota) | Medium | Medium | M |
| 15 | Accounts, payments, print-on-demand | Only once 1–12 are done | High | L |

**My suggested first build batch:** 2, 3, 4, 5, 6, 7, then 9 (the redesign), with a time-boxed
prototype of 11 in parallel. F1 is what turns this from a demo into something people would
pay for, and its result decides how the redesign handles resolution and download sizes.

---

### Sources (looked up 2026-11-16)
- Render free tier: [Render, free tiers 2026](https://render.com/articles/platforms-with-a-real-free-tier-for-developers-in-2026), [costbench](https://costbench.com/software/developer-tools/render/free-plan), [livemy.app](https://livemy.app/blog/render-pricing), [DEV review](https://dev.to/build996/render-free-hosting-review-2026-deploy-web-apps-databases-and-cron-jobs-for-free-b39)
- Render / Cloudflare client IP: [Render feedback on X-Forwarded-For](https://feedback.render.com/features/p/send-the-correct-xforwardedfor), [Render community](https://community.render.com/t/accessing-client-ips-in-a-node-express-app/36282), [Cloudflare request headers](https://developers.cloudflare.com/fundamentals/reference/http-request-headers)
- Hugging Face Spaces (considered, uncertain in 2026): [HF docs](https://huggingface.co/docs/hub/main/spaces-gpus), [forum complaint](https://discuss.huggingface.co/t/official-community-complaint-revert-free-cpu-basic-spaces-and-remove-anti-developer-sdk-restrictions/177703)
- Workers AI: [Cloudflare pricing docs](https://developers.cloudflare.com/workers-ai/platform/pricing/), [third-party Flux estimate](https://learnwithhasan.com/guide/free-ai-images-claude-code-cloudflare/)
- Gemini free tier: [pecollective guide](https://pecollective-com.pages.dev/tools/gemini-free-tier-guide/), [agentdeals](https://agentdeals.dev/gemini-api-pricing-changes)
- Supabase: [toolradar](https://toolradar.com/tools/supabase/pricing), [Supabase changelog](https://supabase.com/changelog?types=policy)
- Sentry: [costbench](https://costbench.com/software/developer-tools/sentry/free-plan/), [observability-pricing](https://observability-pricing.pages.dev/vendor/sentry/)
- UptimeRobot: [help centre](https://help.uptimerobot.com/en/articles/11604710-who-is-uptimerobot-s-free-plan-for), [pricing](https://uptimerobot.com/pricing/?orderBy=name0)
- Cloudflare Pages: [limits](https://developers.cloudflare.com/pages/platform/limits), [pages.cloudflare.com](https://pages.cloudflare.com/)
- Payments: [Lemon Squeezy vs Stripe](https://dodopayments.com/blogs/lemon-squeezy-vs-stripe/), [merchant of record 2026](https://alexcloudstar.com/md/blog/merchant-of-record-indie-saas-2026/)
- On-device models: [Magenta style-transfer-js](https://magenta.tensorflow.org/blog/2018/12/20/style-transfer-js/), [reiinakano port](https://awesome.ecosyste.ms/projects/github.com%2Freiinakano%2Farbitrary-image-stylization-tfjs), [MediaPipe tasks-vision](https://www.npmjs.com/package/@mediapipe/tasks-vision), [MediaPipe iOS issue](https://github.com/google-ai-edge/mediapipe/issues/6142)
- Competitors: [Prisma App Store](https://apps.apple.com/app/prisma-art-filters-photo-effects/id1122649984), [shotkit Prisma guide](https://shotkit.com/prisma-app/), [NightCafe pricing](https://toolradar.com/tools/nightcafe/pricing), [costbench NightCafe](https://costbench.com/software/ai-image-generators/nightcafe/), [Fotor on Capterra](https://www.capterra.com/p/181589/Fotor/pricing/), [Fotor pricing tiers](https://flowith.io/blog/fotor-pricing-2026-free-vs-pro-vs-pro-plus)
