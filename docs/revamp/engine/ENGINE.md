# On-device engine (PROPOSAL F1, F2, F3)

**Decision:** run the server's own model in the browser. The style network of Magenta's
*arbitrary-image-stylization-v1-256* (TF Hub, the model `backend/nst_engine.py` loads) was
converted to TF.js with uint8 weights and split at the style parameters. Presets ship
precomputed parameters, so a first visit downloads **7.8 MB** (TF.js 0.9 MB + transformer
6.9 MB) and the result matches the server. The smaller Magenta.js models were rejected on
quality.

All numbers below were measured on 2026-11-16 on the build machine: Windows 11, AMD Ryzen
APU with integrated Radeon Vega graphics (`gcn-5`), Playwright 1.65's headless Chromium
through ANGLE/Direct3D 11. That is a **low-end GPU**; a discrete or Apple-silicon GPU is
several times faster. WebGPU could not be measured (the headless build lacks
`dxcompiler.dll`), so the engine uses WebGL2.

## 1. Options compared

Three CC0 photos (`frontend/assets/samples/{dog,city,lake}.jpg`) x four styles
(Starry Night, The Scream, Udnie, Shipwreck of the Minotaur, as they were in
`images/style_image/` before the presets refresh), raw stylization at 768 px long side.
"Difference" is the mean absolute pixel difference (0-255) from the server's float32
TensorFlow output for the same inputs.

| Option | Download for a preset | Difference vs server (mean, min-max) | Verdict |
|---|---|---|---|
| Magenta.js distilled MobileNet + separable transformer (`@magenta/image` 0.2.1 / reiinakano) | 12.2 MB | 29.6 (23.3-37.8) | wrong palette, blotchy, swirls lost |
| Magenta.js distilled MobileNet + full transformer | 17.7 MB | 27.9 (20.8-34.8) | same problems |
| Server model, float16 weights | 13.0 MB transformer (+34.4 MB style net for custom styles) | 2.8 (1.7-5.0) | identical to the eye |
| **Server model, uint8 weights (chosen)** | **6.9 MB** transformer (+17.5 MB style net for custom styles) | **5.9 (4.7-7.7)** | identical to the eye, even in 2x crops |

Side by side: `compare-dog.jpg`, `compare-city.jpg`, `compare-lake.jpg` (columns: server,
float16, uint8, Magenta.js separable), and `compare-crops.jpg` (2x crops). The Magenta.js
checkpoints are a different, smaller network (5 residual blocks, 100-d style bottleneck,
distilled style predictor) than the server's (20 residual blocks, 768-d Inception
bottleneck), which is why they look so different.

About the Magenta.js checkpoints: `@magenta/image` points at
`storage.googleapis.com/magentadata/js/checkpoints/style/arbitrary/{predictor,transformer}`
in the TF.js 0.x format (`tensorflowjs_model.pb` + `weights_manifest.json`), which
`tf.loadGraphModel` in TF.js 4.x cannot read. The same weights exist as TF.js 1.x
`model.json` files in `github.com/reiinakano/arbitrary-image-stylization-tfjs`
(`saved_model_style_js` 9.5 MB, `saved_model_transformer_js` 7.8 MB,
`saved_model_transformer_separable_js` 2.3 MB), which do load; those were benchmarked.

## 2. How the model was converted

`docs/revamp/engine/tools/convert_model.py` (run in a throwaway venv with
`tensorflowjs==4.22.0`): freeze the SavedModel, replace the debug `CheckNumerics` ops,
inline the functions with Grappler, then split the graph where the 768-d Inception
bottleneck is projected into the 86 gamma/beta vectors of the conditional instance norms:

- `style/`: `style_image` [1,256,256,3] -> `style_params` [1,10438] (Inception-v3 +
  projections, 17.5 MB uint8). Only downloaded when someone uses their own painting.
- `transformer/`: `content_image` [1,H,W,3] + `style_params` -> `stylized` (6.9 MB uint8).

The split was checked against the original SavedModel: max abs difference 4.3e-6.
Projections are affine, so mixing two styles by interpolating their parameters equals
interpolating bottlenecks (what the paper does). Preset parameters are computed in
float32 from the original SavedModel by `scripts/precompute_style_vectors.py`, stored as
float16 (`frontend/models/style-vectors/<key>.bin`, 20.9 KB each, index in
`frontend/models/style-vectors.json`); float16 storage changes the output by at most
0.0011 (mean 7e-5). Rerun the script whenever `data/styles.json` or a style image changes.

## 3. Making it run at 768-1024 px in WebGL

Two problems showed up and are fixed in the engine:

1. **GPU limits.** TF.js lowers each convolution to one im2col matrix. At 768 px the
   final 9x9 convolution's matrix needs ~4 GB, so WebGL failed with "Texture total
   allocation size is too large" and lost the context; at 1024 px TF.js asked for a
   21304x21305 texture. `tiled-conv.js` wraps the WebGL `Conv2D`/`FusedConv2D` kernels and
   computes large `VALID` convolutions in horizontal bands (slice input rows, convolve,
   concatenate, flush between bands). The result is bit-identical (all of this model's
   convolutions are `VALID` after explicit mirror padding) and 1024 px now works
   (`browser-1024-city.jpg`). Disabling im2col instead made each draw so long that
   Windows reset the GPU.
2. **Shader compilation per input shape.** WebGL compiles a shader set for each new input
   size (+17-28 s per new size on this GPU; `WEBGL_USE_SHAPES_UNIFORMS` did not help). The
   engine (a) snaps the inference size to a few aspect-ratio buckets (1, 5:4, sqrt(2) for
   both 4:3 and 3:2, 16:9, 9:4; at most 8% stretch, undone when the result is resized to
   the photo's true aspect), and (b) precompiles the 4:3/3:2 bucket during `warmup()` with
   TF.js's `ENGINE_COMPILE_ONLY` + `KHR_parallel_shader_compile`. Chrome keeps compiled
   programs: a second warm-up in the same browser took 0.34 s instead of 9.3 s.

## 4. Measured performance (this low-end GPU)

| Long side | Inference | First run of a new shape (without precompile) |
|---|---|---|
| 512 (342 x 512 content) | 3.4-3.7 s | 14-17 s |
| 768 | 8.3-8.7 s | 25-38 s |
| 1024 | 15.5-16.1 s | 44-51 s |

For reference, Magenta.js separable ran 0.84 s at 512 and 2.1 s at 768 on the same GPU
and lost the WebGL context at 1024; the server model on this CPU (TensorFlow, 4 threads)
takes 19-27 s at 768 and ~2.7 s at the free tier's 320 px.

- **Warm-up** (first visit): 7.8 MB download (0.4-3.2 s on localhost), shader compile
  9.3-9.7 s, one 512 px timing probe 3.6 s: ~14 s total, run on idle or on first "Paint
  it" with real byte progress. Later visits: models from Cache Storage (one cache per
  manifest version; model URLs carry `?v=<manifest version>`, so the server caches them
  for a year), side remembered, compile cached by the browser.
- **Inference size policy:** `maxInferSide` = the largest of 1024 / 768 / 512 whose
  predicted time (probe x pixel ratio) fits 6 s, capped at 1024 on desktop and 768 on
  phones/tablets (`pointer: coarse` or mobile UA) or when `navigator.deviceMemory <= 4`.
  This machine gets 512 (still 2.5x the server's pixels). The choice is stored in
  `localStorage` (`pictura.engine.side`); `createBrowserEngine({ maxInferSide })` forces it.
- **Memory:** `tf.memory().numTensors` stayed at 590 (the model weights, 24.4 MB) after
  each of 5 stylizations: no tensor leak. GPU memory peaks around 0.5 GB at 768-1024 px.
- **Compositor (live slider)** at 1200 x 801: `render({ sharpen: false })` 25-41 ms in
  Chromium (26-37 ms in Node), `sharpen: true` 158-278 ms, precompute 180-280 ms.
  UI (`ui/studio.js`): live frames (slider drag, brushing) render with `sharpen: false`
  into a reused buffer (`into`), from a second, **display-size** compositor when the frame
  shows the painting at under 85% of its size (photo and stylization resized to the
  canvas's on-screen width x `devicePixelRatio`, built on idle once the result is shown;
  the full-size compositor is used until it exists or if building it fails). The canvas
  shrinks to that size while dragging and CSS scales it. The full-size `sharpen: true`
  render waits until the slider has been still for 350 ms (`FINAL_DELAY_MS`); a new press
  or input cancels it. Running it on release blocked the main thread for 110-310 ms and
  froze the thumb when the slider was grabbed again straight away.
- **Segmentation** (DeepLab-v3): first measured on the GPU delegate (1.5 s to load on
  localhost, 3.5 s for the first 1600 px photo; found `["dog"]` on the dog sample, 18.8%
  subject). The GPU delegate's first run compiles shaders on the main thread (a ~4 s
  freeze here, ~8 s with software GL), so `segment.js` now uses the **CPU delegate** by
  default (~1.3 s) and segments a copy downscaled to at most **768 px** (DeepLab works at
  257 px). MediaPipe also downloads its 11.2 MB wasm from jsDelivr on first use.

## 5. Off the main thread (Web Worker)

WebGL shader compiles and GPU waits ran on the page's main thread, so the first warm-up
froze scrolling and Cancel for seconds (minutes with software GL). The browser engine now
runs in a module Web Worker:

- `worker.js` runs `createBrowserEngine` with TF.js WebGL on an `OffscreenCanvas`.
  `worker-engine.js` (`createWorkerEngine`) is the page-side proxy with the same API; it
  sends photos as `ImageBitmap` copies and receives the raw stylization as a transferred
  buffer. It sets `engine.offMainThread = true`.
- In the worker TF.js is loaded with a dynamic `import()` of the same self-hosted files
  (`importScripts()` is not available in module workers); on the page they are classic
  scripts with SRI (section 8).
- **Fallback to the in-page engine** (`browser-engine.js`, imported only then;
  `offMainThread` becomes `false`) when the worker cannot start or does not say hello
  within 20 s, when there is no `OffscreenCanvas` WebGL2 (Safari before 17), or when
  TF.js fails in the worker (`ENGINE_WORKER`). `warmup(onProgress, { allowInPage: false })`
  (the studio's idle prefetch) never falls back, so a prefetch cannot freeze the page: it
  rejects with `ENGINE_WORKER` instead, and a later `warmup()` (default
  `allowInPage: true`, e.g. on "Paint it") can still fall back. If `browser-engine.js`
  itself cannot be imported for that fallback, the proxy rejects with `ENGINE_UNAVAILABLE`
  and the studio paints on the server.
  A crashed worker (usually out of memory) rejects with `ENGINE_RESOURCE` and sets
  `pictura.engine.browserFailedAt`, like a lost context.
- **Cancel** rejects at once with an `AbortError`; the worker finishes its current GPU pass
  and drops the result.
- `?worker=0` (or `createEngine({ worker: false })`) keeps the browser engine on the page,
  for comparison and debugging.

Measured during the integration work on this machine (main-thread long tasks during the
first warm-up; ranges are over repeated runs):

| Setup | Before (in page) | After (worker) |
|---|---|---|
| Real GPU | 9,006 ms of long tasks, longest 3,394 ms | 173-246 ms in total, longest 105-132 ms |
| Software GL (headless Chromium) | 315,548 ms of long tasks | 1,465-1,986 ms in total |

Cancel while painting in the worker answered in 18-134 ms.

## 6. Compositor parity (F2)

`composite.js` is a line-for-line port of `run_nst` phases 3-4 and
`color_palette_transfer`, including Pillow's exact extended-box Gaussian blur and its
integer UnsharpMask (`|diff| > threshold`, C integer division). Test:
`venv/Scripts/python.exe docs/revamp/engine/parity/composite_parity.py` runs the Python
pipeline (using `nst_engine`'s own helpers) and the JS port (through Node) on the same
inputs. Result: mean absolute difference **0.0000** (rounded to 4 places) in all 21 cases
(intensity 0/25/50/75/100% at 320 and 640 px, no-sharpen, mask, 1200 px, three palette
cases, three blur radii); max 1-3 levels on a handful of pixels from float rounding.
The mask blur radius scales with size: `6 * longSide / 320` (6 px at the server's 320 px).

## 7. API (BUILD-PLAN section 3) and extras

- `index.js`: `createEngine({ prefer, worker, warmup, onProgress, browser, server })`,
  `preferredEngineKind()`, and re-exports `EngineError`, `createServerEngine`,
  `createWorkerEngine` and `hasWebGL2`. It no longer re-exports `createBrowserEngine`:
  `browser-engine.js` is imported on demand (by the worker, or for the in-page fallback),
  so loading `index.js` does not pull in the engine. `?engine=browser|server` overrides
  the choice; `?worker=0` keeps the browser engine on the page. The browser engine is
  chosen when a page canvas gives WebGL2 with float render targets and the browser engine
  has not failed on this device in the last 7 days. If the in-page engine's code cannot
  load (it is used directly with `?worker=0`, `worker: false`, or when `workerCapable()`
  finds no `Worker` or no `OffscreenCanvas` WebGL2), `createEngine()` rejects with
  `ENGINE_UNAVAILABLE` (not cached; the next call tries again) unless `warmup: true`,
  which falls back to the server engine. Callers that do not pass `warmup: true` must
  catch it and call `createEngine({ prefer: 'server' })`.
- Engine: `kind`, `label`, `maxInferSide` (final after `warmup()`),
  `warmup(onProgress, { allowInPage })` (`allowInPage` exists on the worker proxy only,
  default `true`; see section 5),
  `stylize({ content, style, style2, mix, signal, onProgress })` returning the raw
  stylization at the photo's aspect with long side `maxInferSide`. Extras on the browser
  engine: `offMainThread`, `prepare({ width, height })` (precompile a photo's shape before
  "Paint it"), `timings`, `ready`, `dispose()`; `getStyleParams(style)` on the in-page
  engine only.
- Progress phases: `download` (bytes of TF.js + transformer), `download-style` (bytes of
  the style network, custom styles only), `compile`, `style`, `upload` (server), `paint`,
  `ready`, `done`. Every caller of a running `warmup()` gets the events.
- Errors: `EngineError` with `code` (see `errors.js`). `ENGINE_RESOURCE` (lost WebGL
  context, out of GPU memory) and `ENGINE_UNAVAILABLE` mean "switch to
  `createEngine({ prefer: 'server' })`". A lost context also sets
  `localStorage['pictura.engine.browserFailedAt']`, so the next visit starts on the server.
  Aborting rejects with a `DOMException` named `AbortError` (checked between steps; a
  running GPU pass cannot be interrupted).
- `server-engine.js`: `POST /api/transfer` with `output=raw`, `intensity=1`; the photo is
  uploaded as a JPEG at most 1024 px (custom styles at most 512 px); messages for
  413/429/502/503/504/offline; `maxInferSide` updates from the first response (320 on the
  free tier). Style mixing on the server blends style images, not parameters, so a mixed
  result differs slightly between engines.
- `composite.js`: `createCompositor`, `paletteTransfer`, `resizeImageData`, plus
  `pilGaussianBlurPlane`, `makeImageData` (the unsharp mask is internal to
  `render({ sharpen: true })`; the separate `unsharpMask` export was removed).
  `render({ intensity, mask, sharpen, into })`: `into` is an `ImageData` of the
  compositor's size to overwrite instead of allocating a new one (every pixel is
  rewritten; any other size is ignored and a new one is made). The mask may be any size;
  white = styled; alpha counts (a transparent brush canvas = unstyled); set
  `mask.version` (a number) when you edit a mask in place so the cached blur is redone.
- `segment.js`: `createSegmenter({ modelUrl, delegate })` -> `segment(image)` ->
  `{ subjectMask, foundLabels }`; CPU delegate by default (no main-thread shader compile;
  iOS Safari also returns wrong labels on the GPU, mediapipe#6142), input downscaled to
  768 px; `delegate: 'GPU'` falls back to CPU if it fails. "Style the background" = the
  inverted mask.

## 8. Browser support

- Browser engine: any browser with WebGL2 + `EXT_color_buffer_float` (or half float):
  Chrome/Edge 56+, Firefox 51+, Safari 15+ (iOS 15+). Verified in Chromium (desktop and
  Pixel 7 emulation) only; Safari/Firefox and real phones were not available here.
- Falls back to the server engine without WebGL2 (verified with WebGL disabled), after a
  context loss (verified with `WEBGL_lose_context`), or when the UI catches
  `ENGINE_RESOURCE`/`ENGINE_UNAVAILABLE`.
- Off the main thread: needs module workers and `OffscreenCanvas` WebGL2 (roughly
  Chrome/Edge 80+, Firefox 114+, Safari 17+); otherwise the same engine runs on the page.
  Only Chromium was verified here.
- CSP: works under the backend's policy (`script-src 'self' https://cdn.jsdelivr.net
  'wasm-unsafe-eval'; connect-src 'self' https://cdn.jsdelivr.net; worker-src 'self'
  blob:`), no console errors. TF.js 4.22.0 is **self-hosted** in
  `frontend/vendor/tfjs/4.22.0/` (unmodified `es2017` npm builds, same sha384 hashes as
  on jsDelivr; see the folder's `README.md`). On the page it is loaded as three classic
  scripts with SRI; in the worker by `import()` from our own origin. jsDelivr is now used
  only by MediaPipe (`segment.js`): its ES module is a dynamic `import()` of an exact
  version, which cannot carry SRI, and its wasm is fetched from there.
  `.gitattributes` marks `frontend/vendor/**` and `frontend/models/**` as `-text`, so a
  Windows checkout cannot rewrite line endings and break the SRI hashes or model bytes.

## 9. Limits

- Low-end GPUs get 512 px inference and ~4 s per painting after a ~14 s first warm-up.
  The compositor still renders at the photo's resolution (detail re-injection uses the
  full-resolution photo), so results stay sharp.
- A new aspect bucket costs a compile (~10 s here) the first time in a session; call
  `engine.prepare({ width, height })` right after a photo is chosen to hide it.
- Panoramas beyond 9:4 (more than 8% from a bucket) use their exact shape and compile.
- Custom styles download the 17.5 MB style network once (cached afterwards).

## 10. Licences

| Item | Licence | Where |
|---|---|---|
| Magenta arbitrary-image-stylization-v1-256 weights | Apache-2.0 | `frontend/models/arbitrary-stylization-v1-256/LICENSE`, `README.md` |
| DeepLab-v3 `.tflite` (MediaPipe) | Apache-2.0 (TensorFlow Models DeepLab; confirm against the model card before a commercial launch) | `frontend/models/deeplab_v3/` |
| TensorFlow.js 4.22.0 (core, webgl, converter) | Apache-2.0 | `frontend/vendor/tfjs/4.22.0/LICENSE` (self-hosted) |
| MediaPipe Tasks Vision 0.10.35 | Apache-2.0 | cdn.jsdelivr.net |
