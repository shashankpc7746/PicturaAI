// On-device style transfer with TensorFlow.js (WebGL).
//
// Model: the same Magenta "arbitrary-image-stylization-v1-256" network the
// server runs (Ghiasi et al. 2017), converted to TF.js and split in two:
//   style/        style image (256x256) -> style parameters [1, N]
//                 (Inception-v3 bottleneck + the per-layer gamma/beta projections)
//   transformer/  content image + style parameters -> stylized image
// Preset styles ship precomputed parameters (models/style-vectors.json), so the
// common path downloads only the transformer. The style network is fetched the
// first time someone uses their own painting.
//
// The parameters are affine in the style bottleneck, so mixing two styles by
// interpolating their parameters is exactly the same as interpolating bottlenecks.

import { installBandedConv } from './tiled-conv.js';
import { resizeImageData, makeImageData } from './composite.js';
import { EngineError } from './errors.js';

export { EngineError };

const TF_VERSION = '4.22.0';
const CDN = 'https://cdn.jsdelivr.net/npm';
const TF_SCRIPTS = [
  { src: `${CDN}/@tensorflow/tfjs-core@${TF_VERSION}/dist/tf-core.es2017.min.js`,
    integrity: 'sha384-pwFlZMvh0feP+i0lcqJy+EGMsa8y8MInwWkUgARHZEHUGyTNqxnqfi38fzMU994Z', bytes: 255317 },
  { src: `${CDN}/@tensorflow/tfjs-backend-webgl@${TF_VERSION}/dist/tf-backend-webgl.es2017.min.js`,
    integrity: 'sha384-vLZ29FEcQ9QWLdF6cKts4jNSPRioiSHmaggrIjdw0FbgdUp3CV0qUb4Mvp1C/ehV', bytes: 370446 },
  { src: `${CDN}/@tensorflow/tfjs-converter@${TF_VERSION}/dist/tf-converter.es2017.min.js`,
    integrity: 'sha384-jYgcd4ez9iEjyoeMMxPS7PCpt0FCX9Mc3ovkX5gEI0BnKmxJOiBBi9wWaPyzE40X', bytes: 286781 },
];

const DEFAULT_MODEL_BASE = new URL('../../../models/arbitrary-stylization-v1-256/', import.meta.url).href;
const DEFAULT_VECTORS_URL = new URL('../../../models/style-vectors.json', import.meta.url).href;
const MODEL_CACHE_PREFIX = 'pictura-models-';
const STYLE_SIDE = 256;
const LS_SIDE = 'pictura.engine.side';
const LS_FAILED = 'pictura.engine.browserFailedAt';

// Inference shapes are snapped to a few aspect-ratio buckets: WebGL compiles a
// set of shaders per input shape (10-30 s on a weak integrated GPU), so reusing
// shapes matters far more than the few percent of stretch a bucket introduces.
// 4:3 and 3:2 (most cameras and phones) share the sqrt(2) bucket (6% stretch at most).
const BUCKETS = [1, 5 / 4, Math.SQRT2, 16 / 9, 9 / 4];
const MAX_STRETCH = 1.08;

// ── Small utilities ─────────────────────────────────────────────────────────

function abortError() {
  return new DOMException('The operation was aborted.', 'AbortError');
}
function checkAbort(signal) {
  if (signal && signal.aborted) throw abortError();
}

function lsGet(key) { try { return localStorage.getItem(key); } catch { return null; } }
function lsSet(key, v) { try { localStorage.setItem(key, v); } catch { /* storage blocked */ } }

export function isMobileLike() {
  const coarseOnly = typeof matchMedia === 'function'
    && matchMedia('(pointer: coarse)').matches && !matchMedia('(any-pointer: fine)').matches;
  const ua = (typeof navigator !== 'undefined' && navigator.userAgent) || '';
  const iPadOS = /Macintosh/.test(ua) && typeof navigator !== 'undefined' && navigator.maxTouchPoints > 1;
  return coarseOnly || /Android|iPhone|iPad|iPod|Mobile/i.test(ua) || iPadOS;
}

/** Largest inference side this device should attempt (before timing it). */
export function deviceSideCap() {
  const mem = typeof navigator !== 'undefined' ? navigator.deviceMemory : undefined;
  if ((typeof mem === 'number' && mem <= 4) || isMobileLike()) return 768;
  return 1024;
}

/** Quick capability check: WebGL2 with renderable float textures. */
export function hasWebGL2() {
  try {
    // A page canvas: when WebGL is disabled by policy, OffscreenCanvas can still
    // report a context in some Chromium builds.
    const c = typeof document !== 'undefined' ? document.createElement('canvas') : new OffscreenCanvas(1, 1);
    const gl = c.getContext('webgl2');
    if (!gl) return false;
    const ok = !!(gl.getExtension('EXT_color_buffer_float') || gl.getExtension('EXT_color_buffer_half_float'));
    const lose = gl.getExtension('WEBGL_lose_context');
    if (lose) lose.loseContext();
    return ok;
  } catch {
    return false;
  }
}

export function recentBrowserFailure(days = 7) {
  const t = Number(lsGet(LS_FAILED) || 0);
  return t > 0 && Date.now() - t < days * 864e5;
}

function loadScript({ src, integrity }) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-pictura-src="${src}"]`);
    if (existing && existing.dataset.loaded === '1') { resolve(); return; }
    const s = existing || document.createElement('script');
    s.addEventListener('load', () => { s.dataset.loaded = '1'; resolve(); }, { once: true });
    s.addEventListener('error', () => reject(new EngineError('Could not load the on-device engine.', 'ENGINE_UNAVAILABLE')), { once: true });
    if (!existing) {
      s.src = src;
      s.integrity = integrity;
      s.crossOrigin = 'anonymous';
      s.async = false;
      s.dataset.picturaSrc = src;
      document.head.appendChild(s);
    }
  });
}

let tfPromise = null;
/** Loads TF.js (core + WebGL backend + converter) once, with SRI. */
function loadTf(onBytes) {
  if (!tfPromise) {
    tfPromise = (async () => {
      for (const script of TF_SCRIPTS) {
        // Sequential: each bundle extends the global `tf` from the previous one.
        await loadScript(script);
        onBytes(script.bytes);
      }
      const tf = globalThis.tf;
      if (!tf || !tf.loadGraphModel) throw new EngineError('The on-device engine did not start.', 'ENGINE_UNAVAILABLE');
      const ok = await tf.setBackend('webgl');
      if (!ok) throw new EngineError('This browser cannot run the on-device engine.', 'ENGINE_UNAVAILABLE');
      await tf.ready();
      installBandedConv(tf);
      return tf;
    })();
    tfPromise.catch(() => { tfPromise = null; });
  }
  return tfPromise;
}

// Cache-first fetch for model files (Cache Storage, one cache per model
// version; older versions are deleted), counting downloaded bytes.
async function openModelCache(version) {
  try {
    const name = `${MODEL_CACHE_PREFIX}${version}`;
    for (const old of await caches.keys()) {
      if (old.startsWith(MODEL_CACHE_PREFIX) && old !== name) caches.delete(old);
    }
    return await caches.open(name);
  } catch {
    return null;   // no Cache Storage (insecure context, private mode): plain HTTP cache
  }
}

function makeModelFetch(onBytes, version) {
  return async (input, init) => {
    const url = typeof input === 'string' ? input : input.url;
    const cache = await openModelCache(version);
    if (cache) {
      const hit = await cache.match(url);
      if (hit) {
        const len = Number(hit.headers.get('content-length') || 0);
        if (len) onBytes(len);
        return hit;
      }
    }
    const res = await fetch(url, init);
    if (!res.ok || !res.body) {
      if (!res.ok) throw new EngineError(`Model download failed (${res.status}).`, 'ENGINE_UNAVAILABLE');
      return res;
    }
    const reader = res.body.getReader();
    const chunks = [];
    let size = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      size += value.byteLength;
      onBytes(value.byteLength);
    }
    const headers = new Headers(res.headers);
    headers.set('content-length', String(size));
    const body = new Blob(chunks);
    const copy = new Response(body, { status: 200, headers });
    if (cache) cache.put(url, copy.clone()).catch(() => {});
    return copy;
  };
}

// ── IndexedDB cache for style parameters ────────────────────────────────────

const IDB_NAME = 'pictura-engine';
const IDB_STORE = 'style-params';
let idbPromise = null;
function idb() {
  if (!idbPromise) {
    idbPromise = new Promise((resolve) => {
      try {
        const req = indexedDB.open(IDB_NAME, 1);
        req.onupgradeneeded = () => req.result.createObjectStore(IDB_STORE);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => resolve(null);
        req.onblocked = () => resolve(null);
      } catch { resolve(null); }
    });
  }
  return idbPromise;
}
async function idbGet(key) {
  const db = await idb();
  if (!db) return null;
  return new Promise((resolve) => {
    try {
      const req = db.transaction(IDB_STORE, 'readonly').objectStore(IDB_STORE).get(key);
      req.onsuccess = () => resolve(req.result instanceof Float32Array ? req.result : null);
      req.onerror = () => resolve(null);
    } catch { resolve(null); }
  });
}
async function idbPut(key, value) {
  const db = await idb();
  if (!db) return;
  try { db.transaction(IDB_STORE, 'readwrite').objectStore(IDB_STORE).put(value, key); } catch { /* quota */ }
}

// IEEE 754 half -> float.
function halfToFloat32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    if (e === 0) out[i] = s * 5.960464477539063e-8 * f;
    else if (e === 31) out[i] = f ? NaN : s * Infinity;
    else out[i] = s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

// FNV-1a over a sample of pixels: a cheap cache key for a custom style image.
function hashPixels(data) {
  let h = 0x811c9dc5;
  const step = Math.max(1, Math.floor(data.length / 65536));
  for (let i = 0; i < data.length; i += step) {
    h ^= data[i];
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16) + ':' + data.length;
}

function drawableSize(src) {
  const w = src.naturalWidth || src.videoWidth || src.width;
  const h = src.naturalHeight || src.videoHeight || src.height;
  return [w, h];
}

function toImageData(src, w, h) {
  return resizeImageData(src, w, h);
}

/** Picks the inference shape for a content of size (w, h) at long side `side`. */
export function inferenceShape(w, h, side) {
  const landscape = w >= h;
  const aspect = landscape ? w / h : h / w;
  let best = BUCKETS[0];
  for (const b of BUCKETS) if (Math.abs(Math.log(aspect / b)) < Math.abs(Math.log(aspect / best))) best = b;
  const useBucket = Math.exp(Math.abs(Math.log(aspect / best))) <= MAX_STRETCH;
  const a = useBucket ? best : aspect;
  const short = Math.max(16, Math.round(side / a / 4) * 4);
  const long = Math.round(side / 4) * 4;
  const out = Math.max(1, Math.round(side / aspect));
  return {
    infer: landscape ? [long, short] : [short, long],          // [w, h] fed to the network
    output: landscape ? [side, out] : [out, side],             // [w, h] returned (true aspect)
  };
}

// ── Engine ──────────────────────────────────────────────────────────────────

/**
 * @param {object} [options]
 * @param {string} [options.modelBase]   base URL of models/arbitrary-stylization-v1-256/
 * @param {string} [options.vectorsUrl]  URL of models/style-vectors.json ('' to disable)
 * @param {(key:string)=>Promise<string>|string} [options.styleImageUrl]  preset key -> image URL
 * @param {number} [options.maxInferSide] force the inference long side (skips the speed probe)
 * @param {number} [options.timeBudgetMs] target time for one stylization when picking the side (6000)
 */
export function createBrowserEngine(options = {}) {
  const modelBase = options.modelBase || DEFAULT_MODEL_BASE;
  const vectorsUrl = options.vectorsUrl === undefined ? DEFAULT_VECTORS_URL : options.vectorsUrl;
  const cap = Math.min(deviceSideCap(), options.maxInferSide || Infinity);
  const budget = options.timeBudgetMs || 6000;
  const forcedSide = options.maxInferSide || null;

  let tf = null;
  let manifest = null;
  let transformer = null;
  let styleModel = null;
  let styleModelPromise = null;
  let warmPromise = null;
  let vectorsIndexPromise = null;
  let presetUrlMap = null;
  let broken = false;
  const paramCache = new Map();
  const compiled = new Set();

  const engine = {
    kind: 'browser',
    label: 'on this device',
    maxInferSide: Math.min(cap, Number(lsGet(LS_SIDE)) || 768),
    timings: {},
    warmup,
    stylize,
    prepare,
    getStyleParams,
    dispose,
    get ready() { return !!transformer && !broken; },
  };

  function fail(e) {
    if (e && (e.name === 'AbortError' || e instanceof EngineError)) return e;
    const msg = String((e && e.message) || e);
    if (broken || /context|lost|memory|alloc|texture|OOM|out of|too large/i.test(msg)) {
      return new EngineError('Your device ran out of graphics memory for this picture.', 'ENGINE_RESOURCE', e);
    }
    return new EngineError('The on-device engine failed.', 'ENGINE_FAILED', e);
  }

  function watchContext() {
    try {
      const gl = tf.backend().gpgpu.gl;
      gl.canvas.addEventListener('webglcontextlost', (ev) => {
        broken = true;
        lsSet(LS_FAILED, String(Date.now()));
        ev.preventDefault();
      });
    } catch { /* not the webgl backend */ }
  }

  async function loadManifest() {
    if (manifest) return manifest;
    const res = await fetch(new URL('manifest.json', modelBase));
    if (!res.ok) throw new EngineError('Model files are missing.', 'ENGINE_UNAVAILABLE');
    manifest = await res.json();
    return manifest;
  }

  function progressCounter(onProgress, total, phase) {
    let loaded = 0;
    return (n) => {
      loaded += n;
      if (onProgress) onProgress({ loaded: Math.min(loaded, total), total, phase });
    };
  }

  // Everyone waiting on the warm-up hears its progress (an idle prefetch may
  // start it before the person presses "Paint it").
  const listeners = new Set();
  function emit(p) {
    for (const fn of listeners) { try { fn(p); } catch { /* listener error */ } }
  }

  function warmup(onProgress) {
    if (broken) return Promise.reject(new EngineError('The on-device engine stopped.', 'ENGINE_RESOURCE'));
    if (transformer && warmPromise && engine.timings.warmupMs) {
      if (onProgress) onProgress({ loaded: 1, total: 1, phase: 'ready' });
      return warmPromise;
    }
    if (onProgress) listeners.add(onProgress);
    if (!warmPromise) {
      warmPromise = doWarmup()
        .catch((e) => { warmPromise = null; throw fail(e); })
        .finally(() => listeners.clear());
    }
    return warmPromise;
  }

  async function doWarmup() {
    const t0 = performance.now();
    const m = await loadManifest();
    const scriptBytes = TF_SCRIPTS.reduce((a, s) => a + s.bytes, 0);
    const tick = progressCounter(emit, scriptBytes + m.transformer.bytes, 'download');
    tick(0);
    tf = await loadTf(tick);
    watchContext();
    if (isMobileLike()) tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
    transformer = await tf.loadGraphModel(new URL(m.transformer.path, modelBase).href, { fetchFunc: makeModelFetch(tick, m.version) });
    engine.timings.loadMs = performance.now() - t0;

    // Inference size: forced, remembered from an earlier visit, or measured now
    // by timing one 512 px stylization and taking the largest side whose
    // predicted time (cost ~ pixels) fits the budget.
    emit({ loaded: 0, total: 1, phase: 'compile' });
    const t1 = performance.now();
    let side = forcedSide || Number(lsGet(LS_SIDE)) || 0;
    if (!side) {
      const probeSide = Math.min(512, cap);
      const [pw, ph] = inferenceShape(4, 3, probeSide).infer;
      await compileShape(pw, ph);
      const t2 = performance.now();
      await runOnce(pw, ph, new Float32Array(m.n));
      const probeMs = performance.now() - t2;
      engine.timings.probeMs = probeMs;
      side = probeSide;
      for (const s of [1024, 768]) {
        if (s <= cap && probeMs * (s / probeSide) ** 2 <= budget) { side = s; break; }
      }
      if (!forcedSide) lsSet(LS_SIDE, String(side));
    }
    engine.maxInferSide = Math.min(side, cap);
    await compileShape(...inferenceShape(4, 3, engine.maxInferSide).infer);
    engine.timings.compileMs = performance.now() - t1 - (engine.timings.probeMs || 0);
    engine.timings.warmupMs = performance.now() - t0;
    emit({ loaded: 1, total: 1, phase: 'ready' });
  }

  // Compile all shaders for one input shape without running them.
  async function compileShape(w, h) {
    const key = `${w}x${h}`;
    if (compiled.has(key)) return;
    const backend = tf.backend();
    if (typeof backend.checkCompileCompletionAsync !== 'function') return;
    tf.env().set('ENGINE_COMPILE_ONLY', true);
    let out = null;
    const c = tf.zeros([1, h, w, 3]);
    const p = tf.zeros([1, manifest.n]);
    try {
      out = transformer.execute({ content_image: c, style_params: p }, 'stylized');
      await backend.checkCompileCompletionAsync();
      backend.getUniformLocations();
    } finally {
      tf.env().set('ENGINE_COMPILE_ONLY', false);
      tf.dispose([c, p, out]);
    }
    compiled.add(key);
  }

  async function runOnce(w, h, params) {
    const out = tf.tidy(() => transformer.execute(
      { content_image: tf.zeros([1, h, w, 3]), style_params: tf.tensor2d(params, [1, manifest.n]) }, 'stylized'));
    await out.data();
    out.dispose();
  }

  /** Optional: compile shaders for a photo's shape ahead of "Paint it". */
  async function prepare({ width, height }) {
    await warmup();
    const { infer } = inferenceShape(width, height, engine.maxInferSide);
    await compileShape(infer[0], infer[1]);
  }

  // ── Style parameters ──

  function vectorsIndex() {
    if (!vectorsUrl) return Promise.resolve(null);
    if (!vectorsIndexPromise) {
      vectorsIndexPromise = fetch(vectorsUrl)
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null);
    }
    return vectorsIndexPromise;
  }

  async function presetImageUrl(key) {
    if (options.styleImageUrl) return options.styleImageUrl(key);
    if (!presetUrlMap) {
      presetUrlMap = new Map();
      try {
        const res = await fetch('/api/styles');
        if (res.ok) {
          const body = await res.json();
          const list = Array.isArray(body) ? body : body.styles || [];
          for (const s of list) if (s && s.key && s.image) presetUrlMap.set(s.key, s.image);
        }
      } catch { /* offline: use the default path */ }
    }
    return presetUrlMap.get(key) || `/app/assets/styles/${encodeURIComponent(key)}-512.jpg`;
  }

  async function loadStyleModel(onProgress) {
    if (styleModel) return styleModel;
    if (!styleModelPromise) {
      styleModelPromise = (async () => {
        const m = await loadManifest();
        const tick = progressCounter(onProgress, m.style.bytes, 'download-style');
        tick(0);
        styleModel = await tf.loadGraphModel(new URL(m.style.path, modelBase).href, { fetchFunc: makeModelFetch(tick, m.version) });
        return styleModel;
      })();
      styleModelPromise.catch(() => { styleModelPromise = null; });
    }
    return styleModelPromise;
  }

  async function paramsFromImage(image, cacheKey, onProgress, signal) {
    const [w, h] = drawableSize(image);
    if (!w || !h) throw new EngineError('The style image is empty.', 'ENGINE_INPUT');
    const px = toImageData(image, STYLE_SIDE, STYLE_SIDE);
    const key = cacheKey || `img:${manifest.version}:${hashPixels(px.data)}`;
    if (paramCache.has(key)) return paramCache.get(key);
    const stored = await idbGet(key);
    if (stored && stored.length === manifest.n) { paramCache.set(key, stored); return stored; }
    const model = await loadStyleModel(onProgress);
    checkAbort(signal);
    if (onProgress) onProgress({ loaded: 0, total: 1, phase: 'style' });
    const t = tf.tidy(() => {
      const x = tf.expandDims(tf.div(tf.cast(tf.browser.fromPixels(px), 'float32'), 255), 0);
      // Same light blur the server applies to style images.
      return model.execute({ style_image: tf.avgPool(x, 3, 1, 'same') }, 'style_params');
    });
    const params = new Float32Array(await t.data());
    t.dispose();
    paramCache.set(key, params);
    idbPut(key, params);
    return params;
  }

  async function presetParams(key, onProgress, signal) {
    const cacheKey = `preset:${manifest.version}:${key}`;
    if (paramCache.has(cacheKey)) return paramCache.get(cacheKey);
    const stored = await idbGet(cacheKey);
    if (stored && stored.length === manifest.n) { paramCache.set(cacheKey, stored); return stored; }
    const index = await vectorsIndex();
    const entry = index && index.model === manifest.name && index.version === manifest.version && index.styles && index.styles[key];
    if (entry) {
      try {
        const res = await fetch(new URL(entry.file, vectorsUrl));
        if (res.ok) {
          const buf = await res.arrayBuffer();
          const params = index.dtype === 'float16' ? halfToFloat32(new Uint16Array(buf)) : new Float32Array(buf);
          if (params.length === manifest.n) {
            paramCache.set(cacheKey, params);
            idbPut(cacheKey, params);
            return params;
          }
        }
      } catch { /* fall back to computing it */ }
    }
    // No precomputed vector: run the style network on the preset image.
    const url = await presetImageUrl(key);
    const img = await loadImageBitmap(url);
    return paramsFromImage(img, cacheKey, onProgress, signal);
  }

  /** Style parameters for {key} or {image}; cached in memory and IndexedDB. */
  async function getStyleParams(style, onProgress, signal) {
    await warmup();
    if (!style) throw new EngineError('Pick a style first.', 'ENGINE_INPUT');
    if (style.key) return presetParams(style.key, onProgress, signal);
    if (style.image) return paramsFromImage(style.image, null, onProgress, signal);
    throw new EngineError('Unknown style.', 'ENGINE_INPUT');
  }

  // ── Stylize ──

  async function stylize({ content, style, style2 = null, mix = 0.5, signal, onProgress } = {}) {
    try {
      checkAbort(signal);
      if (broken) throw new EngineError('The on-device engine stopped.', 'ENGINE_RESOURCE');
      if (!content) throw new EngineError('Add a photo first.', 'ENGINE_INPUT');
      const t0 = performance.now();
      await warmup(onProgress);
      checkAbort(signal);
      let params = await getStyleParams(style, onProgress, signal);
      if (style2) {
        const p2 = await getStyleParams(style2, onProgress, signal);
        const k = Math.max(0, Math.min(1, Number(mix)));
        const mixed = new Float32Array(params.length);
        for (let i = 0; i < mixed.length; i++) mixed[i] = (1 - k) * params[i] + k * p2[i];
        params = mixed;
      }
      checkAbort(signal);
      const [cw, ch] = drawableSize(content);
      if (!cw || !ch) throw new EngineError('The photo is empty.', 'ENGINE_INPUT');
      const side = engine.maxInferSide;
      const { infer, output } = inferenceShape(cw, ch, side);
      if (!compiled.has(`${infer[0]}x${infer[1]}`)) {
        if (onProgress) onProgress({ loaded: 0, total: 1, phase: 'compile' });
        await compileShape(infer[0], infer[1]);
        checkAbort(signal);
      }
      if (onProgress) onProgress({ loaded: 0, total: 1, phase: 'paint' });
      const input = toImageData(content, infer[0], infer[1]);
      const t1 = performance.now();
      const out = tf.tidy(() => {
        const x = tf.expandDims(tf.div(tf.cast(tf.browser.fromPixels(input), 'float32'), 255), 0);
        const y = transformer.execute({ content_image: x, style_params: tf.tensor2d(params, [1, manifest.n]) }, 'stylized');
        return tf.clipByValue(tf.squeeze(y, [0]), 0, 1);
      });
      let data;
      try {
        data = await out.data();
      } finally {
        out.dispose();
      }
      if (broken) throw new EngineError('The on-device engine stopped.', 'ENGINE_RESOURCE');
      checkAbort(signal);
      engine.timings.inferMs = performance.now() - t1;
      const [iw, ih] = infer;
      const raw = makeImageData(iw, ih);
      const d = raw.data;
      for (let i = 0, j = 0; i < iw * ih; i++, j += 3) {
        const o = i * 4;
        d[o] = data[j] * 255; d[o + 1] = data[j + 1] * 255; d[o + 2] = data[j + 2] * 255; d[o + 3] = 255;
      }
      const stylized = iw === output[0] && ih === output[1] ? raw : resizeImageData(raw, output[0], output[1]);
      engine.timings.stylizeMs = performance.now() - t0;
      if (onProgress) onProgress({ loaded: 1, total: 1, phase: 'done' });
      return { stylized };
    } catch (e) {
      throw fail(e);
    }
  }

  function dispose() {
    if (transformer) transformer.dispose();
    if (styleModel) styleModel.dispose();
    transformer = null; styleModel = null; styleModelPromise = null; warmPromise = null;
    compiled.clear();
  }

  return engine;
}

async function loadImageBitmap(url) {
  const res = await fetch(url);
  if (!res.ok) throw new EngineError('Could not load the style image.', 'ENGINE_INPUT');
  const blob = await res.blob();
  return createImageBitmap(blob);
}
