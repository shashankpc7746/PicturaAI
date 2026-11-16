// The browser engine with the same API (BUILD-PLAN section 3), run in worker.js.
//
// WebGL shader compiles and GPU waits used to block the page for seconds (minutes
// with software GL), so Cancel and scrolling froze. In the worker the page stays
// responsive and Cancel answers at once (the worker finishes its current GPU pass
// and drops the result). Falls back to the in-page engine when the worker cannot
// start, has no OffscreenCanvas WebGL2 (Safari before 17), or TF.js fails there.

import { EngineError } from './errors.js';
import { deviceSideCap, isMobileLike, localStore, LS_SIDE, LS_FAILED } from './support.js';

const HELLO_TIMEOUT_MS = 20000;

function abortError() {
  return new DOMException('The operation was aborted.', 'AbortError');
}

/** OffscreenCanvas WebGL2 and module workers: worth trying the worker. */
export function workerCapable() {
  try {
    return typeof Worker !== 'undefined' && typeof OffscreenCanvas !== 'undefined'
      && !!new OffscreenCanvas(1, 1).getContext('webgl2');
  } catch {
    return false;
  }
}

function toError(e) {
  if (e.name === 'AbortError') return abortError();
  return new EngineError(e.message || 'The on-device engine failed.', e.code || 'ENGINE_FAILED');
}

// A private copy for the worker (transferring would detach the caller's bitmap).
const bitmapOf = (src) => createImageBitmap(src);

/**
 * @param {object} [options] same as createBrowserEngine (styleImageUrl is not supported)
 */
export function createWorkerEngine(options = {}) {
  const store = options.store || localStore;
  const mobile = isMobileLike();
  const { store: _s, styleImageUrl: _u, ...workerOptions } = options;
  let worker = null;
  let starting = null;
  let inPage = null;
  let warmPromise = null;
  let ready = false;
  let broken = false;
  let seq = 0;
  const pending = new Map();
  const listeners = new Set();

  const engine = {
    kind: 'browser',
    label: 'on this device',
    offMainThread: true,
    maxInferSide: Math.min(deviceSideCap(mobile), options.maxInferSide || Infinity, Number(store.get(LS_SIDE)) || 768),
    timings: {},
    warmup,
    stylize,
    prepare,
    dispose,
    get ready() { return inPage ? inPage.ready : ready && !broken; },
  };

  // Loaded only when needed, so the page does not download it for nothing.
  async function useInPage() {
    if (!inPage) {
      if (worker) worker.terminate();
      worker = null;
      const { createBrowserEngine } = await import('./browser-engine.js');
      inPage = inPage || createBrowserEngine(options);
      engine.offMainThread = false;
    }
    return inPage;
  }

  function sync(v) {
    if (!v) return;
    engine.maxInferSide = v.maxInferSide;
    engine.timings = v.timings;
    ready = v.ready;
  }

  function onMessage({ data: m }) {
    if (m.type === 'store') { store.set(m.key, m.value); return; }
    const p = pending.get(m.id);
    if (!p) return;
    if (m.type === 'progress') { if (p.onProgress) try { p.onProgress(m.p); } catch { /* listener error */ } }
    else if (m.type === 'result') { sync(m.value); p.resolve(m.value); }
    else if (m.type === 'error') p.reject(toError(m.error));
  }

  // The worker died (usually out of memory): fail what is running, like a lost context.
  function onCrash(ev) {
    if (ev && ev.preventDefault) ev.preventDefault();
    broken = true;
    store.set(LS_FAILED, String(Date.now()));
    const err = new EngineError('Your device ran out of graphics memory for this picture.', 'ENGINE_RESOURCE');
    for (const p of pending.values()) p.reject(err);
  }

  /** Resolves to the running worker, or null when the in-page engine must be used. */
  function start() {
    if (!starting) {
      starting = new Promise((resolve) => {
        let w;
        const give = (ok) => {
          clearTimeout(timer);
          if (ok) {
            worker = w;
            w.onmessage = onMessage;
            w.onerror = onCrash;
            resolve(w);
          } else {
            if (w) w.terminate();
            resolve(null);
          }
        };
        const timer = setTimeout(() => give(false), HELLO_TIMEOUT_MS);
        try {
          w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module', name: 'pictura-engine' });
        } catch {
          give(false);
          return;
        }
        w.onmessage = ({ data }) => { if (data && data.type === 'hello') give(!!data.ok); };
        w.onerror = (ev) => { if (ev.preventDefault) ev.preventDefault(); give(false); };
        w.postMessage({
          op: 'init',
          mobile,
          options: workerOptions,
          store: { [LS_SIDE]: store.get(LS_SIDE), [LS_FAILED]: store.get(LS_FAILED) },
        });
      });
    }
    return starting;
  }

  function call(op, payload = {}, { signal, onProgress, transfer = [] } = {}) {
    if (broken || !worker) return Promise.reject(new EngineError('The on-device engine stopped.', 'ENGINE_RESOURCE'));
    const id = ++seq;
    return new Promise((resolve, reject) => {
      const done = () => {
        pending.delete(id);
        if (signal) signal.removeEventListener('abort', onAbort);
      };
      const onAbort = () => {
        done();
        if (worker) worker.postMessage({ op: 'abort', id });
        reject(abortError());
      };
      pending.set(id, { onProgress, resolve: (v) => { done(); resolve(v); }, reject: (e) => { done(); reject(e); } });
      if (signal) {
        if (signal.aborted) { onAbort(); return; }
        signal.addEventListener('abort', onAbort, { once: true });
      }
      worker.postMessage({ op, id, ...payload }, transfer);
    });
  }

  function emit(p) {
    for (const fn of listeners) { try { fn(p); } catch { /* listener error */ } }
  }

  function warmup(onProgress) {
    if (inPage) return inPage.warmup(onProgress);
    if (broken) return Promise.reject(new EngineError('The on-device engine stopped.', 'ENGINE_RESOURCE'));
    if (ready && warmPromise) {
      if (onProgress) onProgress({ loaded: 1, total: 1, phase: 'ready' });
      return warmPromise;
    }
    if (onProgress) listeners.add(onProgress);
    if (!warmPromise) {
      warmPromise = (async () => {
        if (!(await start())) return (await useInPage()).warmup(emit);
        try {
          await call('warmup', {}, { onProgress: emit });
        } catch (e) {
          if (e.code !== 'ENGINE_WORKER') throw e;
          return (await useInPage()).warmup(emit);
        }
        return undefined;
      })()
        .catch((e) => { warmPromise = null; throw e; })
        .finally(() => listeners.clear());
    }
    return warmPromise;
  }

  // Rejects as soon as `signal` aborts, even if `promise` is still running.
  function abortable(promise, signal) {
    if (!signal) return promise;
    if (signal.aborted) return Promise.reject(abortError());
    return new Promise((resolve, reject) => {
      const onAbort = () => reject(abortError());
      signal.addEventListener('abort', onAbort, { once: true });
      promise.then(resolve, reject).finally(() => signal.removeEventListener('abort', onAbort));
    });
  }

  async function styleMessage(style, transfer) {
    if (!style || style.key) return style ? { key: style.key } : null;
    if (!style.image) throw new EngineError('Unknown style.', 'ENGINE_INPUT');
    const image = await bitmapOf(style.image);
    transfer.push(image);
    return { image };
  }

  async function stylize(args = {}) {
    const { content, style, style2 = null, mix = 0.5, signal, onProgress } = args;
    if (signal && signal.aborted) throw abortError();
    if (!content) throw new EngineError('Add a photo first.', 'ENGINE_INPUT');
    if (!style) throw new EngineError('Pick a style first.', 'ENGINE_INPUT');
    if (!inPage) await abortable(warmup(onProgress), signal);
    if (inPage) return inPage.stylize(args);
    const transfer = [];
    const msg = { mix };
    try {
      msg.content = await bitmapOf(content);
      transfer.push(msg.content);
      msg.style = await styleMessage(style, transfer);
      msg.style2 = await styleMessage(style2, transfer);
    } catch (e) {
      for (const b of transfer) b.close();
      throw e instanceof EngineError ? e : new EngineError('Could not read the picture.', 'ENGINE_INPUT', e);
    }
    const v = await call('stylize', msg, { signal, onProgress, transfer });
    return { stylized: new ImageData(new Uint8ClampedArray(v.buffer), v.width, v.height) };
  }

  async function prepare({ width, height }) {
    await warmup();
    if (inPage) return inPage.prepare({ width, height });
    await call('prepare', { width, height });
    return undefined;
  }

  function dispose() {
    if (inPage) inPage.dispose();
    if (worker) worker.terminate();
    for (const p of pending.values()) p.reject(abortError());
    worker = null; starting = null; inPage = null; warmPromise = null; ready = false;
  }

  return engine;
}
