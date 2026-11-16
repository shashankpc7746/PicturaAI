// PicturaAI engine entry point. The UI talks only to this module and composite.js.
//
//   const engine = await createEngine();          // browser engine if WebGL2, else server
//   await engine.warmup(({ loaded, total, phase }) => ...);   // optional, e.g. on idle
//   const { stylized } = await engine.stylize({ content, style: { key: 'starry_night' } });
//
// The browser engine runs in a Web Worker when it can (engine.offMainThread).
// If a browser-engine call rejects with err.code 'ENGINE_RESOURCE' or
// 'ENGINE_UNAVAILABLE', switch with `await createEngine({ prefer: 'server' })`.
// Testing: ?engine=browser or ?engine=server forces an engine; ?worker=0 keeps
// the browser engine on the page.

import { hasWebGL2, recentBrowserFailure } from './support.js';
import { createWorkerEngine, workerCapable } from './worker-engine.js';
import { createServerEngine } from './server-engine.js';
import { EngineError } from './errors.js';

// createBrowserEngine (the in-page engine) lives in browser-engine.js, loaded on demand.
export { EngineError, createServerEngine, createWorkerEngine, hasWebGL2 };

const instances = {};

function param(name) {
  try {
    return new URLSearchParams(location.search).get(name);
  } catch {
    return null;
  }
}

function urlChoice() {
  const v = param('engine');
  return v === 'browser' || v === 'server' ? v : null;
}

/** Which engine createEngine() would pick right now, without creating it. */
export function preferredEngineKind(options = {}) {
  const forced = urlChoice();
  if (forced) return forced;
  if (options.prefer === 'server') return 'server';
  if (!hasWebGL2()) return 'server';
  if (options.prefer !== 'browser' && recentBrowserFailure()) return 'server';
  return 'browser';
}

function serverEngine(options) {
  return instances.server || (instances.server = createServerEngine(options.server));
}

/**
 * Picks the browser engine (TF.js, WebGL2) when supported, else the server engine.
 * @param {object} [options]
 * @param {'auto'|'browser'|'server'} [options.prefer='auto']
 * @param {boolean} [options.worker=true] run the browser engine in a Web Worker when possible
 * @param {boolean} [options.warmup=false] load the browser engine now and fall back to
 *   the server engine if that fails (otherwise call engine.warmup() yourself, e.g. on idle)
 * @param {(p:{loaded:number,total:number,phase:string})=>void} [options.onProgress]
 * @param {object} [options.browser]  options for createBrowserEngine
 * @param {object} [options.server]   options for createServerEngine
 * @returns {Promise<object>} Engine (see docs/revamp/BUILD-PLAN.md section 3)
 */
export async function createEngine(options = {}) {
  if (preferredEngineKind(options) === 'server') return serverEngine(options);
  if (!instances.browser) {
    const inWorker = options.worker !== false && param('worker') !== '0' && workerCapable();
    instances.browser = inWorker ? createWorkerEngine(options.browser)
      : import('./browser-engine.js').then((m) => m.createBrowserEngine(options.browser));
  }
  const engine = await instances.browser;
  if (options.warmup) {
    try {
      await engine.warmup(options.onProgress);
    } catch (e) {
      if ((e && e.name === 'AbortError') || urlChoice() === 'browser') throw e;
      return serverEngine(options);
    }
  }
  return engine;
}
