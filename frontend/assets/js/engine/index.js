// PicturaAI engine entry point. The UI talks only to this module and composite.js.
//
//   const engine = await createEngine();          // browser engine if WebGL2, else server
//   await engine.warmup(({ loaded, total, phase }) => ...);   // optional, e.g. on idle
//   const { stylized } = await engine.stylize({ content, style: { key: 'starry_night' } });
//
// If a browser-engine call rejects with err.code 'ENGINE_RESOURCE' or
// 'ENGINE_UNAVAILABLE', switch with `await createEngine({ prefer: 'server' })`.
// Testing: add ?engine=browser or ?engine=server to the page URL.

import { createBrowserEngine, hasWebGL2, recentBrowserFailure } from './browser-engine.js';
import { createServerEngine } from './server-engine.js';
import { EngineError } from './errors.js';

export { EngineError, createBrowserEngine, createServerEngine, hasWebGL2 };

const instances = {};

function urlChoice() {
  try {
    const v = new URLSearchParams(location.search).get('engine');
    return v === 'browser' || v === 'server' ? v : null;
  } catch {
    return null;
  }
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

/**
 * Picks the browser engine (TF.js, WebGL2) when supported, else the server engine.
 * @param {object} [options]
 * @param {'auto'|'browser'|'server'} [options.prefer='auto']
 * @param {boolean} [options.warmup=false] load the browser engine now and fall back to
 *   the server engine if that fails (otherwise call engine.warmup() yourself, e.g. on idle)
 * @param {(p:{loaded:number,total:number,phase:string})=>void} [options.onProgress]
 * @param {object} [options.browser]  options for createBrowserEngine
 * @param {object} [options.server]   options for createServerEngine
 * @returns {Promise<object>} Engine (see docs/revamp/BUILD-PLAN.md section 3)
 */
export async function createEngine(options = {}) {
  const kind = preferredEngineKind(options);
  if (kind === 'server') {
    return instances.server || (instances.server = createServerEngine(options.server));
  }
  const engine = instances.browser || (instances.browser = createBrowserEngine(options.browser));
  if (options.warmup) {
    try {
      await engine.warmup(options.onProgress);
    } catch (e) {
      if (e && e.name === 'AbortError') throw e;
      if (urlChoice() === 'browser') throw e;
      return instances.server || (instances.server = createServerEngine(options.server));
    }
  }
  return engine;
}
