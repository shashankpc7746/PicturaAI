// Small capability checks shared by the engine modules. Kept apart from
// browser-engine.js so the page can choose an engine without loading it.

export const LS_SIDE = 'pictura.engine.side';
export const LS_FAILED = 'pictura.engine.browserFailedAt';

/** localStorage, or nothing (workers, blocked storage). */
export const localStore = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, v) { try { localStorage.setItem(key, v); } catch { /* storage blocked */ } },
};

export function isMobileLike() {
  const coarseOnly = typeof matchMedia === 'function'
    && matchMedia('(pointer: coarse)').matches && !matchMedia('(any-pointer: fine)').matches;
  const ua = (typeof navigator !== 'undefined' && navigator.userAgent) || '';
  const iPadOS = /Macintosh/.test(ua) && navigator.maxTouchPoints > 1;
  return coarseOnly || /Android|iPhone|iPad|iPod|Mobile/i.test(ua) || iPadOS;
}

/** Largest inference side this device should attempt (before timing it). */
export function deviceSideCap(mobile = isMobileLike()) {
  const mem = typeof navigator !== 'undefined' ? navigator.deviceMemory : undefined;
  return (typeof mem === 'number' && mem <= 4) || mobile ? 768 : 1024;
}

/** WebGL2 with renderable float textures (a page canvas when there is a page). */
export function hasWebGL2() {
  try {
    // With WebGL disabled by policy, OffscreenCanvas can still report a context
    // in some Chromium builds, so prefer a page canvas.
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

/** The browser engine lost its GPU context on this device in the last `days`. */
export function recentBrowserFailure(days = 7, store = localStore) {
  const t = Number(store.get(LS_FAILED) || 0);
  return t > 0 && Date.now() - t < days * 864e5;
}
