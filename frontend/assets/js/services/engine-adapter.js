// The only place the UI touches the painting engine (BUILD-PLAN section 3).
// engine/composite.js is the only compositor. If engine/index.js cannot load
// (blocked or broken module), painting falls back to the server engine.

let enginePromise = null;
let compositePromise = null;
let serverPromise = null;

const serverModule = () => serverPromise || (serverPromise = import('../engine/server-engine.js'));

function engineModule() {
  if (!enginePromise) {
    enginePromise = import('../engine/index.js').catch((e) => {
      console.warn('PicturaAI: the engine module did not load, painting on the server.', e);
      return null;
    });
  }
  return enginePromise;
}

function composite() {
  if (!compositePromise) {
    compositePromise = import('../engine/composite.js');
    compositePromise.catch(() => { compositePromise = null; });
  }
  return compositePromise;
}

/** Start loading our own engine code (small; no TF.js until warmup). */
export const preloadEngineCode = () => Promise.all([engineModule(), composite().catch(() => null)]);

/** 'browser' | 'server': what getEngine() would choose now. */
export async function engineKindHint() {
  const m = await engineModule();
  try { return m ? m.preferredEngineKind() : 'server'; } catch { return 'server'; }
}

/** @returns {Promise<{kind,label,maxInferSide,warmup,stylize,ready?,offMainThread?}>} */
export async function getEngine(options = {}) {
  const m = await engineModule();
  if (m) return m.createEngine(options);
  const s = await serverModule();
  return s.createServerEngine();
}

export async function createCompositor(contentImageData, stylizedImageData) {
  return (await composite()).createCompositor(contentImageData, stylizedImageData);
}

export async function paletteTransfer(contentImageData, styleImageData, opts) {
  return (await composite()).paletteTransfer(contentImageData, styleImageData, opts);
}

export async function resizeImageData(src, width, height) {
  return (await composite()).resizeImageData(src, width, height);
}

let segmenterPromise = null;
/** One-tap subject masks (MediaPipe, loaded on first use). */
export function getSegmenter() {
  if (!segmenterPromise) {
    segmenterPromise = import('../engine/segment.js').then((m) => m.createSegmenter());
    segmenterPromise.catch(() => { segmenterPromise = null; });
  }
  return segmenterPromise;
}

/**
 * "Describe it" fallback when /api/describe-style is unreachable: the server
 * resolves the words to the closest preset as part of the transfer.
 * @returns {Promise<{stylized: ImageData, resolved: {key, title}|null}>}
 */
export async function stylizeFromPrompt({ content, prompt, signal }) {
  const engine = (await serverModule()).createServerEngine();
  const { stylized } = await engine.stylize({ content, style: { prompt }, signal });
  return { stylized, resolved: engine.resolvedStyle || null };
}
