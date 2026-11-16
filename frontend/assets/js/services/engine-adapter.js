// The only place the UI touches the painting engine (BUILD-PLAN section 3).
//
// Uses frontend/assets/js/engine/ when it loads. Everything under
// "TEMPORARY FALLBACK" below runs only when that module is missing or broken;
// it calls the server directly and blends on a canvas, so the studio works end
// to end either way. Integration can delete the fallback once the engine ships.

import { apiFetch, base64ToBlob } from './api.js';
import { toCanvas, canvasToBlob, canvasToImageData, blobToImage } from './image-io.js';

let modsPromise = null;

function loadModules() {
  if (!modsPromise) {
    modsPromise = Promise.all([import('../engine/index.js'), import('../engine/composite.js')])
      .then(([engine, composite]) => ({ engine, composite }))
      .catch((e) => {
        console.warn('PicturaAI: engine module unavailable, using the built-in server fallback.', e);
        return null;
      });
  }
  return modsPromise;
}

/** Start loading our own engine code (small; no TF.js until warmup). */
export const preloadEngineCode = () => loadModules();

/** 'browser' | 'server': what getEngine() would choose now. */
export async function engineKindHint() {
  const m = await loadModules();
  if (!m) return 'server';
  try { return m.engine.preferredEngineKind ? m.engine.preferredEngineKind() : 'server'; } catch { return 'server'; }
}

/** @returns {Promise<{kind,label,maxInferSide,warmup,stylize,ready?}>} */
export async function getEngine(options = {}) {
  const m = await loadModules();
  if (!m) return fallbackEngine;
  return m.engine.createEngine(options);
}

export async function createCompositor(contentImageData, stylizedImageData) {
  const m = await loadModules();
  return m ? m.composite.createCompositor(contentImageData, stylizedImageData)
    : fallbackCompositor(contentImageData, stylizedImageData);
}

export async function paletteTransfer(contentImageData, styleImageData, opts) {
  const m = await loadModules();
  if (m) return m.composite.paletteTransfer(contentImageData, styleImageData, opts);
  return fallbackPalette(contentImageData, styleImageData, opts);
}

export async function resizeImageData(src, width, height) {
  const m = await loadModules();
  if (m) return m.composite.resizeImageData(src, width, height);
  const c = document.createElement('canvas');
  c.width = width; c.height = height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  if (src instanceof ImageData) {
    const tmp = document.createElement('canvas');
    tmp.width = src.width; tmp.height = src.height;
    tmp.getContext('2d').putImageData(src, 0, 0);
    ctx.drawImage(tmp, 0, 0, width, height);
  } else {
    ctx.drawImage(src, 0, 0, width, height);
  }
  return ctx.getImageData(0, 0, width, height);
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
  return serverStylize({ content, style: { prompt }, signal });
}

// ── TEMPORARY FALLBACK (remove after integration) ───────────────────────────

async function appendStyle(form, style, imageField, presetField) {
  if (!style) return;
  if (style.key) form.append(presetField, style.key);
  else if (style.image) form.append(imageField, await canvasToBlob(toCanvas(style.image, 512), 'image/jpeg', 0.9), 'style.jpg');
  else if (style.prompt) form.append('text_prompt', style.prompt);
}

async function serverStylize({ content, style, style2 = null, mix = 0.5, signal, onProgress }) {
  if (onProgress) onProgress({ loaded: 0, total: 1, phase: 'upload' });
  const form = new FormData();
  form.append('content_image', await canvasToBlob(toCanvas(content, 1024), 'image/jpeg', 0.92), 'photo.jpg');
  await appendStyle(form, style, 'style_image', 'style_preset');
  if (style2) {
    await appendStyle(form, style2, 'style_image_2', 'style_preset_2');
    form.append('style_mix_ratio', String(mix));
  }
  form.append('style_weight', '1');
  form.append('intensity', '1');
  form.append('output', 'raw');
  const body = await apiFetch('/api/transfer', { method: 'POST', body: form, signal });
  if (!body || typeof body.result !== 'string') throw new Error('The server did not return a picture. Please try again.');
  if (onProgress) onProgress({ loaded: 1, total: 1, phase: 'paint' });
  const img = await blobToImage(base64ToBlob(body.result));
  const stylized = canvasToImageData(toCanvas(img));
  fallbackEngine.maxInferSide = Math.max(stylized.width, stylized.height);
  return {
    stylized,
    resolved: body.resolved_style_key ? { key: body.resolved_style_key, title: body.resolved_style_name } : null,
  };
}

const fallbackEngine = {
  kind: 'server',
  label: 'on our server',
  maxInferSide: 320,
  ready: true,
  async warmup(onProgress) { if (onProgress) onProgress({ loaded: 1, total: 1, phase: 'ready' }); },
  stylize: serverStylize,
};

function rgbStats(img) {
  const d = img.data; const n = d.length / 4;
  const sum = [0, 0, 0]; const sq = [0, 0, 0];
  for (let i = 0; i < d.length; i += 4) {
    for (let c = 0; c < 3; c++) { sum[c] += d[i + c]; sq[c] += d[i + c] * d[i + c]; }
  }
  return [0, 1, 2].map((c) => {
    const mean = sum[c] / n;
    return [mean, Math.sqrt(Math.max(0, sq[c] / n - mean * mean)) + 1e-6];
  });
}

function fallbackPalette(content, style, { strength = 1, style2ImageData = null, mix = 0.5 } = {}) {
  const cs = rgbStats(content);
  let ss = rgbStats(style);
  if (style2ImageData) {
    const s2 = rgbStats(style2ImageData);
    ss = ss.map(([m, s], c) => [(1 - mix) * m + mix * s2[c][0], (1 - mix) * s + mix * s2[c][1]]);
  }
  const out = new ImageData(content.width, content.height);
  const o = out.data; const d = content.data;
  for (let i = 0; i < d.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const v = (d[i + c] - cs[c][0]) * (ss[c][1] / cs[c][1]) + ss[c][0];
      o[i + c] = d[i + c] + (v - d[i + c]) * strength;
    }
    o[i + 3] = 255;
  }
  return out;
}

function fallbackCompositor(content, stylized) {
  const { width: w, height: h } = stylized;
  const c = content.width === w && content.height === h ? content : null;
  if (!c) throw new Error('Compositor needs images of the same size.');
  const n = w * h * 4;
  function maskAlpha(mask) {
    if (!mask) return null;
    const cv = document.createElement('canvas');
    cv.width = mask.width; cv.height = mask.height;
    cv.getContext('2d').putImageData(mask, 0, 0);
    const scaled = document.createElement('canvas');
    scaled.width = w; scaled.height = h;
    const sctx = scaled.getContext('2d', { willReadFrequently: true });
    sctx.filter = `blur(${Math.max(1, Math.round(Math.max(w, h) / 160))}px)`;
    sctx.drawImage(cv, 0, 0, w, h);
    return sctx.getImageData(0, 0, w, h).data;
  }
  return {
    width: w,
    height: h,
    render({ intensity = 1, mask = null } = {}) {
      const a = Math.max(0, Math.min(1, intensity));
      const m = maskAlpha(mask);
      const out = new ImageData(w, h);
      const o = out.data; const cd = c.data; const sd = stylized.data;
      for (let i = 0; i < n; i += 4) {
        const k = m ? a * ((m[i] / 255) * (m[i + 3] / 255)) : a;
        o[i] = cd[i] + (sd[i] - cd[i]) * k;
        o[i + 1] = cd[i + 1] + (sd[i + 1] - cd[i + 1]) * k;
        o[i + 2] = cd[i + 2] + (sd[i + 2] - cd[i + 2]) * k;
        o[i + 3] = 255;
      }
      return out;
    },
  };
}
