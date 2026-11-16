// Server engine: the same model, run by the PicturaAI backend.
// POST /api/transfer with output=raw returns the raw stylization (no intensity
// blend), so the browser compositor finishes both engines identically.

import { resizeImageData } from './composite.js';
import { EngineError } from './errors.js';

// Messages a person can act on (kept in step with the backend's limits).
const STATUS_MESSAGES = {
  413: 'That image is too large for the server. Please use a smaller photo.',
  429: 'The server is busy with other pictures. Please wait a moment and try again.',
  502: 'The server is waking up or restarting. Please try again in about a minute.',
  503: 'The server is waking up or restarting. Please try again in about a minute.',
  504: 'The server took too long to answer. Please try again in a moment.',
};

const UPLOAD_SIDE = 1024;       // the server works at <= NST_MAX_DIM anyway; keep uploads small
const STYLE_UPLOAD_SIDE = 512;  // the server resizes style images to 256 px

function abortError() {
  return new DOMException('The operation was aborted.', 'AbortError');
}

function drawableSize(src) {
  return [src.naturalWidth || src.width, src.naturalHeight || src.height];
}

async function toJpegBlob(src, maxSide, quality = 0.92) {
  const [w, h] = drawableSize(src);
  const k = Math.min(1, maxSide / Math.max(w, h));
  const tw = Math.max(1, Math.round(w * k));
  const th = Math.max(1, Math.round(h * k));
  const img = resizeImageData(src, tw, th);
  let canvas;
  if (typeof OffscreenCanvas !== 'undefined') {
    canvas = new OffscreenCanvas(tw, th);
    canvas.getContext('2d').putImageData(img, 0, 0);
    return canvas.convertToBlob({ type: 'image/jpeg', quality });
  }
  canvas = document.createElement('canvas');
  canvas.width = tw; canvas.height = th;
  canvas.getContext('2d').putImageData(img, 0, 0);
  return new Promise((resolve, reject) => canvas.toBlob(
    (b) => (b ? resolve(b) : reject(new EngineError('Could not prepare the photo.', 'ENGINE_INPUT'))), 'image/jpeg', quality));
}

async function errorFromResponse(res) {
  let detail = '';
  try {
    const body = await res.json();
    detail = body.detail || body.error || '';
  } catch { /* HTML or empty body */ }
  if (typeof detail !== 'string') detail = '';
  const known = STATUS_MESSAGES[res.status];
  const message = res.status === 429 || res.status === 413 || res.status >= 502 ? known : (detail || known);
  return new EngineError(message || `Something went wrong on the server (${res.status}). Please try again.`,
    res.status === 429 ? 'SERVER_BUSY' : res.status >= 500 ? 'SERVER_UNAVAILABLE' : 'SERVER_REJECTED');
}

function base64ToBlob(b64, type = 'image/jpeg') {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type });
}

/**
 * @param {{ endpoint?: string, fetchImpl?: typeof fetch }} [options]
 */
export function createServerEngine(options = {}) {
  const endpoint = options.endpoint || '/api/transfer';
  const doFetch = options.fetchImpl || ((...a) => fetch(...a));

  const engine = {
    kind: 'server',
    label: 'on our server',
    // The server's output size (NST_OUTPUT_DIM, 320 on the free tier); updated
    // from the first response.
    maxInferSide: 320,
    timings: {},
    async warmup(onProgress) {
      if (onProgress) onProgress({ loaded: 1, total: 1, phase: 'ready' });
    },
    stylize,
  };

  async function appendStyle(form, style, field, presetField) {
    if (!style) return;
    if (style.key) form.append(presetField, style.key);
    else if (style.image) form.append(field, await toJpegBlob(style.image, STYLE_UPLOAD_SIDE, 0.9), 'style.jpg');
    else throw new EngineError('Unknown style.', 'ENGINE_INPUT');
  }

  async function stylize({ content, style, style2 = null, mix = 0.5, signal, onProgress } = {}) {
    if (signal && signal.aborted) throw abortError();
    if (!content) throw new EngineError('Add a photo first.', 'ENGINE_INPUT');
    if (!style) throw new EngineError('Pick a style first.', 'ENGINE_INPUT');
    const t0 = performance.now();
    if (onProgress) onProgress({ loaded: 0, total: 1, phase: 'upload' });
    const form = new FormData();
    form.append('content_image', await toJpegBlob(content, UPLOAD_SIDE), 'photo.jpg');
    await appendStyle(form, style, 'style_image', 'style_preset');
    if (style2) {
      await appendStyle(form, style2, 'style_image_2', 'style_preset_2');
      form.append('style_mix_ratio', String(Math.max(0, Math.min(1, Number(mix)))));
    }
    form.append('output', 'raw');
    form.append('intensity', '1');

    let res;
    try {
      res = await doFetch(endpoint, { method: 'POST', body: form, signal });
    } catch (e) {
      if (e && e.name === 'AbortError') throw e;
      throw new EngineError(typeof navigator !== 'undefined' && navigator.onLine === false
        ? 'You appear to be offline. Check your connection and try again.'
        : 'Could not reach the server. It may be waking up; please try again in a minute.', 'SERVER_UNAVAILABLE', e);
    }
    if (!res.ok) throw await errorFromResponse(res);
    if (onProgress) onProgress({ loaded: 0, total: 1, phase: 'paint' });
    let body;
    try { body = await res.json(); } catch (e) {
      throw new EngineError('The server sent an unreadable answer. Please try again.', 'SERVER_UNAVAILABLE', e);
    }
    if (!body || typeof body.result !== 'string') {
      throw new EngineError('The server did not return a picture. Please try again.', 'SERVER_UNAVAILABLE');
    }
    // Older servers ignore output=raw and send the finished image: still usable.
    engine.rawOutput = body.output === 'raw';
    const bitmap = await createImageBitmap(base64ToBlob(body.result));
    if (signal && signal.aborted) throw abortError();
    const [w, h] = [bitmap.width, bitmap.height];
    const stylized = resizeImageData(bitmap, w, h);
    if (bitmap.close) bitmap.close();
    engine.maxInferSide = Math.max(w, h);
    engine.timings.stylizeMs = performance.now() - t0;
    engine.resolvedStyle = body.resolved_style_key ? { key: body.resolved_style_key, title: body.resolved_style_name } : null;
    if (onProgress) onProgress({ loaded: 1, total: 1, phase: 'done' });
    return { stylized };
  }

  return engine;
}
