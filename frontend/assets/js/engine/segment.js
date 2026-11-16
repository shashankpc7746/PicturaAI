// One-tap subject / background masks with MediaPipe Image Segmenter (DeepLab-v3,
// 21 PASCAL VOC classes), on the device. Loaded lazily on first use.
//
//   const seg = await createSegmenter();
//   const { subjectMask, foundLabels } = await seg.segment(imageBitmapOrCanvas);
//   // subjectMask: ImageData at the image's size, white = subject (any non-background class)
//   // "Style the background" = the same mask inverted.

import { EngineError } from './errors.js';
import { isMobileLike } from './browser-engine.js';

const MP_VERSION = '0.10.35';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const DEFAULT_MODEL_URL = new URL('../../../models/deeplab_v3/deeplab_v3.tflite', import.meta.url).href;
// Classes smaller than this share of the picture are not reported as "found".
const MIN_LABEL_SHARE = 0.005;

function isAppleTouch() {
  const ua = navigator.userAgent || '';
  return /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
}

function sizeOf(image) {
  return [image.naturalWidth || image.videoWidth || image.width, image.naturalHeight || image.videoHeight || image.height];
}

let visionPromise = null;
function loadVision() {
  if (!visionPromise) {
    // ES module import from the pinned CDN build (dynamic import() cannot carry
    // SRI; the version is exact and the CSP limits scripts to this host).
    visionPromise = import(/* webpackIgnore: true */ `${MP_BASE}/vision_bundle.mjs`)
      .then(async (vision) => {
        const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
        return { vision, fileset };
      });
    visionPromise.catch(() => { visionPromise = null; });
  }
  return visionPromise;
}

async function createTask(vision, fileset, modelUrl, delegate) {
  return vision.ImageSegmenter.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: modelUrl, delegate },
    runningMode: 'IMAGE',
    outputCategoryMask: true,
    outputConfidenceMasks: false,
  });
}

/**
 * @param {{ modelUrl?: string, delegate?: 'GPU'|'CPU' }} [options]
 * @returns {Promise<{ segment(image): Promise<{subjectMask: ImageData, foundLabels: string[]}>,
 *                     labels: string[], delegate: string, close(): void }>}
 */
export async function createSegmenter(options = {}) {
  const modelUrl = options.modelUrl || DEFAULT_MODEL_URL;
  // iOS/iPadOS Safari returns wrong labels with the GPU delegate
  // (google-ai-edge/mediapipe#6142), so use the CPU there.
  let delegate = options.delegate || (isAppleTouch() ? 'CPU' : 'GPU');
  let task;
  try {
    const { vision, fileset } = await loadVision();
    try {
      task = await createTask(vision, fileset, modelUrl, delegate);
    } catch (e) {
      if (delegate !== 'GPU') throw e;
      delegate = 'CPU';
      task = await createTask(vision, fileset, modelUrl, delegate);
    }
  } catch (e) {
    throw new EngineError('Automatic selection is not available in this browser. You can still paint the area by hand.', 'ENGINE_UNAVAILABLE', e);
  }
  const labels = task.getLabels ? task.getLabels() : [];
  // Mobile: segment a smaller copy (DeepLab works at 257 px anyway).
  const maxSide = isMobileLike() ? 1024 : 1600;

  async function segment(image) {
    const [w, h] = sizeOf(image);
    if (!w || !h) throw new EngineError('The photo is empty.', 'ENGINE_INPUT');
    let input = image;
    if (Math.max(w, h) > maxSide || (image.data && !image.getContext)) {
      const k = Math.min(1, maxSide / Math.max(w, h));
      const c = document.createElement('canvas');
      c.width = Math.max(1, Math.round(w * k)); c.height = Math.max(1, Math.round(h * k));
      const ctx = c.getContext('2d');
      if (image.data && !image.getContext) {
        const tmp = document.createElement('canvas');
        tmp.width = w; tmp.height = h;
        tmp.getContext('2d').putImageData(image, 0, 0);
        ctx.drawImage(tmp, 0, 0, c.width, c.height);
      } else {
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(image, 0, 0, c.width, c.height);
      }
      input = c;
    }
    let result;
    try {
      result = task.segment(input);
    } catch (e) {
      throw new EngineError('Automatic selection failed on this picture. You can paint the area by hand.', 'ENGINE_FAILED', e);
    }
    try {
      const mask = result.categoryMask;
      if (!mask) throw new EngineError('Automatic selection found nothing.', 'ENGINE_FAILED');
      const cats = mask.getAsUint8Array();
      const mw = mask.width, mh = mask.height;
      const counts = new Uint32Array(256);
      const small = new ImageData(mw, mh);
      const d = small.data;
      for (let i = 0, j = 0; i < cats.length; i++, j += 4) {
        const c = cats[i];
        counts[c]++;
        const v = c === 0 || c === 255 ? 0 : 255;   // 0 = background (255 = none in some models)
        d[j] = d[j + 1] = d[j + 2] = v;
        d[j + 3] = 255;
      }
      const foundLabels = [];
      for (let c = 1; c < 255; c++) {
        if (counts[c] / cats.length >= MIN_LABEL_SHARE) foundLabels.push(labels[c] || `class ${c}`);
      }
      let subjectMask = small;
      if (mw !== w || mh !== h) {
        const src = document.createElement('canvas');
        src.width = mw; src.height = mh;
        src.getContext('2d').putImageData(small, 0, 0);
        const dst = document.createElement('canvas');
        dst.width = w; dst.height = h;
        const ctx = dst.getContext('2d', { willReadFrequently: true });
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(src, 0, 0, w, h);
        subjectMask = ctx.getImageData(0, 0, w, h);
      }
      return { subjectMask, foundLabels };
    } finally {
      if (result && result.close) result.close();
    }
  }

  return {
    segment,
    labels,
    delegate,
    close() { try { task.close(); } catch { /* already closed */ } },
  };
}
