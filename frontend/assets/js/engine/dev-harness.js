// Developer harness for the engine modules (no UI polish; exercised by tests).
import { createEngine } from './index.js';
import { createCompositor, paletteTransfer, resizeImageData } from './composite.js';
import { createSegmenter } from './segment.js';

const $ = (id) => document.getElementById(id);
const logEl = $('log');
const state = { engine: null, photo: null, photoData: null, styleImage: null, stylized: null, compositor: null,
  mask: null, segmenter: null, abort: null, styles: [] };

function log(...parts) {
  const line = parts.map((p) => (typeof p === 'string' ? p : JSON.stringify(p))).join(' ');
  logEl.textContent += line + '\n';
  logEl.scrollTop = logEl.scrollHeight;
  console.info('[harness]', line);
}

function draw(canvasId, imageData) {
  const c = $(canvasId);
  c.width = imageData.width; c.height = imageData.height;
  c.getContext('2d').putImageData(imageData, 0, 0);
}

function showEngine() {
  const e = state.engine;
  $('engine-info').textContent = e
    ? JSON.stringify({ kind: e.kind, label: e.label, maxInferSide: e.maxInferSide, timings: e.timings }, null, 1)
    : 'No engine yet.';
}

async function loadBitmap(src) {
  if (src instanceof Blob) return createImageBitmap(src, { imageOrientation: 'from-image' });
  const res = await fetch(src);
  if (!res.ok) throw new Error(`${src}: ${res.status}`);
  return createImageBitmap(await res.blob(), { imageOrientation: 'from-image' });
}

async function loadStyles() {
  let list = [];
  try {
    const res = await fetch('/api/styles');
    if (res.ok) {
      const body = await res.json();
      list = (Array.isArray(body) ? body : body.styles || []).map((s) => ({ key: s.key, title: s.title || s.name || s.key }));
    }
  } catch { /* offline */ }
  if (!list.length) {
    try {
      const v = await (await fetch('/app/models/style-vectors.json')).json();
      list = Object.keys(v.styles).map((key) => ({ key, title: key }));
    } catch { /* none */ }
  }
  state.styles = list;
  for (const sel of [$('sel-style'), $('sel-style2')]) {
    for (const s of list) {
      const o = document.createElement('option');
      o.value = s.key; o.textContent = s.title;
      sel.appendChild(o);
    }
  }
}

function currentStyle() {
  if (state.styleImage) return { image: state.styleImage };
  return { key: $('sel-style').value };
}

async function ensurePhoto() {
  if (!state.photo) {
    state.photo = await loadBitmap($('sel-photo').value);
  }
  return state.photo;
}

async function create(prefer) {
  state.engine = await createEngine(prefer ? { prefer } : {});
  log('engine', state.engine.kind, state.engine.label);
  showEngine();
  return { kind: state.engine.kind, label: state.engine.label, maxInferSide: state.engine.maxInferSide };
}

async function warmup() {
  if (!state.engine) await create();
  const t0 = performance.now();
  await state.engine.warmup((p) => {
    $('warm-progress').value = p.total ? p.loaded / p.total : 0;
    if (p.phase !== 'download' || p.loaded === p.total) log('progress', p);
  });
  const ms = Math.round(performance.now() - t0);
  log('warmup done in', ms, 'ms; maxInferSide', state.engine.maxInferSide);
  showEngine();
  return { ms, maxInferSide: state.engine.maxInferSide, timings: state.engine.timings };
}

function rebuildCompositor() {
  const { stylized } = state;
  const photo = state.photo;
  // Composite at the photo's size, capped at 1200 px (the live-slider budget).
  const k = Math.min(1, 1200 / Math.max(photo.width, photo.height));
  const w = Math.round(photo.width * k), h = Math.round(photo.height * k);
  const t0 = performance.now();
  state.photoData = resizeImageData(photo, w, h);
  const up = resizeImageData(stylized, w, h);
  state.compositor = createCompositor(state.photoData, up);
  const ms = performance.now() - t0;
  log('compositor ready', `${w}x${h}`, Math.round(ms), 'ms');
  return ms;
}

function currentMask() {
  if (!$('chk-mask').checked || !state.mask) return null;
  if (!$('chk-invert').checked) return state.mask;
  if (!state.inverted) {
    const m = state.mask;
    const inv = new ImageData(m.width, m.height);
    for (let i = 0; i < m.data.length; i += 4) {
      const v = 255 - m.data[i];
      inv.data[i] = inv.data[i + 1] = inv.data[i + 2] = v; inv.data[i + 3] = 255;
    }
    state.inverted = inv;
  }
  return state.inverted;
}

function render(sharpen = $('chk-sharpen').checked) {
  if (!state.compositor) return null;
  const t0 = performance.now();
  const out = state.compositor.render({ intensity: Number($('rng-intensity').value), mask: currentMask(), sharpen });
  const ms = performance.now() - t0;
  draw('cv-out', out);
  return ms;
}

async function stylize() {
  if (!state.engine) await create();
  const photo = await ensurePhoto();
  state.abort = new AbortController();
  const style2 = $('sel-style2').value ? { key: $('sel-style2').value } : null;
  const t0 = performance.now();
  const { stylized } = await state.engine.stylize({
    content: photo, style: currentStyle(), style2, mix: Number($('rng-mix').value),
    signal: state.abort.signal, onProgress: (p) => { if (p.phase !== 'download') log('stylize', p.phase); },
  });
  const ms = Math.round(performance.now() - t0);
  state.stylized = stylized;
  draw('cv-raw', stylized);
  const compositorMs = rebuildCompositor();
  const renderMs = render();
  log('stylize', `${stylized.width}x${stylized.height}`, ms, 'ms; render', Math.round(renderMs), 'ms');
  showEngine();
  return { ms, width: stylized.width, height: stylized.height, compositorMs, renderMs, timings: state.engine.timings };
}

async function leakTest(runs = 5) {
  if (!state.engine) await create();
  const photo = await ensurePhoto();
  const counts = [];
  for (let i = 0; i < runs; i++) {
    await state.engine.stylize({ content: photo, style: currentStyle() });
    const tf = globalThis.tf;
    counts.push(tf ? tf.memory().numTensors : null);
  }
  const tf = globalThis.tf;
  const mem = tf ? tf.memory() : null;
  log('leak test numTensors after each run:', counts, 'numBytes', mem && mem.numBytes);
  return { counts, numBytes: mem && mem.numBytes, numBytesInGPU: mem && mem.numBytesInGPU };
}

async function segment() {
  const photo = await ensurePhoto();
  const t0 = performance.now();
  if (!state.segmenter) state.segmenter = await createSegmenter();
  const t1 = performance.now();
  const { subjectMask, foundLabels } = await state.segmenter.segment(photo);
  const t2 = performance.now();
  state.mask = subjectMask; state.inverted = null;
  draw('cv-mask', subjectMask);
  log('segment: load', Math.round(t1 - t0), 'ms, run', Math.round(t2 - t1), 'ms, delegate', state.segmenter.delegate, 'found', foundLabels);
  let white = 0;
  for (let i = 0; i < subjectMask.data.length; i += 4) if (subjectMask.data[i] > 127) white++;
  return { loadMs: t1 - t0, runMs: t2 - t1, foundLabels, delegate: state.segmenter.delegate,
    width: subjectMask.width, height: subjectMask.height, subjectShare: white / (subjectMask.width * subjectMask.height) };
}

async function palette() {
  const photo = await ensurePhoto();
  const k = Math.min(1, 1024 / Math.max(photo.width, photo.height));
  const content = resizeImageData(photo, Math.round(photo.width * k), Math.round(photo.height * k));
  let styleImg = state.styleImage;
  if (!styleImg) {
    const key = $('sel-style').value;
    const res = await fetch('/api/styles');
    const body = await res.json();
    const s = (Array.isArray(body) ? body : body.styles).find((x) => x.key === key);
    styleImg = await loadBitmap(s.image);
  }
  const k2 = Math.min(1, 512 / Math.max(styleImg.width, styleImg.height));
  const styleData = resizeImageData(styleImg, Math.round(styleImg.width * k2), Math.round(styleImg.height * k2));
  const t0 = performance.now();
  const out = paletteTransfer(content, styleData, { strength: Number($('rng-palette').value) });
  const ms = performance.now() - t0;
  draw('cv-out', out);
  log('palette transfer', `${out.width}x${out.height}`, Math.round(ms), 'ms');
  return { ms };
}

function sliderBench() {
  if (!state.compositor) return null;
  const fast = [], sharp = [];
  for (let i = 0; i <= 10; i++) {
    $('rng-intensity').value = String(i / 10);
    fast.push(Math.round(render(false)));
  }
  for (let i = 0; i < 3; i++) sharp.push(Math.round(render(true)));
  log('render ms sharpen:false', fast, 'sharpen:true', sharp);
  return { fast, sharp, size: [state.compositor.width, state.compositor.height] };
}

function guard(fn) {
  return async () => {
    try { await fn(); } catch (e) { log('ERROR', e.name, e.code || '', e.message); }
  };
}

$('btn-create').addEventListener('click', guard(() => create()));
$('btn-warmup').addEventListener('click', guard(warmup));
$('btn-server').addEventListener('click', guard(() => create('server')));
$('btn-stylize').addEventListener('click', guard(stylize));
$('btn-cancel').addEventListener('click', () => state.abort && state.abort.abort());
$('btn-leak').addEventListener('click', guard(leakTest));
$('btn-segment').addEventListener('click', guard(segment));
$('btn-palette').addEventListener('click', guard(palette));
$('rng-intensity').addEventListener('input', () => render(false));
$('rng-intensity').addEventListener('change', () => render(true));
for (const id of ['chk-mask', 'chk-invert', 'chk-sharpen']) $(id).addEventListener('change', () => render());
$('sel-photo').addEventListener('change', () => { state.photo = null; state.mask = null; });
$('file-photo').addEventListener('change', guard(async () => {
  const f = $('file-photo').files[0];
  if (f) { state.photo = await loadBitmap(f); state.mask = null; log('photo', f.name, state.photo.width, state.photo.height); }
}));
$('file-style').addEventListener('change', guard(async () => {
  const f = $('file-style').files[0];
  state.styleImage = f ? await loadBitmap(f) : null;
  log('custom style', f ? f.name : 'cleared');
}));
$('sel-style').addEventListener('change', () => { state.styleImage = null; });

// Automation hooks for scripted tests.
window.harness = {
  create, warmup, stylize, leakTest, segment, palette, sliderBench,
  setPhoto: async (url) => { $('sel-photo').value = url; state.photo = await loadBitmap(url); state.mask = null; },
  setStyle: (key) => { $('sel-style').value = key; state.styleImage = null; },
  setStyleImage: async (url) => { state.styleImage = await loadBitmap(url); },
  setStyle2: (key, mix) => { $('sel-style2').value = key || ''; $('rng-mix').value = String(mix ?? 0.5); },
  setIntensity: (v) => { $('rng-intensity').value = String(v); return render(); },
  useMask: (on, invert = false) => { $('chk-mask').checked = on; $('chk-invert').checked = invert; return render(); },
  engineInfo: () => state.engine && { kind: state.engine.kind, maxInferSide: state.engine.maxInferSide, timings: state.engine.timings },
};
loadStyles().then(() => { window.harnessReady = true; log('ready;', state.styles.length, 'styles'); });
