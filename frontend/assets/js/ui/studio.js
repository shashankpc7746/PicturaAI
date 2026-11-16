// Studio flow: photo in, painting out. Owns the result pixels, the live intensity
// slider, region painting, and the result actions (download, share, animate, save).
import { $, $$, h, announce, formatBytes, paintRange, rovingRadios, checkRadio, prefersReducedMotion } from '../core/dom.js';
import { store } from '../core/store.js';
import { createFrame } from './frame.js';
import { createMaskPainter } from './mask-painter.js';
import { toast } from './toasts.js';
import { artistLine } from '../services/styles-data.js';
import {
  checkFile, loadImageFile, loadImageUrl, toCanvas, imageDataToCanvas, canvasToImageData, canvasToBlob, blobToImage, WORK_SIDE,
} from '../services/image-io.js';
import {
  getEngine, createCompositor, paletteTransfer, resizeImageData, getSegmenter, stylizeFromPrompt,
} from '../services/engine-adapter.js';
import { exportSizes, exportBlob, saveBlob, fileName, canShareFiles, shareBlob } from '../services/export.js';
import { videoSupport, recordReveal } from '../services/video.js';
import { putPainting } from '../services/gallery-db.js';

const MODE_HINTS = {
  full: "The painting's brushwork and colours across the whole photo.",
  palette: "Only the painting's colours. Your photo keeps its detail.",
  region: 'Style only the parts you brush. The rest stays a photo.',
};

let frame;
let mask;
let current = null;        // { kind: 'nst'|'palette', content, stylized, compositor?, meta, sig }
let abortCtl = null;
let photoSeq = 0;
let warm = { promise: null, last: null, listener: null };
let gallery = null;
let shareOk = false;
const canvasIds = new WeakMap();
let canvasSeq = 0;

// ── Signatures: has anything changed since the result was painted? ─────────
function styleSig(s) {
  if (!s) return '';
  if (s.canvas && !canvasIds.has(s.canvas)) canvasIds.set(s.canvas, ++canvasSeq);
  return `${s.source}:${s.key || s.prompt || ''}:${s.canvas ? canvasIds.get(s.canvas) : ''}`;
}
function signature(st) {
  const mix = st.mixOpen && st.style2 ? `${st.style2.key}@${st.mix}` : '';
  return `${st.photo ? st.photo.id : 0}|${styleSig(st.style)}|${mix}|${st.mode === 'palette' ? 'p' : 'n'}`;
}
const isDirty = (st) => !st.result || st.result.sig !== signature(st);

function blocker(st) {
  if (st.phase === 'generating') return 'Painting';
  if (!st.photo) return 'Add a photo first';
  if (!st.style) return 'Pick a painting first';
  if (st.mode === 'region' && !st.maskHasContent) return 'Brush over the area to paint first';
  return '';
}

// ── Photo ───────────────────────────────────────────────────────────────
function setPhoto(img, info) {
  const canvas = toCanvas(img, WORK_SIDE);
  const prev = store.get().photo;
  if (prev && prev.url && prev.url.startsWith('blob:') && prev.url !== info.url) URL.revokeObjectURL(prev.url);
  const photo = { id: ++photoSeq, canvas, width: info.width || img.naturalWidth, height: info.height || img.naturalHeight, ...info };
  frame.photo.classList.remove('is-loading');
  frame.showPhoto(info.url, info.alt, canvas.width / canvas.height);
  frame.hideResult();
  frame.hideError();
  mask.reset(canvas.width, canvas.height);
  current = null;
  store.set({ photo, result: null, phase: 'idle', error: null, maskHasContent: false });
}

async function usePhotoFile(file) {
  const reason = checkFile(file);
  if (reason) { showInputError(reason); return; }
  frame.photo.classList.add('is-loading');
  try {
    const { img, url, width, height } = await loadImageFile(file);
    setPhoto(img, { url, width, height, name: file.name, bytes: file.size, sample: null, alt: `Your photo: ${file.name}` });
    markSample(null);
    prefetchEngine();
    announce('Photo added.');
  } catch (e) {
    frame.photo.classList.remove('is-loading');
    showInputError(e.message);
  }
}

function showInputError(message) {
  store.set({ error: { message, input: true } });
  frame.showError(message, { retryLabel: 'Choose another photo' });
}

let samples = {};
async function useSample(key) {
  const sm = samples[key];
  if (!sm) return;
  frame.photo.classList.add('is-loading');
  try {
    const img = await loadImageUrl(sm.url);
    setPhoto(img, { url: sm.url, name: sm.title, bytes: 0, sample: key, caption: sm.title, credit: sm.credit, alt: `Sample photo: ${sm.title}` });
    markSample(key);
  } catch {
    frame.photo.classList.remove('is-loading');
    toast("That sample couldn't be loaded. Try another one or upload your photo.", { type: 'error' });
  }
}

function markSample(key) {
  $$('[data-sample]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.sample === key)));
}

// ── Engine warm-up (lazy: only once someone shows intent) ────────────────
// Skipped on touch devices and data saver (the model is about 8 MB; phones load it
// on the first "Paint it" with visible progress), and when the engine would run on
// the page's main thread, where compiling shaders would freeze the page.
function prefetchEngine() {
  if (warm.promise || navigator.connection?.saveData || matchMedia('(pointer: coarse)').matches) return;
  const run = async () => {
    try {
      const engine = await getEngine();
      store.set({ engineKind: engine.kind });
      if (engine.kind !== 'browser' || engine.ready || !engine.offMainThread) return;
      warm.promise = engine.warmup((p) => { warm.last = p; if (warm.listener) warm.listener(p); });
      await warm.promise;
    } catch { warm.promise = null; /* painting will retry and fall back to the server */ }
  };
  if ('requestIdleCallback' in window) requestIdleCallback(run, { timeout: 3000 }); else setTimeout(run, 1200);
}

// ── Painting ────────────────────────────────────────────────────────────
function veilForProgress(p, state) {
  if (!p) return;
  const mb = (n) => (n / 1048576).toFixed(1);
  if (p.phase === 'download' && p.total > 1) {
    state.sub = `${mb(p.loaded)} of ${mb(p.total)} MB, only on the first visit`;
    frame.veil.text('Preparing the studio', state.sub);
    frame.veil.progress(p.loaded / p.total);
    state.downloading = true;
    return;
  }
  state.downloading = false;
  const text = { compile: 'Getting the brushes ready', style: 'Studying the painting', upload: 'Sending your photo', paint: 'Painting' }[p.phase];
  if (text) { frame.veil.text(text, state.sub = ''); frame.veil.progress(null); }
}

function styleArg(s) {
  if (!s) return null;
  if (s.source === 'preset') return { key: s.key };
  if (s.canvas) return { image: s.canvas };
  return null;
}

function metaFor(st, resolved = null) {
  const s = st.style;
  const meta = { source: s.source, title: s.title, artist: s.artist || '', year: s.year || '', key: s.key || null };
  if (resolved) Object.assign(meta, { source: 'preset', key: resolved.key, title: resolved.title });
  if (resolved) {
    const full = st.styles.find((x) => x.key === resolved.key);
    if (full) Object.assign(meta, { artist: full.artist, year: full.year });
  }
  if (st.mixOpen && st.style2) meta.mixWith = st.style2.title;
  return meta;
}

async function runStyle(st, signal, onProgress) {
  const content = st.photo.canvas;
  const style2 = st.mixOpen && st.style2 ? { key: st.style2.key } : null;
  let stylized; let resolved = null;
  if (st.style.source === 'prompt') {
    ({ stylized, resolved } = await stylizeFromPrompt({ content, prompt: st.style.prompt, signal }));
    store.set({ engineKind: 'server' });
  } else {
    let engine = await getEngine();
    store.set({ engineKind: engine.kind });
    const attempt = async (eng) => {
      if (eng.kind === 'browser' && !eng.ready) {
        const wp = warm.promise || eng.warmup((p) => { warm.last = p; if (warm.listener) warm.listener(p); });
        warm.promise = wp;
        if (warm.last) onProgress(warm.last);
        await Promise.race([wp, new Promise((_, rej) => signal.addEventListener('abort',
          () => rej(new DOMException('Cancelled', 'AbortError')), { once: true }))]);
      }
      return eng.stylize({ content, style: styleArg(st.style), style2, mix: st.mix, signal, onProgress });
    };
    try {
      ({ stylized } = await attempt(engine));
    } catch (e) {
      if (e.name === 'AbortError' || engine.kind !== 'browser' || e.code === 'ENGINE_INPUT') throw e;
      warm.promise = null;
      engine = await getEngine({ prefer: 'server' });
      store.set({ engineKind: engine.kind });
      toast("This device couldn't paint it, so our server is painting it instead.");
      ({ stylized } = await attempt(engine));
    }
  }
  const contentData = await resizeImageData(content, stylized.width, stylized.height);
  const compositor = await createCompositor(contentData, stylized);
  return { kind: 'nst', content: contentData, stylized, compositor, meta: metaFor(st, resolved) };
}

async function styleImageData(s, styles) {
  if (s.canvas) return canvasToImageData(toCanvas(s.canvas, 512));
  const preset = styles.find((x) => x.key === s.key);
  if (!preset) throw new Error('Pick a painting from the gallery for colours only.');
  return canvasToImageData(toCanvas(await loadImageUrl(preset.image), 512));
}

async function runPalette(st) {
  let style = st.style;
  if (style.source === 'prompt') throw new Error('Colours only needs a painting. Pick one from the gallery or upload your own.');
  const content = canvasToImageData(toCanvas(st.photo.canvas, 1024));
  const sData = await styleImageData(style, st.styles);
  const s2Data = st.mixOpen && st.style2 ? await styleImageData(st.style2, st.styles) : null;
  const opts = { style2ImageData: s2Data, mix: st.mix };
  const stylized = await paletteTransfer(content, sData, { ...opts, strength: 1 });
  return { kind: 'palette', content, stylized, styleData: sData, opts, meta: metaFor(st) };
}

export async function paint() {
  const st = store.get();
  if (blocker(st)) return;
  abortCtl = new AbortController();
  const { signal } = abortCtl;
  const t0 = performance.now();
  const vstate = { sub: '', downloading: false };
  frame.hideError();
  frame.veil.show(st.mode === 'palette' ? 'Matching colours' : 'Painting');
  store.set({ phase: 'generating', error: null });
  announce('Painting started.');
  const onProgress = (p) => veilForProgress(p, vstate);
  warm.listener = onProgress;
  const timer = setInterval(() => {
    if (vstate.downloading) return;
    const secs = Math.round((performance.now() - t0) / 1000);
    const asleep = store.get().engineKind === 'server' && secs >= 8;
    const elapsed = `${secs} ${secs === 1 ? 'second' : 'seconds'}`;
    frame.veil.text(null, asleep ? `${elapsed}. The server was asleep, so the first picture can take up to a minute.` : elapsed);
  }, 1000);
  try {
    const out = st.mode === 'palette' ? await runPalette(st) : await runStyle(st, signal, onProgress);
    if (signal.aborted) throw new DOMException('Cancelled', 'AbortError');
    out.sig = signature(st);
    showResult(out, { unveil: true });
  } catch (e) {
    if (e && e.name === 'AbortError') {
      store.set({ phase: current ? 'result' : 'idle' });
      toast('Painting cancelled.');
    } else {
      if (!e || (!e.code && !e.status)) console.error(e);   // expected server/engine errors are shown, not logged
      const message = (e && e.message) || 'Something went wrong. Please try again.';
      const smaller = e && (e.code === 'ENGINE_RESOURCE' || e.status === 413 || /too large|memory/i.test(message));
      store.set({ phase: 'error', error: { message, smaller } });
      frame.showError(message, { smaller });
    }
  } finally {
    clearInterval(timer);
    warm.listener = null;
    frame.veil.hide();
    abortCtl = null;
  }
}

function showResult(out, { unveil = false, saved = null } = {}) {
  current = out;
  const st = store.get();
  store.set({
    phase: 'result',
    error: null,
    result: { sig: out.sig, kind: out.kind, meta: out.meta, savedId: saved, mode: st.mode },
  });
  render(true);
  frame.showResult({ unveil, compareEnabled: st.mode !== 'region' });
  frame.setCompare(0);
  updateLabel(true);
  updateSizes();
  announce(`Your painting is ready: after ${out.meta.title}.`);
  const r = frame.el.getBoundingClientRect();
  if (r.top < 0 || r.top > innerHeight * 0.5) {
    frame.el.scrollIntoView({ block: 'start', behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
  }
}

// ── Rendering (live) ─────────────────────────────────────────────────────
function lerp(a, b, t) {
  const out = new ImageData(a.width, a.height);
  const o = out.data; const x = a.data; const y = b.data;
  for (let i = 0; i < o.length; i++) o[i] = x[i] + (y[i] - x[i]) * t;
  return out;
}

function compose(final, intensity = store.get().intensity) {
  const st = store.get();
  if (current.kind === 'nst') {
    return current.compositor.render({ intensity, mask: st.mode === 'region' ? mask.getMask() : null, sharpen: final });
  }
  return final
    ? null   // computed async in render()
    : lerp(current.content, current.stylized, intensity);
}

let rafId = 0;
let paletteToken = 0;
function render(final) {
  if (!current) return;
  cancelAnimationFrame(rafId);
  rafId = 0;
  if (current.kind === 'palette' && final) {
    const token = ++paletteToken;
    paletteTransfer(current.content, current.styleData, { ...current.opts, strength: store.get().intensity })
      .then((img) => { if (token === paletteToken && current && current.kind === 'palette') imageDataToCanvas(img, frame.result); });
    return;
  }
  imageDataToCanvas(compose(final), frame.result);
}
function renderSoon() {
  if (!rafId) rafId = requestAnimationFrame(() => { rafId = 0; if (current) imageDataToCanvas(compose(false), frame.result); });
}

// ── Wall label ───────────────────────────────────────────────────────────
function updateLabel(fresh = false) {
  const st = store.get();
  const title = $('#label-title');
  const l1 = $('#label-line1');
  const l2 = $('#label-line2');
  const label = $('#wall-label');
  if (st.phase === 'result' || (current && st.result)) {
    const m = st.result.meta;
    const pct = `${Math.round(st.intensity * 100)}%`;
    if (m.source === 'preset') {
      title.replaceChildren(document.createTextNode(st.result.kind === 'palette' ? 'In the colours of ' : 'After '), h('em', { text: m.title }));
    } else if (m.source === 'custom') {
      title.textContent = st.result.kind === 'palette' ? 'In the colours of your painting' : 'After your painting';
    } else {
      title.textContent = `After ${m.title}`;
    }
    const who = m.source === 'preset' ? artistLine(m) : m.source === 'custom' ? 'Your own painting' : 'A style made from your words';
    l1.textContent = m.mixWith ? `${who}, mixed with ${m.mixWith}` : who;
    l2.textContent = st.result.kind === 'palette' ? `Colour strength ${pct}`
      : st.mode === 'region' ? `Intensity ${pct}, brushed areas only` : `Intensity ${pct}`;
  } else if (st.photo) {
    if (st.photo.sample) {
      title.textContent = 'Sample photo';
      l1.textContent = st.photo.caption || '';
      l2.textContent = st.photo.credit || '';
    } else {
      title.textContent = 'Your photo';
      l1.textContent = st.photo.name || '';
      l2.textContent = `${st.photo.width} × ${st.photo.height} pixels${st.photo.bytes ? `, ${formatBytes(st.photo.bytes)}` : ''}`;
    }
  } else {
    title.textContent = 'No photo yet';
    l1.textContent = 'Upload a photo to begin.';
    l2.textContent = '';
  }
  if (fresh) {
    label.classList.remove('is-fresh');
    void label.offsetWidth;
    label.classList.add('is-fresh');
  }
}

// ── Buttons and bar ──────────────────────────────────────────────────────
function updateButtons() {
  const st = store.get();
  const reason = blocker(st);
  const generating = st.phase === 'generating';
  const dirty = isDirty(st);
  const paintBtn = $('#paint-btn');
  paintBtn.disabled = !!reason;
  $('#paint-reason').textContent = generating ? '' : reason;

  const showActions = !!current && st.result && st.phase !== 'generating';
  $('#result-actions').hidden = !showActions;
  const barShowsResult = showActions && !dirty && st.phase === 'result';
  $('#bar-paint').hidden = barShowsResult;
  $('#bar-paint').disabled = !!reason;
  $('#bar-download').hidden = !barShowsResult;
  $('#bar-share').hidden = !barShowsResult || !shareOk;
  $('#bar-reason').textContent = barShowsResult || generating ? '' : reason;
  const save = $('#save-btn');
  const saved = !!(st.result && st.result.savedId);
  save.disabled = saved;
  save.querySelector('span').textContent = saved ? 'Saved' : 'Save to my gallery';
}

function updateRegion() {
  const st = store.get();
  const region = st.mode === 'region';
  $('#region-tools').hidden = !region;
  mask.setEnabled(region && !!st.photo && st.phase !== 'generating');
  frame.setCompareEnabled(!region && !!current);
  $('#mode-hint').textContent = MODE_HINTS[st.mode];
}

// ── Download menu ────────────────────────────────────────────────────────
function updateSizes() {
  const c = frame.result;
  if (!c.width) return;
  const sizes = exportSizes(c.width, c.height);
  for (const [k, v] of Object.entries(sizes)) {
    const el = document.querySelector(`[data-size-label="${k}"]`);
    if (el) el.textContent = `${v.width} × ${v.height}${v.crop ? ', cropped' : ''}`;
  }
}

function initDownloadMenu() {
  const menu = $('#download-menu');
  const items = $$('.menu__item', menu);
  let trigger = null;

  function place() {
    const r = trigger.getBoundingClientRect();
    const w = menu.offsetWidth;
    const left = Math.max(8, Math.min(r.right - w, innerWidth - w - 8));
    menu.style.left = `${Math.max(8, r.left + w > innerWidth - 8 ? left : r.left)}px`;
    if (r.top > innerHeight / 2) {
      menu.style.top = '';
      menu.style.bottom = `${innerHeight - r.top + 8}px`;
    } else {
      menu.style.bottom = '';
      menu.style.top = `${r.bottom + 8}px`;
    }
  }
  function open(btn) {
    trigger = btn;
    updateSizes();
    menu.hidden = false;
    place();
    btn.setAttribute('aria-expanded', 'true');
    items[0].focus();
    setTimeout(() => document.addEventListener('pointerdown', outside), 0);
    window.addEventListener('resize', close, { once: true });
  }
  function close(refocus = true) {
    if (menu.hidden) return;
    menu.hidden = true;
    document.removeEventListener('pointerdown', outside);
    if (trigger) {
      trigger.setAttribute('aria-expanded', 'false');
      if (refocus === true) trigger.focus();
    }
  }
  function outside(e) { if (!menu.contains(e.target) && e.target !== trigger && !trigger.contains(e.target)) close(false); }

  for (const btn of [$('#download-btn'), $('#bar-download')]) {
    btn.addEventListener('click', () => (menu.hidden ? open(btn) : close()));
  }
  menu.addEventListener('keydown', (e) => {
    const i = items.indexOf(document.activeElement);
    if (e.key === 'Escape') { e.preventDefault(); close(); }
    else if (e.key === 'ArrowDown') { e.preventDefault(); items[(i + 1) % items.length].focus(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); }
    else if (e.key === 'Tab') close(false);
  });
  menu.addEventListener('click', async (e) => {
    const item = e.target.closest('[data-size]');
    if (!item) return;
    close();
    await download(item.dataset.size);
  });
}

async function finalCanvas() {
  if (current.kind === 'palette') {
    const img = await paletteTransfer(current.content, current.styleData, { ...current.opts, strength: store.get().intensity });
    return imageDataToCanvas(img, document.createElement('canvas'));
  }
  return imageDataToCanvas(compose(true), document.createElement('canvas'));
}

function resultTitle() {
  const m = store.get().result.meta;
  return m.source === 'custom' ? 'After your painting' : `After ${m.title}`;
}

async function download(kind) {
  if (!current) return;
  try {
    const src = await finalCanvas();
    const size = exportSizes(src.width, src.height)[kind];
    const blob = await exportBlob(src, size);
    saveBlob(blob, fileName(resultTitle(), kind === 'full' ? '' : kind));
    toast(`Downloaded, ${size.width} × ${size.height} pixels.`);
  } catch (e) {
    toast(e.message || "The download couldn't be created.", { type: 'error' });
  }
}

async function share() {
  if (!current) return;
  try {
    const src = await finalCanvas();
    const blob = await exportBlob(src, { width: src.width, height: src.height });
    await shareBlob(blob, fileName(resultTitle()), resultTitle());
  } catch (e) {
    if (e && e.name === 'AbortError') return;
    toast("Sharing didn't work here. Download the picture and share it from your files.", { type: 'error' });
  }
}

// ── Animate ──────────────────────────────────────────────────────────────
async function animate(btn) {
  if (!current || btn.disabled) return;
  const label = btn.querySelector('span');
  btn.disabled = true;
  btn.classList.add('is-busy');
  try {
    const st = store.get();
    const src = current.content;
    const k = Math.min(1, 720 / Math.max(src.width, src.height));
    const w = Math.max(2, Math.round((src.width * k) / 2) * 2);
    const hh = Math.max(2, Math.round((src.height * k) / 2) * 2);
    const c = await resizeImageData(current.content, w, hh);
    const s = await resizeImageData(current.stylized, w, hh);
    let frameAt;
    if (current.kind === 'nst') {
      const comp = await createCompositor(c, s);
      const m = st.mode === 'region' ? mask.getMask() : null;
      frameAt = (t) => comp.render({ intensity: t * st.intensity, mask: m, sharpen: false });
    } else {
      frameAt = (t) => lerp(c, s, t * st.intensity);
    }
    const { blob, ext } = await recordReveal({
      width: w, height: hh, frame: frameAt,
      onProgress: (f) => { label.textContent = `Recording ${Math.round(f * 100)}%`; },
    });
    saveBlob(blob, fileName(resultTitle(), 'reveal', ext));
    toast(`Video downloaded, ${ext.toUpperCase()}.`);
  } catch (e) {
    toast(e.message || "The video couldn't be recorded.", { type: 'error' });
  } finally {
    label.textContent = 'Animate';
    btn.disabled = false;
    btn.classList.remove('is-busy');
  }
}

// ── Save to my gallery / open from it ────────────────────────────────────
async function save() {
  const st = store.get();
  if (!current || !st.result || st.result.savedId) return;
  try {
    const src = await finalCanvas();
    const id = (crypto.randomUUID && crypto.randomUUID()) || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const toBlob = (img, q = 0.9) => canvasToBlob(imageDataToCanvas(img, document.createElement('canvas')), 'image/jpeg', q);
    const rec = {
      id,
      createdAt: Date.now(),
      title: resultTitle(),
      meta: st.result.meta,
      kind: current.kind,
      mode: st.mode,
      intensity: st.intensity,
      width: src.width,
      height: src.height,
      image: await canvasToBlob(src, 'image/jpeg', 0.92),
      thumb: await canvasToBlob(toCanvas(src, 480), 'image/jpeg', 0.85),
      photo: await toBlob(current.content),
      stylized: await toBlob(current.stylized, 0.95),
      mask: st.mode === 'region' ? await canvasToBlob(mask.toCanvas(), 'image/png') : null,
      paletteOpts: current.kind === 'palette' ? { mix: current.opts.mix } : null,
      styleData: current.kind === 'palette' ? await toBlob(current.styleData) : null,
    };
    await putPainting(rec);
    store.set({ result: { ...store.get().result, savedId: id } });
    if (gallery) gallery.refresh();
    toast('Saved to my gallery.', { action: { label: 'View', run: () => $('#gallery').scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth' }) } });
  } catch (e) {
    toast(e.message || "The painting couldn't be saved.", { type: 'error' });
  }
}

const blobData = async (blob) => canvasToImageData(toCanvas(await blobToImage(blob)));

export async function openSaved(rec) {
  if (store.get().phase === 'generating') { toast('Wait for the current painting to finish.'); return; }
  try {
    const content = await blobData(rec.photo);
    const stylized = await blobData(rec.stylized);
    const url = URL.createObjectURL(rec.photo);
    const img = await loadImageUrl(url);
    setPhoto(img, { url, name: rec.title, bytes: 0, sample: null, alt: `Photo used for ${rec.title}` });
    markSample(null);
    let out;
    if (rec.kind === 'palette' && rec.styleData) {
      out = { kind: 'palette', content, stylized, styleData: await blobData(rec.styleData), opts: { mix: rec.paletteOpts?.mix ?? 0.5 }, meta: rec.meta };
    } else {
      out = { kind: 'nst', content, stylized, compositor: await createCompositor(content, stylized), meta: rec.meta };
    }
    if (rec.mask) mask.setFromImageData(await blobData(rec.mask));
    const mode = rec.kind === 'palette' ? 'palette' : rec.mask ? 'region' : 'full';
    setIntensity(rec.intensity);
    setMode(mode, { silent: true });
    if (rec.meta && rec.meta.source === 'preset') {
      const s = store.get().styles.find((x) => x.key === rec.meta.key);
      if (s) store.set({ style: { source: 'preset', key: s.key, title: s.title, artist: s.artist, year: s.year, image: s.image } });
    }
    store.set({ maskHasContent: !!rec.mask });
    out.sig = signature(store.get());
    $('#studio').scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
    showResult(out, { unveil: true, saved: rec.id });
  } catch (e) {
    console.error(e);
    toast("That painting couldn't be opened.", { type: 'error' });
  }
}

// ── Controls ─────────────────────────────────────────────────────────────
function setIntensity(v) {
  const input = $('#intensity');
  input.value = String(Math.round(v * 100));
  paintRange(input);
  $('#intensity-value').textContent = `${input.value}%`;
  store.set({ intensity: v });
}

function setMode(mode, { silent = false } = {}) {
  const items = $$('#mode .segmented__item');
  checkRadio(items, items.find((el) => el.dataset.mode === mode));
  const prev = store.get().mode;
  if (prev === mode) return;
  store.set({ mode });
  if (silent || !current) return;
  // Full style and Paint a region share the same painting: just re-render.
  if ((mode === 'palette') !== (prev === 'palette')) {
    const st = store.get();
    if (mode === 'palette' && !blocker(st)) paint();
    return;
  }
  render(true);
  updateLabel();
}

function initControls() {
  rovingRadios($('#mode'), '.segmented__item', (el) => setMode(el.dataset.mode));

  const intensity = $('#intensity');
  paintRange(intensity);
  intensity.addEventListener('input', () => {
    paintRange(intensity);
    $('#intensity-value').textContent = `${intensity.value}%`;
    store.set({ intensity: Number(intensity.value) / 100 });
    renderSoon();
    if (current) updateLabel();
  });
  intensity.addEventListener('change', () => { render(true); updateLabel(); });

  const brushItems = $$('[data-brush]');
  rovingRadios(brushItems[0].parentElement, '[data-brush]', (el) => {
    checkRadio(brushItems, el);
    mask.setBrush(el.dataset.brush);
  });
  const size = $('#brush-size');
  paintRange(size);
  size.addEventListener('input', () => {
    paintRange(size);
    $('#brush-size-value').textContent = `${size.value} px`;
    mask.setSize(Number(size.value));
  });
  $('#mask-clear').addEventListener('click', () => mask.clear());
  $('#mask-invert').addEventListener('click', () => mask.invert());
  $('#seg-subject').addEventListener('click', (e) => segment(e.currentTarget, false));
  $('#seg-background').addEventListener('click', (e) => segment(e.currentTarget, true));

  $('#paint-btn').addEventListener('click', paint);
  $('#bar-paint').addEventListener('click', paint);
  // The reason in the bar takes you to what is missing.
  $('#bar-reason').addEventListener('click', () => {
    const st = store.get();
    const smooth = prefersReducedMotion() ? 'auto' : 'smooth';
    if (!st.photo) { $('#photo-input').click(); return; }
    if (!st.style) {
      $('#style-panel').scrollIntoView({ block: 'start', behavior: smooth });
      const card = $('.style-card[tabindex="0"]') || $('.style-card[data-key]');
      if (card) card.focus({ preventScroll: true });
      return;
    }
    frame.el.scrollIntoView({ block: 'center', behavior: smooth });
  });
  $('#cancel-paint').addEventListener('click', () => abortCtl && abortCtl.abort());
  $('#error-retry').addEventListener('click', () => {
    const err = store.get().error;
    frame.hideError();
    if (err && err.input) { store.set({ error: null }); $('#photo-input').click(); return; }
    paint();
  });
  $('#error-dismiss').addEventListener('click', () => {
    frame.hideError();
    store.set({ phase: current ? 'result' : 'idle', error: null });
  });
  $('#error-smaller').addEventListener('click', () => {
    const st = store.get();
    if (!st.photo) return;
    const small = toCanvas(st.photo.canvas, 768);
    store.set({ photo: { ...st.photo, canvas: small } });
    frame.hideError();
    paint();
  });

  $('#upload-photo').addEventListener('click', () => $('#photo-input').click());
  $('#photo-input').addEventListener('change', (e) => {
    const f = e.target.files[0];
    e.target.value = '';
    if (f) usePhotoFile(f);
  });
  $$('[data-sample]').forEach((b) => b.addEventListener('click', () => useSample(b.dataset.sample)));

  shareOk = canShareFiles();
  $('#share-btn').hidden = !shareOk;
  $('#share-btn').addEventListener('click', share);
  $('#bar-share').addEventListener('click', share);
  if (!videoSupport()) $('#animate-btn').hidden = true;
  $('#animate-btn').addEventListener('click', (e) => animate(e.currentTarget));
  $('#save-btn').addEventListener('click', save);
  initDownloadMenu();
}

async function segment(btn, invert) {
  const st = store.get();
  if (!st.photo) return;
  const label = btn.lastChild;
  const original = label.textContent;
  btn.disabled = true;
  btn.classList.add('is-busy');
  label.textContent = ' Finding the subject';
  try {
    const seg = await getSegmenter();
    const { subjectMask, foundLabels } = await seg.segment(st.photo.canvas);
    if (!foundLabels || !foundLabels.length) {
      toast("We couldn't find a clear subject. Brush over it instead.");
    } else {
      mask.setFromImageData(subjectMask, { invert });
      const what = foundLabels.slice(0, 2).join(' and ');
      toast(invert ? `Selected everything except the ${what}.` : `Selected the ${what}.`);
    }
  } catch (e) {
    console.warn(e);
    toast("Subject detection isn't available in this browser. Brush over the area instead.", { type: 'error' });
  } finally {
    btn.disabled = false;
    btn.classList.remove('is-busy');
    label.textContent = original;
  }
}

// ── Boot ─────────────────────────────────────────────────────────────────
async function initSamples() {
  const base = 'assets/samples/';
  const defaults = { dog: 'Golden retriever in a forest', city: 'Old town street in Innsbruck', lake: 'Clouds mirrored in a mountain lake' };
  let credits = [];
  try {
    const res = await fetch(`${base}credits.json`);
    if (res.ok) credits = await res.json();
  } catch { /* captions fall back to defaults */ }
  for (const key of Object.keys(defaults)) {
    const c = Array.isArray(credits) ? credits.find((x) => x.file === `${key}.jpg`) : null;
    samples[key] = {
      url: `${base}${key}.jpg`,
      title: (c && c.title) || defaults[key],
      credit: c ? `Photo by ${c.author}, ${c.license}` : 'Public-domain sample photo',
    };
  }
  // Hide chips whose sample is missing.
  await Promise.all(Object.keys(samples).map(async (key) => {
    try {
      const res = await fetch(samples[key].url, { method: 'HEAD' });
      if (!res.ok) throw new Error();
    } catch {
      delete samples[key];
      const chip = document.querySelector(`[data-sample="${key}"]`);
      if (chip) chip.hidden = true;
    }
  }));
  if (!Object.keys(samples).length) $('#samples').hidden = true;
}

async function initFirstPhoto() {
  const img = frame.photo;
  const loaded = img.complete ? img.naturalWidth > 0
    : await new Promise((resolve) => {
      img.addEventListener('load', () => resolve(true), { once: true });
      img.addEventListener('error', () => resolve(false), { once: true });
    });
  await samplesReady;
  if (loaded && samples.dog) {
    setPhoto(img, { url: img.getAttribute('src'), name: samples.dog.title, bytes: 0, sample: 'dog', caption: samples.dog.title, credit: samples.dog.credit, alt: img.alt });
  } else {
    frame.showEmpty();
    img.removeAttribute('src');
    markSample(null);
    updateLabel();
  }
}

let samplesReady;
export function initStudio({ galleryApi } = {}) {
  gallery = galleryApi || null;
  frame = createFrame({ onFiles: usePhotoFile });
  mask = createMaskPainter($('#frame-mask'), {
    onChange: () => { if (current && current.kind === 'nst') renderSoon(); },
    onCommit: () => {
      store.set({ maskHasContent: mask.hasContent });
      if (current && current.kind === 'nst') render(true);
    },
  });
  initControls();
  samplesReady = initSamples();
  initFirstPhoto();

  store.subscribe((st, prev) => {
    if (st.style !== prev.style && st.style) prefetchEngine();
    updateButtons();
    if (st.mode !== prev.mode || st.photo !== prev.photo || st.phase !== prev.phase) updateRegion();
    if (st.photo !== prev.photo || st.phase !== prev.phase || st.result !== prev.result) updateLabel();
  });
  updateButtons();
  updateRegion();
  return { paint, openSaved, setGallery: (g) => { gallery = g; } };
}
