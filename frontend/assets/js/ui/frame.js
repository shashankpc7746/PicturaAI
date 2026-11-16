// The gilt frame: photo, result canvas, compare slider, veil, error and drop states.
import { $, prefersReducedMotion } from '../core/dom.js';

export function createFrame({ onFiles }) {
  const frame = $('#frame');
  const stage = $('#frame-stage');
  const photo = $('#frame-photo');
  const result = $('#frame-result');
  const compare = $('#compare');
  const handle = $('#compare-handle');
  const tag = $('#tag-original');
  const empty = $('#frame-empty');
  const veil = $('#veil');
  const veilText = $('#veil-text');
  const veilSub = $('#veil-sub');
  const veilProgress = $('#veil-progress');
  const veilBar = $('#veil-bar');
  const error = $('#frame-error');
  const errorText = $('#frame-error-text');
  const errorSmaller = $('#error-smaller');
  const studio = $('#studio');

  let pos = 0;

  function setAspect(ar) {
    if (ar > 0) studio.style.setProperty('--ar', String(Math.round(ar * 10000) / 10000));
  }

  function showPhoto(url, alt, ar) {
    setAspect(ar);
    frame.classList.remove('is-empty');
    empty.hidden = true;
    if (photo.getAttribute('src') !== url) photo.src = url;
    photo.alt = alt;
  }

  function showEmpty() {
    frame.classList.add('is-empty');
    empty.hidden = false;
    result.hidden = true;
    compare.hidden = true;
    tag.hidden = true;
  }

  // ── Compare ──────────────────────────────────────────────────────────
  function setCompare(pct, { announce = true } = {}) {
    pos = Math.max(0, Math.min(100, Math.round(pct)));
    stage.style.setProperty('--pos', `${pos}%`);
    handle.setAttribute('aria-valuenow', String(pos));
    if (announce) {
      handle.setAttribute('aria-valuetext', pos === 0 ? 'Showing the painting'
        : pos === 100 ? 'Showing your photo' : `Your photo on the left ${pos}%, painting on the right`);
    }
    tag.hidden = compare.hidden || pos < 12;
  }

  function pctFromEvent(e) {
    const r = stage.getBoundingClientRect();
    return ((e.clientX - r.left) / r.width) * 100;
  }

  handle.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    compare.classList.remove('is-intro');
    const move = (ev) => setCompare(pctFromEvent(ev));
    const up = () => {
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('pointerup', up);
      handle.removeEventListener('pointercancel', up);
    };
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', up);
    handle.addEventListener('pointercancel', up);
  });
  handle.addEventListener('keydown', (e) => {
    const step = { ArrowLeft: -5, ArrowDown: -5, ArrowRight: 5, ArrowUp: 5, PageDown: -25, PageUp: 25 }[e.key];
    if (step != null) { e.preventDefault(); setCompare(pos + step); }
    else if (e.key === 'Home') { e.preventDefault(); setCompare(0); }
    else if (e.key === 'End') { e.preventDefault(); setCompare(100); }
  });

  // ── Result ───────────────────────────────────────────────────────────
  function showResult({ unveil = false, compareEnabled = true } = {}) {
    result.hidden = false;
    setCompareEnabled(compareEnabled);
    if (unveil) {
      result.classList.remove('is-unveiling');
      void result.offsetWidth;   // restart the animation
      result.classList.add('is-unveiling');
      const done = () => result.classList.remove('is-unveiling');
      result.addEventListener('animationend', done, { once: true });
      if (compareEnabled && !prefersReducedMotion()) {
        compare.classList.remove('is-intro');
        setTimeout(() => compare.classList.add('is-intro'), 750);
      }
    }
  }

  function hideResult() {
    result.hidden = true;
    compare.hidden = true;
    tag.hidden = true;
  }

  function setCompareEnabled(on) {
    compare.hidden = !on || result.hidden;
    if (compare.hidden) { stage.style.setProperty('--pos', '0%'); tag.hidden = true; }
    else setCompare(pos, { announce: true });
  }

  // ── Veil (generating) ────────────────────────────────────────────────
  const veilApi = {
    show(text = 'Painting') {
      veil.hidden = false;
      veilText.textContent = text;
      veilSub.textContent = '';
      this.progress(null);
    },
    text(main, sub = '') {
      if (main != null) veilText.textContent = main;
      veilSub.textContent = sub;
    },
    /** fraction 0..1, or null for indeterminate */
    progress(fraction) {
      if (fraction == null) {
        veilProgress.classList.add('is-indeterminate');
        veilBar.style.width = '';
        veilProgress.removeAttribute('aria-valuenow');
      } else {
        const p = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
        veilProgress.classList.remove('is-indeterminate');
        veilBar.style.width = `${p}%`;
        veilProgress.setAttribute('aria-valuenow', String(p));
      }
    },
    hide() { veil.hidden = true; },
  };

  // ── Error ────────────────────────────────────────────────────────────
  function showError(message, { smaller = false, retryLabel = 'Try again' } = {}) {
    errorText.textContent = message;
    errorSmaller.hidden = !smaller;
    $('#error-retry-text').textContent = retryLabel;
    error.hidden = false;
    $('#error-retry').focus({ preventScroll: true });
  }
  const hideError = () => { error.hidden = true; };

  // ── Drag and drop anywhere on the frame ──────────────────────────────
  let depth = 0;
  const hasFiles = (e) => Array.from(e.dataTransfer?.types || []).includes('Files');
  frame.addEventListener('dragenter', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth++;
    frame.classList.add('is-dragover');
  });
  frame.addEventListener('dragover', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  });
  frame.addEventListener('dragleave', () => {
    depth = Math.max(0, depth - 1);
    if (!depth) frame.classList.remove('is-dragover');
  });
  frame.addEventListener('drop', (e) => {
    e.preventDefault();
    depth = 0;
    frame.classList.remove('is-dragover');
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) onFiles(file);
  });
  // A photo dropped beside the frame should not navigate away from the studio.
  window.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener('drop', (e) => { if (hasFiles(e)) e.preventDefault(); });

  return {
    el: frame, stage, photo, result,
    setAspect, showPhoto, showEmpty, showResult, hideResult, setCompare, setCompareEnabled,
    veil: veilApi, showError, hideError,
    get comparePos() { return pos; },
  };
}
