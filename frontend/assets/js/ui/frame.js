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
  const tagPainting = $('#tag-painting');
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
  // The first result of a visit slides the divider out to the middle once, so it is
  // clear that the frame compares the photo with the painting. Any touch of the
  // slider stops it and keeps the user's position from then on.
  let introduced = false;
  let sweepTimer = 0;
  let sweepFrame = 0;
  function stopSweep() {
    clearTimeout(sweepTimer);
    cancelAnimationFrame(sweepFrame);
    sweepTimer = 0;
    sweepFrame = 0;
  }
  function sweepTo(target, delay) {
    stopSweep();
    if (prefersReducedMotion()) { setCompare(target); return; }
    sweepTimer = setTimeout(() => {
      const from = pos;
      const start = performance.now();
      const dur = 700;
      const step = (now) => {
        const t = Math.min(1, (now - start) / dur);
        const eased = 1 - Math.pow(1 - t, 3);
        setCompare(from + (target - from) * eased, { announce: t === 1 });
        sweepFrame = t < 1 ? requestAnimationFrame(step) : 0;
      };
      sweepFrame = requestAnimationFrame(step);
    }, delay);
  }

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
    tagPainting.hidden = true;
    clearResultLabel();
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
    tagPainting.hidden = compare.hidden || pos > 88;
    compare.dataset.at = pos === 0 ? 'start' : pos === 100 ? 'end' : 'mid';
  }

  function pctFromEvent(e) {
    const r = stage.getBoundingClientRect();
    return ((e.clientX - r.left) / r.width) * 100;
  }

  handle.addEventListener('focus', () => { introduced = true; stopSweep(); });
  handle.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    introduced = true;
    stopSweep();
    handle.setPointerCapture(e.pointerId);
    compare.classList.remove('is-intro');
    // .is-dragging (app.css) keeps the held look while the pointer is off the handle.
    compare.classList.add('is-dragging');
    const move = (ev) => setCompare(pctFromEvent(ev));
    const up = () => {
      compare.classList.remove('is-dragging');
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('pointerup', up);
      handle.removeEventListener('pointercancel', up);
      handle.removeEventListener('lostpointercapture', up);
    };
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', up);
    handle.addEventListener('pointercancel', up);
    handle.addEventListener('lostpointercapture', up);
  });
  handle.addEventListener('keydown', (e) => {
    const step = { ArrowLeft: -5, ArrowDown: -5, ArrowRight: 5, ArrowUp: 5, PageDown: -25, PageUp: 25 }[e.key];
    if (step != null) { e.preventDefault(); setCompare(pos + step); }
    else if (e.key === 'Home') { e.preventDefault(); setCompare(0); }
    else if (e.key === 'End') { e.preventDefault(); setCompare(100); }
  });

  // ── Result ───────────────────────────────────────────────────────────
  // The canvas covers the photo, so it carries the text alternative and the photo
  // is hidden from assistive tech while it is covered.
  function setResultLabel(label) {
    if (result.hidden || !label) return;
    result.setAttribute('role', 'img');
    result.setAttribute('aria-label', label);
    photo.setAttribute('aria-hidden', 'true');
  }
  function clearResultLabel() {
    result.removeAttribute('role');
    result.removeAttribute('aria-label');
    photo.removeAttribute('aria-hidden');
  }

  function showResult({ unveil = false, compareEnabled = true, label = '' } = {}) {
    result.hidden = false;
    setResultLabel(label);
    setCompareEnabled(compareEnabled);
    if (unveil) {
      result.classList.remove('is-unveiling');
      void result.offsetWidth;   // restart the animation
      result.classList.add('is-unveiling');
      const done = () => result.classList.remove('is-unveiling');
      result.addEventListener('animationend', done, { once: true });
    }
    if (compareEnabled && !introduced && pos === 0) {
      introduced = true;
      // after the brush-wipe unveiling (700 ms) has finished
      sweepTo(50, unveil ? 900 : 300);
    }
  }

  function hideResult() {
    stopSweep();
    result.hidden = true;
    compare.hidden = true;
    tag.hidden = true;
    tagPainting.hidden = true;
    clearResultLabel();
  }

  function setCompareEnabled(on) {
    compare.hidden = !on || result.hidden;
    if (compare.hidden) { stopSweep(); stage.style.setProperty('--pos', '0%'); tag.hidden = true; tagPainting.hidden = true; }
    else setCompare(pos, { announce: true });
  }

  // ── Overlays ─────────────────────────────────────────────────────────
  // While the veil or the error card covers the frame, the compare handle under it
  // must not be focusable or operable.
  function syncOverlay() {
    compare.inert = !error.hidden || (!veil.hidden && !veil.classList.contains('is-leaving'));
  }

  // ── Veil (generating) ────────────────────────────────────────────────
  // It fades out as the brush wipe starts instead of vanishing in one frame. The
  // duration and easing come from the motion tokens; reduced motion hides it at once.
  let veilFade = null;
  function token(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }
  function endVeilFade() {
    if (!veilFade) return;
    const a = veilFade;
    veilFade = null;
    a.onfinish = null;
    a.cancel();
    veil.classList.remove('is-leaving');
    veil.inert = false;
  }
  const veilApi = {
    show(text = 'Painting') {
      endVeilFade();
      veil.hidden = false;
      veilText.textContent = text;
      veilSub.textContent = '';
      this.progress(null);
      syncOverlay();
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
    hide() {
      if (veil.hidden || veilFade) return;
      const ms = parseFloat(token('--t-base', '200ms')) || 0;
      if (prefersReducedMotion() || !ms || typeof veil.animate !== 'function') {
        veil.hidden = true;
        syncOverlay();
        return;
      }
      veil.classList.add('is-leaving');
      veil.inert = true;   // Cancel can't be pressed while it fades
      syncOverlay();
      const a = veil.animate([{ opacity: 1 }, { opacity: 0 }],
        { duration: ms, easing: token('--ease', 'ease-out'), fill: 'forwards' });
      veilFade = a;
      const done = () => {
        if (veilFade !== a) return;   // a new show() took over
        endVeilFade();
        veil.hidden = true;
        syncOverlay();
      };
      a.onfinish = done;
      setTimeout(done, ms + 80);   // finish events can stall in a background tab
    },
    get visible() { return !veil.hidden && !veilFade; },
  };

  // ── Error ────────────────────────────────────────────────────────────
  function showError(message, { smaller = false, retryLabel = 'Try again' } = {}) {
    errorSmaller.hidden = !smaller;
    $('#error-retry-text').textContent = retryLabel;
    // The role="alert" element is the message itself. Empty it, show the card, then
    // set the text: inserting text into a shown alert is what screen readers announce
    // (the same message twice in a row is announced again, too).
    errorText.textContent = '';
    error.hidden = false;
    errorText.textContent = message;
    syncOverlay();
    $('#error-retry').focus({ preventScroll: true });
  }
  const hideError = () => { error.hidden = true; errorText.textContent = ''; syncOverlay(); };
  // Esc dismisses the card, like its Dismiss button.
  error.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    e.preventDefault();
    $('#error-dismiss').click();
  });

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
    setAspect, showPhoto, showEmpty, showResult, hideResult, setResultLabel, setCompare, setCompareEnabled,
    veil: veilApi, showError, hideError,
    get comparePos() { return pos; },
  };
}
