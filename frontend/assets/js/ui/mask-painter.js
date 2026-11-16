// "Paint a region": brush a mask over the photo with mouse, touch or pen.
// The mask lives in an offscreen canvas (white = styled); the visible overlay
// shows it as a translucent gilt hatch.

const MASK_SIDE = 1024;

// The hatch is designed in CSS pixels (a 14 px tile with 2.5 px lines). The overlay
// canvas is ~1024 px wide but shown at ~360 px on a phone, so the tile is scaled by
// canvas px per CSS px; otherwise the lines shrink below a pixel and the brushed
// area almost disappears on a busy photo.
function makeHatch(ctx, k = 1) {
  const T = Math.max(8, Math.round(14 * k));
  const t = document.createElement('canvas');
  t.width = t.height = T;
  const c = t.getContext('2d');
  c.scale(T / 14, T / 14);
  c.fillStyle = 'rgba(201, 164, 76, 0.34)';
  c.fillRect(0, 0, 14, 14);
  c.strokeStyle = 'rgba(221, 187, 102, 0.95)';
  c.lineWidth = 2.5;
  c.beginPath();
  c.moveTo(-2, 16); c.lineTo(16, -2);
  c.moveTo(-2, 2); c.lineTo(2, -2);
  c.moveTo(12, 16); c.lineTo(16, 12);
  c.stroke();
  return ctx.createPattern(t, 'repeat');
}

export function createMaskPainter(overlay, { onChange, onCommit } = {}) {
  const data = document.createElement('canvas');
  const dctx = data.getContext('2d', { willReadFrequently: true });
  const octx = overlay.getContext('2d');
  let hatch = null;
  let hatchK = 0;
  let enabled = false;
  let brush = 'draw';
  let size = 36;            // CSS pixels
  let version = 0;
  let cached = null;
  let hasContent = false;
  let drawing = null;       // { id, x, y }
  let raf = 0;
  let revealTimer = 0;

  function reset(width, height) {
    const k = MASK_SIDE / Math.max(width, height);
    data.width = overlay.width = Math.max(1, Math.round(width * k));
    data.height = overlay.height = Math.max(1, Math.round(height * k));
    hatch = null;
    clear(false);
  }

  /** Rebuild the hatch when the overlay's on-screen scale changes (resize, rotation). */
  function ensureHatch() {
    const cssW = overlay.getBoundingClientRect().width;
    if (!cssW && hatch) return;   // hidden: keep what we have
    const k = cssW ? Math.round((overlay.width / cssW) * 4) / 4 || 1 : 1;
    if (!hatch || k !== hatchK) {
      hatch = makeHatch(octx, k);
      hatchK = k;
    }
  }

  function refresh() {
    raf = 0;
    ensureHatch();
    octx.globalCompositeOperation = 'source-over';
    octx.clearRect(0, 0, overlay.width, overlay.height);
    octx.drawImage(data, 0, 0);
    octx.globalCompositeOperation = 'source-in';
    octx.fillStyle = hatch;
    octx.fillRect(0, 0, overlay.width, overlay.height);
    octx.globalCompositeOperation = 'source-over';
  }
  const schedule = () => { if (!raf) raf = requestAnimationFrame(refresh); };

  function changed(commit) {
    version++;
    cached = null;
    schedule();
    if (commit) {
      hasContent = scanContent();
      onCommit && onCommit();
    } else {
      onChange && onChange();
    }
  }

  function scanContent() {
    const px = dctx.getImageData(0, 0, data.width, data.height).data;
    for (let i = 3; i < px.length; i += 16) if (px[i] > 8) return true;
    return false;
  }

  function point(e) {
    const r = overlay.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * data.width,
      y: ((e.clientY - r.top) / r.height) * data.height,
      scale: data.width / r.width,
    };
  }

  function stroke(from, to) {
    dctx.save();
    dctx.globalCompositeOperation = brush === 'erase' ? 'destination-out' : 'source-over';
    dctx.strokeStyle = '#fff';
    dctx.fillStyle = '#fff';
    dctx.lineCap = 'round';
    dctx.lineJoin = 'round';
    dctx.lineWidth = size * to.scale;
    dctx.beginPath();
    if (from.x === to.x && from.y === to.y) {
      dctx.arc(to.x, to.y, dctx.lineWidth / 2, 0, Math.PI * 2);
      dctx.fill();
    } else {
      dctx.moveTo(from.x, from.y);
      dctx.lineTo(to.x, to.y);
      dctx.stroke();
    }
    dctx.restore();
  }

  overlay.addEventListener('pointerdown', (e) => {
    if (!enabled || (e.pointerType === 'mouse' && e.button !== 0)) return;
    e.preventDefault();
    overlay.setPointerCapture(e.pointerId);
    const p = point(e);
    drawing = { id: e.pointerId, ...p };
    overlay.classList.add('is-drawing');
    stroke(p, p);
    changed(false);
  });
  overlay.addEventListener('pointermove', (e) => {
    if (!drawing || e.pointerId !== drawing.id) return;
    const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    let last = drawing;
    for (const ev of events.length ? events : [e]) {
      const p = point(ev);
      stroke(last, p);
      last = p;
    }
    drawing = { id: drawing.id, ...last };
    changed(false);
  });
  const end = (e) => {
    if (!drawing || e.pointerId !== drawing.id) return;
    drawing = null;
    overlay.classList.remove('is-drawing');
    changed(true);
  };
  overlay.addEventListener('pointerup', end);
  overlay.addEventListener('pointercancel', end);

  function clear(notify = true) {
    dctx.clearRect(0, 0, data.width, data.height);
    hasContent = false;
    if (notify) changed(true);
    else { version++; cached = null; schedule(); }
  }

  function invert() {
    const img = dctx.getImageData(0, 0, data.width, data.height);
    const px = img.data;
    for (let i = 0; i < px.length; i += 4) {
      px[i] = px[i + 1] = px[i + 2] = 255;
      px[i + 3] = 255 - px[i + 3];
    }
    dctx.putImageData(img, 0, 0);
    changed(true);
  }

  /** Replace the mask with an ImageData (white = styled). */
  function setFromImageData(mask, { invert: inv = false } = {}) {
    const tmp = document.createElement('canvas');
    tmp.width = mask.width; tmp.height = mask.height;
    const tctx = tmp.getContext('2d');
    const out = tctx.createImageData(mask.width, mask.height);
    const s = mask.data; const o = out.data;
    for (let i = 0; i < s.length; i += 4) {
      const l = (s[i] * 0.299 + s[i + 1] * 0.587 + s[i + 2] * 0.114) * (s[i + 3] / 255);
      o[i] = o[i + 1] = o[i + 2] = 255;
      o[i + 3] = inv ? 255 - l : l;
    }
    tctx.putImageData(out, 0, 0);
    dctx.clearRect(0, 0, data.width, data.height);
    dctx.imageSmoothingQuality = 'high';
    dctx.drawImage(tmp, 0, 0, data.width, data.height);
    changed(true);
  }

  /** The mask as ImageData (white = styled); a new object only when it changed. */
  function getMask() {
    if (!cached) {
      cached = dctx.getImageData(0, 0, data.width, data.height);
      cached.version = version;
    }
    return cached;
  }

  return {
    reset,
    clear: () => clear(true),
    invert,
    setFromImageData,
    getMask,
    toCanvas: () => data,
    get hasContent() { return hasContent; },
    setEnabled(on) {
      enabled = on;
      overlay.hidden = !on;
      if (on) schedule();
    },
    setBrush(kind) { brush = kind; },
    setSize(px) { size = px; },
    /** Show the full hatch for a moment (.is-editing) after a button changed the area. */
    reveal(ms = 1600) {
      clearTimeout(revealTimer);
      overlay.classList.add('is-editing');
      revealTimer = setTimeout(() => overlay.classList.remove('is-editing'), ms);
    },
  };
}
