// Small DOM helpers. User and server text always goes in through textContent.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const SVG_NS = 'http://www.w3.org/2000/svg';
const SPRITE = 'assets/icons/icons.svg';

// Attributes that start a fetch go last, after `loading`/`decoding`/`sizes`: an <img>
// whose src is set before loading="lazy" starts downloading at once whenever the
// file is already in the HTTP cache (every repeat visit).
const FETCH_ATTRS = ['srcset', 'src'];

/** Create an element: h('button', { class: 'btn', text: 'Save', onclick: fn }, child...) */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  const entries = Object.entries(attrs);
  entries.sort(([a], [b]) => FETCH_ATTRS.indexOf(a) - FETCH_ATTRS.indexOf(b));
  for (const [k, v] of entries) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c.nodeType ? c : String(c));
  }
  return el;
}

/** Inline SVG icon from the sprite (decorative). */
export function icon(name, cls = 'icon') {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', cls);
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS(SVG_NS, 'use');
  use.setAttribute('href', `${SPRITE}#${name}`);
  svg.append(use);
  return svg;
}

export const prefersReducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

export function formatBytes(n) {
  if (n < 1024 * 1024) return `${Math.max(1, Math.round(n / 1024))} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDate(ts) {
  try {
    return new Intl.DateTimeFormat(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }).format(ts);
  } catch {
    return new Date(ts).toLocaleString();
  }
}

/** Fill a range input's track up to its value (CSS reads --fill). */
export function paintRange(input) {
  const min = Number(input.min) || 0;
  const max = Number(input.max) || 100;
  const pct = ((Number(input.value) - min) / (max - min)) * 100;
  input.style.setProperty('--fill', `${pct}%`);
}

/**
 * Scroll `el` fully into view inside every scroller around it (and the page), each
 * one only as far as needed. Unlike scrollIntoView({ block: 'nearest' }), an item
 * taller (or wider) than what a scroller shows has its start aligned instead of
 * being left where it is, half cut off. Each scroller's scroll-padding is kept clear
 * (the gallery's "more below" fade, the style panel's stuck Paint it button, the
 * sticky header and the phone action bar).
 */
export function revealInScrollers(el, { behavior } = {}) {
  if (!el || !el.isConnected) return;
  behavior = behavior || (prefersReducedMotion() ? 'auto' : 'smooth');
  const root = document.scrollingElement || document.documentElement;
  // Where the element will be once the inner scrollers have moved (they all start
  // at once, so the outer ones are measured against the planned position).
  let shiftX = 0;
  let shiftY = 0;
  const delta = (start, end, lo, hi) => {
    if (end - start > hi - lo || start < lo) return start - lo;
    if (end > hi) return end - hi;
    return 0;
  };
  for (let p = el.parentElement; p; p = p.parentElement) {
    const isRoot = p === root;
    const cs = getComputedStyle(p);
    const canY = p.scrollHeight > p.clientHeight + 1 && (isRoot || /auto|scroll/.test(cs.overflowY));
    const canX = p.scrollWidth > p.clientWidth + 1 && (isRoot || /auto|scroll/.test(cs.overflowX));
    if (!canX && !canY) continue;
    const r = el.getBoundingClientRect();
    const pr = isRoot ? { left: 0, top: 0 } : p.getBoundingClientRect();
    const box = isRoot
      ? { left: 0, top: 0, right: innerWidth, bottom: innerHeight }
      : { left: pr.left + p.clientLeft, top: pr.top + p.clientTop,
          right: pr.left + p.clientLeft + p.clientWidth, bottom: pr.top + p.clientTop + p.clientHeight };
    const pad = (v) => parseFloat(v) || 0;
    let dx = 0;
    let dy = 0;
    if (canY) {
      dy = delta(r.top - shiftY, r.bottom - shiftY,
        box.top + pad(cs.scrollPaddingTop), box.bottom - pad(cs.scrollPaddingBottom));
    }
    if (canX) {
      dx = delta(r.left - shiftX, r.right - shiftX,
        box.left + pad(cs.scrollPaddingLeft), box.right - pad(cs.scrollPaddingRight));
    }
    // Only as far as the scroller can go, so the outer ones get the true remainder.
    dy = Math.max(-p.scrollTop, Math.min(dy, p.scrollHeight - p.clientHeight - p.scrollTop));
    dx = Math.max(-p.scrollLeft, Math.min(dx, p.scrollWidth - p.clientWidth - p.scrollLeft));
    if (Math.abs(dx) >= 1 || Math.abs(dy) >= 1) {
      p.scrollBy({ left: Math.round(dx), top: Math.round(dy), behavior });
      shiftX += Math.round(dx);
      shiftY += Math.round(dy);
    }
    if (cs.position === 'fixed') break;
  }
}

/** Roving tabindex + arrow keys for a group of role="radio" elements. */
export function rovingRadios(container, selector, onPick) {
  container.addEventListener('keydown', (e) => {
    const items = $$(selector, container).filter((el) => !el.hidden && !el.classList.contains('is-skeleton'));
    const i = items.indexOf(document.activeElement);
    if (i < 0) return;
    let next = null;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = items[(i + 1) % items.length];
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = items[(i - 1 + items.length) % items.length];
    else if (e.key === 'Home') next = items[0];
    else if (e.key === 'End') next = items[items.length - 1];
    else if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); onPick(items[i]); return; }
    if (!next) return;
    e.preventDefault();
    // focus() alone leaves a partly visible item half cut off in a scrolling rail,
    // and scrollIntoView 'nearest' does nothing for a card taller than the gallery.
    next.focus({ preventScroll: true });
    revealInScrollers(next);
    onPick(next);
  });
  container.addEventListener('click', (e) => {
    const item = e.target.closest(selector);
    if (item && container.contains(item)) onPick(item);
  });
}

/** Mark one radio checked and make it the group's single tab stop. */
export function checkRadio(items, chosen) {
  let any = false;
  for (const el of items) {
    const on = el === chosen;
    any = any || on;
    el.setAttribute('aria-checked', String(on));
    el.tabIndex = on ? 0 : -1;
  }
  if (!any && items[0]) items[0].tabIndex = 0;
}

export function announce(text) {
  const el = document.getElementById('sr-status');
  if (!el) return;
  el.textContent = '';
  requestAnimationFrame(() => { el.textContent = text; });
}
