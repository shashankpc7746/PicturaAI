// Small DOM helpers. User and server text always goes in through textContent.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const SVG_NS = 'http://www.w3.org/2000/svg';
const SPRITE = 'assets/icons/icons.svg';

/** Create an element: h('button', { class: 'btn', text: 'Save', onclick: fn }, child...) */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
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
    next.focus();
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
