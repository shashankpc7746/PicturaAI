// Toasts in a polite live region. Optional action (e.g. Undo).
import { h, icon } from '../core/dom.js';

const region = () => document.getElementById('toasts');

/**
 * @param {string} message
 * @param {{ type?: 'info'|'error', action?: { label: string, run: () => void }, duration?: number }} [opts]
 * @returns {() => void} dismiss
 */
export function toast(message, { type = 'info', action = null, duration } = {}) {
  const root = region();
  if (!root) return () => {};
  const ms = duration ?? (action ? 7000 : 4500);
  const el = h('div', { class: `toast toast--${type}` },
    icon(type === 'error' ? 'alert' : 'check', 'icon toast__icon'),
    h('span', { class: 'toast__text', text: message }));
  let timer = null;
  const dismiss = () => {
    clearTimeout(timer);
    if (!el.isConnected || el.classList.contains('is-leaving')) return;
    el.classList.add('is-leaving');
    setTimeout(() => el.remove(), 220);
  };
  if (action) {
    el.append(h('button', {
      class: 'toast__action',
      type: 'button',
      text: action.label,
      onclick: () => { dismiss(); action.run(); },
    }));
  }
  while (root.children.length >= 3) root.firstElementChild.remove();
  root.append(el);
  const arm = () => { timer = setTimeout(dismiss, ms); };
  el.addEventListener('pointerenter', () => clearTimeout(timer));
  el.addEventListener('pointerleave', arm);
  el.addEventListener('focusin', () => clearTimeout(timer));
  el.addEventListener('focusout', arm);
  arm();
  return dismiss;
}
