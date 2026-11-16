// "The collection": every preset with its wall label and a "Use this style" button.
import { $, h } from '../core/dom.js';
import { artistLine } from '../services/styles-data.js';

export function renderCollection(styles, { onUse }) {
  const list = $('#collection');
  list.replaceChildren(...styles.map((s) => h('li', { class: 'piece' },
    h('div', { class: 'piece__hang' },
      h('img', {
        class: 'piece__img', src: s.card, alt: `${s.title} by ${s.artist || 'an unknown artist'}`,
        width: 480, height: Math.round(480 / s.aspect), loading: 'lazy', decoding: 'async',
      })),
    h('div', { class: 'piece__label' },
      h('h3', { class: 'piece__title', text: s.title }),
      h('p', { class: 'piece__artist', text: artistLine(s) }),
      s.description ? h('p', { class: 'piece__desc', text: s.description }) : null,
      h('div', { class: 'piece__actions' },
        h('button', { class: 'btn btn--on-label', type: 'button', text: 'Use this style', onclick: () => onUse(s.key) }),
        s.page ? h('a', { class: 'piece__link', href: s.page, text: 'About this painting', 'aria-label': `About ${s.title}` }) : null)))));
  $('#collection-status').hidden = true;
}

export function renderCollectionStatus(text) {
  const el = $('#collection-status');
  el.textContent = text;
  el.hidden = !text;
}
