// "My gallery" section: list, open in the frame, download, delete with undo.
import { $, h, icon, formatDate, announce } from '../core/dom.js';
import { toast } from './toasts.js';
import { listPaintings, putPainting, deletePainting, clearPaintings, storageNote } from '../services/gallery-db.js';
import { saveBlob, fileName } from '../services/export.js';

export function initMyGallery({ onOpen }) {
  const list = $('#my-gallery');
  const empty = $('#gallery-empty');
  const note = $('#gallery-note');
  const clearBtn = $('#gallery-clear');
  const baseNote = note.textContent;
  let urls = [];

  async function refresh() {
    let items = [];
    try {
      items = await listPaintings();
    } catch (e) {
      empty.textContent = e.message || 'Saved paintings are not available in this browser.';
      empty.hidden = false;
      list.replaceChildren();
      return;
    }
    urls.forEach((u) => URL.revokeObjectURL(u));
    urls = [];
    list.replaceChildren(...items.map(item));
    empty.hidden = items.length > 0;
    clearBtn.hidden = items.length === 0;
    const extra = items.length ? await storageNote() : '';
    note.textContent = extra ? `${baseNote} ${extra}` : baseNote;
  }

  function item(rec) {
    const url = URL.createObjectURL(rec.thumb || rec.image);
    urls.push(url);
    const li = h('li', { class: 'saved', 'data-id': rec.id },
      // The picture is a large mouse/touch target; keyboard users get the Open button below.
      h('button', { class: 'saved__open', type: 'button', tabindex: '-1', 'aria-hidden': 'true', onclick: () => onOpen(rec) },
        h('img', { src: url, alt: '', loading: 'lazy', decoding: 'async' })),
      h('div', {},
        h('p', { class: 'saved__title', text: rec.title }),
        h('p', { class: 'saved__date', text: formatDate(rec.createdAt) })),
      h('div', { class: 'saved__actions' },
        h('button', { class: 'icon-btn', type: 'button', 'aria-label': `Open ${rec.title} in the frame`, title: 'Open in the frame', onclick: () => onOpen(rec) }, icon('open')),
        h('button', { class: 'icon-btn', type: 'button', 'aria-label': `Download ${rec.title}`, title: 'Download', onclick: () => saveBlob(rec.image, fileName(rec.title)) }, icon('download')),
        h('button', { class: 'icon-btn', type: 'button', 'aria-label': `Delete ${rec.title}`, title: 'Delete', onclick: () => remove(rec, li) }, icon('trash'))));
    return li;
  }

  async function remove(rec, li) {
    const next = li.nextElementSibling || li.previousElementSibling;
    try {
      await deletePainting(rec.id);
    } catch (e) {
      toast(e.message, { type: 'error' });
      return;
    }
    li.classList.add('is-leaving');
    setTimeout(refresh, 200);
    const focusTarget = next && next.querySelector('.icon-btn');
    if (focusTarget) focusTarget.focus();
    else { const t = $('#gallery-title'); t.setAttribute('tabindex', '-1'); t.focus({ preventScroll: true }); }
    toast(`Deleted ${rec.title}.`, {
      action: { label: 'Undo', run: async () => { await putPainting(rec); await refresh(); announce('Painting restored.'); } },
    });
  }

  clearBtn.addEventListener('click', async () => {
    let backup = [];
    try { backup = await listPaintings(); await clearPaintings(); } catch (e) { toast(e.message, { type: 'error' }); return; }
    await refresh();
    toast(`Cleared ${backup.length} ${backup.length === 1 ? 'painting' : 'paintings'}.`, {
      action: { label: 'Undo', run: async () => { for (const r of backup) await putPainting(r); await refresh(); } },
    });
  });

  refresh();
  return { refresh };
}
