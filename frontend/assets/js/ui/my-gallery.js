// "My gallery" section: list, open in the frame, download, delete with undo.
// The list is updated by id, not rebuilt: rebuilding replays every item's fade-in
// and destroys the button that has keyboard focus.
import { $, h, icon, formatDate, announce } from '../core/dom.js';
import { toast } from './toasts.js';
import { listPaintings, putPainting, deletePainting, clearPaintings, storageNote } from '../services/gallery-db.js';
import { saveBlob, fileName } from '../services/export.js';

/**
 * @param {{ onOpen: (rec) => void, onChange?: (ids: Set<string>, info: { available: boolean }) => void }} opts
 */
export function initMyGallery({ onOpen, onChange = () => {} }) {
  const list = $('#my-gallery');
  const empty = $('#gallery-empty');
  const note = $('#gallery-note');
  const clearBtn = $('#gallery-clear');
  const baseNote = note.textContent;
  const emptyText = empty.textContent;
  const nodes = new Map();   // id -> { li, url }
  let firstLoad = true;

  function drop(id) {
    const n = nodes.get(id);
    if (!n) return;
    n.li.remove();
    URL.revokeObjectURL(n.url);
    nodes.delete(id);
  }

  async function updateMeta(count) {
    empty.textContent = emptyText;
    empty.hidden = count > 0;
    clearBtn.hidden = count === 0;
    const extra = count ? await storageNote() : '';
    note.textContent = extra ? `${baseNote} ${extra}` : baseNote;
  }

  async function refresh() {
    let items = [];
    try {
      items = await listPaintings();
    } catch (e) {
      for (const id of [...nodes.keys()]) drop(id);
      empty.textContent = e.message || 'Saved paintings are not available in this browser.';
      empty.hidden = false;
      clearBtn.hidden = true;
      onChange(new Set(), { available: e.code !== 'GALLERY_UNAVAILABLE' });
      return;
    }
    const ids = new Set(items.map((r) => r.id));
    for (const id of [...nodes.keys()]) if (!ids.has(id)) drop(id);
    // Insert new items in their slot. Kept items are already in order, so they are
    // never moved (a moved node would replay its animation).
    items.forEach((rec, i) => {
      let n = nodes.get(rec.id);
      if (!n) {
        n = item(rec);
        if (!firstLoad) n.li.classList.add('is-new');
        nodes.set(rec.id, n);
      }
      if (list.children[i] !== n.li) list.insertBefore(n.li, list.children[i] || null);
    });
    firstLoad = false;
    onChange(ids, { available: true });
    await updateMeta(items.length);
  }

  function item(rec) {
    const url = URL.createObjectURL(rec.thumb || rec.image);
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
    li.addEventListener('animationend', () => li.classList.remove('is-new'));
    return { li, url };
  }

  async function remove(rec, li) {
    const next = li.nextElementSibling || li.previousElementSibling;
    try {
      await deletePainting(rec.id);
    } catch (e) {
      toast(e.message, { type: 'error' });
      return;
    }
    // Focus moves to the neighbour now; that node is kept, so focus stays put.
    const focusTarget = next && next.querySelector('.icon-btn');
    if (focusTarget) focusTarget.focus();
    else { const t = $('#gallery-title'); t.setAttribute('tabindex', '-1'); t.focus({ preventScroll: true }); }
    li.classList.add('is-leaving');
    let gone = false;
    const finish = () => {
      if (gone) return;
      gone = true;
      drop(rec.id);
      onChange(new Set(nodes.keys()), { available: true });
      updateMeta(nodes.size);
    };
    li.addEventListener('animationend', (e) => { if (e.target === li) finish(); });
    setTimeout(finish, 260);   // no animationend with reduced motion or in a background tab
    toast(`Deleted ${rec.title}.`, {
      action: { label: 'Undo', run: async () => { await putPainting(rec); finish(); await refresh(); announce('Painting restored.'); } },
    });
  }

  clearBtn.addEventListener('click', async () => {
    let backup = [];
    try { backup = await listPaintings(); await clearPaintings(); } catch (e) { toast(e.message, { type: 'error' }); return; }
    await refresh();
    const t = $('#gallery-title');
    t.setAttribute('tabindex', '-1');
    t.focus({ preventScroll: true });   // the Clear button is hidden now
    toast(`Cleared ${backup.length} ${backup.length === 1 ? 'painting' : 'paintings'}.`, {
      action: { label: 'Undo', run: async () => { for (const r of backup) await putPainting(r); await refresh(); } },
    });
  });

  refresh();
  return { refresh };
}
