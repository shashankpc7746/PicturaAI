// Style column: Gallery / Your painting / Describe it, plus "Mix with a second style".
import { $, $$, h, icon, rovingRadios, checkRadio, paintRange } from '../core/dom.js';
import { store } from '../core/store.js';
import { artistLine, matchStyleLocally } from '../services/styles-data.js';
import { describeStyle, base64ToBlob } from '../services/api.js';
import { checkFile, loadImageFile, toCanvas, blobToImage } from '../services/image-io.js';

const grid = () => $('#style-grid');

export function presetStyle(s) {
  return { source: 'preset', key: s.key, title: s.title, artist: s.artist, year: s.year, image: s.image };
}

// ── Gallery cards ─────────────────────────────────────────────────────────
export function renderSkeleton(count = 8) {
  const g = grid();
  g.replaceChildren();
  g.setAttribute('aria-busy', 'true');
  const aspects = [1.26, 0.81, 1.5, 1, 0.78, 1.33, 1.1, 0.9];
  for (let i = 0; i < count; i++) {
    const card = h('div', { class: 'style-card is-skeleton', 'aria-hidden': 'true' },
      h('div', { class: 'style-card__img' }), h('span', { class: 'style-card__title' }), h('span', { class: 'style-card__meta' }));
    card.style.setProperty('--a', String(aspects[i % aspects.length]));
    g.append(card);
  }
}

export function renderStatus(kind, onRetry) {
  const box = $('#styles-status');
  box.replaceChildren();
  box.classList.toggle('is-error', kind === 'error');
  if (kind === 'waking') {
    box.append(h('p', { text: 'Waking up the studio. After a quiet spell this takes up to a minute.' }));
  } else if (kind === 'error') {
    box.append(h('p', { text: "We couldn't load the paintings. Check your connection and try again." }),
      h('button', { class: 'btn btn--secondary', type: 'button', text: 'Try again', onclick: onRetry }));
  }
  box.hidden = kind !== 'waking' && kind !== 'error';
}

function card(s) {
  const el = h('div', {
    class: 'style-card',
    role: 'radio',
    'aria-checked': 'false',
    tabindex: '-1',
    'data-key': s.key,
    'aria-label': [s.title, artistLine(s)].filter(Boolean).join(', '),
  },
  h('img', {
    class: 'style-card__img',
    src: s.card,
    srcset: s.thumb && s.thumb !== s.card ? `${s.thumb} 160w, ${s.card} 480w` : null,
    sizes: '(max-width: 767px) 42vw, (max-width: 1023px) 22vw, 220px',
    alt: '',
    width: 480,
    height: Math.round(480 / s.aspect),
    loading: 'lazy',
    decoding: 'async',
  }),
  h('span', { class: 'style-card__title', text: s.title }),
  h('span', { class: 'style-card__meta', text: artistLine(s) }),
  h('span', { class: 'style-card__check' }, icon('check')));
  el.style.setProperty('--a', String(s.aspect));
  return el;
}

export function renderStyles(styles) {
  const g = grid();
  g.replaceChildren(...styles.map(card));
  g.setAttribute('aria-busy', 'false');
  renderMixRail(styles);
  syncSelection(store.get());
}

function syncSelection(state) {
  const key = state.style && state.style.source === 'preset' ? state.style.key : null;
  const items = $$('.style-card[role="radio"]', grid());
  checkRadio(items, items.find((el) => el.dataset.key === key) || null);
  const items2 = $$('.mix-thumb', $('#mix-rail'));
  checkRadio(items2, items2.find((el) => state.style2 && el.dataset.key === state.style2.key) || null);
  renderSelected(state);
}

function renderSelected(state) {
  const el = $('#selected-style');
  const s = state.style;
  if (!s) { el.textContent = 'No painting chosen yet'; return; }
  const parts = [document.createTextNode('Selected: '), h('strong', { text: s.title })];
  const line = s.source === 'preset' ? artistLine(s) : s.source === 'custom' ? 'your upload'
    : s.source === 'generated' ? 'made from your words' : 'matched when you paint';
  if (line) parts.push(document.createTextNode(`, ${line}`));
  if (state.mixOpen && state.style2) parts.push(document.createTextNode(`, mixed with ${state.style2.title}`));
  el.replaceChildren(...parts);
}

// ── Tabs ──────────────────────────────────────────────────────────────────
function initTabs() {
  const tabs = $$('.tabs__tab');
  const select = (tab, focus = false) => {
    for (const t of tabs) {
      const on = t === tab;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      $(`#${t.getAttribute('aria-controls')}`).hidden = !on;
    }
    if (focus) tab.focus();
  };
  tabs.forEach((t) => t.addEventListener('click', () => select(t)));
  $('.tabs').addEventListener('keydown', (e) => {
    const i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (next == null) return;
    e.preventDefault();
    select(tabs[(next + tabs.length) % tabs.length], true);
  });
  return { show: (id) => select($(`#${id}`)) };
}

// ── Your painting ─────────────────────────────────────────────────────────
function initCustom() {
  const input = $('#style-input');
  const tile = $('#style-drop');
  const preview = $('#custom-preview');
  const hint = $('#custom-hint');
  const title = $('#custom-title');
  const remove = $('#custom-remove');
  const hintText = hint.textContent;
  let url = null;

  async function use(file) {
    const reason = checkFile(file);
    hint.classList.toggle('is-error', !!reason);
    if (reason) { hint.textContent = reason; return; }
    try {
      const { img, url: u } = await loadImageFile(file);
      if (url) URL.revokeObjectURL(url);
      url = u;
      preview.src = u;
      preview.alt = `Your painting: ${file.name}`;
      preview.hidden = false;
      tile.classList.add('has-image');
      title.textContent = file.name;
      hint.textContent = 'This painting is selected. Paint it to use it.';
      hint.classList.remove('is-error');
      $('#custom-upload-text').textContent = 'Use a different painting';
      remove.hidden = false;
      store.set({ style: { source: 'custom', title: 'Your painting', name: file.name, canvas: toCanvas(img, 512) } });
    } catch (e) {
      hint.textContent = e.message;
      hint.classList.add('is-error');
    }
  }

  $('#custom-upload').addEventListener('click', () => input.click());
  input.addEventListener('change', () => { if (input.files[0]) use(input.files[0]); input.value = ''; });
  remove.addEventListener('click', () => {
    if (url) URL.revokeObjectURL(url);
    url = null;
    preview.hidden = true;
    preview.removeAttribute('src');
    tile.classList.remove('has-image');
    title.textContent = 'Use a painting you own';
    hint.textContent = hintText;
    $('#custom-upload-text').textContent = 'Upload a painting';
    remove.hidden = true;
    if (store.get().style?.source === 'custom') store.set({ style: null });
    $('#custom-upload').focus();
  });
  tile.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); tile.classList.add('is-dragover'); });
  tile.addEventListener('dragleave', () => tile.classList.remove('is-dragover'));
  tile.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    tile.classList.remove('is-dragover');
    if (e.dataTransfer.files[0]) use(e.dataTransfer.files[0]);
  });
}

// ── Describe it ───────────────────────────────────────────────────────────
function initDescribe() {
  const form = $('#describe-form');
  const input = $('#describe-input');
  const count = $('#describe-count');
  const submit = $('#describe-submit');
  const box = $('#describe-result');
  const img = $('#describe-img');
  const titleEl = $('#describe-title');
  const textEl = $('#describe-text');
  // Without Workers AI the server only matches words to a preset: say so.
  const generates = (window.PICTURA_CONFIG || {}).describeGenerates !== false;
  const idleLabel = generates ? 'Create style' : 'Find a style';
  submit.textContent = idleLabel;
  if (!generates) $('label[for="describe-input"]').textContent = "Describe a style in a few words. We'll find the closest painting.";

  input.addEventListener('input', () => { count.textContent = `${input.value.length} of 200 characters`; });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });

  function show({ title, text, src = null, error = false }) {
    box.hidden = false;
    box.classList.toggle('is-error', error);
    titleEl.textContent = title;
    textEl.textContent = text;
    img.hidden = !src;
    if (src) img.src = src; else img.removeAttribute('src');
  }

  function useMatch(s, message) {
    store.set({ style: presetStyle(s) });
    show({ title: `Closest classic style: ${s.title}`, text: message, src: s.thumb });
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const prompt = input.value.replace(/\s+/g, ' ').trim();
    if (!prompt) {
      show({ title: 'Add a few words first', text: 'For example: soft watercolour of a misty harbour.', error: true });
      input.focus();
      return;
    }
    submit.disabled = true;
    submit.textContent = generates ? 'Creating style' : 'Finding a style';
    submit.classList.add('is-busy');
    box.setAttribute('aria-busy', 'true');
    const styles = store.get().styles;
    try {
      const res = await describeStyle(prompt);
      if (res.source === 'generated' && res.image) {
        const blob = base64ToBlob(res.image, res.media_type || 'image/jpeg');
        const picture = await blobToImage(blob);
        const url = URL.createObjectURL(blob);
        store.set({ style: { source: 'generated', title: `"${prompt}"`, prompt, canvas: toCanvas(picture, 512) } });
        show({ title: 'Style created from your words', text: 'It is selected. Paint it to see it on your photo.', src: url });
      } else if (res.source === 'matched' && res.key) {
        const s = styles.find((x) => x.key === res.key);
        if (s) useMatch(s, res.message || `We matched the closest classic style: ${s.title}.`);
        else show({ title: res.title || 'Matched a classic style', text: res.message || '' });
      } else {
        throw Object.assign(new Error('unexpected'), { status: 500 });
      }
    } catch (err) {
      if (err && err.status === 404 && err.message !== 'Not Found') {
        // The server understood but nothing matched (or describing is turned off).
        show({ title: 'No style matched those words', text: err.message, error: true });
      } else {
        const s = matchStyleLocally(styles, prompt);
        if (s) {
          useMatch(s, "The style generator isn't available right now, so we matched the closest classic style.");
        } else {
          store.set({ style: { source: 'prompt', title: `"${prompt}"`, prompt } });
          show({ title: 'Your words are selected', text: "We'll match them to the closest classic style when you paint." });
        }
      }
    } finally {
      submit.disabled = false;
      submit.textContent = idleLabel;
      submit.classList.remove('is-busy');
      box.removeAttribute('aria-busy');
    }
  });
}

// ── Mix with a second style ──────────────────────────────────────────────
function renderMixRail(styles) {
  const rail = $('#mix-rail');
  rail.replaceChildren(...styles.map((s) => h('button', {
    class: 'mix-thumb',
    type: 'button',
    role: 'radio',
    'aria-checked': 'false',
    tabindex: '-1',
    'data-key': s.key,
    'aria-label': `${s.title}, ${artistLine(s)}`,
    title: s.title,
  }, h('img', { src: s.thumb, alt: '', width: 56, height: 56, loading: 'lazy', decoding: 'async' }))));
  const first = rail.querySelector('.mix-thumb');
  if (first && !rail.querySelector('[tabindex="0"]')) first.tabIndex = 0;
}

function initMix() {
  const toggle = $('#mix-toggle');
  const body = $('#mix-body');
  const ratio = $('#mix-ratio');
  const value = $('#mix-value');
  toggle.addEventListener('click', () => {
    const open = toggle.getAttribute('aria-expanded') !== 'true';
    toggle.setAttribute('aria-expanded', String(open));
    body.hidden = !open;
    store.set({ mixOpen: open });
  });
  rovingRadios($('#mix-rail'), '.mix-thumb', (el) => {
    const s = store.get().styles.find((x) => x.key === el.dataset.key);
    if (s) store.set({ style2: presetStyle(s) });
  });
  ratio.addEventListener('input', () => {
    paintRange(ratio);
    value.textContent = `${ratio.value}%`;
    store.set({ mix: Number(ratio.value) / 100 });
  });
  paintRange(ratio);
}

// ── Public ────────────────────────────────────────────────────────────────
export function initStylePicker() {
  const tabs = initTabs();
  rovingRadios(grid(), '.style-card[role="radio"]', (el) => {
    const s = store.get().styles.find((x) => x.key === el.dataset.key);
    if (s) store.set({ style: presetStyle(s) });
  });
  initCustom();
  initDescribe();
  initMix();
  store.subscribe((state, prev) => {
    if (state.style !== prev.style || state.style2 !== prev.style2 || state.mixOpen !== prev.mixOpen) syncSelection(state);
  });
  return {
    selectPreset(key) {
      const s = store.get().styles.find((x) => x.key === key);
      if (!s) return;
      tabs.show('tab-gallery');
      store.set({ style: presetStyle(s) });
      const el = grid().querySelector(`[data-key="${CSS.escape(key)}"]`);
      if (el) el.scrollIntoView({ block: 'nearest', inline: 'center' });
    },
  };
}
