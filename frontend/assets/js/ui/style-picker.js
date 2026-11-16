// Style column: Gallery / Your painting / Describe it, plus "Mix with a second style".
import { $, $$, h, icon, announce, rovingRadios, checkRadio, paintRange, revealInScrollers, prefersReducedMotion } from '../core/dom.js';
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
  const name = [s.title, artistLine(s)].filter(Boolean).join(', ');
  const el = h('div', {
    class: 'style-card',
    role: 'radio',
    'aria-checked': 'false',
    tabindex: '-1',
    'data-key': s.key,
    'aria-label': name,
    // Short screens show one line of title and artist: the full text on hover.
    title: name,
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
  scheduleHint();
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

  // The hint is not a live region (it would also read the reset text on Remove),
  // so results are announced through #sr-status.
  async function use(file) {
    const reason = checkFile(file);
    hint.classList.toggle('is-error', !!reason);
    if (reason) { hint.textContent = reason; announce(reason); return; }
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
      store.set({ style: { source: 'custom', title: 'Your painting', name: file.name, canvas: toCanvas(img, 512, { background: '#fff' }) } });
      announce('Painting added. Paint it to use it.');
    } catch (e) {
      hint.textContent = e.message;
      hint.classList.add('is-error');
      announce(e.message);
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
  // Only file drags light the tile, and moving over the tile's own children
  // (text, buttons) must not switch the highlight off and on.
  const hasFiles = (e) => Array.from(e.dataTransfer?.types || []).includes('Files');
  tile.addEventListener('dragenter', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    tile.classList.add('is-dragover');
  });
  tile.addEventListener('dragover', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = 'copy';
    tile.classList.add('is-dragover');
  });
  tile.addEventListener('dragleave', (e) => {
    if (e.relatedTarget && tile.contains(e.relatedTarget)) return;
    tile.classList.remove('is-dragover');
  });
  tile.addEventListener('drop', (e) => {
    if (!hasFiles(e)) return;
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

  let swatchUrl = null;   // blob: URL of the last generated swatch, revoked when replaced
  function show({ title, text, src = null, error = false }) {
    box.hidden = false;
    box.classList.toggle('is-error', error);
    titleEl.textContent = title;
    textEl.textContent = text;
    img.hidden = !src;
    if (src) img.src = src; else img.removeAttribute('src');
    if (swatchUrl && swatchUrl !== src) { URL.revokeObjectURL(swatchUrl); swatchUrl = null; }
    // On a short screen the answer can land under the stuck Paint it button.
    if (!box.closest('[hidden]')) requestAnimationFrame(() => revealInPanel(box));
  }

  // The title already names the style, so the second line says who painted it
  // instead of repeating it.
  const selectedLine = (s) => {
    const who = artistLine(s);
    return `${who ? `${who}. ` : ''}It is selected. Paint it to see it on your photo.`;
  };
  // A late answer must not replace a painting the person picked meanwhile.
  const pickedMeanwhile = 'You picked another painting meanwhile, so that one stays selected.';

  function useMatch(s, message, select = true) {
    if (select) store.set({ style: presetStyle(s) });
    show({ title: `Closest classic style: ${s.title}`, text: select ? message : pickedMeanwhile, src: s.thumb });
  }

  let busy = false;
  let reqSeq = 0;
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (busy) return;
    const prompt = input.value.replace(/\s+/g, ' ').trim();
    if (!prompt) {
      show({ title: 'Add a few words first', text: 'For example: soft watercolour of a misty harbour.', error: true });
      input.focus();
      return;
    }
    // aria-disabled, not disabled: a disabled button drops keyboard focus to <body>.
    busy = true;
    const req = ++reqSeq;
    const before = store.get().style;
    const stillCurrent = () => req === reqSeq && store.get().style === before;
    submit.setAttribute('aria-disabled', 'true');
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
        const select = stillCurrent();
        if (select) store.set({ style: { source: 'generated', title: `"${prompt}"`, prompt, canvas: toCanvas(picture, 512, { background: '#fff' }) } });
        show({
          title: 'Style created from your words',
          text: select ? 'It is selected. Paint it to see it on your photo.' : `${pickedMeanwhile} Describe it again to use this style.`,
          src: url,
        });
        swatchUrl = url;
      } else if (res.source === 'matched' && res.key) {
        const s = styles.find((x) => x.key === res.key);
        // With a reason other than "not configured", the server's message starts
        // with why it matched instead of generating; keep that.
        const explained = res.reason && res.reason !== 'not_configured' && res.message;
        if (s) useMatch(s, explained || selectedLine(s), stillCurrent());
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
          useMatch(s, "The style generator isn't available right now, so we matched the closest classic style.", stillCurrent());
        } else if (stillCurrent()) {
          store.set({ style: { source: 'prompt', title: `"${prompt}"`, prompt } });
          show({ title: 'Your words are selected', text: "We'll try to match them to a classic style when you paint." });
        } else {
          show({ title: 'No style matched those words', text: 'Try a painter, a colour or a mood.', error: true });
        }
      }
    } finally {
      busy = false;
      submit.removeAttribute('aria-disabled');
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
    // The opened rail and slider would otherwise sit below the stuck Paint it button.
    if (open) requestAnimationFrame(() => revealInPanel(body));
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

// ── "More below" hint ────────────────────────────────────────────────────
// On wide screens the gallery tab is a fixed-height scroller and its last visible
// row fades out. A quiet line under it says how many paintings there are, while
// some are cut off; at the end of the list it hides (keeping its line, so the
// panel does not jump) and .is-scrolled-end drops the fade. On tablets and phones
// the gallery is not a vertical scroller, so the hint stays hidden. Plain text in
// reading order, not a live region: it is not announced as it changes.
let scheduleHint = () => {};
function initScrollHint() {
  const panel = $('#panel-gallery');
  const hint = $('#gallery-hint');
  const text = $('#gallery-hint-text');
  if (!panel || !hint || !text) return;
  let raf = 0;
  const update = () => {
    raf = 0;
    const count = $$('.style-card[role="radio"]', grid()).length;
    const overflow = panel.scrollHeight - panel.clientHeight;
    const scrolls = !panel.hidden && count > 0 && overflow > 4
      && getComputedStyle(panel).overflowY !== 'visible';
    const atEnd = !scrolls || panel.scrollTop >= overflow - 4;
    panel.classList.toggle('is-scrolled-end', atEnd);
    hint.hidden = !scrolls;
    hint.classList.toggle('is-done', atEnd);
    if (count) text.textContent = `${count} ${count === 1 ? 'painting' : 'paintings'}. Scroll for more.`;
  };
  scheduleHint = () => { if (!raf) raf = requestAnimationFrame(update); };
  panel.addEventListener('scroll', scheduleHint, { passive: true });
  if (typeof ResizeObserver === 'function') {
    const ro = new ResizeObserver(scheduleHint);
    ro.observe(panel);
    ro.observe(grid());
  } else {
    window.addEventListener('resize', scheduleHint);
  }
  scheduleHint();
}

// ── Panel scroll (desktop) ───────────────────────────────────────────────
// On a desktop the style panel is height-capped. When its content does not fit (a
// short laptop screen, "Mix" open, a tall "Describe it" answer) it scrolls as a
// whole, with Paint it / Download stuck to its bottom edge. The stuck block's
// height (--cta-h) is the panel's bottom scroll-padding, so keyboard focus and
// revealInPanel() land clear of it; and while content is hidden below it the panel
// has .has-more-below (a deeper fade) and shows the "More below" pill.
const panelEl = () => $('#style-panel');
const panelScrolls = (p) => getComputedStyle(p).overflowY !== 'visible';

/** Scroll the desktop panel just enough to show `el` above Paint it (no-op elsewhere). */
function revealInPanel(el) {
  const p = panelEl();
  if (p && p.contains(el) && panelScrolls(p)) revealInScrollers(el);
}

function initPanelScroll() {
  const panel = panelEl();
  const cta = $('.style-panel__cta');
  const more = $('#panel-more');
  if (!panel || !cta || !more) return;
  let raf = 0;
  const update = () => {
    raf = 0;
    const scrolls = panelScrolls(panel);
    if (scrolls) panel.style.setProperty('--cta-h', `${Math.ceil(cta.offsetHeight)}px`);
    const below = scrolls && panel.scrollHeight - panel.clientHeight - panel.scrollTop > 4;
    panel.classList.toggle('has-more-below', below);
    more.hidden = !below;
  };
  const schedule = () => { if (!raf) raf = requestAnimationFrame(update); };
  panel.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule);
  if (typeof ResizeObserver === 'function') {
    // The panel stops growing at its cap, so watch what is inside it too: a tab
    // switch, Mix opening, an answer or a wrapped line all change what is below.
    const ro = new ResizeObserver(schedule);
    ro.observe(panel);
    for (const child of panel.children) ro.observe(child);
  }
  more.addEventListener('click', () => {
    panel.scrollBy({
      top: Math.max(80, panel.clientHeight - cta.offsetHeight - 64),
      behavior: prefersReducedMotion() ? 'auto' : 'smooth',
    });
  });
  schedule();
}

// ── Public ────────────────────────────────────────────────────────────────
export function initStylePicker() {
  const tabs = initTabs();
  initScrollHint();
  initPanelScroll();
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
