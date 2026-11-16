// PicturaAI studio entry point (native ES modules, no build step).
import { $, prefersReducedMotion } from './core/dom.js';
import { store } from './core/store.js';
import { getStyles } from './services/api.js';
import { normalizeStyles } from './services/styles-data.js';
import { engineKindHint, preloadEngineCode } from './services/engine-adapter.js';
import { initHeader } from './ui/header.js';
import { initStylePicker, renderSkeleton, renderStyles, renderStatus } from './ui/style-picker.js';
import { renderCollection, renderCollectionStatus } from './ui/collection.js';
import { initMyGallery } from './ui/my-gallery.js';
import { initStudio } from './ui/studio.js';

const config = window.PICTURA_CONFIG || {};

initHeader();
const picker = initStylePicker();
const studio = initStudio();
const gallery = initMyGallery({
  onOpen: (rec) => studio.openSaved(rec),
  onChange: (ids, info) => studio.setGalleryIds(ids, info),
});
studio.setGallery(gallery);

if (config.describeEnabled === false) {
  $('#tab-describe').hidden = true;
}

// ── Styles: retry with backoff while a sleeping server wakes up ───────────
const RETRY_DELAYS = [2000, 4000, 6000, 8000, 10000, 15000, 15000, 15000];

async function loadStyles(attempt = 0) {
  if (attempt === 0) renderSkeleton();
  try {
    const styles = normalizeStyles(await getStyles());
    if (!styles.length) throw new Error('No styles');
    store.set({ styles, stylesStatus: 'ready' });
    renderStatus(null);
    renderStyles(styles);
    // Style pages link here as /app/?style=<key>#studio
    const wanted = new URLSearchParams(location.search).get('style');
    if (wanted && !store.get().style) picker.selectPreset(wanted);
    renderCollection(styles, {
      onUse: (key) => {
        picker.selectPreset(key);
        $('#studio').scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
        const card = document.querySelector(`.style-card[data-key="${CSS.escape(key)}"]`);
        if (card) card.focus({ preventScroll: true });
      },
    });
  } catch {
    const retrying = attempt < RETRY_DELAYS.length;
    store.set({ stylesStatus: retrying ? 'waking' : 'error' });
    renderStatus(retrying ? 'waking' : 'error', () => loadStyles(0));
    renderCollectionStatus(retrying ? 'Loading the collection.' : "We couldn't load the collection. Refresh the page to try again.");
    if (retrying) setTimeout(() => loadStyles(attempt + 1), RETRY_DELAYS[attempt]);
  }
}
loadStyles();

// ── Honest copy about where the painting happens ─────────────────────────
const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 600));
idle(async () => {
  preloadEngineCode();
  const kind = await engineKindHint();
  if (kind === 'browser') {
    $('#studio-lede').textContent = 'Pick a photo and a painting. It runs on your device, so your photo stays private.';
  }
});
