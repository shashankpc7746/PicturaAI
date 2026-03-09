/**
 * PicturaAI — Frontend Application
 * ==================================
 * Pictura (Latin: "a painting") — AI-powered Neural Style Transfer Studio
 * Handles: style loading, image upload, job submission,
 * WebSocket progress streaming, result rendering, gallery.
 */

const API = '';   // same origin; change to 'http://localhost:8000' for dev
let currentJobId = null;
let currentWS = null;
let selectedPreset = null;
let resultB64 = null;
let contentFile = null;
let styleFile = null;
let _doneReceived = false;
let _polling = false;
let styleMixEnabled = false;
let styleFile2 = null;
let selectedPreset2 = null;

// ── Mask painting state ──────────────────────────────────────────
let maskMode = false;
let maskBrushType = 'draw';   // 'draw' or 'erase'
let maskBrushSize = 30;
let maskPainting = false;
let maskHasContent = false;   // whether user painted anything

// ── Generation History ───────────────────────────────────────────
const MAX_HISTORY = 10;
let generationHistory = [];   // [{id, b64, style, timestamp}]
let activeHistoryIdx = -1;

// ── Navbar scroll effect ─────────────────────────────────────────
window.addEventListener('scroll', () => {
  document.getElementById('navbar').classList.toggle('scrolled', window.scrollY > 40);
});

// ── On load ──────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  loadStyles();
  setupDnD('contentUploadZone', 'contentInput', handleContentFile);
  setupDnD('customPanel', 'styleCustomInput', handleStyleFile);
  setupDnD('customPanel2', 'styleCustomInput2', handleStyleFile2);
  document.getElementById('contentInput').addEventListener('change', e => handleContentFile(e.target.files[0]));
  document.getElementById('styleCustomInput').addEventListener('change', e => handleStyleFile(e.target.files[0]));
  document.getElementById('styleCustomInput2').addEventListener('change', e => handleStyleFile2(e.target.files[0]));
  initBASlider();
});

// ── Slider helpers ───────────────────────────────────────────────
function updateSlider(el, labelId) {
  document.getElementById(labelId).textContent = parseFloat(el.value).toLocaleString();
}
function updateLRSlider(el) {
  document.getElementById('lrVal').textContent = parseFloat(el.value).toFixed(3);
}

// ── Style tab switch ─────────────────────────────────────────────
function switchStyleTab(tab) {
  const isPreset = tab === 'presets';
  document.getElementById('tabPresets').classList.toggle('active', isPreset);
  document.getElementById('tabCustom').classList.toggle('active', !isPreset);
  document.getElementById('presetPanel').style.display = isPreset ? 'grid' : 'none';
  document.getElementById('customPanel').style.display = isPreset ? 'none' : 'flex';
  if (!isPreset) { selectedPreset = null; clearPresetSelection(); }
}

// ── Load styles from API ─────────────────────────────────────────
async function loadStyles() {
  try {
    const res = await fetch(`${API}/api/styles`);
    const styles = await res.json();
    renderPresets(styles);
    renderPresets2(styles);
    renderGallery(styles);
  } catch (e) {
    document.getElementById('presetPanel').innerHTML =
      '<div class="preset-loading"><span>⚠ Could not load styles. Is the server running?</span></div>';
  }
}

function renderPresets(styles) {
  const grid = document.getElementById('presetPanel');
  grid.innerHTML = '';
  styles.forEach(s => {
    const card = document.createElement('div');
    card.className = 'preset-card';
    card.id = `preset-${s.key}`;
    card.innerHTML = `
      ${s.thumbnail
        ? `<img src="data:image/jpeg;base64,${s.thumbnail}" alt="${s.name}" loading="lazy" />`
        : `<div style="background:var(--surface-2);width:100%;height:100%;display:flex;align-items:center;justify-content:center;font-size:28px;">🎨</div>`}
      <div class="preset-caption">${s.name}</div>
      <div class="preset-check">✓</div>
    `;
    card.addEventListener('click', () => selectPreset(s.key, card));
    card.title = `${s.name} by ${s.artist} — ${s.description}`;
    grid.appendChild(card);
  });
}

function renderGallery(styles) {
  const grid = document.getElementById('galleryGrid');
  grid.innerHTML = '';
  styles.forEach(s => {
    const card = document.createElement('div');
    card.className = 'gallery-card';
    card.innerHTML = `
      ${s.thumbnail
        ? `<img src="data:image/jpeg;base64,${s.thumbnail}" alt="${s.name}" loading="lazy" />`
        : `<div style="background:var(--surface-2);width:100%;height:100%;"></div>`}
      <div class="gallery-info">
        <h4>${s.name}</h4>
        <p>${s.artist} · ${s.description}</p>
        <button class="gallery-apply" onclick="applyFromGallery('${s.key}', event)">Use This Style →</button>
      </div>
    `;
    grid.appendChild(card);
  });
}

function selectPreset(key, el) {
  clearPresetSelection();
  el.classList.add('active');
  selectedPreset = key;
  styleFile = null;
  toast(`Style selected: ${el.querySelector('.preset-caption').textContent}`, 'info');
}

function clearPresetSelection() {
  document.querySelectorAll('.preset-card.active').forEach(el => el.classList.remove('active'));
  selectedPreset = null;
}

function applyFromGallery(key, e) {
  e.stopPropagation();
  // Switch to presets tab and select
  switchStyleTab('presets');
  const card = document.getElementById(`preset-${key}`);
  if (card) { selectPreset(key, card); }
  document.getElementById('studio').scrollIntoView({ behavior: 'smooth' });
}

// ── Drag & Drop ──────────────────────────────────────────────────
function setupDnD(zoneId, inputId, handler) {
  const zone = document.getElementById(zoneId);
  if (!zone) return;
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handler(file);
    else toast('Please drop an image file', 'error');
  });
}

// ── File handlers ────────────────────────────────────────────────
function handleContentFile(file) {
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) { toast('Image too large (max 10MB)', 'error'); return; }
  contentFile = file;
  showImagePreview(file, 'contentPreview', 'contentPlaceholder', 'contentClear');
  // Update compare box
  const reader = new FileReader();
  reader.onload = e => { document.getElementById('compareContent').src = e.target.result; };
  reader.readAsDataURL(file);
  // Show mask toolbar and reset mask
  document.getElementById('maskToolbar').classList.remove('hidden');
  resetMaskCanvas();
}

function handleStyleFile(file) {
  if (!file) return;
  styleFile = file;
  clearPresetSelection();
  showImagePreview(file, 'styleCustomPreview', 'stylePlaceholder', 'styleClear');
}

// ── Style Mixing ─────────────────────────────────────────────────
function toggleStyleMix() {
  styleMixEnabled = !styleMixEnabled;
  const toggle = document.getElementById('styleMixToggle');
  const body = document.getElementById('styleMixBody');
  toggle.classList.toggle('active', styleMixEnabled);
  if (styleMixEnabled) {
    body.classList.remove('hidden');
  } else {
    body.classList.add('hidden');
    // Clear second style when disabled
    styleFile2 = null;
    selectedPreset2 = null;
    clearPresetSelection2();
    resetUploadZone('styleCustomPreview2', 'stylePlaceholder2', 'styleClear2', 'styleCustomInput2');
  }
}

function switchStyleTab2(tab) {
  const isPreset = tab === 'presets';
  document.getElementById('tabPresets2').classList.toggle('active', isPreset);
  document.getElementById('tabCustom2').classList.toggle('active', !isPreset);
  document.getElementById('presetPanel2').style.display = isPreset ? 'grid' : 'none';
  document.getElementById('customPanel2').style.display = isPreset ? 'none' : 'flex';
  if (!isPreset) { selectedPreset2 = null; clearPresetSelection2(); }
}

function renderPresets2(styles) {
  const grid = document.getElementById('presetPanel2');
  if (!grid) return;
  grid.innerHTML = '';
  styles.forEach(s => {
    const card = document.createElement('div');
    card.className = 'preset-card';
    card.id = `preset2-${s.key}`;
    card.innerHTML = `
      ${s.thumbnail
        ? `<img src="data:image/jpeg;base64,${s.thumbnail}" alt="${s.name}" loading="lazy" />`
        : `<div style="background:var(--surface-2);width:100%;height:100%;display:flex;align-items:center;justify-content:center;font-size:20px;">🎨</div>`}
      <div class="preset-caption">${s.name}</div>
      <div class="preset-check">✓</div>
    `;
    card.addEventListener('click', () => selectPreset2(s.key, card));
    card.title = `${s.name} by ${s.artist}`;
    grid.appendChild(card);
  });
}

function selectPreset2(key, el) {
  clearPresetSelection2();
  el.classList.add('active');
  selectedPreset2 = key;
  styleFile2 = null;
  toast(`Second style: ${el.querySelector('.preset-caption').textContent}`, 'info');
}

function clearPresetSelection2() {
  document.querySelectorAll('#presetPanel2 .preset-card.active').forEach(el => el.classList.remove('active'));
  selectedPreset2 = null;
}

function handleStyleFile2(file) {
  if (!file) return;
  styleFile2 = file;
  clearPresetSelection2();
  showImagePreview(file, 'styleCustomPreview2', 'stylePlaceholder2', 'styleClear2');
}

function updateMixRatio(val) {
  const a = 100 - parseInt(val);
  const b = parseInt(val);
  document.getElementById('mixRatioVal').textContent = `${a} / ${b}`;
}

function showImagePreview(file, previewId, placeholderId, clearId) {
  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById(previewId);
    img.src = e.target.result;
    img.classList.remove('hidden');
    document.getElementById(placeholderId).classList.add('hidden');
    document.getElementById(clearId).classList.remove('hidden');
  };
  reader.readAsDataURL(file);
}

function clearImage(type, e) {
  e.stopPropagation();
  if (type === 'content') {
    contentFile = null;
    resetUploadZone('contentPreview', 'contentPlaceholder', 'contentClear', 'contentInput');
    // Hide mask UI
    if (maskMode) toggleMaskMode();
    document.getElementById('maskToolbar').classList.add('hidden');
    document.getElementById('maskCanvas').classList.add('hidden');
    maskHasContent = false;
  } else if (type === 'style2') {
    styleFile2 = null;
    resetUploadZone('styleCustomPreview2', 'stylePlaceholder2', 'styleClear2', 'styleCustomInput2');
  } else {
    styleFile = null;
    resetUploadZone('styleCustomPreview', 'stylePlaceholder', 'styleClear', 'styleCustomInput');
  }
}

function resetUploadZone(previewId, placeholderId, clearId, inputId) {
  document.getElementById(previewId).classList.add('hidden');
  document.getElementById(previewId).src = '';
  document.getElementById(placeholderId).classList.remove('hidden');
  document.getElementById(clearId).classList.add('hidden');
  document.getElementById(inputId).value = '';
}

// ── Start Transfer ───────────────────────────────────────────────
async function startTransfer() {
  if (!contentFile) { toast('Please upload a content image first', 'error'); return; }
  if (!selectedPreset && !styleFile) { toast('Please pick a style preset or upload one', 'error'); return; }

  const btn = document.getElementById('generateBtn');
  btn.disabled = true;
  btn.querySelector('.btn-text').textContent = 'Sending to AI…';

  closeWS();
  _doneReceived = false;
  _polling = false;
  setOutputState('progress');
  document.getElementById('outputActions').style.display = 'none';
  document.getElementById('resultCompare').style.display = 'none';

  // Reset progress UI for fresh run
  document.getElementById('progressBar').style.width = '0%';
  document.getElementById('progressPct').textContent = '0%';
  document.getElementById('progressStep').textContent = 'Phase 0 / 4';
  document.getElementById('progressLabel').textContent = 'Initialising AI…';
  document.getElementById('progressLoss').textContent = '';
  const prevImg = document.getElementById('progressPreview');
  prevImg.src = '';
  prevImg.style.opacity = '0';
  const barFill = document.querySelector('.progress-bar-fill');
  const spinner = document.querySelector('.progress-spinner');
  if (barFill) { barFill.classList.remove('paused'); barFill.classList.remove('completed'); }
  if (spinner) { spinner.classList.remove('paused'); spinner.classList.remove('completed'); spinner.innerHTML = ''; }

  // Build form data
  const form = new FormData();
  form.append('content_image', contentFile);
  if (styleFile) form.append('style_image', styleFile);
  if (selectedPreset) form.append('style_preset', selectedPreset);
  // Style intensity slider sends 0-100, convert to 0.0-1.0 alpha
  const styleIntensity = parseFloat(document.getElementById('styleWeight').value) / 100;
  form.append('style_weight', styleIntensity.toString());
  form.append('content_weight', '1');
  form.append('tv_weight', '0');
  form.append('num_steps', '3');
  form.append('learning_rate', '0.02');

  // Style Mixing — optional second style
  if (styleMixEnabled) {
    if (styleFile2) form.append('style_image_2', styleFile2);
    if (selectedPreset2) form.append('style_preset_2', selectedPreset2);
    const mixRatio = parseFloat(document.getElementById('mixRatio').value) / 100;
    form.append('style_mix_ratio', mixRatio.toString());
  }

  // Regional Styling — optional mask
  if (maskHasContent) {
    const maskBlob = getMaskBlob();
    if (maskBlob) form.append('mask_image', maskBlob, 'mask.png');
  }

  try {
    const res = await fetch(`${API}/api/transfer`, { method: 'POST', body: form });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Server error'); }
    const data = await res.json();
    currentJobId = data.job_id;
    toast('Job started! Streaming progress…', 'success');
    btn.querySelector('.btn-text').textContent = 'Processing…';
    connectWS(currentJobId);
  } catch (err) {
    showError(err.message);
    btn.disabled = false;
    btn.querySelector('.btn-text').textContent = 'Generate Artwork';
  }
}

// ── WebSocket ────────────────────────────────────────────────────
function connectWS(jobId) {
  const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
  const host = location.host || 'localhost:8000';
  const ws = new WebSocket(`${wsProto}://${host}/ws/${jobId}`);
  currentWS = ws;

  ws.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'progress') onProgress(msg);
    else if (msg.type === 'done') onDone(msg);
    else if (msg.type === 'error') onWsError(msg);
    // ping: ignore
  };
  ws.onerror = () => { if (!_doneReceived && !_polling) pollFallback(jobId); };
  ws.onclose = () => { if (!_doneReceived && !_polling && currentJobId) pollFallback(currentJobId); };
}

function closeWS() {
  if (currentWS) { currentWS.close(); currentWS = null; }
}

// ── Progress handler ─────────────────────────────────────────────
function onProgress(msg) {
  const pct = msg.percent ?? 0;
  const step = msg.step ?? 0;
  const total = msg.total ?? 3;

  document.getElementById('progressBar').style.width = `${pct}%`;
  document.getElementById('progressPct').textContent = `${pct}%`;
  document.getElementById('progressStep').textContent = total <= 5
    ? `Phase ${step} / ${total}` : `Step ${step} / ${total}`;

  // Label based on phase for the fast model
  if (pct >= 100) {
    document.getElementById('progressLabel').textContent = 'Artwork ready!';
    const bar = document.querySelector('.progress-bar-fill');
    const spin = document.querySelector('.progress-spinner');
    if (bar) { bar.classList.add('completed'); }
    if (spin) {
      spin.classList.add('completed');
      spin.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
    }
    // Safety: if onDone hasn't fired within 4s, poll the server
    if (!_doneReceived && currentJobId) {
      const jid = currentJobId;
      setTimeout(() => { if (!_doneReceived) pollFallback(jid); }, 4000);
    }
  } else if (pct < 20) {
    document.getElementById('progressLabel').textContent = 'Loading AI model…';
  } else if (pct < 40) {
    document.getElementById('progressLabel').textContent = 'Preprocessing images…';
  } else if (pct < 60) {
    document.getElementById('progressLabel').textContent = 'Applying style…';
  } else if (pct < 85) {
    document.getElementById('progressLabel').textContent = 'Enhancing details…';
  } else {
    document.getElementById('progressLabel').textContent = 'Sharpening & finalising…';
  }

  if (msg.loss != null && msg.loss > 0) {
    document.getElementById('progressLoss').textContent = `Loss: ${msg.loss.toExponential(2)}`;
  } else {
    document.getElementById('progressLoss').textContent = '';
  }
  if (msg.preview) {
    const img = document.getElementById('progressPreview');
    img.src = `data:image/jpeg;base64,${msg.preview}`;
    img.style.opacity = '1';
  }
}

// ── Done handler ─────────────────────────────────────────────────
function onDone(msg) {
  _doneReceived = true;
  closeWS();
  resultB64 = msg.result;
  const imgSrc = `data:image/jpeg;base64,${resultB64}`;
  const resultImg = document.getElementById('resultImage');
  resultImg.src = imgSrc;
  document.getElementById('compareResult').src = imgSrc;

  document.getElementById('resultMeta').textContent = `Styled with Magenta Arbitrary Style Transfer · PicturaAI`;

  setOutputState('result');
  document.getElementById('outputActions').style.display = 'flex';
  document.getElementById('resultCompare').style.display = 'block';

  // Reset Before/After slider to 50 %
  const baImgBefore = document.querySelector('.ba-img-before');
  const baHandle = document.getElementById('baHandle');
  if (baImgBefore) baImgBefore.style.clipPath = 'inset(0 50% 0 0)';
  if (baHandle) baHandle.style.left = '50%';

  const btn = document.getElementById('generateBtn');
  btn.disabled = false;
  btn.querySelector('.btn-text').textContent = 'Regenerate Artwork';
  addToHistory(resultB64, currentJobId);
  toast('🎨 PicturaAI — your masterpiece is ready!', 'success');
}

function onWsError(msg) {
  closeWS();
  showError(msg.message || 'An error occurred');
}

// ── Poll fallback (if WS fails) ──────────────────────────────────
async function pollFallback(jobId) {
  _polling = true;
  try {
    const res = await fetch(`${API}/api/jobs/${jobId}`);
    const data = await res.json();
    if (data.status === 'done') {
      onDone({ result: data.result });
    } else if (data.status === 'error') {
      showError(data.error);
    } else {
      onProgress({ percent: data.progress || 0, preview: data.preview });
      setTimeout(() => pollFallback(jobId), 2000);
    }
  } catch (e) {
    setTimeout(() => pollFallback(jobId), 3000);
  }
}

// ── Output state ─────────────────────────────────────────────────
function setOutputState(state) {
  document.getElementById('outputIdle').classList.add('hidden');
  document.getElementById('outputProgress').classList.add('hidden');
  document.getElementById('outputResult').classList.add('hidden');
  document.getElementById('outputError').classList.add('hidden');
  document.getElementById(`output${state.charAt(0).toUpperCase() + state.slice(1)}`).classList.remove('hidden');
}

function showError(msg) {
  document.getElementById('errorMessage').textContent = msg;
  setOutputState('error');
  const btn = document.getElementById('generateBtn');
  btn.disabled = false;
  btn.querySelector('.btn-text').textContent = 'Generate Artwork';
  toast(msg, 'error');
}

// ── Download ─────────────────────────────────────────────────────
function downloadResult() {
  if (currentJobId) {
    const a = document.createElement('a');
    a.href = `${API}/api/result/${currentJobId}`;
    a.download = `pictura_${currentJobId.substr(0, 8)}.jpg`;
    a.click();
    toast('Downloading your artwork…', 'info');
  } else if (resultB64) {
    const a = document.createElement('a');
    a.href = `data:image/jpeg;base64,${resultB64}`;
    a.download = 'pictura_result.jpg';
    a.click();
  }
}

// ── Reset ────────────────────────────────────────────────────────
function resetStudio() {
  closeWS();
  currentJobId = null;
  resultB64 = null;

  resetUploadZone('contentPreview', 'contentPlaceholder', 'contentClear', 'contentInput');
  resetUploadZone('styleCustomPreview', 'stylePlaceholder', 'styleClear', 'styleCustomInput');
  clearPresetSelection();
  contentFile = null;
  styleFile = null;

  // Reset style mixing
  if (styleMixEnabled) toggleStyleMix();
  styleFile2 = null;
  selectedPreset2 = null;

  // Reset mask
  if (maskMode) toggleMaskMode();
  document.getElementById('maskToolbar').classList.add('hidden');
  document.getElementById('maskCanvas').classList.add('hidden');
  maskHasContent = false;

  document.getElementById('progressBar').style.width = '0%';
  document.getElementById('progressPct').textContent = '0%';
  document.getElementById('progressStep').textContent = 'Step 0 / 0';
  const barFill = document.querySelector('.progress-bar-fill');
  const spinner = document.querySelector('.progress-spinner');
  if (barFill) { barFill.classList.remove('paused'); barFill.classList.remove('completed'); }
  if (spinner) { spinner.classList.remove('paused'); spinner.classList.remove('completed'); spinner.innerHTML = ''; }

  setOutputState('idle');
  document.getElementById('outputActions').style.display = 'none';

  const btn = document.getElementById('generateBtn');
  btn.disabled = false;
  btn.querySelector('.btn-text').textContent = 'Generate Artwork';
}

// ── Toast ────────────────────────────────────────────────────────
function toast(msg, type = 'info') {
  const icons = { success: '✅', error: '❌', info: 'ℹ️' };
  const t = document.createElement('div');
  t.className = `toast toast-${type}`;
  t.innerHTML = `<span>${icons[type]}</span><span>${msg}</span>`;
  document.getElementById('toastContainer').appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transform = 'translateX(110%)'; t.style.transition = 'all 0.4s'; setTimeout(() => t.remove(), 400); }, 4000);
}

// ── Generation History ────────────────────────────────────────────
function addToHistory(b64, jobId) {
  const styleName = selectedPreset
    ? document.querySelector(`#preset-${selectedPreset} .preset-caption`)?.textContent || selectedPreset
    : 'Custom';
  const entry = {
    id: jobId || Date.now().toString(),
    b64,
    style: styleName,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  };
  generationHistory.unshift(entry);
  if (generationHistory.length > MAX_HISTORY) generationHistory.pop();
  activeHistoryIdx = 0;
  renderHistory();
}

function renderHistory() {
  const strip = document.getElementById('historyStrip');
  const scroll = document.getElementById('historyScroll');
  const countEl = document.getElementById('historyCount');
  if (!strip || !scroll) return;

  if (generationHistory.length === 0) {
    strip.style.display = 'none';
    return;
  }
  strip.style.display = 'block';
  countEl.textContent = generationHistory.length;

  scroll.innerHTML = '';
  generationHistory.forEach((entry, i) => {
    const card = document.createElement('div');
    card.className = 'history-card' + (i === activeHistoryIdx ? ' active' : '');
    card.title = `${entry.style} · ${entry.timestamp}`;
    card.innerHTML = `
      <img src="data:image/jpeg;base64,${entry.b64}" alt="History ${i + 1}" draggable="false" />
      <span class="history-card-num">${i + 1}</span>
      <button class="history-card-del" title="Remove">✕</button>
    `;
    card.addEventListener('click', (e) => {
      if (e.target.closest('.history-card-del')) return;
      loadHistoryItem(i);
    });
    card.querySelector('.history-card-del').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteHistoryItem(i);
    });
    scroll.appendChild(card);
  });
}

function loadHistoryItem(idx) {
  const entry = generationHistory[idx];
  if (!entry) return;
  activeHistoryIdx = idx;
  resultB64 = entry.b64;
  const imgSrc = `data:image/jpeg;base64,${entry.b64}`;
  document.getElementById('resultImage').src = imgSrc;
  document.getElementById('compareResult').src = imgSrc;
  document.getElementById('resultMeta').textContent = `${entry.style} · ${entry.timestamp} · PicturaAI`;

  // Reset BA slider
  const baImgBefore = document.querySelector('.ba-img-before');
  const baHandle = document.getElementById('baHandle');
  if (baImgBefore) baImgBefore.style.clipPath = 'inset(0 50% 0 0)';
  if (baHandle) baHandle.style.left = '50%';

  renderHistory();
}

function deleteHistoryItem(idx) {
  generationHistory.splice(idx, 1);
  if (activeHistoryIdx === idx) activeHistoryIdx = Math.min(0, generationHistory.length - 1);
  else if (activeHistoryIdx > idx) activeHistoryIdx--;
  renderHistory();
  if (generationHistory.length === 0) {
    document.getElementById('historyStrip').style.display = 'none';
  }
}

function clearHistory() {
  generationHistory = [];
  activeHistoryIdx = -1;
  renderHistory();
}

// ── Before / After Comparison Slider ─────────────────────────────
function initBASlider() {
  const slider = document.getElementById('baSlider');
  if (!slider) return;
  const beforeImg = slider.querySelector('.ba-img-before');
  const handle = document.getElementById('baHandle');
  let dragging = false;

  function setPosition(x) {
    const rect = slider.getBoundingClientRect();
    let pct = ((x - rect.left) / rect.width) * 100;
    pct = Math.max(0, Math.min(100, pct));
    beforeImg.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    handle.style.left = pct + '%';
  }

  slider.addEventListener('mousedown', e => { e.preventDefault(); dragging = true; setPosition(e.clientX); });
  slider.addEventListener('touchstart', e => { dragging = true; setPosition(e.touches[0].clientX); }, { passive: true });
  window.addEventListener('mousemove', e => { if (dragging) { e.preventDefault(); setPosition(e.clientX); } });
  window.addEventListener('touchmove', e => { if (dragging) setPosition(e.touches[0].clientX); }, { passive: true });
  window.addEventListener('mouseup', () => { dragging = false; });
  window.addEventListener('touchend', () => { dragging = false; });
}

// ── Mask Painting (Regional Styling) ─────────────────────────────
function resetMaskCanvas() {
  const canvas = document.getElementById('maskCanvas');
  const preview = document.getElementById('contentPreview');
  // Size canvas to match the displayed preview image
  canvas.width = preview.naturalWidth || 400;
  canvas.height = preview.naturalHeight || 400;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  maskHasContent = false;
  canvas.classList.add('hidden');
  canvas.classList.remove('active');
}

function toggleMaskMode() {
  maskMode = !maskMode;
  const canvas = document.getElementById('maskCanvas');
  const btn = document.getElementById('maskToggleBtn');
  const controls = document.getElementById('maskControls');
  const hint = document.getElementById('maskHint');

  btn.classList.toggle('active', maskMode);
  if (maskMode) {
    // Size canvas to the preview element's display size for proper coordinate mapping
    const preview = document.getElementById('contentPreview');
    canvas.width = preview.naturalWidth || 400;
    canvas.height = preview.naturalHeight || 400;
    canvas.classList.remove('hidden');
    canvas.classList.add('active');
    controls.classList.remove('hidden');
    hint.style.display = '';
    setMaskBrush('draw');
  } else {
    canvas.classList.remove('active');
    controls.classList.add('hidden');
    hint.style.display = 'none';
    // Keep canvas visible if user painted on it (so they see the overlay)
    if (!maskHasContent) canvas.classList.add('hidden');
  }
}

function setMaskBrush(type) {
  maskBrushType = type;
  document.getElementById('brushDraw').classList.toggle('active', type === 'draw');
  document.getElementById('brushErase').classList.toggle('active', type === 'erase');
}

function updateBrushSize(val) {
  maskBrushSize = parseInt(val);
  document.getElementById('brushSizeVal').textContent = val;
}

function fillMask() {
  const canvas = document.getElementById('maskCanvas');
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(34, 197, 94, 0.45)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  maskHasContent = true;
}

function clearMask() {
  const canvas = document.getElementById('maskCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  maskHasContent = false;
}

function getMaskBlob() {
  // Convert the painted mask canvas to a binary (white = style, black = original) PNG
  const src = document.getElementById('maskCanvas');
  const w = src.width, h = src.height;
  const srcCtx = src.getContext('2d');
  const data = srcCtx.getImageData(0, 0, w, h);

  const out = document.createElement('canvas');
  out.width = w; out.height = h;
  const oCtx = out.getContext('2d');
  const oData = oCtx.createImageData(w, h);

  for (let i = 0; i < data.data.length; i += 4) {
    // Any painted pixel (alpha > 0) becomes white (style), else black (original)
    const val = data.data[i + 3] > 10 ? 255 : 0;
    oData.data[i] = val;
    oData.data[i + 1] = val;
    oData.data[i + 2] = val;
    oData.data[i + 3] = 255;
  }
  oCtx.putImageData(oData, 0, 0);

  // Synchronous conversion to blob — use toBlob workaround via dataURL
  const dataURL = out.toDataURL('image/png');
  const bin = atob(dataURL.split(',')[1]);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new Blob([arr], { type: 'image/png' });
}

// ── Mask canvas drawing events ───────────────────────────────────
(function initMaskEvents() {
  const canvas = document.getElementById('maskCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: (clientX - rect.left) * scaleX, y: (clientY - rect.top) * scaleY };
  }

  function paint(e) {
    if (!maskPainting || !maskMode) return;
    const { x, y } = getPos(e);
    const scaleFactor = canvas.width / canvas.getBoundingClientRect().width;
    const r = maskBrushSize * scaleFactor / 2;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    if (maskBrushType === 'draw') {
      ctx.fillStyle = 'rgba(34, 197, 94, 0.45)';
      ctx.globalCompositeOperation = 'source-over';
      ctx.fill();
    } else {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.fillStyle = 'rgba(0,0,0,1)';
      ctx.fill();
    }
    ctx.globalCompositeOperation = 'source-over';
    if (maskBrushType === 'draw') maskHasContent = true;
  }

  canvas.addEventListener('mousedown', e => { e.preventDefault(); maskPainting = true; paint(e); });
  canvas.addEventListener('touchstart', e => { maskPainting = true; paint(e); }, { passive: true });
  canvas.addEventListener('mousemove', paint);
  canvas.addEventListener('touchmove', paint, { passive: true });
  window.addEventListener('mouseup', () => { maskPainting = false; });
  window.addEventListener('touchend', () => { maskPainting = false; });
})();
