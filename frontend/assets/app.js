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

// ── Navbar scroll effect ─────────────────────────────────────────
window.addEventListener('scroll', () => {
  document.getElementById('navbar').classList.toggle('scrolled', window.scrollY > 40);
});

// ── On load ──────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  loadStyles();
  setupDnD('contentUploadZone', 'contentInput', handleContentFile);
  setupDnD('customPanel', 'styleCustomInput', handleStyleFile);
  document.getElementById('contentInput').addEventListener('change', e => handleContentFile(e.target.files[0]));
  document.getElementById('styleCustomInput').addEventListener('change', e => handleStyleFile(e.target.files[0]));
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
}

function handleStyleFile(file) {
  if (!file) return;
  styleFile = file;
  clearPresetSelection();
  showImagePreview(file, 'styleCustomPreview', 'stylePlaceholder', 'styleClear');
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
  const baBefore = document.getElementById('baBefore');
  const baHandle = document.getElementById('baHandle');
  if (baBefore && baHandle) {
    baBefore.style.width = '50%';
    baHandle.style.left = '50%';
  }

  const btn = document.getElementById('generateBtn');
  btn.disabled = false;
  btn.querySelector('.btn-text').textContent = 'Regenerate Artwork';
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

// ── Before / After Comparison Slider ─────────────────────────────
function initBASlider() {
  const slider = document.getElementById('baSlider');
  if (!slider) return;
  const before = document.getElementById('baBefore');
  const handle = document.getElementById('baHandle');
  const beforeImg = before.querySelector('.ba-img-before');
  let dragging = false;

  function setPosition(x) {
    const rect = slider.getBoundingClientRect();
    let pct = ((x - rect.left) / rect.width) * 100;
    pct = Math.max(0, Math.min(100, pct));
    before.style.width = pct + '%';
    handle.style.left = pct + '%';
    // Keep the before-image aligned to the full slider width
    beforeImg.style.width = (rect.width) + 'px';
  }

  slider.addEventListener('mousedown', e => { dragging = true; setPosition(e.clientX); });
  slider.addEventListener('touchstart', e => { dragging = true; setPosition(e.touches[0].clientX); }, { passive: true });
  window.addEventListener('mousemove', e => { if (dragging) setPosition(e.clientX); });
  window.addEventListener('touchmove', e => { if (dragging) setPosition(e.touches[0].clientX); }, { passive: true });
  window.addEventListener('mouseup', () => { dragging = false; });
  window.addEventListener('touchend', () => { dragging = false; });

  // Ensure before-image width stays correct on resize
  new ResizeObserver(() => {
    const rect = slider.getBoundingClientRect();
    beforeImg.style.width = rect.width + 'px';
  }).observe(slider);
}
