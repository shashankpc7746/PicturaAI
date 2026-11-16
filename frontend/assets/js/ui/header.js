// Header: mobile menu sheet (focus trap, Esc closes) and the current-section marker.
// Works on the studio and on static pages (assets/js/page.js).

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function initHeader() {
  const sheet = document.getElementById('menu-sheet');
  const opener = document.querySelector('[data-menu-open]');
  if (sheet && opener) initSheet(sheet, opener);
  initCurrentSection();
}

function initSheet(sheet, opener) {
  let lastFocus = null;

  function open() {
    lastFocus = document.activeElement;
    sheet.hidden = false;
    sheet.classList.remove('is-closing');
    opener.setAttribute('aria-expanded', 'true');
    document.documentElement.style.overflow = 'hidden';
    const first = sheet.querySelector('.menu-sheet__nav a');
    (first || sheet.querySelector(FOCUSABLE))?.focus();
    document.addEventListener('keydown', onKey);
  }

  function close({ restoreFocus = true } = {}) {
    if (sheet.hidden) return;
    document.removeEventListener('keydown', onKey);
    opener.setAttribute('aria-expanded', 'false');
    document.documentElement.style.overflow = '';
    const done = () => { sheet.hidden = true; sheet.classList.remove('is-closing'); };
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) done();
    else { sheet.classList.add('is-closing'); setTimeout(done, 190); }
    if (restoreFocus && lastFocus) lastFocus.focus();
  }

  function onKey(e) {
    if (e.key === 'Escape') { e.preventDefault(); close(); return; }
    if (e.key !== 'Tab') return;
    const items = Array.from(sheet.querySelectorAll(FOCUSABLE)).filter((el) => el.offsetParent !== null);
    if (!items.length) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }

  opener.addEventListener('click', open);
  sheet.addEventListener('click', (e) => {
    if (e.target.closest('[data-menu-close]')) close();
    else if (e.target.closest('a[href]')) close({ restoreFocus: false });
  });
  matchMedia('(min-width: 768px)').addEventListener('change', (e) => { if (e.matches) close({ restoreFocus: false }); });
}

function initCurrentSection() {
  const links = Array.from(document.querySelectorAll('.site-nav a[href^="#"]'));
  if (!links.length || !('IntersectionObserver' in window)) return;
  const byId = new Map(links.map((a) => [a.getAttribute('href').slice(1), a]));
  const visible = new Map();
  const io = new IntersectionObserver((entries) => {
    for (const en of entries) visible.set(en.target.id, en.isIntersecting ? en.intersectionRatio : 0);
    let best = null; let ratio = 0;
    for (const [id, r] of visible) if (r > ratio) { best = id; ratio = r; }
    links.forEach((a) => a.removeAttribute('aria-current'));
    if (best && byId.get(best)) byId.get(best).setAttribute('aria-current', 'true');
  }, { rootMargin: '-30% 0px -50% 0px', threshold: [0, 0.01, 0.25, 0.5] });
  for (const id of byId.keys()) {
    const sec = document.getElementById(id);
    if (sec) io.observe(sec);
  }
}
