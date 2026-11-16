// Normalise /api/styles entries. Supports the new shape (slug, title, year, image,
// card, thumb, aspect, credit, page) and the legacy one (name + base64 thumbnail).

function assetUrl(path) {
  if (!path) return '';
  if (/^(https?:|data:|blob:|\/)/.test(path)) return path;
  return path;   // relative to /app/, which is where the page lives
}

export function normalizeStyle(s) {
  const legacyThumb = s.thumbnail ? `data:image/jpeg;base64,${s.thumbnail}` : '';
  const card = assetUrl(s.card) || legacyThumb || assetUrl(s.image);
  const aspect = Number(s.aspect) > 0 ? Number(s.aspect) : 1;
  return {
    key: s.key,
    slug: s.slug || s.key,
    title: s.title || s.name || s.key,
    artist: s.artist || '',
    year: s.year ? String(s.year) : '',
    description: s.description || '',
    image: assetUrl(s.image) || card,     // <= 512 px, for on-device work and colour matching
    card,
    thumb: assetUrl(s.thumb) || card,
    aspect: Math.min(2.2, Math.max(0.45, aspect)),
    credit: s.credit || '',
    page: s.page || (s.slug ? `styles/${s.slug}/` : ''),
  };
}

export const normalizeStyles = (list) => (Array.isArray(list) ? list : (list && list.styles) || [])
  .filter((s) => s && s.key)
  .map(normalizeStyle);

export function artistLine(s) {
  return [s.artist, s.year].filter(Boolean).join(', ');
}

/** Closest preset for some words (only used when /api/describe-style is unreachable). */
export function matchStyleLocally(styles, prompt) {
  const words = String(prompt).toLowerCase().match(/[a-z]{3,}/g) || [];
  let best = null;
  let bestScore = 0;
  for (const s of styles) {
    const hay = `${s.title} ${s.artist} ${s.description}`.toLowerCase();
    const score = words.reduce((n, w) => n + (hay.includes(w) ? 1 : 0), 0);
    if (score > bestScore) { best = s; bestScore = score; }
  }
  return best;
}
