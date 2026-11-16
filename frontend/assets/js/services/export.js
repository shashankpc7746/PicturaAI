// Download sizes, file saving and the Web Share API.
import { canvasToBlob } from './image-io.js';

const PRINT_LONG = 1754;   // A4 at 150 dpi
const IG_WIDTH = 1080;

// A size this much bigger than the painting is an enlargement and is labelled so
// (the server engine paints at 320 px, so Print is about 5x).
const ENLARGED = 1.05;

/**
 * Target sizes for a picture of w x h. Instagram allows 4:5 to 1.91:1, so outside that
 * we crop. `scale` is how much each size enlarges the painting (1 = no enlargement),
 * and `enlarged` is true when that would visibly soften it.
 */
export function exportSizes(w, h) {
  const ar = w / h;
  const igAr = Math.min(1.91, Math.max(0.8, ar));
  const k = PRINT_LONG / Math.max(w, h);
  const ig = { width: IG_WIDTH, height: Math.round(IG_WIDTH / igAr), crop: igAr !== ar };
  // A crop keeps the full height (wide photos) or the full width (tall ones).
  ig.scale = igAr < ar ? ig.height / h : ig.width / w;
  const sizes = {
    full: { width: w, height: h, scale: 1 },
    instagram: ig,
    print: { width: Math.round(w * k), height: Math.round(h * k), scale: k },
  };
  for (const s of Object.values(sizes)) s.enlarged = s.scale > ENLARGED;
  return sizes;
}

/** Render the source canvas at a size, centre-cropping when the aspect differs. */
export function renderAt(source, { width, height }) {
  if (width === source.width && height === source.height) return source;
  const out = document.createElement('canvas');
  out.width = width; out.height = height;
  const ctx = out.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  const sAr = source.width / source.height;
  const tAr = width / height;
  let sw = source.width; let sh = source.height; let sx = 0; let sy = 0;
  if (sAr > tAr + 1e-3) { sw = Math.round(sh * tAr); sx = Math.round((source.width - sw) / 2); }
  else if (sAr < tAr - 1e-3) { sh = Math.round(sw / tAr); sy = Math.round((source.height - sh) / 2); }
  ctx.drawImage(source, sx, sy, sw, sh, 0, 0, width, height);
  return out;
}

export async function exportBlob(source, size) {
  return canvasToBlob(renderAt(source, size), 'image/jpeg', 0.92);
}

export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 15000);
}

export function fileName(title, suffix = '', ext = 'jpg') {
  const slug = String(title || 'painting').toLowerCase().normalize('NFKD')
    .replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'painting';
  return `pictura-${slug}${suffix ? `-${suffix}` : ''}.${ext}`;
}

/** True when this browser can share image files. */
export function canShareFiles() {
  try {
    if (!navigator.share || !navigator.canShare) return false;
    const probe = new File([new Blob(['x'], { type: 'image/jpeg' })], 'p.jpg', { type: 'image/jpeg' });
    return navigator.canShare({ files: [probe] });
  } catch {
    return false;
  }
}

export async function shareBlob(blob, filename, title) {
  const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
  await navigator.share({ files: [file], title, text: 'Painted with PicturaAI' });
}
