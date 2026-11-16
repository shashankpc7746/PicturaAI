// Reading, checking and converting images in the browser.

export const MAX_BYTES = 25 * 1024 * 1024;
export const MAX_MEGAPIXELS = 50;
export const WORK_SIDE = 1600;      // the photo is kept at most this big in memory
const TYPES = { 'image/jpeg': 'JPG', 'image/png': 'PNG', 'image/webp': 'WEBP' };
// Rounded up to one decimal, so a size just over a limit never reads as the limit.
const up1 = (n) => (Math.ceil(n * 10) / 10).toFixed(1);

/** Quick checks before decoding. Returns a reason string, or null when fine. */
export function checkFile(file) {
  if (!file) return 'No file was chosen.';
  const okType = TYPES[file.type] || (!file.type && /\.(jpe?g|png|webp)$/i.test(file.name));
  if (!okType) {
    const ext = (file.name.split('.').pop() || '').toUpperCase();
    return `${ext && ext.length <= 5 ? `That ${ext} file` : 'That file'} isn't supported. Use a JPG, PNG or WEBP image.`;
  }
  if (file.size > MAX_BYTES) {
    return `That image is ${up1(file.size / 1048576)} MB, over the 25 MB limit. Use a smaller file.`;
  }
  return null;
}

/** Decode a file through <img> (respects EXIF orientation). */
export function loadImageFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => {
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      if (!w || !h) { URL.revokeObjectURL(url); reject(new Error("We couldn't read that image. Try another file.")); return; }
      const mp = (w * h) / 1e6;
      if (mp > MAX_MEGAPIXELS) {
        URL.revokeObjectURL(url);
        reject(new Error(`That image is ${up1(mp)} megapixels, over the ${MAX_MEGAPIXELS}-megapixel limit. Use a smaller image.`));
        return;
      }
      resolve({ img, url, width: w, height: h });
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("We couldn't open that file. It may be damaged, or not really an image."));
    };
    img.src = url;
  });
}

export function loadImageUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Image could not be loaded.'));
    img.src = url;
  });
}

function sizeOf(src) {
  return [src.naturalWidth || src.videoWidth || src.width, src.naturalHeight || src.videoHeight || src.height];
}

/**
 * Draw any image source into a new canvas no larger than maxSide (high quality, stepped).
 * `background` flattens transparency onto a colour: photos and style images need it,
 * because JPEG encoding and the compositor would otherwise turn transparent pixels
 * black. Masks must keep their alpha, so it is off by default.
 */
export function toCanvas(src, maxSide = Infinity, { background = null } = {}) {
  let [w, h] = sizeOf(src);
  const k = Math.min(1, maxSide / Math.max(w, h));
  const tw = Math.max(1, Math.round(w * k));
  const th = Math.max(1, Math.round(h * k));
  let cur = src;
  // Halve in steps so large reductions average every pixel.
  while (w / 2 >= tw && h / 2 >= th) {
    w = Math.round(w / 2); h = Math.round(h / 2);
    const step = document.createElement('canvas');
    step.width = w; step.height = h;
    const sctx = step.getContext('2d');
    sctx.imageSmoothingQuality = 'high';
    sctx.drawImage(cur, 0, 0, w, h);
    cur = step;
  }
  const out = document.createElement('canvas');
  out.width = tw; out.height = th;
  const ctx = out.getContext('2d', { willReadFrequently: false });
  ctx.imageSmoothingQuality = 'high';
  if (background) {
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, tw, th);
  }
  ctx.drawImage(cur, 0, 0, tw, th);
  return out;
}

export function imageDataToCanvas(imageData, canvas = document.createElement('canvas')) {
  if (canvas.width !== imageData.width) canvas.width = imageData.width;
  if (canvas.height !== imageData.height) canvas.height = imageData.height;
  canvas.getContext('2d').putImageData(imageData, 0, 0);
  return canvas;
}

export function canvasToImageData(canvas) {
  return canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, canvas.width, canvas.height);
}

export function canvasToBlob(canvas, type = 'image/jpeg', quality = 0.92) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Could not create the image file.'))), type, quality);
  });
}

export function blobToImage(blob) {
  const url = URL.createObjectURL(blob);
  return loadImageUrl(url).finally(() => setTimeout(() => URL.revokeObjectURL(url), 0));
}
