// PicturaAI compositor: a port of the post-processing in backend/nst_engine.py
// (run_nst phases 3-4 and color_palette_transfer), so the browser engine and the
// server engine produce the same picture from the same raw stylization.
//
// Pure functions over ImageData-like objects ({ data, width, height }). The only
// DOM use is resizeImageData, which draws through a (Offscreen)Canvas.
//
// Server pipeline being reproduced, per pixel in float [0, 1]:
//   1. blended = a*S + (1-a)*C, then scaled so its Rec.709 luminance moves toward
//      the content's: target = k*lumC + (1-k)*lumB with k = 1 - 0.6a.
//   2. + (0.15 + 0.30a) * (C - avgpool5x5(C))   high-frequency detail re-injection
//   3. optional mask: m*result + (1-m)*C, mask Gaussian-blurred (radius 6 at 320 px)
//   4. to 8 bits (truncating, like numpy astype(uint8)), then Pillow's
//      UnsharpMask(radius 1.2, percent 40 + 50a, threshold 2).
// Pillow's Gaussian blur is an extended box blur (3 passes per axis, 8-bit
// rounding after each pass); it is ported exactly so the numbers match.

const LUM_R = 0.2126;
const LUM_G = 0.7152;
const LUM_B = 0.0722;
const SERVER_MASK_RADIUS = 6;   // px, at the server's 320 px output
const SERVER_MASK_SIDE = 320;
const INV_2_24 = 1 / 16777216; // exact power of two: same as integer >> 24 after the store truncates

// ── ImageData helpers ───────────────────────────────────────────────────────

export function makeImageData(width, height, data) {
  const px = data || new Uint8ClampedArray(width * height * 4);
  if (typeof ImageData !== 'undefined') return new ImageData(px, width, height);
  return { data: px, width, height, colorSpace: 'srgb' };
}

function assertSameSize(a, b, what) {
  if (!a || !b || a.width !== b.width || a.height !== b.height) {
    throw new RangeError(`${what}: images must be the same size (resize first)`);
  }
}

// ── Pillow-compatible blurs ─────────────────────────────────────────────────

// Pillow _gaussian_blur_radius(): box radius (fractional) that approximates a
// Gaussian with standard deviation `radius` over `passes` box passes.
function pilBoxRadius(radius, passes) {
  const sigma2 = (radius * radius) / passes;
  const L = Math.sqrt(12.0 * sigma2 + 1.0);
  const l = Math.floor((L - 1.0) / 2.0);
  let a = (2 * l + 1) * (l * (l + 1) - 3 * sigma2);
  a /= 6 * (sigma2 - (l + 1) * (l + 1));
  return Math.fround(l + a);
}

// One extended-box-blur pass along a line of an 8-bit plane, Pillow fixed point.
// src/dst: typed arrays; start/stride address the line; n = line length.
// `line` is scratch space of length >= n + 2 * (radius + 1).
function boxLine(src, dst, start, stride, n, radius, ww, fw, line) {
  const pad = radius + 1;
  const first = src[start];
  const lastV = src[start + (n - 1) * stride];
  // Edge-replicated copy: line[pad + i] = src[clamp(i)].
  for (let i = 0; i < pad; i++) line[i] = first;
  for (let i = 0, p = start; i < n; i++, p += stride) line[pad + i] = src[p];
  for (let i = pad + n, e = n + 2 * pad; i < e; i++) line[i] = lastV;
  // acc = sum of the window centred on x = -1.
  let acc = 0;
  for (let k = 0; k < 2 * radius + 1; k++) acc += line[k];
  for (let x = 0, p = start; x < n; x++, p += stride) {
    const out = x + pad - radius - 1;   // index of x - radius - 1
    const inn = x + pad + radius;       // index of x + radius
    acc += line[inn] - line[out];
    const bulk = acc * ww + (line[out] + line[inn + 1]) * fw;
    dst[p] = (bulk + 8388608) * INV_2_24;
  }
}

// Pillow ImagingBoxBlur on one 8-bit plane: `passes` horizontal passes, then
// `passes` vertical passes, rounding to 8 bits after each pass.
function pilBoxBlurPlane(plane, w, h, floatRadius, passes) {
  const radius = Math.floor(floatRadius);
  const ww = Math.floor(16777216 / (floatRadius * 2 + 1));
  const fw = Math.floor((16777216 - (radius * 2 + 1) * ww) / 2);
  let a = plane;
  let b = new Uint8Array(w * h);
  if (radius === 0) {
    // Small blurs (the unsharp mask's 1.2 px) are a 3-tap kernel [fw, ww, fw].
    for (let p = 0; p < passes; p++) { threeTapH(a, b, w, h, ww, fw); [a, b] = [b, a]; }
    for (let p = 0; p < passes; p++) { threeTapV(a, b, w, h, ww, fw); [a, b] = [b, a]; }
    return a;
  }
  const line = new Uint8Array(Math.max(w, h) + 2 * (radius + 1));
  for (let p = 0; p < passes; p++) {
    for (let y = 0; y < h; y++) boxLine(a, b, y * w, 1, w, radius, ww, fw, line);
    [a, b] = [b, a];
  }
  // Vertical passes on a transposed copy (row access is much more cache friendly).
  let t = transposePlane(a, w, h, b);
  let u = a;
  for (let p = 0; p < passes; p++) {
    for (let x = 0; x < w; x++) boxLine(t, u, x * h, 1, h, radius, ww, fw, line);
    [t, u] = [u, t];
  }
  return transposePlane(t, h, w, u);
}

// Box pass with integer radius 0: out = in*ww + (left + right)*fw, edges clamped.
function threeTapH(src, dst, w, h, ww, fw) {
  for (let y = 0; y < h; y++) {
    const row = y * w;
    const end = row + w - 1;
    for (let p = row; p <= end; p++) {
      const l = src[p === row ? p : p - 1];
      const r = src[p === end ? p : p + 1];
      dst[p] = (src[p] * ww + (l + r) * fw + 8388608) * INV_2_24;
    }
  }
}

function threeTapV(src, dst, w, h, ww, fw) {
  for (let y = 0; y < h; y++) {
    const row = y * w;
    const up = (y === 0 ? 0 : y - 1) * w;
    const down = (y === h - 1 ? y : y + 1) * w;
    for (let x = 0; x < w; x++) {
      dst[row + x] = (src[row + x] * ww + (src[up + x] + src[down + x]) * fw + 8388608) * INV_2_24;
    }
  }
}

// (w x h) -> (h x w); dst must not alias src.
function transposePlane(src, w, h, dst) {
  const B = 32;
  for (let y0 = 0; y0 < h; y0 += B) {
    const y1 = Math.min(h, y0 + B);
    for (let x0 = 0; x0 < w; x0 += B) {
      const x1 = Math.min(w, x0 + B);
      for (let y = y0; y < y1; y++) {
        for (let x = x0, s = y * w + x0; x < x1; x++, s++) dst[x * h + y] = src[s];
      }
    }
  }
  return dst;
}

/** Pillow ImageFilter.GaussianBlur(radius) on an 8-bit plane. Returns a new plane. */
export function pilGaussianBlurPlane(plane, w, h, radius) {
  if (radius <= 0) return Uint8Array.from(plane);
  return pilBoxBlurPlane(Uint8Array.from(plane), w, h, pilBoxRadius(radius, 3), 3);
}

// TF avg_pool(ksize=k, strides=1, padding='SAME'): mean over the in-bounds part
// of the k x k window (padding is not counted). Separable, on a float plane.
function avgPoolSame(src, w, h, k) {
  const r = (k - 1) >> 1;          // k = 5 -> 2 before, 2 after
  const before = r;
  const after = k - 1 - r;
  const tmp = new Float32Array(w * h);
  const cntX = new Float32Array(w);
  for (let x = 0; x < w; x++) cntX[x] = Math.min(w - 1, x + after) - Math.max(0, x - before) + 1;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let s = 0;
      const x0 = Math.max(0, x - before);
      const x1 = Math.min(w - 1, x + after);
      for (let i = x0; i <= x1; i++) s += src[row + i];
      tmp[row + x] = s;
    }
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - before);
    const y1 = Math.min(h - 1, y + after);
    const cy = y1 - y0 + 1;
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let j = y0; j <= y1; j++) s += tmp[j * w + x];
      out[y * w + x] = s / (cntX[x] * cy);
    }
  }
  return out;
}

// ── Compositor ──────────────────────────────────────────────────────────────

/**
 * Precomputes everything that does not depend on intensity, so render() is
 * cheap enough for a live slider.
 * @param {ImageData} contentImageData  the photo
 * @param {ImageData} stylizedImageData raw stylization, same size as the photo
 */
export function createCompositor(contentImageData, stylizedImageData) {
  assertSameSize(contentImageData, stylizedImageData, 'createCompositor');
  const w = contentImageData.width;
  const h = contentImageData.height;
  const n = w * h;
  const cd = contentImageData.data;
  const sd = stylizedImageData.data;

  // Interleaved float copies in [0, 1]: content C, difference D = S - C, their
  // luminances, and the content's high-frequency detail HF. With these, the
  // blend at intensity a is C + a*D, and its luminance is lumC + a*dLum.
  const C = new Float32Array(n * 3);
  const D = new Float32Array(n * 3);
  const LC = new Float32Array(n);
  const DL = new Float32Array(n);
  const planes = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
  for (let i = 0, j = 0, k = 0; i < n; i++, j += 4, k += 3) {
    const cr = cd[j] / 255, cg = cd[j + 1] / 255, cb = cd[j + 2] / 255;
    const sr = sd[j] / 255, sg = sd[j + 1] / 255, sb = sd[j + 2] / 255;
    C[k] = cr; C[k + 1] = cg; C[k + 2] = cb;
    D[k] = sr - cr; D[k + 1] = sg - cg; D[k + 2] = sb - cb;
    const lc = cr * LUM_R + cg * LUM_G + cb * LUM_B;
    LC[i] = lc;
    DL[i] = sr * LUM_R + sg * LUM_G + sb * LUM_B - lc;
    planes[0][i] = cr; planes[1][i] = cg; planes[2][i] = cb;
  }
  // High-frequency detail layer of the content (alpha independent).
  const HF = new Float32Array(n * 3);
  for (let ch = 0; ch < 3; ch++) {
    const plane = planes[ch];
    const blur = avgPoolSame(plane, w, h, 5);
    for (let i = 0, k = ch; i < n; i++, k += 3) HF[k] = plane[i] - blur[i];
  }

  // Mask cache: blurring a 1200 px mask costs a few ms, so keep the last one.
  let lastMask = null;
  let lastMaskVersion = -1;
  let lastMaskPlane = null;

  function maskPlane(mask) {
    const version = mask.version ?? 0;
    if (mask === lastMask && version === lastMaskVersion) return lastMaskPlane;
    const m = mask.width === w && mask.height === h ? mask : resizeImageData(mask, w, h);
    const md = m.data;
    const l8 = new Uint8Array(n);
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      // Pillow convert('L') (ITU-R 601-2), weighted by alpha so a transparent
      // brush canvas counts as "not painted".
      const L = (md[j] * 19595 + md[j + 1] * 38470 + md[j + 2] * 7471 + 0x8000) >> 16;
      l8[i] = md[j + 3] === 255 ? L : Math.round((L * md[j + 3]) / 255);
    }
    const radius = (SERVER_MASK_RADIUS * Math.max(w, h)) / SERVER_MASK_SIDE;
    const blurred = pilGaussianBlurPlane(l8, w, h, radius);
    const f = new Float32Array(n);
    for (let i = 0; i < n; i++) f[i] = blurred[i] / 255;
    lastMask = mask; lastMaskVersion = version; lastMaskPlane = f;
    return f;
  }

  const rgb8 = [new Uint8Array(n), new Uint8Array(n), new Uint8Array(n)];

  /**
   * @param {{intensity:number, mask?:ImageData|null, sharpen?:boolean}} opts
   *   intensity 0..1; mask white = styled (any size, resized to fit);
   *   sharpen applies the server's final unsharp mask (slower; use false while dragging).
   * @returns {ImageData}
   */
  function render({ intensity = 1, mask = null, sharpen = true } = {}) {
    const a = Math.max(0, Math.min(1, Number(intensity) || 0));
    const keep = 1.0 - a * 0.6;
    const strength = 0.15 + a * 0.30;
    const m = mask ? maskPlane(mask) : null;
    const lumGain = (1 - keep) * a;   // target luminance = lumC + lumGain * dLum
    const out = makeImageData(w, h);
    const od = out.data;
    const [or, og, ob] = rgb8;
    for (let i = 0, k = 0, j = 0; i < n; i++, k += 3, j += 4) {
      const cr = C[k], cg = C[k + 1], cb = C[k + 2];
      // 1. luminance-preserving blend
      const lumB = LC[i] + a * DL[i];
      const ratio = (LC[i] + lumGain * DL[i]) / (lumB + 1e-7);
      let r = (cr + a * D[k]) * ratio;
      let g = (cg + a * D[k + 1]) * ratio;
      let b = (cb + a * D[k + 2]) * ratio;
      r = r < 0 ? 0 : r > 1 ? 1 : r;
      g = g < 0 ? 0 : g > 1 ? 1 : g;
      b = b < 0 ? 0 : b > 1 ? 1 : b;
      // 2. detail re-injection
      r += strength * HF[k]; g += strength * HF[k + 1]; b += strength * HF[k + 2];
      r = r < 0 ? 0 : r > 1 ? 1 : r;
      g = g < 0 ? 0 : g > 1 ? 1 : g;
      b = b < 0 ? 0 : b > 1 ? 1 : b;
      // 3. regional mask
      if (m !== null) {
        const t = m[i];
        r = t * r + (1 - t) * cr;
        g = t * g + (1 - t) * cg;
        b = t * b + (1 - t) * cb;
      }
      // 4a. to 8 bits, truncating like np.clip(x*255).astype(uint8)
      if (sharpen) {
        or[i] = r * 255; og[i] = g * 255; ob[i] = b * 255;
      } else {
        od[j] = (r * 255) | 0; od[j + 1] = (g * 255) | 0; od[j + 2] = (b * 255) | 0; od[j + 3] = 255;
      }
    }
    if (sharpen) unsharpInto(rgb8, w, h, 1.2, Math.trunc(40 + a * 50), 2, od);
    return out;
  }

  return { width: w, height: h, render };
}

// Pillow UnsharpMask: diff = orig - gaussian(orig); where |diff| > threshold,
// out = clip8(orig + diff * percent / 100) in integer maths, else orig.
function unsharpInto(planes, w, h, radius, percent, threshold, od) {
  const n = w * h;
  for (let ch = 0; ch < 3; ch++) {
    const src = planes[ch];
    const blur = pilGaussianBlurPlane(src, w, h, radius);
    for (let i = 0, j = ch; i < n; i++, j += 4) {
      const v = src[i];
      const diff = v - blur[i];
      // Pillow: abs(diff) > threshold, integer maths (C division truncates).
      if (diff > threshold || diff < -threshold) {
        const nv = v + Math.trunc((diff * percent) / 100);
        od[j] = nv <= 0 ? 0 : nv >= 255 ? 255 : nv;
      } else {
        od[j] = v;
      }
    }
  }
  for (let j = 3; j < od.length; j += 4) od[j] = 255;
}

/** Pillow-exact UnsharpMask on an ImageData (exported for tests and reuse). */
export function unsharpMask(imageData, { radius = 1.2, percent = 90, threshold = 2 } = {}) {
  const w = imageData.width, h = imageData.height, n = w * h, d = imageData.data;
  const planes = [new Uint8Array(n), new Uint8Array(n), new Uint8Array(n)];
  for (let i = 0, j = 0; i < n; i++, j += 4) { planes[0][i] = d[j]; planes[1][i] = d[j + 1]; planes[2][i] = d[j + 2]; }
  const out = makeImageData(w, h);
  unsharpInto(planes, w, h, radius, percent, threshold, out.data);
  return out;
}

// ── Palette transfer (port of color_palette_transfer) ───────────────────────

function srgbToLin(v) { return v > 0.04045 ? Math.pow((v + 0.055) / 1.055, 2.4) : v / 12.92; }
function labF(t) { return t > 0.008856451679035631 ? Math.cbrt(t) : t / 0.12841854934601665 + 4 / 29; }
const LIN_LUT = new Float64Array(256);
for (let i = 0; i < 256; i++) LIN_LUT[i] = srgbToLin(i / 255);

// RGB bytes -> planar LAB (Float32), same constants as the server.
function toLab(imageData) {
  const n = imageData.width * imageData.height;
  const d = imageData.data;
  const L = new Float32Array(n), A = new Float32Array(n), B = new Float32Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    const r = LIN_LUT[d[j]], g = LIN_LUT[d[j + 1]], b = LIN_LUT[d[j + 2]];
    const x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047;
    const y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    const z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883;
    const fx = labF(x), fy = labF(y), fz = labF(z);
    L[i] = 116 * fy - 16; A[i] = 500 * (fx - fy); B[i] = 200 * (fy - fz);
  }
  return [L, A, B];
}

function meanStd(plane) {
  let s = 0;
  for (let i = 0; i < plane.length; i++) s += plane[i];
  const mean = s / plane.length;
  let v = 0;
  for (let i = 0; i < plane.length; i++) { const d = plane[i] - mean; v += d * d; }
  return [mean, Math.sqrt(v / plane.length)];
}

function labStats(imageData) {
  // The server measures style statistics on a <= 512 px copy.
  const side = Math.max(imageData.width, imageData.height);
  let img = imageData;
  if (side > 512 && typeof document !== 'undefined') {
    const k = 512 / side;
    img = resizeImageData(imageData, Math.max(1, Math.round(imageData.width * k)), Math.max(1, Math.round(imageData.height * k)));
  }
  return toLab(img).map(meanStd);
}

function labToRgbByte(L, a, b) {
  const fy = (L + 16) / 116, fx = a / 500 + fy, fz = fy - b / 200;
  const d = 6 / 29;
  const x = (fx > d ? fx * fx * fx : 3 * d * d * (fx - 4 / 29)) * 0.95047;
  const y = fy > d ? fy * fy * fy : 3 * d * d * (fy - 4 / 29);
  const z = (fz > d ? fz * fz * fz : 3 * d * d * (fz - 4 / 29)) * 1.08883;
  const lin = [
    x * 3.2404542 + y * -1.5371385 + z * -0.4985314,
    x * -0.9692660 + y * 1.8760108 + z * 0.0415560,
    x * 0.0556434 + y * -0.2040259 + z * 1.0572252,
  ];
  for (let k = 0; k < 3; k++) {
    let v = lin[k] < 0 ? 0 : lin[k] > 1 ? 1 : lin[k];
    v = v > 0.0031308 ? 1.055 * Math.pow(v, 1 / 2.4) - 0.055 : 12.92 * v;
    v *= 255;
    lin[k] = v <= 0 ? 0 : v >= 255 ? 255 : Math.trunc(v);
  }
  return lin;
}

/**
 * Colours-only mode: match the content's LAB mean/std to the style's.
 * @param {ImageData} contentImageData
 * @param {ImageData} styleImageData
 * @param {{strength?:number, style2ImageData?:ImageData|null, mix?:number}} opts
 *   strength 0..1; mix 0..1 = weight of style2 (0 = all style 1).
 * @returns {ImageData} same size as the content
 */
export function paletteTransfer(contentImageData, styleImageData, { strength = 1, style2ImageData = null, mix = 0.5 } = {}) {
  const st = Math.max(0, Math.min(1, Number(strength)));
  const mx = Math.max(0, Math.min(1, Number(mix)));
  const w = contentImageData.width, h = contentImageData.height, n = w * h;
  const lab = toLab(contentImageData);
  const cStats = lab.map(meanStd);
  const sStats = labStats(styleImageData);
  const s2Stats = style2ImageData ? labStats(style2ImageData) : null;
  const params = [0, 1, 2].map((ch) => {
    const cMean = cStats[ch][0], cStd = cStats[ch][1] + 1e-6;
    let sMean = sStats[ch][0], sStd = sStats[ch][1] + 1e-6;
    if (s2Stats) {
      sMean = (1 - mx) * sMean + mx * s2Stats[ch][0];
      sStd = (1 - mx) * sStd + mx * (s2Stats[ch][1] + 1e-6);
    }
    return { cMean, scale: sStd / cStd, sMean };
  });
  const out = makeImageData(w, h);
  const od = out.data;
  const [Lp, Ap, Bp] = lab;
  const [p0, p1, p2] = params;
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    const L = (1 - st) * Lp[i] + st * ((Lp[i] - p0.cMean) * p0.scale + p0.sMean);
    const A = (1 - st) * Ap[i] + st * ((Ap[i] - p1.cMean) * p1.scale + p1.sMean);
    const B = (1 - st) * Bp[i] + st * ((Bp[i] - p2.cMean) * p2.scale + p2.sMean);
    const rgb = labToRgbByte(L, A, B);
    od[j] = rgb[0]; od[j + 1] = rgb[1]; od[j + 2] = rgb[2]; od[j + 3] = 255;
  }
  return out;
}

// ── Resizing ────────────────────────────────────────────────────────────────

function newCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}

/**
 * High-quality resize of an ImageData (or anything drawable: canvas, ImageBitmap,
 * image) to exactly width x height. Large reductions are done in halving steps so
 * every browser averages all source pixels instead of skipping some.
 * @returns {ImageData}
 */
export function resizeImageData(source, width, height) {
  width = Math.max(1, Math.round(width));
  height = Math.max(1, Math.round(height));
  let cur;
  let cw, ch;
  if (source && source.data && typeof source.width === 'number' && !(source.getContext)) {
    cw = source.width; ch = source.height;
    if (cw === width && ch === height) return makeImageData(width, height, new Uint8ClampedArray(source.data));
    cur = newCanvas(cw, ch);
    const img = source instanceof ImageData ? source : new ImageData(new Uint8ClampedArray(source.data), cw, ch);
    cur.getContext('2d').putImageData(img, 0, 0);
  } else {
    cur = source;
    cw = source.width; ch = source.height;
  }
  while (cw / 2 >= width && ch / 2 >= height) {
    const nw = Math.max(width, Math.round(cw / 2));
    const nh = Math.max(height, Math.round(ch / 2));
    const next = newCanvas(nw, nh);
    const ctx = next.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(cur, 0, 0, cw, ch, 0, 0, nw, nh);
    cur = next; cw = nw; ch = nh;
  }
  const out = newCanvas(width, height);
  const ctx = out.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(cur, 0, 0, cw, ch, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
}
