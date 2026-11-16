// Node side of the composite parity test (run by composite_parity.py).
// Reads raw RGBA buffers written by the Python side, runs the browser compositor
// from frontend/assets/js/engine/composite.js, and writes raw RGBA results back.
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const [, , workDir, compositePath] = process.argv;
const mod = await import(pathToFileURL(compositePath).href);
const cases = JSON.parse(readFileSync(join(workDir, 'cases.json'), 'utf8'));

function load(name) {
  const meta = cases.images[name];
  const data = new Uint8ClampedArray(readFileSync(join(workDir, `${name}.rgba`)));
  return { data, width: meta.w, height: meta.h };
}
function save(name, img) {
  writeFileSync(join(workDir, `${name}.js.rgba`), Buffer.from(img.data.buffer, img.data.byteOffset, img.data.byteLength));
}

const timings = {};
const compositors = {};
for (const c of cases.composite) {
  const key = `${c.content}|${c.stylized}`;
  if (!compositors[key]) {
    const t0 = performance.now();
    compositors[key] = mod.createCompositor(load(c.content), load(c.stylized));
    timings[`precompute ${c.content}`] = performance.now() - t0;
  }
  const mask = c.mask ? load(c.mask) : null;
  const t0 = performance.now();
  const out = compositors[key].render({ intensity: c.alpha, mask, sharpen: c.sharpen });
  timings[c.name] = performance.now() - t0;
  save(c.name, out);
}
for (const c of cases.palette) {
  const t0 = performance.now();
  const out = mod.paletteTransfer(load(c.content), load(c.style), {
    strength: c.strength, style2ImageData: c.style2 ? load(c.style2) : null, mix: c.mix,
  });
  timings[c.name] = performance.now() - t0;
  save(c.name, out);
}
for (const c of cases.blur) {
  const img = load(c.src);
  const n = img.width * img.height;
  const plane = new Uint8Array(n);
  for (let i = 0; i < n; i++) plane[i] = img.data[i * 4];
  const b = mod.pilGaussianBlurPlane(plane, img.width, img.height, c.radius);
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) { out[i * 4] = out[i * 4 + 1] = out[i * 4 + 2] = b[i]; out[i * 4 + 3] = 255; }
  save(c.name, { data: out });
}
// Live-slider timing at a realistic size: 5 renders without sharpening, 2 with.
if (cases.speed) {
  const comp = compositors[cases.speed.key];
  const t = [];
  for (let i = 0; i < 5; i++) { const t0 = performance.now(); comp.render({ intensity: i / 4, sharpen: false }); t.push(performance.now() - t0); }
  const ts = [];
  for (let i = 0; i < 2; i++) { const t0 = performance.now(); comp.render({ intensity: 0.5 + i / 4, sharpen: true }); ts.push(performance.now() - t0); }
  timings['speed render sharpen:false (ms each)'] = t.map((x) => Math.round(x));
  timings['speed render sharpen:true (ms each)'] = ts.map((x) => Math.round(x));
}
writeFileSync(join(workDir, 'timings.json'), JSON.stringify(timings, null, 1));
