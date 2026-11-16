// Module worker that runs the browser engine off the page's main thread:
// TF.js WebGL on an OffscreenCanvas, so shader compiles and GPU waits never
// block input. Spoken to only by worker-engine.js; messages are
//   in:  {op:'init', store, mobile, options} | {op:'warmup'|'prepare'|'stylize', id, ...} | {op:'abort', id}
//   out: {type:'hello', ok} | {type:'progress'|'result'|'error', id, ...} | {type:'store', key, value}

import { createBrowserEngine } from './browser-engine.js';
import { hasWebGL2 } from './support.js';

let engine = null;
const controllers = new Map();
const send = (msg, transfer = []) => self.postMessage(msg, transfer);

function info() {
  return { maxInferSide: engine.maxInferSide, timings: { ...engine.timings }, ready: engine.ready };
}

function closeBitmaps(m) {
  for (const b of [m.content, m.style && m.style.image, m.style2 && m.style2.image]) {
    if (b && typeof b.close === 'function') b.close();
  }
}

async function run(m, signal, onProgress) {
  if (m.op === 'warmup') await engine.warmup(onProgress);
  else if (m.op === 'prepare') await engine.prepare(m);
  else if (m.op === 'stylize') {
    const { stylized } = await engine.stylize({ content: m.content, style: m.style, style2: m.style2, mix: m.mix, signal, onProgress });
    const { width, height } = stylized;
    const buffer = stylized.data.buffer;
    return [{ ...info(), width, height, buffer }, [buffer]];
  } else throw new Error(`Unknown op ${m.op}`);
  return [info(), []];
}

self.onmessage = async ({ data: m }) => {
  if (m.op === 'abort') {
    const ctl = controllers.get(m.id);
    if (ctl) ctl.abort();
    return;
  }
  if (m.op === 'init') {
    const values = new Map(Object.entries(m.store || {}));
    const store = {
      get: (k) => (values.has(k) ? values.get(k) : null),
      set: (k, v) => { values.set(k, v); send({ type: 'store', key: k, value: v }); },
    };
    const ok = typeof OffscreenCanvas !== 'undefined' && hasWebGL2();
    if (ok) engine = createBrowserEngine({ ...m.options, store, mobile: m.mobile });
    send({ type: 'hello', ok });
    return;
  }
  const ctl = new AbortController();
  controllers.set(m.id, ctl);
  try {
    const [value, transfer] = await run(m, ctl.signal, (p) => send({ id: m.id, type: 'progress', p }));
    send({ id: m.id, type: 'result', value }, transfer);
  } catch (e) {
    send({ id: m.id, type: 'error', error: { name: (e && e.name) || 'Error', message: String((e && e.message) || e), code: (e && e.code) || null } });
  } finally {
    controllers.delete(m.id);
    closeBitmaps(m);
  }
};
