// "Animate": record the photo turning into the painting (0 to 100% intensity)
// with MediaRecorder from a canvas. MP4 where supported, WebM otherwise.

const TYPES = [
  'video/mp4;codecs=avc1.42E01E',
  'video/mp4',
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
];

export function videoSupport() {
  if (typeof MediaRecorder === 'undefined' || !HTMLCanvasElement.prototype.captureStream) return null;
  const type = TYPES.find((t) => { try { return MediaRecorder.isTypeSupported(t); } catch { return false; } });
  return type ? { type, ext: type.startsWith('video/mp4') ? 'mp4' : 'webm' } : null;
}

const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

/**
 * @param {{ width:number, height:number, frame:(t:number)=>ImageData, durationMs?:number,
 *           holdMs?:number, onProgress?:(f:number)=>void, signal?:AbortSignal }} opts
 *   frame(t) returns the picture at style amount t (0 = photo, 1 = painting).
 * @returns {Promise<{ blob: Blob, ext: string }>}
 */
export async function recordReveal({ width, height, frame, durationMs = 2800, holdMs = 1200, onProgress, signal }) {
  const support = videoSupport();
  if (!support) throw new Error("This browser can't record video.");
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.putImageData(frame(0), 0, 0);
  const stream = canvas.captureStream(30);
  const rec = new MediaRecorder(stream, { mimeType: support.type, videoBitsPerSecond: 6_000_000 });
  const chunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  const stopped = new Promise((resolve) => { rec.onstop = resolve; });
  rec.start(250);

  const lead = 500;
  const total = lead + durationMs + holdMs;
  const t0 = performance.now();
  await new Promise((resolve) => {
    function tick(now) {
      if (signal && signal.aborted) { resolve(); return; }
      const t = now - t0;
      const amount = ease(Math.max(0, Math.min(1, (t - lead) / durationMs)));
      ctx.putImageData(frame(amount), 0, 0);
      if (onProgress) onProgress(Math.min(1, t / total));
      if (t >= total) resolve(); else requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
  rec.stop();
  stream.getTracks().forEach((tr) => tr.stop());
  await stopped;
  if (signal && signal.aborted) throw new DOMException('Recording cancelled.', 'AbortError');
  return { blob: new Blob(chunks, { type: support.type.split(';')[0] }), ext: support.ext };
}
