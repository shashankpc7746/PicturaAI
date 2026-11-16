// Banded convolutions for the TF.js WebGL backend (docs/revamp/engine/ENGINE.md §3).
// TF.js lowers each convolution to one im2col matrix + matmul; at 768-1024 px the
// model's large 'VALID' convolutions exceed GPU texture limits, and one huge draw
// can trip the OS GPU watchdog (lost context). When the im2col matrix would be
// large, Conv2D / FusedConv2D are computed in horizontal bands instead (slice the
// rows a band needs, convolve, concatenate): bit-identical for VALID padding.

const DEFAULT_BAND_ELEMENTS = 1 << 23; // im2col elements per band (32 MB as float32)

let installed = false;

/**
 * Install the banded convolution wrappers on the 'webgl' backend. Idempotent.
 * @param {object} tf  the TF.js namespace (tfjs-core with the webgl backend registered)
 * @param {{maxBandElements?: number}} [opts]
 */
export function installBandedConv(tf, opts = {}) {
  if (installed) return;
  const limit = opts.maxBandElements || DEFAULT_BAND_ELEMENTS;
  const slice = tf.getKernel('Slice', 'webgl');
  const concat = tf.getKernel('Concat', 'webgl');
  if (!slice || !concat) return;
  for (const name of ['Conv2D', 'FusedConv2D']) {
    const orig = tf.getKernel(name, 'webgl');
    if (!orig) continue;
    tf.unregisterKernel(name, 'webgl');
    tf.registerKernel({
      kernelName: name,
      backendName: 'webgl',
      setupFunc: orig.setupFunc,
      disposeFunc: orig.disposeFunc,
      kernelFunc: (args) => bandedConv(args, orig.kernelFunc, slice.kernelFunc, concat.kernelFunc, limit),
    });
  }
  installed = true;
}

function pair(v) {
  return Array.isArray(v) ? [v[0], v.length > 1 ? v[1] : v[0]] : [v, v];
}

function bandedConv(args, convFn, sliceFn, concatFn, limit) {
  const { inputs, backend, attrs } = args;
  const { x, filter } = inputs;
  const pad = attrs.pad;
  const fmt = attrs.dataFormat || 'NHWC';
  if (pad !== 'valid' || fmt !== 'NHWC' || x.shape.length !== 4 || x.shape[0] !== 1) {
    return convFn(args);
  }
  const [, inH, inW, inC] = x.shape;
  const [kh, kw] = filter.shape;
  const [sh, sw] = pair(attrs.strides ?? 1);
  const [dh, dw] = pair(attrs.dilations ?? 1);
  const effKh = (kh - 1) * dh + 1;
  const effKw = (kw - 1) * dw + 1;
  const outH = Math.floor((inH - effKh) / sh) + 1;
  const outW = Math.floor((inW - effKw) / sw) + 1;
  if (outH <= 0 || outW <= 0) return convFn(args);
  const perRow = outW * kh * kw * inC;          // im2col elements per output row
  if (perRow * outH <= limit) return convFn(args);

  const rowsPerBand = Math.max(1, Math.floor(limit / perRow));
  const parts = [];
  const gl = backend.gpgpu && backend.gpgpu.gl;
  try {
    for (let r0 = 0; r0 < outH; r0 += rowsPerBand) {
      const r1 = Math.min(outH, r0 + rowsPerBand);
      const inStart = r0 * sh;
      const inRows = (r1 - 1) * sh + effKh - inStart;
      const xs = sliceFn({ inputs: { x }, backend, attrs: { begin: [0, inStart, 0, 0], size: [1, inRows, inW, inC] } });
      try {
        parts.push(convFn({ inputs: { ...inputs, x: xs }, backend, attrs }));
      } finally {
        backend.disposeIntermediateTensorInfo(xs);
      }
      // Hand each band to the GPU so no single submission runs for seconds.
      if (gl) gl.flush();
    }
    if (parts.length === 1) return parts[0];
    return concatFn({ inputs: parts, backend, attrs: { axis: 1 } });
  } finally {
    if (parts.length > 1) for (const p of parts) backend.disposeIntermediateTensorInfo(p);
  }
}
