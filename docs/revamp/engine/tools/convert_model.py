"""Convert the server's style-transfer SavedModel into the browser engine's TF.js model.

Produces frontend/models/arbitrary-stylization-v1-256/:
    transformer/  content_image + style_params -> stylized   (uint8 weights)
    style/        style_image (256x256) -> style_params      (uint8 weights)
    manifest.json name, version, parameter layout, file sizes

Run it in a THROWAWAY virtualenv (never the app's venv), because tensorflowjs
pulls in packages the app does not need:

    uv venv %TEMP%/tfjs --python 3.12
    uv pip install --python %TEMP%/tfjs/Scripts/python.exe tensorflow==2.20.0 tensorflow-hub packaging six h5py "setuptools<81"
    uv pip install --python %TEMP%/tfjs/Scripts/python.exe --no-deps tensorflowjs==4.22.0
    %TEMP%/tfjs/Scripts/python.exe docs/revamp/engine/tools/convert_model.py --saved-model .tfhub_cache/<hash>

(tensorflowjs imports tensorflow_decision_forests and jax at module level even
for plain graph conversion; this script stubs both so they need not be installed.)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "frontend" / "models" / "arbitrary-stylization-v1-256"
NAME = "arbitrary-stylization-v1-256"
VERSION = 1


def _stub_optional_imports() -> None:
    import keras  # noqa: F401  (import before the jax stub: keras probes jax)
    for name in ("tensorflow_decision_forests", "jax", "jax.experimental", "jax.experimental.jax2tf"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["jax"].experimental = sys.modules["jax.experimental"]
    sys.modules["jax.experimental"].jax2tf = sys.modules["jax.experimental.jax2tf"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved-model", required=True, help="the server's cached TF Hub SavedModel directory")
    ap.add_argument("--quantize", default="uint8", choices=["uint8", "float16", "none"])
    args = ap.parse_args()

    import tensorflow as tf
    _stub_optional_imports()
    from tensorflowjs.converters import tf_saved_model_conversion_v2 as conv
    import precompute_style_vectors as graphs

    gd = graphs.inlined_graph(args.saved_model)
    layout = graphs.param_layout(gd)
    n = sum(p["size"] for p in layout)
    parts = {"style": (graphs.style_graph(gd, layout), "style_params"),
             "transformer": (graphs.transformer_graph(gd, layout), "stylized")}
    qmap = None if args.quantize == "none" else {args.quantize: True}

    manifest = {"name": NAME, "version": VERSION, "n": n,
                "source": "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2",
                "quantization": args.quantize, "styleInput": 256}
    with tempfile.TemporaryDirectory() as tmp:
        for part, (graph, output) in parts.items():
            pb = Path(tmp) / f"{part}.pb"
            pb.write_bytes(graph.SerializeToString())
            dest = OUT / part
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            conv.convert_tf_frozen_model(str(pb), output, str(dest), quantization_dtype_map=qmap,
                                         skip_op_check=True, strip_debug_ops=True,
                                         weight_shard_size_bytes=4 * 1024 * 1024)
            size = sum(f.stat().st_size for f in dest.iterdir() if f.is_file())
            manifest[part] = {"path": f"{part}/model.json", "bytes": size}
            print(f"{part}: {size / 1e6:.2f} MB")
    manifest["layout"] = layout
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}/manifest.json (N = {n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
