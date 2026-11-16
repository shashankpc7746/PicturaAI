"""Precompute the browser engine's style parameters for every preset style.

The on-device engine (frontend/assets/js/engine/browser-engine.js) runs the same
Magenta arbitrary-image-stylization-v1-256 network as the server, split in two:
a style network (style image -> N style parameters) and a transformer. Shipping
the parameters for the presets means visitors never download the 17 MB style
network unless they upload their own painting.

For each style in data/styles.json this script:
  1. loads images/style_image/<file> exactly like the server does
     (nst_engine.load_and_preprocess at 256x256 + 3x3 average pool),
  2. runs the style half of the server's own SavedModel (float32, on CPU),
  3. writes frontend/models/style-vectors/<key>.bin (float16, little endian)
     and the index frontend/models/style-vectors.json.

The engine works without these files (it then computes the parameters in the
browser), so rerun this whenever the presets or the model change:

    venv/Scripts/python.exe scripts/precompute_style_vectors.py

The graph helpers here (freeze, inline, split) are also used by
docs/revamp/engine/tools/convert_model.py, which produced the TF.js model.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

MODEL_DIR = ROOT / "frontend" / "models" / "arbitrary-stylization-v1-256"
VECTORS_JSON = ROOT / "frontend" / "models" / "style-vectors.json"
VECTORS_DIR = ROOT / "frontend" / "models" / "style-vectors"
BOTTLENECK = "StatefulPartitionedCall/bottleneck/Mean"
PARAM_PREFIX = "StatefulPartitionedCall/style_params/"


# ── Graph helpers (shared with the conversion tool) ─────────────────────────

def inlined_graph(saved_model_dir: str):
    """SavedModel -> one frozen, function-inlined GraphDef.

    Inputs: `placeholder` (content) and `placeholder_1` (style), output `Identity`.
    """
    import tensorflow as tf
    from tensorflow.core.protobuf import config_pb2
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    from tensorflow.python.grappler import tf_optimizer

    model = tf.saved_model.load(saved_model_dir)
    frozen = convert_variables_to_constants_v2(model.signatures["serving_default"])
    gd = frozen.graph.as_graph_def()
    # CheckNumerics (a debug op) blocks function inlining and has no TF.js kernel.
    for nodes in [gd.node] + [f.node_def for f in gd.library.function]:
        for node in nodes:
            if node.op == "CheckNumerics":
                node.op = "Identity"
                if "message" in node.attr:
                    del node.attr["message"]
    graph = tf.Graph()
    with graph.as_default():
        tf.compat.v1.import_graph_def(gd, name="")
        meta = tf.compat.v1.train.export_meta_graph(graph=graph, graph_def=graph.as_graph_def())
    meta.collection_def["train_op"].node_list.value.extend(["Identity"])
    config = config_pb2.ConfigProto()
    rewrite = config.graph_options.rewrite_options
    rewrite.optimizers.extend(["function", "constfold", "pruning", "dependency", "debug_stripper"])
    rewrite.min_graph_nodes = -1
    out = tf_optimizer.OptimizeGraph(config, meta)
    return tf.compat.v1.graph_util.extract_sub_graph(out, ["Identity"])


def param_layout(gd) -> list[dict]:
    """The style-parameter tensors (gamma/beta of every conditional instance norm),
    in a fixed order, with their sizes and offsets in the flat [1, N] vector."""
    from tensorflow.python.framework import tensor_util

    by_name = {n.name: n for n in gd.node}
    squeezes = sorted((n for n in gd.node if n.op == "Squeeze" and n.name.startswith(PARAM_PREFIX)),
                      key=lambda n: n.name)
    layout, offset = [], 0
    for node in squeezes:
        bias = by_name[by_name[node.input[0]].input[1]]
        size = int(tensor_util.MakeNdarray(bias.attr["value"].tensor).shape[0])
        layout.append({"name": node.name[len(PARAM_PREFIX):], "size": size, "offset": offset})
        offset += size
    return layout


def style_graph(gd, layout):
    """Style half: `style_image` [1,256,256,3] -> `style_params` [1, N]."""
    import numpy as np
    import tensorflow as tf
    from tensorflow.core.framework import types_pb2
    from tensorflow.python.framework import tensor_util

    g = copy.deepcopy(gd)
    for node in g.node:
        if node.name == "placeholder_1":
            node.name = "style_image"
        node.input[:] = ["style_image" if i == "placeholder_1" else i for i in node.input]
    axis = g.node.add()
    axis.name, axis.op = "style_params/axis", "Const"
    axis.attr["dtype"].type = types_pb2.DT_INT32
    axis.attr["value"].tensor.CopyFrom(tensor_util.make_tensor_proto(np.array(1, np.int32)))
    cat = g.node.add()
    cat.name, cat.op = "style_params", "ConcatV2"
    cat.input.extend([PARAM_PREFIX + p["name"] for p in layout] + [axis.name])
    cat.attr["N"].i = len(layout)
    cat.attr["T"].type = types_pb2.DT_FLOAT
    cat.attr["Tidx"].type = types_pb2.DT_INT32
    return tf.compat.v1.graph_util.extract_sub_graph(g, ["style_params"])


def transformer_graph(gd, layout):
    """Transformer half: `content_image` [1,H,W,3] + `style_params` [1,N] -> `stylized`."""
    import numpy as np
    import tensorflow as tf
    from tensorflow.core.framework import tensor_shape_pb2, types_pb2
    from tensorflow.python.framework import tensor_util

    n_params = sum(p["size"] for p in layout)
    g = copy.deepcopy(gd)
    for node in g.node:
        if node.name == "placeholder":
            node.name = "content_image"
        node.input[:] = ["content_image" if i == "placeholder" else
                         ("content_image" + i[len("placeholder"):] if i.startswith("placeholder:") else i)
                         for i in node.input]
    ph = g.node.add()
    ph.name, ph.op = "style_params", "Placeholder"
    ph.attr["dtype"].type = types_pb2.DT_FLOAT
    ph.attr["shape"].shape.CopyFrom(tensor_shape_pb2.TensorShapeProto(
        dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=s) for s in (1, n_params)]))
    slots = {PARAM_PREFIX + p["name"]: p for p in layout}

    def const(name, values):
        c = g.node.add()
        c.name, c.op = name, "Const"
        c.attr["dtype"].type = types_pb2.DT_INT32
        c.attr["value"].tensor.CopyFrom(tensor_util.make_tensor_proto(np.array(values, np.int32)))
        return name

    for node in list(g.node):
        p = slots.get(node.name)
        if p is None:
            continue
        begin = const(node.name + "/begin", [0, p["offset"]])
        size = const(node.name + "/size", [1, p["size"]])
        node.op = "Slice"
        del node.input[:]
        node.input.extend(["style_params", begin, size])
        node.ClearField("attr")
        node.attr["T"].type = types_pb2.DT_FLOAT
        node.attr["Index"].type = types_pb2.DT_INT32
    g = tf.compat.v1.graph_util.extract_sub_graph(g, ["Identity"])
    for node in g.node:
        if node.name == "Identity":
            node.name = "stylized"
    return g


def default_saved_model() -> str:
    sys.path.insert(0, str(ROOT / "backend"))
    from model_store import ensure_model  # downloads/caches the server model if needed
    return str(ensure_model())


# ── Precompute ──────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saved-model", help="server SavedModel dir (default: the backend's cached model)")
    ap.add_argument("--styles", default=str(ROOT / "data" / "styles.json"))
    ap.add_argument("--images", default=str(ROOT / "images" / "style_image"))
    args = ap.parse_args()

    styles_path = Path(args.styles)
    if not styles_path.exists():
        print(f"{styles_path} not found; nothing to do (the engine computes vectors in the browser).")
        return 0
    styles = json.loads(styles_path.read_text(encoding="utf-8"))["styles"]
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))

    import numpy as np
    import tensorflow as tf
    sys.path.insert(0, str(ROOT / "backend"))
    from nst_engine import STYLE_IMG_SIZE, load_and_preprocess

    gd = inlined_graph(args.saved_model or default_saved_model())
    layout = param_layout(gd)
    n = sum(p["size"] for p in layout)
    if n != manifest["n"] or [p["name"] for p in layout] != [p["name"] for p in manifest["layout"]]:
        print("The parameter layout does not match frontend/models/.../manifest.json; reconvert the model.")
        return 1
    graph = tf.Graph()
    with graph.as_default():
        tf.compat.v1.import_graph_def(style_graph(gd, layout), name="")
    sess = tf.compat.v1.Session(graph=graph)

    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    index = {"model": manifest["name"], "version": manifest["version"], "n": n, "dtype": "float16",
             "styles": {}}
    keep = set()
    for style in styles:
        key, fname = style["key"], style["file"]
        src = Path(args.images) / fname
        if not src.exists():
            print(f"  skip {key}: {src} missing")
            continue
        t = load_and_preprocess(src.read_bytes(), target_size=(STYLE_IMG_SIZE, STYLE_IMG_SIZE))
        t = tf.nn.avg_pool(t, ksize=[3, 3], strides=[1, 1], padding="SAME").numpy()
        params = sess.run("style_params:0", {"style_image:0": t})[0].astype(np.float32)
        out = VECTORS_DIR / f"{key}.bin"
        out.write_bytes(params.astype("<f2").tobytes())
        keep.add(out.name)
        index["styles"][key] = {"file": f"style-vectors/{key}.bin", "source": fname}
        print(f"  {key}: {len(params)} params, |max| {np.abs(params).max():.3f}")
    for stale in VECTORS_DIR.glob("*.bin"):
        if stale.name not in keep:
            stale.unlink()
    VECTORS_JSON.write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {VECTORS_JSON.relative_to(ROOT)} ({len(index['styles'])} styles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
