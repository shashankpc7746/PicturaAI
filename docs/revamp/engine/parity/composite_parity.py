"""Parity test: browser compositor (composite.js) vs the server's Python pipeline.

Runs the post-processing of backend/nst_engine.run_nst (phases 3-4) and
color_palette_transfer in Python on fixed inputs, runs composite.js on the same
inputs through Node, and reports per-case mean / max absolute differences
(0-255 scale). Fails if a mean difference exceeds the limit.

Usage (from the repo root):
    venv/Scripts/python.exe docs/revamp/engine/parity/composite_parity.py [--keep DIR]

Needs: the backend venv (TensorFlow, Pillow, NumPy) and Node 18+.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

import nst_engine as ne  # noqa: E402

MEAN_LIMIT = 0.5   # mean abs diff per channel value (0-255) allowed
COMPOSITE_JS = ROOT / "frontend" / "assets" / "js" / "engine" / "composite.js"
NODE_SIDE = Path(__file__).with_name("composite_parity.mjs")


def server_composite(content: np.ndarray, stylized: np.ndarray, alpha: float,
                     mask: np.ndarray | None, mask_radius: float, sharpen: bool) -> np.ndarray:
    """run_nst phases 3-4, line for line, using nst_engine's own helpers."""
    content_hr = tf.constant(content[None].astype(np.float32) / 255.0)
    stylized_tensor = tf.constant(stylized[None].astype(np.float32) / 255.0)
    blended = ne._luminance_preserving_blend(content_hr, stylized_tensor, alpha)
    blended = ne._reinject_details(content_hr, blended, strength=0.15 + alpha * 0.30)
    if mask is not None:
        h, w = content.shape[:2]
        mask_img = Image.fromarray(mask).convert("L").resize((w, h), ne._LANCZOS)
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=mask_radius))
        m = tf.reshape(tf.constant(np.array(mask_img, dtype=np.float32) / 255.0), [1, h, w, 1])
        blended = tf.clip_by_value(m * blended + (1.0 - m) * content_hr, 0.0, 1.0)
    pil = ne.tensor_to_pil(blended)
    if sharpen:
        pil = ne._unsharp_mask(pil, radius=1.2, percent=int(40 + alpha * 50), threshold=2)
    return np.array(pil)


def server_palette(content: np.ndarray, style: np.ndarray, strength: float,
                   style2: np.ndarray | None, mix: float) -> np.ndarray:
    """color_palette_transfer without the JPEG round trip (same maths)."""
    c_lab = ne._rgb_to_lab(content.astype(np.float32))
    s_lab = ne._rgb_to_lab(style.astype(np.float32))
    s2_lab = ne._rgb_to_lab(style2.astype(np.float32)) if style2 is not None else None
    out = np.empty_like(c_lab)
    for ch in range(3):
        c_mean, c_std = c_lab[:, :, ch].mean(), c_lab[:, :, ch].std() + 1e-6
        s_mean, s_std = s_lab[:, :, ch].mean(), s_lab[:, :, ch].std() + 1e-6
        if s2_lab is not None:
            s_mean = (1 - mix) * s_mean + mix * s2_lab[:, :, ch].mean()
            s_std = (1 - mix) * s_std + mix * (s2_lab[:, :, ch].std() + 1e-6)
        t = (c_lab[:, :, ch] - c_mean) * (s_std / c_std) + s_mean
        out[:, :, ch] = (1 - strength) * c_lab[:, :, ch] + strength * t
    return np.clip(ne._lab_to_rgb(out), 0, 255).astype(np.uint8)


def fit(img: Image.Image, side: int) -> Image.Image:
    img = img.convert("RGB")
    k = side / max(img.size)
    return img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), ne._LANCZOS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", help="keep the work directory here")
    ap.add_argument("--photo", default=str(ROOT / "frontend/assets/samples/dog.jpg"))
    ap.add_argument("--style", default=str(ROOT / "images/style_image/starry_night.jpg"))
    ap.add_argument("--style2", default=str(ROOT / "images/style_image/the_scream.jpg"))
    args = ap.parse_args()

    work = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="pictura-parity-"))
    work.mkdir(parents=True, exist_ok=True)
    images: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}

    def put(name: str, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr.astype(np.uint8))
        rgba = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=2) if arr.shape[2] == 3 else arr
        (work / f"{name}.rgba").write_bytes(rgba.tobytes())
        images[name] = {"w": arr.shape[1], "h": arr.shape[0]}
        arrays[name] = arr[:, :, :3]

    photo = Image.open(args.photo)
    style = Image.open(args.style)
    for side in (320, 640, 1200):
        c = fit(photo, side)
        # A stand-in "stylization": the painting stretched to the photo's size.
        # The compositor maths does not care where the pixels came from.
        s = style.convert("RGB").resize(c.size, ne._LANCZOS)
        put(f"content{side}", np.array(c))
        put(f"stylized{side}", np.array(s))
        m = Image.new("L", c.size, 0)
        ImageDraw.Draw(m).ellipse([c.width * 0.2, c.height * 0.15, c.width * 0.7, c.height * 0.9], fill=255)
        put(f"mask{side}", np.stack([np.array(m)] * 3, axis=2))
    put("style512", np.array(fit(style, 512)))
    put("style2_512", np.array(fit(Image.open(args.style2), 512)))

    composite = []
    for side in (320, 640):
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            composite.append({"name": f"comp{side}_a{int(a*100)}", "content": f"content{side}",
                              "stylized": f"stylized{side}", "alpha": a, "sharpen": True})
        composite.append({"name": f"comp{side}_a75_nosharpen", "content": f"content{side}",
                          "stylized": f"stylized{side}", "alpha": 0.75, "sharpen": False})
        composite.append({"name": f"comp{side}_a75_mask", "content": f"content{side}",
                          "stylized": f"stylized{side}", "alpha": 0.75, "sharpen": True,
                          "mask": f"mask{side}"})
    palette = [
        {"name": "pal_full", "content": "content640", "style": "style512", "strength": 1.0, "mix": 0.5},
        {"name": "pal_60", "content": "content640", "style": "style512", "strength": 0.6, "mix": 0.5},
        {"name": "pal_mix30", "content": "content640", "style": "style512", "strength": 1.0,
         "style2": "style2_512", "mix": 0.3},
    ]
    blur = [{"name": f"blur_r{r}", "src": "content640", "radius": r} for r in (1.2, 6, 12)]
    cases = {"images": images, "composite": composite, "palette": palette, "blur": blur,
             "speed": {"key": "content1200|stylized1200"}}
    # make the speed compositor exist
    composite.append({"name": "comp1200_a50", "content": "content1200", "stylized": "stylized1200",
                      "alpha": 0.5, "sharpen": True})
    (work / "cases.json").write_text(json.dumps(cases))

    node = shutil.which("node")
    if not node:
        print("node not found", file=sys.stderr)
        return 2
    subprocess.run([node, str(NODE_SIDE), str(work), str(COMPOSITE_JS)], check=True)

    def js(name: str, w: int, h: int) -> np.ndarray:
        return np.frombuffer((work / f"{name}.js.rgba").read_bytes(), np.uint8).reshape(h, w, 4)[:, :, :3]

    failed = False
    print(f"{'case':28s} {'mean':>7s} {'max':>5s} {'>2':>7s}")
    rows = []
    for c in composite:
        side_img = arrays[c["content"]]
        h, w = side_img.shape[:2]
        side = max(w, h)
        mask = arrays[c["mask"]] if c.get("mask") else None
        ref = server_composite(arrays[c["content"]], arrays[c["stylized"]], c["alpha"], mask,
                               6 * side / 320, c["sharpen"])
        rows.append((c["name"], ref, js(c["name"], w, h)))
    for c in palette:
        h, w = arrays[c["content"]].shape[:2]
        ref = server_palette(arrays[c["content"]], arrays[c["style"]], c["strength"],
                             arrays.get(c.get("style2")) if c.get("style2") else None, c["mix"])
        rows.append((c["name"], ref, js(c["name"], w, h)))
    for c in blur:
        src = Image.fromarray(arrays[c["src"]][:, :, 0])
        ref = np.array(src.filter(ImageFilter.GaussianBlur(radius=c["radius"])))[:, :, None].repeat(3, 2)
        h, w = arrays[c["src"]].shape[:2]
        rows.append((c["name"], ref, js(c["name"], w, h)))
    summary = {}
    for name, ref, got in rows:
        d = np.abs(ref.astype(np.int16) - got.astype(np.int16))
        mean, mx, over2 = float(d.mean()), int(d.max()), float((d > 2).mean())
        summary[name] = {"mean_abs_diff": round(mean, 4), "max_abs_diff": mx, "frac_over_2": round(over2, 5)}
        flag = "  FAIL" if mean > MEAN_LIMIT else ""
        failed |= bool(flag)
        print(f"{name:28s} {mean:7.4f} {mx:5d} {over2:7.4f}{flag}")
    timings = json.loads((work / "timings.json").read_text())
    print("JS timings (Node, ms):", json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in timings.items()}))
    (work / "summary.json").write_text(json.dumps({"diffs": summary, "timings": timings}, indent=1))
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    print("FAILED" if failed else "OK: all mean differences within", MEAN_LIMIT)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
