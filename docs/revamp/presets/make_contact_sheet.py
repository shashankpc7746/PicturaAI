"""Stylise the sample photos with every preset and build a contact sheet.

Rows are styles (the artwork first), columns are the three sample photos. Used to
check that every preset gives a clear, distinct result with the real model.

    venv/Scripts/python.exe docs/revamp/presets/make_contact_sheet.py \
        [--styles-dir DIR] [--keys a,b,c] [--out docs/revamp/presets/contact-sheet.jpg]

Without --styles-dir the presets in data/styles.json are used. Stylised images are
cached in --cache (default: a temp folder) so re-runs only compute new styles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

# Bigger working size than the 320 px free-tier default, for a clearer preview.
os.environ.setdefault("NST_MAX_DIM", "512")
os.environ.setdefault("NST_OUTPUT_DIM", "512")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "backend"))

from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402

SAMPLES = ["dog", "city", "lake"]
CELL = 240
CAPTION = 34
PAD = 6


def styles_from_json() -> list[tuple[str, Path, str]]:
    data = json.loads((REPO / "data/styles.json").read_text(encoding="utf-8"))
    return [
        (s["key"], REPO / "images/style_image" / s["file"], f'{s["title"]}, {s["artist"]} ({s["year"]})')
        for s in data["styles"]
    ]


def styles_from_dir(d: Path, keys: list[str] | None) -> list[tuple[str, Path, str]]:
    files = {p.stem: p for p in sorted(d.glob("*.jpg"))}
    order = keys or list(files)
    return [(k, files[k], k) for k in order]


def stylise_all(styles, cache: Path) -> None:
    import nst_engine  # heavy import: TensorFlow

    cache.mkdir(parents=True, exist_ok=True)
    samples = {s: (REPO / f"frontend/assets/samples/{s}.jpg").read_bytes() for s in SAMPLES}
    for key, path, _ in styles:
        style_bytes = path.read_bytes()
        digest = hashlib.sha1(style_bytes).hexdigest()[:10]
        for s, content in samples.items():
            out = cache / f"{key}-{digest}-{s}.jpg"
            if out.exists():
                continue
            out.write_bytes(nst_engine.run_nst(content, style_bytes, style_weight=1.0))
            print("stylised", key, s, flush=True)


def fit(im: Image.Image, box: int) -> Image.Image:
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((box, box), Image.Resampling.LANCZOS)
    return im


def caption_font(size: int = 13) -> ImageFont.ImageFont:
    """A TrueType font with accents and dashes (Pillow's built-in font lacks them)."""
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def build_sheet(styles, cache: Path, out: Path) -> None:
    font = caption_font()
    cols = 1 + len(SAMPLES)
    row_h = CELL + CAPTION
    sheet = Image.new("RGB", (cols * (CELL + PAD) + PAD, len(styles) * row_h + PAD + 22), (245, 242, 236))
    draw = ImageDraw.Draw(sheet)
    for c, name in enumerate(["artwork"] + SAMPLES):
        draw.text((PAD + c * (CELL + PAD), 4), name, fill=(60, 55, 50), font=font)
    for r, (key, path, label) in enumerate(styles):
        y = 22 + r * row_h
        digest = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
        tiles = [Image.open(path)] + [Image.open(cache / f"{key}-{digest}-{s}.jpg") for s in SAMPLES]
        for c, tile in enumerate(tiles):
            t = fit(tile, CELL)
            x = PAD + c * (CELL + PAD)
            sheet.paste(t, (x + (CELL - t.width) // 2, y + (CELL - t.height) // 2))
        draw.text((PAD, y + CELL + 6), f"{key}  |  {label}", fill=(30, 28, 25), font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "JPEG", quality=82, optimize=True)
    print("wrote", out, sheet.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--styles-dir", type=Path)
    ap.add_argument("--keys", help="comma-separated order/subset (with --styles-dir)")
    ap.add_argument("--out", type=Path, default=REPO / "docs/revamp/presets/contact-sheet.jpg")
    ap.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "pictura-contact-cache")
    args = ap.parse_args()
    keys = args.keys.split(",") if args.keys else None
    styles = styles_from_dir(args.styles_dir, keys) if args.styles_dir else styles_from_json()
    stylise_all(styles, args.cache)
    build_sheet(styles, args.cache, args.out)


if __name__ == "__main__":
    main()
