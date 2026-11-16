"""Build the web images for every style preset listed in data/styles.json.

For each style it reads images/style_image/<file> and writes into
frontend/assets/styles/:

    <key>-512.jpg    long side 512 px, JPEG q90   (input for the in-browser engine)
    <key>-1024.webp  exactly twice the 512 file's size (the source size if that is
                     smaller), WebP q80: the HiDPI copy of the hero on the style pages
                     (scripts/build_pages.py adds it to the srcset when it exists)
    <key>-480.webp   480 px wide, WebP q80        (gallery card; height capped at 720)
    <key>-160.webp   160 px on the short side, WebP q80  (small picker; crops cleanly
                     to a square with CSS object-fit: cover)

It also checks the styles.json contract (required fields, description length, paths,
aspect ratio) and removes generated files whose key is no longer listed.

The script is idempotent: outputs newer than their source are kept unless --force is
given, and re-running it produces identical files.

    venv/Scripts/python.exe scripts/build_style_assets.py [--force] [--check]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
STYLES_JSON = REPO / "data" / "styles.json"
SOURCE_DIR = REPO / "images" / "style_image"
FRONTEND = REPO / "frontend"
OUT_DIR = FRONTEND / "assets" / "styles"

REQUIRED = ("key", "slug", "title", "artist", "year", "medium", "description", "tags", "file",
            "image", "card", "thumb", "aspect", "credit", "source_url", "license")
GENERATED = re.compile(r"^(?P<key>[a-z0-9_]+)-(512\.jpg|1024\.webp|480\.webp|160\.webp)$")
LANCZOS = Image.Resampling.LANCZOS


def outputs_for(key: str) -> dict[str, str]:
    return {
        "image": f"assets/styles/{key}-512.jpg",
        "card": f"assets/styles/{key}-480.webp",
        "thumb": f"assets/styles/{key}-160.webp",
    }


def validate(data: dict) -> list[str]:
    errors: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    seen_keys: set[str] = set()
    seen_slugs: set[str] = set()
    for i, s in enumerate(data.get("styles", [])):
        where = s.get("key", f"#{i}")
        for field in REQUIRED:
            if field not in s or s[field] in ("", None, []):
                errors.append(f"{where}: missing {field}")
        key = s.get("key", "")
        if not re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", key):
            errors.append(f"{where}: key must be snake_case")
        if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", s.get("slug", "")):
            errors.append(f"{where}: slug must be kebab-case")
        if key in seen_keys or s.get("slug") in seen_slugs:
            errors.append(f"{where}: duplicate key or slug")
        seen_keys.add(key)
        seen_slugs.add(s.get("slug"))
        if len(s.get("description", "")) > 90:
            errors.append(f"{where}: description longer than 90 characters")
        for field, path in outputs_for(key).items():
            if s.get(field) != path:
                errors.append(f"{where}: {field} should be {path!r}")
        src = SOURCE_DIR / s.get("file", "")
        if not src.is_file():
            errors.append(f"{where}: source image {src} not found")
            continue
        with Image.open(src) as im:
            w, h = ImageOps.exif_transpose(im).size
        if abs(round(w / h, 2) - float(s.get("aspect", 0))) > 0.011:
            errors.append(f"{where}: aspect should be {round(w / h, 2)} (image is {w}x{h})")
        if any(t != t.lower() or " " in t for t in s.get("tags", [])):
            errors.append(f"{where}: tags must be lowercase single words")
    return errors


def resized(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    return im.resize(size, LANCZOS) if im.size != size else im.copy()


def hidpi_path(key: str) -> Path:
    return OUT_DIR / f"{key}-1024.webp"


def hidpi_size(source: tuple[int, int], base: tuple[int, int]) -> tuple[int, int]:
    """Twice the 512 file's size, or the source size when the source is smaller."""
    w2, h2 = base[0] * 2, base[1] * 2
    return (w2, h2) if w2 <= source[0] and h2 <= source[1] else source


def build_one(style: dict, force: bool) -> list[Path]:
    src = SOURCE_DIR / style["file"]
    key = style["key"]
    targets = {
        "512": OUT_DIR / f"{key}-512.jpg",
        "480": OUT_DIR / f"{key}-480.webp",
        "160": OUT_DIR / f"{key}-160.webp",
        "1024": hidpi_path(key),
    }
    src_mtime = src.stat().st_mtime
    if not force and all(t.exists() and t.stat().st_mtime >= src_mtime for t in targets.values()):
        return []

    with Image.open(src) as raw:
        im = ImageOps.exif_transpose(raw).convert("RGB")
    w, h = im.size
    written: list[Path] = []

    # 512: long side 512 (never upscale).
    scale = min(1.0, 512 / max(w, h))
    big = resized(im, (max(1, round(w * scale)), max(1, round(h * scale))))
    big.save(targets["512"], "JPEG", quality=90, optimize=True, progressive=True)
    written.append(targets["512"])

    # 1024: the 512 file at 2x (never upscaled), for HiDPI screens on the style pages.
    hidpi = resized(im, hidpi_size((w, h), big.size))
    hidpi.save(targets["1024"], "WEBP", quality=80, method=6)
    written.append(targets["1024"])

    # 480: 480 wide, but keep very tall works at most 720 high.
    scale = min(480 / w, 720 / h, 1.0)
    card = resized(im, (max(1, round(w * scale)), max(1, round(h * scale))))
    card.save(targets["480"], "WEBP", quality=80, method=6)
    written.append(targets["480"])

    # 160: short side 160.
    scale = min(1.0, 160 / min(w, h))
    thumb = resized(im, (max(1, round(w * scale)), max(1, round(h * scale))))
    thumb.save(targets["160"], "WEBP", quality=80, method=6)
    written.append(targets["160"])
    return written


def remove_stale(keys: set[str]) -> list[Path]:
    removed = []
    for p in OUT_DIR.iterdir():
        m = GENERATED.match(p.name)
        if m and m.group("key") not in keys:
            p.unlink()
            removed.append(p)
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild every image")
    ap.add_argument("--check", action="store_true", help="only validate styles.json and existing outputs")
    args = ap.parse_args()

    data = json.loads(STYLES_JSON.read_text(encoding="utf-8"))
    errors = validate(data)
    if errors:
        print("styles.json problems:", *errors, sep="\n  ")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    styles = data["styles"]
    if args.check:
        missing = [str(FRONTEND / s[f]) for s in styles for f in ("image", "card", "thumb")
                   if not (FRONTEND / s[f]).is_file()]
        missing += [str(hidpi_path(s["key"])) for s in styles if not hidpi_path(s["key"]).is_file()]
        if missing:
            print("missing outputs:", *missing, sep="\n  ")
            return 1
        wrong = []
        for s in styles:
            with Image.open(SOURCE_DIR / s["file"]) as src, Image.open(FRONTEND / s["image"]) as base, \
                    Image.open(hidpi_path(s["key"])) as hi:
                want = hidpi_size(ImageOps.exif_transpose(src).size, base.size)
                if hi.size != want or hi.format != "WEBP":
                    wrong.append(f"{hidpi_path(s['key']).name} is {hi.format} {hi.width}x{hi.height}, "
                                 f"expected WEBP {want[0]}x{want[1]}")
        if wrong:
            print("outdated outputs (run without --check):", *wrong, sep="\n  ")
            return 1
        print(f"ok: {len(styles)} styles, all outputs present (with the 1024 px HiDPI copies)")
        return 0

    written = [p for s in styles for p in build_one(s, args.force)]
    removed = remove_stale({s["key"] for s in styles})
    for p in written:
        with Image.open(p) as im:
            print(f"wrote {p.relative_to(REPO).as_posix()}  {im.width}x{im.height}  {p.stat().st_size // 1024} KB")
    for p in removed:
        print(f"removed {p.relative_to(REPO).as_posix()}")
    print(f"{len(styles)} styles, {len(written)} files written, {len(removed)} removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
