#!/usr/bin/env python3
"""
Build the static SEO pages — PicturaAI
=======================================
Reads ``data/styles.json`` (the preset catalogue), ``scripts/style_notes.json``
(the longer note for each painting) and ``frontend/assets/samples/credits.json``,
and writes:

* ``frontend/styles/index.html``         the collection, every painting with its wall label
* ``frontend/styles/<slug>/index.html``  one page per painting
* ``frontend/credits.html``              artwork, sample photo, model, software and font credits
* ``frontend/404.html``                  served by StaticFiles for unknown paths under /app/
* the head, header and footer of ``frontend/privacy.html`` and ``frontend/terms.html``;
  their text between the ``page-content`` markers is written by hand and kept as is

The output is deterministic (no timestamps, sorted input), so ``--check`` can tell
whether the committed pages match the data:

    venv/Scripts/python.exe scripts/build_pages.py            # write the pages
    venv/Scripts/python.exe scripts/build_pages.py --check    # exit 1 if any page is stale

Absolute URLs (canonical, Open Graph, JSON-LD) use ``PUBLIC_SITE_URL`` (default
https://picturaai-k7q6.onrender.com) or ``--site-url``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image  # Pillow is already a backend dependency

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
STYLES_JSON = ROOT / "data" / "styles.json"
NOTES_JSON = ROOT / "scripts" / "style_notes.json"
SAMPLES_JSON = FRONTEND / "assets" / "samples" / "credits.json"
DEFAULT_SITE_URL = "https://picturaai-k7q6.onrender.com"
GITHUB_URL = "https://github.com/shashankpc7746/PicturaAI"
PRESETS_DOC_URL = f"{GITHUB_URL}/blob/main/docs/revamp/presets/README.md"
# The fonts are self-hosted (frontend/assets/fonts/README.md; @font-face in tokens.css,
# font-display: swap). These pages do not preload them: on a slow connection the
# preloads took bandwidth from the hero painting (the LCP image), measured +550 ms LCP
# on Slow 4G for a style page, while the fonts arrived only ~0.7 s sooner.
RELATED_COUNT = 4

LICENSE_URLS = {
    "public domain": "https://creativecommons.org/publicdomain/mark/1.0/",
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
}


# ── Data ───────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Style:
    key: str
    slug: str
    title: str
    artist: str
    year: str
    medium: str
    description: str
    tags: Tuple[str, ...]
    image: str
    card: str
    credit: str
    source_url: str
    license: str
    note: Tuple[str, ...]
    meta: str


def load_styles() -> List[Style]:
    data = json.loads(STYLES_JSON.read_text(encoding="utf-8"))
    notes = json.loads(NOTES_JSON.read_text(encoding="utf-8"))
    styles: List[Style] = []
    problems: List[str] = []
    for raw in data["styles"]:
        key = raw["key"]
        entry = notes.get(key)
        if not entry or not entry.get("note") or not entry.get("meta"):
            problems.append(f"{key}: no note/meta in {NOTES_JSON.name}")
            continue
        if len(entry["meta"]) > 160:
            problems.append(f"{key}: meta description is over 160 characters")
        for asset in (raw["image"], raw["card"]):
            if not (FRONTEND / asset).is_file():
                problems.append(f"{key}: missing {asset}")
        styles.append(Style(
            key=key, slug=raw["slug"], title=raw["title"], artist=raw["artist"],
            year=raw["year"], medium=raw["medium"], description=raw["description"],
            tags=tuple(raw.get("tags", [])), image=raw["image"], card=raw["card"],
            credit=raw["credit"], source_url=raw["source_url"], license=raw["license"],
            note=tuple(entry["note"]), meta=entry["meta"],
        ))
    extra = sorted(set(k for k in notes if not k.startswith("_")) - {s.key for s in styles})
    problems.extend(f"{k}: note has no preset in styles.json" for k in extra)
    if problems:
        raise SystemExit("build_pages: " + "; ".join(problems))
    return styles


_size_cache: Dict[str, Tuple[int, int]] = {}


def image_size(asset: str) -> Tuple[int, int]:
    if asset not in _size_cache:
        with Image.open(FRONTEND / asset) as im:
            _size_cache[asset] = im.size
    return _size_cache[asset]


# Smallest horizontal chrome around the hung painting on a one-column layout:
# 2 x 16 px gutter + 2 x 10 px mat padding + 2 x 8 px frame border (pages.css
# .artwork__img clamps). Using the minimum makes ``sizes`` an overestimate, which only
# picks a slightly larger file; layout is unchanged because img has max-width: 100%.
HERO_CHROME_MIN = 68


def hero_srcset(s: Style, w: int) -> str:
    """srcset/sizes for the hung painting, or "" when no sharper copy exists.

    ``<key>-512.jpg`` stays the src (the engine and /api/styles use it). A
    ``<key>-1024.webp`` beside it (long side 1024, from scripts/build_style_assets.py)
    is offered to HiDPI screens so the hero is not upscaled 2x. ``sizes`` never exceeds
    the 512 file's width, so the rendered size is the same as without a srcset.
    """
    large = s.image.rsplit("/", 1)[0] + f"/{s.key}-1024.webp"
    if not (FRONTEND / large).is_file():
        return ""
    lw, _ = image_size(large)
    if lw <= w:
        return ""
    return (f' srcset="{app_url(s.image)} {w}w, {app_url(large)} {lw}w"'
            f' sizes="(max-width: {w + HERO_CHROME_MIN}px) calc(100vw - {HERO_CHROME_MIN}px), {w}px"')


def related_styles(style: Style, styles: List[Style]) -> List[Style]:
    """Paintings that share the most tags, then catalogue order (deterministic)."""
    order = {s.key: i for i, s in enumerate(styles)}
    tags = set(style.tags)
    others = [s for s in styles if s.key != style.key]
    others.sort(key=lambda s: (-len(tags & set(s.tags)), order[s.key]))
    return others[:RELATED_COUNT]


# ── HTML helpers ───────────────────────────────────────────────────────────────

def e(text: str) -> str:
    return escape(text, quote=True)


def app_url(path: str) -> str:
    """A site-relative URL under /app/ for a path relative to frontend/."""
    return "/app/" + path.lstrip("/")


def served_file(url_path: str) -> bool:
    """Whether a /app/... URL maps to a file under frontend/."""
    return url_path.startswith("/app/") and (FRONTEND / url_path[len("/app/"):]).is_file()


def artist_line(s: Style) -> str:
    return f"{s.artist}, {s.year}" if s.year else s.artist


def json_ld(data: object) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False)
    # Keep the data block from ever closing the <script> element.
    text = text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return f'<script type="application/ld+json">\n{text}\n</script>'


def license_url(name: str) -> Optional[str]:
    return LICENSE_URLS.get(name.strip().lower())


def icon(name: str, cls: str = "icon") -> str:
    return (f'<svg class="{cls}" aria-hidden="true">'
            f'<use href="/app/assets/icons/icons.svg#{name}"/></svg>')


NAV = (
    ("studio", "Studio", "/app/#studio"),
    ("styles", "Styles", "/app/styles/"),
    ("gallery", "My gallery", "/app/#gallery"),
    ("how", "How it works", "/app/#how-it-works"),
)
FOOTER = (
    ("credits", "Credits", "/app/credits.html"),
    ("privacy", "Privacy", "/app/privacy.html"),
    ("terms", "Terms", "/app/terms.html"),
)


def nav_links(current: str, indent: str) -> str:
    out = []
    for key, label, href in NAV:
        cur = ' aria-current="page"' if key == current else ""
        out.append(f'{indent}<a href="{href}"{cur}>{label}</a>')
    return "\n".join(out)


def footer_links(current: str) -> str:
    out = []
    for key, label, href in FOOTER:
        cur = ' aria-current="page"' if key == current else ""
        out.append(f'        <li><a href="{href}"{cur}>{label}</a></li>')
    return "\n".join(out)


@dataclass
class Page:
    path: str                 # relative to frontend/, e.g. "styles/klimt-the-kiss/index.html"
    url_path: str             # public path, e.g. "/app/styles/klimt-the-kiss/"
    title: str
    description: str
    body: str                 # the content of <div class="container page__inner">
    current: str = ""         # nav/footer key to mark with aria-current
    og_image: str = "assets/og/og-default.jpg"
    og_image_alt: str = "A photo of a dog turned into a painting, hanging in a gilt frame on a green wall"
    og_type: str = "website"
    ld: Tuple[object, ...] = ()
    extra_css: Tuple[str, ...] = ()
    robots: str = ""
    twitter_card: str = "summary_large_image"
    hand_written: bool = False


def render(page: Page, site: str) -> str:
    canonical = site + page.url_path
    og_w, og_h = image_size(page.og_image)
    css = ["assets/css/tokens.css", "assets/css/base.css", *page.extra_css, "assets/css/pages.css"]
    css_links = "\n".join(f'  <link rel="stylesheet" href="{app_url(c)}">' for c in css)
    ld = "".join("\n  " + json_ld(d).replace("\n", "\n  ") for d in page.ld)
    if page.hand_written:
        note = ("The head, header and footer are generated by scripts/build_pages.py. "
                "Edit only the text between the page-content markers, then run the script.")
    else:
        note = "Generated by scripts/build_pages.py. Edit the data or the script, not this file."
    robots = f'\n  <meta name="robots" content="{e(page.robots)}">' if page.robots else ""
    manifest = ('\n  <link rel="manifest" href="/app/manifest.webmanifest">'
                if served_file("/app/manifest.webmanifest") else "")
    canonical_tag = "" if page.robots else f'\n  <link rel="canonical" href="{e(canonical)}">'
    og_url = "" if page.robots else f'\n  <meta property="og:url" content="{e(canonical)}">'
    return f"""<!doctype html>
<!-- {note} -->
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>{e(page.title)}</title>
  <meta name="description" content="{e(page.description)}">{robots}{canonical_tag}
  <meta name="theme-color" content="#142622">
  <link rel="icon" href="/app/assets/favicon.ico" sizes="32x32">
  <link rel="icon" type="image/svg+xml" href="/app/assets/icons/favicon.svg">
  <link rel="apple-touch-icon" href="/app/assets/apple-touch-icon.png">{manifest}
  <meta property="og:site_name" content="PicturaAI">
  <meta property="og:type" content="{page.og_type}">
  <meta property="og:title" content="{e(page.title)}">
  <meta property="og:description" content="{e(page.description)}">{og_url}
  <meta property="og:image" content="{e(site + app_url(page.og_image))}">
  <meta property="og:image:width" content="{og_w}">
  <meta property="og:image:height" content="{og_h}">
  <meta property="og:image:alt" content="{e(page.og_image_alt)}">
  <meta name="twitter:card" content="{page.twitter_card}">
  <meta name="twitter:title" content="{e(page.title)}">
  <meta name="twitter:description" content="{e(page.description)}">
  <meta name="twitter:image" content="{e(site + app_url(page.og_image))}">
  <meta name="twitter:image:alt" content="{e(page.og_image_alt)}">

{css_links}
  <script type="module" src="/app/assets/js/page.js"></script>
  <link rel="modulepreload" href="/app/assets/js/ui/header.js">{ld}
</head>
<body>
  <a class="skip-link" href="#content">Skip to the content</a>

  <header class="site-header">
    <div class="container site-header__inner">
      <a class="wordmark" href="/app/#studio" aria-label="PicturaAI, open the studio">
        {icon("frame-mark")}
        PicturaAI
      </a>
      <nav class="site-nav" aria-label="Main">
{nav_links(page.current, "        ")}
      </nav>
      <button class="menu-button" type="button" aria-expanded="false" aria-controls="menu-sheet" data-menu-open>
        {icon("menu")}
        Menu
      </button>
    </div>
  </header>

  <div class="menu-sheet" id="menu-sheet" role="dialog" aria-modal="true" aria-label="Menu" hidden>
    <div class="menu-sheet__backdrop" data-menu-close></div>
    <div class="menu-sheet__panel">
      <div class="menu-sheet__top">
        <span class="wordmark">
          {icon("frame-mark")}
          PicturaAI
        </span>
        <button class="btn btn--quiet" type="button" data-menu-close>
          {icon("close")}
          Close
        </button>
      </div>
      <nav class="menu-sheet__nav" aria-label="Main">
{nav_links(page.current, "        ")}
      </nav>
    </div>
  </div>

  <main class="page" id="content" tabindex="-1">
    <div class="container page__inner">
{page.body.rstrip()}
    </div>
  </main>

  <footer class="site-footer">
    <div class="container site-footer__inner">
      <div class="site-footer__brand">
        <span class="wordmark">
          {icon("frame-mark")}
          PicturaAI
        </span>
        <p>Pictura is Latin for a painting.</p>
      </div>
      <ul class="site-footer__links" role="list">
{footer_links(page.current)}
        <li>
          <a href="{GITHUB_URL}" rel="noopener" target="_blank">
            {icon("github", "icon icon--sm")}
            GitHub
          </a>
        </li>
      </ul>
      <p class="site-footer__small">Artworks are reproduced from public-domain sources. Sample photos are CC0.</p>
    </div>
  </footer>
</body>
</html>
"""


def breadcrumb(*items: Tuple[str, Optional[str]]) -> str:
    parts = []
    for label, href in items:
        if href:
            parts.append(f'<a href="{href}">{e(label)}</a>')
        else:
            parts.append(f'<span aria-current="page">{e(label)}</span>')
    sep = ' <span aria-hidden="true">/</span> '
    return f'      <nav class="breadcrumb" aria-label="Breadcrumb">{sep.join(parts)}</nav>\n'


def breadcrumb_ld(site: str, items: Iterable[Tuple[str, str]]) -> dict:
    return {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {"@type": "ListItem", "position": i, "name": name, "item": site + path}
            for i, (name, path) in enumerate(items, start=1)
        ],
    }


def use_href(s: Style) -> str:
    return f"/app/?style={s.key}#studio"


def style_path(s: Style) -> str:
    return f"/app/styles/{s.slug}/"


# ── Style pages ────────────────────────────────────────────────────────────────

NOTE_LOOK_PREFIX = "How it comes out: "


def style_page(s: Style, styles: List[Style], site: str) -> Page:
    w, h = image_size(s.image)
    alt = f"{s.title} by {artist_line(s)}"
    about = [p for p in s.note if not p.startswith(NOTE_LOOK_PREFIX)]
    look = [p[len(NOTE_LOOK_PREFIX):] for p in s.note if p.startswith(NOTE_LOOK_PREFIX)]
    look = [p[:1].upper() + p[1:] for p in look]
    lic_url = license_url(s.license)
    lic = (f'<a href="{e(lic_url)}" rel="license noopener">{e(s.license)}</a>' if lic_url else e(s.license))

    # The notes sit inside .artwork so pages.css can hang them beside the painting on
    # wide screens (under the wall label) and below it on narrower ones.
    prose = ["        <article class=\"prose artwork__notes\">", "          <h2>About the painting</h2>"]
    prose += [f"          <p>{e(p)}</p>" for p in about]
    if look:
        prose.append("          <h2>How it looks on your photo</h2>")
        prose += [f"          <p>{e(p)}</p>" for p in look]
    prose.append("        </article>")

    rel_items = []
    for r in related_styles(s, styles):
        rw, rh = image_size(r.card)
        rel_items.append(
            f'        <li><a class="related__card" href="{style_path(r)}">'
            f'<img src="{app_url(r.card)}" alt="" width="{rw}" height="{rh}" loading="lazy" decoding="async">'
            f'<span class="related__title">{e(r.title)}</span>'
            f'<span class="related__meta">{e(artist_line(r))}</span></a></li>'
        )

    body = (
        breadcrumb(("Studio", "/app/"), ("Styles", "/app/styles/"), (s.title, None))
        + f"""      <section class="artwork" aria-labelledby="artwork-title">
        <figure class="artwork__hang">
          <img class="artwork__img" src="{app_url(s.image)}"{hero_srcset(s, w)} alt="{e(alt)}" width="{w}" height="{h}" fetchpriority="high" decoding="async">
        </figure>
        <div class="artwork__label">
          <h1 class="artwork__title" id="artwork-title">{e(s.title)}</h1>
          <p class="artwork__artist">{e(artist_line(s))}</p>
          <p class="artwork__medium">{e(s.medium)}</p>
          <p class="artwork__desc">{e(s.description)}</p>
          <a class="btn btn--primary" href="{e(use_href(s))}">
            {icon("brush")}
            Use this style
          </a>
          <p class="artwork__credit">{e(s.credit)} <a href="{e(s.source_url)}" rel="noopener">Source on Wikimedia Commons</a>. Licence: {lic}.</p>
        </div>
"""
        + "\n".join(prose) + "\n"
        + """      </section>
      <section class="page__inner" aria-labelledby="related-title">
        <h2 id="related-title">More from the collection</h2>
        <ul class="related" role="list">
"""
        + "\n".join(rel_items) + "\n"
        + """        </ul>
        <p><a class="text-link" href="/app/styles/">See all the paintings</a></p>
      </section>
"""
    )

    canonical = site + style_path(s)
    artwork: dict = {
        "@context": "https://schema.org",
        "@type": "VisualArtwork",
        "name": s.title,
        "creator": {"@type": "Person", "name": s.artist},
        "artMedium": s.medium,
        "description": s.description,
        "image": site + app_url(s.image),
        "url": canonical,
        "creditText": s.credit,
        "isBasedOn": s.source_url,
    }
    if re.fullmatch(r"\d{4}", s.year):
        artwork["dateCreated"] = s.year
    if lic_url:
        artwork["license"] = lic_url
    crumbs = breadcrumb_ld(site, [("Studio", "/app/"), ("Styles", "/app/styles/"), (s.title, style_path(s))])

    return Page(
        path=f"styles/{s.slug}/index.html",
        url_path=style_path(s),
        title=f"{s.title} by {s.artist} as a photo style | PicturaAI",
        description=s.meta,
        body=body,
        current="styles",
        og_image=s.image,
        og_image_alt=alt,
        og_type="article",
        twitter_card="summary",   # paintings are not 2:1; a large card would crop them
        ld=(artwork, crumbs),
    )


# ── The collection ─────────────────────────────────────────────────────────────

def collection_page(styles: List[Style], site: str) -> Page:
    items = []
    for i, s in enumerate(styles):
        w, h = image_size(s.card)
        lazy = "" if i < 4 else ' loading="lazy"'
        items.append(f"""        <li class="piece">
          <div class="piece__hang">
            <img class="piece__img" src="{app_url(s.card)}" alt="{e(s.title)} by {e(s.artist)}" width="{w}" height="{h}"{lazy} decoding="async">
          </div>
          <div class="piece__label">
            <h2 class="piece__title">{e(s.title)}</h2>
            <p class="piece__artist">{e(artist_line(s))}</p>
            <p class="piece__desc">{e(s.description)}</p>
            <div class="piece__actions">
              <a class="btn btn--on-label" href="{e(use_href(s))}">Use this style<span class="visually-hidden">: {e(s.title)}</span></a>
              <a class="piece__link" href="{style_path(s)}">About this painting<span class="visually-hidden">: {e(s.title)}</span></a>
            </div>
          </div>
        </li>""")
    body = (
        breadcrumb(("Studio", "/app/"), ("Styles", None))
        + """      <header class="page__head">
        <h1>The collection</h1>
        <p class="page__lede">Fourteen paintings and prints, from a 1495 Japanese ink landscape to Matisse and Kandinsky. Every one is in the public domain. Pick one to paint your photo in its style, or read about it first.</p>
      </header>
      <ul class="collection" role="list">
"""
        + "\n".join(items) + "\n"
        + """      </ul>
      <p class="muted">Where each image comes from, and why it is in the public domain, is listed on the <a class="text-link" href="/app/credits.html">credits page</a>.</p>
"""
    )
    collection = {
        "@context": "https://schema.org",
        "@type": "CollectionPage",
        "name": "The PicturaAI collection",
        "url": site + "/app/styles/",
        "description": "Public-domain paintings you can use as styles for your photos.",
        "mainEntity": {
            "@type": "ItemList",
            "numberOfItems": len(styles),
            "itemListElement": [
                {"@type": "ListItem", "position": i, "url": site + style_path(s), "name": s.title}
                for i, s in enumerate(styles, start=1)
            ],
        },
    }
    crumbs = breadcrumb_ld(site, [("Studio", "/app/"), ("Styles", "/app/styles/")])
    return Page(
        path="styles/index.html",
        url_path="/app/styles/",
        title="Painting styles for your photos: the collection | PicturaAI",
        description=("Fourteen public-domain paintings to use as photo styles: Van Gogh, Hokusai, "
                     "Klimt, Munch, Seurat, Matisse, Kandinsky and more. Free, no sign-up."),
        body=body,
        current="styles",
        ld=(collection, crumbs),
        extra_css=collection_css(),
    )


def collection_css() -> Tuple[str, ...]:
    """The .collection / .piece wall-label rules live in base.css (shared with the
    studio) or pages.css. Fall back to the studio's app.css only if neither has them:
    app.css also pads the body for the studio's mobile action bar (a gap under the
    footer on a static page)."""
    css_dir = FRONTEND / "assets" / "css"
    for name in ("base.css", "pages.css"):
        if re.search(r"^\.piece\s*\{", (css_dir / name).read_text(encoding="utf-8"), re.M):
            return ()
    return ("assets/css/app.css",)


# ── Credits ────────────────────────────────────────────────────────────────────

# Extra notes from docs/revamp/presets/README.md that belong next to a credit.
CREDIT_NOTES = {
    "the_shipwreck": "Museum photograph by Benoît Prieur, who released it as CC0.",
    "zodiac": "Paper margins cropped. The Commons file name says 1869 by mistake; the work dates from 1896.",
    "improvisation_27": "Mount margins cropped. The Metropolitan Museum of Art releases the image as CC0 Open Access.",
}

SOFTWARE = (
    ("Arbitrary Image Stylization v1-256", "Google Magenta",
     "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2",
     "Apache License 2.0", "/app/models/arbitrary-stylization-v1-256/LICENSE",
     "The style transfer model, used both on the server and, converted to TensorFlow.js, in your browser. "
     "Ghiasi, Lee, Kudlur, Dumoulin and Shlens, “Exploring the structure of a real-time, arbitrary "
     "neural artistic stylization network”, BMVC 2017."),
    ("DeepLab-v3 (MediaPipe Image Segmenter model)", "Google",
     "https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter",
     "Apache License 2.0", "/app/models/deeplab_v3/LICENSE",
     "Finds the subject for “Style the subject” and “Style the background”, on your device."),
    ("TensorFlow.js 4.22.0", "Google", "https://github.com/tensorflow/tfjs",
     "Apache License 2.0", "/app/vendor/tfjs/4.22.0/LICENSE", "Runs the model in the browser."),
    ("MediaPipe Tasks Vision 0.10.35", "Google", "https://github.com/google-ai-edge/mediapipe",
     "Apache License 2.0", None, "Runs the segmentation model. Loaded from cdn.jsdelivr.net."),
    ("TensorFlow", "Google", "https://github.com/tensorflow/tensorflow",
     "Apache License 2.0", None, "Runs the model on the server."),
    ("FastAPI", "Sebastián Ramírez", "https://github.com/fastapi/fastapi",
     "MIT License", None, "The web server."),
    ("Pillow", "Jeffrey A. Clark and contributors", "https://github.com/python-pillow/Pillow",
     "MIT-CMU License", None, "Reads, checks and writes images on the server."),
    ("NumPy", "The NumPy developers", "https://github.com/numpy/numpy",
     "BSD 3-Clause License", None, "Image maths on the server."),
)

FONTS = (
    ("Libre Caslon Display", "Pablo Impallari", "https://fonts.google.com/specimen/Libre+Caslon+Display",
     "/app/assets/fonts/libre-caslon-display/OFL.txt", "Titles and painting names."),
    ("Schibsted Grotesk", "Bakken & Bæck for Schibsted", "https://fonts.google.com/specimen/Schibsted+Grotesk",
     "/app/assets/fonts/schibsted-grotesk/OFL.txt", "Everything else."),
)


def credits_page(styles: List[Style], site: str) -> Page:
    samples = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    art = []
    for s in styles:
        lic_url = license_url(s.license)
        lic = (f'<a href="{e(lic_url)}" rel="license noopener">{e(s.license)}</a>' if lic_url else e(s.license))
        note = f" {e(CREDIT_NOTES[s.key])}" if s.key in CREDIT_NOTES else ""
        art.append(
            f'          <li><a href="{style_path(s)}">{e(s.title)}</a>. {e(s.credit)}{note} '
            f'<a href="{e(s.source_url)}" rel="noopener">Source file<span class="visually-hidden"> for {e(s.title)}</span></a>. '
            f'Licence: {lic}.</li>'
        )
    photos = []
    for p in samples:
        lic_url = license_url(p["license"])
        lic = (f'<a href="{e(lic_url)}" rel="license noopener">{e(p["license"])}</a>' if lic_url else e(p["license"]))
        photos.append(
            f'          <li>{e(p["title"])}. Photo by {e(p["author"])}. '
            f'<a href="{e(p["source_url"])}" rel="noopener">Source file<span class="visually-hidden"> for {e(p["title"])}</span></a>. '
            f'Licence: {lic}.</li>'
        )
    soft = []
    for name, author, url, lic, lic_file, what in SOFTWARE:
        if lic_file and not served_file(lic_file):
            lic_file = None   # e.g. a vendored library that is not in this checkout
        lic_html = f'<a href="{lic_file}">{e(lic)}</a>' if lic_file else e(lic)
        soft.append(
            f'          <li><a href="{e(url)}" rel="noopener">{e(name)}</a>, {e(author)}. {e(what)} Licence: {lic_html}.</li>'
        )
    fonts = []
    for name, author, url, lic_file, what in FONTS:
        if not served_file(lic_file):
            raise SystemExit(f"build_pages: font licence {lic_file} is missing")
        fonts.append(
            f'          <li><a href="{e(url)}" rel="noopener">{e(name)}</a> by {e(author)}. {e(what)} '
            f'Licence: <a href="{e(lic_file)}">SIL Open Font License 1.1</a>. '
            f'Served from our own server; the files come from Google Fonts.</li>'
        )

    body = (
        breadcrumb(("Studio", "/app/"), ("Credits", None))
        + """      <header class="page__head">
        <h1>Credits</h1>
        <p class="page__lede">The paintings, photos, models, software and fonts that PicturaAI is built from, with their sources and licences.</p>
      </header>
      <article class="prose">
        <h2>Paintings</h2>
        <p>All fourteen paintings are in the public domain in both the United States and the European Union: each artist died before 1956, and each work was published or exhibited before 1931. Photographs and scans of them add no new copyright. Every image comes from Wikimedia Commons and was resized and re-encoded for the web. The full reasoning for each work is in the <a href=\"""" + PRESETS_DOC_URL + """\" rel="noopener">presets notes on GitHub</a>.</p>
        <ul>
"""
        + "\n".join(art) + "\n"
        + """        </ul>

        <h2>Sample photos</h2>
        <p>The three photos under “Try a sample” are released under CC0, so they need no credit. We credit them anyway.</p>
        <ul>
"""
        + "\n".join(photos) + "\n"
        + """        </ul>

        <h2>Models and software</h2>
        <ul>
"""
        + "\n".join(soft) + "\n"
        + """        </ul>

        <h2>Fonts</h2>
        <ul>
"""
        + "\n".join(fonts) + "\n"
        + f"""        </ul>

        <h2>PicturaAI itself</h2>
        <p>The PicturaAI code is open source under the MIT License. It is on <a href="{GITHUB_URL}" rel="noopener">GitHub</a>.</p>
        <p>Paintings you upload under “Your painting” are not part of the collection and are not kept. Only use paintings you have the right to use; see the <a href="/app/terms.html">terms</a>.</p>
      </article>
"""
    )
    return Page(
        path="credits.html",
        url_path="/app/credits.html",
        title="Credits: paintings, photos, models and fonts | PicturaAI",
        description=("Sources and licences for every painting, sample photo, model, library and font "
                     "used by PicturaAI. All fourteen paintings are in the public domain."),
        body=body,
        current="credits",
    )


# ── Hand-written pages ─────────────────────────────────────────────────────────
# privacy.html and terms.html: the text between the two markers is written by hand
# and kept as it is; the head, header and footer around it are generated here.

CONTENT_START = "<!-- page-content:start (hand-written; build_pages.py keeps it) -->"
CONTENT_END = "<!-- page-content:end -->"

HAND_PAGES = (
    ("privacy.html", "privacy", "Privacy | PicturaAI",
     "How PicturaAI handles your photos: painted on your device when it can, otherwise on our "
     "server, which keeps no copy. No accounts, no cookies, no tracking."),
    ("terms.html", "terms", "Terms | PicturaAI",
     "The short terms for using PicturaAI: a free service with no warranty. Your photos stay yours; "
     "only upload what you have the right to use."),
)


def hand_page(path: str, current: str, title: str, description: str) -> Page:
    file = FRONTEND / path
    if not file.is_file():
        raise SystemExit(f"build_pages: {path} is missing; it holds hand-written content")
    text = file.read_text(encoding="utf-8")
    start, end = text.find(CONTENT_START), text.find(CONTENT_END)
    if start < 0 or end < start:
        raise SystemExit(f"build_pages: {path} has no page-content markers")
    content = text[start:end + len(CONTENT_END)]
    return Page(path=path, url_path=f"/app/{path}", title=title, description=description,
                body="      " + content + "\n", current=current,
                hand_written=True)


# ── 404 ────────────────────────────────────────────────────────────────────────

def not_found_page(site: str) -> Page:
    body = """      <header class="page__head">
        <h1>This page isn’t on the wall</h1>
        <p class="page__lede">The link may be old, or the address may have a typo.</p>
      </header>
      <div class="prose">
        <p>You can go back to the <a href="/app/">studio</a>, browse <a href="/app/styles/">the collection</a>, or read the <a href="/app/credits.html">credits</a>.</p>
      </div>
"""
    return Page(
        path="404.html",
        url_path="/app/404.html",
        title="Page not found | PicturaAI",
        description="This page does not exist. Go back to the PicturaAI studio.",
        body=body,
        robots="noindex",
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def build(site: str) -> Dict[Path, str]:
    styles = load_styles()
    pages = [collection_page(styles, site), credits_page(styles, site), not_found_page(site)]
    pages += [hand_page(*args) for args in HAND_PAGES]
    pages += [style_page(s, styles, site) for s in styles]
    return {FRONTEND / p.path: render(p, site) for p in pages}


def clean_site_url(raw: str) -> str:
    url = (raw or "").strip().rstrip("/")
    if not re.fullmatch(r"https?://[^/\s]+(/[^\s]*)?", url):
        return DEFAULT_SITE_URL
    return url


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if a generated page is missing or out of date")
    ap.add_argument("--site-url", default=os.environ.get("PUBLIC_SITE_URL", DEFAULT_SITE_URL),
                    help="absolute site URL for canonical/OG/JSON-LD (default: $PUBLIC_SITE_URL or %(default)s)")
    args = ap.parse_args(argv)
    site = clean_site_url(args.site_url)
    outputs = build(site)

    styles_dir = FRONTEND / "styles"
    expected = {p.resolve() for p in outputs}
    stray = sorted(p for p in styles_dir.rglob("*.html") if p.resolve() not in expected) if styles_dir.is_dir() else []

    if args.check:
        stale = [p for p, html in outputs.items()
                 if not p.is_file() or p.read_text(encoding="utf-8") != html]
        for p in stale:
            print(f"out of date: {p.relative_to(ROOT).as_posix()}")
        for p in stray:
            print(f"not generated (remove it): {p.relative_to(ROOT).as_posix()}")
        if stale or stray:
            print("Run: python scripts/build_pages.py")
            return 1
        print(f"{len(outputs)} pages up to date")
        return 0

    written = 0
    for path, html in outputs.items():
        if path.is_file() and path.read_text(encoding="utf-8") == html:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8", newline="\n")
        written += 1
    for p in stray:
        p.unlink()
        print(f"removed {p.relative_to(ROOT).as_posix()}")
    print(f"{written} of {len(outputs)} pages written ({site})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
