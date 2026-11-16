"""Static assets: self-hosted fonts, module preloads, and the HiDPI hero on style pages."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from playwright.sync_api import expect

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
JS_DIR = FRONTEND / "assets" / "js"
STYLE_PAGE = "/app/styles/van-gogh-starry-night/"

# A static import (not a dynamic import(...)), possibly spread over several lines.
STATIC_IMPORT = re.compile(r"""^\s*import\s+(?:[^'";]*?\s+from\s+)?['"](\.{1,2}/[^'"]+)['"]""", re.M | re.S)
FONT_HOSTS = ("fonts.googleapis.com", "fonts.gstatic.com")


def _static_graph(entry: Path) -> set:
    seen: set = set()
    todo = [entry]
    while todo:
        path = todo.pop()
        for spec in STATIC_IMPORT.findall(path.read_text(encoding="utf-8")):
            dep = (path.parent / spec).resolve()
            if dep not in seen:
                seen.add(dep)
                todo.append(dep)
    return seen


def test_modulepreloads_match_the_static_import_graph():
    """index.html preloads exactly the modules main.js loads at startup (no engine modules)."""
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    preloads = {(FRONTEND / href).resolve()
                for href in re.findall(r'<link rel="modulepreload" href="([^"]+)"', html)}
    graph = _static_graph(JS_DIR / "main.js")
    assert graph, "no static imports found in main.js"
    assert not any("engine" in p.parts for p in graph), "an engine module is imported statically"
    missing = sorted(p.relative_to(FRONTEND).as_posix() for p in graph - preloads)
    extra = sorted(p.relative_to(FRONTEND).as_posix() for p in preloads - graph)
    assert not missing and not extra, f"modulepreload out of step: missing {missing}, not imported {extra}"


@pytest.mark.parametrize("path", ["/app/?engine=server", STYLE_PAGE, "/app/privacy.html"])
def test_fonts_are_self_hosted(open_page, path):
    requests: list = []
    w = open_page(1440, path=path, wait_for_styles=False,
                  setup=lambda p: p.on("request", lambda r: requests.append(r.url)))
    page = w.page
    page.evaluate("document.fonts.ready.then(() => true)")
    assert not [u for u in requests if any(h in u for h in FONT_HOSTS)], "a page still asks Google Fonts"
    fonts = [u for u in requests if u.endswith(".woff2")]
    assert fonts and all("/app/assets/fonts/" in u for u in fonts), fonts
    loaded = page.evaluate("""() => [...document.fonts].filter((f) => f.status === 'loaded')
        .map((f) => f.family.replace(/["']/g, ''))""")
    assert "Schibsted Grotesk" in loaded and "Libre Caslon Display" in loaded, loaded
    assert not page.locator('link[href*="fonts.googleapis.com"], link[href*="fonts.gstatic.com"]').count()


@pytest.mark.parametrize("dpr,suffix", [(1, "starry_night-512.jpg"), (2, "starry_night-1024.webp")])
def test_style_hero_picks_the_right_file_for_the_screen(open_page, dpr, suffix):
    w = open_page(1440, path=STYLE_PAGE, wait_for_styles=False, device_scale_factor=dpr)
    hero = w.page.locator(".artwork__img")
    expect(hero).to_be_visible()
    w.page.wait_for_function("() => document.querySelector('.artwork__img').complete")
    src = hero.evaluate("(img) => img.currentSrc")
    assert src.endswith(suffix), f"DPR {dpr} loaded {src}"
    # The rendered size does not depend on which file was picked.
    width = hero.evaluate("(img) => img.getBoundingClientRect().width")
    assert 500 < width < 620, width
