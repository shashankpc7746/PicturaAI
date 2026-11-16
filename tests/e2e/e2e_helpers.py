"""Small helpers shared by the e2e tests (locators by role/label, waits, pixels)."""
from __future__ import annotations

import io
import re
from typing import Tuple

from PIL import Image
from playwright.sync_api import Locator, Page, expect

PAINT_TIMEOUT_MS = 120_000

# Viewports used across the suite (width -> height).
VIEWPORTS = {360: 780, 390: 844, 768: 1024, 1024: 768, 1440: 900}


def style_radio(page: Page, title: str) -> Locator:
    """A painting card in the studio gallery, by its title (accessible name starts with it)."""
    return page.get_by_role("radiogroup", name="Paintings").get_by_role(
        "radio", name=re.compile("^" + re.escape(title)))


def mode_radio(page: Page, name: str) -> Locator:
    return page.get_by_role("radio", name=name, exact=True)


def paint_button(page: Page) -> Locator:
    """The visible "Paint it" button (panel button on desktop, bottom bar on phones)."""
    return page.get_by_role("button", name="Paint it", exact=True).filter(visible=True).first


def label_title(page: Page) -> Locator:
    return page.locator("#label-title")


def wait_for_painting(page: Page, timeout: int = PAINT_TIMEOUT_MS) -> None:
    """Wait for the veil to appear and go away, then for a result in the frame."""
    veil = page.locator("#veil")
    try:
        veil.wait_for(state="visible", timeout=5_000)
    except Exception:  # very fast results can skip the visible state
        pass
    veil.wait_for(state="hidden", timeout=timeout)
    expect(page.locator("#frame-result")).to_be_visible()


def paint(page: Page, timeout: int = PAINT_TIMEOUT_MS) -> None:
    btn = paint_button(page)
    expect(btn).to_be_enabled()
    btn.click()
    wait_for_painting(page, timeout)


def result_checksum(page: Page) -> int:
    """A cheap checksum of the result canvas (red channel, every 10th pixel)."""
    return page.evaluate("""() => {
      const c = document.getElementById('frame-result');
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      let s = 0; for (let i = 0; i < d.length; i += 40) s += d[i]; return s; }""")


def png_bytes(size: Tuple[int, int], color=(120, 90, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def overflow(page: Page) -> Tuple[int, int]:
    return tuple(page.evaluate(
        "[document.documentElement.scrollWidth, document.documentElement.clientWidth]"))


def clipped_controls(page: Page) -> list:
    """Visible control groups whose content is wider than the box (clipped text/buttons)."""
    return page.evaluate("""() => Array.from(document.querySelectorAll(
        '.segmented, .plinth, .style-panel, .studio__label, .tool-row, .action-bar__inner, .frame-error__card, .veil__status'))
      .filter(e => e.offsetParent && e.scrollWidth > e.clientWidth + 1)
      .map(e => e.className + ' ' + e.scrollWidth + '>' + e.clientWidth)""")
