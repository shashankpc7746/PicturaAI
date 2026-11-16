"""Short laptop screens (e.g. 1920x1080 at 125 % = 1536x690) and phone toasts.

Regression tests for the height-capped desktop style panel: Paint it, the Mix
toggle and a full row of paintings must stay visible in every tab, keyboard focus
in the gallery must keep the focused card fully in view, and on phones a toast
must not cover the intensity slider under the frame.
"""
from __future__ import annotations

import pytest
from playwright.sync_api import expect

from e2e_helpers import style_radio, wait_for_painting

# Share of an element's height visible inside the style panel and above the stuck
# call to action (100 = fully visible).
VISIBLE_IN_PANEL = """(sel) => {
  const panel = document.getElementById('style-panel').getBoundingClientRect();
  const cta = document.querySelector('.style-panel__cta').getBoundingClientRect();
  const el = document.querySelector(sel);
  if (!el || !el.offsetParent) return 0;
  const r = el.getBoundingClientRect();
  const insideCta = el.closest('.style-panel__cta');
  const top = Math.max(r.top, panel.top);
  const bottom = Math.min(r.bottom, panel.bottom, insideCta ? Infinity : cta.top);
  return Math.round(100 * Math.max(0, bottom - top) / r.height);
}"""

FOCUSED_CARD_VISIBLE = """() => {
  const card = document.activeElement;
  const g = document.querySelector('.tab-panel--gallery').getBoundingClientRect();
  const r = card.getBoundingClientRect();
  return Math.round(100 * Math.max(0, Math.min(r.bottom, g.bottom) - Math.max(r.top, g.top)) / r.height);
}"""


@pytest.mark.parametrize("width,height", [(1536, 690), (1366, 625)])
def test_panel_keeps_paint_and_mix_visible_in_every_tab(open_page, width, height):
    page = open_page(width, height=height).page
    style_radio(page, "The Starry Night").click()
    assert page.evaluate(VISIBLE_IN_PANEL, "#paint-btn") == 100
    assert page.evaluate(VISIBLE_IN_PANEL, "#mix-toggle") == 100

    page.get_by_role("tab", name="Your painting").click()
    assert page.evaluate(VISIBLE_IN_PANEL, "#mix-toggle") == 100

    page.get_by_role("tab", name="Describe it").click()
    page.locator("textarea").fill("starry night swirl")
    page.get_by_role("button", name="Find a style").click()
    expect(page.locator(".describe-result")).to_be_visible()
    page.wait_for_timeout(400)   # the panel scrolls the answer into view
    assert page.evaluate(VISIBLE_IN_PANEL, "#mix-toggle") == 100

    page.get_by_role("tab", name="Gallery").click()
    page.locator("#mix-toggle").click()
    page.wait_for_timeout(400)
    assert page.evaluate(VISIBLE_IN_PANEL, "#mix-ratio") == 100
    assert page.evaluate(VISIBLE_IN_PANEL, "#paint-btn") == 100


def test_gallery_focus_stays_fully_visible_on_a_short_screen(open_page):
    page = open_page(1536, height=690, reduced_motion="reduce").page
    page.locator('.style-card[tabindex="0"]').focus()
    for key in ["ArrowRight"] * 6 + ["ArrowLeft"] * 6:
        page.keyboard.press(key)
        page.wait_for_timeout(60)
        assert page.evaluate(FOCUSED_CARD_VISIBLE) == 100, f"focused card cut off after {key}"


def test_phone_toast_does_not_cover_the_intensity_slider(open_page):
    page = open_page(360, height=640, has_touch=True, is_mobile=True, device_scale_factor=2).page
    style_radio(page, "The Starry Night").tap()
    page.locator("#action-bar").get_by_role("button", name="Paint it").tap()
    wait_for_painting(page)
    save = page.get_by_role("button", name="Save to my gallery")
    save.scroll_into_view_if_needed()
    save.tap()
    expect(page.get_by_text("Saved to my gallery.")).to_be_visible()
    page.locator(".frame-ledge").scroll_into_view_if_needed()
    hit = page.evaluate("""() => {
      const r = document.getElementById('intensity').getBoundingClientRect();
      const e = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      return e && e.id;
    }""")
    assert hit == "intensity", f"something covers the intensity slider: {hit}"
