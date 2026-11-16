"""Phone-width flows: bottom action bar, touch painting of a region, reduced motion."""
from __future__ import annotations

import re

from playwright.sync_api import expect

from e2e_helpers import label_title, mode_radio, overflow, style_radio, wait_for_painting

PHONE = dict(has_touch=True, is_mobile=True, device_scale_factor=2)


def test_phone_bar_paint_and_download(open_page):
    page = open_page(390, reduced_motion="reduce", **PHONE).page
    bar_paint = page.locator("#action-bar").get_by_role("button", name="Paint it")
    expect(bar_paint).to_be_visible()
    expect(page.locator("#bar-reason")).to_have_text("Pick a painting first")

    style_radio(page, "The Great Wave").tap()
    expect(page.locator("#bar-reason")).to_have_text("")
    bar_paint.tap()
    wait_for_painting(page)
    expect(label_title(page)).to_have_text(re.compile(r"^After The Great Wave"))

    # Reduced motion: no brush wipe, only a cross-fade.
    names = page.evaluate("() => document.getAnimations().map(a => a.animationName || '')")
    assert not any("unveil" in n for n in names), names

    # After a result the bar offers Download; changing the style brings Paint it back.
    bar_download = page.locator("#action-bar").get_by_role("button", name="Download")
    expect(bar_download).to_be_visible()
    expect(bar_paint).to_be_hidden()
    bar_download.tap()
    expect(page.get_by_role("menu", name="Download size")).to_be_visible()
    page.keyboard.press("Escape")
    style_radio(page, "The Kiss").tap()
    expect(bar_paint).to_be_visible()

    sw, cw = overflow(page)
    assert sw <= cw, f"page scrolls sideways after a result ({sw} > {cw})"


def test_region_mask_with_touch(open_page):
    w = open_page(390, **PHONE)
    page = w.page
    style_radio(page, "The Kiss").tap()
    mode_radio(page, "Paint a region").tap()
    expect(page.locator("#bar-reason")).to_have_text("Brush over the area to paint first")
    bar_paint = page.locator("#action-bar").get_by_role("button", name="Paint it")
    expect(bar_paint).to_be_disabled()

    page.evaluate("document.querySelector('#frame').scrollIntoView({block: 'center', behavior: 'instant'})")
    page.wait_for_timeout(200)
    scroll_before = page.evaluate("scrollY")
    box = page.locator("#frame-mask").bounding_box()
    assert box, "mask canvas is not shown in region mode"

    cdp = page.context.new_cdp_session(page)

    def touch(kind: str, x: float = 0, y: float = 0) -> None:
        points = [] if kind == "touchEnd" else [{"x": x, "y": y}]
        cdp.send("Input.dispatchTouchEvent", {"type": kind, "touchPoints": points})

    x, y = box["x"] + 30, box["y"] + box["height"] / 2
    touch("touchStart", x, y)
    for i in range(20):
        touch("touchMove", x + i * 8, y + i * 2)
    touch("touchEnd")
    page.wait_for_timeout(400)

    assert page.evaluate("scrollY") == scroll_before, "the page scrolled while painting the mask"
    expect(bar_paint).to_be_enabled()
    expect(page.locator("#bar-reason")).to_have_text("")
