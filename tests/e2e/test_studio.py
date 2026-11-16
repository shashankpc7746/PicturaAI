"""Studio flows on the server engine (?engine=server), desktop width."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from PIL import Image
from playwright.sync_api import Page, expect

from e2e_helpers import (
    label_title, mode_radio, paint, paint_button, png_bytes, result_checksum, style_radio,
)

ROOT = Path(__file__).resolve().parents[2]
CUSTOM_STYLE = ROOT / "images" / "style_image" / "the_kiss.jpg"


def _pick(page: Page, title: str) -> None:
    style_radio(page, title).click()
    expect(style_radio(page, title)).to_have_attribute("aria-checked", "true")


def _api_requests(page: Page) -> list:
    seen: list = []
    page.on("request", lambda r: seen.append(r.url) if "/api/" in r.url else None)
    return seen


def test_keyboard_pick_paint_and_result_controls(open_page, tmp_path):
    page = open_page(1440).page
    api = _api_requests(page)

    # Arrow keys move the selection in the painting radio group (roving tabindex).
    first = style_radio(page, "The Starry Night")
    first.focus()
    page.keyboard.press("ArrowRight")
    second = style_radio(page, "The Scream")
    expect(second).to_be_focused()
    expect(second).to_have_attribute("aria-checked", "true")
    page.keyboard.press("ArrowLeft")
    expect(first).to_have_attribute("aria-checked", "true")
    expect(page.locator("#selected-style")).to_contain_text("The Starry Night")
    expect(page.locator("#paint-reason")).to_have_text("")

    # Paint it: the result appears in the frame with a wall label.
    paint(page)
    expect(label_title(page)).to_have_text(re.compile(r"^After The Starry Night"))
    expect(page.locator("#label-line2")).to_have_text("Intensity 75%")
    assert any("/api/transfer" in u for u in api)

    # Compare slider on the frame: arrow keys move it by 5%.
    compare = page.get_by_role("slider", name="Compare with your photo")
    compare.focus()
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowRight")
    expect(compare).to_have_attribute("aria-valuenow", "10")
    page.keyboard.press("End")
    expect(compare).to_have_attribute("aria-valuenow", "100")
    page.keyboard.press("Home")
    expect(compare).to_have_attribute("aria-valuenow", "0")

    # Live intensity: the frame changes without a new request.
    n_before = len(api)
    intensity = page.get_by_label("Intensity")
    intensity.fill("0")
    intensity.dispatch_event("change")
    page.wait_for_timeout(250)
    at_0 = result_checksum(page)
    intensity.fill("100")
    intensity.dispatch_event("change")
    page.wait_for_timeout(250)
    at_100 = result_checksum(page)
    assert at_0 != at_100, "moving the intensity slider did not change the picture"
    assert len(api) == n_before, f"intensity made a request: {api[n_before:]}"
    expect(page.locator("#label-line2")).to_have_text("Intensity 100%")
    intensity.fill("75")
    intensity.dispatch_event("change")

    # Download menu: sizes shown, Esc closes and returns focus, files have the right size.
    download = page.get_by_role("button", name="Download", exact=True).filter(visible=True).first
    download.click()
    menu = page.get_by_role("menu", name="Download size")
    expect(menu).to_be_visible()
    expect(menu.get_by_role("menuitem", name=re.compile("Instagram"))).to_contain_text("1080 ×")
    expect(menu.get_by_role("menuitem", name=re.compile("Print"))).to_contain_text("×")
    page.keyboard.press("Escape")
    expect(menu).to_be_hidden()
    expect(download).to_be_focused()

    for item, check in (("Instagram", lambda s: s[0] == 1080),
                        ("Print", lambda s: max(s) == 1754),
                        ("Full size", lambda s: min(s) > 0)):
        download.click()
        with page.expect_download() as dl:
            menu.get_by_role("menuitem", name=re.compile(item)).click()
        path = tmp_path / dl.value.suggested_filename
        dl.value.save_as(path)
        assert path.suffix in (".jpg", ".jpeg"), path.name
        with Image.open(path) as im:
            assert check(im.size), f"{item} download has size {im.size}"


def test_colours_only(open_page):
    page = open_page(1440).page
    api = _api_requests(page)
    _pick(page, "The Great Wave")
    mode_radio(page, "Colours only").click()
    expect(mode_radio(page, "Colours only")).to_have_attribute("aria-checked", "true")
    paint(page)
    expect(label_title(page)).to_have_text(re.compile(r"^In the colours of The Great Wave"))
    expect(page.locator("#label-line2")).to_contain_text("Colour strength")
    assert not any("/api/transfer" in u for u in api), "colours only should run in the browser"


def test_mix_two_styles(open_page):
    page = open_page(1440).page
    _pick(page, "The Starry Night")
    page.get_by_role("button", name="Mix with a second style").click()
    page.get_by_role("radiogroup", name="Second painting").get_by_role(
        "radio", name=re.compile("^The Scream")).click()
    expect(page.locator("#selected-style")).to_contain_text("mixed with The Scream")
    paint(page)
    expect(page.locator("#label-line1")).to_contain_text("mixed with The Scream")


def test_region_mask_with_mouse(open_page):
    page = open_page(1440).page
    _pick(page, "The Starry Night")
    paint(page)
    mode_radio(page, "Paint a region").click()
    expect(page.get_by_role("button", name="Style the subject")).to_be_visible()
    expect(paint_button(page)).to_be_disabled()
    expect(page.locator("#paint-reason")).to_have_text("Brush over the area to paint first")

    page.locator("#frame").scroll_into_view_if_needed()
    box = page.locator("#frame-mask").bounding_box()
    assert box, "mask canvas is not shown in region mode"
    y = box["y"] + box["height"] * 0.5
    page.mouse.move(box["x"] + box["width"] * 0.1, y)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.3, y, steps=8)
    page.mouse.up()
    page.wait_for_timeout(300)

    # Brushed area is styled; an unbrushed corner matches the photo.
    diff = _brushed_vs_corner(page)
    assert diff[0] > 25 and diff[1] < 25, f"brushed vs unbrushed difference: {diff}"
    expect(paint_button(page)).to_be_enabled()
    expect(page.locator("#label-line2")).to_contain_text("brushed areas only")

    # Invert swaps the mask: the brushed stroke now shows the photo, the corner is styled.
    mask_before, result_before = _mask_alpha_sum(page), result_checksum(page)
    assert mask_before > 0, "the brush stroke is not shown on the mask overlay"
    page.get_by_role("button", name="Invert").click()
    page.wait_for_function(
        "(before) => { const c = document.getElementById('frame-result'); const d = c.getContext('2d')"
        ".getImageData(0, 0, c.width, c.height).data; let s = 0; for (let i = 0; i < d.length; i += 40) s += d[i];"
        " return s !== before; }", arg=result_before, timeout=5_000)
    page.wait_for_function(f"(before) => ({MASK_ALPHA_JS})() !== before", arg=mask_before, timeout=5_000)
    inverted = _brushed_vs_corner(page)
    assert inverted[0] < 25 and inverted[1] > 25, f"after Invert, brushed vs unbrushed difference: {inverted}"
    expect(paint_button(page)).to_be_enabled()

    page.get_by_role("button", name="Clear", exact=True).click()
    expect(paint_button(page)).to_be_disabled()
    page.wait_for_function(f"() => ({MASK_ALPHA_JS})() === 0", timeout=5_000)


def _brushed_vs_corner(page: Page) -> list:
    """Result vs photo colour difference at the brushed stroke (20%, 50%) and a corner (85%, 15%)."""
    return page.evaluate("""() => {
      const c = document.getElementById('frame-result'); const ctx = c.getContext('2d');
      const p = document.getElementById('frame-photo'); const t = document.createElement('canvas');
      t.width = c.width; t.height = c.height; const tc = t.getContext('2d'); tc.drawImage(p, 0, 0, c.width, c.height);
      const at = (x, y) => { const a = ctx.getImageData(x, y, 1, 1).data, b = tc.getImageData(x, y, 1, 1).data;
        return Math.abs(a[0]-b[0]) + Math.abs(a[1]-b[1]) + Math.abs(a[2]-b[2]); };
      return [at(Math.round(c.width*0.2), Math.round(c.height*0.5)), at(Math.round(c.width*0.85), Math.round(c.height*0.15))]; }""")


# Sum of the alpha channel of the visible mask overlay (#frame-mask, redrawn on the next frame).
MASK_ALPHA_JS = """() => {
  const c = document.getElementById('frame-mask');
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let s = 0; for (let i = 3; i < d.length; i += 4) s += d[i]; return s; }"""


def _mask_alpha_sum(page: Page) -> int:
    return page.evaluate(MASK_ALPHA_JS)


def test_my_gallery_save_delete_undo_open(open_page):
    page = open_page(1440).page
    _pick(page, "The Starry Night")
    paint(page)
    save = page.get_by_role("button", name=re.compile("Save to my gallery"))
    save.click()
    expect(page.locator("#save-btn")).to_have_text(re.compile(r"^\s*Saved\s*$"))
    expect(page.locator("#save-btn")).to_be_disabled()

    gallery = page.locator("#my-gallery")
    expect(gallery.get_by_role("listitem")).to_have_count(1)
    expect(page.get_by_text("Paintings you save appear here.")).to_be_hidden()

    page.get_by_role("button", name=re.compile(r"^Delete After The Starry Night")).click()
    expect(gallery.get_by_role("listitem")).to_have_count(0)
    expect(page.get_by_text("Paintings you save appear here.")).to_be_visible()

    page.locator("#toasts").get_by_role("button", name="Undo").click()
    expect(gallery.get_by_role("listitem")).to_have_count(1)

    # Start fresh, then open the saved painting back into the frame.
    page.get_by_role("button", name="City", exact=True).click()
    expect(label_title(page)).to_have_text("Sample photo")
    page.get_by_role("button", name=re.compile(r"^Open After The Starry Night")).click()
    expect(page.locator("#frame-result")).to_be_visible()
    expect(label_title(page)).to_have_text(re.compile(r"^After The Starry Night"))
    expect(page.locator("#save-btn")).to_be_disabled()


def test_describe_it_matches_a_preset(open_page):
    page = open_page(1440).page
    page.get_by_role("tab", name="Describe it").click()
    expect(page.get_by_role("tab", name="Describe it")).to_have_attribute("aria-selected", "true")
    box = page.get_by_label(re.compile("^Describe a style"))
    box.fill("starry night sky with swirls")
    # Without Cloudflare credentials the server only matches words to a preset.
    submit = page.locator("#describe-form").get_by_role("button")
    expect(submit).to_have_text("Find a style")
    with page.expect_response("**/api/describe-style") as resp:
        submit.click()
    assert resp.value.status == 200
    assert resp.value.json()["source"] == "matched"
    expect(page.locator("#describe-title")).to_have_text(re.compile(r"^Closest classic style: The Starry Night"))
    expect(page.locator("#selected-style")).to_contain_text("The Starry Night")
    expect(paint_button(page)).to_be_enabled()


def test_describe_it_falls_back_when_endpoint_is_down(open_page):
    w = open_page(1440)
    w.allow(r"status of 502")
    page = w.page
    page.route("**/api/describe-style", lambda r: r.fulfill(
        status=502, content_type="text/html", body="<html>bad gateway</html>"))
    page.get_by_role("tab", name="Describe it").click()
    page.get_by_label(re.compile("^Describe a style")).fill("great wave on the sea")
    page.locator("#describe-form").get_by_role("button").click()
    expect(page.locator("#describe-title")).to_have_text(re.compile(r"^Closest classic style: The Great Wave"))
    expect(page.locator("#describe-text")).to_contain_text("isn't available")


def test_server_error_shows_message_in_frame(open_page):
    """A sleeping or overloaded server answers 503 with an HTML page."""
    w = open_page(1440)
    w.allow(r"status of 503")
    page = w.page
    page.route("**/api/transfer", lambda r: r.fulfill(
        status=503, content_type="text/html", body="<html><body>Service Unavailable</body></html>"))
    _pick(page, "The Scream")
    paint_button(page).click()
    card = page.locator("#frame-error")
    # The alert is the message alone (no button names read with it).
    alert = card.get_by_role("alert")
    expect(alert).to_be_visible(timeout=30_000)
    text = alert.inner_text()
    assert text.strip() and "<html" not in text.lower(), f"error message: {text!r}"
    expect(alert.get_by_role("button")).to_have_count(0)
    expect(card.get_by_role("button", name="Try again")).to_be_visible()
    expect(card.get_by_role("button", name="Try again")).to_be_focused()
    card.get_by_role("button", name="Dismiss").click()
    expect(card).to_be_hidden()
    expect(page.locator("#frame-error-text")).to_have_text("")
    expect(label_title(page)).to_have_text("Sample photo")

    # Try again after the server recovers.
    page.unroute("**/api/transfer")
    paint(page)
    expect(label_title(page)).to_have_text(re.compile(r"^After The Scream"))


def test_cancel_painting(open_page):
    page = open_page(1440).page
    held = []
    page.route("**/api/transfer", lambda r: held.append(r))   # never answered until cancel
    _pick(page, "Udnie")
    paint_button(page).click()
    expect(page.locator("#veil")).to_be_visible()
    page.wait_for_timeout(300)
    # Measured in the page: from the Cancel click to the veil being hidden.
    page.evaluate("""() => {
      window.__cancelMs = null;
      const veil = document.getElementById('veil');
      document.getElementById('cancel-paint').addEventListener('click', () => {
        const t0 = performance.now();
        const check = () => { if (veil.hidden || !veil.offsetParent) window.__cancelMs = performance.now() - t0;
                              else requestAnimationFrame(check); };
        check();
      }, { once: true, capture: true });
    }""")
    page.get_by_role("button", name="Cancel").click()
    expect(page.locator("#veil")).to_be_hidden(timeout=10_000)
    page.wait_for_function("window.__cancelMs !== null", timeout=5_000)
    cancel_ms = page.evaluate("window.__cancelMs")
    assert cancel_ms < 1000, f"Cancel took {cancel_ms:.0f} ms to hide the veil"
    expect(page.locator("#toasts")).to_contain_text("Painting cancelled")
    expect(page.locator("#frame-result")).to_be_hidden()
    for r in held:
        try:
            r.abort()
        except Exception:
            pass
    page.unroute("**/api/transfer")


def test_upload_validation_and_custom_painting(open_page):
    page = open_page(1440).page
    photo_input = page.locator("#photo-input")

    photo_input.set_input_files({"name": "notes.pdf", "mimeType": "application/pdf",
                                 "buffer": b"%PDF-1.4 not an image"})
    card = page.locator("#frame-error")
    alert = card.get_by_role("alert")
    expect(alert).to_be_visible()
    expect(alert).to_contain_text("isn't supported")
    expect(card.get_by_role("button", name="Choose another photo")).to_be_visible()
    card.get_by_role("button", name="Dismiss").click()

    photo_input.set_input_files({"name": "me.png", "mimeType": "image/png", "buffer": png_bytes((900, 1200))})
    expect(label_title(page)).to_have_text("Your photo")
    expect(page.locator("#label-line2")).to_contain_text("900 × 1200")
    ar = page.evaluate("parseFloat(getComputedStyle(document.getElementById('studio')).getPropertyValue('--ar'))")
    assert abs(ar - 0.75) < 0.01, f"frame aspect {ar}"

    page.get_by_role("tab", name="Your painting").click()
    page.locator("#style-input").set_input_files(str(CUSTOM_STYLE))
    expect(page.locator("#custom-preview")).to_be_visible()
    expect(page.locator("#selected-style")).to_contain_text("Your painting")
    paint(page)
    expect(label_title(page)).to_have_text("After your painting")


def test_animate_downloads_a_video(open_page):
    page = open_page(1440).page
    animate = page.get_by_role("button", name="Animate")
    supported = page.evaluate(
        "import('/app/assets/js/services/video.js').then((m) => !!m.videoSupport())")
    if not supported:
        pytest.skip("this Chromium has no MediaRecorder video type, so Animate is hidden by design")
    _pick(page, "The Starry Night")
    paint(page)
    expect(animate).to_be_visible()
    with page.expect_download(timeout=60_000) as dl:
        animate.click()
    assert dl.value.suggested_filename.endswith((".webm", ".mp4")), dl.value.suggested_filename
    expect(animate).to_be_enabled()
