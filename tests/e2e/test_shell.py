"""The page shell: headers, first load, layout at every target width, the menu sheet."""
from __future__ import annotations

import json
import re
import urllib.request

import pytest
from playwright.sync_api import expect

from e2e_helpers import VIEWPORTS, clipped_controls, overflow, paint_button


def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as res:
        return res.status, {k.lower(): v for k, v in res.headers.items()}, res.read()


def test_health(server_url):
    status, headers, body = _get(server_url + "/api/health")
    assert status == 200
    assert b'"ok"' in body


def test_security_headers_on_app(server_url):
    status, h, body = _get(server_url + "/app/")
    assert status == 200 and b"<html" in body.lower()
    assert h.get("x-content-type-options") == "nosniff"
    assert h.get("x-frame-options") == "DENY"
    assert h.get("referrer-policy")
    assert "camera=()" in h.get("permissions-policy", "")
    csp = h.get("content-security-policy") or h.get("content-security-policy-report-only")
    assert csp, "no Content-Security-Policy header"
    directives = {d.strip().split(" ")[0]: d.strip() for d in csp.split(";") if d.strip()}
    script_src = directives.get("script-src", "")
    assert "'self'" in script_src
    assert "'unsafe-inline'" not in script_src and "'unsafe-eval'" not in script_src
    assert "frame-ancestors" not in directives or "'none'" in directives["frame-ancestors"]
    # No HSTS on plain HTTP (it would pin localhost).
    assert "strict-transport-security" not in h


def test_config_js_is_public_and_script(server_url):
    status, h, body = _get(server_url + "/app/config.js")
    assert status == 200
    assert "javascript" in h.get("content-type", "")
    text = body.decode()
    assert "window.PICTURA_CONFIG" in text
    assert "describeGenerates" in text
    assert not re.search(r"token\W+[A-Za-z0-9_-]{20,}", text, re.I), "config.js looks like it leaks a token"


def test_studio_loads_with_sample(open_page, server_url):
    _, _, styles_body = _get(server_url + "/api/styles")
    presets = json.loads(styles_body)
    assert isinstance(presets, list) and presets, "/api/styles returned no presets"
    w = open_page(1440)
    page = w.page
    expect(page.get_by_role("heading", level=1)).to_have_text("Turn a photo into a painting")
    expect(page.locator("#label-title")).to_have_text("Sample photo")
    expect(page.get_by_role("button", name="Dog", exact=True)).to_have_attribute("aria-pressed", "true")
    photo = page.locator("#frame-photo")
    expect(photo).to_be_visible()
    assert page.evaluate("document.getElementById('frame-photo').naturalWidth") > 0
    cards = page.get_by_role("radiogroup", name="Paintings").get_by_role("radio")
    # One card per preset in /api/styles (data/styles.json), no more and no fewer.
    expect(cards).to_have_count(len(presets))
    # One tab stop in the radio group.
    assert page.locator('.style-card[role="radio"][tabindex="0"]').count() == 1
    # Paint is disabled with the reason shown until a painting is chosen.
    expect(paint_button(page)).to_be_disabled()
    expect(page.locator("#paint-reason")).to_have_text("Pick a painting first")
    expect(page.get_by_role("navigation", name="Main").first).to_be_visible()


@pytest.mark.parametrize("width", list(VIEWPORTS))
def test_no_horizontal_overflow(open_page, width):
    w = open_page(width, has_touch=width < 1024)
    page = w.page
    page.wait_for_load_state("networkidle")
    page.evaluate("document.fonts.ready.then(() => true)")
    for where in ("top", "bottom"):
        if where == "bottom":
            page.evaluate("window.scrollTo({top: document.body.scrollHeight, behavior: 'instant'})")
        sw, cw = overflow(page)
        assert sw <= cw, f"{width}px {where}: page scrolls sideways ({sw} > {cw})"
    assert clipped_controls(page) == []


def test_menu_sheet_focus_trap_at_360(open_page):
    w = open_page(360, has_touch=True)
    page = w.page
    menu_btn = page.get_by_role("button", name="Menu", exact=True)
    expect(menu_btn).to_be_visible()
    expect(page.locator(".site-nav")).to_be_hidden()
    menu_btn.click()
    sheet = page.get_by_role("dialog", name="Menu")
    expect(sheet).to_be_visible()
    expect(menu_btn).to_have_attribute("aria-expanded", "true")
    for _ in range(8):
        page.keyboard.press("Tab")
        assert page.evaluate("!!document.activeElement.closest('#menu-sheet')"), "focus left the menu sheet"
    page.keyboard.press("Shift+Tab")
    assert page.evaluate("!!document.activeElement.closest('#menu-sheet')")
    page.keyboard.press("Escape")
    expect(sheet).to_be_hidden()
    expect(menu_btn).to_be_focused()
    expect(menu_btn).to_have_attribute("aria-expanded", "false")


def test_styles_unavailable_shows_status(open_page):
    """/api/styles down (a sleeping Render instance answers 503 HTML): a status, not a crash."""
    w = open_page(
        1440, wait_for_styles=False,
        setup=lambda p: p.route("**/api/styles*", lambda r: r.fulfill(
            status=503, content_type="text/html", body="<html>waking up</html>")),
    )
    w.allow(r"status of 503")
    page = w.page
    status = page.locator("#styles-status")
    expect(status).to_be_visible(timeout=15_000)
    expect(status).not_to_be_empty()
    expect(paint_button(page)).to_be_disabled()


def test_csp_blocks_inline_script_and_is_watched(open_page):
    """The CSP is enforced in the browser, and the suite's violation listener sees it."""
    w = open_page(1440)
    w.allow(r"(Refused to execute|Executing) inline script")
    page = w.page
    ran = page.evaluate("""() => new Promise((resolve) => {
      const s = document.createElement('script');
      s.textContent = 'window.__inlineRan = true';
      document.head.append(s);
      setTimeout(() => resolve(!!window.__inlineRan), 100);
    })""")
    assert ran is False, "an inline script ran: the CSP allows 'unsafe-inline'"
    violations = w.csp_violations()
    assert any(v.startswith("script-src") for v in violations), violations
    page.evaluate("window.__cspViolations = []")
