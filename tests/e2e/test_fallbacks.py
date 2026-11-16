"""Engine fallbacks and config.js wiring.

Each fallback is made to fail early with routes (a blocked worker script, blocked
TF.js files, a blocked engine module), so these tests never run TF.js inference and
stay fast. The one test that really paints with the in-page engine is ``slow``.

Routes are set on the browser context so they also catch requests made by the
engine's Web Worker.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional

import pytest
from playwright.sync_api import Page, Route, expect

from e2e_helpers import label_title, paint_button, style_radio, wait_for_painting

ENGINE_INFO_JS = """() => import('/app/assets/js/services/engine-adapter.js')
  .then((m) => m.getEngine())
  .then((e) => ({ kind: e.kind, off: e.offMainThread ?? null }))"""

SERVER_TOAST = "server is painting it"
# Toasts disappear after a few seconds; keep every toast's text for the assertions.
TOAST_RECORDER = """
window.__toastLog = [];
document.addEventListener('DOMContentLoaded', () => {
  const box = document.getElementById('toasts');
  if (!box) return;
  new MutationObserver((records) => {
    for (const r of records) for (const n of r.addedNodes) {
      if (n.textContent && n.textContent.trim()) window.__toastLog.push(n.textContent.trim());
    }
  }).observe(box, { childList: true, subtree: true });
});
"""
# Aborted routes show up as failed loads in the console; they are what the test does.
BLOCKED_LOAD = r"Failed to load resource|net::ERR_FAILED|Failed to fetch dynamically imported module"


def _routes(*pairs, record: Optional[List[str]] = None, init_script: str = "") -> Callable[[Page], None]:
    """open_page setup: context routes, plus every request URL appended to ``record``
    from before the first navigation (the engine starts loading on idle)."""
    def setup(page: Page) -> None:
        if record is not None:
            page.context.on("request", lambda r: record.append(r.url))
        page.context.add_init_script(TOAST_RECORDER)
        if init_script:
            page.context.add_init_script(init_script)
        for pattern, handler in pairs:
            page.context.route(pattern, handler)
    return setup


def _abort(route: Route) -> None:
    route.abort()


def _config(body: str) -> Callable[[Route], None]:
    return lambda r: r.fulfill(status=200, content_type="application/javascript", body=body)


def _require_browser_engine(page: Page, *, worker: bool = False) -> None:
    """Skip when this Chromium would never pick the browser engine (no usable WebGL2)."""
    caps = page.evaluate("""async () => {
      const s = await import('/app/assets/js/engine/support.js');
      const w = await import('/app/assets/js/engine/worker-engine.js');
      return { webgl2: s.hasWebGL2(), worker: w.workerCapable() };
    }""")
    if not caps["webgl2"]:
        pytest.skip("this Chromium has no WebGL2 with float render targets, so it always uses the server")
    if worker and not caps["worker"]:
        pytest.skip("this Chromium has no OffscreenCanvas WebGL2, so the engine never starts a worker")


def _paint(page: Page, title: str = "The Starry Night", timeout: Optional[int] = None) -> None:
    style_radio(page, title).click()
    expect(style_radio(page, title)).to_have_attribute("aria-checked", "true")
    paint_button(page).click()
    if timeout:
        wait_for_painting(page, timeout)
    else:
        wait_for_painting(page)
    expect(label_title(page)).to_have_text(re.compile(r"^After " + re.escape(title)))
    expect(page.locator("#frame-error")).to_be_hidden()


def _toasts(page: Page) -> str:
    return " | ".join(page.evaluate("window.__toastLog || []"))


def _transfer_calls(reqs: List[str]) -> List[str]:
    return [u for u in reqs if "/api/transfer" in u]


# ── Engine fallbacks (fast: the browser engine fails before inference) ────
def test_worker_blocked_moves_engine_to_the_page(open_page):
    """engine/worker.js cannot load -> the browser engine runs on the page.

    TF.js is blocked too, so the in-page engine then fails quickly and the server
    paints; what this checks is that the engine moved to the page (offMainThread
    false, browser-engine.js loaded by the page) instead of failing outright.
    """
    reqs: List[str] = []
    w = open_page(1440, path="/app/", setup=_routes(
        ("**/assets/js/engine/worker.js", _abort), ("**/vendor/tfjs/**", _abort), record=reqs))
    w.allow(BLOCKED_LOAD)
    page = w.page
    _require_browser_engine(page, worker=True)
    _paint(page)
    assert page.evaluate(ENGINE_INFO_JS) == {"kind": "browser", "off": False}
    assert any(u.endswith("/assets/js/engine/browser-engine.js") for u in reqs), \
        "the page never loaded browser-engine.js after the worker failed"
    assert _transfer_calls(reqs), "the server did not paint after the in-page engine failed"
    assert SERVER_TOAST in _toasts(page), f"toasts: {_toasts(page)!r}"


def test_tfjs_blocked_falls_back_to_server_with_toast(open_page):
    """TF.js cannot load (ENGINE_UNAVAILABLE) -> the server paints and the person is told."""
    reqs: List[str] = []
    w = open_page(1440, path="/app/", setup=_routes(("**/vendor/tfjs/**", _abort), record=reqs))
    w.allow(BLOCKED_LOAD)
    page = w.page
    _require_browser_engine(page)
    _paint(page)
    assert _transfer_calls(reqs), "no /api/transfer request: the server engine did not paint"
    assert SERVER_TOAST in _toasts(page), f"toasts: {_toasts(page)!r}"


def test_webgl_backend_failing_in_worker_moves_engine_to_the_page(open_page):
    """TF.js loads in the worker but its WebGL backend does not (ENGINE_WORKER) -> in-page engine.

    The stub backend is served to the page as well, where Subresource Integrity rejects
    it, so the in-page engine fails fast and the server paints.
    """
    reqs: List[str] = []
    w = open_page(1440, path="/app/", setup=_routes(
        ("**/vendor/tfjs/*/tf-backend-webgl.es2017.min.js",
         lambda r: r.fulfill(status=200, content_type="text/javascript", body="/* no backend in this test */")),
        record=reqs))
    w.allow(BLOCKED_LOAD)
    w.allow(r"integrity|digest")
    page = w.page
    _require_browser_engine(page, worker=True)
    _paint(page)
    assert page.evaluate(ENGINE_INFO_JS) == {"kind": "browser", "off": False}
    assert _transfer_calls(reqs)
    assert SERVER_TOAST in _toasts(page), f"toasts: {_toasts(page)!r}"


def test_engine_module_blocked_uses_server_engine(open_page):
    """engine/index.js cannot load -> the adapter uses the server engine, with no on-device claim."""
    reqs: List[str] = []
    w = open_page(1440, path="/app/", setup=_routes(("**/assets/js/engine/index.js", _abort), record=reqs))
    w.allow(BLOCKED_LOAD)
    page = w.page
    _paint(page)
    info = page.evaluate(ENGINE_INFO_JS)
    assert info["kind"] == "server", info
    assert _transfer_calls(reqs)
    expect(page.locator("#studio-lede")).not_to_contain_text("on your device")
    assert SERVER_TOAST not in _toasts(page), f"toasts: {_toasts(page)!r}"


def test_no_worker_and_browser_engine_module_blocked_uses_server(open_page):
    """?worker=0 keeps the engine on the page; if browser-engine.js cannot load, the server paints."""
    reqs: List[str] = []
    w = open_page(1440, path="/app/?worker=0", setup=_routes(
        ("**/assets/js/engine/browser-engine.js", _abort), record=reqs))
    w.allow(BLOCKED_LOAD)
    page = w.page
    _require_browser_engine(page)
    _paint(page)
    assert any(u.endswith("/assets/js/engine/browser-engine.js") for u in reqs), \
        "browser-engine.js was never requested, so this case was not exercised"
    assert not any(u.endswith("/assets/js/engine/worker.js") for u in reqs), "?worker=0 still started a worker"
    assert _transfer_calls(reqs), "no /api/transfer request: the server engine did not paint"
    assert SERVER_TOAST in _toasts(page), f"toasts: {_toasts(page)!r}"


def test_no_webgl_uses_server_engine(open_page):
    """No WebGL at all -> the server engine, chosen up front (no fallback toast, no errors)."""
    no_webgl = """
      for (const C of [self.HTMLCanvasElement, self.OffscreenCanvas]) {
        if (!C) continue;
        const get = C.prototype.getContext;
        C.prototype.getContext = function (type, ...rest) {
          return /webgl/i.test(String(type)) ? null : get.call(this, type, ...rest);
        };
      }"""
    reqs: List[str] = []
    w = open_page(1440, path="/app/", setup=_routes(record=reqs, init_script=no_webgl))
    page = w.page
    _paint(page)
    assert page.evaluate(ENGINE_INFO_JS)["kind"] == "server"
    assert _transfer_calls(reqs)
    assert not any("/vendor/tfjs/" in u for u in reqs), "TF.js was downloaded without WebGL"
    assert SERVER_TOAST not in _toasts(page), f"toasts: {_toasts(page)!r}"


# ── config.js ─────────────────────────────────────────────────────────────
def test_describe_disabled_hides_the_tab(open_page):
    w = open_page(1440, setup=_routes(("**/app/config.js", _config(
        "window.PICTURA_CONFIG = Object.freeze({describeEnabled: false, describeGenerates: false});"))))
    page = w.page
    expect(page.get_by_role("tab", name="Gallery", exact=True)).to_be_visible()
    expect(page.get_by_role("tab", name="Your painting")).to_be_visible()
    expect(page.locator("#tab-describe")).to_be_hidden()
    expect(page.get_by_role("tab", name="Describe it")).to_have_count(0)


def test_describe_generates_says_create_style(open_page):
    w = open_page(1440, setup=_routes(("**/app/config.js", _config(
        "window.PICTURA_CONFIG = Object.freeze({describeEnabled: true, describeGenerates: true});"))))
    page = w.page
    page.get_by_role("tab", name="Describe it").click()
    expect(page.locator("#describe-form").get_by_role("button")).to_have_text("Create style")


def test_missing_config_js_still_works(open_page):
    w = open_page(1440, setup=_routes(("**/app/config.js", lambda r: r.fulfill(status=404, body=""))))
    w.allow(r"status of 404")
    page = w.page
    expect(page.get_by_role("tab", name="Describe it")).to_be_visible()
    _paint(page, "The Scream")


# ── Slow: the in-page engine really paints ────────────────────────────────
@pytest.mark.slow
def test_worker_blocked_in_page_engine_paints(open_page):
    """engine/worker.js cannot load -> the in-page engine paints with no server call."""
    reqs: List[str] = []
    w = open_page(1440, path="/app/", setup=_routes(("**/assets/js/engine/worker.js", _abort), record=reqs))
    w.allow(BLOCKED_LOAD)
    page = w.page
    page.set_default_timeout(15 * 60 * 1000)
    _require_browser_engine(page, worker=True)
    _paint(page, timeout=15 * 60 * 1000)
    assert page.evaluate(ENGINE_INFO_JS) == {"kind": "browser", "off": False}
    assert not _transfer_calls(reqs), "the in-page engine fell back to the server"
