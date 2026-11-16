"""On-device engine smoke test (TF.js on WebGL2). Slow in headless Chromium, where
WebGL runs on SwiftShader (CPU): run with ``pytest tests/e2e -m slow``."""
from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from e2e_helpers import label_title, paint_button, style_radio, wait_for_painting

BROWSER_PAINT_TIMEOUT_MS = 15 * 60 * 1000


@pytest.mark.slow
def test_browser_engine_paints_on_device(open_page):
    w = open_page(1440, path="/app/?engine=browser")
    page = w.page
    page.set_default_timeout(BROWSER_PAINT_TIMEOUT_MS)
    if not page.evaluate("!!document.createElement('canvas').getContext('webgl2')"):
        pytest.skip("this Chromium has no WebGL2")
    origin = server_url_of(page)
    transfer_calls, foreign_model_files, model_files = [], [], []

    def on_request(r):
        if "/api/transfer" in r.url:
            transfer_calls.append(r.url)
        if r.url.endswith(".bin") or "/models/" in r.url:
            (model_files if r.url.startswith(origin + "/app/models/") else foreign_model_files).append(r.url)

    # Context-level, so requests made by the engine's Web Worker are seen too.
    page.context.on("request", on_request)

    style_radio(page, "The Starry Night").click()
    paint_button(page).click()
    wait_for_painting(page, timeout=BROWSER_PAINT_TIMEOUT_MS)
    expect(label_title(page)).to_have_text(re.compile(r"^After The Starry Night"))
    assert not transfer_calls, "the browser engine fell back to the server"
    assert model_files, "no model file was requested: was the browser engine used?"
    assert not foreign_model_files, f"model files came from elsewhere: {foreign_model_files[:3]}"


def server_url_of(page) -> str:
    return page.evaluate("location.origin")
