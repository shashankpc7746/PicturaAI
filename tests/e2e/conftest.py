"""End-to-end test fixtures: a real PicturaAI server and headless Chromium.

The ``server_url`` fixture starts ``uvicorn main:app`` from ``backend/`` on
port 8230 (``E2E_PORT``) with ``NST_MAX_DIM=320``, waits for ``/api/health``
and stops it at the end of the session. Set ``E2E_BASE_URL`` (for example
``http://127.0.0.1:8000``) to test a server you started yourself instead.

Every page made with ``open_page`` records console errors, uncaught page
errors and Content-Security-Policy violations; the fixture fails the test at
teardown if any were seen that the test did not expect (``watch.allow(...)``).

Tests marked ``@pytest.mark.slow`` are skipped unless ``-m slow`` is given.
"""
from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Pattern

import pytest
from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from e2e_helpers import VIEWPORTS

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
PORT = int(os.environ.get("E2E_PORT", "8230"))
STARTUP_TIMEOUT_S = float(os.environ.get("E2E_STARTUP_TIMEOUT", "180"))



# ── Markers ──────────────────────────────────────────────────────────────
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long test (browser engine); run with -m slow")


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    if "slow" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason="slow test: run with -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


# ── Server ───────────────────────────────────────────────────────────────
def _health_ok(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/api/health", timeout=3) as res:
            return res.status == 200 and b'"ok"' in res.read()
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _server_env() -> dict:
    env = dict(os.environ)
    # Never reach real third parties from tests: no Workers AI, Sentry or analytics.
    for name in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "SENTRY_DSN",
                 "SENTRY_DSN_FRONTEND", "CF_WEB_ANALYTICS_TOKEN", "CLIENT_IP_HEADER",
                 "CSP_REPORT_ONLY", "STYLES_FILE"):
        env.pop(name, None)
    env.update({
        "NST_MAX_DIM": "320",
        "NST_OUTPUT_DIM": "320",
        "DESCRIBE_ENABLED": "true",
        # One browser makes many requests; production limits would turn into 429s.
        "RATE_LIMIT_TRANSFER": "1000/minute",
        "RATE_LIMIT_PALETTE": "1000/minute",
        "RATE_LIMIT_INTERPOLATE": "1000/minute",
        "RATE_LIMIT_DESCRIBE": "1000/minute",
        "PYTHONUNBUFFERED": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
    })
    return env


@pytest.fixture(scope="session")
def server_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    external = os.environ.get("E2E_BASE_URL", "").strip().rstrip("/")
    if external:
        if not _health_ok(external):
            pytest.exit(f"E2E_BASE_URL={external} is not healthy (GET /api/health)", returncode=1)
        yield external
        return

    base = f"http://127.0.0.1:{PORT}"
    if _port_in_use(PORT):
        pytest.exit(f"port {PORT} is already in use; stop that server or set E2E_PORT/E2E_BASE_URL",
                    returncode=1)
    log_path = tmp_path_factory.mktemp("server") / "uvicorn.log"
    log = open(log_path, "wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(BACKEND_DIR), env=_server_env(), stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while not _health_ok(base):
            if proc.poll() is not None:
                log.flush()
                tail = log_path.read_text(errors="replace")[-3000:]
                pytest.exit(f"server exited with {proc.returncode} during startup:\n{tail}", returncode=1)
            if time.monotonic() > deadline:
                pytest.exit(f"server did not become healthy in {STARTUP_TIMEOUT_S:.0f}s; log: {log_path}",
                            returncode=1)
            time.sleep(0.5)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log.close()


# ── Browser ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def chromium() -> Iterator[Browser]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=os.environ.get("E2E_HEADED") != "1")
        yield browser
        browser.close()


# Recorded in the page before any of its own scripts run. Init scripts are not
# subject to the page's CSP, so this listener sees every violation.
CSP_LISTENER = """
window.__cspViolations = [];
document.addEventListener('securitypolicyviolation', (e) => {
  window.__cspViolations.push(`${e.effectiveDirective} blocked ${e.blockedURI || 'inline'}`
    + (e.sourceFile ? ` (${e.sourceFile}:${e.lineNumber})` : ''));
});
"""


@dataclass
class PageWatch:
    """Console errors, page errors and CSP violations seen by one page."""

    page: Page
    errors: List[str] = field(default_factory=list)
    allowed: List[Pattern[str]] = field(default_factory=list)

    def allow(self, pattern: str) -> None:
        """Ignore console errors matching this regex (e.g. an intentionally mocked 503)."""
        self.allowed.append(re.compile(pattern, re.I))

    def unexpected_errors(self) -> List[str]:
        return [e for e in self.errors if not any(p.search(e) for p in self.allowed)]

    def csp_violations(self) -> List[str]:
        try:
            return list(self.page.evaluate("window.__cspViolations || []"))
        except Exception:  # page closed or navigating
            return []


# The fonts are self-hosted under /app/assets/fonts/, so the suite needs no internet
# for them and the layout checks always measure the real typefaces
# (test_assets.py checks that no page asks Google Fonts).


@pytest.fixture
def open_page(chromium: Browser, server_url: str) -> Iterator[Callable[..., PageWatch]]:
    """Factory: open_page(width=1440, path='/app/?engine=server', **context_options) -> PageWatch."""
    contexts: List[BrowserContext] = []
    watches: List[PageWatch] = []

    def _open(width: int = 1440, path: str = "/app/?engine=server", *, height: Optional[int] = None,
              wait_for_styles: bool = True, setup: Optional[Callable[[Page], None]] = None,
              **ctx_opts) -> PageWatch:
        ctx = chromium.new_context(
            viewport={"width": width, "height": height or VIEWPORTS.get(width, 900)},
            accept_downloads=True, **ctx_opts,
        )
        contexts.append(ctx)
        ctx.add_init_script(CSP_LISTENER)
        page = ctx.new_page()
        page.set_default_timeout(30_000)
        watch = PageWatch(page)
        page.on("console", lambda m: watch.errors.append(f"console: {m.text}") if m.type == "error" else None)
        page.on("pageerror", lambda e: watch.errors.append(f"pageerror: {e}"))
        watches.append(watch)
        if setup:
            setup(page)
        page.goto(server_url + path, wait_until="load")
        if wait_for_styles:
            page.get_by_role("radiogroup", name="Paintings").get_by_role("radio").first.wait_for()
        return watch

    yield _open

    problems = []
    for w in watches:
        errs = w.unexpected_errors()
        csp = w.csp_violations()
        if errs:
            problems.append("console/page errors: " + "; ".join(errs[:5]))
        if csp:
            problems.append("CSP violations: " + "; ".join(csp[:5]))
    for ctx in contexts:
        ctx.close()
    if problems:
        pytest.fail("\n".join(problems), pytrace=False)
