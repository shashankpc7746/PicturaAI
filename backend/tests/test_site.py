"""Styles API, health, config.js, security headers, robots/sitemap, docs toggle."""
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

import main
from conftest import BACKEND_DIR
from site_config import DEFAULT_SITE_URL, MEDIAPIPE_CDN, MEDIAPIPE_VERSION, SiteConfig, is_docs_path, is_https


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


# ── /api/styles & /api/health ──────────────────────────────────────────────────

def test_styles_list_shape_and_cache(client):
    r = client.get("/api/styles")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "public, max-age=3600"
    body = r.json()
    assert [s["key"] for s in body] == ["starry_night", "the_scream", "udnie"]
    s = body[0]
    assert s["title"] == s["name"] == "The Starry Night"
    assert s["slug"] == "van-gogh-starry-night" and s["artist"] == "Vincent van Gogh"
    assert s["year"] == "1889" and s["aspect"] == 1.26 and s["credit"].startswith("Vincent")
    assert s["image"] == "/app/assets/styles/starry_night-512.jpg"
    assert s["card"] == "/app/assets/styles/starry_night-480.webp"
    assert s["thumb"] == "/app/assets/styles/starry_night-160.webp"
    assert s["page"] == "/app/styles/van-gogh-starry-night/"
    assert "thumbnail" not in s and "file" not in s and "tags" not in s


def test_styles_legacy_thumbnail_on_request(client):
    import base64, io
    from PIL import Image
    body = client.get("/api/styles?include=thumbnail").json()
    thumb = Image.open(io.BytesIO(base64.b64decode(body[0]["thumbnail"])))
    assert thumb.format == "JPEG" and max(thumb.size) == 200
    # The plain list stays light afterwards.
    assert "thumbnail" not in client.get("/api/styles").json()[0]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert client.head("/api/health").status_code == 200


# ── /app/config.js ─────────────────────────────────────────────────────────────

def test_config_js_served_before_static_mount(client):
    r = client.get("/app/config.js")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/javascript")
    assert r.headers["cache-control"] == "no-cache"
    prefix, _, rest = r.text.partition("window.PICTURA_CONFIG = Object.freeze(")
    cfg = json.loads(rest.rsplit(");", 1)[0])
    assert cfg == {
        "siteUrl": DEFAULT_SITE_URL,
        "sentryDsn": None,
        "cfWebAnalyticsToken": None,
        "describeEnabled": True,
        "describeGenerates": False,
    }


def test_config_js_escapes_hostile_values():
    cfg = SiteConfig(cf_web_analytics_token='abc</script><script>alert(1)</script>" ')
    js = cfg.config_js()
    assert "</script>" not in js and "<" not in js.split("Object.freeze(")[1]
    payload = js.split("Object.freeze(", 1)[1].rsplit(");", 1)[0]
    assert json.loads(payload)["cfWebAnalyticsToken"].startswith("abc</script>")


def test_config_never_contains_server_secrets(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "server-secret-token")
    monkeypatch.setenv("SENTRY_DSN", "https://backendkey@o1.ingest.sentry.io/1")
    js = SiteConfig.from_env(describe_generates=True).config_js()
    assert "server-secret-token" not in js and "backendkey" not in js


def test_site_url_is_sanitised(monkeypatch):
    monkeypatch.setenv("PUBLIC_SITE_URL", "https://example.org/")
    assert SiteConfig.from_env().site_url == "https://example.org"
    monkeypatch.setenv("PUBLIC_SITE_URL", "javascript:alert(1)")
    assert SiteConfig.from_env().site_url == DEFAULT_SITE_URL


def test_describe_flags(monkeypatch):
    assert SiteConfig.from_env(describe_generates=True).describe_generates is True
    monkeypatch.setenv("DESCRIBE_ENABLED", "false")
    cfg = SiteConfig.from_env(describe_generates=True)
    assert cfg.describe_enabled is False and cfg.describe_generates is False


# ── Security headers ───────────────────────────────────────────────────────────

def _csp(headers) -> dict:
    out = {}
    for part in headers["content-security-policy"].split(";"):
        name, *values = part.split()
        out[name] = values
    return out


def test_app_csp_and_headers(client):
    r = client.get("/app/")
    h = r.headers
    assert h["x-content-type-options"] == "nosniff"
    assert h["x-frame-options"] == "DENY"
    assert h["referrer-policy"] == "strict-origin-when-cross-origin"
    assert h["permissions-policy"] == "camera=(), microphone=(), geolocation=()"
    assert "strict-transport-security" not in h  # plain http
    csp = _csp(h)
    assert csp["default-src"] == ["'self'"]
    assert csp["script-src"] == ["'self'", MEDIAPIPE_CDN, "'wasm-unsafe-eval'"]
    assert MEDIAPIPE_CDN == "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/"
    assert "'unsafe-inline'" not in " ".join(sum(csp.values(), []))
    assert "'unsafe-eval'" not in csp["script-src"]
    assert csp["style-src"] == ["'self'", "https://fonts.googleapis.com"]
    assert csp["font-src"] == ["'self'", "https://fonts.gstatic.com"]
    assert csp["img-src"] == ["'self'", "data:", "blob:"]
    assert csp["media-src"] == ["'self'", "blob:"]
    assert csp["connect-src"] == ["'self'", MEDIAPIPE_CDN]
    # Only the pinned MediaPipe path, never the whole CDN host.
    assert "https://cdn.jsdelivr.net" not in csp["script-src"] + csp["connect-src"]
    assert csp["worker-src"] == ["'self'", "blob:"]
    assert csp["object-src"] == ["'none'"]
    assert csp["base-uri"] == ["'self'"]
    assert csp["frame-ancestors"] == ["'none'"]


def test_mediapipe_version_matches_segment_js():
    import re
    segment = (BACKEND_DIR.parent / "frontend" / "assets" / "js" / "engine" / "segment.js").read_text(encoding="utf-8")
    match = re.search(r"const MP_VERSION = '([^']+)'", segment)
    assert match, "MP_VERSION not found in segment.js"
    assert match.group(1) == MEDIAPIPE_VERSION
    assert "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}" in segment


def test_csp_also_on_api_responses(client):
    assert "content-security-policy" in client.get("/api/health").headers


def test_hsts_only_over_https(client):
    assert "strict-transport-security" not in client.get("/api/health").headers
    r = client.get("/api/health", headers={"x-forwarded-proto": "https"})
    assert r.headers["strict-transport-security"] == "max-age=31536000; includeSubDomains"
    r = client.get("/api/health", headers={"x-forwarded-proto": "http"})
    assert "strict-transport-security" not in r.headers


def test_is_https_helper():
    assert is_https("https", None)
    assert is_https("http", "https, http")
    assert not is_https("http", "http")
    assert not is_https("http", None)


def test_optional_third_parties_only_when_configured():
    plain = SiteConfig().app_csp()
    assert "cloudflareinsights" not in plain and "sentry" not in plain

    cf = SiteConfig(cf_web_analytics_token="tok").app_csp()
    assert f"script-src 'self' {MEDIAPIPE_CDN} 'wasm-unsafe-eval' https://static.cloudflareinsights.com" in cf
    assert "https://cloudflareinsights.com" in cf.split("connect-src", 1)[1].split(";")[0]

    sentry = SiteConfig(sentry_dsn_frontend="https://k@o1.ingest.us.sentry.io/2").app_csp()
    connect = sentry.split("connect-src", 1)[1].split(";")[0]
    assert "https://browser.sentry-cdn.com" in sentry.split("script-src", 1)[1].split(";")[0]
    assert "https://*.ingest.sentry.io" in connect and "https://*.ingest.us.sentry.io" in connect

    self_hosted = SiteConfig(sentry_dsn_frontend="https://k@sentry.example.com/3").app_csp()
    assert "https://sentry.example.com" in self_hosted.split("connect-src", 1)[1].split(";")[0]


def test_report_only_mode():
    cfg = SiteConfig(csp_report_only=True)
    assert cfg.csp_header_name() == "Content-Security-Policy-Report-Only"


def test_docs_get_relaxed_policy_and_still_work(client):
    r = client.get("/docs")
    assert r.status_code == 200 and "swagger-ui" in r.text
    csp = _csp(r.headers)
    assert "'unsafe-inline'" in csp["script-src"] and "https://cdn.jsdelivr.net" in csp["script-src"]
    assert csp["frame-ancestors"] == ["'none'"]
    assert client.get("/redoc").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert is_docs_path("/docs/oauth2-redirect") and not is_docs_path("/docsx")


def test_openapi_documents_new_and_legacy_fields(client):
    spec = client.get("/openapi.json").json()
    schema_ref = spec["paths"]["/api/transfer"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    props = spec["components"]["schemas"][schema_ref.rsplit("/", 1)[1]]["properties"]
    assert props["output"]["enum"] == ["final", "raw"] and props["output"]["default"] == "final"
    assert "Alias of style_weight" in props["intensity"]["description"]
    for legacy in ("content_weight", "tv_weight", "num_steps", "learning_rate"):
        assert props[legacy]["description"].startswith("Ignored")
    assert "/api/describe-style" in spec["paths"]


def test_api_docs_can_be_disabled(tmp_path):
    code = (
        "from fastapi.testclient import TestClient; import main; c = TestClient(main.app); "
        "print([c.get(p).status_code for p in ('/docs', '/redoc', '/openapi.json', '/api/health')])"
    )
    env = {**os.environ, "ENABLE_API_DOCS": "false", "TF_CPP_MIN_LOG_LEVEL": "3"}
    proc = subprocess.run([sys.executable, "-c", code], cwd=BACKEND_DIR, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "[404, 404, 404, 200]"


# ── robots.txt & sitemap.xml ───────────────────────────────────────────────────

def test_robots_txt(client):
    r = client.get("/robots.txt")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert "Disallow: /api/" in r.text
    assert f"Sitemap: {DEFAULT_SITE_URL}/sitemap.xml" in r.text


def test_sitemap_lists_app_style_pages_and_static_pages(client):
    r = client.get("/sitemap.xml")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/xml")
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = [e.text for e in ET.fromstring(r.content).findall("s:url/s:loc", ns)]
    base = f"{DEFAULT_SITE_URL}/app"
    assert locs[0] == f"{base}/"
    for slug in ("van-gogh-starry-night", "munch-the-scream", "picabia-udnie"):
        assert f"{base}/styles/{slug}/" in locs
    for page in ("privacy.html", "terms.html", "credits.html"):
        assert f"{base}/{page}" in locs
    assert len(locs) == len(set(locs))


def test_sitemap_styles_index_and_custom_site_url():
    cfg = SiteConfig(site_url="https://pictura.example")
    urls = cfg.sitemap_urls(["a-b"], include_styles_index=True)
    assert urls[:3] == ["https://pictura.example/app/", "https://pictura.example/app/styles/",
                        "https://pictura.example/app/styles/a-b/"]
    assert "Sitemap: https://pictura.example/sitemap.xml" in cfg.robots_txt()


# ── Optional Sentry ────────────────────────────────────────────────────────────

def test_sentry_off_without_dsn():
    assert main.SENTRY_DSN == "" and main.SENTRY_ENABLED is False


def test_sentry_missing_package_is_tolerated(monkeypatch):
    monkeypatch.setattr(main, "SENTRY_DSN", "https://k@o1.ingest.sentry.io/1")
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)  # makes the import fail
    assert main._init_sentry() is False


@pytest.mark.parametrize("dsn, enabled", [
    ("https://publickey@o0.ingest.sentry.io/0", True),
    ("not a dsn", False),
])
def test_sentry_init_in_a_fresh_process(dsn, enabled):
    code = (
        "import sentry_sdk; from fastapi.testclient import TestClient; import main; "
        "print(main.SENTRY_ENABLED, sentry_sdk.get_client().is_active(), "
        "TestClient(main.app).get('/api/health').status_code)"
    )
    env = {**os.environ, "SENTRY_DSN": dsn, "TF_CPP_MIN_LOG_LEVEL": "3"}
    proc = subprocess.run([sys.executable, "-c", code], cwd=BACKEND_DIR, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == f"{enabled} {enabled} 200"
