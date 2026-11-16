"""POST /api/describe-style and the Workers AI client (HTTP always mocked)."""
import base64
import io
import json
import urllib.error
from datetime import date

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import describe_style as ds
import main


def _png_b64(size=(1024, 1024), color=(20, 120, 200)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _jpeg_b64(size=(1024, 768)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 80, 20)).save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self, n=-1):
        return self._body if n < 0 else self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code: int, payload: dict):
    return urllib.error.HTTPError(
        "https://api.cloudflare.com/x", code, "err", {}, io.BytesIO(json.dumps(payload).encode())
    )


CFG = ds.WorkersAIConfig(account_id="0123456789abcdef0123456789abcdef", api_token="t0ken-SECRET", timeout=5)


# ── Client unit tests ──────────────────────────────────────────────────────────

def test_generate_swatch_request_and_success():
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"], seen["timeout"] = req.full_url, timeout
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        body = {"result": {"image": _jpeg_b64()}, "success": True, "errors": [], "messages": []}
        return _FakeResp(json.dumps(body).encode())

    jpeg = ds.generate_swatch(CFG, "neon city at night", urlopen=fake_urlopen)
    assert seen["url"] == (
        "https://api.cloudflare.com/client/v4/accounts/0123456789abcdef0123456789abcdef"
        "/ai/run/@cf/black-forest-labs/flux-1-schnell"
    )
    assert seen["auth"] == "Bearer t0ken-SECRET" and seen["timeout"] == 5
    assert seen["body"] == {"prompt": "a painterly texture study in the style of: neon city at night", "steps": 4}
    img = Image.open(io.BytesIO(jpeg))
    assert img.format == "JPEG" and img.size == (512, 384)


def test_quota_error_is_reported_as_quota(caplog):
    def fake_urlopen(req, timeout):
        raise _http_error(429, {"success": False, "errors": [{"code": 3036, "message": "daily free allocation"}]})

    with pytest.raises(ds.DescribeUnavailable) as exc:
        ds.generate_swatch(CFG, "x y", urlopen=fake_urlopen)
    assert exc.value.reason == "quota"
    assert "t0ken-SECRET" not in caplog.text


def test_capacity_429_is_a_plain_failure():
    def fake_urlopen(req, timeout):
        raise _http_error(429, {"success": False, "errors": [{"code": 3040}]})

    with pytest.raises(ds.DescribeUnavailable) as exc:
        ds.generate_swatch(CFG, "x y", urlopen=fake_urlopen)
    assert exc.value.reason == "failed"


@pytest.mark.parametrize("error", [
    TimeoutError("timed out"),
    urllib.error.URLError(TimeoutError("timed out")),
    ConnectionResetError("reset"),
])
def test_timeouts_and_network_errors(error, caplog):
    def fake_urlopen(req, timeout):
        raise error

    with pytest.raises(ds.DescribeUnavailable) as exc:
        ds.generate_swatch(CFG, "x y", urlopen=fake_urlopen)
    assert exc.value.reason == "failed"
    assert "t0ken-SECRET" not in caplog.text


@pytest.mark.parametrize("body", [
    b"not json",
    json.dumps({"success": False, "errors": [{"code": 5007}]}).encode(),
    json.dumps({"success": True, "result": {}}).encode(),
    json.dumps({"success": True, "result": {"image": "!!!notbase64"}}).encode(),
    json.dumps({"success": True, "result": {"image": base64.b64encode(b"<html>").decode()}}).encode(),
])
def test_bad_provider_replies_fail_cleanly(body):
    with pytest.raises(ds.DescribeUnavailable) as exc:
        ds.generate_swatch(CFG, "x y", urlopen=lambda req, timeout: _FakeResp(body))
    assert exc.value.reason == "failed"


def test_not_configured_and_account_validation():
    for cfg in (ds.WorkersAIConfig(), ds.WorkersAIConfig(account_id="abc/../x", api_token="t"),
                ds.WorkersAIConfig(account_id="0123456789abcdef", api_token="")):
        assert not cfg.configured
        with pytest.raises(ds.DescribeUnavailable) as exc:
            ds.generate_swatch(cfg, "x y", urlopen=lambda *a, **k: pytest.fail("must not call"))
        assert exc.value.reason == "not_configured"


def test_clean_prompt():
    assert ds.clean_prompt("  neon\x00 city\n\tat​ night\x1b[31m ") == "neon city at night[31m"
    assert ds.clean_prompt(None) == "" and ds.clean_prompt("\x07\x08") == ""


def test_daily_quota_resets_each_day():
    day = [date(2026, 11, 16)]
    q = ds.DailyQuota(2, today=lambda: day[0])
    assert q.try_acquire() and q.try_acquire() and not q.try_acquire()
    assert q.remaining == 0
    day[0] = date(2026, 11, 17)
    assert q.remaining == 2 and q.try_acquire()
    q.exhaust()
    assert not q.try_acquire()


# ── Endpoint tests ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def _app_client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def client(_app_client, monkeypatch):
    main.limiter.reset()
    monkeypatch.setattr(main, "DESCRIBE_QUOTA", ds.DailyQuota(50))
    return _app_client


def _configure(monkeypatch, generate):
    monkeypatch.setattr(main, "DESCRIBE_CFG", CFG)
    monkeypatch.setattr(main, "generate_swatch", generate)


def test_not_configured_falls_back_to_matcher(client):
    r = client.post("/api/describe-style", data={"prompt": "swirling starry sky"})
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    assert r.json() == {
        "source": "matched", "key": "starry_night", "title": "The Starry Night",
        "artist": "Vincent van Gogh", "reason": "not_configured",
        "message": "We matched the closest classic style: The Starry Night.",
    }


def test_json_body_is_accepted(client):
    r = client.post("/api/describe-style", json={"prompt": "munch"})
    assert r.status_code == 200 and r.json()["key"] == "the_scream"


def test_generated_success(client, monkeypatch):
    calls = []

    def fake(cfg, prompt):
        calls.append(prompt)
        return base64.b64decode(_jpeg_b64((512, 384)))

    _configure(monkeypatch, fake)
    r = client.post("/api/describe-style", data={"prompt": "  neon\x00 cyberpunk city "})
    body = r.json()
    assert r.status_code == 200 and body["source"] == "generated"
    assert body["media_type"] == "image/jpeg" and body["prompt"] == "neon cyberpunk city"
    assert Image.open(io.BytesIO(base64.b64decode(body["image"]))).format == "JPEG"
    assert calls == ["neon cyberpunk city"]
    assert main.DESCRIBE_QUOTA.remaining == 49


def test_quota_error_falls_back_and_stops_calling(client, monkeypatch):
    calls = []

    def fake(cfg, prompt):
        calls.append(prompt)
        raise ds.DescribeUnavailable("quota")

    _configure(monkeypatch, fake)
    r = client.post("/api/describe-style", data={"prompt": "van gogh"})
    body = r.json()
    assert body["source"] == "matched" and body["reason"] == "quota"
    assert body["message"].startswith("Today's describe limit is used up.")
    # Provider said the day's allocation is gone: no further calls today.
    r2 = client.post("/api/describe-style", data={"prompt": "van gogh"})
    assert r2.json()["reason"] == "daily_limit" and len(calls) == 1


def test_daily_limit(client, monkeypatch):
    monkeypatch.setattr(main, "DESCRIBE_QUOTA", ds.DailyQuota(1))
    _configure(monkeypatch, lambda cfg, p: base64.b64decode(_jpeg_b64((64, 64))))
    assert client.post("/api/describe-style", data={"prompt": "abstract"}).json()["source"] == "generated"
    body = client.post("/api/describe-style", data={"prompt": "abstract"}).json()
    assert body["source"] == "matched" and body["reason"] == "daily_limit" and body["key"] == "udnie"


def test_timeout_falls_back(client, monkeypatch):
    def fake(cfg, prompt):
        raise ds.DescribeUnavailable("failed", "network error or timeout")

    _configure(monkeypatch, fake)
    body = client.post("/api/describe-style", data={"prompt": "munch scream"}).json()
    assert body["reason"] == "failed" and body["key"] == "the_scream"
    assert body["message"].startswith("The style generator isn't available right now.")


def test_unexpected_error_falls_back(client, monkeypatch):
    def fake(cfg, prompt):
        raise RuntimeError("boom")

    _configure(monkeypatch, fake)
    assert client.post("/api/describe-style", data={"prompt": "munch"}).json()["reason"] == "failed"


def test_no_match_is_404_with_a_hint(client, monkeypatch):
    r = client.post("/api/describe-style", data={"prompt": "neon cyberpunk"})
    assert r.status_code == 404 and r.json()["detail"].startswith("No classic style matches")
    _configure(monkeypatch, lambda cfg, p: (_ for _ in ()).throw(ds.DescribeUnavailable("failed")))
    r = client.post("/api/describe-style", data={"prompt": "neon cyberpunk"})
    assert r.status_code == 404
    assert r.json()["detail"].startswith("The style generator isn't available right now. No classic style")


@pytest.mark.parametrize("data", [{}, {"prompt": ""}, {"prompt": "   \x00\x01 "}, {"prompt": "a" * 201}])
def test_prompt_validation(client, data):
    r = client.post("/api/describe-style", data=data)
    assert r.status_code == 400 and "1 to 200 characters" in r.json()["detail"]


def test_prompt_200_chars_ok_and_bad_json(client):
    assert client.post("/api/describe-style", data={"prompt": "munch " + "a" * 194}).status_code == 200
    r = client.post("/api/describe-style", content=b"{oops", headers={"content-type": "application/json"})
    assert r.status_code == 400
    r = client.post("/api/describe-style", json=["munch"])
    assert r.status_code == 400


def test_oversized_body_rejected(client):
    r = client.post("/api/describe-style", data={"prompt": "a" * 20000})
    assert r.status_code == 413


def test_rate_limited_per_ip(client):
    ip = {"cf-connecting-ip": "198.51.100.9"}
    codes = [client.post("/api/describe-style", data={"prompt": "munch"}, headers=ip).status_code
             for _ in range(6)]
    assert codes == [200] * 5 + [429]


def test_disabled(client, monkeypatch):
    monkeypatch.setattr(main, "SITE", main.SiteConfig(describe_enabled=False))
    assert client.post("/api/describe-style", data={"prompt": "munch"}).status_code == 404
