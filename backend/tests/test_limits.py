"""Memory and abuse limits: pixel budgets, request size, form parts, queue guard,
text_prompt cap, no stored transfer results, uvicorn proxy trust."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import describe_style as ds
import main


def _jpeg(size=(96, 64), color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def _png(size, mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, 0).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def _app_client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def client(_app_client, monkeypatch):
    """App client with the model and palette pipeline replaced by recorders."""
    main.limiter.reset()
    calls = []

    def fake(name):
        def run(*args, **kwargs):
            calls.append(name)
            return _jpeg((10, 10))
        return run

    monkeypatch.setattr(main, "run_nst", fake("run_nst"))
    monkeypatch.setattr(main, "color_palette_transfer", fake("palette"))
    monkeypatch.setattr(main, "run_interpolation_gif", lambda *a, **k: b"GIF89a")
    _app_client.calls = calls
    yield _app_client
    assert main._inflight_jobs == 0, "a job slot leaked"


def _transfer(client, files=None, data=None, path="/api/transfer"):
    files = {"content_image": ("c.jpg", _jpeg(), "image/jpeg"), **(files or {})}
    return client.post(path, data={"style_preset": "starry_night", **(data or {})}, files=files)


# ── Pixel budgets (decompression bombs) ────────────────────────────────────────

@pytest.fixture(scope="module")
def bomb_png():
    # 11000 x 11000 = 121 MP: inside Pillow's "warning only" range (64–128 MP),
    # a few KB on the wire, ~480 MB to decode and convert.
    return _png((11000, 11000), mode="1")


@pytest.mark.parametrize("path", ["/api/transfer", "/api/palette-transfer"])
def test_pixel_bomb_content_rejected_before_decoding(client, bomb_png, path):
    assert len(bomb_png) < 64 * 1024
    r = _transfer(client, files={"content_image": ("b.png", bomb_png, "image/png")}, path=path)
    assert r.status_code == 413 and "too many pixels" in r.json()["detail"]
    assert client.calls == []


def test_pixel_bomb_mask_and_style_rejected(client, bomb_png):
    r = _transfer(client, files={"mask_image": ("m.png", bomb_png, "image/png")})
    assert r.status_code == 413 and r.json()["detail"].startswith("mask_image")
    r = client.post("/api/transfer", files={"content_image": ("c.jpg", _jpeg(), "image/jpeg"),
                                            "style_image": ("s.png", bomb_png, "image/png")})
    assert r.status_code == 413 and r.json()["detail"].startswith("style_image")
    assert client.calls == []


def test_over_pillow_hard_limit_is_413_not_400(client):
    huge = _png((12000, 12000), mode="1")  # 144 MP: Pillow raises DecompressionBombError
    r = _transfer(client, files={"content_image": ("b.png", huge, "image/png")})
    assert r.status_code == 413 and client.calls == []


def test_jpeg_gets_the_larger_budget(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_PIXELS", 1000)  # 96 x 64 = 6144 px
    r = _transfer(client, files={"content_image": ("c.png", _png((96, 64)), "image/png")})
    assert r.status_code == 413
    assert _transfer(client).status_code == 200  # same size as JPEG
    monkeypatch.setattr(main, "MAX_UPLOAD_PIXELS_JPEG", 1000)
    assert _transfer(client).status_code == 413


def test_describe_swatch_refuses_huge_provider_image():
    import base64
    with pytest.raises(ds.DescribeUnavailable):
        ds._to_swatch_jpeg(base64.b64encode(_png((5000, 5000), mode="1")).decode())


# ── Request size and form parts ────────────────────────────────────────────────

def test_declared_content_length_over_cap_is_413(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_REQUEST_BYTES", 50_000)
    r = _transfer(client, files={"style_image": ("s.bin", b"x" * 60_000, "image/jpeg")})
    assert r.status_code == 413 and r.json()["detail"] == "Request is too large"
    # Same headers as any other response.
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "content-security-policy" in r.headers
    assert client.calls == []


def test_chunked_body_over_cap_is_413(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_REQUEST_BYTES", 50_000)
    boundary = "pictura-test"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"style_preset\"\r\n\r\nstarry_night\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"content_image\"; filename=\"c.jpg\"\r\n"
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + b"x" * 80_000 + f"\r\n--{boundary}--\r\n".encode()

    def chunks():  # no Content-Length: httpx sends Transfer-Encoding: chunked
        for i in range(0, len(body), 8192):
            yield body[i:i + 8192]

    r = client.post("/api/transfer", content=chunks(),
                    headers={"content-type": f"multipart/form-data; boundary={boundary}"})
    assert "content-length" not in r.request.headers
    assert r.status_code == 413 and client.calls == []


def test_get_and_non_api_posts_untouched(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_REQUEST_BYTES", 10)
    assert client.get("/api/health").status_code == 200


def test_too_many_file_parts_rejected(client):
    files = [("content_image", ("c.jpg", _jpeg(), "image/jpeg"))]
    files += [(f"junk{i}", (f"j{i}.bin", b"x" * 100, "application/octet-stream")) for i in range(5)]
    r = client.post("/api/transfer", data={"style_preset": "starry_night"}, files=files)
    assert r.status_code == 400 and "Too many files" in r.json()["detail"]
    assert client.calls == []


def test_too_many_fields_rejected(client):
    data = {"style_preset": "starry_night", **{f"f{i}": "1" for i in range(30)}}
    r = client.post("/api/transfer", data=data, files={"content_image": ("c.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 400 and "Too many fields" in r.json()["detail"]


def test_all_four_uploads_still_accepted(client):
    r = _transfer(client, files={
        "style_image": ("s.jpg", _jpeg(), "image/jpeg"),
        "style_image_2": ("s2.jpg", _jpeg(), "image/jpeg"),
        "mask_image": ("m.png", _png((96, 64), mode="L"), "image/png"),
    }, data={"style_mix_ratio": "0.3", "intensity": "0.5"})
    assert r.status_code == 200 and client.calls == ["run_nst"]


# ── Queue guard runs before any upload is read ─────────────────────────────────

@pytest.mark.parametrize("path", ["/api/transfer", "/api/palette-transfer", "/api/interpolate"])
def test_queue_full_is_429_without_reading_uploads(client, monkeypatch, path):
    async def must_not_read(*args, **kwargs):
        raise AssertionError("upload read while the queue was full")

    monkeypatch.setattr(main, "read_image_upload", must_not_read)
    monkeypatch.setattr(main, "_inflight_jobs", main.MAX_PENDING_JOBS)
    r = _transfer(client, path=path)
    assert r.status_code == 429
    monkeypatch.setattr(main, "_inflight_jobs", 0)


@pytest.mark.parametrize("path", ["/api/transfer", "/api/palette-transfer", "/api/interpolate"])
def test_slot_released_when_upload_is_rejected(client, path):
    r = _transfer(client, files={"content_image": ("x.jpg", b"not an image", "image/jpeg")}, path=path)
    assert r.status_code == 400
    r = _transfer(client, data={"style_preset": "la_muse"}, path=path)
    assert r.status_code == 404
    # The `client` fixture asserts no slot leaked.


# ── text_prompt is capped like /api/describe-style ─────────────────────────────

@pytest.mark.parametrize("path", ["/api/transfer", "/api/palette-transfer"])
def test_long_text_prompt_rejected_before_matching(client, monkeypatch, path):
    matched = []
    monkeypatch.setattr(main, "resolve_style_from_text", lambda p: matched.append(p))
    files = {"content_image": ("c.jpg", _jpeg(), "image/jpeg")}
    r = client.post(path, data={"text_prompt": "munch " + "a" * 195}, files=files)
    assert r.status_code == 400 and "1 to 200 characters" in r.json()["detail"]
    r = client.post(path, data={"text_prompt": "a " * 30_000}, files=files)
    assert r.status_code == 400
    assert matched == [] and client.calls == []


def test_text_prompt_is_cleaned_and_still_matches(client):
    r = client.post("/api/transfer", data={"text_prompt": "  picabia\x00 geometric "},
                    files={"content_image": ("c.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["resolved_style_key"] == "udnie"


# ── Transfers are not stored on the server ─────────────────────────────────────

@pytest.mark.parametrize("output", ["final", "raw"])
def test_transfer_result_not_persisted(client, output):
    before_files = set(main.OUTPUT_DIR.iterdir())
    before_jobs = dict(main.jobs)
    r = _transfer(client, data={"output": output})
    assert r.status_code == 200 and r.json()["result"]
    assert "job_id" not in r.json()
    assert set(main.OUTPUT_DIR.iterdir()) == before_files
    assert main.jobs == before_jobs


def test_palette_result_not_persisted(client):
    before_files = set(main.OUTPUT_DIR.iterdir())
    before_jobs = dict(main.jobs)
    assert _transfer(client, path="/api/palette-transfer").status_code == 200
    assert set(main.OUTPUT_DIR.iterdir()) == before_files and main.jobs == before_jobs


# ── uvicorn entry point: forwarded headers trusted only from known proxies ─────

def test_forwarded_ips_default_is_loopback(monkeypatch):
    monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)
    opts = main.uvicorn_options()
    assert opts["proxy_headers"] is True and opts["forwarded_allow_ips"] == "127.0.0.1"
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "10.0.0.5")
    assert main.uvicorn_options()["forwarded_allow_ips"] == "10.0.0.5"
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "  ")
    assert main.uvicorn_options()["forwarded_allow_ips"] == "127.0.0.1"
