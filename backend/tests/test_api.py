"""HTTP-level tests that don't need the neural model (palette path, jobs, limits)."""
import base64
import io
import os
import uuid

os.environ["RATE_LIMIT_PALETTE"] = "3/minute"
os.environ["CLIENT_IP_HEADER"] = "cf-connecting-ip"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import main  # noqa: E402


@pytest.fixture(scope="module")
def _app_client():
    # One lifespan per module: shutdown closes the shared worker executor.
    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def client(_app_client):
    main.limiter.reset()
    return _app_client


def _jpeg(size=(64, 48), color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def _palette(client, headers=None):
    return client.post(
        "/api/palette-transfer",
        files={"content_image": ("c.jpg", _jpeg(), "image/jpeg")},
        data={"style_preset": "space"},
        headers=headers or {},
    )


def test_favicon_served_at_root(client):
    r = client.get("/favicon.ico")
    assert r.status_code == 200 and r.headers["content-type"] == "image/x-icon"


def test_palette_transfer_returns_image(client):
    r = _palette(client, {"cf-connecting-ip": "203.0.113.1"})
    assert r.status_code == 200
    img = Image.open(io.BytesIO(base64.b64decode(r.json()["result"])))
    assert img.size == (64, 48)


def test_rejects_non_image(client):
    r = client.post(
        "/api/palette-transfer",
        files={"content_image": ("x.jpg", b"not an image", "image/jpeg")},
        data={"style_preset": "space"},
    )
    assert r.status_code == 400


def test_spoofed_x_forwarded_for_does_not_reset_limit(client):
    ip = {"cf-connecting-ip": "203.0.113.7"}
    for i in range(3):
        assert _palette(client, {**ip, "x-forwarded-for": f"10.0.0.{i}"}).status_code == 200
    assert _palette(client, {**ip, "x-forwarded-for": "10.9.9.9"}).status_code == 429
    # A genuinely different visitor still gets through.
    assert _palette(client, {"cf-connecting-ip": "203.0.113.8"}).status_code == 200


def test_job_result_read_from_disk_and_path_not_leaked(client, tmp_path):
    job_id = str(uuid.uuid4())
    gif = tmp_path / f"{job_id}.gif"
    gif.write_bytes(b"GIF89a-test")
    main.jobs[job_id] = {"id": job_id, "status": "done", "progress": 100,
                         "result_path": str(gif), "result_type": "gif"}
    try:
        body = client.get(f"/api/jobs/{job_id}").json()
        assert base64.b64decode(body["result"]) == b"GIF89a-test"
        assert "result_path" not in body
    finally:
        main.jobs.pop(job_id, None)
