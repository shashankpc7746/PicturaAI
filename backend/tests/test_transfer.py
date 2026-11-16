"""/api/transfer: intensity alias, output=raw, preset resolution; run_nst raw path."""
import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
from conftest import STYLE_IMAGES_DIR


def _jpeg(size=(96, 64), color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def _png_mask(size=(96, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("L", size, 255).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def _app_client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def calls(_app_client, monkeypatch):
    """Replace the model call with a recorder (these tests check the HTTP layer)."""
    main.limiter.reset()
    seen = []

    def fake_run_nst(content_bytes, style_bytes, **kwargs):
        seen.append({"content": content_bytes, "style": style_bytes, **kwargs})
        return _jpeg((10, 10))

    monkeypatch.setattr(main, "run_nst", fake_run_nst)
    return seen


def _post(client, data=None, files=None):
    files = {"content_image": ("c.jpg", _jpeg(), "image/jpeg"), **(files or {})}
    return client.post("/api/transfer", data={"style_preset": "starry_night", **(data or {})}, files=files)


def test_defaults_unchanged(_app_client, calls):
    r = _post(_app_client)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done" and body["output"] == "final"
    assert Image.open(io.BytesIO(base64.b64decode(body["result"]))).size == (10, 10)
    (call,) = calls
    assert call["style_weight"] == pytest.approx(1e-2) and call["raw"] is False
    assert call["style"] == (STYLE_IMAGES_DIR / "starry_night.jpg").read_bytes()


def test_intensity_is_an_alias_and_wins(_app_client, calls):
    _post(_app_client, {"intensity": "0.75"})
    _post(_app_client, {"intensity": "0.3", "style_weight": "0.9"})
    _post(_app_client, {"style_weight": "0.6"})
    assert [c["style_weight"] for c in calls] == [pytest.approx(0.75), pytest.approx(0.3), pytest.approx(0.6)]


@pytest.mark.parametrize("value", ["-0.1", "1.5", "abc"])
def test_intensity_out_of_range_rejected(_app_client, calls, value):
    assert _post(_app_client, {"intensity": value}).status_code == 422
    assert calls == []


def test_raw_output_skips_mask(_app_client, calls):
    r = _post(_app_client, {"output": "raw", "intensity": "0.5"},
              files={"mask_image": ("m.png", _png_mask(), "image/png")})
    assert r.status_code == 200 and r.json()["output"] == "raw"
    assert calls[0]["raw"] is True and calls[0]["mask_bytes"] is None


def test_final_output_passes_mask(_app_client, calls):
    _post(_app_client, files={"mask_image": ("m.png", _png_mask(), "image/png")})
    assert calls[0]["mask_bytes"] is not None and calls[0]["raw"] is False


def test_unknown_output_rejected(_app_client, calls):
    assert _post(_app_client, {"output": "png"}).status_code == 422


def test_legacy_fields_still_accepted(_app_client, calls):
    r = _post(_app_client, {"content_weight": "1", "tv_weight": "2", "num_steps": "10", "learning_rate": "0.1"})
    assert r.status_code == 200


def test_text_prompt_resolves_with_catalogue_title(_app_client, calls):
    r = _app_client.post("/api/transfer", data={"text_prompt": "picabia geometric"},
                         files={"content_image": ("c.jpg", _jpeg(), "image/jpeg")})
    body = r.json()
    assert body["resolved_style_key"] == "udnie" and body["resolved_style_name"] == "Udnie"
    r = _app_client.post("/api/transfer", data={"text_prompt": "neon cyberpunk"},
                         files={"content_image": ("c.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 400


def test_unknown_preset_404(_app_client, calls):
    assert _post(_app_client, {"style_preset": "la_muse"}).status_code == 404


def test_second_preset_from_catalogue(_app_client, calls):
    _post(_app_client, {"style_preset_2": "the_scream", "style_mix_ratio": "0.4"})
    assert calls[0]["style_bytes_2"] == (STYLE_IMAGES_DIR / "the_scream.jpg").read_bytes()


# ── Real model: the raw pipeline in nst_engine ─────────────────────────────────

def test_run_nst_raw_is_stylized_image_at_output_size():
    from nst_engine import OUTPUT_MAX_DIM, run_nst

    rng = np.random.default_rng(0)
    arr = (rng.random((240, 400, 3)) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    content = buf.getvalue()
    style = (STYLE_IMAGES_DIR / "starry_night.jpg").read_bytes()

    raw = run_nst(content, style, raw=True)
    raw_img = Image.open(io.BytesIO(raw))
    assert raw_img.format == "JPEG"
    expected_w = min(400, OUTPUT_MAX_DIM)
    assert raw_img.size == (expected_w, round(240 * expected_w / 400))

    # Raw ignores intensity and mask entirely.
    assert run_nst(content, style, raw=True, style_weight=0.0, mask_bytes=_png_mask()) == raw
    # ...and differs from the finished image.
    final = run_nst(content, style, style_weight=1.0)
    assert final != raw and Image.open(io.BytesIO(final)).size == raw_img.size
