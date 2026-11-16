"""Cache-Control, compression and content types for the static frontend under /app."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main

FRONTEND = Path(main.FRONTEND_DIR)
IMMUTABLE = "public, max-age=31536000, immutable"
DAY = "public, max-age=86400"
GZIP = {"Accept-Encoding": "gzip"}


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


def _url(path: Path) -> str:
    return "/app/" + path.relative_to(FRONTEND).as_posix()


def _first(pattern: str) -> str:
    found = sorted(FRONTEND.glob(pattern))
    if not found:
        pytest.skip(f"no file matches frontend/{pattern}")
    return _url(found[0])


@pytest.mark.parametrize("path", ["/app/", "/app/index.html", "/app/assets/js/main.js",
                                  "/app/assets/js/engine/worker.js", "/app/assets/css/app.css",
                                  "/app/assets/icons/icons.svg", "/app/config.js"])
def test_unversioned_files_are_revalidated(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-cache"


def test_modules_revalidate_with_etag(client):
    first = client.get("/app/assets/js/main.js")
    etag = first.headers["etag"]
    again = client.get("/app/assets/js/main.js", headers={"If-None-Match": etag})
    assert again.status_code == 304
    assert again.headers["cache-control"] == "no-cache"


def test_model_files_are_immutable_but_index_files_revalidate(client):
    shard = _first("models/arbitrary-stylization-v1-256/transformer/*.bin")
    assert client.get(shard).headers["cache-control"] == IMMUTABLE
    assert client.get(_first("models/style-vectors/*.bin")).headers["cache-control"] == IMMUTABLE
    for index in ("/app/models/arbitrary-stylization-v1-256/manifest.json", "/app/models/style-vectors.json"):
        r = client.get(index)
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"


def test_vendored_tfjs_is_immutable(client):
    r = client.get("/app/vendor/tfjs/4.22.0/tf-core.es2017.min.js")
    assert r.status_code == 200
    assert r.headers["cache-control"] == IMMUTABLE
    assert r.headers["content-type"].startswith("text/javascript")


def test_artwork_and_samples_cached_for_a_day(client):
    card = client.get(_first("assets/styles/*.webp"))
    assert card.headers["cache-control"] == DAY
    assert card.headers["content-type"] == "image/webp"
    assert client.get("/app/assets/samples/dog.jpg").headers["cache-control"] == DAY


def test_missing_files_get_no_long_cache(client):
    r = client.get("/app/models/arbitrary-stylization-v1-256/transformer/nope.bin")
    assert r.status_code == 404
    assert r.headers.get("cache-control") != IMMUTABLE


def test_text_is_gzipped(client):
    for path in ("/app/models/arbitrary-stylization-v1-256/transformer/model.json",
                 "/app/vendor/tfjs/4.22.0/tf-backend-webgl.es2017.min.js",
                 "/app/assets/js/ui/studio.js", "/app/assets/css/app.css", "/app/"):
        r = client.get(path, headers=GZIP)
        assert r.status_code == 200, path
        assert r.headers.get("content-encoding") == "gzip", path
        assert "accept-encoding" in r.headers.get("vary", "").lower(), path
    # The client decodes transparently: the JSON is intact.
    assert "modelTopology" in client.get(
        "/app/models/arbitrary-stylization-v1-256/transformer/model.json", headers=GZIP).json()


def test_compressed_formats_are_not_gzipped(client):
    for path in (_first("models/arbitrary-stylization-v1-256/transformer/*.bin"),
                 "/app/models/deeplab_v3/deeplab_v3.tflite",
                 _first("assets/styles/*.webp"), "/app/assets/samples/dog.jpg"):
        r = client.get(path, headers=GZIP)
        assert r.status_code == 200, path
        assert "content-encoding" not in r.headers, path
    assert client.get("/app/models/deeplab_v3/deeplab_v3.tflite").headers["content-type"] == "application/octet-stream"


def test_small_responses_are_not_gzipped(client):
    r = client.get("/api/health", headers=GZIP)
    assert "content-encoding" not in r.headers


def test_web_app_manifest_type(client):
    path = FRONTEND / "manifest.webmanifest"
    if not path.is_file():
        pytest.skip("no web app manifest")
    r = client.get("/app/manifest.webmanifest")
    assert r.headers["content-type"].startswith("application/manifest+json")
    assert r.headers["cache-control"] == "no-cache"
