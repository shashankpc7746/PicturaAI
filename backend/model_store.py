"""
Locate (and if needed, download) the Magenta arbitrary-style SavedModel.

PicturaAI used to load the model through ``tensorflow_hub``. That library only
downloads and unpacks the archive and then calls ``tf.saved_model.load``,
but it drags in ``tf-keras`` and needs ``pkg_resources`` (removed from recent
setuptools and absent from Python 3.12 images). Loading the SavedModel directly
gives byte-identical results with fewer dependencies.

The cache layout matches what tensorflow_hub produced
(``$TFHUB_CACHE_DIR/<sha1 of the handle>/``), so existing caches keep working.

Run ``python backend/model_store.py`` to pre-download (used by the Dockerfile).
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger("model_store")

MODEL_HANDLE = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
# tfhub.dev redirects this to the model archive hosted on Kaggle.
DOWNLOAD_URL = MODEL_HANDLE + "?tf-hub-format=compressed"
USER_AGENT = "PicturaAI-model-fetch/1.0"


def cache_root() -> Path:
    default = Path(__file__).resolve().parent.parent / ".tfhub_cache"
    return Path(os.environ.get("TFHUB_CACHE_DIR", str(default)))


def model_dir() -> Path:
    """Directory holding saved_model.pb for the style-transfer model."""
    override = os.environ.get("NST_MODEL_DIR")
    if override:
        return Path(override)
    return cache_root() / hashlib.sha1(MODEL_HANDLE.encode("utf8")).hexdigest()


def _is_complete(path: Path) -> bool:
    return (path / "saved_model.pb").is_file() and (path / "variables").is_dir()


def ensure_model() -> Path:
    """Return the model directory, downloading and unpacking it if missing."""
    target = model_dir()
    if _is_complete(target):
        return target

    logger.info("Downloading style-transfer model (~90 MB) to %s …", target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        archive = Path(tmp) / "model.tar.gz"
        req = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=300) as resp, open(archive, "wb") as fh:
            shutil.copyfileobj(resp, fh)

        unpacked = Path(tmp) / "model"
        unpacked.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            # "data" filter rejects absolute paths, links and traversal (PEP 706).
            tar.extractall(unpacked, filter="data")
        if not _is_complete(unpacked):
            raise RuntimeError("Downloaded model archive is missing saved_model.pb/variables")

        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(unpacked), str(target))

    (target.parent / f"{target.name}.descriptor.txt").write_text(f"Module: {MODEL_HANDLE}\n")
    logger.info("Model ready at %s", target)
    return target


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(ensure_model())
