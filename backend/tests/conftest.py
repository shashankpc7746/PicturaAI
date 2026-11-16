import atexit
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Tests import backend modules directly (main.py, nst_engine.py).
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

STYLE_IMAGES_DIR = BACKEND_DIR.parent / "images" / "style_image"

# A small, fixed style catalogue so the tests don't depend on the real
# data/styles.json (which changes as presets are curated). It points at real
# images in images/style_image/.
FIXTURE_STYLES = {
    "version": 1,
    "styles": [
        {
            "key": "starry_night",
            "slug": "van-gogh-starry-night",
            "title": "The Starry Night",
            "artist": "Vincent van Gogh",
            "year": "1889",
            "medium": "Oil on canvas",
            "description": "Swirling night sky in deep blues and yellows.",
            "tags": ["starry", "night", "swirl", "van", "gogh", "sky", "cosmic"],
            "file": "starry_night.jpg",
            "image": "assets/styles/starry_night-512.jpg",
            "card": "assets/styles/starry_night-480.webp",
            "thumb": "assets/styles/starry_night-160.webp",
            "aspect": 1.26,
            "credit": "Vincent van Gogh, The Starry Night, 1889. Museum of Modern Art, New York.",
            "source_url": "https://commons.wikimedia.org/wiki/File:Starry_Night.jpg",
            "license": "Public domain",
        },
        {
            "key": "the_scream",
            "slug": "munch-the-scream",
            "title": "The Scream",
            "artist": "Edvard Munch",
            "year": "1893",
            "description": "Blood-orange sky and wavering lines.",
            "tags": ["scream", "munch", "expressionist", "orange", "anxiety"],
            "file": "the_scream.jpg",
            "image": "assets/styles/the_scream-512.jpg",
            "card": "assets/styles/the_scream-480.webp",
            "thumb": "assets/styles/the_scream-160.webp",
            "aspect": 0.81,
            "credit": "Edvard Munch, The Scream, 1893. National Museum, Oslo.",
            "license": "Public domain",
        },
        {
            "key": "udnie",
            "slug": "picabia-udnie",
            "title": "Udnie",
            "artist": "Francis Picabia",
            "year": "1913",
            "description": "Fragmented, interlocking planes close to abstraction.",
            "tags": ["udnie", "picabia", "cubism", "abstract", "geometric"],
            "file": "udnie.jpg",
            "image": "assets/styles/udnie-512.jpg",
            "card": "assets/styles/udnie-480.webp",
            "thumb": "assets/styles/udnie-160.webp",
            "aspect": 1.0,
            "credit": "Francis Picabia, Udnie, 1913. Centre Pompidou, Paris.",
            "license": "Public domain",
        },
    ],
}

_fixture_dir = Path(tempfile.mkdtemp(prefix="pictura-tests-"))
atexit.register(shutil.rmtree, _fixture_dir, True)
FIXTURE_STYLES_FILE = _fixture_dir / "styles.json"
FIXTURE_STYLES_FILE.write_text(json.dumps(FIXTURE_STYLES), encoding="utf-8")
# main.py reads this at import time.
os.environ["STYLES_FILE"] = str(FIXTURE_STYLES_FILE)

# Keep the suite deterministic regardless of the developer's shell: no real
# Cloudflare/Sentry calls, default site URL and docs on.
for _name in (
    "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "SENTRY_DSN", "SENTRY_DSN_FRONTEND",
    "CF_WEB_ANALYTICS_TOKEN", "PUBLIC_SITE_URL", "DESCRIBE_ENABLED", "ENABLE_API_DOCS",
    "CSP_REPORT_ONLY", "DESCRIBE_DAILY_LIMIT", "ALLOWED_ORIGINS",
):
    os.environ.pop(_name, None)
