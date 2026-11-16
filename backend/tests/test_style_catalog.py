"""Loading/validating data/styles.json and the keyword matcher."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import BACKEND_DIR, FIXTURE_STYLES, STYLE_IMAGES_DIR
from style_catalog import StyleCatalogError, load_catalog, parse_catalog

REPO_ROOT = BACKEND_DIR.parent


def _write(tmp_path: Path, data) -> Path:
    path = tmp_path / "styles.json"
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")
    return path


def _catalog(tmp_path, data=None):
    return load_catalog(_write(tmp_path, data or FIXTURE_STYLES), STYLE_IMAGES_DIR)


def test_loads_fixture_in_order(tmp_path):
    cat = _catalog(tmp_path)
    assert [p.key for p in cat] == ["starry_night", "the_scream", "udnie"]
    starry = cat.get("starry_night")
    assert starry.title == "The Starry Night" and starry.aspect == 1.26
    assert "gogh" in starry.tags
    assert cat.image_path("udnie") == STYLE_IMAGES_DIR / "udnie.jpg"
    assert cat.image_path("nope") is None and cat.get(None) is None


def test_real_catalogue_is_valid_if_present():
    real = REPO_ROOT / "data" / "styles.json"
    if not real.exists():
        pytest.skip("data/styles.json not created yet")
    cat = load_catalog(real, STYLE_IMAGES_DIR)
    assert len(cat) >= 3


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(version=2), "version"),
    (lambda d: d.update(styles=[]), "non-empty"),
    (lambda d: d["styles"][0].pop("title"), "'title'"),
    (lambda d: d["styles"][0].update(key="Starry Night"), "snake_case"),
    (lambda d: d["styles"][0].update(slug="Van Gogh"), "kebab-case"),
    (lambda d: d["styles"][1].update(key="starry_night"), "duplicate style key"),
    (lambda d: d["styles"][1].update(slug="van-gogh-starry-night"), "duplicate style slug"),
    (lambda d: d["styles"][0].update(file="missing.jpg"), "not found"),
    (lambda d: d["styles"][0].update(file="../main.py"), "plain file name"),
    (lambda d: d["styles"][0].update(image="/etc/passwd"), "relative path"),
    (lambda d: d["styles"][0].update(card="../../secret.webp"), "relative path"),
    (lambda d: d["styles"][0].update(aspect="wide"), "aspect"),
    (lambda d: d["styles"][0].update(tags="starry"), "tags"),
])
def test_invalid_catalogue_is_rejected_with_reason(tmp_path, mutate, message):
    data = copy.deepcopy(FIXTURE_STYLES)
    mutate(data)
    with pytest.raises(StyleCatalogError, match=message):
        _catalog(tmp_path, data)


def test_missing_and_malformed_files(tmp_path):
    with pytest.raises(StyleCatalogError, match="not found"):
        load_catalog(tmp_path / "nope.json", STYLE_IMAGES_DIR)
    with pytest.raises(StyleCatalogError, match="not valid JSON"):
        load_catalog(_write(tmp_path, "{not json"), STYLE_IMAGES_DIR)


@pytest.mark.parametrize("prompt, expected", [
    ("starry swirling sky", "starry_night"),
    ("Van Gogh please", "starry_night"),
    ("something like munch", "the_scream"),
    ("geometric abstract", "udnie"),
    ("Blood-orange sky", "the_scream"),        # description words count
    ("FRANCIS PICABIA", "udnie"),              # artist, case-insensitive
    ("the scream", "the_scream"),              # key phrase bonus
])
def test_matcher_uses_tags_title_artist_description(tmp_path, prompt, expected):
    assert _catalog(tmp_path).match_text(prompt).key == expected


@pytest.mark.parametrize("prompt", ["", "   ", "x", "neon cyberpunk", "!!!"])
def test_matcher_returns_none_without_a_match(tmp_path, prompt):
    assert _catalog(tmp_path).match_text(prompt) is None


def test_matcher_tie_keeps_catalogue_order(tmp_path):
    data = copy.deepcopy(FIXTURE_STYLES)
    for s in data["styles"]:
        s["tags"] = ["shared"]
    assert _catalog(tmp_path, data).match_text("shared").key == "starry_night"


def test_parse_catalog_rejects_non_object():
    with pytest.raises(StyleCatalogError):
        parse_catalog(["not", "an", "object"], STYLE_IMAGES_DIR, Path("x"))


def test_app_refuses_to_start_without_a_catalogue(tmp_path):
    env = {**os.environ, "STYLES_FILE": str(tmp_path / "missing.json"), "TF_CPP_MIN_LOG_LEVEL": "3"}
    proc = subprocess.run(
        [sys.executable, "-c", "import main"],
        cwd=BACKEND_DIR, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode != 0
    assert "PicturaAI cannot start: style catalogue not found" in proc.stderr
