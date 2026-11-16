"""
Style preset catalogue — PicturaAI
===================================
Loads the built-in style presets from ``data/styles.json`` (schema in
``docs/revamp/BUILD-PLAN.md`` §4) and matches free text to the closest preset.

The file is validated strictly at startup: a typo in the catalogue should stop
the deploy with a clear message, not surface later as a broken style card.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

_KEY_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_WORD_RE = re.compile(r"[a-z0-9]+")
# A web asset path relative to frontend/: no scheme, no leading slash, no "..".
_ASSET_RE = re.compile(r"^(?!/)(?!.*\.\.)[A-Za-z0-9_./-]+$")


class StyleCatalogError(ValueError):
    """The style catalogue file is missing or invalid."""


@dataclass(frozen=True)
class StylePreset:
    key: str
    slug: str
    title: str
    artist: str
    file: str
    year: str = ""
    medium: str = ""
    description: str = ""
    tags: frozenset = field(default_factory=frozenset)
    image: str = ""
    card: str = ""
    thumb: str = ""
    aspect: float = 1.0
    credit: str = ""
    source_url: str = ""
    license: str = ""

    def searchable_text(self) -> str:
        """Lower-case text the keyword matcher searches besides the tags."""
        return " ".join([
            self.key.replace("_", " "),
            self.title.lower(),
            self.artist.lower(),
            self.description.lower(),
        ])


@dataclass(frozen=True)
class StyleCatalog:
    presets: Dict[str, StylePreset]
    images_dir: Path
    source: Path

    def __contains__(self, key: object) -> bool:
        return key in self.presets

    def __iter__(self):
        return iter(self.presets.values())

    def __len__(self) -> int:
        return len(self.presets)

    def get(self, key: Optional[str]) -> Optional[StylePreset]:
        if not key:
            return None
        return self.presets.get(key)

    def image_path(self, key: Optional[str]) -> Optional[Path]:
        preset = self.get(key)
        if preset is None:
            return None
        return self.images_dir / preset.file

    def match_text(self, prompt: str) -> Optional[StylePreset]:
        """Map a free-text description to the best matching preset, or None.

        Scoring (unchanged from the original matcher): +4 for each prompt word
        that is one of the style's tags, +2 for each word found in its key,
        title, artist or description, +4 if the key itself appears as a phrase.
        Ties keep the earlier preset in catalogue order.
        """
        text = (prompt or "").strip().lower()
        if not text:
            return None
        tokens = [t for t in _WORD_RE.findall(text) if len(t) > 1]
        if not tokens:
            return None

        best: Optional[StylePreset] = None
        best_score = 0
        for preset in self.presets.values():
            searchable = preset.searchable_text()
            score = 0
            for token in tokens:
                if token in preset.tags:
                    score += 4
                if token in searchable:
                    score += 2
            if preset.key.replace("_", " ") in text:
                score += 4
            if score > best_score:
                best_score = score
                best = preset
        return best if best_score > 0 else None


# ── Loading & validation ───────────────────────────────────────────────────────

def _req_str(entry: Dict[str, Any], name: str, where: str) -> str:
    value = entry.get(name)
    if not isinstance(value, str) or not value.strip():
        raise StyleCatalogError(f"{where}: '{name}' must be a non-empty string")
    return value.strip()


def _opt_str(entry: Dict[str, Any], name: str, where: str) -> str:
    value = entry.get(name, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise StyleCatalogError(f"{where}: '{name}' must be a string")
    return value.strip()


def _asset(entry: Dict[str, Any], name: str, where: str) -> str:
    value = _opt_str(entry, name, where)
    if value and not _ASSET_RE.match(value):
        raise StyleCatalogError(f"{where}: '{name}' must be a relative path inside frontend/ (got {value!r})")
    return value


def _tags(entry: Dict[str, Any], where: str) -> frozenset:
    raw = entry.get("tags", [])
    if not isinstance(raw, list) or not all(isinstance(t, str) for t in raw):
        raise StyleCatalogError(f"{where}: 'tags' must be a list of strings")
    return frozenset(t.strip().lower() for t in raw if t.strip())


def _parse_entry(entry: Any, index: int, images_dir: Path) -> StylePreset:
    where = f"styles[{index}]"
    if not isinstance(entry, dict):
        raise StyleCatalogError(f"{where}: must be an object")
    key = _req_str(entry, "key", where)
    where = f"style '{key}'"
    if not _KEY_RE.match(key):
        raise StyleCatalogError(f"{where}: key must be snake_case (a-z, 0-9, _)")
    slug = _req_str(entry, "slug", where)
    if not _SLUG_RE.match(slug):
        raise StyleCatalogError(f"{where}: slug must be kebab-case (a-z, 0-9, -)")

    file_name = _req_str(entry, "file", where)
    if Path(file_name).name != file_name or file_name in (".", ".."):
        raise StyleCatalogError(f"{where}: 'file' must be a plain file name inside {images_dir}")
    if not (images_dir / file_name).is_file():
        raise StyleCatalogError(f"{where}: style image {file_name!r} not found in {images_dir}")

    aspect = entry.get("aspect", 1.0)
    if isinstance(aspect, bool) or not isinstance(aspect, (int, float)) or not (0.05 <= aspect <= 20):
        raise StyleCatalogError(f"{where}: 'aspect' must be a number between 0.05 and 20")

    return StylePreset(
        key=key,
        slug=slug,
        title=_req_str(entry, "title", where),
        artist=_req_str(entry, "artist", where),
        file=file_name,
        year=_opt_str(entry, "year", where),
        medium=_opt_str(entry, "medium", where),
        description=_opt_str(entry, "description", where),
        tags=_tags(entry, where),
        image=_asset(entry, "image", where),
        card=_asset(entry, "card", where),
        thumb=_asset(entry, "thumb", where),
        aspect=float(aspect),
        credit=_opt_str(entry, "credit", where),
        source_url=_opt_str(entry, "source_url", where),
        license=_opt_str(entry, "license", where),
    )


def parse_catalog(data: Any, images_dir: Path, source: Path) -> StyleCatalog:
    if not isinstance(data, dict):
        raise StyleCatalogError("top level must be an object with a 'styles' list")
    if data.get("version") != 1:
        raise StyleCatalogError(f"unsupported 'version' {data.get('version')!r} (expected 1)")
    entries = data.get("styles")
    if not isinstance(entries, list) or not entries:
        raise StyleCatalogError("'styles' must be a non-empty list")

    presets: Dict[str, StylePreset] = {}
    slugs: set[str] = set()
    for i, entry in enumerate(entries):
        preset = _parse_entry(entry, i, images_dir)
        if preset.key in presets:
            raise StyleCatalogError(f"duplicate style key {preset.key!r}")
        if preset.slug in slugs:
            raise StyleCatalogError(f"duplicate style slug {preset.slug!r}")
        presets[preset.key] = preset
        slugs.add(preset.slug)
    return StyleCatalog(presets=presets, images_dir=images_dir, source=source)


def load_catalog(path: Path, images_dir: Path) -> StyleCatalog:
    """Read and validate the catalogue. Raises StyleCatalogError with a clear reason."""
    path = Path(path)
    if not path.is_file():
        raise StyleCatalogError(f"style catalogue not found at {path} (set STYLES_FILE)")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise StyleCatalogError(f"could not read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise StyleCatalogError(f"{path} is not valid JSON: {exc}") from None
    try:
        return parse_catalog(data, Path(images_dir), path)
    except StyleCatalogError as exc:
        raise StyleCatalogError(f"{path}: {exc}") from None
