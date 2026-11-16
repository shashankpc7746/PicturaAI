"""
"Describe it" — generate a style swatch from text with Cloudflare Workers AI
============================================================================
Optional: only active when ``CLOUDFLARE_ACCOUNT_ID`` and ``CLOUDFLARE_API_TOKEN``
are set. The caller falls back to the keyword matcher whenever this module
raises (not configured, daily cap reached, quota used up, timeout, bad reply).

REST call (https://developers.cloudflare.com/workers-ai/models/flux-1-schnell/):
    POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell
    Authorization: Bearer <token>
    {"prompt": "...", "steps": 4}
    -> {"result": {"image": "<base64 JPEG>"}, "success": true, "errors": [], "messages": []}
Free-allocation exhaustion is HTTP 429 with error code 3036 ("Account limited").

The token is only ever placed in the Authorization header; it is never logged.
No new dependency: plain urllib, run by the caller in a worker thread.
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
import re
import threading
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Optional

from PIL import Image  # pyre-ignore[21]

logger = logging.getLogger("describe_style")

MODEL = "@cf/black-forest-labs/flux-1-schnell"
API_ROOT = "https://api.cloudflare.com/client/v4/accounts"
PROMPT_TEMPLATE = "a painterly texture study in the style of: {prompt}"
PROMPT_MAX_CHARS = 200
SWATCH_MAX_DIM = 512
MAX_RESPONSE_BYTES = 12 * 1024 * 1024
QUOTA_ERROR_CODE = 3036  # "You have used up your daily free allocation of 10,000 neurons."
USER_AGENT = "PicturaAI/2.2 (+describe-style)"

_ACCOUNT_RE = re.compile(r"^[A-Za-z0-9]{8,64}$")
_DROP_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn"}


class DescribeUnavailable(Exception):
    """Generation is not possible right now; ``reason`` says why."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason  # "not_configured" | "daily_limit" | "quota" | "failed"


# ── Prompt hygiene ─────────────────────────────────────────────────────────────

def clean_prompt(raw: Optional[str]) -> str:
    """Drop control/format characters and collapse whitespace."""
    if not raw:
        return ""
    kept = []
    for ch in raw:
        if ch in "\t\n\r":
            kept.append(" ")
        elif unicodedata.category(ch) not in _DROP_CATEGORIES:
            kept.append(ch)
    return " ".join("".join(kept).split())


# ── Daily cap (in memory; resets at 00:00 UTC and on restart) ──────────────────

class DailyQuota:
    def __init__(self, limit: int, today: Callable[[], date] | None = None):
        self.limit = max(0, int(limit))
        self._today = today or (lambda: datetime.now(timezone.utc).date())
        self._day: Optional[date] = None
        self._used = 0
        self._lock = threading.Lock()

    def _roll(self) -> None:
        day = self._today()
        if day != self._day:
            self._day = day
            self._used = 0

    def try_acquire(self) -> bool:
        with self._lock:
            self._roll()
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    def exhaust(self) -> None:
        """Provider said the allocation is used up: stop calling it until tomorrow."""
        with self._lock:
            self._roll()
            self._used = max(self._used, self.limit)

    @property
    def remaining(self) -> int:
        with self._lock:
            self._roll()
            return max(0, self.limit - self._used)


# ── Workers AI client ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorkersAIConfig:
    account_id: str = ""
    api_token: str = ""
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "WorkersAIConfig":
        return cls(
            account_id=os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip(),
            api_token=os.environ.get("CLOUDFLARE_API_TOKEN", "").strip(),
            timeout=float(os.environ.get("DESCRIBE_TIMEOUT_SECONDS", "30")),
        )

    @property
    def configured(self) -> bool:
        return bool(self.account_id and self.api_token and _ACCOUNT_RE.match(self.account_id))

    @property
    def url(self) -> str:
        return f"{API_ROOT}/{self.account_id}/ai/run/{MODEL}"


def _error_codes(body: bytes) -> list[int]:
    try:
        payload = json.loads(body.decode("utf-8", "replace"))
        return [int(e.get("code")) for e in payload.get("errors", []) if isinstance(e, dict) and "code" in e]
    except (ValueError, TypeError, AttributeError):
        return []


def _to_swatch_jpeg(image_b64: str) -> bytes:
    """Decode the provider's image, verify it, and re-encode a small clean JPEG."""
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        raise DescribeUnavailable("failed", "image field is not valid base64") from None
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(raw)) as img:
            img = img.convert("RGB")
            img.thumbnail((SWATCH_MAX_DIM, SWATCH_MAX_DIM))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=90)
            return buf.getvalue()
    except Exception:
        raise DescribeUnavailable("failed", "image could not be decoded") from None


def generate_swatch(
    cfg: WorkersAIConfig,
    prompt: str,
    urlopen: Callable = urllib.request.urlopen,
) -> bytes:
    """Generate a style swatch for an already-cleaned prompt. Blocking: run off the event loop.

    Returns JPEG bytes (long side ≤ SWATCH_MAX_DIM). Raises DescribeUnavailable.
    """
    if not cfg.configured:
        raise DescribeUnavailable("not_configured")

    body = json.dumps({"prompt": PROMPT_TEMPLATE.format(prompt=prompt), "steps": 4}).encode("utf-8")
    req = urllib.request.Request(
        cfg.url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {cfg.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urlopen(req, timeout=cfg.timeout) as resp:
            data = resp.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        codes = _error_codes(exc.read(64 * 1024) if exc.fp else b"")
        if exc.code == 429 and QUOTA_ERROR_CODE in codes:
            logger.warning("Workers AI daily allocation used up (HTTP 429, code %s)", QUOTA_ERROR_CODE)
            raise DescribeUnavailable("quota") from None
        logger.warning("Workers AI request failed: HTTP %s, error codes %s", exc.code, codes or "none")
        raise DescribeUnavailable("failed", f"HTTP {exc.code}") from None
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        logger.warning("Workers AI request did not complete: %s", type(reason).__name__)
        raise DescribeUnavailable("failed", "network error or timeout") from None

    if len(data) > MAX_RESPONSE_BYTES:
        raise DescribeUnavailable("failed", "response too large")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise DescribeUnavailable("failed", "response is not JSON") from None
    if not isinstance(payload, dict) or not payload.get("success", False):
        logger.warning("Workers AI returned success=false, error codes %s", _error_codes(data) or "none")
        raise DescribeUnavailable("failed", "provider reported an error")
    result = payload.get("result")
    image_b64 = result.get("image") if isinstance(result, dict) else None
    if not isinstance(image_b64, str) or not image_b64:
        raise DescribeUnavailable("failed", "response has no image")
    return _to_swatch_jpeg(image_b64)
