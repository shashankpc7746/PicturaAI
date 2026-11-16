"""
FastAPI Backend — PicturaAI
============================
Endpoints:
  POST /api/transfer         – Stylize a photo (synchronous; base64 JPEG in the response, not stored)
  POST /api/palette-transfer – Colour-only transfer (synchronous, not stored)
  POST /api/interpolate      – Start a 0→100% style GIF job, returns job_id
  POST /api/describe-style   – Text → generated style swatch (Workers AI) or closest preset
  GET  /api/jobs/{job_id}    – Poll GIF job status & result
  WS   /ws/{job_id}          – WebSocket stream for real-time progress
  GET  /api/styles           – List built-in style presets (from data/styles.json)
  GET  /api/health           – Cheap liveness check
  GET  /api/result/{job_id}  – Download a finished GIF job's result
  DELETE /api/jobs/{job_id}  – Cancel / cleanup a GIF job
  GET  /app/config.js        – Public browser config (window.PICTURA_CONFIG)
  GET  /robots.txt, /sitemap.xml
"""

import os
import warnings
import logging as _logging

# Suppress TF noise BEFORE any TF import (nst_engine imports TF at module level)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = os.environ.get("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
_logging.getLogger("tensorflow").setLevel(_logging.ERROR)

import asyncio
import base64
import inspect
import json
import logging
import mimetypes
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlsplit, urlunsplit

from fastapi import (  # pyre-ignore[21]
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from slowapi import Limiter, _rate_limit_exceeded_handler  # pyre-ignore[21]
from slowapi.errors import RateLimitExceeded  # pyre-ignore[21]
from slowapi.util import get_remote_address  # pyre-ignore[21]
from fastapi.middleware.cors import CORSMiddleware  # pyre-ignore[21]
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response  # pyre-ignore[21]
from fastapi.routing import APIRoute  # pyre-ignore[21]
from fastapi.staticfiles import StaticFiles  # pyre-ignore[21]
from starlette.middleware.gzip import GZipMiddleware  # pyre-ignore[21]

try:
    from starlette.middleware.gzip import DEFAULT_EXCLUDED_CONTENT_TYPES  # pyre-ignore[21]
except ImportError:  # older Starlette: no per-type exclusions
    DEFAULT_EXCLUDED_CONTENT_TYPES = ()

from nst_engine import run_nst, pil_to_bytes, run_interpolation_gif, color_palette_transfer  # pyre-ignore[21]
from describe_style import (  # pyre-ignore[21]
    PROMPT_MAX_CHARS,
    DailyQuota,
    DescribeUnavailable,
    WorkersAIConfig,
    clean_prompt,
    generate_swatch,
)
from site_config import SiteConfig, env_bool, is_docs_path, is_https  # pyre-ignore[21]
from style_catalog import StyleCatalog, StyleCatalogError, StylePreset, load_catalog  # pyre-ignore[21]
from PIL import Image  # pyre-ignore[21]
import io

# Pillow 10+ moved LANCZOS to Image.Resampling
_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS  # type: ignore[attr-defined]

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)
logger = logging.getLogger("nst_api")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
UPLOAD_DIR  = BASE_DIR / "uploads"
OUTPUT_DIR  = BASE_DIR / "outputs"
STYLES_DIR  = BASE_DIR.parent / "images" / "style_image"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

for d in (UPLOAD_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Static file types, independent of the host's MIME registry (Windows can map
# .js to text/plain, and some Python builds don't know .webp or .tflite).
for _ext, _type in ((".js", "text/javascript"), (".mjs", "text/javascript"), (".webp", "image/webp"),
                    (".webmanifest", "application/manifest+json"), (".tflite", "application/octet-stream"),
                    (".bin", "application/octet-stream"), (".svg", "image/svg+xml")):
    mimetypes.add_type(_type, _ext)

# ── Optional error reporting (Sentry) ──────────────────────────────────────────
# Only imported when SENTRY_DSN is set, so the app runs (and stays lighter on
# the 512 MB tier) without it. No request bodies, PII or local variables are
# sent: uploads are people's photos.
SENTRY_DSN = os.environ.get("SENTRY_DSN", "").strip()


def _init_sentry() -> bool:
    if not SENTRY_DSN:
        return False
    try:
        import sentry_sdk  # pyre-ignore[21]
    except ImportError:
        logger.warning("SENTRY_DSN is set but sentry-sdk is not installed; error reporting is off")
        return False
    try:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            environment=os.environ.get("SENTRY_ENVIRONMENT", "production"),
            traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0")),
            send_default_pii=False,
            max_request_body_size="never",
            include_local_variables=False,
        )
    except Exception:
        logger.exception("Sentry could not be initialised; error reporting is off")
        return False
    logger.info("Sentry error reporting enabled")
    return True


SENTRY_ENABLED = _init_sentry()

# ── Style presets (data/styles.json) ───────────────────────────────────────────
STYLES_FILE = Path(os.environ.get("STYLES_FILE") or (BASE_DIR.parent / "data" / "styles.json"))
try:
    CATALOG: StyleCatalog = load_catalog(STYLES_FILE, STYLES_DIR)
except StyleCatalogError as _exc:
    # Fail at startup with one clear line instead of serving a broken studio.
    raise RuntimeError(f"PicturaAI cannot start: {_exc}") from None

# ── Upload limits & validation ─────────────────────────────────────────────────
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
# Last-resort decompression-bomb guard for every Pillow decode in the process:
# PIL warns above this and raises DecompressionBombError above 2× this cap.
Image.MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(64_000_000)))
# Pixel budgets per upload, checked from the header before anything is decoded.
# A few KB of PNG can declare 100+ MP, and decoding it (plus convert("RGB"))
# costs ~4 bytes per pixel on top of the ~455 MB the model already holds, so
# non-JPEG images get a small budget. JPEGs are decoded at a reduced DCT scale
# (Image.draft in nst_engine.open_image), so they may be larger.
MAX_UPLOAD_PIXELS = int(os.environ.get("MAX_UPLOAD_PIXELS", str(12_000_000)))
MAX_UPLOAD_PIXELS_JPEG = int(os.environ.get("MAX_UPLOAD_PIXELS_JPEG", str(40_000_000)))


def _pixel_budget(fmt: Optional[str]) -> int:
    budget = MAX_UPLOAD_PIXELS_JPEG if fmt in ("JPEG", "MPO") else MAX_UPLOAD_PIXELS
    return min(budget, Image.MAX_IMAGE_PIXELS or budget)


async def read_image_upload(upload: Optional[UploadFile], field: str) -> Optional[bytes]:
    """Read an uploaded file with size and pixel caps and verify it is an image.

    Returns None if the upload is absent or empty. Raises HTTPException on
    oversized (413) or invalid (400) files. Only the header is parsed here;
    nothing is decoded.
    """
    if upload is None:
        return None
    data = bytearray()
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{field} exceeds the {MAX_UPLOAD_MB} MB upload limit")
    if not data:
        return None
    raw = bytes(data)
    del data  # don't hold two copies while verifying
    try:
        # The budget is never above MAX_IMAGE_PIXELS, so Pillow's 64–128 MP
        # "warning only" range is rejected here too (no global warning filters:
        # they would also affect the worker thread).
        with Image.open(io.BytesIO(raw)) as img:
            width, height = img.size
            too_many_pixels = width * height > _pixel_budget(img.format)
            if not too_many_pixels:
                img.verify()
    except Image.DecompressionBombError:
        too_many_pixels = True
    except Exception:
        raise HTTPException(400, f"{field} is not a valid image file")
    if too_many_pixels:
        logger.info("Rejected %s: too many pixels", field)
        raise HTTPException(413, f"{field} has too many pixels. Please use a smaller image.")
    return raw

# ── In-memory job store ────────────────────────────────────────────────────────
jobs: Dict[str, dict] = {}
ws_clients: Dict[str, List[WebSocket]] = {}

executor = ThreadPoolExecutor(max_workers=1)

# ── Concurrency guard ──────────────────────────────────────────────────────────
# The single worker thread processes one heavy TF job at a time; cap how many
# more may wait in line so a request flood can't pile bytes up in memory.
# Handlers take a slot BEFORE reading any upload into memory (Starlette has
# already spooled large parts to temporary files by then), so rejected
# requests never hold image bytes. Admitted requests then wait for `_work_lock`
# before reading theirs, so only the job about to run holds image bytes; the
# others wait with their uploads still in those temporary files.
MAX_PENDING_JOBS = int(os.environ.get("MAX_PENDING_JOBS", "3"))
_inflight_jobs = 0
_work_lock = asyncio.Lock()  # recreated per lifespan (see lifespan below)


def _acquire_job_slot():
    """Raise 429 if the processing queue is full. Call from the event loop only."""
    global _inflight_jobs
    if _inflight_jobs >= MAX_PENDING_JOBS:
        raise HTTPException(429, "Server is busy processing other images. Please try again in a moment.")
    _inflight_jobs += 1


def _release_job_slot():
    global _inflight_jobs
    _inflight_jobs = max(0, _inflight_jobs - 1)

# ── Job / output cleanup ───────────────────────────────────────────────────────
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", "3600"))


def _cleanup_expired_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    for job_id in list(jobs):
        job = jobs.get(job_id) or {}
        ts = job.get("finished_at") or job.get("created_at") or job.get("started_at")
        if ts and ts < cutoff and job.get("status") in ("done", "error"):
            jobs.pop(job_id, None)
            ws_clients.pop(job_id, None)
            result_path = job.get("result_path")
            if result_path:
                Path(result_path).unlink(missing_ok=True)
    # Remove orphaned result files (e.g. left over from a previous process).
    # Only touch generated formats so files like .gitkeep survive.
    try:
        for f in OUTPUT_DIR.iterdir():
            if f.is_file() and f.suffix in (".jpg", ".gif") and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except OSError:
        pass


async def _cleanup_loop():
    while True:
        await asyncio.sleep(600)
        try:
            _cleanup_expired_jobs()
        except Exception:
            logger.exception("Job cleanup pass failed")

# ── Capture the main event loop at startup ─────────────────────────────────────
_main_loop: asyncio.AbstractEventLoop | None = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Modern lifespan handler (replaces deprecated on_event)."""
    global _main_loop, executor, _work_lock
    _main_loop = asyncio.get_running_loop()
    _work_lock = asyncio.Lock()
    # A fresh worker per lifespan: shutdown below closes it, and a closed pool
    # can't be reused (matters when the app is started more than once in-process).
    executor = ThreadPoolExecutor(max_workers=1)
    logger.info("Event loop captured for thread-safe WS broadcasting.")
    _cleanup_expired_jobs()
    cleanup_task = asyncio.create_task(_cleanup_loop())
    port = int(os.environ.get("PORT", 8000))
    docs_line = f"    ➜  API docs:  http://localhost:{port}/docs\n" if ENABLE_API_DOCS else ""
    logger.info(f"\n\n    🎨 PicturaAI is running!\n    ➜  Frontend:  http://localhost:{port}/app\n{docs_line}")
    logger.info("Loaded %d style presets from %s", len(CATALOG), CATALOG.source)
    yield
    # Shutdown: clean up background task and executor
    cleanup_task.cancel()
    executor.shutdown(wait=False)

# ── Public config, describe-style settings ─────────────────────────────────────
DESCRIBE_CFG = WorkersAIConfig.from_env()
DESCRIBE_QUOTA = DailyQuota(int(os.environ.get("DESCRIBE_DAILY_LIMIT", "50")))
SITE = SiteConfig.from_env(describe_generates=DESCRIBE_CFG.configured)
ENABLE_API_DOCS = env_bool("ENABLE_API_DOCS", True)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PicturaAI — Neural Style Transfer",
    description="Instant Neural Style Transfer powered by Google Magenta's pre-trained model. *Pictura* — Latin for 'a painting'.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if ENABLE_API_DOCS else None,
    redoc_url="/redoc" if ENABLE_API_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_API_DOCS else None,
)

# ── Request size and form part limits ──────────────────────────────────────────
# FastAPI parses the whole multipart body before a handler (and its rate limit)
# runs, with Starlette's defaults: 1000 files, 1000 fields, and up to 1 MB of
# every file part kept in memory. Cap the body and the part counts instead.
# Largest legitimate request: content + style + style_2 + mask, plus small fields.
MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", str(4 * MAX_UPLOAD_BYTES + 64 * 1024)))
FORM_MAX_FILES = 4
FORM_MAX_FIELDS = 24
FORM_MAX_FIELD_BYTES = 64 * 1024  # text fields only (file parts are spooled to disk)


class RequestTooLarge(HTTPException):
    def __init__(self) -> None:
        super().__init__(413, "Request is too large")


class BodySizeLimitMiddleware:
    """Reject POST /api/* bodies over `max_bytes` (pure ASGI, wraps `receive`).

    A declared Content-Length over the cap gets 413 without reading the body;
    otherwise streamed bytes are counted, so chunked bodies or a false
    Content-Length can't get around it.
    """

    def __init__(self, app: Any, max_bytes: Callable[[], int]) -> None:
        self.app = app
        self.max_bytes = max_bytes  # read per request

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or not scope["path"].startswith("/api/"):
            await self.app(scope, receive, send)
            return
        max_bytes = self.max_bytes()
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > max_bytes):
            await JSONResponse({"detail": "Request is too large"}, status_code=413)(scope, receive, send)
            return

        received = 0
        started = False

        async def limited_receive() -> dict:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > max_bytes:
                    raise RequestTooLarge()  # → 413 via the exception handler
            return message

        async def tracking_send(message: dict) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except RequestTooLarge:
            if started:
                raise
            await JSONResponse({"detail": "Request is too large"}, status_code=413)(scope, receive, send)


class _FormLimitedRequest(Request):
    def form(self, *, max_files: int | float = FORM_MAX_FILES, max_fields: int | float = FORM_MAX_FIELDS,
             max_part_size: int = FORM_MAX_FIELD_BYTES):  # type: ignore[override]
        return super().form(
            max_files=min(max_files, FORM_MAX_FILES),
            max_fields=min(max_fields, FORM_MAX_FIELDS),
            max_part_size=min(max_part_size, FORM_MAX_FIELD_BYTES),
        )


class _FormLimitedRoute(APIRoute):
    """Parse form bodies with tight part limits (too many parts → 400)."""

    def get_route_handler(self):  # type: ignore[override]
        handler = super().get_route_handler()

        async def limited_handler(request: Request) -> Response:
            return await handler(_FormLimitedRequest(request.scope, request.receive))

        return limited_handler


# Before any route is declared, so every API route uses it.
app.router.route_class = _FormLimitedRoute
# Registered first, so it runs inside CORS and the header middleware (its 413
# gets the same CORS and security headers as any other response).
app.add_middleware(BodySizeLimitMiddleware, max_bytes=lambda: MAX_REQUEST_BYTES)

# ── Per-IP rate limiting ───────────────────────────────────────────────────────
# Complements the queue guard: the guard caps total server load, these limits
# stop a single client from hogging all the slots. Behind a proxy the real
# client IP comes from CLIENT_IP_HEADER, or from X-Forwarded-For when the proxy's
# address is listed in FORWARDED_ALLOW_IPS (see uvicorn_options below).
RATE_LIMIT_TRANSFER    = os.environ.get("RATE_LIMIT_TRANSFER", "10/minute")
RATE_LIMIT_INTERPOLATE = os.environ.get("RATE_LIMIT_INTERPOLATE", "3/minute")
RATE_LIMIT_PALETTE     = os.environ.get("RATE_LIMIT_PALETTE", "20/minute")
RATE_LIMIT_DESCRIBE    = os.environ.get("RATE_LIMIT_DESCRIBE", "5/minute")

# Which request header carries the real visitor IP. Render sits behind
# Cloudflare, which *overwrites* CF-Connecting-IP with the connecting address,
# whereas X-Forwarded-For is only appended to — its first entry is whatever the
# client sent, so keying on it lets anyone dodge the limits. Leave this empty
# on hosts without such a proxy (a client could forge the header there); the
# limits then key on the socket peer, which is the proxy itself unless
# FORWARDED_ALLOW_IPS names it (coarser limits, but they can't be bypassed).
CLIENT_IP_HEADER = os.environ.get("CLIENT_IP_HEADER", "").strip().lower()


def client_ip(request: Request) -> str:
    """Rate-limit key: trusted proxy header if configured, else the socket peer."""
    if CLIENT_IP_HEADER:
        value = request.headers.get(CLIENT_IP_HEADER, "").split(",")[0].strip()
        if value:
            return value
    return get_remote_address(request)


limiter = Limiter(key_func=client_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: origins configurable via env (comma-separated). Wildcard origins must not
# be combined with credentials, so credentials stay off (the API uses no cookies).
_allowed_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# No `interest-cohort` (FLoC was removed from browsers in 2022): Chromium logs
# "Unrecognized feature: 'interest-cohort'" on every page load, and its
# successor `browsing-topics` triggers the same warning in plain Chromium.
PERMISSIONS_POLICY = "camera=(), microphone=(), geolocation=()"
HSTS_VALUE = "max-age=31536000; includeSubDomains"


# ── Static file caching ────────────────────────────────────────────────────────
# The frontend has no build step, so HTML, ES modules, CSS and icons keep their
# names across deploys: browsers must revalidate them (ETag, so usually a 304).
# Model files are requested with ?v=<manifest version> and TF.js sits in a
# versioned folder, so both can be cached for a year; a changed model file must
# get a new name or a new manifest version. The two index files that carry the
# version are revalidated. Artwork and sample images change rarely: one day.
CACHE_IMMUTABLE = "public, max-age=31536000, immutable"
CACHE_DAY = "public, max-age=86400"
CACHE_REVALIDATE = "no-cache"
MODEL_INDEX_FILES = ("manifest.json", "style-vectors.json")


def static_cache_control(path: str, status: int = 200) -> Optional[str]:
    """Cache-Control for a frontend path under /app (None: leave it unset)."""
    if not path.startswith("/app") or status not in (200, 206, 304):
        return None
    if path.startswith("/app/models/"):
        return CACHE_REVALIDATE if path.rsplit("/", 1)[-1] in MODEL_INDEX_FILES else CACHE_IMMUTABLE
    if path.startswith("/app/vendor/"):
        return CACHE_IMMUTABLE
    if path.startswith(("/app/assets/styles/", "/app/assets/samples/")):
        return CACHE_DAY
    return CACHE_REVALIDATE


# Compress text (HTML, JS, CSS, JSON: the model.json files are ~1 MB each), but
# not formats that are already compressed (images, model weights). Starlette
# adds Vary: Accept-Encoding and skips Range (206) responses. Registered before
# the header middleware below, so it wraps the route's own (unstreamed) response
# and `minimum_size` applies.
GZIP_EXCLUDED_TYPES = DEFAULT_EXCLUDED_CONTENT_TYPES + ("application/octet-stream",)
_gzip_options: Dict[str, Any] = {"minimum_size": 1000, "compresslevel": 6}
if "exclude_content_types" in inspect.signature(GZipMiddleware.__init__).parameters:
    _gzip_options["exclude_content_types"] = GZIP_EXCLUDED_TYPES
app.add_middleware(GZipMiddleware, **_gzip_options)


# ── Same-origin redirects ──────────────────────────────────────────────────────
# Starlette builds absolute Location URLs from request.url (StaticFiles adds the
# trailing slash to /app and /app/styles/<slug>, the router strips it from
# /docs/). Behind Render's proxy the peer is not in FORWARDED_ALLOW_IPS, so the
# scheme stays "http" and an https visitor would be sent to http:// for one hop.
# A path-only Location keeps whatever scheme and host the browser used, without
# trusting X-Forwarded-* (which the rate limits deliberately don't trust either).
def relative_location(location: str, scheme: str, netloc: str) -> str:
    """Return `location` as a path (with query) if it points at this origin."""
    try:
        parts = urlsplit(location)
    except ValueError:
        return location
    if parts.scheme.lower() != scheme.lower() or parts.netloc.lower() != netloc.lower():
        return location
    # "//evil.example" or "/\evil.example" would read as another host: collapse
    # the leading slashes so the result can only ever be a path on this site.
    path = "/" + parts.path.lstrip("/\\")
    return urlunsplit(("", "", path, parts.query, parts.fragment))


@app.middleware("http")
async def _security_headers(request, call_next):
    response = await call_next(request)
    path = request.url.path
    location = response.headers.get("location")
    if location:
        relative = relative_location(location, request.url.scheme, request.url.netloc)
        if relative != location:
            response.headers["location"] = relative
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", PERMISSIONS_POLICY)
    csp = SiteConfig.docs_csp() if is_docs_path(path) else SITE.app_csp()
    response.headers.setdefault(SITE.csp_header_name(), csp)
    # HSTS only over HTTPS (Render terminates TLS and sets X-Forwarded-Proto);
    # browsers ignore it on plain HTTP and it would pin localhost otherwise.
    if is_https(request.url.scheme, request.headers.get("x-forwarded-proto")):
        response.headers.setdefault("Strict-Transport-Security", HSTS_VALUE)
    cache_control = static_cache_control(path, response.status_code)
    if cache_control:
        response.headers.setdefault("Cache-Control", cache_control)
    return response


# Must be registered before the /app mount, which would otherwise shadow it.
@app.get("/app/config.js", include_in_schema=False)
async def public_config_js():
    return Response(
        SITE.config_js(),
        media_type="application/javascript; charset=utf-8",
        headers={"Cache-Control": "no-cache"},
    )


# Serve frontend static files
app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ── Style presets ──────────────────────────────────────────────────────────────
def get_style_path(style_key: Optional[str]) -> Optional[Path]:
    return CATALOG.image_path(style_key)


def resolve_style_from_text(prompt: str) -> Optional[str]:
    """Map a free-text style prompt to the best matching preset key (or None)."""
    preset = CATALOG.match_text(prompt)
    return preset.key if preset else None


NO_TEXT_MATCH_MESSAGE = (
    "We couldn't match those words to a painting. "
    "Try words like 'starry night', 'ink wash' or 'gold'."
)


def _resolve_text_prompt(raw: str) -> str:
    """Validate a `text_prompt` form field like /api/describe-style does, then match it.

    The matcher runs on the event loop, so the length is capped before any work.
    """
    too_long = f"Describe the style in 1 to {PROMPT_MAX_CHARS} characters."
    if len(raw) > PROMPT_MAX_CHARS * 4:
        raise HTTPException(400, too_long)
    prompt = clean_prompt(raw)
    if not 1 <= len(prompt) <= PROMPT_MAX_CHARS:
        raise HTTPException(400, too_long)
    key = resolve_style_from_text(prompt)
    if key is None:
        raise HTTPException(400, NO_TEXT_MATCH_MESSAGE)
    return key


def _style_title(style_key: Optional[str]) -> Optional[str]:
    preset = CATALOG.get(style_key)
    return preset.title if preset else None


def _web_url(asset: str) -> Optional[str]:
    return f"/app/{asset}" if asset else None


def _style_summary(preset: StylePreset) -> Dict[str, Any]:
    return {
        "key": preset.key,
        "slug": preset.slug,
        "title": preset.title,
        "name": preset.title,  # legacy field (the old UI reads `name`)
        "artist": preset.artist,
        "year": preset.year,
        "medium": preset.medium,
        "description": preset.description,
        "image": _web_url(preset.image),
        "card": _web_url(preset.card),
        "thumb": _web_url(preset.thumb),
        "page": f"/app/styles/{preset.slug}/",
        "aspect": preset.aspect,
        "credit": preset.credit,
        "source_url": preset.source_url,
        "license": preset.license,
    }

# ── Helper: broadcast via WS ───────────────────────────────────────────────────
async def _broadcast(job_id: str, payload: dict):
    clients = ws_clients.get(job_id, [])
    dead = []
    for ws in clients:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.remove(ws)

def _sync_broadcast(job_id: str, payload: dict):
    """Thread-safe broadcast from worker thread."""
    loop = _main_loop
    if loop is None or loop.is_closed():
        return
    try:
        asyncio.run_coroutine_threadsafe(_broadcast(job_id, payload), loop)
    except Exception:
        pass

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
async def root():
    # Straight to /app/ (the /app mount would otherwise add a second hop).
    return RedirectResponse(url="/app/", status_code=301)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Browsers request this path regardless of <link rel="icon">.
    return FileResponse(FRONTEND_DIR / "assets" / "favicon.ico", media_type="image/x-icon",
                        headers={"Cache-Control": CACHE_DAY})

@app.get("/robots.txt", include_in_schema=False)
async def robots_txt():
    return Response(SITE.robots_txt(), media_type="text/plain; charset=utf-8",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/sitemap.xml", include_in_schema=False)
async def sitemap_xml():
    has_styles_index = (FRONTEND_DIR / "styles" / "index.html").is_file()
    body = SITE.sitemap_xml((p.slug for p in CATALOG), include_styles_index=has_styles_index)
    return Response(body, media_type="application/xml",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/health")
@app.head("/api/health", include_in_schema=False)
async def health():
    """Cheap liveness check for the host and uptime monitors (no model, no disk)."""
    return {"status": "ok"}


STYLES_CACHE_CONTROL = "public, max-age=3600"
_styles_list: Optional[List[dict]] = None
_styles_with_thumbs: Optional[List[dict]] = None


def _legacy_thumbnail(preset: StylePreset) -> Optional[str]:
    path = CATALOG.image_path(preset.key)
    if not path or not path.exists():
        return None
    with Image.open(path) as src:
        img = src.convert("RGB")
    img.thumbnail((200, 200), _LANCZOS)
    return base64.b64encode(pil_to_bytes(img, quality=70)).decode()


@app.get("/api/styles")
async def list_styles(include: Optional[str] = None):
    """Built-in presets. Image URLs point at static files under /app/.

    `?include=thumbnail` adds the old base64 `thumbnail` field (200 px JPEG),
    kept for older clients; it makes the response ~150 KB larger.
    """
    # Presets are static: build each variant once and reuse it.
    global _styles_list, _styles_with_thumbs
    wants_thumbs = "thumbnail" in {p.strip() for p in (include or "").split(",")}
    if wants_thumbs:
        if _styles_with_thumbs is None:
            def build() -> List[dict]:
                return [{**_style_summary(p), "thumbnail": _legacy_thumbnail(p)} for p in CATALOG]
            # Decoding the paintings is slow-ish: keep it off the event loop.
            _styles_with_thumbs = await asyncio.get_running_loop().run_in_executor(None, build)
        body = _styles_with_thumbs
    else:
        if _styles_list is None:
            _styles_list = [_style_summary(p) for p in CATALOG]
        body = _styles_list
    return JSONResponse(body, headers={"Cache-Control": STYLES_CACHE_CONTROL})

@app.post("/api/transfer")
@limiter.limit(RATE_LIMIT_TRANSFER)
async def start_transfer(
    request: Request,
    content_image: UploadFile = File(...),
    style_image: Optional[UploadFile] = File(None),
    style_preset: Optional[str] = Form(None),
    text_prompt: Optional[str] = Form(None),
    intensity: Optional[float] = Form(
        None, ge=0.0, le=1.0,
        description="Style intensity 0–1 (0 = photo, 1 = full style). Alias of style_weight; wins if both are sent.",
    ),
    output: Literal["final", "raw"] = Form(
        "final",
        description="`final` (default): the finished image with intensity blend, detail and sharpening. "
                    "`raw`: the stylized image only, at the output size (JPEG q95), for clients that "
                    "composite intensity and masks themselves; intensity, style_weight and mask_image are ignored.",
    ),
    style_weight: float = Form(1e-2, description="Legacy name for `intensity` (0–1). Prefer `intensity`."),
    content_weight: float = Form(1e4, description="Ignored. Accepted so older clients keep working."),
    tv_weight: float = Form(30.0, description="Ignored. Accepted so older clients keep working."),
    num_steps: int = Form(300, description="Ignored. Accepted so older clients keep working."),
    learning_rate: float = Form(0.02, description="Ignored. Accepted so older clients keep working."),
    style_image_2: Optional[UploadFile] = File(None),
    style_preset_2: Optional[str] = Form(None),
    style_mix_ratio: float = Form(0.5, description="Mix of the second style, 0 = only style 1, 1 = only style 2."),
    mask_image: Optional[UploadFile] = File(None, description="White = styled. Ignored when output=raw."),
):
    if intensity is not None:
        style_weight = intensity
    raw_output = output == "raw"

    # Validate
    resolved_style_key: Optional[str] = None
    if style_image is None and style_preset is None:
        if text_prompt and text_prompt.strip():
            resolved_style_key = _resolve_text_prompt(text_prompt)
            style_preset = resolved_style_key
        else:
            raise HTTPException(400, "Provide style_image, style_preset, or text_prompt")

    num_steps = max(50, min(num_steps, 1000))

    # ── Run synchronously to avoid OOM-restart losing job state ──────────
    # On constrained environments (free tier), the async worker pattern fails
    # because the process can be killed mid-inference and the job store is lost.
    resolved_style_name = _style_title(resolved_style_key)

    # Queue check first: a rejected request must not read its uploads into memory.
    _acquire_job_slot()
    try:
        async with _work_lock:
            content_bytes = await read_image_upload(content_image, "content_image")
            if content_bytes is None:
                raise HTTPException(400, "content_image is empty")

            style_bytes = await read_image_upload(style_image, "style_image")
            if style_bytes is None:
                assert style_preset is not None  # validated above (style_image is None ⇒ style_preset must exist)
                style_path = get_style_path(style_preset)
                if not style_path or not style_path.exists():
                    raise HTTPException(404, f"Style preset '{style_preset}' not found")
                style_bytes = style_path.read_bytes()

            # ── Optional second style (style mixing) ─────────────────────────
            style_bytes_2 = await read_image_upload(style_image_2, "style_image_2")
            if style_bytes_2 is None and style_preset_2 is not None:
                style_path_2 = get_style_path(style_preset_2)
                if style_path_2 and style_path_2.exists():
                    style_bytes_2 = style_path_2.read_bytes()

            # ── Optional region mask (regional styling; the client masks raw output) ─
            mask_bytes = None if raw_output else await read_image_upload(mask_image, "mask_image")

            loop = asyncio.get_running_loop()
            result_bytes = await loop.run_in_executor(
                executor,
                lambda: run_nst(
                    content_bytes, style_bytes,
                    style_weight=style_weight,
                    content_weight=content_weight,
                    tv_weight=tv_weight,
                    num_steps=num_steps,
                    learning_rate=learning_rate,
                    style_bytes_2=style_bytes_2,
                    style_mix_ratio=style_mix_ratio,
                    mask_bytes=mask_bytes,
                    raw=raw_output,
                )
            )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Transfer failed")
        raise HTTPException(500, "Style transfer failed. Please try again with a different image.")
    finally:
        _release_job_slot()

    # The result goes straight back in the response and is not stored on the
    # server (data minimisation: nothing reads a saved copy of a transfer).
    result_b64 = base64.b64encode(result_bytes).decode()

    # Release the raw bytes before building the response
    del result_bytes
    import gc
    gc.collect()

    return JSONResponse({
        "status": "done",
        "result": result_b64,
        "output": output,
        "resolved_style_key": resolved_style_key,
        "resolved_style_name": resolved_style_name,
    })

def _job_result_b64(job: dict) -> Optional[str]:
    """Read a finished job's output from disk (results are not kept in memory)."""
    path = job.get("result_path")
    if job.get("status") != "done" or not path:
        return None
    try:
        return base64.b64encode(Path(path).read_bytes()).decode()
    except OSError:
        return None


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    resp = {k: v for k, v in job.items() if k not in ("result", "preview", "result_path")}
    if job.get("status") == "done":
        resp["result"] = _job_result_b64(job)
    resp["preview"] = job.get("preview")
    return JSONResponse(resp)

@app.get("/api/result/{job_id}")
async def download_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(400, f"Job is {job['status']}, not done yet")
    path = job.get("result_path")
    if not path or not Path(path).exists():
        raise HTTPException(500, "Result file missing")
    if job.get("result_type") == "gif":
        return FileResponse(path, media_type="image/gif", filename=f"styled_{job_id}.gif")
    return FileResponse(path, media_type="image/jpeg", filename=f"styled_{job_id}.jpg")

@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    job = jobs.pop(job_id, None)
    ws_clients.pop(job_id, None)
    if not job:
        raise HTTPException(404, "Job not found")
    result_path = job.get("result_path")
    if result_path:
        Path(result_path).unlink(missing_ok=True)
    return {"message": "Job removed"}


# ── Style Interpolation Animation (GIF) ───────────────────────────────────────

def _interpolation_worker(
    job_id: str,
    content_bytes: bytes,
    style_bytes: bytes,
    num_frames: int,
    frame_duration_ms: int,
):
    job = jobs[job_id]
    job["status"] = "processing"
    job["started_at"] = time.time()

    def progress_callback(step: int, total: int, loss: float, img_bytes: bytes):
        pct = round(step / total * 100)
        job["progress"] = pct
        _sync_broadcast(job_id, {"type": "progress", "step": step, "total": total, "percent": pct})

    try:
        gif_bytes = run_interpolation_gif(
            content_bytes, style_bytes,
            num_frames=num_frames,
            frame_duration_ms=frame_duration_ms,
            progress_callback=progress_callback,
        )
        out_path = OUTPUT_DIR / f"{job_id}.gif"
        out_path.write_bytes(gif_bytes)

        result_b64 = base64.b64encode(gif_bytes).decode()
        del gif_bytes
        job["status"] = "done"
        job["result_path"] = str(out_path)
        job["result_type"] = "gif"
        job["progress"] = 100
        job["finished_at"] = time.time()
        _sync_broadcast(job_id, {"type": "done", "percent": 100, "result": result_b64, "result_type": "gif"})
    except Exception:
        logger.exception(f"Interpolation job {job_id} failed")
        job["status"] = "error"
        job["error"] = "Animation generation failed. Please try again."
        job["finished_at"] = time.time()
        _sync_broadcast(job_id, {"type": "error", "message": job["error"]})
    finally:
        _release_job_slot()


@app.post("/api/interpolate")
@limiter.limit(RATE_LIMIT_INTERPOLATE)
async def start_interpolation(
    request: Request,
    content_image: UploadFile = File(...),
    style_image: Optional[UploadFile] = File(None),
    style_preset: Optional[str] = Form(None),
    num_frames: int = Form(10),
    frame_duration: int = Form(200),
):
    if style_image is None and style_preset is None:
        raise HTTPException(400, "Provide either style_image or style_preset")

    num_frames = max(5, min(num_frames, 20))
    frame_duration = max(50, min(frame_duration, 500))

    # Queue check before reading uploads. Once the job is submitted, the worker
    # releases the slot; until then, every exit path here releases it.
    _acquire_job_slot()
    job_id = str(uuid.uuid4())
    submitted = False
    try:
        async with _work_lock:
            content_bytes = await read_image_upload(content_image, "content_image")
            if content_bytes is None:
                raise HTTPException(400, "content_image is empty")

            style_bytes = await read_image_upload(style_image, "style_image")
            if style_bytes is None:
                assert style_preset is not None
                style_path = get_style_path(style_preset)
                if not style_path or not style_path.exists():
                    raise HTTPException(404, f"Style preset '{style_preset}' not found")
                style_bytes = style_path.read_bytes()

            jobs[job_id] = {
                "id": job_id, "status": "queued", "progress": 0,
                "created_at": time.time(), "result_type": "gif",
            }
            ws_clients[job_id] = []
            executor.submit(_interpolation_worker, job_id, content_bytes, style_bytes, num_frames, frame_duration)  # type: ignore[arg-type]
            submitted = True
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to queue interpolation job")
        raise HTTPException(500, "Could not start the animation job. Please try again.")
    finally:
        if not submitted:
            _release_job_slot()
            jobs.pop(job_id, None)
            ws_clients.pop(job_id, None)

    return JSONResponse({"job_id": job_id, "status": "queued"})


# ── Color Palette Transfer ─────────────────────────────────────────────────────

@app.post("/api/palette-transfer")
@limiter.limit(RATE_LIMIT_PALETTE)
async def palette_transfer(
    request: Request,
    content_image: UploadFile = File(...),
    style_image: Optional[UploadFile] = File(None),
    style_preset: Optional[str] = Form(None),
    text_prompt: Optional[str] = Form(None),
    strength: float = Form(1.0),
    style_image_2: Optional[UploadFile] = File(None),
    style_preset_2: Optional[str] = Form(None),
    style_mix_ratio: float = Form(0.5),
):
    resolved_style_key: Optional[str] = None
    if style_image is None and style_preset is None:
        if text_prompt and text_prompt.strip():
            resolved_style_key = _resolve_text_prompt(text_prompt)
            style_preset = resolved_style_key
        else:
            raise HTTPException(400, "Provide style_image, style_preset, or text_prompt")

    strength = max(0.0, min(1.0, strength))

    # Queue check before reading uploads, then run the CPU-heavy transfer in the
    # worker thread so it never blocks the event loop.
    _acquire_job_slot()
    try:
        async with _work_lock:
            content_bytes = await read_image_upload(content_image, "content_image")
            if content_bytes is None:
                raise HTTPException(400, "content_image is empty")

            style_bytes = await read_image_upload(style_image, "style_image")
            if style_bytes is None:
                assert style_preset is not None
                style_path = get_style_path(style_preset)
                if not style_path or not style_path.exists():
                    raise HTTPException(404, f"Style preset '{style_preset}' not found")
                style_bytes = style_path.read_bytes()

            # ── Optional second style (style mixing) ─────────────────────────
            style_bytes_2 = await read_image_upload(style_image_2, "style_image_2")
            if style_bytes_2 is None and style_preset_2 is not None:
                style_path_2 = get_style_path(style_preset_2)
                if style_path_2 and style_path_2.exists():
                    style_bytes_2 = style_path_2.read_bytes()

            loop = asyncio.get_running_loop()
            result_bytes = await loop.run_in_executor(
                executor,
                lambda: color_palette_transfer(
                    content_bytes, style_bytes, strength=strength,
                    style_bytes_2=style_bytes_2, style_mix_ratio=style_mix_ratio,
                )
            )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Palette transfer failed")
        raise HTTPException(500, "Palette transfer failed. Please try again with a different image.")
    finally:
        _release_job_slot()
    result_b64 = base64.b64encode(result_bytes).decode()

    return JSONResponse({
        "result": result_b64,
        "resolved_style_key": resolved_style_key,
        "resolved_style_name": _style_title(resolved_style_key),
    })


# ── Describe it: text → generated swatch, or the closest preset ────────────────
DESCRIBE_MAX_BODY_BYTES = 16 * 1024
# Why generation was skipped, shown before the fallback (none when it is simply off).
_DESCRIBE_REASONS = {
    "not_configured": "",
    "daily_limit": "Today's describe limit is used up.",
    "quota": "Today's describe limit is used up.",
    "failed": "The style generator isn't available right now.",
}
_DESCRIBE_NO_MATCH = (
    "No classic style matches that description. "
    "Try a painter, a colour or a mood, like \"Van Gogh\", \"gold\" or \"stormy sea\"."
)


async def _read_describe_prompt(request: Request) -> Any:
    """Accept `prompt` as a form field or as JSON {"prompt": "..."}."""
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > DESCRIBE_MAX_BODY_BYTES:
        raise HTTPException(413, "Request is too large")
    ctype = request.headers.get("content-type", "").lower()
    if ctype.startswith("application/json"):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > DESCRIBE_MAX_BODY_BYTES:
                raise HTTPException(413, "Request is too large")
        try:
            data = json.loads(bytes(body) or b"null")
        except ValueError:
            raise HTTPException(400, "Body must be JSON like {\"prompt\": \"...\"}")
        return data.get("prompt") if isinstance(data, dict) else None
    form = await request.form(max_files=0, max_fields=8, max_part_size=DESCRIBE_MAX_BODY_BYTES)
    return form.get("prompt")


@app.post(
    "/api/describe-style",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                media: {
                    "schema": {
                        "type": "object",
                        "required": ["prompt"],
                        "properties": {"prompt": {
                            "type": "string", "minLength": 1, "maxLength": PROMPT_MAX_CHARS,
                            "description": "What the style should look like, e.g. \"neon city at night\".",
                        }},
                    }
                }
                for media in ("application/x-www-form-urlencoded", "multipart/form-data", "application/json")
            },
        }
    },
)
@limiter.limit(RATE_LIMIT_DESCRIBE)
async def describe_style(request: Request):
    """Turn a short description into a style.

    With Cloudflare Workers AI configured (and under the daily cap) this returns
    `{"source": "generated", "image": <base64 JPEG ≤512 px>, "media_type": "image/jpeg"}`.
    Otherwise, or when generation fails, it returns the closest built-in preset:
    `{"source": "matched", "key", "title", "artist", "message", "reason"}`.
    404 when nothing matches either.
    """
    if not SITE.describe_enabled:
        raise HTTPException(404, "Describe it is turned off on this server.")

    raw_prompt = await _read_describe_prompt(request)
    if not isinstance(raw_prompt, str) or len(raw_prompt) > PROMPT_MAX_CHARS * 4:
        raise HTTPException(400, f"Describe the style in 1 to {PROMPT_MAX_CHARS} characters.")
    prompt = clean_prompt(raw_prompt)
    if not 1 <= len(prompt) <= PROMPT_MAX_CHARS:
        raise HTTPException(400, f"Describe the style in 1 to {PROMPT_MAX_CHARS} characters.")

    no_store = {"Cache-Control": "no-store"}
    if not DESCRIBE_CFG.configured:
        reason = "not_configured"
    elif not DESCRIBE_QUOTA.try_acquire():
        reason = "daily_limit"
    else:
        try:
            # Network I/O: default thread pool, not the single TF worker.
            jpeg = await asyncio.get_running_loop().run_in_executor(
                None, lambda: generate_swatch(DESCRIBE_CFG, prompt)
            )
            return JSONResponse({
                "source": "generated",
                "image": base64.b64encode(jpeg).decode(),
                "media_type": "image/jpeg",
                "prompt": prompt,
            }, headers=no_store)
        except DescribeUnavailable as exc:
            reason = exc.reason
            if reason == "quota":
                DESCRIBE_QUOTA.exhaust()
        except Exception:
            logger.exception("Describe-style generation failed unexpectedly")
            reason = "failed"

    why = _DESCRIBE_REASONS.get(reason, _DESCRIBE_REASONS["failed"])
    preset = CATALOG.match_text(prompt)
    if preset is None:
        raise HTTPException(404, f"{why} {_DESCRIBE_NO_MATCH}".strip())
    if why:
        message = f"{why} We matched the closest classic style instead: {preset.title}."
    else:
        message = f"We matched the closest classic style: {preset.title}."
    return JSONResponse({
        "source": "matched",
        "key": preset.key,
        "title": preset.title,
        "artist": preset.artist,
        "message": message,
        "reason": reason,
    }, headers=no_store)

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    # Only accept subscriptions for jobs that actually exist — otherwise an
    # attacker could grow ws_clients unboundedly with random ids.
    if job_id not in jobs:
        await websocket.close(code=4404)
        return
    if job_id not in ws_clients:
        ws_clients[job_id] = []
    ws_clients[job_id].append(websocket)

    job = jobs.get(job_id, {})
    # Send current state immediately on connect
    if job.get("status") == "done":
        await websocket.send_text(json.dumps({
            "type": "done", "percent": 100, "result": _job_result_b64(job),
            "result_type": job.get("result_type"),
        }))
    elif job.get("status") == "error":
        await websocket.send_text(json.dumps({
            "type": "error", "message": job.get("error", "Unknown error")
        }))
    else:
        await websocket.send_text(json.dumps({
            "type": "progress",
            "percent": job.get("progress", 0),
            "preview": job.get("preview"),
        }))

    try:
        while True:
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        clients = ws_clients.get(job_id)
        if clients and websocket in clients:
            clients.remove(websocket)
        if clients is not None and not clients:
            ws_clients.pop(job_id, None)

# ── Entry point ────────────────────────────────────────────────────────────────
def uvicorn_options() -> Dict[str, Any]:
    """Options for `python backend/main.py` (the Dockerfile CMD).

    X-Forwarded-For / X-Forwarded-Proto are trusted only from the peers listed in
    FORWARDED_ALLOW_IPS (default 127.0.0.1, uvicorn's own default). Never "*":
    uvicorn then takes the leftmost X-Forwarded-For entry, which the client
    writes, and the per-IP rate limits could be dodged with a forged header.
    Behind a proxy, set CLIENT_IP_HEADER (cf-connecting-ip on Render) or set
    FORWARDED_ALLOW_IPS to the proxy's own address on purpose.
    """
    return {
        "host": "0.0.0.0",
        "port": int(os.environ.get("PORT", 8000)),
        "reload": False,
        "log_level": "info",
        "proxy_headers": True,
        "forwarded_allow_ips": os.environ.get("FORWARDED_ALLOW_IPS", "").strip() or "127.0.0.1",
    }


if __name__ == "__main__":
    import uvicorn  # pyre-ignore[21]
    uvicorn.run("main:app", **uvicorn_options())
