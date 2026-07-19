"""
FastAPI Backend — PicturaAI
============================
Endpoints:
  POST /api/transfer        – Start a new NST job, returns job_id
  GET  /api/jobs/{job_id}   – Poll job status & result
  WS   /ws/{job_id}         – WebSocket stream for real-time progress
  GET  /api/styles           – List built-in style presets
  POST /api/styles/preview  – Quick low-res style preview
  GET  /api/result/{job_id} – Download final image
  DELETE /api/jobs/{job_id} – Cancel / cleanup
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
import json
import logging
import re
import shutil
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

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
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse  # pyre-ignore[21]
from fastapi.staticfiles import StaticFiles  # pyre-ignore[21]

from nst_engine import run_nst, pil_to_bytes, run_interpolation_gif, color_palette_transfer  # pyre-ignore[21]
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

# ── Upload limits & validation ─────────────────────────────────────────────────
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
# Reject decompression bombs: PIL raises DecompressionBombError above 2× this cap.
Image.MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(64_000_000)))


async def read_image_upload(upload: Optional[UploadFile], field: str) -> Optional[bytes]:
    """Read an uploaded file with a size cap and verify it is a decodable image.

    Returns None if the upload is absent or empty. Raises HTTPException on
    oversized or invalid files.
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
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
    except Exception:
        raise HTTPException(400, f"{field} is not a valid image file")
    return raw

# ── In-memory job store ────────────────────────────────────────────────────────
jobs: Dict[str, dict] = {}
ws_clients: Dict[str, List[WebSocket]] = {}

executor = ThreadPoolExecutor(max_workers=1)

# ── Concurrency guard ──────────────────────────────────────────────────────────
# The single worker thread processes one heavy TF job at a time; cap how many
# more may wait in line so a request flood can't pile bytes up in memory.
MAX_PENDING_JOBS = int(os.environ.get("MAX_PENDING_JOBS", "3"))
_inflight_jobs = 0


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
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    logger.info("Event loop captured for thread-safe WS broadcasting.")
    _cleanup_expired_jobs()
    cleanup_task = asyncio.create_task(_cleanup_loop())
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"\n\n    🎨 PicturaAI is running!\n    ➜  Frontend:  http://localhost:{port}/app\n    ➜  API docs:  http://localhost:{port}/docs\n")
    yield
    # Shutdown: clean up background task and executor
    cleanup_task.cancel()
    executor.shutdown(wait=False)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PicturaAI — Neural Style Transfer",
    description="Instant Neural Style Transfer powered by Google Magenta's pre-trained model. *Pictura* — Latin for 'a painting'.",
    version="2.1.0",
    lifespan=lifespan,
)

# ── Per-IP rate limiting ───────────────────────────────────────────────────────
# Complements the queue guard: the guard caps total server load, these limits
# stop a single client from hogging all the slots. Requires proxy headers to be
# trusted (see uvicorn.run below) so the real client IP is seen behind Render.
RATE_LIMIT_TRANSFER    = os.environ.get("RATE_LIMIT_TRANSFER", "10/minute")
RATE_LIMIT_INTERPOLATE = os.environ.get("RATE_LIMIT_INTERPOLATE", "3/minute")
RATE_LIMIT_PALETTE     = os.environ.get("RATE_LIMIT_PALETTE", "20/minute")

limiter = Limiter(key_func=get_remote_address)
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


@app.middleware("http")
async def _security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response

# Serve frontend static files
app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# ── Style presets ──────────────────────────────────────────────────────────────
STYLE_PRESETS = {
    "starry_night":        {"file": "starry_night.jpg",        "name": "Starry Night",        "artist": "Van Gogh",     "description": "Swirling cosmic energy and deep blues"},
    "the_scream":          {"file": "the_scream.jpg",          "name": "The Scream",          "artist": "Munch",        "description": "Anguished expressionist curves"},
    "great_wave":          {"file": "The_Great_Wave_off_Kanagawa.jpg", "name": "The Great Wave", "artist": "Hokusai",  "description": "Bold Japanese woodblock print"},
    "la_muse":             {"file": "la_muse.jpg",             "name": "La Muse",             "artist": "Picasso",      "description": "Cubist fragments and vivid tones"},
    "rain_princess":       {"file": "rain_princess.jpg",       "name": "Rain Princess",       "artist": "Afremov",      "description": "Rainy street reflected in warm colour"},
    "udnie":               {"file": "udnie.jpg",               "name": "Udnie",               "artist": "Picabia",      "description": "Abstract art-deco swirls"},
    "the_shipwreck":       {"file": "the_shipwreck_of_the_minotaur.jpg", "name": "The Shipwreck", "artist": "Turner",  "description": "Dramatic seascape in oils"},
    "aquarelle":           {"file": "aquarelle.jpg",           "name": "Aquarelle",           "artist": "Unknown",      "description": "Soft watercolour washes"},
    "chinese_style":       {"file": "chinese_style.jpg",       "name": "Chinese Ink",         "artist": "Traditional",  "description": "Delicate ink brush strokes"},
    "space":               {"file": "space.jpg",               "name": "Space",               "artist": "Digital",      "description": "Nebulae and cosmic textures"},
    "hampson":             {"file": "hampson.jpg",             "name": "Hampson",             "artist": "Hampson",      "description": "Bold illustrative style"},
    "mountain":            {"file": "mountain.jpg",            "name": "Mountain",            "artist": "Nature",       "description": "Rugged mountain textures"},
    "paris":               {"file": "paris.jpg",               "name": "Paris",               "artist": "Photography",  "description": "Parisian street atmosphere"},
}

STYLE_TEXT_TAGS: Dict[str, set[str]] = {
    "starry_night": {"starry", "night", "swirl", "swirling", "van", "gogh", "impressionist", "postimpressionist", "cosmic", "dreamy", "sky"},
    "the_scream": {"scream", "expressionist", "dramatic", "anguish", "munch", "emotional", "intense"},
    "great_wave": {"wave", "ocean", "sea", "japanese", "hokusai", "ukiyoe", "water", "coast", "kanagawa"},
    "la_muse": {"cubist", "cubism", "picasso", "geometric", "fragments", "portrait", "modern"},
    "rain_princess": {"rain", "rainy", "street", "lights", "reflection", "afremov", "city", "evening"},
    "udnie": {"abstract", "artdeco", "deco", "dynamic", "picabia", "energetic", "surreal"},
    "the_shipwreck": {"ship", "shipwreck", "storm", "sea", "turner", "dramatic", "oil", "seascape"},
    "aquarelle": {"watercolor", "watercolour", "soft", "wash", "pastel", "aquarelle", "delicate"},
    "chinese_style": {"ink", "brush", "chinese", "traditional", "minimal", "monochrome", "calligraphy"},
    "space": {"space", "galaxy", "nebula", "cosmic", "sci", "fi", "stars", "universe", "futuristic"},
    "hampson": {"illustration", "graphic", "bold", "comic", "linework", "hampson"},
    "mountain": {"mountain", "nature", "landscape", "forest", "rocks", "earthy", "outdoor"},
    "paris": {"paris", "street", "city", "romantic", "european", "photography", "urban"},
}

def get_style_path(style_key: str) -> Optional[Path]:
    if style_key not in STYLE_PRESETS:
        return None
    return STYLES_DIR / STYLE_PRESETS[style_key]["file"]


def resolve_style_from_text(prompt: str) -> Optional[str]:
    """Map free-text style prompt to the best matching built-in style preset."""
    text = prompt.strip().lower()
    if not text:
        return None

    tokens = [t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 1]
    if not tokens:
        return None

    best_key: Optional[str] = None
    best_score = 0

    for key, info in STYLE_PRESETS.items():
        searchable = " ".join([
            key.replace("_", " "),
            str(info.get("name", "")).lower(),
            str(info.get("artist", "")).lower(),
            str(info.get("description", "")).lower(),
        ])
        tags = STYLE_TEXT_TAGS.get(key, set())

        score = 0
        for token in tokens:
            if token in tags:
                score += 4
            if token in searchable:
                score += 2

        if key.replace("_", " ") in text:
            score += 4

        if score > best_score:
            best_score = score
            best_key = key

    # If nothing matches, avoid surprising random picks.
    if best_score <= 0:
        return None
    return best_key

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
@app.get("/")
@app.head("/")
async def root():
    return RedirectResponse(url="/app", status_code=301)

_styles_cache: Optional[List[dict]] = None

@app.get("/api/styles")
async def list_styles():
    # Presets are static — build the thumbnail list once and reuse it
    # (this endpoint doubles as the health check, so it is called often).
    global _styles_cache
    if _styles_cache is None:
        result = []
        for key, info in STYLE_PRESETS.items():
            style_path = get_style_path(key)
            thumbnail_b64 = None
            if style_path and style_path.exists():
                img = Image.open(style_path).convert("RGB")
                img.thumbnail((200, 200), _LANCZOS)
                thumbnail_b64 = base64.b64encode(pil_to_bytes(img, quality=70)).decode()
            result.append({
                "key": key,
                "name": info["name"],
                "artist": info["artist"],
                "description": info["description"],
                "thumbnail": thumbnail_b64,
            })
        _styles_cache = result
    return JSONResponse(_styles_cache)

@app.post("/api/transfer")
@limiter.limit(RATE_LIMIT_TRANSFER)
async def start_transfer(
    request: Request,
    content_image: UploadFile = File(...),
    style_image: Optional[UploadFile] = File(None),
    style_preset: Optional[str] = Form(None),
    text_prompt: Optional[str] = Form(None),
    style_weight: float = Form(1e-2),
    content_weight: float = Form(1e4),
    tv_weight: float = Form(30.0),
    num_steps: int = Form(300),
    learning_rate: float = Form(0.02),
    style_image_2: Optional[UploadFile] = File(None),
    style_preset_2: Optional[str] = Form(None),
    style_mix_ratio: float = Form(0.5),
    mask_image: Optional[UploadFile] = File(None),
):
    # Validate
    resolved_style_key: Optional[str] = None
    if style_image is None and style_preset is None:
        if text_prompt and text_prompt.strip():
            resolved_style_key = resolve_style_from_text(text_prompt)
            if resolved_style_key is None:
                raise HTTPException(400, "No matching style found for the provided text_prompt")
            style_preset = resolved_style_key
        else:
            raise HTTPException(400, "Provide style_image, style_preset, or text_prompt")

    num_steps = max(50, min(num_steps, 1000))

    content_bytes = await read_image_upload(content_image, "content_image")
    if content_bytes is None:
        raise HTTPException(400, "content_image is empty")

    style_bytes = await read_image_upload(style_image, "style_image")
    if style_bytes is None:
        assert style_preset is not None  # already validated above (style_image is None ⇒ style_preset must exist)
        style_path = get_style_path(style_preset)
        if not style_path or not style_path.exists():
            raise HTTPException(404, f"Style preset '{style_preset}' not found")
        style_bytes = style_path.read_bytes()

    # ── Optional second style (style mixing) ─────────────────────────────
    style_bytes_2 = await read_image_upload(style_image_2, "style_image_2")
    if style_bytes_2 is None and style_preset_2 is not None:
        style_path_2 = get_style_path(style_preset_2)
        if style_path_2 and style_path_2.exists():
            style_bytes_2 = style_path_2.read_bytes()

    # ── Optional region mask (regional styling) ──────────────────────────
    mask_bytes = await read_image_upload(mask_image, "mask_image")

    job_id = str(uuid.uuid4())

    # ── Run synchronously to avoid OOM-restart losing job state ──────────
    # On constrained environments (free tier), the async worker pattern fails
    # because the process can be killed mid-inference and the job store is lost.
    resolved_style_name = STYLE_PRESETS[resolved_style_key]["name"] if resolved_style_key else None

    _acquire_job_slot()
    try:
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
            )
        )
    except Exception:
        logger.exception("Transfer failed")
        raise HTTPException(500, "Style transfer failed. Please try again with a different image.")
    finally:
        _release_job_slot()

    # Save result to disk for the download endpoint
    out_path = OUTPUT_DIR / f"{job_id}.jpg"
    out_path.write_bytes(result_bytes)
    result_b64 = base64.b64encode(result_bytes).decode()

    # Store only lightweight metadata + disk path (NOT the base64) to save memory
    jobs[job_id] = {
        "id": job_id,
        "status": "done",
        "progress": 100,
        "result_path": str(out_path),
        "created_at": time.time(),
        "finished_at": time.time(),
    }

    # Release the raw bytes before building the response
    del result_bytes
    import gc
    gc.collect()

    return JSONResponse({
        "job_id": job_id,
        "status": "done",
        "result": result_b64,
        "resolved_style_key": resolved_style_key,
        "resolved_style_name": resolved_style_name,
    })

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    resp = {k: v for k, v in job.items() if k not in ("result", "preview")}
    if job.get("status") == "done":
        resp["result"] = job.get("result")
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
        job["status"] = "done"
        job["result_path"] = str(out_path)
        job["result"] = result_b64
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

    _acquire_job_slot()
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id, "status": "queued", "progress": 0,
        "created_at": time.time(), "result_type": "gif",
    }
    ws_clients[job_id] = []

    try:
        executor.submit(_interpolation_worker, job_id, content_bytes, style_bytes, num_frames, frame_duration)  # type: ignore[arg-type]
    except Exception:
        _release_job_slot()
        jobs.pop(job_id, None)
        ws_clients.pop(job_id, None)
        logger.exception("Failed to queue interpolation job")
        raise HTTPException(500, "Could not start the animation job. Please try again.")

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
            resolved_style_key = resolve_style_from_text(text_prompt)
            if resolved_style_key is None:
                raise HTTPException(400, "No matching style found for the provided text_prompt")
            style_preset = resolved_style_key
        else:
            raise HTTPException(400, "Provide style_image, style_preset, or text_prompt")

    strength = max(0.0, min(1.0, strength))

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

    # ── Optional second style (style mixing) ─────────────────────────────
    style_bytes_2 = await read_image_upload(style_image_2, "style_image_2")
    if style_bytes_2 is None and style_preset_2 is not None:
        style_path_2 = get_style_path(style_preset_2)
        if style_path_2 and style_path_2.exists():
            style_bytes_2 = style_path_2.read_bytes()

    # Run the CPU-heavy transfer in the worker thread so it never blocks the
    # event loop (it previously ran inline in this async handler).
    _acquire_job_slot()
    try:
        loop = asyncio.get_running_loop()
        result_bytes = await loop.run_in_executor(
            executor,
            lambda: color_palette_transfer(
                content_bytes, style_bytes, strength=strength,
                style_bytes_2=style_bytes_2, style_mix_ratio=style_mix_ratio,
            )
        )
    except Exception:
        logger.exception("Palette transfer failed")
        raise HTTPException(500, "Palette transfer failed. Please try again with a different image.")
    finally:
        _release_job_slot()
    result_b64 = base64.b64encode(result_bytes).decode()

    return JSONResponse({
        "result": result_b64,
        "resolved_style_key": resolved_style_key,
        "resolved_style_name": STYLE_PRESETS[resolved_style_key]["name"] if resolved_style_key else None,
    })

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
            "type": "done", "percent": 100, "result": job.get("result")
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
if __name__ == "__main__":
    import uvicorn  # pyre-ignore[21]
    port = int(os.environ.get("PORT", 8000))
    # proxy_headers + forwarded_allow_ips: trust X-Forwarded-For from the
    # hosting platform's proxy (Render) so rate limiting sees real client IPs.
    uvicorn.run(
        "main:app", host="0.0.0.0", port=port, reload=False, log_level="info",
        proxy_headers=True, forwarded_allow_ips="*",
    )
