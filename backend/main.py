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
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware  # pyre-ignore[21]
from fastapi.responses import FileResponse, JSONResponse  # pyre-ignore[21]
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

# ── In-memory job store ────────────────────────────────────────────────────────
jobs: Dict[str, dict] = {}
ws_clients: Dict[str, List[WebSocket]] = {}

executor = ThreadPoolExecutor(max_workers=2)

# ── Capture the main event loop at startup ─────────────────────────────────────
_main_loop: asyncio.AbstractEventLoop | None = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Modern lifespan handler (replaces deprecated on_event)."""
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    logger.info("Event loop captured for thread-safe WS broadcasting.")
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"\n\n    🎨 PicturaAI is running!\n    ➜  Frontend:  http://localhost:{port}/app\n    ➜  API docs:  http://localhost:{port}/docs\n")
    yield
    # Shutdown: clean up executor
    executor.shutdown(wait=False)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PicturaAI — Neural Style Transfer",
    description="Instant Neural Style Transfer powered by Google Magenta's pre-trained model. *Pictura* — Latin for 'a painting'.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# ── Worker ─────────────────────────────────────────────────────────────────────
def _nst_worker(
    job_id: str,
    content_bytes: bytes,
    style_bytes: bytes,
    style_weight: float,
    content_weight: float,
    tv_weight: float,
    num_steps: int,
    learning_rate: float,
    style_bytes_2: bytes | None = None,
    style_mix_ratio: float = 0.5,
    mask_bytes: bytes | None = None,
):
    job = jobs[job_id]
    job["status"] = "processing"
    job["started_at"] = time.time()

    def progress_callback(step: int, total: int, loss: float, img_bytes: bytes):
        pct = round(step / total * 100)
        preview_b64 = base64.b64encode(img_bytes).decode()
        job["progress"] = pct
        job["loss"] = loss
        job["preview"] = preview_b64
        _sync_broadcast(job_id, {
            "type": "progress",
            "step": step,
            "total": total,
            "percent": pct,
            "loss": loss,
            "preview": preview_b64,
        })

    try:
        result_bytes = run_nst(
            content_bytes, style_bytes,
            style_weight=style_weight,
            content_weight=content_weight,
            tv_weight=tv_weight,
            num_steps=num_steps,
            learning_rate=learning_rate,
            progress_callback=progress_callback,
            style_bytes_2=style_bytes_2,
            style_mix_ratio=style_mix_ratio,
            mask_bytes=mask_bytes,
        )
        out_path = OUTPUT_DIR / f"{job_id}.jpg"
        out_path.write_bytes(result_bytes)

        result_b64 = base64.b64encode(result_bytes).decode()
        job["status"] = "done"
        job["result_path"] = str(out_path)
        job["result"] = result_b64
        job["progress"] = 100
        job["finished_at"] = time.time()
        _sync_broadcast(job_id, {
            "type": "done",
            "percent": 100,
            "result": result_b64,
        })
    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        job["status"] = "error"
        job["error"] = str(e)
        _sync_broadcast(job_id, {"type": "error", "message": str(e)})

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/api/styles")
async def list_styles():
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
    return JSONResponse(result)

@app.post("/api/transfer")
async def start_transfer(
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

    content_bytes = await content_image.read()

    if style_image is not None:
        style_bytes = await style_image.read()
    else:
        assert style_preset is not None  # already validated above (style_image is None ⇒ style_preset must exist)
        style_path = get_style_path(style_preset)
        if not style_path or not style_path.exists():
            raise HTTPException(404, f"Style preset '{style_preset}' not found")
        assert style_path is not None  # narrowed: HTTPException raised if None
        style_bytes = style_path.read_bytes()

    # ── Optional second style (style mixing) ─────────────────────────────
    style_bytes_2: bytes | None = None
    if style_image_2 is not None:
        raw = await style_image_2.read()
        if len(raw) > 0:
            style_bytes_2 = raw
    elif style_preset_2 is not None:
        style_path_2 = get_style_path(style_preset_2)
        if style_path_2 and style_path_2.exists():
            style_bytes_2 = style_path_2.read_bytes()

    # ── Optional region mask (regional styling) ──────────────────────────
    mask_bytes: bytes | None = None
    if mask_image is not None:
        raw = await mask_image.read()
        if len(raw) > 0:
            mask_bytes = raw

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "created_at": time.time(),
        "style_preset": style_preset,
        "num_steps": num_steps,
    }
    ws_clients[job_id] = []

    # Run in thread pool (TF not async-friendly)
    executor.submit(  # type: ignore[arg-type]
        _nst_worker,
        job_id, content_bytes, style_bytes,
        style_weight, content_weight, tv_weight,
        num_steps, learning_rate,
        style_bytes_2, style_mix_ratio,
        mask_bytes,
    )

    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "resolved_style_key": resolved_style_key,
        "resolved_style_name": STYLE_PRESETS[resolved_style_key]["name"] if resolved_style_key else None,
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
    except Exception as e:
        logger.exception(f"Interpolation job {job_id} failed")
        job["status"] = "error"
        job["error"] = str(e)
        _sync_broadcast(job_id, {"type": "error", "message": str(e)})


@app.post("/api/interpolate")
async def start_interpolation(
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

    content_bytes = await content_image.read()

    if style_image is not None:
        style_bytes = await style_image.read()
    else:
        assert style_preset is not None
        style_path = get_style_path(style_preset)
        if not style_path or not style_path.exists():
            raise HTTPException(404, f"Style preset '{style_preset}' not found")
        style_bytes = style_path.read_bytes()

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id, "status": "queued", "progress": 0,
        "created_at": time.time(), "result_type": "gif",
    }
    ws_clients[job_id] = []

    executor.submit(_interpolation_worker, job_id, content_bytes, style_bytes, num_frames, frame_duration)  # type: ignore[arg-type]

    return JSONResponse({"job_id": job_id, "status": "queued"})


# ── Color Palette Transfer ─────────────────────────────────────────────────────

@app.post("/api/palette-transfer")
async def palette_transfer(
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
    content_bytes = await content_image.read()

    if style_image is not None:
        style_bytes = await style_image.read()
    else:
        assert style_preset is not None
        style_path = get_style_path(style_preset)
        if not style_path or not style_path.exists():
            raise HTTPException(404, f"Style preset '{style_preset}' not found")
        style_bytes = style_path.read_bytes()

    # ── Optional second style (style mixing) ─────────────────────────────
    style_bytes_2: bytes | None = None
    if style_image_2 is not None:
        raw = await style_image_2.read()
        if len(raw) > 0:
            style_bytes_2 = raw
    elif style_preset_2 is not None:
        style_path_2 = get_style_path(style_preset_2)
        if style_path_2 and style_path_2.exists():
            style_bytes_2 = style_path_2.read_bytes()

    result_bytes = color_palette_transfer(
        content_bytes, style_bytes, strength=strength,
        style_bytes_2=style_bytes_2, style_mix_ratio=style_mix_ratio,
    )
    result_b64 = base64.b64encode(result_bytes).decode()

    return JSONResponse({
        "result": result_b64,
        "resolved_style_key": resolved_style_key,
        "resolved_style_name": STYLE_PRESETS[resolved_style_key]["name"] if resolved_style_key else None,
    })

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
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
        if job_id in ws_clients and websocket in ws_clients[job_id]:
            ws_clients[job_id].remove(websocket)

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn  # pyre-ignore[21]
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, log_level="info")
