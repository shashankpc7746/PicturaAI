# ── PicturaAI — Production Deployment ──────────────────────────────
# Docker image for Render / Hugging Face Spaces / any container host

FROM python:3.12-slim

# System deps for Pillow / TF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user early
RUN useradd -m -u 1000 user

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Magenta style-transfer model (~90 MB) in its own layer, so
# code changes don't trigger a re-download. model_store.py has no app imports.
ENV TFHUB_CACHE_DIR=/app/.tfhub_cache
COPY backend/model_store.py ./backend/model_store.py
RUN python backend/model_store.py

# Copy project files
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY images/style_image/ ./images/style_image/

# Create runtime directories and hand everything to the runtime user
RUN mkdir -p backend/uploads backend/outputs && chown -R user:user /app

# Switch to non-root user
USER user

# Render uses PORT env var; default to 8000 for local Docker runs
ENV PORT=8000
# Stream logs immediately (no stdout buffering)
ENV PYTHONUNBUFFERED=1
# ── Memory optimization for constrained environments ──
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1
ENV NST_MAX_DIM=320
EXPOSE 8000

# Run the FastAPI server
CMD ["python", "backend/main.py"]
