# ── PicturaAI — Production Deployment ──────────────────────────────
# Docker image for Render / Hugging Face Spaces / any container host

FROM python:3.10-slim

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

# Copy project files
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY images/style_image/ ./images/style_image/

# Create runtime directories
RUN mkdir -p backend/uploads backend/outputs

# Pre-download the Magenta model into a persistent path accessible by user
ENV TFHUB_CACHE_DIR=/app/.tfhub_cache
RUN mkdir -p /app/.tfhub_cache && \
    python -c "import os; os.environ['TFHUB_CACHE_DIR']='/app/.tfhub_cache'; import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'); print('Model cached!')" && \
    chown -R user:user /app

# Switch to non-root user
USER user

# Render uses PORT env var; default to 8000 for local Docker runs
ENV PORT=8000
EXPOSE 8000

# Run the FastAPI server
CMD ["python", "backend/main.py"]
