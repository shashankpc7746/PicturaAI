# ── PicturaAI — Hugging Face Spaces Deployment ─────────────────────
# Docker-based HF Space running FastAPI + TensorFlow Hub (Magenta)

FROM python:3.10-slim

# System deps for Pillow / TF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies first (cached layer)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY images/ ./images/

# Pre-download the Magenta model so first request is fast
RUN python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'); print('Model cached!')"

# Switch to non-root user
USER user

# HF Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "backend/main.py"]
