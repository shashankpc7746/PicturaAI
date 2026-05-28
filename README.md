# PicturaAI

<div align="center">
  <img src="frontend/assets/logo.png" alt="PicturaAI Logo" width="120" />
  <h1>PicturaAI</h1>
  <p><strong>Neural Style Transfer Studio for turning photos into artwork.</strong></p>
  <p>Pictura means a painting. Upload a photo, choose or describe a style, and generate stylized artwork in a single flow.</p>

  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Frontend-Vanilla%20JS-222222?style=for-the-badge&logo=javascript&logoColor=F7DF1E" />
  <img src="https://img.shields.io/badge/Deploy-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" />
</div>

---

## Overview

PicturaAI is a full-stack Neural Style Transfer application built around Google Magenta's arbitrary image stylization model. It combines a FastAPI backend, a vanilla HTML/CSS/JavaScript frontend, and a quality-focused post-processing pipeline to produce stylized images quickly while preserving the structure of the original photo.

The project is designed as a complete interactive studio rather than a simple model demo. Users can upload custom style images, mix two styles, paint regional masks, transfer only the color palette, generate interpolation GIFs, revisit previous generations, and describe a style through text.

---

## Features

| Category | What's Included |
|---|---|
| Transfer modes | Standard NST, palette-only transfer, text-to-style preset matching |
| Creative controls | Style mixing, regional masking, style intensity slider |
| Output tools | Before/after compare, download, animation GIF, history strip |
| Runtime UX | Live WebSocket progress, REST fallback, preset gallery |

---

## Architecture

```text
PicturaAI/
├── backend/
│   ├── main.py              # FastAPI app, routes, WebSocket, job management
│   ├── nst_engine.py        # TF Hub model, image pipeline, post-processing
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html           # Single-page studio UI
│   └── assets/
│       ├── app.js           # Client logic, WS, uploads, history
│       ├── style.css        # Dark glassmorphism theme
│       └── logo.png / favicons
├── images/
│   └── style_image/         # 13 built-in style presets
├── Dockerfile               # Production container (Render / HF Spaces)
├── render.yaml              # Render deploy blueprint
└── run.py                   # Local dev launcher
```

---

## Quick Start (Local)

### Prerequisites

- Python 3.10+
- 4 GB RAM minimum
- Internet connection (first run downloads the TF Hub model ~100 MB)

### Setup

```bash
git clone https://github.com/shashankpc7746/PicturaAI.git
cd PicturaAI
python -m venv venv

# Windows
venv\Scripts\pip install -r backend\requirements.txt

# macOS / Linux
venv/bin/pip install -r backend/requirements.txt
```

### Run

```bash
python run.py
```

Open **http://localhost:8000/app** in your browser.

The first request triggers model download and caching. Subsequent starts are instant.

---

## Deployment on Render

Render is the recommended hosting platform for PicturaAI. It supports Docker, persistent storage, and the memory needed for TensorFlow inference.

### Option A: One-Click Deploy

1. Push this repo to GitHub.
2. Go to [render.com/deploy](https://render.com/deploy).
3. Connect your GitHub repo — Render auto-detects `render.yaml`.
4. Click **Deploy**. Done.

### Option B: Manual Setup

1. Sign up at [render.com](https://render.com).
2. Click **New → Web Service**.
3. Connect your GitHub repository.
4. Configure:
   - **Environment:** Docker
   - **Plan:** Free (512 MB RAM) — sufficient for TF inference
   - **Health Check Path:** `/api/styles`
5. Add environment variables (optional, for cleaner logs):
   - `TF_CPP_MIN_LOG_LEVEL` = `2`
   - `TF_ENABLE_ONEDNN_OPTS` = `0`
6. Click **Create Web Service**.

### What happens during deploy

1. Render builds the Docker image (~5 min first time).
2. The Dockerfile pre-downloads the Magenta model during build.
3. Your app starts on the assigned URL (e.g., `https://picturaai.onrender.com`).
4. The frontend is served at `/app`, API docs at `/docs`.

### Important notes for Render

- **Cold starts:** On the free plan, Render spins down after 15 min of inactivity. First request after sleep takes ~30–60s (model reload).
- **RAM:** TensorFlow + model needs ~400 MB. The free plan (512 MB) is tight but works. Upgrade to Starter ($7/mo) for always-on.
- **Build time:** First deploy takes ~8–10 min (model download during Docker build). Subsequent deploys are faster due to layer caching.
- **Automation:** You can install the [Render MCP Server](https://render.com/docs/mcp-server) to manage deployments directly from your AI editor.

---

## Why Render over Vercel?

| Concern | Render | Vercel |
|---|---|---|
| Long-running inference (2–5s) | ✅ Supported | ❌ 10s serverless timeout |
| TensorFlow + large model | ✅ Docker, persistent filesystem | ❌ 250 MB function size limit |
| WebSocket connections | ✅ Native support | ❌ Not supported in serverless |
| Docker deployment | ✅ First-class | ❌ Not supported |
| Persistent uploads/outputs | ✅ Disk storage | ❌ Ephemeral filesystem |

Vercel is great for static sites and edge functions, but PicturaAI needs a real server with persistent state, WebSockets, and enough memory for TensorFlow.

---

## API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Entry page |
| GET | `/app` | Frontend studio |
| GET | `/api/styles` | Built-in styles with thumbnails |
| POST | `/api/transfer` | Start style transfer job |
| GET | `/api/jobs/{job_id}` | Job status and progress |
| GET | `/api/result/{job_id}` | Download result JPEG |
| DELETE | `/api/jobs/{job_id}` | Remove job |
| POST | `/api/interpolate` | Generate interpolation GIF |
| POST | `/api/palette-transfer` | Color palette transfer |
| WS | `/ws/{job_id}` | Live progress stream |
| GET | `/docs` | Swagger UI |

### Transfer inputs

- `content_image` — your photo (required)
- `style_image` or `style_preset` or `text_prompt` — the style source
- `style_weight` — intensity 0.0–1.0
- `style_image_2` / `style_preset_2` — optional second style for mixing
- `style_mix_ratio` — blend ratio between two styles
- `mask_image` — optional regional mask (white = apply style)

---

## Built-In Styles

| Style | Artist | Character |
|---|---|---|
| Starry Night | Van Gogh | Swirling cosmic energy |
| The Scream | Munch | Expressive dramatic curves |
| The Great Wave | Hokusai | Japanese woodblock print |
| La Muse | Picasso | Cubist geometry |
| Rain Princess | Afremov | Warm rainy reflections |
| Udnie | Picabia | Abstract dynamic motion |
| The Shipwreck | Turner | Stormy seascape |
| Aquarelle | Unknown | Soft watercolor wash |
| Chinese Ink | Traditional | Brush and ink minimalism |
| Space | Digital | Cosmic nebula textures |
| Hampson | Illustration | Bold graphic style |
| Mountain | Nature | Rugged landscape |
| Paris | Photography | Urban atmosphere |

---

## Development

```bash
# Run with auto-reload
cd backend
../venv/Scripts/python -m uvicorn main:app --reload

# View API docs
# http://localhost:8000/docs
```

---

## License

MIT © 2026 Shashank
