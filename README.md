# PicturaAI — Neural Style Transfer Studio

<div align="center">
  <h1>✦ PicturaAI</h1>
  <p><strong><em>Pictura</em> — Latin for "a painting." Instant Neural Style Transfer powered by Google Magenta + FastAPI</strong></p>
  <img src="images/generated images/Style-transfer-image.jpeg" alt="Demo Output" width="600" />
  <br/><br/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.16+-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-green?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/License-MIT-purple?style=for-the-badge" />
</div>

---

## 🎨 What Is PicturaAI?

**Pictura** is the Latin word for *"a painting."*  
PicturaAI is a **full-stack, production-ready** Neural Style Transfer application.  
Upload any photo, choose a famous painting style (or bring your own), and watch an AI **blend the content of your image with the brushstrokes, colours, and textures of the artwork** — in real time.

Built on top of a 1-year-old Jupyter notebook prototype, this version adds:

| Feature | Notebook | PicturaAI v2 |
|---------|----------|-------------|
| Interface | Jupyter sliders | Full web studio |
| Style options | 1 at a time | 13 presets + custom upload |
| Progress | tqdm bar | Real-time WebSocket preview |
| Deployment | Google Colab | Local / any server |
| Download | Manual | One-click JPEG download |
| Architecture | Single script | Frontend + FastAPI backend |

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- At least 4 GB RAM (8 GB recommended for quality output)

### 2. Clone / navigate to project

```bash
cd "Neural Style Transfer"
```

### 3. Create virtual environment *(already done if you see `venv/`)*

```bash
python -m venv venv
```

### 4. Install dependencies

```bash
.\venv\Scripts\pip install -r backend\requirements.txt
```

> **Note:** TensorFlow (~600 MB) will download on first install. Be patient!

### 5. Start the server

**Option A — Double-click:**
```
start_server.bat
```

**Option B — Python:**
```bash
python run.py
```

**Option C — Direct uvicorn:**
```bash
.\venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
*(run from the `backend/` directory)*

### 6. Open your browser

```
http://localhost:8000/app
```

---

## 🗂️ Project Structure

```
Neural Style Transfer/
│
├── backend/                     ← FastAPI server
│   ├── main.py                  ← API routes, WebSocket, job manager
│   ├── nst_engine.py            ← Core NST logic (Magenta pre-trained model)
│   ├── requirements.txt         ← Python dependencies
│   ├── uploads/                 ← Temp uploaded images (auto-created)
│   └── outputs/                 ← Generated result images (auto-created)
│
├── frontend/                    ← Pure HTML/CSS/JS SPA
│   ├── index.html               ← Main page (hero, studio, gallery)
│   └── assets/
│       ├── style.css            ← Dark glassmorphism design system
│       └── app.js               ← WebSocket client, upload logic, UI
│
├── images/
│   ├── content_image/           ← Sample content photos
│   ├── style_image/             ← Built-in art style images (13 styles)
│   └── generated images/        ← Past results
│
├── venv/                        ← Python virtual environment
├── run.py                       ← Python server launcher
├── start_server.bat             ← Windows batch launcher
├── NST_Manual.ipynb             ← Original prototype notebook
└── README.md                    ← This file
```

---

## 🎛️ Studio Controls

| Control | Description | Range |
|---------|-------------|-------|
| **Style Strength** | How strongly the art style is applied | 0.001 – 0.1 |
| **Content Fidelity** | How closely the output matches your photo | 1,000 – 100,000 |
| **Smoothness (TV)** | Total variation — reduces noise/grain | 0 – 100 |
| **Iterations** | More = better quality, longer time | 50 – 1,000 |
| **Learning Rate** | Adam optimizer step size | 0.005 – 0.05 |

### Recommended presets

| Goal | Style | Content | TV | Steps |
|------|-------|---------|-----|-------|
| Painterly look | 0.05 | 5,000 | 20 | 500 |
| Subtle texture | 0.005 | 20,000 | 50 | 200 |
| Maximum art | 0.1 | 1,000 | 10 | 800 |

---

## 🖼️ Available Art Styles

| Style | Artist | Character |
|-------|--------|-----------|
| Starry Night | Van Gogh | Swirling cosmic energy |
| The Scream | Munch | Anguished expressionist curves |
| The Great Wave | Hokusai | Bold Japanese woodblock |
| La Muse | Picasso | Cubist fragments |
| Rain Princess | Afremov | Rainy street in warm colour |
| Udnie | Picabia | Abstract art-deco swirls |
| The Shipwreck | Turner | Dramatic seascape |
| Aquarelle | — | Soft watercolour washes |
| Chinese Ink | Traditional | Delicate ink brush strokes |
| Space | Digital | Nebulae and cosmic textures |
| Hampson | Hampson | Bold illustrative style |
| Mountain | Nature | Rugged mountain textures |
| Paris | Photography | Parisian street atmosphere |

---

## 🤖 Technical Architecture

### NST Algorithm
- **Model:** Google Magenta's `arbitrary-image-stylization-v1-256` (pre-trained, instant)
- **Style layers:** `block1a_activation` → `block4b_activation` (7 layers)
- **Content layer:** `block5a_activation`
- **Style loss:** Mean squared error on **Gram matrices**
- **Content loss:** Mean squared error on raw feature maps
- **Regularisation:** Total Variation loss (controls grain)
- **Optimizer:** Adam (lr=0.02, β₁=0.99, ε=1e-1)

### Backend (FastAPI)
- Asynchronous job queue with `ThreadPoolExecutor` (TF not async-safe)
- **WebSocket** streaming — sends base64 preview every ~5% of steps
- REST fallback polling for environments where WS is blocked
- Automatic `uploads/` and `outputs/` cleanup on cancel

### Frontend
- **Zero framework** — vanilla HTML + CSS + JS
- WebSocket client with automatic polling fallback
- Drag-and-drop image upload with live previews
- Glassmorphism dark UI with CSS animations

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`                    | Redirect to studio |
| `GET`  | `/app`                 | Serve frontend |
| `GET`  | `/api/styles`          | List all style presets + thumbnails |
| `POST` | `/api/transfer`        | Start NST job → returns `job_id` |
| `GET`  | `/api/jobs/{id}`       | Poll job status, progress, preview |
| `GET`  | `/api/result/{id}`     | Download final JPEG |
| `DELETE` | `/api/jobs/{id}`    | Cancel & cleanup job |
| `WS`   | `/ws/{job_id}`         | Real-time progress stream |
| `GET`  | `/docs`                | Swagger UI |

---

## 🛠️ Development

```bash
# Run with hot reload
.\venv\Scripts\python -m uvicorn main:app --reload
# (from backend/ directory)

# Swagger UI (auto-generated)
http://localhost:8000/docs
```

---

## 📄 License

MIT © 2026 Shashank · PicturaAI Neural Style Transfer Studio
