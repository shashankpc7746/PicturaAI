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
</div>

---

## Overview

PicturaAI is a full-stack Neural Style Transfer application built around Google Magenta's arbitrary image stylization model. It combines a FastAPI backend, a vanilla HTML/CSS/JavaScript frontend, and a quality-focused post-processing pipeline to produce stylized images quickly while preserving the structure of the original photo.

The project is designed as a complete interactive studio rather than a simple model demo. Users can upload custom style images, mix two styles, paint regional masks, transfer only the color palette, generate interpolation GIFs, revisit previous generations, and describe a style through text.

## At a Glance

| Category | Included |
|---|---|
| Transfer modes | Standard NST, palette-only transfer, text-to-style preset matching |
| Creative controls | Style mixing, regional masking, style intensity |
| Output tools | Before/after compare, download, animation GIF, history |
| Runtime UX | Live WebSocket progress, REST fallback, preset gallery |

---

## What PicturaAI Can Do

- Generate stylized artwork from a content photo and one selected style.
- Use 13 built-in presets drawn from iconic visual styles.
- Accept custom uploaded style images.
- Blend two styles with an adjustable mix ratio.
- Paint a regional mask so style is applied only where needed.
- Transfer only the color palette while keeping structure and texture intact.
- Create looping style interpolation GIFs.
- Store recent results in a generation history strip.
- Compare original and result using a before/after slider.
- Resolve text prompts like "watercolor portrait" or "cosmic night sky" to the closest built-in preset.
- Stream live progress updates through WebSockets with polling fallback.
- Download the final image directly from the UI.

---

## Feature Breakdown

### 1. Core Style Transfer

The main pipeline uses Magenta's arbitrary image stylization model for fast forward-pass inference. This keeps generation responsive while still producing visually rich results.

### 2. Style Mixing

Users can pick a second style and control the ratio between style A and style B. This enables composite outputs such as mixing Van Gogh-like motion with Picasso-style geometry.

### 3. Regional Styling

The app supports mask painting over the content image. Styled regions follow the mask while untouched regions preserve the original content. This is useful for selective edits such as stylizing only the background, sky, clothing, or foreground subject.

### 4. Color Palette Transfer

Palette-only mode transfers color characteristics from the selected style into the content image without applying full texture stylization. This uses LAB-space mean and standard deviation matching for fast, model-free color adaptation.

### 5. Text-to-Style

Users can type a style prompt directly in the studio. The backend resolves that prompt to the closest matching preset using keyword scoring over preset names, artists, descriptions, and style tags. This gives the user a natural-language entry point without requiring a separate text-to-image model.

### 6. Style Interpolation Animation

PicturaAI can generate a looping GIF that sweeps style intensity from low to high and back again. This is useful for demos, social sharing, and visually comparing how stylization evolves across the same image.

### 7. Generation History

Recent generations are stored in a lightweight gallery strip so users can jump back to earlier results, compare variants, and re-download outputs without re-running the model.

---

## Built-In Styles

| Style | Artist / Source | Visual Character |
|---|---|---|
| Starry Night | Van Gogh | Swirling, energetic night-sky motion |
| The Scream | Munch | Expressive and dramatic curves |
| The Great Wave | Hokusai | Strong wave forms and Japanese print texture |
| La Muse | Picasso | Cubist, fragmented geometry |
| Rain Princess | Afremov | Reflective rainy streets and warm color |
| Udnie | Picabia | Abstract, dynamic motion |
| The Shipwreck | Turner | Stormy seascape energy |
| Aquarelle | Unknown | Soft watercolor wash |
| Chinese Ink | Traditional | Brushwork and ink minimalism |
| Space | Digital | Cosmic glow and nebula textures |
| Hampson | Illustration | Bold graphic stylization |
| Mountain | Nature | Rugged earthy landscape feel |
| Paris | Photography | Urban street atmosphere |

---

## Architecture

### Backend

- FastAPI application serving REST endpoints and WebSocket updates.
- Thread pool execution for TensorFlow work.
- Built-in style preset registry with thumbnail generation.
- Prompt-to-style resolver for text-driven preset selection.
- Palette transfer and interpolation endpoints in addition to the main NST pipeline.

### Frontend

- Single-page interface built with vanilla HTML, CSS, and JavaScript.
- Studio workflow for content upload, style selection, prompt entry, mask painting, generation, comparison, history, and download.
- WebSocket client for progress streaming with automatic polling fallback.

### Model and Image Pipeline

- Google Magenta Arbitrary Style Transfer via TensorFlow Hub.
- Content and style preprocessing.
- Optional second-style blending.
- Optional regional mask compositing.
- Luminance-preserving blend and detail reinjection.
- Final sharpening and JPEG export.

---

## Project Structure

```text
PicturaAI/
├── backend/
│   ├── main.py
│   ├── nst_engine.py
│   ├── requirements.txt
│   ├── uploads/
│   └── outputs/
├── frontend/
│   ├── index.html
│   └── assets/
│       ├── app.js
│       ├── style.css
│       ├── logo.png
│       └── favicon.ico
├── images/
│   ├── content_image/
│   ├── generated images/
│   └── style_image/
├── NST_Manual.ipynb
├── Dockerfile
├── run.py
├── start_server.bat
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10 or newer
- 4 GB RAM minimum
- Internet connection for first model download

### Clone the repository

```bash
git clone https://github.com/shashankpc7746/PicturaAI.git
cd PicturaAI
```

### Create a virtual environment

```bash
python -m venv venv
```

### Install dependencies

Windows:

```bash
.\venv\Scripts\pip install -r backend\requirements.txt
```

macOS / Linux:

```bash
venv/bin/pip install -r backend/requirements.txt
```

### Start the app

Windows helper:

```bash
start_server.bat
```

Python launcher:

```bash
python run.py
```

Direct backend run:

```bash
cd backend
../venv/Scripts/python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Open in browser

```text
http://localhost:8000/app
```

The first run downloads the TensorFlow Hub model and caches it locally.

---

## Studio Workflow

1. Upload a content image.
2. Choose a preset, upload a custom style, or enter a text prompt.
3. Optionally enable style mixing, palette transfer, or mask painting.
4. Generate the result and watch live progress.
5. Compare, download, animate, or revisit the output from history.

---

## API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | / | Serves the main entry page |
| GET | /app | Serves the frontend studio |
| GET | /api/styles | Returns built-in styles with metadata and thumbnails |
| POST | /api/transfer | Starts a style transfer job |
| GET | /api/jobs/{job_id} | Returns job status, progress, and preview |
| GET | /api/result/{job_id} | Downloads a completed JPEG result |
| DELETE | /api/jobs/{job_id} | Removes a job and its result |
| POST | /api/interpolate | Generates an interpolation GIF |
| POST | /api/palette-transfer | Performs palette-only transfer |
| WS | /ws/{job_id} | Streams live progress for a running job |
| GET | /docs | Swagger UI |

### Main transfer inputs

- `content_image`
- `style_image` or `style_preset` or `text_prompt`
- `style_weight`
- `style_image_2`
- `style_preset_2`
- `style_mix_ratio`
- `mask_image`

### Example: preset-based transfer

```bash
curl -X POST http://localhost:8000/api/transfer \
  -F "content_image=@photo.jpg" \
  -F "style_preset=starry_night" \
  -F "style_weight=0.75"
```

### Example: text-to-style transfer

```bash
curl -X POST http://localhost:8000/api/transfer \
  -F "content_image=@photo.jpg" \
  -F "text_prompt=watercolor portrait with soft pastel tones" \
  -F "style_weight=0.75"
```

### Example: palette-only transfer

```bash
curl -X POST http://localhost:8000/api/palette-transfer \
  -F "content_image=@photo.jpg" \
  -F "text_prompt=cosmic blue nebula" \
  -F "strength=0.85"
```

---

## Quality Pipeline

PicturaAI does more than run a raw model pass. The image pipeline is tuned to keep the result visually sharp and readable.

1. Content is resized for efficient inference.
2. Style is applied through Magenta's arbitrary style transfer model.
3. Optional style mixing combines style statistics from two style sources.
4. Optional regional mask blending limits where style appears.
5. Luminance is preserved to protect subject structure.
6. Fine details are reintroduced from the original content image.
7. Final sharpening improves output crispness.

---

## Development

### Run with auto-reload

```bash
cd backend
../venv/Scripts/python -m uvicorn main:app --reload
```

### View API docs

```text
http://localhost:8000/docs
```

---

## Deployment

### Docker

```bash
docker build -t picturaai .
docker run -p 8000:8000 picturaai
```

### Generic cloud deployment

1. Push the repository to GitHub.
2. Connect the repository to your hosting platform.
3. Set the build command to install backend requirements.
4. Set the start command to run the backend server.
5. Expose port 8000.

---

## Notes

- Text-to-Style currently maps prompts to the nearest built-in preset. It does not generate a brand-new style image.
- The first launch is slower because TensorFlow Hub downloads and caches the model.
- GIF interpolation and high-resolution transfers can take noticeably longer on CPU-only systems.
- Palette transfer changes color statistics only. It does not copy brush texture or composition.

---

## Changelog

### v1.5

- Cleaned and expanded project documentation.
- Documented Text-to-Style as a first-class feature.
- Clarified architecture, workflow, API usage, and deployment.

### v1.4

- Added Text-to-Style prompt input in the studio.
- Added backend prompt matcher for built-in style presets.
- Added `text_prompt` support to transfer and palette APIs.

### v1.3

- Added style interpolation animation.
- Added color palette transfer mode.

### v1.2

- Added before/after slider.
- Added style mixing.
- Added regional styling.
- Added generation history.

### v1.1

- Improved output quality and sharpness.
- Raised processing resolution.
- Improved progress pipeline and UI polish.

### v1.0

- Initial full-stack release.

---

## License

MIT © 2026 Shashank
