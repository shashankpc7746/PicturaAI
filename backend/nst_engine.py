"""
Neural Style Transfer Engine — PicturaAI
==========================================
Fast feed-forward NST using Google Magenta's pre-trained
Arbitrary Image Stylization model (TF Hub).

One forward pass ≈ 2–5 seconds on CPU, <1 second on GPU.
No iterative optimisation needed — instant results.
"""

import tensorflow as tf  # pyre-ignore[21]
import tensorflow_hub as hub  # pyre-ignore[21]
import numpy as np  # pyre-ignore[21]
from PIL import Image, __version__ as PIL_VERSION  # pyre-ignore[21]
import io
import time
import logging
import os
from typing import Callable, Optional

# Suppress non-critical TensorFlow warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide INFO & WARNING from TF C++
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # suppress oneDNN info messages
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
tf.get_logger().setLevel("ERROR")  # suppress TF Python-level warnings

logger = logging.getLogger("nst_engine")

# Pillow 10+ moved LANCZOS to Image.Resampling
_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS  # type: ignore[attr-defined]

# ── Constants ──────────────────────────────────────────────────────────────────
CONTENT_MAX_DIM = 512       # Max dimension for the content image
STYLE_IMG_SIZE  = 256       # Style model expects 256×256 (recommended)
HUB_MODEL_URL = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"

# ── Image utilities ────────────────────────────────────────────────────────────

def load_and_preprocess(
    image_bytes: bytes,
    target_size: Optional[tuple] = None,
    max_dim: int = CONTENT_MAX_DIM,
) -> tf.Tensor:
    """
    Load image from raw bytes → [1, H, W, 3] float32 tensor in [0, 1].

    If target_size is given (H, W), resize to exactly that.
    Otherwise, resize preserving aspect ratio so the longest side ≤ max_dim.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if target_size:
        img = img.resize((target_size[1], target_size[0]), _LANCZOS)
    else:
        w, h = img.size
        scale = max_dim / max(h, w)
        if scale < 1.0:  # only downscale, never upscale
            img = img.resize((int(w * scale), int(h * scale)), _LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0
    return tf.expand_dims(tf.constant(arr), axis=0)


def tensor_to_pil(tensor: tf.Tensor) -> Image.Image:
    """Convert a [1, H, W, 3] float tensor (0–1) → PIL Image."""
    t = tensor.numpy()  # type: ignore[union-attr]
    if t.ndim == 4:
        t = t[0]
    t = np.clip(t * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(t)


def pil_to_bytes(img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()


# ── Pre-trained Model (Singleton) ──────────────────────────────────────────────

class StyleTransferModel:
    """
    Singleton wrapper around Google Magenta's Arbitrary Image Stylization model.

    On first call, the TF Hub module is downloaded (~100 MB) and cached locally.
    Subsequent calls load from cache instantly. The model itself runs a single
    forward pass — no iterative optimisation, no gradient descent.
    """
    _instance: Optional["StyleTransferModel"] = None
    _model = None

    def __init__(self):
        logger.info("Loading Magenta Arbitrary Style Transfer model from TF Hub …")
        t0 = time.time()
        self._model = hub.load(HUB_MODEL_URL)
        elapsed = time.time() - t0
        logger.info(f"Model loaded in {elapsed:.1f}s — ready for instant style transfer!")

    @classmethod
    def get(cls) -> "StyleTransferModel":
        instance = cls._instance
        if instance is None:
            instance = cls()
            cls._instance = instance
        return instance  # pyre-ignore[7]

    def stylize(
        self,
        content_tensor: tf.Tensor,
        style_tensor: tf.Tensor,
    ) -> tf.Tensor:
        """
        Run style transfer in a single forward pass.

        Args:
            content_tensor: [1, H, W, 3] float32 in [0, 1]
            style_tensor:   [1, 256, 256, 3] float32 in [0, 1]

        Returns:
            [1, H, W, 3] stylized image tensor
        """
        outputs = self._model(tf.constant(content_tensor), tf.constant(style_tensor))  # type: ignore[misc]  # pyre-ignore[29]
        return outputs[0]


# ── Main API (backward-compatible signature) ───────────────────────────────────

def run_nst(
    content_bytes: bytes,
    style_bytes: bytes,
    style_weight: float = 1e-2,
    content_weight: float = 1e4,
    tv_weight: float = 30.0,
    num_steps: int = 300,
    learning_rate: float = 0.02,
    progress_callback: Optional[Callable[[int, int, float, bytes], None]] = None,
) -> bytes:
    """
    Run Neural Style Transfer using the pre-trained Magenta model.

    The style_weight, content_weight, tv_weight, num_steps, and learning_rate
    parameters are accepted for backward compatibility with the API but are
    NOT used — the pre-trained model handles everything in one pass.

    Args:
        content_bytes:     Raw bytes of the content image.
        style_bytes:       Raw bytes of the style image.
        progress_callback: Called with (step, total, loss, jpg_bytes).

    Returns:
        JPEG bytes of the stylized image.
    """
    model = StyleTransferModel.get()
    total_steps = 3  # We report 3 phases to keep the frontend progress bar useful

    # ── Phase 1: Preprocessing ──────────────────────────────────────────────
    if progress_callback is not None:
        progress_callback(0, total_steps, 0.0, b"")

    logger.info("Preprocessing content & style images …")
    t0 = time.time()
    content_tensor = load_and_preprocess(content_bytes, max_dim=CONTENT_MAX_DIM)
    style_tensor = load_and_preprocess(
        style_bytes,
        target_size=(STYLE_IMG_SIZE, STYLE_IMG_SIZE),
    )
    # Light blur on style image for better stylisation (recommended by TF docs)
    style_tensor = tf.nn.avg_pool(style_tensor, ksize=[3, 3], strides=[1, 1], padding="SAME")

    if progress_callback is not None:
        progress_callback(1, total_steps, 0.0, b"")

    # ── Phase 2: Style Transfer (single forward pass!) ──────────────────────
    logger.info("Running style transfer (single forward pass) …")
    stylized_tensor = model.stylize(content_tensor, style_tensor)

    # ── Phase 2.5: Blend with content using style_weight as alpha ────────────
    # style_weight = 0.0 → pure content, 1.0 → fully stylized
    alpha = max(0.0, min(1.0, style_weight))
    logger.info(f"Blending: {alpha*100:.0f}% style + {(1-alpha)*100:.0f}% content")

    # The model may output slightly different dimensions due to internal padding.
    # Resize stylized output to match content tensor shape before blending.
    content_shape = tf.shape(content_tensor)[1:3]  # type: ignore[index]  # [H, W]
    stylized_tensor = tf.image.resize(stylized_tensor, content_shape)

    blended_tensor = alpha * stylized_tensor + (1.0 - alpha) * content_tensor  # type: ignore[operator]

    elapsed = time.time() - t0
    logger.info(f"Style transfer complete in {elapsed:.1f}s")

    if progress_callback is not None:
        pil_result = tensor_to_pil(blended_tensor)
        preview_bytes = pil_to_bytes(pil_result, quality=75)
        progress_callback(2, total_steps, 0.0, preview_bytes)

    # ── Phase 3: Final output ───────────────────────────────────────────────
    pil_result = tensor_to_pil(blended_tensor)
    result_bytes = pil_to_bytes(pil_result, quality=95)

    if progress_callback is not None:
        progress_callback(total_steps, total_steps, 0.0, result_bytes)

    logger.info(f"Total pipeline: {time.time() - t0:.1f}s | output: {len(result_bytes)/1024:.0f} KB")
    return result_bytes
