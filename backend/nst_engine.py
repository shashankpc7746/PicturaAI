"""
Neural Style Transfer Engine — PicturaAI
==========================================
Fast feed-forward NST using Google Magenta's pre-trained
Arbitrary Image Stylization model (TF Hub).

One forward pass ≈ 2–5 seconds on CPU, <1 second on GPU.
No iterative optimisation needed — instant results.
"""

import io
import logging
import os
import time
import warnings
from typing import Callable, Optional

# Suppress non-critical TensorFlow warnings — MUST be set BEFORE importing TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = os.environ.get("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf  # pyre-ignore[21]  # noqa: E402
import tensorflow_hub as hub  # pyre-ignore[21]  # noqa: E402
import numpy as np  # pyre-ignore[21]  # noqa: E402
from PIL import Image, __version__ as PIL_VERSION  # pyre-ignore[21]  # noqa: E402

tf.get_logger().setLevel("ERROR")  # suppress TF Python-level warnings

logger = logging.getLogger("nst_engine")

# Pillow 10+ moved LANCZOS to Image.Resampling
_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS  # type: ignore[attr-defined]
from PIL import ImageFilter  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
CONTENT_MAX_DIM = 768       # Raised from 512 for sharper output
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


# ── Detail-preservation & post-processing utilities ────────────────────────────

def _rgb_to_luminance(tensor: tf.Tensor) -> tf.Tensor:
    """Extract perceptual luminance (Rec. 709) from [1,H,W,3] RGB tensor."""
    return tf.reduce_sum(
        tensor * tf.constant([[[0.2126, 0.7152, 0.0722]]]),  # type: ignore[operator]
        axis=-1, keepdims=True,
    )


def _extract_high_freq(tensor: tf.Tensor, sigma: int = 3) -> tf.Tensor:
    """Return the high-frequency detail layer (tensor − blurred)."""
    blurred = tf.nn.avg_pool(tensor, ksize=[sigma, sigma], strides=[1, 1], padding="SAME")
    return tensor - blurred  # type: ignore[operator]


def _luminance_preserving_blend(
    content: tf.Tensor,
    stylized: tf.Tensor,
    alpha: float,
) -> tf.Tensor:
    """
    Blend that preserves content luminance structure.
    At any alpha, the luminance channel is always taken from content,
    while chrominance (colour) smoothly shifts toward the style.
    This keeps edges and structures sharp even at 100 % style.
    """
    # Simple alpha blend for colour
    blended = alpha * stylized + (1.0 - alpha) * content  # type: ignore[operator]

    # Replace luminance of blended with a mix biased toward content luminance
    lum_content = _rgb_to_luminance(content)
    lum_blended = _rgb_to_luminance(blended)

    # How much content luminance to keep: at alpha=1 keep 40%, at alpha=0 keep 100%
    lum_keep = 1.0 - alpha * 0.6
    target_lum = lum_keep * lum_content + (1.0 - lum_keep) * lum_blended  # type: ignore[operator]

    # Adjust blended so its luminance matches the target
    ratio = target_lum / (lum_blended + 1e-7)  # type: ignore[operator]
    result = blended * ratio  # type: ignore[operator]
    return tf.clip_by_value(result, 0.0, 1.0)  # type: ignore[return-value]


def _reinject_details(
    content: tf.Tensor,
    stylized: tf.Tensor,
    strength: float = 0.35,
) -> tf.Tensor:
    """
    Extract high-frequency details from the content image and add them
    back into the stylized image to recover edges and fine textures.
    """
    hf = _extract_high_freq(content, sigma=5)
    return tf.clip_by_value(stylized + strength * hf, 0.0, 1.0)  # type: ignore[operator]


def _unsharp_mask(img: Image.Image, radius: float = 1.5, percent: int = 60, threshold: int = 2) -> Image.Image:
    """Apply PIL UnsharpMask for final edge crispness."""
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


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
    style_bytes_2: Optional[bytes] = None,
    style_mix_ratio: float = 0.5,
    mask_bytes: Optional[bytes] = None,
) -> bytes:
    """
    Run Neural Style Transfer using the pre-trained Magenta model.

    The style_weight, content_weight, tv_weight, num_steps, and learning_rate
    parameters are accepted for backward compatibility with the API but are
    NOT used — the pre-trained model handles everything in one pass.

    Args:
        content_bytes:     Raw bytes of the content image.
        style_bytes:       Raw bytes of the primary style image.
        style_bytes_2:     Optional raw bytes of a second style image (style mixing).
        style_mix_ratio:   Blend ratio 0–1. 0 = 100 % Style A, 1 = 100 % Style B.
        progress_callback: Called with (step, total, loss, jpg_bytes).

    Returns:
        JPEG bytes of the stylized image.
    """
    model = StyleTransferModel.get()
    total_steps = 4  # 4 phases: preprocess, stylize, enhance, finalize

    # ── Phase 1: Preprocessing ──────────────────────────────────────────────
    if progress_callback is not None:
        progress_callback(0, total_steps, 0.0, b"")

    mixing = style_bytes_2 is not None
    logger.info("Preprocessing content & style images%s …", " (mixing 2 styles)" if mixing else "")
    t0 = time.time()
    content_tensor = load_and_preprocess(content_bytes, max_dim=CONTENT_MAX_DIM)
    style_tensor = load_and_preprocess(
        style_bytes,
        target_size=(STYLE_IMG_SIZE, STYLE_IMG_SIZE),
    )
    # Light blur on style image for better stylisation (recommended by TF docs)
    style_tensor = tf.nn.avg_pool(style_tensor, ksize=[3, 3], strides=[1, 1], padding="SAME")

    # ── Style Mixing: blend two style tensors if a second style is provided ──
    if mixing:
        style_tensor_2 = load_and_preprocess(
            style_bytes_2,
            target_size=(STYLE_IMG_SIZE, STYLE_IMG_SIZE),
        )
        style_tensor_2 = tf.nn.avg_pool(style_tensor_2, ksize=[3, 3], strides=[1, 1], padding="SAME")
        ratio = max(0.0, min(1.0, style_mix_ratio))
        style_tensor = (1.0 - ratio) * style_tensor + ratio * style_tensor_2  # type: ignore[operator]
        logger.info(f"Style mix: {(1-ratio)*100:.0f}% Style A + {ratio*100:.0f}% Style B")

    if progress_callback is not None:
        progress_callback(1, total_steps, 0.0, b"")

    # ── Phase 2: Style Transfer (single forward pass!) ──────────────────────
    logger.info("Running style transfer (single forward pass) …")
    stylized_tensor = model.stylize(content_tensor, style_tensor)

    # Resize stylized output to match content tensor shape before blending.
    content_shape = tf.shape(content_tensor)[1:3]  # type: ignore[index]  # [H, W]
    stylized_tensor = tf.image.resize(stylized_tensor, content_shape)

    if progress_callback is not None:
        pil_preview = tensor_to_pil(stylized_tensor)
        progress_callback(2, total_steps, 0.0, pil_to_bytes(pil_preview, quality=60))

    # ── Phase 3: Detail-preserving blend + enhancement ──────────────────────
    alpha = max(0.0, min(1.0, style_weight))
    logger.info(f"Blending: {alpha*100:.0f}% style + {(1-alpha)*100:.0f}% content")

    # 3a. Luminance-preserving blend keeps content structure sharp
    blended_tensor = _luminance_preserving_blend(content_tensor, stylized_tensor, alpha)

    # 3b. Re-inject high-frequency content details (edges, textures)
    #     Strength scales with alpha: more detail injection at higher style
    detail_strength = 0.15 + alpha * 0.30  # 0.15 at 0%, 0.45 at 100%
    blended_tensor = _reinject_details(content_tensor, blended_tensor, strength=detail_strength)

    # 3c. Regional mask: if provided, only apply style where mask is white
    if mask_bytes is not None:
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        # Resize mask to match content tensor spatial dimensions
        content_shape = tf.shape(content_tensor)  # type: ignore[arg-type]
        h, w = int(content_shape[1]), int(content_shape[2])
        mask_img = mask_img.resize((w, h), _LANCZOS)
        # Light Gaussian blur to soften mask edges (avoid harsh boundaries)
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=6))
        mask_arr = np.array(mask_img, dtype=np.float32) / 255.0
        mask_tensor = tf.reshape(tf.constant(mask_arr), [1, h, w, 1])
        blended_tensor = mask_tensor * blended_tensor + (1.0 - mask_tensor) * content_tensor  # type: ignore[operator]
        blended_tensor = tf.clip_by_value(blended_tensor, 0.0, 1.0)  # type: ignore[assignment]
        logger.info("Regional mask applied — style restricted to painted areas")

    elapsed = time.time() - t0
    logger.info(f"Style transfer + enhancement in {elapsed:.1f}s")

    if progress_callback is not None:
        pil_result = tensor_to_pil(blended_tensor)  # type: ignore[arg-type]
        preview_bytes = pil_to_bytes(pil_result, quality=75)
        progress_callback(3, total_steps, 0.0, preview_bytes)

    # ── Phase 4: Final output with sharpening ───────────────────────────────
    pil_result = tensor_to_pil(blended_tensor)  # type: ignore[arg-type]

    # Adaptive unsharp mask: sharpen more at higher style intensity
    sharp_pct = int(40 + alpha * 50)  # 40% at low style → 90% at full style
    pil_result = _unsharp_mask(pil_result, radius=1.2, percent=sharp_pct, threshold=2)

    result_bytes = pil_to_bytes(pil_result, quality=95)

    if progress_callback is not None:
        progress_callback(total_steps, total_steps, 0.0, result_bytes)

    logger.info(f"Total pipeline: {time.time() - t0:.1f}s | output: {len(result_bytes)/1024:.0f} KB")
    return result_bytes
