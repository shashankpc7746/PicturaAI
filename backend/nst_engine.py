"""
Neural Style Transfer Engine
==============================
Production-grade NST using EfficientNetB0 feature extraction.
Supports real-time progress streaming via callback.
"""

import tensorflow as tf  # pyre-ignore[21]
import numpy as np  # pyre-ignore[21]
from PIL import Image  # pyre-ignore[21]
import io
import time
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── Layer configuration ────────────────────────────────────────────────────────
STYLE_LAYER_NAMES = [
    "block1a_activation",
    "block2a_activation",
    "block2b_activation",
    "block3a_activation",
    "block3b_activation",
    "block4a_activation",
    "block4b_activation",
]
CONTENT_LAYER_NAME = ["block5a_activation"]
NUM_STYLE_LAYERS = len(STYLE_LAYER_NAMES)

# ── Image utilities ────────────────────────────────────────────────────────────
MAX_DIM = 512   # Max dimension for processing (keeps it fast)

def load_and_preprocess(image_bytes: bytes, max_dim: int = MAX_DIM) -> tf.Tensor:
    """Load image from bytes, resize preserving aspect ratio, normalize to [0,1]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return tf.expand_dims(tf.constant(arr), axis=0)  # shape: [1, H, W, 3]

def tensor_to_pil(tensor: tf.Tensor) -> Image.Image:
    """Convert a [1, H, W, 3] float tensor (0–1) to a PIL Image."""
    t = tensor.numpy()
    if t.ndim == 4:
        t = t[0]
    t = np.clip(t * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(t)

def pil_to_bytes(img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()

# ── Model ──────────────────────────────────────────────────────────────────────
class NSTModel:
    """Singleton wrapper around the EfficientNetB0 feature extractor."""
    _instance: Optional["NSTModel"] = None

    def __init__(self):
        logger.info("Loading EfficientNetB0 backbone …")
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet"
        )
        base.trainable = False
        outputs = (
            [base.get_layer(n).output for n in STYLE_LAYER_NAMES]
            + [base.get_layer(n).output for n in CONTENT_LAYER_NAME]
        )
        self.model = tf.keras.Model(inputs=base.input, outputs=outputs)
        self.model.trainable = False
        logger.info("Model ready.")

    @classmethod
    def get(cls) -> "NSTModel":
        # Use a local variable so Pyre2 can narrow the type correctly.
        # Class variables typed as Optional are not narrowed by Pyre2 after
        # assignment, but local variables are — this is the idiomatic fix.
        instance = cls._instance
        if instance is None:
            instance = cls()
            cls._instance = instance
        return instance  # pyre-ignore[7]

    def __call__(self, img_tensor: tf.Tensor):
        preprocessed = tf.keras.applications.efficientnet.preprocess_input(
            img_tensor * 255.0
        )
        return self.model(preprocessed)

# ── Losses ─────────────────────────────────────────────────────────────────────
def gram_matrix(tensor: tf.Tensor) -> tf.Tensor:
    """Gram matrix for a single-batch feature map."""
    # tensor shape: [1, H, W, C]  →  [H*W, C]
    shape = tf.shape(tensor)
    a = tf.reshape(tensor, [-1, shape[-1]])
    gram = tf.matmul(a, a, transpose_a=True)
    n = tf.cast(shape[1] * shape[2], tf.float32)
    return gram / n

def style_loss(gen_features, target_grams):
    losses = [
        tf.reduce_mean(tf.square(gram_matrix(g) - t))
        for g, t in zip(gen_features, target_grams)
    ]
    return tf.add_n(losses) / NUM_STYLE_LAYERS

def content_loss(gen_feature, target_feature):
    return tf.reduce_mean(tf.square(gen_feature - target_feature))

def total_variation_loss(img: tf.Tensor) -> tf.Tensor:
    """Encourages spatial smoothness in the generated image."""
    return tf.image.total_variation(img)[0]

# ── Training step ──────────────────────────────────────────────────────────────
@tf.function
def train_step(
    image: tf.Variable,
    model: NSTModel,
    target_style_grams,
    target_content_feat: tf.Tensor,
    optimizer: tf.keras.optimizers.Optimizer,
    style_weight: float,
    content_weight: float,
    tv_weight: float,
):
    with tf.GradientTape() as tape:
        features = model(image)
        gen_style_feats = features[:NUM_STYLE_LAYERS]
        gen_content_feat = features[NUM_STYLE_LAYERS]

        s_loss = style_loss(gen_style_feats, target_style_grams)
        c_loss = content_loss(gen_content_feat, target_content_feat)
        tv_loss = total_variation_loss(image)
        loss = style_weight * s_loss + content_weight * c_loss + tv_weight * tv_loss

    grads = tape.gradient(loss, image)
    optimizer.apply_gradients([(grads, image)])
    image.assign(tf.clip_by_value(image, 0.0, 1.0))
    return loss, s_loss, c_loss

# ── Main API ───────────────────────────────────────────────────────────────────
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
    Run Neural Style Transfer.

    Args:
        content_bytes:     Raw bytes of the content image.
        style_bytes:       Raw bytes of the style image.
        style_weight:      Weight for style loss.
        content_weight:    Weight for content loss.
        tv_weight:         Weight for total-variation regularisation.
        num_steps:         Total optimisation steps.
        learning_rate:     Adam learning rate.
        progress_callback: Called every 25 steps with (step, total, loss, jpg_bytes).

    Returns:
        JPEG bytes of the styled image.
    """
    nst_model = NSTModel.get()

    content_tensor = load_and_preprocess(content_bytes)
    style_tensor   = load_and_preprocess(style_bytes)

    # Extract targets
    style_features_target = nst_model(style_tensor)[:NUM_STYLE_LAYERS]
    target_style_grams = [gram_matrix(f) for f in style_features_target]
    target_content_feat = nst_model(content_tensor)[NUM_STYLE_LAYERS]

    # Initialise generated image from content
    image = tf.Variable(content_tensor, trainable=True, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)

    REPORT_EVERY = max(1, num_steps // 20)   # ~20 progress updates
    t0 = time.time()

    for step in range(1, num_steps + 1):
        loss, s_loss, c_loss = train_step(
            image, nst_model, target_style_grams, target_content_feat,
            optimizer, style_weight, content_weight, tv_weight
        )

        if progress_callback is not None and (step % REPORT_EVERY == 0 or step == num_steps):
            pil_img = tensor_to_pil(image)
            img_bytes = pil_to_bytes(pil_img, quality=75)
            elapsed = time.time() - t0
            logger.debug(f"Step {step}/{num_steps} | loss={loss:.2f} | {elapsed:.1f}s")
            progress_callback(step, num_steps, float(loss), img_bytes)

    return pil_to_bytes(tensor_to_pil(image), quality=95)
