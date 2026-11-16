# arbitrary-stylization-v1-256 (TF.js, uint8)

The browser engine's copy of the exact model the PicturaAI server runs.

- **Origin:** Google Magenta, *Arbitrary Image Stylization v1-256*, TF Hub handle
  `https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2`
  (Ghiasi, Lee, Kudlur, Dumoulin, Shlens, "Exploring the structure of a real-time,
  arbitrary neural artistic stylization network", BMVC 2017).
- **Licence:** Apache License 2.0 (see `LICENSE`). Converted and split by PicturaAI;
  the weights are unchanged apart from uint8 quantization.
- **How it was made:** `docs/revamp/engine/tools/convert_model.py` freezes the
  SavedModel, inlines its functions, and splits it at the style parameters:
  - `transformer/` (6.9 MB): `content_image` [1,H,W,3] in 0..1 + `style_params`
    [1,10438] → `stylized` [1,H,W,3] in 0..1. Downloaded on first use.
  - `style/` (17.5 MB): `style_image` [1,256,256,3] in 0..1 (3x3 average-pooled, like
    the server) → `style_params` [1,10438]. Downloaded only for custom styles.
  - `manifest.json`: sizes and the layout of the 86 gamma/beta vectors inside
    `style_params` (the conditional instance norm parameters of every layer).
- **Preset parameters:** `../style-vectors.json` + `../style-vectors/<key>.bin`
  (float16), computed in float32 from the original SavedModel by
  `scripts/precompute_style_vectors.py`.

Quantization check (docs/revamp/engine/ENGINE.md): against the server's float32
TensorFlow output at 768 px, mean absolute difference 5.9/255 (uint8) vs 2.8/255
(float16); side by side the results are indistinguishable.
