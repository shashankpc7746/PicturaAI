# DeepLab-v3 (MediaPipe Image Segmenter)

Used by `frontend/assets/js/engine/segment.js` for one-tap "Style the subject" /
"Style the background".

- **File:** `deeplab_v3.tflite` (2,780,176 bytes, SHA-256
  `ff36e24d40547fe9e645e2f4e8745d1876d6e38b332d39a82f0bf0f5d1d561b3`), unchanged.
- **Downloaded from:**
  `https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite`
  (the model linked from Google's MediaPipe Image Segmenter documentation).
- **What it does:** 257x257 semantic segmentation into the 21 PASCAL VOC classes
  (background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
  dining table, dog, horse, motorbike, person, potted plant, sheep, sofa, train, tv).
  Labels are embedded in the file's metadata.
- **Licence:** Apache License 2.0 (`LICENSE`), the licence of the DeepLab
  implementation in TensorFlow Models that MediaPipe packages. The MediaPipe page does
  not print a separate licence for the .tflite file; check its model card before any
  commercial launch.
- **Runtime:** `@mediapipe/tasks-vision@0.10.35` (Apache-2.0) from cdn.jsdelivr.net.
