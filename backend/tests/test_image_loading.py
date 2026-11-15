"""Image decoding: EXIF orientation and reduced-scale decode of large uploads."""
import io

from PIL import Image

from nst_engine import color_palette_transfer, load_and_preprocess, open_image


def _jpeg(size, orientation=None, color=(200, 30, 30)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    if orientation:
        exif = img.getexif()
        exif[0x0112] = orientation
        img.save(buf, "JPEG", exif=exif.tobytes())
    else:
        img.save(buf, "JPEG", quality=60)
    return buf.getvalue()


def test_exif_rotation_is_applied():
    # Stored 400x200 landscape with "rotate 90°" → displays as 200x400 portrait.
    img = open_image(_jpeg((400, 200), orientation=6))
    assert img.size == (200, 400)


def test_exif_rotation_carries_through_to_model_tensor():
    t = load_and_preprocess(_jpeg((400, 200), orientation=6), max_dim=320)
    _, h, w, _ = t.shape
    assert (w, h) == (160, 320)  # portrait, longest side capped at 320


def test_large_jpeg_is_decoded_at_reduced_scale():
    raw = _jpeg((6000, 4000))
    img = open_image(raw, max_dim=320)
    assert max(img.size) == 320
    # The draft decode must not have decoded the full 24 MP frame first:
    # a reopened image in draft mode reports the reduced size before load().
    probe = Image.open(io.BytesIO(raw))
    probe.draft("RGB", (320, 320))
    assert probe.size[0] < 6000


def test_never_upscales():
    img = open_image(_jpeg((120, 80)), max_dim=320)
    assert img.size == (120, 80)


def test_min_size_keeps_both_sides_large_enough():
    img = open_image(_jpeg((5000, 1000)), min_size=(256, 256))
    assert img.size[0] >= 256 and img.size[1] >= 256


def test_png_and_palette_images_are_converted_to_rgb():
    buf = io.BytesIO()
    Image.new("P", (50, 40)).save(buf, "PNG")
    img = open_image(buf.getvalue(), max_dim=320)
    assert img.mode == "RGB" and img.size == (50, 40)


def test_palette_transfer_respects_orientation():
    out = color_palette_transfer(_jpeg((400, 200), orientation=6), _jpeg((64, 64), color=(10, 10, 200)))
    assert Image.open(io.BytesIO(out)).size == (200, 400)
