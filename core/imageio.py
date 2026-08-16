"""Image loading and saving.

Uses OpenCV for decode/encode (it handles 16-bit and EXIF-free paths
predictably) and normalises everything to RGB float32 in [0, 1], which is
the only image representation the rest of the app deals with.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

SUPPORTED_READ = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
SUPPORTED_WRITE = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")

IMAGE_FILTER = (
    "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;"
    "PNG (*.png);;JPEG (*.jpg *.jpeg);;All files (*)"
)


def load_image(path: str, max_side: int | None = None) -> np.ndarray:
    """Load an image as RGB float32 in [0, 1].

    ``max_side`` optionally downsamples very large photos so the G-buffer
    and its GPU textures stay within a sane memory budget.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    # imdecode via fromfile keeps non-ASCII paths working on Windows, where
    # cv2.imread silently fails on them.
    raw = np.fromfile(path, dtype=np.uint8)
    data = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if data is None:
        raise ValueError(f"Could not decode image: {path}")

    if data.dtype == np.uint16:
        image = data.astype(np.float32) / 65535.0
    elif data.dtype == np.uint8:
        image = data.astype(np.float32) / 255.0
    else:
        image = np.clip(data.astype(np.float32), 0.0, 1.0)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported channel count {image.shape[2]} in {path}")

    if max_side:
        height, width = image.shape[:2]
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / float(longest)
            image = cv2.resize(
                image,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )

    return np.ascontiguousarray(image.astype(np.float32))


def save_image(path: str, image: np.ndarray, quality: int = 95) -> None:
    """Write RGB float32 [0, 1] (or uint8) to disk, inferring format."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_WRITE:
        raise ValueError(f"Unsupported output format: {ext}")

    data = to_uint8(image)
    bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR) if data.ndim == 3 else data

    params: list[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 4]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]

    ok, encoded = cv2.imencode(ext, bgr, params)
    if not ok:
        raise IOError(f"Failed to encode {path}")

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    encoded.tofile(path)


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return (np.clip(image.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def encode_exr_like_png16(path: str, data: np.ndarray) -> None:
    """Write a single-channel float map as 16-bit PNG.

    Depth is the one buffer where 8 bits visibly bands, and 16-bit PNG is
    readable everywhere without an OpenEXR dependency.  The value range is
    normalised, so the companion JSON records the original min/max.
    """
    finite = data[np.isfinite(data)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 1.0
    span = max(hi - lo, 1e-8)
    normalised = np.clip((data - lo) / span, 0.0, 1.0)
    as_u16 = (normalised * 65535.0 + 0.5).astype(np.uint16)
    ok, encoded = cv2.imencode(".png", as_u16)
    if not ok:
        raise IOError(f"Failed to encode {path}")
    encoded.tofile(path)
