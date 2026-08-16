"""Shared plumbing for the inference engines.

Holds device selection, colour-space conversion, and the image filters the
fallback estimators are built from.  Nothing here imports Qt, so the whole
pipeline can be driven from a script or a test.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

LOGGER = logging.getLogger("imagelighting.pipeline")

#: Longest edge the estimators run at.  Inference cost scales with area and
#: the G-buffer is upsampled back to full resolution afterwards, so this is
#: the main quality/latency dial.
DEFAULT_MAX_SIDE = 1024


@dataclass(frozen=True)
class DeviceInfo:
    torch_device: str
    name: str
    fp16: bool

    def __str__(self) -> str:
        return f"{self.name} ({self.torch_device}{', fp16' if self.fp16 else ''})"


@lru_cache(maxsize=1)
def select_device() -> DeviceInfo:
    """Pick the best available torch device, degrading to CPU in silence."""
    if os.environ.get("IMAGELIGHTING_FORCE_CPU"):
        return DeviceInfo("cpu", "CPU (forced)", False)
    try:
        import torch
    except ImportError:
        return DeviceInfo("cpu", "CPU (torch unavailable)", False)

    if torch.cuda.is_available():
        return DeviceInfo("cuda", torch.cuda.get_device_name(0), True)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return DeviceInfo("mps", "Apple MPS", False)
    return DeviceInfo("cpu", "CPU", False)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


_warmed_up = False


def warm_up_torch() -> None:
    """Initialise torch's intra-op thread pool on the calling thread.

    Must be called from the main thread before any worker thread runs
    inference.  On Windows, letting torch lazily create its OpenMP pool
    from a secondary thread corrupts the process heap on the *second*
    inference -- the first succeeds, and the crash lands later in whatever
    unrelated allocation happens to touch the damaged arena, typically
    deep inside numpy or the transformers image processor.  The failure is
    a bare ``0xC0000374`` with no Python exception, so it is worth
    preventing rather than debugging twice.

    A trivial matmul is enough to force the pool into existence, and
    unlike ``set_num_threads(1)`` it costs no CPU inference throughput.
    """
    global _warmed_up
    if _warmed_up or not torch_available():
        return
    import torch

    torch.zeros(64, 64).matmul(torch.zeros(64, 64))
    _warmed_up = True
    LOGGER.debug("torch thread pool warmed up on the main thread")


# --------------------------------------------------------------------------
# Colour space
# --------------------------------------------------------------------------

def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """Decode sRGB in [0, 1] to linear light."""
    x = np.clip(image.astype(np.float32), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Encode linear light back to sRGB in [0, 1]."""
    x = np.clip(image.astype(np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055).astype(np.float32)


def luminance(linear_rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 relative luminance of a linear-light RGB image."""
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return (linear_rgb.astype(np.float32) @ weights).astype(np.float32)


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------

def fit_to_max_side(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Scale a size down so its longest edge is ``max_side``, keeping aspect."""
    longest = max(width, height)
    if longest <= max_side:
        return int(width), int(height)
    scale = max_side / float(longest)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def resize_image(image: np.ndarray, width: int, height: int, *, smooth: bool = True) -> np.ndarray:
    if image.shape[1] == width and image.shape[0] == height:
        return image
    shrinking = width * height < image.shape[0] * image.shape[1]
    if not smooth:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    out = cv2.resize(image, (int(width), int(height)), interpolation=interp)
    return np.ascontiguousarray(out.astype(np.float32))


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Edge-aware smoothing of ``src`` under the structure of ``guide``.

    Implements He et al.'s grayscale guided filter with box means.  Used by
    the de-lighter to separate low-frequency shading from albedo detail
    without bleeding across object boundaries the way a Gaussian would.
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    ksize = (int(radius) * 2 + 1, int(radius) * 2 + 1)

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, -1, ksize, borderType=cv2.BORDER_REFLECT)

    mean_g = box(guide)
    mean_s = box(src)
    corr_gg = box(guide * guide)
    corr_gs = box(guide * src)
    var_g = corr_gg - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_s
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return (box(a) * guide + box(b)).astype(np.float32)


def percentile_normalize(
    data: np.ndarray, low: float = 1.0, high: float = 99.0
) -> np.ndarray:
    """Normalise to [0, 1] using percentiles so outliers do not flatten it."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo = float(np.percentile(finite, low))
    hi = float(np.percentile(finite, high))
    if hi - lo < 1e-8:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def disparity_to_depth(
    disparity: np.ndarray, near: float = 0.6, far: float = 6.0
) -> np.ndarray:
    """Convert a relative inverse-depth (disparity) map to a depth map.

    Monocular models such as Depth Anything predict disparity up to an
    unknown affine transform, so we normalise it to [0, 1] and map it onto a
    plausible ``[near, far]`` metric window.  The mapping is done in
    disparity space (``1/z`` linear in the prediction), which preserves the
    perspective relationships the shader relies on far better than lerping
    depth directly.
    """
    d = percentile_normalize(disparity, 1.0, 99.0)
    inv_near = 1.0 / max(float(near), 1e-4)
    inv_far = 1.0 / max(float(far), 1e-4)
    inv_z = inv_far + d * (inv_near - inv_far)
    depth = 1.0 / np.maximum(inv_z, 1e-6)
    return depth.astype(np.float32)


def smooth_depth_edges(depth: np.ndarray) -> np.ndarray:
    """Remove model noise from depth while keeping occlusion edges sharp.

    Deliberately *self*-guided rather than guided by the image.  Using the
    photograph as the guide sharpens silhouettes but also copies albedo
    texture straight into the geometry -- a checkerboard floor comes back as
    a corrugated one, and the normal estimator, which differentiates this
    map, turns that corrugation into speckle.

    The result is clamped to the range the model actually predicted: the
    filter's edge response overshoots at strong discontinuities, and a
    negative or near-zero depth would blow up the unprojection.  Bounding by
    the input rather than by fixed limits keeps this correct for metric
    depth in metres and for a remapped relative scale alike.
    """
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.ones_like(depth, dtype=np.float32)
    lo = max(float(finite.min()), 1e-4)
    hi = max(float(finite.max()), lo + 1e-4)

    normalised = percentile_normalize(depth, 0.0, 100.0)
    filtered = guided_filter(normalised, depth, radius=3, eps=2e-4)
    return np.clip(filtered, lo, hi).astype(np.float32)
