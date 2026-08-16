"""Surface normal estimation.

Two paths, both producing camera-space unit normals:

* **Analytic (default).**  Differentiate the unprojected 3D point cloud and
  take the cross product of the two tangents.  Because it works on metric
  positions rather than raw depth, it stays correct under perspective and
  needs no extra model download.
* **Deep (opt-in).**  A DSINE/Omnidata-style normal network exported to
  ONNX, pointed at by ``IMAGELIGHTING_NORMAL_ONNX``.  These networks recover
  relief that depth-derived normals wash out, at the cost of a separate
  weights file.

Convention: normals live in the same camera space as the position map --
X right, Y up, Z away from the camera -- so a surface facing the viewer has
a negative Z component.
"""

from __future__ import annotations

import os
import threading
from typing import Callable

import cv2
import numpy as np

from .base import (
    DEFAULT_MAX_SIDE,
    LOGGER,
    fit_to_max_side,
    guided_filter,
    resize_image,
)
from .stablenormal_backend import StableNormalBackend
from .stablenormal_backend import is_available as stablenormal_available
from core.gbuffer import CameraIntrinsics

NORMAL_ONNX_ENV = "IMAGELIGHTING_NORMAL_ONNX"

#: Total depth-smoothing scale before differentiation, as a fraction of the
#: longest edge.  Chosen by sweep: it is the point where the noise floor
#: stops improving materially while silhouettes are still intact.
NORMAL_SMOOTH_FRACTION = 0.008

#: How many passes that scale is split across.  Not a performance knob --
#: it controls how wide a halo the filter leaves around every object.  A
#: guided filter fits a local linear model, so a single pass of radius r
#: ramps across a depth discontinuity over about r pixels, and that ramp
#: renders as a band of wrongly-oriented normals hugging every silhouette.
#: Several passes of radius r/n reach the same smoothing on flat surfaces
#: while each one can only ramp over r/n, so the halo shrinks in proportion.
#: Measured on a 1504 px interior: one pass at r=12 left a halo score of
#: 1.13, three passes at r=4 cut it to 0.66 for 140 ms more, and a bilateral
#: good enough to beat that cost 4.5x as much.
NORMAL_SMOOTH_PASSES = 3

#: Median passes run before the guided filter.  A median cannot ramp across
#: a step -- it returns one side or the other -- so iterating it drives the
#: depth toward a piecewise-constant signal and sharpens exactly the
#: occlusion boundaries the guided filter would otherwise soften.  OpenCV
#: caps float32 medians at a 5x5 kernel, hence repetition rather than a
#: wider window.  Three passes take the halo from 0.664 to 0.643 for 30 ms;
#: past six the returns stop being worth the milliseconds.
NORMAL_MEDIAN_PASSES = 3

#: Pixel baseline of the one-sided derivatives used to build the tangents.
#: Two was the best of 1/2/4 at both test resolutions, so it is left fixed
#: rather than scaled with the image; the resolution-dependent part of the
#: noise is already handled by the smoothing above.
DERIVATIVE_BASELINE = 2

#: Guided-filter epsilon as a fraction of the robust depth span, squared.
#: Sets what counts as "noise to average" versus "edge to keep": comfortably
#: above the wobble on a flat wall, far below a real occlusion step.
NORMAL_EPS_FRACTION = 0.012


class NormalEngine:
    """Estimates surface normals, preferring a deep model when configured."""

    def __init__(
        self,
        onnx_path: str | None = None,
        max_side: int = DEFAULT_MAX_SIDE,
        smoothing: float = 1.0,
        use_neural: bool = True,
    ) -> None:
        self.onnx_path = onnx_path or os.environ.get(NORMAL_ONNX_ENV) or None
        self.max_side = int(max_side)
        self.smoothing = float(smoothing)
        self.use_neural = bool(use_neural)
        self._session = None
        self._backend = "analytic point-cloud gradients"
        self._lock = threading.Lock()
        self._load_failed = False
        # Constructed unconditionally; loads nothing until first use, so an
        # absent optional dependency costs nothing here.
        self._stable = StableNormalBackend()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_deep(self) -> bool:
        return self._session is not None or self._stable.loaded

    @property
    def neural_available(self) -> bool:
        return stablenormal_available()

    def offload(self) -> None:
        """Hand back VRAM until the next import needs it."""
        self._stable.offload()

    def release(self) -> None:
        self._stable.release()
        self._session = None

    # -- model loading -----------------------------------------------------
    def _ensure_session(self) -> bool:
        if self._session is not None:
            return True
        if self._load_failed or not self.onnx_path:
            return False
        if not os.path.isfile(self.onnx_path):
            LOGGER.warning("Normal model %s not found; using analytic normals", self.onnx_path)
            self._load_failed = True
            return False

        with self._lock:
            if self._session is not None:
                return True
            try:
                import onnxruntime as ort

                providers = [
                    p
                    for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                    if p in ort.get_available_providers()
                ]
                self._session = ort.InferenceSession(self.onnx_path, providers=providers)
                self._backend = f"ONNX {os.path.basename(self.onnx_path)} ({providers[0]})"
                return True
            except Exception as exc:
                LOGGER.warning("Failed to load normal ONNX model: %s", exc)
                self._load_failed = True
                return False

    # -- inference ---------------------------------------------------------
    def estimate(
        self,
        image_srgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        progress: Callable[[str], None] | None = None,
    ) -> np.ndarray:
        """Return an ``H x W x 3`` float32 map of unit camera-space normals."""
        if self.use_neural and stablenormal_available():
            try:
                normal = self._stable.estimate(image_srgb, progress=progress)
                self._backend = self._stable.backend
                return normal
            except Exception as exc:
                # A failed download, a VRAM exhaustion, a diffusers change:
                # none of these should cost the user their import.
                LOGGER.warning("StableNormal failed, falling back: %s", exc)
                LOGGER.debug("StableNormal failure detail", exc_info=True)

        if self._ensure_session():
            if progress:
                progress("Normals: ONNX model")
            try:
                return self._infer_onnx(image_srgb)
            except Exception as exc:
                LOGGER.warning("Normal inference failed, falling back to analytic: %s", exc)
                self._session = None
                self._load_failed = True

        if progress:
            progress("Normals: point-cloud gradients")
        self._backend = "analytic point-cloud gradients"
        return self.from_point_cloud(depth, intrinsics, self.smoothing)

    def _infer_onnx(self, image_srgb: np.ndarray) -> np.ndarray:
        height, width = image_srgb.shape[:2]
        work_w, work_h = fit_to_max_side(width, height, self.max_side)
        # Most exported normal networks want a size divisible by 32.
        work_w = max(32, (work_w // 32) * 32)
        work_h = max(32, (work_h // 32) * 32)
        work = resize_image(image_srgb, work_w, work_h)

        tensor = np.transpose(work, (2, 0, 1))[None].astype(np.float32)
        inputs = {self._session.get_inputs()[0].name: tensor}
        output = self._session.run(None, inputs)[0]

        normal = np.squeeze(output)
        if normal.shape[0] == 3:
            normal = np.transpose(normal, (1, 2, 0))
        # Networks export either [-1, 1] directly or a [0, 1] encoded map.
        if float(normal.min()) >= -0.01:
            normal = normal * 2.0 - 1.0
        normal = resize_image(normal.astype(np.float32), width, height)
        # Exported maps use +Z toward the viewer; our camera space uses -Z.
        normal[..., 2] *= -1.0
        return normalize_map(normal)

    # -- analytic path -----------------------------------------------------
    @staticmethod
    def from_point_cloud(
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        smoothing: float = 1.0,
        edge_threshold: float = 0.06,
    ) -> np.ndarray:
        """Cross product of the two surface tangents of the 3D point cloud.

        ``edge_threshold`` is the relative depth gradient above which a
        pixel is treated as a silhouette rather than a surface.  At an
        occlusion boundary the two tangents straddle a depth cliff, so
        their cross product points sideways and the shader renders a dark
        rim around every object.  Those pixels get bent back toward the
        viewer instead -- not because that is their true orientation, but
        because a single depth map does not record the true orientation of
        a surface it never saw, and facing the camera is the least wrong
        answer available.
        """
        depth = depth.astype(np.float32)
        if smoothing > 0.0:
            height, width = depth.shape[:2]
            longest = max(width, height)
            span = float(np.percentile(depth, 95.0) - np.percentile(depth, 5.0))

            # Medians first, aimed squarely at the plateau-and-jump pattern
            # left by upsampling a depth map from the network's internal
            # resolution, and at keeping occlusion boundaries as steps. A
            # median collapses those where a linear filter would only ramp
            # between them, which is what keeps object silhouettes from
            # rendering with crenellated rims or a halo of false geometry.
            for _ in range(NORMAL_MEDIAN_PASSES):
                depth = cv2.medianBlur(depth, 5)

            # Then an edge-preserving smooth, whose support scales with the
            # image. This is the fix for the noise that dominates large
            # photographs, and the reason it has to be proportional is worth
            # spelling out.
            #
            # Turning depth into a normal divides the depth gradient by the
            # pixel footprint, so the sensitivity to depth error scales with
            # fx/Z -- and fx grows with image width. On a 1504 px interior
            # shot fx is ~1383, and a depth wobble of one millimetre per
            # pixel already tilts the normal by ten degrees. On a surface
            # facing the camera the true gradient is nearly zero, so on
            # walls and ceilings that wobble is *all* you see: the shading
            # breaks into the streaks and crosshatch this filter removes.
            # A support fixed in pixels under-smooths precisely as
            # resolution, and therefore fx, goes up.
            #
            # Self-guided rather than image-guided, for the same reason as
            # in smooth_depth_edges: guiding by the photograph would print
            # its albedo texture into the geometry.
            #
            # Applied as several narrow passes rather than one wide one --
            # see NORMAL_SMOOTH_PASSES. One wide pass reaches the same noise
            # floor but leaves a visible halo of mis-oriented normals around
            # every object, because the guided filter's linear model ramps
            # across a depth cliff over roughly its own radius.
            radius = max(
                2,
                int(round(NORMAL_SMOOTH_FRACTION * smoothing * longest
                          / NORMAL_SMOOTH_PASSES)),
            )
            eps = float(max(NORMAL_EPS_FRACTION * span, 1e-4) ** 2)
            for _ in range(NORMAL_SMOOTH_PASSES):
                depth = guided_filter(depth, depth, radius=radius, eps=eps)

        position = intrinsics.unproject(depth)

        # One-sided differences chosen per pixel, rather than a centred
        # Scharr: a centred stencil spans occlusion boundaries and invents
        # the connecting surface. See directed_tangents.
        d_du, d_dv = directed_tangents(position, DERIVATIVE_BASELINE)

        normal = np.cross(d_du, d_dv)
        normal = normalize_map(normal)

        # Orient every normal toward the camera.  The test has to be against
        # the actual view direction, not the sign of Nz: a floor or a wall
        # seen edge-on has Nz near zero, so an Nz-based test would flip
        # neighbouring pixels at random and speckle the whole surface.
        view = -normalize_map(position)
        backfacing = np.sum(normal * view, axis=-1) < 0.0
        normal[backfacing] *= -1.0

        if edge_threshold > 0.0:
            silhouette = _depth_edge_mask(depth, edge_threshold)[..., None]
            normal = normalize_map(normal * (1.0 - silhouette) + view * silhouette)

        return normal


def directed_tangents(position: np.ndarray, baseline: int) -> tuple[np.ndarray, np.ndarray]:
    """Surface tangents from one-sided differences, per axis.

    A centred stencil straddles an occlusion boundary: the pixel to one
    side is on the object and the pixel to the other is on the background,
    so the difference describes a surface that connects them and does not
    exist.  Taking both one-sided differences and keeping whichever has the
    smaller change in depth keeps the stencil on a single surface layer,
    because the one that would cross the boundary is exactly the one with
    the large jump.

    Worth knowing what this does and does not buy: it only helps where the
    boundary really is a step.  This depth comes from a network that
    predicts at a lower internal resolution and upsamples, so boundaries
    arrive already spread over several pixels -- and inside a ramp both
    one-sided differences are moderate, leaving nothing to reject.  Applied
    to raw depth it is markedly worse than a centred stencil (halo 1.42
    against 0.64); applied after the smoothing above it is modestly better
    (0.59) and about twice as fast.  It is the smoothing that keeps the
    boundary sharp enough for this to have something to bite on.
    """
    k = max(1, int(baseline))
    height, width = position.shape[:2]

    # One replicated border, then four views of it.  Padding per-neighbour
    # instead would copy this array four times, and at a few megapixels
    # that dominates the cost of the whole estimator.
    padded = cv2.copyMakeBorder(position, k, k, k, k, cv2.BORDER_REPLICATE)
    center = padded[k:k + height, k:k + width]

    left = center - padded[k:k + height, 0:width]
    right = padded[k:k + height, 2 * k:2 * k + width] - center
    up = center - padded[0:height, k:k + width]
    down = padded[2 * k:2 * k + height, k:k + width] - center

    use_left = (np.abs(left[..., 2]) < np.abs(right[..., 2]))[..., None]
    use_up = (np.abs(up[..., 2]) < np.abs(down[..., 2]))[..., None]
    return np.where(use_left, left, right), np.where(use_up, up, down)


def _depth_edge_mask(depth: np.ndarray, threshold: float) -> np.ndarray:
    """Soft mask of occlusion boundaries, from the relative depth gradient.

    The gradient is divided by depth so the threshold means the same thing
    near and far: a distant object's silhouette spans fewer depth units per
    pixel than a close one's, but both are equally a cliff.
    """
    depth = depth.astype(np.float32)
    # Scharr's 3x3 kernel has a gain of 32; dividing it out makes the
    # threshold a plain per-pixel fraction of depth.
    gu = cv2.Scharr(depth, cv2.CV_32F, 1, 0, borderType=cv2.BORDER_REFLECT) / 32.0
    gv = cv2.Scharr(depth, cv2.CV_32F, 0, 1, borderType=cv2.BORDER_REFLECT) / 32.0
    relative = np.sqrt(gu * gu + gv * gv) / np.maximum(depth, 1e-4)

    low = float(threshold)
    high = float(threshold) * 3.0
    mask = np.clip((relative - low) / max(high - low, 1e-6), 0.0, 1.0)
    # Widen slightly: the cross product is corrupted a pixel or two either
    # side of the cliff, not only exactly on it.
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=1.2, borderType=cv2.BORDER_REFLECT)


def normalize_map(vectors: np.ndarray) -> np.ndarray:
    """Normalise an ``H x W x 3`` field, substituting a viewer-facing normal."""
    vectors = vectors.astype(np.float32)
    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe = length > 1e-6
    out = np.divide(vectors, np.maximum(length, 1e-6), dtype=np.float32)
    out = np.where(safe, out, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    return np.ascontiguousarray(out.astype(np.float32))


def encode_normal_for_display(normal: np.ndarray) -> np.ndarray:
    """Pack camera-space normals into the conventional RGB normal-map look.

    Flips Z so that surfaces facing the viewer read as the familiar
    lavender-blue, matching how every other tool displays a normal map.
    """
    display = normal.copy()
    display[..., 2] *= -1.0
    return np.clip(display * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
