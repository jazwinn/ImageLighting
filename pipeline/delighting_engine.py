"""Intrinsic decomposition: strip baked lighting into a flat albedo map.

Relighting an image that already contains its original illumination gives
double shadows and double highlights.  This module solves ``I = A * S`` for
the reflectance ``A``, so the deferred shader can multiply in *only* the
virtual lights.

Three solvers, tried in order:

* **Neural (preferred, optional).**  compphoto/Intrinsic -- see
  :mod:`pipeline.intrinsic_backend`.  Substantially the best of the three,
  and the one to use when it is installed and its academic-only licence
  suits your purpose.

* **ONNX (opt-in).**  Any intrinsic-decomposition network exported to ONNX,
  pointed at by ``IMAGELIGHTING_ALBEDO_ONNX``.

* **Analytic (default).**  An inverse-shading solver in three stages:

  1. **Geometric shading.**  Fit image luminance to a second-order
     spherical-harmonic basis evaluated on the estimated normals.  Nine
     coefficients describe any distant illumination of a Lambertian
     surface, so this recovers the smooth directional shading that
     geometry explains.

  2. **Cast shadows.**  The order-1 SH coefficients *are* the dominant
     light direction.  Knowing it, we can raymarch the depth buffer from
     every pixel toward that light and find which pixels the original
     illumination could not reach -- the same screen-space trace the
     viewport shader runs, applied to the photograph's own light.  The
     resulting visibility mask is appended to the basis as a tenth term
     and re-fitted, so the solve decides for itself how much of the image
     is cast shadow.  This is what lets hard shadows come out, which no
     amount of filtering achieves: a cast shadow and a dark patch of paint
     are identical to a filter, and distinguishable only by geometry.

  3. **Residual falloff.**  Whatever is left after the geometric model --
     vignetting, inverse-square falloff, bounce light -- is captured as a
     genuinely smooth low-pass of the log residual and divided out too.

  A final per-channel pass neutralises the colour cast of the illuminant.

Limitations worth stating plainly: the light-direction recovery assumes one
dominant source, the shadow trace can only see the single depth layer the
camera captured, and a shadow whose occluder is outside the frame is
invisible to it.  Hard shadow edges are attenuated rather than erased.  For
production work, point ``IMAGELIGHTING_ALBEDO_ONNX`` at a learned model.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .base import (
    DEFAULT_MAX_SIDE,
    LOGGER,
    fit_to_max_side,
    linear_to_srgb,
    luminance,
    resize_image,
    srgb_to_linear,
)
from .intrinsic_backend import IntrinsicDecomposer
from .intrinsic_backend import is_available as intrinsic_available

ALBEDO_ONNX_ENV = "IMAGELIGHTING_ALBEDO_ONNX"


@dataclass
class DelightSettings:
    """Tunables exposed to the UI."""

    #: Weight on the geometry-explained shading (SH fit + cast shadows).
    geometry_weight: float = 0.9
    #: Weight on the smooth residual illumination (falloff, vignetting,
    #: bounce).  Raising it flattens large-scale brightness variation, at
    #: the risk of also flattening large-scale *albedo* variation.
    residual_weight: float = 0.5
    #: Low-pass scale for the residual, as a fraction of the longest edge.
    detail_radius: float = 0.07
    #: Overall strength: 0 leaves the image untouched, 1 divides out the
    #: full estimated shading.
    strength: float = 1.0
    #: Neutralises the illuminant's colour so albedo is white-balanced.
    color_cast_removal: float = 0.6
    #: Guards against exploding albedo in near-black pixels.
    shading_floor: float = 0.22
    #: Blends the source back in.  A perfect decomposition is impossible,
    #: and slight residual shading reads better than over-flattening.
    preserve_contrast: float = 0.0
    #: Prefer compphoto/Intrinsic when it is installed.  Turning this off
    #: forces the analytic solver, whose sliders above then apply.
    use_neural: bool = True
    #: Raymarch the depth buffer to find the original light's cast shadows.
    trace_cast_shadows: bool = True
    #: Resolution cap for that raymarch; the mask is smooth, so tracing it
    #: at full resolution costs seconds and buys nothing.
    shadow_trace_max_side: int = 640


class DelightingEngine:
    """Produces the flat albedo map and the shading layer it removed."""

    def __init__(
        self,
        onnx_path: str | None = None,
        max_side: int = DEFAULT_MAX_SIDE,
        settings: DelightSettings | None = None,
    ) -> None:
        self.onnx_path = onnx_path or os.environ.get(ALBEDO_ONNX_ENV) or None
        self.max_side = int(max_side)
        self.settings = settings or DelightSettings()
        self._session = None
        self._backend = "analytic inverse shading (SH + Retinex)"
        self._lock = threading.Lock()
        self._load_failed = False
        # Constructed unconditionally but loads nothing until first use, so
        # an absent optional dependency costs nothing here.
        self._intrinsic = IntrinsicDecomposer()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_deep(self) -> bool:
        return self._session is not None or self._intrinsic.loaded

    @property
    def neural_available(self) -> bool:
        """Whether compphoto/Intrinsic can be used at all on this machine."""
        return intrinsic_available()

    # -- model loading -----------------------------------------------------
    def _ensure_session(self) -> bool:
        if self._session is not None:
            return True
        if self._load_failed or not self.onnx_path:
            return False
        if not os.path.isfile(self.onnx_path):
            LOGGER.warning("Albedo model %s not found; using analytic solver", self.onnx_path)
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
                LOGGER.warning("Failed to load albedo ONNX model: %s", exc)
                self._load_failed = True
                return False

    # -- main entry point --------------------------------------------------
    def decompose(
        self,
        image_srgb: np.ndarray,
        normal: np.ndarray,
        depth: np.ndarray | None = None,
        intrinsics=None,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split an image into ``(albedo_srgb, shading)``.

        ``albedo_srgb`` is RGB float32 in [0, 1]; ``shading`` is a positive
        scalar map normalised so its mean is roughly 1.0.  ``depth`` and
        ``intrinsics`` are optional -- without them the solver skips the
        cast-shadow trace and keeps the rest of the decomposition.
        """
        if self.settings.use_neural and intrinsic_available():
            try:
                albedo, shading = self._intrinsic.decompose(image_srgb, progress=progress)
                self._backend = self._intrinsic.backend
                return albedo, shading
            except Exception as exc:
                # Out of VRAM, a broken weights download, a torch.hub
                # failure, or simply a degenerate frame: none of these
                # should cost the user their import.  The traceback goes to
                # debug because the common case here is the expected
                # non-finite-output fallback, which needs no stack.
                LOGGER.warning("Intrinsic decomposition failed, falling back: %s", exc)
                LOGGER.debug("Intrinsic failure detail", exc_info=True)

        if self._ensure_session():
            if progress:
                progress("De-lighting: ONNX intrinsic model")
            try:
                return self._infer_onnx(image_srgb)
            except Exception as exc:
                LOGGER.warning("Albedo inference failed, falling back: %s", exc)
                self._session = None
                self._load_failed = True

        if progress:
            progress("De-lighting: inverse shading solver")
        return self._solve_inverse_shading(image_srgb, normal, depth, intrinsics, progress)

    def _infer_onnx(self, image_srgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = image_srgb.shape[:2]
        work_w, work_h = fit_to_max_side(width, height, self.max_side)
        work_w = max(32, (work_w // 32) * 32)
        work_h = max(32, (work_h // 32) * 32)
        work = resize_image(image_srgb, work_w, work_h)

        tensor = np.transpose(work, (2, 0, 1))[None].astype(np.float32)
        outputs = self._session.run(None, {self._session.get_inputs()[0].name: tensor})

        albedo = np.squeeze(outputs[0])
        if albedo.shape[0] == 3:
            albedo = np.transpose(albedo, (1, 2, 0))
        albedo = np.clip(resize_image(albedo.astype(np.float32), width, height), 0.0, 1.0)

        # Recover the shading the network implied rather than trusting a
        # second output head that may or may not exist.
        shading = self._shading_from_pair(image_srgb, albedo)
        return albedo, shading

    # -- analytic solver ---------------------------------------------------
    def _solve_inverse_shading(
        self,
        image_srgb: np.ndarray,
        normal: np.ndarray,
        depth: np.ndarray | None = None,
        intrinsics=None,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.settings
        height, width = image_srgb.shape[:2]

        linear = srgb_to_linear(image_srgb)
        lum = np.maximum(luminance(linear), 1e-4)
        log_lum = np.log(lum)

        # (1) Fit the SH basis to recover the geometry-explained shading and,
        # from its order-1 terms, the dominant light direction.
        coefficients = self._fit_sh(normal, lum)
        light_dir = self._light_direction(coefficients)

        # (2) Trace that light's cast shadows and re-fit with the visibility
        # term included, letting least squares weigh it against the rest.
        visibility: np.ndarray | None = None
        can_trace = (
            cfg.trace_cast_shadows and depth is not None and intrinsics is not None
        )
        if can_trace:
            if progress:
                progress("De-lighting: tracing original cast shadows")
            visibility = trace_original_shadows(
                depth, intrinsics, light_dir, max_side=cfg.shadow_trace_max_side
            )
            geometric = self._fit_with_shadow(normal, lum, visibility, light_dir)
            self._backend = "inverse shading (SH + traced cast shadows)"
        else:
            geometric = np.maximum(_sh_basis(normal) @ coefficients, 1e-4)
            self._backend = "inverse shading (SH only)"

        geometric = self._normalise_positive(geometric)
        log_geometric = np.log(geometric)

        # (3) Whatever the geometry could not explain, as a genuinely smooth
        # low-pass.  An edge-preserving filter is wrong here: guided by
        # itself it reproduces the input almost exactly (a -> 1 wherever the
        # local variance exceeds eps), so the "base layer" would come back
        # carrying all the albedo texture and the division would flatten
        # contrast instead of removing illumination.
        sigma = max(4.0, cfg.detail_radius * max(width, height))
        residual = log_lum - log_geometric
        log_residual_base = cv2.GaussianBlur(
            residual, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT
        )

        w_geom = float(np.clip(cfg.geometry_weight, 0.0, 1.0))
        w_resid = float(np.clip(cfg.residual_weight, 0.0, 1.0))
        log_shading = w_geom * log_geometric + w_resid * log_residual_base

        # Normalise to unit mean: the absolute scale of the decomposition is
        # unrecoverable, and leaving it free would change overall exposure.
        log_shading -= float(np.mean(log_shading))
        log_shading *= float(np.clip(cfg.strength, 0.0, 1.0))
        shading = np.exp(log_shading).astype(np.float32)
        shading = np.maximum(shading, float(cfg.shading_floor))

        albedo_linear = linear / shading[..., None]

        if cfg.color_cast_removal > 0.0:
            albedo_linear = self._neutralise_illuminant(
                albedo_linear, sigma, float(cfg.color_cast_removal)
            )

        # Re-anchor exposure to the source so the albedo tab is comparable to
        # the original rather than arbitrarily brighter.
        src_mean = float(np.mean(linear)) + 1e-6
        alb_mean = float(np.mean(albedo_linear)) + 1e-6
        albedo_linear *= src_mean / alb_mean

        if cfg.preserve_contrast > 0.0:
            k = float(np.clip(cfg.preserve_contrast, 0.0, 1.0))
            albedo_linear = albedo_linear * (1.0 - k) + linear * k

        albedo_linear = _soft_clip(albedo_linear)
        albedo = linear_to_srgb(albedo_linear)
        return np.ascontiguousarray(albedo), np.ascontiguousarray(shading)

    @staticmethod
    def _solve(basis: np.ndarray, lum: np.ndarray) -> np.ndarray:
        """Least squares of luminance against a per-pixel basis stack.

        Fits on a subsample -- ten unknowns need nowhere near a megapixel of
        equations -- and drops clipped pixels, since blown highlights and
        crushed blacks are the least trustworthy evidence about lighting.
        """
        terms = basis.shape[-1]
        height, width = lum.shape[:2]
        stride = max(1, int(np.sqrt((height * width) / 40000.0)))

        a = basis[::stride, ::stride].reshape(-1, terms)
        b = lum[::stride, ::stride].reshape(-1)
        valid = (b > 0.01) & (b < 0.98)
        if valid.sum() > 64:
            a = a[valid]
            b = b[valid]

        try:
            coefficients, *_ = np.linalg.lstsq(a, b, rcond=None)
        except np.linalg.LinAlgError:
            coefficients = np.zeros(terms, dtype=np.float32)
            coefficients[0] = float(np.mean(b)) / 0.282095
        return coefficients.astype(np.float32)

    def _fit_sh(self, normal: np.ndarray, lum: np.ndarray) -> np.ndarray:
        """Nine SH coefficients describing the smooth illumination."""
        return self._solve(_sh_basis(normal), lum)

    def _fit_with_shadow(
        self,
        normal: np.ndarray,
        lum: np.ndarray,
        visibility: np.ndarray,
        light_dir: np.ndarray,
    ) -> np.ndarray:
        """Re-fit with a cast-shadow term appended to the SH basis.

        The tenth term is ``V * max(N·L, 0)``: the direct contribution of
        the recovered light where it actually reaches the surface.  Its
        coefficient is free, so a scene with no cast shadows simply gets a
        near-zero weight and the fit degrades gracefully to plain SH.
        """
        basis = _sh_basis(normal)
        n_dot_l = np.clip(np.sum(normal * light_dir[None, None, :], axis=-1), 0.0, 1.0)
        direct = (visibility * n_dot_l)[..., None].astype(np.float32)
        extended = np.concatenate([basis, direct], axis=-1)
        coefficients = self._solve(extended, lum)
        return np.maximum(extended @ coefficients, 1e-4)

    @staticmethod
    def _light_direction(coefficients: np.ndarray) -> np.ndarray:
        """Dominant light direction from the order-1 SH coefficients.

        The linear SH terms are proportional to the illumination's centroid
        direction; entries 1, 2 and 3 of the basis are y, z and x in the
        viewer-facing frame, and the Z flip converts back to camera space.
        Returns a unit vector pointing *toward* the light.
        """
        direction = np.array(
            [coefficients[3], coefficients[1], -coefficients[2]], dtype=np.float32
        )
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            # No directional component: treat it as a light at the camera.
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return direction / norm

    @staticmethod
    def _normalise_positive(shading: np.ndarray) -> np.ndarray:
        """Clamp a fitted shading map positive and rescale to unit mean."""
        shading = np.asarray(shading, dtype=np.float32)
        mean = float(np.mean(shading))
        if not np.isfinite(mean) or mean <= 1e-6:
            return np.ones_like(shading, dtype=np.float32)
        # A least-squares fit is free to go negative where the basis is a
        # poor match; floor it well below the mean rather than at zero, so
        # the subsequent log stays finite without flattening real shadow.
        return np.maximum(shading / mean, 0.05).astype(np.float32)

    @staticmethod
    def _neutralise_illuminant(
        albedo_linear: np.ndarray, sigma: float, strength: float
    ) -> np.ndarray:
        """Divide out the low-frequency colour cast of the original light."""
        gray = np.maximum(luminance(albedo_linear), 1e-4)
        log_gray = np.log(gray)
        out = albedo_linear.copy()
        for c in range(3):
            channel = np.maximum(albedo_linear[..., c], 1e-4)
            ratio = np.log(channel) - log_gray
            cast = cv2.GaussianBlur(
                ratio, (0, 0), sigmaX=float(sigma) * 2.0, borderType=cv2.BORDER_REFLECT
            )
            cast -= float(np.mean(cast))
            out[..., c] = channel * np.exp(-cast * strength)
        return out.astype(np.float32)

    def offload(self) -> None:
        """Hand back VRAM until the next import needs it."""
        self._intrinsic.offload()

    def release(self) -> None:
        """Free the neural weights held by the optional backend."""
        self._intrinsic.release()
        self._session = None

    @staticmethod
    def _shading_from_pair(image_srgb: np.ndarray, albedo_srgb: np.ndarray) -> np.ndarray:
        """Recover ``S = I / A`` in linear light from a predicted albedo."""
        lin_i = np.maximum(luminance(srgb_to_linear(image_srgb)), 1e-4)
        lin_a = np.maximum(luminance(srgb_to_linear(albedo_srgb)), 1e-3)
        shading = lin_i / lin_a
        mean = float(np.mean(shading))
        if mean > 1e-6:
            shading /= mean
        return np.clip(shading, 0.02, 8.0).astype(np.float32)


def trace_original_shadows(
    depth: np.ndarray,
    intrinsics,
    light_dir: np.ndarray,
    *,
    steps: int = 32,
    max_side: int = 640,
    reach: float = 0.5,
    bias_scale: float = 0.006,
    thickness_scale: float = 0.5,
    softness: float = 0.004,
) -> np.ndarray:
    """Screen-space raymarch for the *original* light's cast shadows.

    Same algorithm the viewport shader runs, on the CPU and pointed at the
    illumination already baked into the photograph.  For every pixel, walk
    the 3D ray toward the light, reproject each sample through the pinhole
    intrinsics, and compare against the recorded depth: a sample sitting
    behind a recorded surface means something blocked the light.

    Returns visibility in [0, 1] at the input resolution -- 1 lit, 0 fully
    shadowed.  The trace runs at ``max_side`` and is upsampled, which costs
    nothing in quality because the mask is smooth and saves several seconds
    on a large image.

    ``thickness_scale`` bounds how far behind a surface a sample may sit
    and still count as occluded.  With only one depth layer we cannot tell
    a genuine blocker from a distant background, and an unbounded test
    smears shadows across the whole frame.
    """
    height, width = depth.shape[:2]
    work_w, work_h = fit_to_max_side(width, height, max_side)
    small_depth = resize_image(depth, work_w, work_h)
    small_intrinsics = intrinsics.scaled(work_w, work_h)

    position = small_intrinsics.unproject(small_depth)
    # A robust span, not the full peak-to-peak: with metric depth a single
    # patch of distant sky can stretch the range to 150 m, which would make
    # every ray step metres long and march straight past real occluders.
    span = float(np.percentile(small_depth, 95.0) - np.percentile(small_depth, 5.0))
    if span < 1e-5:
        return np.ones((height, width), dtype=np.float32)

    max_distance = span * reach
    bias = span * bias_scale
    thickness = span * thickness_scale

    light_dir = np.asarray(light_dir, dtype=np.float32).reshape(1, 1, 3)
    occluded = np.zeros((work_h, work_w), dtype=bool)

    for step in range(1, int(steps) + 1):
        sample = position + light_dir * (max_distance * step / steps)
        z = sample[..., 2]
        safe_z = np.maximum(z, 1e-4)
        px = sample[..., 0] * small_intrinsics.fx / safe_z + small_intrinsics.cx
        py = -sample[..., 1] * small_intrinsics.fy / safe_z + small_intrinsics.cy

        inside = (px >= 0.0) & (px < work_w) & (py >= 0.0) & (py < work_h) & (z > 1e-3)
        cols = np.clip(px.astype(np.int32), 0, work_w - 1)
        rows = np.clip(py.astype(np.int32), 0, work_h - 1)
        delta = z - small_depth[rows, cols]
        occluded |= inside & (delta > bias) & (delta < thickness)

    visibility = (~occluded).astype(np.float32)
    if softness > 0.0:
        visibility = cv2.GaussianBlur(
            visibility, (0, 0), sigmaX=max(work_w, work_h) * softness,
            borderType=cv2.BORDER_REFLECT,
        )
    return resize_image(visibility, width, height)


def _sh_basis(normal: np.ndarray) -> np.ndarray:
    """Real second-order spherical harmonics evaluated per pixel.

    Normals arrive in camera space (viewer-facing is -Z); the basis is
    evaluated on the flipped-Z vector so the fitted coefficients describe
    lighting in the conventional viewer-facing frame.
    """
    x = normal[..., 0]
    y = normal[..., 1]
    z = -normal[..., 2]
    ones = np.ones_like(x)
    return np.stack(
        [
            0.282095 * ones,
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3.0 * z * z - 1.0),
            1.092548 * x * z,
            0.546274 * (x * x - y * y),
        ],
        axis=-1,
    ).astype(np.float32)


def _soft_clip(linear: np.ndarray, knee: float = 0.85) -> np.ndarray:
    """Roll values off toward 1.0 instead of hard-clipping to it.

    Division by a small shading value can push albedo well past white; a
    hard clamp would flatten those pixels into featureless patches.
    """
    x = np.maximum(linear.astype(np.float32), 0.0)
    over = x > knee
    if np.any(over):
        t = (x[over] - knee) / max(1e-4, (1.0 - knee))
        x[over] = knee + (1.0 - knee) * (t / (1.0 + t))
    return np.clip(x, 0.0, 1.0)


def colorize_depth(depth: np.ndarray, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    """Turbo/Inferno visualisation of a depth map, near = warm."""
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.float32)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    span = max(hi - lo, 1e-6)
    # Invert so near surfaces get the hot end of the ramp.
    normalised = 1.0 - np.clip((depth - lo) / span, 0.0, 1.0)
    as_u8 = (normalised * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(as_u8, colormap)
    return (cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
