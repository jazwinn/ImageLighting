"""Neural intrinsic decomposition via compphoto/Intrinsic.

Wraps the SFU Computational Photography Lab's ordinal-shading pipeline
("Colorful Diffuse Intrinsic Image Decomposition in the Wild", v2 weights).
It is a far stronger de-lighter than the analytic solver in
:mod:`pipeline.delighting_engine`: it flattens shading gradients almost
completely and removes most of a hard cast shadow, which no filter-based
method achieves.

    pip install "intrinsic @ git+https://github.com/compphoto/Intrinsic@main"

LICENCE, read before shipping anything: compphoto/Intrinsic is released for
**academic use only** and the method is patent-protected, with commercial
licensing handled by the SFU Technology Licensing Office.  This is a
stricter constraint than an open-source licence -- there is no compliance
path for commercial use short of negotiating one.  The dependency is
therefore entirely optional: when it is absent the application falls back
to its own analytic solver and works unchanged.

Two implementation details are load-bearing, both established by
measurement rather than from the documentation:

*Working resolution.*  Runtime is flat and fast up to about 1024 px on the
longest edge and then falls off a cliff -- on an 8 GB card the same image
takes 1.4 s at 1024 px and 81 s at 1600 px, as the allocator starts
spilling to host memory.  The pipeline therefore runs at
:data:`WORKING_MAX_SIDE` and the result is lifted back to full resolution.

*Lifting back.*  Rather than upsampling the albedo, which would soften all
its texture, we upsample the *shading* and re-derive albedo at full
resolution from the identity the pipeline itself guarantees,
``linear_image = albedo * diffuse_shading + residual``.  Shading is smooth
by nature so it survives resampling, and every bit of the original image's
detail is retained.  Verified: the identity holds to 0.000000 mean absolute
error, and re-deriving albedo this way matches the network's own
``hr_alb`` to 0.008 mean absolute error.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import threading
from typing import Callable

import numpy as np

from .base import (
    LOGGER,
    fit_to_max_side,
    linear_to_srgb,
    luminance,
    resize_image,
    select_device,
    srgb_to_linear,
)

#: Longest edge the pipeline runs at.  See the module docstring: above this
#: the cost is pathological, not merely higher.
WORKING_MAX_SIDE = 1024

#: Longest edge the pipeline runs at, minimum.  Below roughly 256 px
#: ``chrislib.resolution_util.calculateprocessingres`` raises
#: ``UnboundLocalError`` -- its gradient-threshold loop never executes, so a
#: local it returns is never bound.  Small inputs are therefore upscaled to
#: this before the call and the outputs resampled back down; the pipeline
#: internally rounds up to its 384 px base anyway, so nothing is lost.
WORKING_MIN_SIDE = 384

#: Released weight set. 'v2' is the colourful-diffuse model.
DEFAULT_WEIGHTS = "v2"

#: torch.hub repo the MiDaS backbone pulls in for efficientnet-lite3.
_BACKBONE_REPO = "rwightman/gen-efficientnet-pytorch"


@functools.lru_cache(maxsize=1)
def is_available() -> bool:
    """True when the optional ``intrinsic`` package is installed.

    Deliberately checks the module spec rather than importing: the UI asks
    this question on the main thread while building the inspector, and
    actually importing ``intrinsic`` drags in timm and the MiDaS backbones,
    which would add seconds to application startup for a yes/no answer.
    """
    try:
        return importlib.util.find_spec("intrinsic") is not None
    except (ImportError, ValueError):
        return False


@contextlib.contextmanager
def _trusted_hub_backbone():
    """Allow the one torch.hub repo the backbone needs, and only that one.

    ``torch.hub.load`` asks for confirmation on stdin the first time it
    fetches an untrusted repo.  On the pipeline worker thread there is no
    stdin, so the prompt raises ``EOFError`` and model loading dies.  This
    injects ``trust_repo=True`` for the single repository the MiDaS
    encoder requires, and restores the original function afterwards, so no
    blanket trust is granted to anything else.
    """
    import torch

    original = torch.hub.load

    def patched(repo_or_dir, *args, **kwargs):
        if isinstance(repo_or_dir, str) and repo_or_dir.startswith(_BACKBONE_REPO):
            kwargs.setdefault("trust_repo", True)
        return original(repo_or_dir, *args, **kwargs)

    torch.hub.load = patched
    try:
        yield
    finally:
        torch.hub.load = original


#: Loaded weight sets, shared across every decomposer in the process and
#: keyed by (weights, device).  The five stage networks plus their
#: backbones occupy roughly a gigabyte of VRAM, so a second instance
#: quietly loading its own copy is not an option on an 8 GB card.
_MODEL_CACHE: dict[tuple[str, str], object] = {}
_CACHE_LOCK = threading.Lock()


class IntrinsicDecomposer:
    """Lazy-loading wrapper around the compphoto/Intrinsic pipeline."""

    def __init__(
        self, weights: str = DEFAULT_WEIGHTS, max_side: int = WORKING_MAX_SIDE
    ) -> None:
        self.weights = weights
        self.max_side = int(max_side)
        self._models = None
        self._lock = threading.Lock()
        self._load_failed = False
        self._backend = f"compphoto/Intrinsic {weights}"

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def loaded(self) -> bool:
        return self._models is not None

    # -- loading -----------------------------------------------------------
    def ensure_loaded(self, progress: Callable[[str], None] | None = None) -> bool:
        if self._models is not None:
            return True
        if self._load_failed:
            return False

        with self._lock:
            if self._models is not None:
                return True
            if self._load_failed:
                return False
            try:
                from intrinsic.pipeline import load_models
            except Exception as exc:
                LOGGER.info("compphoto/Intrinsic not installed (%s)", exc)
                self._load_failed = True
                return False

            device = select_device()
            cache_key = (self.weights, device.torch_device)
            with _CACHE_LOCK:
                cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                self._models = cached
                self._backend = (
                    f"compphoto/Intrinsic {self.weights} @ {device.name} "
                    f"({device.torch_device})"
                )
                return True

            if progress:
                progress("De-lighting: loading intrinsic model (first run downloads ~1 GB)")
            try:
                with _trusted_hub_backbone():
                    self._models = load_models(
                        self.weights, device=device.torch_device
                    )
                with _CACHE_LOCK:
                    _MODEL_CACHE[cache_key] = self._models
                self._backend = (
                    f"compphoto/Intrinsic {self.weights} @ {device.name} "
                    f"({device.torch_device})"
                )
                LOGGER.info("Loaded %s", self._backend)
                return True
            except Exception as exc:
                LOGGER.warning("Could not load compphoto/Intrinsic: %s", exc)
                self._load_failed = True
                return False

    # -- inference ---------------------------------------------------------
    def decompose(
        self,
        image_srgb: np.ndarray,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(albedo_srgb, shading)`` at the input resolution.

        ``albedo_srgb`` is RGB float32 in [0, 1]; ``shading`` is a positive
        scalar map normalised to unit mean, matching the contract the rest
        of the pipeline expects.
        """
        if not self.ensure_loaded(progress):
            raise RuntimeError("compphoto/Intrinsic is not available")

        from intrinsic.pipeline import run_pipeline

        height, width = image_srgb.shape[:2]
        work_w, work_h = fit_to_max_side(width, height, self.max_side)
        longest = max(work_w, work_h)
        if longest < WORKING_MIN_SIDE:
            lift = WORKING_MIN_SIDE / float(max(longest, 1))
            work_w = max(1, int(round(work_w * lift)))
            work_h = max(1, int(round(work_h * lift)))
        work = np.ascontiguousarray(
            np.clip(resize_image(image_srgb, work_w, work_h), 0.0, 1.0).astype(np.float32)
        )

        if progress:
            progress(f"De-lighting: intrinsic decomposition at {work_w}x{work_h}")

        device = select_device()
        self.to(device.torch_device)
        # The pipeline expects RGB float in [0, 1], sRGB-encoded (its own
        # `linear=False` default), which is exactly our image convention.
        results = run_pipeline(self._models, work, device=device.torch_device)

        shading = self._as_float(results["dif_shd"])
        residual = self._as_float(results.get("residual"))
        albedo_lr = self._as_float(results.get("hr_alb"))

        # The pipeline rounds its working size up to a multiple of 32, so
        # its outputs rarely match the frame exactly even without a
        # downscale; everything gets resampled back explicitly.
        shading = self._to_size(shading, width, height)

        if residual is not None and shading is not None:
            residual = self._to_size(residual, width, height)
            linear = srgb_to_linear(image_srgb)
            albedo_linear = (linear - residual) / np.maximum(shading, 1e-3)
        elif albedo_lr is not None:
            # No residual head: fall back to resampling the albedo directly.
            albedo_linear = self._to_size(albedo_lr, width, height)
        else:
            raise RuntimeError("Intrinsic pipeline returned neither albedo nor residual")

        # The network emits all-NaN on degenerate input -- a fully black
        # frame is enough to do it.  Raising here rather than returning the
        # NaNs lets the caller fall back to the analytic solver, instead of
        # uploading them as textures and rendering a black hole.
        if not np.all(np.isfinite(albedo_linear)):
            raise ValueError(
                "Intrinsic pipeline returned non-finite albedo "
                "(degenerate input, e.g. a uniformly black frame)"
            )

        albedo = linear_to_srgb(np.clip(albedo_linear, 0.0, 1.0))

        scalar = np.maximum(luminance(shading), 1e-4)
        mean = float(np.mean(scalar))
        if np.isfinite(mean) and mean > 1e-6:
            scalar = scalar / mean

        return (
            np.ascontiguousarray(albedo.astype(np.float32)),
            np.ascontiguousarray(np.clip(scalar, 0.02, 64.0).astype(np.float32)),
        )

    @staticmethod
    def _as_float(value) -> np.ndarray | None:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _to_size(array: np.ndarray, width: int, height: int) -> np.ndarray:
        if array.ndim == 2:
            array = array[..., None]
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        if array.shape[0] == height and array.shape[1] == width:
            return np.ascontiguousarray(array.astype(np.float32))
        return resize_image(array.astype(np.float32), width, height)

    def to(self, device: str) -> None:
        """Move every stage network between host and device memory."""
        if not self._models:
            return
        for model in self._models.values():
            if hasattr(model, "to"):
                try:
                    model.to(device)
                except Exception as exc:  # pragma: no cover
                    LOGGER.debug("Could not move intrinsic model to %s: %s", device, exc)

    def offload(self) -> None:
        """Park the weights in host memory and hand the VRAM back.

        See StableNormalBackend.offload: the three neural stages run in
        sequence, and on an 8 GB card they do not all fit at once.
        """
        self.to("cpu")
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def release(self) -> None:
        """Drop this decomposer's handle and the shared weights with it.

        Called on shutdown, where nothing else is going to want them; the
        cache is cleared too so the VRAM is actually returned rather than
        pinned by a dictionary nobody reads again.
        """
        self._models = None
        with _CACHE_LOCK:
            _MODEL_CACHE.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
