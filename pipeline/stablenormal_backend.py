"""Surface normals straight from RGB, via StableNormal.

Predicting normals from the image sidesteps the ceiling on everything the
analytic estimator can do, because that estimator differentiates depth and
therefore inherits every one of depth's limitations:

* no differentiation, so no ``fx/Z`` noise amplification and no need to
  smooth flat surfaces at all;
* boundaries come from appearance, so an object's silhouette is a step
  rather than a ramp with invented connecting geometry;
* **thin structures survive**.  Chair legs, cable runs and window mullions
  are finer than a monocular depth network can resolve, so no depth-derived
  method can recover them -- but they are plainly visible in the photograph.

Measured on the interior test image against the tuned analytic path:
halo 0.600 -> 0.250 (-58%), noise 0.00718 -> 0.00585 (-19%), at a cost of
about 5.8 s for a 1504 px frame against 0.9 s.

Licence: StableNormal is Apache-2.0 on both the code and the published
weights, so unlike the depth and albedo backends it carries no commercial
restriction.

Two compatibility measures are needed and both are deliberate:

*Weights are pre-fetched.*  ``hubconf`` builds the Hugging Face repo id
with ``os.path.join``, which produces a backslash on Windows and fails repo
id validation.  Downloading into ``local_cache_dir/<version>`` -- exactly
where it then looks -- avoids that path entirely and keeps the checkpoints
under the project's ``models/`` directory like every other model here.

*One diffusers symbol is aliased.*  StableNormal targets diffusers 0.28 and
its custom pipeline imports ``diffusers.models.controlnet``, which has since
moved to ``diffusers.models.controlnets.controlnet``.  Pinning their whole
requirements set is not an option: it also pins transformers 4.36 and torch
2.2, which would break the depth and albedo backends outright.  Aliasing the
one relocated module is the smallest change that works.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import os
import sys
import threading
from typing import Callable

import numpy as np

from .base import LOGGER, resize_image, select_device

#: torch.hub entry point.  The turbo variant is the YOSO feed-forward pass
#: alone; the full ``StableNormal`` adds a diffusion refinement that costs
#: far more time than this application's one-shot import budget allows.
DEFAULT_VARIANT = "StableNormal_turbo"

#: Weight repositories on Hugging Face, per variant.
_WEIGHTS = {
    "StableNormal_turbo": ("yoso-normal-v0-3",),
    "StableNormal": ("yoso-normal-v0-3", "stable-normal-v0-1"),
}

HUB_REPO = "Stable-X/StableNormal"

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "stablenormal"
)

#: Axis signs converting StableNormal's frame to ours (X right, Y up, Z
#: away from the camera).  Determined by fitting all eight sign
#: combinations against known-good analytic normals rather than read off a
#: paper: the winner scores +0.969 mean agreement, and on the 40% of pixels
#: where the X component actually carries signal it scores +0.894 against
#: +0.055 for the alternative sign.
AXIS_SIGNS = np.array([-1.0, 1.0, -1.0], dtype=np.float32)

#: Seed applied before each inference.  StableNormal descends from a
#: diffusion pipeline and samples noise even in the one-step turbo variant,
#: so without this the same photograph yields slightly different normals on
#: every import -- mean agreement 0.9973, with about 4% of pixels differing
#: by more than eight degrees.  Small, but a relighting tool should be
#: reproducible.  Seeding makes it bit-exact.
INFERENCE_SEED = 1234

_PREDICTOR_CACHE: dict[tuple[str, str], object] = {}
_CACHE_LOCK = threading.Lock()


@contextlib.contextmanager
def _quiet_diffusers():
    """Silence diffusers' device-move chatter for the duration of a block."""
    try:
        from diffusers.utils import logging as dlog
    except ImportError:
        yield
        return
    previous = dlog.get_verbosity()
    dlog.set_verbosity_error()
    try:
        yield
    finally:
        dlog.set_verbosity(previous)


@contextlib.contextmanager
def _seeded(seed: int):
    """Run a block from a fixed RNG state, then restore what was there.

    Restoring matters because the seed is global: leaving it set would make
    every later torch operation in the process deterministic as a side
    effect of one inference call.
    """
    import torch

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@functools.lru_cache(maxsize=1)
def is_available() -> bool:
    """True when the optional diffusion stack is installed.

    Checks the module spec rather than importing: the UI asks this on the
    main thread while building the inspector, and importing diffusers costs
    seconds.
    """
    try:
        return importlib.util.find_spec("diffusers") is not None
    except (ImportError, ValueError):
        return False


def _install_diffusers_alias() -> None:
    """Map the pre-0.29 controlnet module path onto its current location."""
    if "diffusers.models.controlnet" in sys.modules:
        return
    try:
        import diffusers.models.controlnets.controlnet as relocated
    except ImportError:
        return  # new enough, or old enough, that no alias is needed
    sys.modules.setdefault("diffusers.models.controlnet", relocated)


def _ensure_weights(variant: str, progress: Callable[[str], None] | None) -> str:
    """Download the checkpoints this variant needs into ``MODELS_DIR``."""
    from huggingface_hub import snapshot_download

    os.makedirs(MODELS_DIR, exist_ok=True)
    for repo in _WEIGHTS.get(variant, _WEIGHTS[DEFAULT_VARIANT]):
        target = os.path.join(MODELS_DIR, repo)
        if os.path.isdir(target) and os.listdir(target):
            continue
        if progress:
            progress(f"Normals: downloading {repo} (first run, ~2 GB)")
        LOGGER.info("Fetching Stable-X/%s into %s", repo, target)
        snapshot_download(repo_id=f"Stable-X/{repo}", local_dir=target)
    return MODELS_DIR


class StableNormalBackend:
    """Lazy-loading wrapper around the StableNormal torch.hub predictor."""

    def __init__(self, variant: str = DEFAULT_VARIANT, resolution: int = 1024) -> None:
        self.variant = variant
        #: Processing resolution handed to the predictor.  Output is
        #: returned at the input resolution regardless.
        self.resolution = int(resolution)
        self._predictor = None
        self._lock = threading.Lock()
        self._load_failed = False
        self._backend = f"StableNormal {variant}"

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    # -- loading -----------------------------------------------------------
    def ensure_loaded(self, progress: Callable[[str], None] | None = None) -> bool:
        if self._predictor is not None:
            return True
        if self._load_failed or not is_available():
            return False

        with self._lock:
            if self._predictor is not None:
                return True
            if self._load_failed:
                return False

            device = select_device()
            key = (self.variant, device.torch_device)
            with _CACHE_LOCK:
                cached = _PREDICTOR_CACHE.get(key)
            if cached is not None:
                self._predictor = cached
                self._backend = f"StableNormal {self.variant} @ {device.name}"
                return True

            try:
                import torch

                _install_diffusers_alias()
                cache_dir = _ensure_weights(self.variant, progress)

                if progress:
                    progress("Normals: loading StableNormal")
                # trust_repo is scoped to this one repository rather than
                # granted globally; torch.hub would otherwise prompt on
                # stdin, which raises EOFError on the worker thread.
                predictor = torch.hub.load(
                    HUB_REPO,
                    self.variant,
                    trust_repo=True,
                    local_cache_dir=cache_dir,
                    device=(
                        "cuda:0" if device.torch_device == "cuda" else device.torch_device
                    ),
                )
                self._predictor = predictor
                with _CACHE_LOCK:
                    _PREDICTOR_CACHE[key] = predictor
                self._backend = f"StableNormal {self.variant} @ {device.name}"
                LOGGER.info("Loaded %s", self._backend)
                return True
            except Exception as exc:
                LOGGER.warning("Could not load StableNormal: %s", exc)
                LOGGER.debug("StableNormal load failure detail", exc_info=True)
                self._load_failed = True
                return False

    # -- inference ---------------------------------------------------------
    def estimate(
        self,
        image_srgb: np.ndarray,
        progress: Callable[[str], None] | None = None,
    ) -> np.ndarray:
        """Return unit normals in our camera space at the input resolution."""
        if not self.ensure_loaded(progress):
            raise RuntimeError("StableNormal is not available")

        from PIL import Image

        height, width = image_srgb.shape[:2]
        if progress:
            progress(f"Normals: StableNormal {self.variant}")

        device = select_device()
        self.to("cuda:0" if device.torch_device == "cuda" else device.torch_device)

        rgb_u8 = (np.clip(image_srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        with _seeded(INFERENCE_SEED):
            result = self._predictor(
                Image.fromarray(rgb_u8),
                resolution=self.resolution,
                match_input_resolution=True,
            )

        encoded = np.asarray(result).astype(np.float32) / 255.0
        if encoded.ndim != 3 or encoded.shape[2] < 3:
            raise ValueError(f"StableNormal returned an unexpected map: {encoded.shape}")
        normal = encoded[..., :3] * 2.0 - 1.0
        normal *= AXIS_SIGNS

        if normal.shape[:2] != (height, width):
            normal = resize_image(normal, width, height)

        length = np.linalg.norm(normal, axis=-1, keepdims=True)
        normal = np.divide(normal, np.maximum(length, 1e-6), dtype=np.float32)

        if not np.all(np.isfinite(normal)):
            raise ValueError("StableNormal returned non-finite normals")
        return np.ascontiguousarray(normal.astype(np.float32))

    # -- residency ---------------------------------------------------------
    def to(self, device: str) -> None:
        """Move the predictor between host and device memory."""
        if self._predictor is None:
            return
        # diffusers warns that an fp16 pipeline cannot run on the CPU. True,
        # but irrelevant: the CPU is only where the weights are parked
        # between imports, never where they are executed.
        with _quiet_diffusers():
            try:
                self._predictor.to(device)
            except Exception as exc:  # pragma: no cover - upstream API drift
                LOGGER.debug("Could not move StableNormal to %s: %s", device, exc)

    def offload(self) -> None:
        """Park the weights in host memory and hand the VRAM back.

        The pipeline runs depth, then normals, then de-lighting, strictly
        in sequence and once per import, so nothing is served by all three
        staying resident.  On an 8 GB card they do not fit: this model
        alone peaks at 4.85 GB, and leaving it there pushed the albedo
        stage from 1.1 s to 10.6 s as the allocator began spilling to host
        memory.  A round trip over PCIe is far cheaper than that.
        """
        self.to("cpu")
        _empty_cuda_cache()

    def release(self) -> None:
        self._predictor = None
        with _CACHE_LOCK:
            _PREDICTOR_CACHE.clear()
        _empty_cuda_cache()
