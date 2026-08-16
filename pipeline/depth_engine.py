"""Monocular depth estimation.

Primary backend is **Ultralytics YOLO26 depth** (``yolo26m-depth.pt``), which
predicts *metric* depth: every pixel is a distance from the camera in metres,
from an unbounded log-depth head covering roughly 0.02-150 m.  That is a
meaningful upgrade for this application over a relative-depth model, because
the rest of the pipeline is physically parameterised -- inverse-square light
falloff, shadow ray lengths and light placement are all distances, and with
metric depth they are distances in real units rather than in an arbitrary
window.

Two fallbacks sit behind it, in order:

* **Depth Anything V2** via ``transformers``, which predicts relative inverse
  depth (disparity).  That is mapped onto a plausible metric window by
  :func:`disparity_to_depth` -- proportionally sensible, not truly metric.
* **An analytic prior** from two weak monocular cues, so the application
  still opens an image with no models and no network at all.

:attr:`DepthEngine.is_metric` reports which regime is live, and the value is
recorded in the G-buffer metadata and the export manifest so downstream
consumers are never left guessing at the units.
"""

from __future__ import annotations

import os
import shutil
import threading
from typing import Callable

import cv2
import numpy as np

from .base import (
    DEFAULT_MAX_SIDE,
    LOGGER,
    disparity_to_depth,
    fit_to_max_side,
    percentile_normalize,
    resize_image,
    select_device,
    smooth_depth_edges,
    torch_available,
)

#: Ultralytics YOLO26 depth checkpoint.  The family is
#: ``yolo26{n,s,m,l,x}-depth.pt``; medium is the default balance of accuracy
#: and the ~30 ms inference this app budgets for.
DEFAULT_YOLO_MODEL = "yolo26m-depth.pt"

#: Relative-depth models tried if Ultralytics is unavailable.
RELATIVE_MODEL_CANDIDATES = (
    "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything/Depth-Anything-V2-Base-hf",
    "LiheYoung/depth-anything-small-hf",
    "Intel/dpt-swinv2-tiny-256",
)

#: Bounds of the YOLO26 depth head, per the Ultralytics task documentation.
METRIC_MIN_M = 0.02
METRIC_MAX_M = 150.0

#: Downloaded checkpoints live here rather than in the working directory.
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)


class DepthEngine:
    """Lazy-loading depth estimator with a metric-first backend chain.

    The model is loaded on first :meth:`estimate` rather than in ``__init__``
    so constructing the engine on the UI thread is free; the load itself
    happens on the worker thread.
    """

    def __init__(
        self,
        model_id: str | None = None,
        max_side: int = DEFAULT_MAX_SIDE,
        near: float = 0.6,
        far: float = 6.0,
    ) -> None:
        #: Explicit override.  A ``.pt`` path or name routes to Ultralytics;
        #: anything else is treated as a Hugging Face model id.
        self.model_id = model_id
        #: Only applies to the relative-depth fallback.  The YOLO backend
        #: letterboxes internally to its trained input size, so pre-resizing
        #: would cost accuracy and save nothing.
        self.max_side = int(max_side)
        self.near = float(near)
        self.far = float(far)

        self._yolo = None
        self._model = None
        self._processor = None
        self._backend = "uninitialised"
        self._kind = "none"
        self._lock = threading.Lock()
        self._load_failed = False

    # -- introspection -----------------------------------------------------
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_deep(self) -> bool:
        return self._yolo is not None or self._model is not None

    @property
    def is_metric(self) -> bool:
        """True when depth values are metres rather than a relative scale."""
        return self._kind == "yolo"

    # -- model loading -----------------------------------------------------
    def _ensure_model(self) -> bool:
        """Load the best available backend once. False means use the prior."""
        if self._kind in ("yolo", "transformers"):
            return True
        if self._load_failed or not torch_available():
            return False

        with self._lock:
            if self._kind in ("yolo", "transformers"):
                return True
            if self._load_failed:
                return False

            explicit_hf = self.model_id is not None and not self.model_id.endswith(".pt")
            if not explicit_hf and self._load_yolo():
                return True
            if self._load_transformers():
                return True

            self._load_failed = True
            self._kind = "analytic"
            self._backend = "analytic gradient (no model available)"
            return False

    # -- Ultralytics YOLO26 ------------------------------------------------
    def _load_yolo(self) -> bool:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            LOGGER.info("Ultralytics unavailable (%s); trying relative-depth models", exc)
            return False

        name = self.model_id or DEFAULT_YOLO_MODEL
        try:
            weights = self._resolve_weights(name)
            LOGGER.info("Loading YOLO26 depth model %s", weights)
            model = YOLO(weights)
            if getattr(model, "task", None) != "depth":
                LOGGER.warning(
                    "%s is a '%s' model, not a depth model; ignoring it",
                    name, getattr(model, "task", "?"),
                )
                return False
            self._yolo = model
            self._kind = "yolo"
            self.model_id = name
            device = select_device()
            # Report the device only, not select_device()'s fp16 *capability*
            # flag: this path deliberately runs at fp32.
            self._backend = f"{name} @ {device.name} ({device.torch_device}, metric)"
            self._stash_weights(name)
            return True
        except Exception as exc:  # network, corrupt download, unsupported build
            LOGGER.warning("Could not load YOLO depth model %s: %s", name, exc)
            return False

    @staticmethod
    def _resolve_weights(name: str) -> str:
        """Prefer a checkpoint already in ``models/``, else let YOLO fetch it.

        Ultralytics auto-downloads a bare name into the *current working
        directory*, which means launching the app from somewhere else
        re-downloads 45 MB.  Keeping our own copy under the project makes
        that a one-time cost.
        """
        if os.path.isabs(name) or os.sep in name or (os.altsep and os.altsep in name):
            return name
        local = os.path.join(MODELS_DIR, name)
        return local if os.path.isfile(local) else name

    @staticmethod
    def _stash_weights(name: str) -> None:
        """Move a freshly downloaded checkpoint out of the CWD into models/."""
        if os.sep in name or os.path.isabs(name):
            return
        target = os.path.join(MODELS_DIR, name)
        if os.path.isfile(target) or not os.path.isfile(name):
            return
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            shutil.move(name, target)
            LOGGER.info("Moved %s into %s", name, MODELS_DIR)
        except OSError as exc:
            # Cosmetic only: the model is loaded and working either way.
            LOGGER.debug("Could not relocate %s: %s", name, exc)

    # -- transformers relative depth ---------------------------------------
    def _load_transformers(self) -> bool:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            LOGGER.warning("transformers unavailable, using analytic depth: %s", exc)
            return False

        device = select_device()
        candidates = (
            (self.model_id,)
            if self.model_id and not self.model_id.endswith(".pt")
            else RELATIVE_MODEL_CANDIDATES
        )
        for candidate in candidates:
            try:
                LOGGER.info("Loading relative depth model %s on %s", candidate, device)
                processor = AutoImageProcessor.from_pretrained(candidate)
                model = AutoModelForDepthEstimation.from_pretrained(candidate)
                model.eval().to(device.torch_device)
                if device.fp16:
                    model.half()
                self._processor = processor
                self._model = model
                self._kind = "transformers"
                self.model_id = candidate
                self._backend = f"{candidate} @ {device} (relative)"
                return True
            except Exception as exc:
                LOGGER.warning("Could not load depth model %s: %s", candidate, exc)
        return False

    # -- inference ---------------------------------------------------------
    def estimate(
        self,
        image_srgb: np.ndarray,
        progress: Callable[[str], None] | None = None,
    ) -> np.ndarray:
        """Return an ``H x W`` float32 depth map matching the input size.

        ``image_srgb`` is RGB float32 in [0, 1] at full resolution.  Depth is
        strictly positive with smaller values closer to the camera; the units
        are metres when :attr:`is_metric`, and an arbitrary but proportionate
        scale otherwise.
        """
        height, width = image_srgb.shape[:2]

        if self._ensure_model() and self._kind == "yolo":
            if progress:
                progress(f"Depth: {self.model_id} (metric)")
            depth = self._infer_yolo(image_srgb)
            return smooth_depth_edges(depth)

        if self._kind == "transformers":
            if progress:
                progress(f"Depth: {self.model_id}")
            work_w, work_h = fit_to_max_side(width, height, self.max_side)
            disparity = self._infer_transformers(resize_image(image_srgb, work_w, work_h))
        else:
            if progress:
                progress("Depth: analytic fallback")
            work_w, work_h = fit_to_max_side(width, height, self.max_side)
            disparity = self._analytic_disparity(resize_image(image_srgb, work_w, work_h))

        disparity = resize_image(disparity, width, height)
        return smooth_depth_edges(disparity_to_depth(disparity, self.near, self.far))

    def _infer_yolo(self, image_srgb: np.ndarray) -> np.ndarray:
        """Run YOLO26 depth and return metric depth at the input resolution.

        Ultralytics takes numpy input in **BGR** order, matching
        ``cv2.imread``.  Handing it RGB is not a no-op -- on the test scene it
        shifts predictions by up to 0.65 m -- so the conversion is mandatory
        rather than cosmetic.  The model letterboxes internally to its trained
        input size and returns a map already aligned to the original frame,
        so there is nothing to resize on either side.
        """
        rgb_u8 = (np.clip(image_srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

        device = select_device()
        # No `imgsz` override: the metric calibration is tied to the trained
        # input size, and forcing another one measurably shifts the scale.
        # No `half`/`quantize` either: inference is ~30 ms at fp32 on a
        # mid-range GPU, so there is nothing to buy.
        results = self._yolo(bgr, device=device.torch_device, verbose=False)

        depth_map = getattr(results[0], "depth", None)
        if depth_map is None:
            raise RuntimeError("YOLO result carried no depth map")

        data = depth_map.data
        depth = data.float().cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        depth = np.squeeze(depth).astype(np.float32)

        if depth.shape != image_srgb.shape[:2]:
            depth = resize_image(depth, image_srgb.shape[1], image_srgb.shape[0])

        # Guard the unprojection against non-finite or non-positive metres.
        depth = np.nan_to_num(depth, nan=METRIC_MAX_M, posinf=METRIC_MAX_M, neginf=METRIC_MIN_M)
        return np.clip(depth, METRIC_MIN_M, METRIC_MAX_M).astype(np.float32)

    def _infer_transformers(self, image_srgb: np.ndarray) -> np.ndarray:
        import torch

        device = select_device()
        rgb_u8 = (np.clip(image_srgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        inputs = self._processor(images=rgb_u8, return_tensors="pt")
        inputs = {k: v.to(device.torch_device) for k, v in inputs.items()}
        if device.fp16:
            inputs = {
                k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()
            }

        with torch.no_grad():
            outputs = self._model(**inputs)
            predicted = outputs.predicted_depth

        if predicted.ndim == 3:
            predicted = predicted.unsqueeze(1)
        predicted = torch.nn.functional.interpolate(
            predicted.float(),
            size=(image_srgb.shape[0], image_srgb.shape[1]),
            mode="bicubic",
            align_corners=False,
        )
        return predicted.squeeze().detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _analytic_disparity(image_srgb: np.ndarray) -> np.ndarray:
        """Plausible stand-in disparity when no depth model is available.

        Combines two weak monocular cues that hold for most photographs: the
        bottom of the frame is usually nearer than the top, and large-scale
        brightness correlates with proximity to the camera in lit scenes.
        This is a usability fallback, not an estimate anyone should trust.
        """
        height, width = image_srgb.shape[:2]
        gray = cv2.cvtColor(np.clip(image_srgb, 0, 1), cv2.COLOR_RGB2GRAY).astype(np.float32)

        ground_plane = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        ground_plane = np.repeat(ground_plane, width, axis=1)

        base = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(width, height) * 0.02)
        base = percentile_normalize(base, 2.0, 98.0)

        disparity = 0.65 * ground_plane + 0.35 * base
        disparity = cv2.GaussianBlur(disparity, (0, 0), sigmaX=max(width, height) * 0.01)
        return percentile_normalize(disparity, 1.0, 99.0)

    def release(self) -> None:
        """Drop model references and free VRAM (called on shutdown)."""
        self._yolo = None
        self._model = None
        self._processor = None
        if torch_available():
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
