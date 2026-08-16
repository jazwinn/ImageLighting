"""Background inference worker.

The three estimators run once per imported image and take seconds; the
viewport must keep rendering at 60+ FPS throughout.  This module owns that
split: a :class:`PipelineWorker` ``QObject`` lives on a dedicated
``QThread``, and the only thing that crosses back to the GUI thread is a
finished :class:`GBuffer` (plus progress strings).

Model instances are held on the worker and reused across images, so a
second import pays no load cost.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from core.gbuffer import CameraIntrinsics, GBuffer
from core.imageio import load_image
from core.qt_compat import QtCore, Signal, Slot
from pipeline.base import DEFAULT_MAX_SIDE, LOGGER, select_device, warm_up_torch
from pipeline.delighting_engine import DelightingEngine, DelightSettings
from pipeline.depth_engine import DepthEngine
from pipeline.normal_engine import NormalEngine


@dataclass
class PipelineConfig:
    """Everything the worker needs to know before it starts."""

    #: Working resolution cap for the estimators.
    inference_max_side: int = DEFAULT_MAX_SIDE
    #: Resolution cap for the G-buffer itself (and therefore the textures).
    buffer_max_side: int = 1600
    camera_fov_y: float = 55.0
    #: Metric window for the *relative*-depth fallback only.  The YOLO26
    #: backend predicts metres directly and ignores these.
    depth_near: float = 0.6
    depth_far: float = 6.0
    #: ``None`` selects YOLO26 metric depth; a ``.pt`` name/path picks a
    #: different Ultralytics checkpoint; anything else is a HF model id.
    depth_model: str | None = None
    normal_onnx: str | None = None
    albedo_onnx: str | None = None
    #: Prefer StableNormal (RGB -> normals) over differentiating depth.
    neural_normals: bool = True
    delight: DelightSettings = field(default_factory=DelightSettings)


class PipelineWorker(QtCore.QObject):
    """Runs depth -> normals -> de-lighting off the GUI thread."""

    started = Signal(str)
    progress = Signal(str, int)  # message, percent
    finished = Signal(object)  # GBuffer
    failed = Signal(str)

    def __init__(self, config: PipelineConfig | None = None) -> None:
        super().__init__()
        self.config = config or PipelineConfig()
        self._depth = DepthEngine(
            model_id=self.config.depth_model,
            max_side=self.config.inference_max_side,
            near=self.config.depth_near,
            far=self.config.depth_far,
        )
        self._normal = NormalEngine(
            onnx_path=self.config.normal_onnx,
            max_side=self.config.inference_max_side,
            use_neural=self.config.neural_normals,
        )
        self._delight = DelightingEngine(
            onnx_path=self.config.albedo_onnx,
            max_side=self.config.inference_max_side,
            settings=self.config.delight,
        )
        self._cancelled = False

    # -- introspection used by the status bar ------------------------------
    def backend_summary(self) -> dict[str, str]:
        return {
            "device": str(select_device()),
            "depth": self._depth.backend,
            "normal": self._normal.backend,
            "albedo": self._delight.backend,
        }

    @Slot()
    def cancel(self) -> None:
        self._cancelled = True

    @Slot(bool)
    def set_neural_normals(self, enabled: bool) -> None:
        """Switch the normal backend.  Takes effect on the next import."""
        self._normal.use_neural = bool(enabled)

    # -- slots -------------------------------------------------------------
    @Slot(str)
    def process_path(self, path: str) -> None:
        """Load an image from disk, then run the full pipeline on it."""
        try:
            image = load_image(path, max_side=self.config.buffer_max_side)
        except Exception as exc:
            LOGGER.exception("Failed to load %s", path)
            self.failed.emit(f"Could not load image:\n{exc}")
            return
        self._run(image, path)

    @Slot(object, str)
    def process_array(self, image: np.ndarray, label: str = "") -> None:
        self._run(image, label or None)

    @Slot(object)
    def redo_delighting(self, payload: object) -> None:
        """Re-run only the de-lighting stage on an existing G-buffer.

        De-lighting has user-facing tunables; changing one should not force
        a depth and normal re-inference that would take seconds.
        """
        buffer, settings = payload  # type: ignore[misc]
        if not isinstance(buffer, GBuffer):
            self.failed.emit("Internal error: expected a G-buffer to re-light")
            return
        try:
            self.started.emit("Re-running de-lighting")
            self._delight.settings = settings
            albedo, shading = self._delight.decompose(
                buffer.original,
                buffer.normal,
                buffer.depth,
                buffer.intrinsics,
                progress=lambda m: self.progress.emit(m, 60),
            )
            buffer.albedo = albedo
            buffer.shading = shading
            buffer.meta["albedo_backend"] = self._delight.backend
            buffer.validate()
            self.progress.emit("De-lighting updated", 100)
            self.finished.emit(buffer)
        except Exception as exc:
            LOGGER.exception("De-lighting failed")
            self.failed.emit(f"De-lighting failed:\n{exc}\n\n{traceback.format_exc()}")

    # -- pipeline ----------------------------------------------------------
    def _run(self, image: np.ndarray, source_path: str | None) -> None:
        self._cancelled = False
        t0 = time.perf_counter()
        try:
            self.started.emit(source_path or "image")
            height, width = image.shape[:2]
            intrinsics = CameraIntrinsics.from_fov(width, height, self.config.camera_fov_y)

            self.progress.emit("Estimating depth…", 5)
            depth = self._depth.estimate(image, progress=lambda m: self.progress.emit(m, 15))
            if self._cancelled:
                return
            t_depth = time.perf_counter()

            self.progress.emit("Estimating surface normals…", 45)
            normal = self._normal.estimate(
                image, depth, intrinsics, progress=lambda m: self.progress.emit(m, 50)
            )
            # Hand the normal model's VRAM back before the next stage claims
            # its own.  These are sequential one-shot stages, and on an 8 GB
            # card they do not fit side by side -- leaving both resident made
            # a 1.1 s de-light take 10.6 s.
            self._normal.offload()
            if self._cancelled:
                return
            t_normal = time.perf_counter()

            self.progress.emit("De-lighting to flat albedo…", 70)
            albedo, shading = self._delight.decompose(
                image, normal, depth, intrinsics,
                progress=lambda m: self.progress.emit(m, 80),
            )
            self._delight.offload()
            if self._cancelled:
                return
            t_albedo = time.perf_counter()

            buffer = GBuffer(
                original=np.ascontiguousarray(image.astype(np.float32)),
                depth=np.ascontiguousarray(depth.astype(np.float32)),
                normal=np.ascontiguousarray(normal.astype(np.float32)),
                albedo=np.ascontiguousarray(albedo.astype(np.float32)),
                shading=np.ascontiguousarray(shading.astype(np.float32)),
                intrinsics=intrinsics,
                source_path=source_path,
                meta={
                    "device": str(select_device()),
                    "depth_backend": self._depth.backend,
                    "depth_is_metric": self._depth.is_metric,
                    "depth_units": "metres" if self._depth.is_metric else "relative",
                    "normal_backend": self._normal.backend,
                    "albedo_backend": self._delight.backend,
                    "depth_ms": round((t_depth - t0) * 1000.0, 1),
                    "normal_ms": round((t_normal - t_depth) * 1000.0, 1),
                    "albedo_ms": round((t_albedo - t_normal) * 1000.0, 1),
                    "total_ms": round((t_albedo - t0) * 1000.0, 1),
                    "resolution": f"{width}x{height}",
                },
            )
            buffer.validate()
            self.progress.emit(f"Ready in {buffer.meta['total_ms']:.0f} ms", 100)
            self.finished.emit(buffer)
        except Exception as exc:
            LOGGER.exception("Pipeline failed")
            self.failed.emit(f"Pipeline failed:\n{exc}\n\n{traceback.format_exc()}")

    def release(self) -> None:
        self._depth.release()
        self._normal.release()
        self._delight.release()


class PipelineController(QtCore.QObject):
    """GUI-side handle that owns the worker thread and its lifetime."""

    started = Signal(str)
    progress = Signal(str, int)
    finished = Signal(object)
    failed = Signal(str)

    _request_path = Signal(str)
    _request_array = Signal(object, str)
    _request_delight = Signal(object)
    _request_neural_normals = Signal(bool)

    def __init__(self, config: PipelineConfig | None = None, parent=None) -> None:
        super().__init__(parent)
        # Do this before the worker thread exists: torch's thread pool has
        # to be created on the main thread or the second inference corrupts
        # the heap.  See warm_up_torch for the full story.
        warm_up_torch()

        self.worker = PipelineWorker(config)
        self.thread = QtCore.QThread()
        self.thread.setObjectName("PipelineWorker")
        self.worker.moveToThread(self.thread)

        self._request_path.connect(self.worker.process_path)
        self._request_array.connect(self.worker.process_array)
        self._request_delight.connect(self.worker.redo_delighting)
        self._request_neural_normals.connect(self.worker.set_neural_normals)

        self.worker.started.connect(self.started)
        self.worker.progress.connect(self.progress)
        self.worker.finished.connect(self.finished)
        self.worker.failed.connect(self.failed)

        self.thread.start()

    def submit_path(self, path: str) -> None:
        self._request_path.emit(path)

    def submit_array(self, image: np.ndarray, label: str = "") -> None:
        self._request_array.emit(image, label)

    def submit_delight(self, buffer: GBuffer, settings: DelightSettings) -> None:
        self._request_delight.emit((buffer, settings))

    def set_neural_normals(self, enabled: bool) -> None:
        self._request_neural_normals.emit(bool(enabled))

    def backend_summary(self) -> dict[str, str]:
        return self.worker.backend_summary()

    def shutdown(self) -> None:
        self.worker.cancel()
        self.thread.quit()
        if not self.thread.wait(5000):
            LOGGER.warning("Pipeline thread did not exit cleanly; terminating")
            self.thread.terminate()
            self.thread.wait(1000)
        self.worker.release()
