"""Application shell: layout, menus, and the wiring between panels.

Owns the single :class:`Scene` instance, the :class:`PipelineController`
(and therefore the worker thread), and the three panels.  The data flow is
one-directional in each phase:

* import  -> controller -> worker thread -> ``GBuffer`` -> viewport + tabs
* editing -> inspector  -> ``Scene``     -> viewport repaint
"""

from __future__ import annotations

import json
import os

import numpy as np

from core.gbuffer import GBuffer
from core.imageio import IMAGE_FILTER
from core.qt_compat import Qt, QtCore, QtGui, QtWidgets
from core.scene import (
    LightType,
    Scene,
    ViewMode,
    default_fill_light,
    default_key_light,
)
from export.exporter import export_gbuffer_archive, export_relit_image
from pipeline.worker import PipelineConfig, PipelineController
from ui.gbuffer_tabs import GBufferTabs
from ui.gl_viewport import GLViewport
from ui.inspector import Inspector
from ui.widgets import ElidedLabel

APP_NAME = "Relighting Studio"
ORG_NAME = "ImageLighting"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: PipelineConfig | None = None) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1680, 980)
        self.setAcceptDrops(True)

        self.scene = Scene()
        self.buffer: GBuffer | None = None
        self._busy = False

        self.controller = PipelineController(config or PipelineConfig(), self)
        self.settings = QtCore.QSettings(ORG_NAME, APP_NAME)

        self._build_ui()
        self._build_menus()
        self._build_statusbar()
        self._connect()
        self._restore_geometry()

        self._set_has_image(False)
        self.status_message.setText("Open an image, or load the sample scene, to begin.")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.viewport = GLViewport(self.scene)
        self.tabs = GBufferTabs()
        self.inspector = Inspector(self.scene)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 4, 8)
        left_layout.addWidget(self.tabs)

        center = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center)
        center_layout.setContentsMargins(4, 8, 4, 8)
        center_layout.setSpacing(6)
        center_layout.addWidget(self.viewport, 1)

        self.hint_bar = QtWidgets.QLabel(
            "Drag a gizmo to move a light   ·   Ctrl-drag or Ctrl-wheel for depth   ·   "
            "Double-click to place on a surface   ·   Shift-drag or middle-drag to pan   ·   "
            "F to reset the view"
        )
        self.hint_bar.setObjectName("metaLabel")
        self.hint_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(self.hint_bar)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(4, 8, 8, 8)
        right_layout.addWidget(self.inspector)

        self.splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(left)
        self.splitter.addWidget(center)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 5)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([380, 850, 450])
        self.setCentralWidget(self.splitter)

    def _build_menus(self) -> None:
        menubar = self.menuBar()

        # --- File ---
        file_menu = menubar.addMenu("&File")
        self.action_open = file_menu.addAction("&Open image…")
        self.action_open.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.action_open.triggered.connect(self.open_image_dialog)

        self.action_sample = file_menu.addAction("Load &sample scene")
        self.action_sample.triggered.connect(self.load_sample)

        self.recent_menu = file_menu.addMenu("Open &recent")
        self._rebuild_recent_menu()

        file_menu.addSeparator()
        self.action_export_image = file_menu.addAction("&Export relit image…")
        self.action_export_image.setShortcut("Ctrl+E")
        self.action_export_image.triggered.connect(self.export_image_dialog)

        self.action_export_gbuffer = file_menu.addAction("Export &G-buffer archive…")
        self.action_export_gbuffer.setShortcut("Ctrl+Shift+E")
        self.action_export_gbuffer.triggered.connect(self.export_gbuffer_dialog)

        file_menu.addSeparator()
        self.action_save_scene = file_menu.addAction("Save lighting preset…")
        self.action_save_scene.triggered.connect(self.save_preset_dialog)
        self.action_load_scene = file_menu.addAction("Load lighting preset…")
        self.action_load_scene.triggered.connect(self.load_preset_dialog)

        file_menu.addSeparator()
        quit_action = file_menu.addAction("&Quit")
        quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)

        # --- Lights ---
        lights_menu = menubar.addMenu("&Lights")
        for kind, shortcut in (
            (LightType.POINT, "Ctrl+1"),
            (LightType.SPOT, "Ctrl+2"),
            (LightType.DIRECTIONAL, "Ctrl+3"),
        ):
            action = lights_menu.addAction(f"Add {kind.label.lower()} light")
            action.setShortcut(shortcut)
            action.triggered.connect(lambda _c=False, k=kind: self.inspector.add_light(k))
        lights_menu.addSeparator()
        remove_action = lights_menu.addAction("Remove selected light")
        remove_action.setShortcut("Del")
        remove_action.triggered.connect(self.inspector.remove_light)
        reset_action = lights_menu.addAction("Reset to default three-point setup")
        reset_action.triggered.connect(self.reset_default_lights)

        # --- View ---
        view_menu = menubar.addMenu("&View")
        self._view_actions: dict[ViewMode, QtGui.QAction] = {}
        group = QtGui.QActionGroup(self)
        group.setExclusive(True)
        for index, (mode, label) in enumerate(
            (
                (ViewMode.BEAUTY, "Beauty (relit)"),
                (ViewMode.ALBEDO, "Albedo"),
                (ViewMode.ORIGINAL, "Original"),
                (ViewMode.NORMAL, "Normals"),
                (ViewMode.DEPTH, "Depth"),
                (ViewMode.SHADING, "Shading mask"),
                (ViewMode.SPECULAR, "Specular only"),
            )
        ):
            action = view_menu.addAction(label)
            action.setCheckable(True)
            action.setShortcut(f"F{index + 1}")
            action.setChecked(mode == ViewMode.BEAUTY)
            action.triggered.connect(lambda _c=False, m=mode: self.inspector.set_view_mode(m))
            group.addAction(action)
            self._view_actions[mode] = action

        view_menu.addSeparator()
        reset_view = view_menu.addAction("Reset view")
        reset_view.setShortcut("F")
        reset_view.triggered.connect(self.viewport.reset_view)

        self.action_continuous = view_menu.addAction("Continuous rendering")
        self.action_continuous.setCheckable(True)
        self.action_continuous.setChecked(True)
        self.action_continuous.setToolTip(
            "Redraw every frame. Turn off to idle the GPU between edits."
        )
        self.action_continuous.toggled.connect(self.viewport.set_continuous)

        view_menu.addSeparator()
        self.action_show_passes = view_menu.addAction("Show G-buffer inspector")
        self.action_show_passes.setCheckable(True)
        self.action_show_passes.setChecked(True)
        self.action_show_passes.toggled.connect(
            lambda v: self.splitter.widget(0).setVisible(v)
        )
        self.action_show_inspector = view_menu.addAction("Show lighting inspector")
        self.action_show_inspector.setCheckable(True)
        self.action_show_inspector.setChecked(True)
        self.action_show_inspector.toggled.connect(
            lambda v: self.splitter.widget(2).setVisible(v)
        )

        # --- Help ---
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("Controls…").triggered.connect(self.show_controls)
        help_menu.addAction("Pipeline backends…").triggered.connect(self.show_backends)
        help_menu.addAction(f"About {APP_NAME}…").triggered.connect(self.show_about)

    def _build_statusbar(self) -> None:
        bar = self.statusBar()

        self.status_message = ElidedLabel("")
        self.status_message.setMinimumWidth(280)
        bar.addWidget(self.status_message, 1)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setFixedWidth(130)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        bar.addPermanentWidget(self.progress)

        self.cursor_label = QtWidgets.QLabel("")
        self.cursor_label.setObjectName("metaLabel")
        self.cursor_label.setMinimumWidth(250)
        bar.addPermanentWidget(self.cursor_label)

        self.fps_label = QtWidgets.QLabel("— fps")
        self.fps_label.setObjectName("metaLabel")
        self.fps_label.setMinimumWidth(62)
        bar.addPermanentWidget(self.fps_label)

        self.gpu_label = QtWidgets.QLabel("GL: initialising")
        self.gpu_label.setObjectName("metaLabel")
        bar.addPermanentWidget(self.gpu_label)

    def _connect(self) -> None:
        self.controller.started.connect(self._on_pipeline_started)
        self.controller.progress.connect(self._on_pipeline_progress)
        self.controller.finished.connect(self._on_pipeline_finished)
        self.controller.failed.connect(self._on_pipeline_failed)

        self.inspector.sceneChanged.connect(self.viewport.update)
        self.inspector.activeLightChanged.connect(lambda _i: self.viewport.update())
        self.inspector.delightRequested.connect(self._on_delight_requested)
        self.inspector.viewModeChanged.connect(self._on_view_mode_changed)
        self.inspector.neuralNormalsToggled.connect(self._on_neural_normals_toggled)

        self.viewport.lightSelected.connect(self._on_viewport_light_selected)
        self.viewport.lightMoved.connect(lambda _i: self.inspector.sync_active_light())
        self.viewport.cursorMoved.connect(self.cursor_label.setText)
        self.viewport.fpsUpdated.connect(lambda f: self.fps_label.setText(f"{f:5.1f} fps"))
        self.viewport.glInitialised.connect(lambda s: self.gpu_label.setText(f"GL: {s}"))
        self.viewport.glFailed.connect(self._on_gl_failed)

        self.tabs.set_capture_callback(self._capture_pass)

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------
    def open_image_dialog(self) -> None:
        start_dir = self.settings.value("last_open_dir", "", type=str)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open image", start_dir, IMAGE_FILTER
        )
        if path:
            self.open_image(path)

    def open_image(self, path: str) -> None:
        if self._busy:
            self.status_message.setText("Still processing the previous image…")
            return
        self.settings.setValue("last_open_dir", os.path.dirname(path))
        self._push_recent(path)
        self.setWindowTitle(f"{APP_NAME} — {os.path.basename(path)}")
        self.controller.submit_path(path)

    def load_sample(self) -> None:
        """Generate the bundled synthetic test scene and run it."""
        from tools.make_sample import ensure_sample_image

        try:
            path = ensure_sample_image()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Sample scene", f"Could not build it:\n{exc}")
            return
        self.open_image(path)

    # -- drag and drop -----------------------------------------------------
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls() and self._first_local_image(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        path = self._first_local_image(event.mimeData())
        if path:
            event.acceptProposedAction()
            self.open_image(path)

    @staticmethod
    def _first_local_image(mime: QtCore.QMimeData) -> str | None:
        from core.imageio import SUPPORTED_READ

        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            if os.path.splitext(path)[1].lower() in SUPPORTED_READ:
                return path
        return None

    # ------------------------------------------------------------------
    # Pipeline callbacks
    # ------------------------------------------------------------------
    def _on_pipeline_started(self, label: str) -> None:
        self._busy = True
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(2)
        self.status_message.setText(f"Processing {os.path.basename(label) or label}…")
        self._set_inputs_enabled(False)

    def _on_pipeline_progress(self, message: str, percent: int) -> None:
        self.progress.setValue(max(0, min(100, int(percent))))
        self.status_message.setText(message)

    def _on_pipeline_finished(self, buffer: object) -> None:
        assert isinstance(buffer, GBuffer)
        self._busy = False
        self.progress.setVisible(False)
        self._set_inputs_enabled(True)

        # A de-lighting recompute hands back the very same object it was
        # given, which is exactly how we tell it apart from a new import.
        first_import = buffer is not self.buffer
        self.buffer = buffer

        self.viewport.set_gbuffer(buffer, reset_view=first_import)
        self.tabs.set_gbuffer(buffer)

        if first_import or not self.scene.lights:
            self.reset_default_lights()

        self._set_has_image(True)
        meta = buffer.meta
        self.status_message.setText(
            f"{meta.get('resolution', '')} · depth {meta.get('depth_ms', 0):.0f} ms · "
            f"normals {meta.get('normal_ms', 0):.0f} ms · "
            f"de-light {meta.get('albedo_ms', 0):.0f} ms · {meta.get('device', '')}"
        )

    def _on_pipeline_failed(self, message: str) -> None:
        self._busy = False
        self.progress.setVisible(False)
        self._set_inputs_enabled(True)
        self.status_message.setText("Pipeline failed.")
        # The traceback is long; show the first line prominently and keep the
        # rest available rather than truncating it away.
        headline = message.strip().splitlines()[0] if message.strip() else "Unknown error"
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        box.setWindowTitle("Pipeline error")
        box.setText(headline)
        box.setDetailedText(message)
        box.exec()

    def _on_gl_failed(self, message: str) -> None:
        self.gpu_label.setText("GL: unavailable")
        QtWidgets.QMessageBox.critical(
            self,
            "OpenGL initialisation failed",
            "Could not create an OpenGL 3.3 core context.\n\n"
            f"{message}\n\n"
            "Update your graphics driver, or run with a software renderer:\n"
            "    set LIBGL_ALWAYS_SOFTWARE=1",
        )

    def _on_delight_requested(self, settings: object) -> None:
        if self.buffer is None or self._busy:
            return
        self.controller.submit_delight(self.buffer, settings)  # type: ignore[arg-type]

    def _on_neural_normals_toggled(self, enabled: bool) -> None:
        """Swap the normal backend and reprocess the current image.

        Normals feed the de-lighting fit and every shading term, so unlike
        the albedo controls there is nothing partial to recompute.
        """
        self.controller.set_neural_normals(enabled)
        if self.buffer is None or self._busy:
            return
        source = self.buffer.source_path
        if source and os.path.isfile(source):
            self.controller.submit_path(source)

    def _on_view_mode_changed(self, mode: object) -> None:
        action = self._view_actions.get(mode)  # type: ignore[arg-type]
        if action is not None and not action.isChecked():
            action.setChecked(True)

    def _on_viewport_light_selected(self, index: int) -> None:
        self.inspector.rebuild_light_list()

    # ------------------------------------------------------------------
    # Lights
    # ------------------------------------------------------------------
    def reset_default_lights(self) -> None:
        """Place a key/fill pair scaled to the reconstructed scene."""
        if self.buffer is None:
            return
        center = self.buffer.scene_center()
        radius = self.buffer.scene_radius()
        self.scene.lights = [default_key_light(center, radius), default_fill_light(center, radius)]
        self.scene.active_index = 0
        # Shadow reach follows the scene's *depth extent*, not the frame
        # width: a shadow ray travels front-to-back through the geometry, and
        # the ray is divided into a fixed number of steps, so an over-long
        # reach buys nothing and coarsens every step until thin occluders
        # start falling between samples and the penumbra goes stippled.
        # The frame width is kept as a floor for very flat, planar scenes.
        span = self.buffer.depth_span()
        self.scene.shadows.max_distance = float(max(span * 1.5, radius * 0.6, 0.4))
        self.scene.shadows.bias = float(max(span * 0.015, 0.002))
        self.inspector.sync_from_scene()
        self.viewport.update()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _default_export_name(self, suffix: str, extension: str) -> str:
        base = "render"
        if self.buffer is not None and self.buffer.source_path:
            base = os.path.splitext(os.path.basename(self.buffer.source_path))[0]
        directory = self.settings.value("last_export_dir", "", type=str)
        return os.path.join(directory, f"{base}_{suffix}{extension}")

    def export_image_dialog(self) -> None:
        if self.buffer is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export relit image",
            self._default_export_name("relit", ".png"),
            "PNG (*.png);;JPEG (*.jpg);;WebP (*.webp)",
        )
        if not path:
            return
        try:
            image = self._render_beauty()
            export_relit_image(path, image)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.settings.setValue("last_export_dir", os.path.dirname(path))
        self.status_message.setText(f"Exported {os.path.basename(path)}")

    def export_gbuffer_dialog(self) -> None:
        if self.buffer is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export G-buffer archive",
            self._default_export_name("gbuffer", ".zip"),
            "ZIP archive (*.zip)",
        )
        if not path:
            return
        try:
            beauty = self._render_beauty()
            export_gbuffer_archive(path, self.buffer, self.scene, beauty)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.settings.setValue("last_export_dir", os.path.dirname(path))
        self.status_message.setText(f"Exported {os.path.basename(path)}")

    def _render_beauty(self) -> np.ndarray:
        """Offscreen beauty render, independent of the current debug view."""
        previous = self.scene.render.view_mode
        self.scene.render.view_mode = ViewMode.BEAUTY
        try:
            return self.viewport.render_to_array()
        finally:
            self.scene.render.view_mode = previous

    def _capture_pass(self, mode: ViewMode) -> np.ndarray | None:
        """Render a single pass offscreen, for the live inspector tab."""
        if self.buffer is None or self.viewport.renderer is None:
            return None
        previous = self.scene.render.view_mode
        self.scene.render.view_mode = mode
        try:
            return self.viewport.render_to_array()
        finally:
            self.scene.render.view_mode = previous

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    def save_preset_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save lighting preset", self._default_export_name("lighting", ".json"),
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.scene.to_dict(), handle, indent=2)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.status_message.setText(f"Saved preset {os.path.basename(path)}")

    def load_preset_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load lighting preset", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            loaded = Scene.from_dict(data)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return

        # Mutate the existing Scene in place: the viewport and inspector both
        # hold a reference to it, and rebinding would leave them out of sync.
        self.scene.lights = loaded.lights
        self.scene.material = loaded.material
        self.scene.shadows = loaded.shadows
        self.scene.render = loaded.render
        self.scene.active_index = loaded.active_index
        self.viewport.scene = self.scene
        self.inspector.sync_from_scene()
        self.viewport.update()
        self.status_message.setText(f"Loaded preset {os.path.basename(path)}")

    # ------------------------------------------------------------------
    # Recent files
    # ------------------------------------------------------------------
    def _push_recent(self, path: str) -> None:
        recent = list(self.settings.value("recent_files", [], type=list) or [])
        path = os.path.abspath(path)
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        self.settings.setValue("recent_files", recent[:8])
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = [
            p for p in (self.settings.value("recent_files", [], type=list) or [])
            if os.path.isfile(p)
        ]
        if not recent:
            action = self.recent_menu.addAction("(nothing yet)")
            action.setEnabled(False)
            return
        for path in recent:
            action = self.recent_menu.addAction(os.path.basename(path))
            action.setToolTip(path)
            action.triggered.connect(lambda _c=False, p=path: self.open_image(p))

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------
    def show_controls(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Viewport controls",
            "<table cellpadding='4'>"
            "<tr><td><b>Left drag on a gizmo</b></td><td>Move the light in the image plane</td></tr>"
            "<tr><td><b>Left click on the image</b></td><td>Move the selected light there</td></tr>"
            "<tr><td><b>Ctrl / Alt + drag</b></td><td>Move the light toward or away from the camera</td></tr>"
            "<tr><td><b>Right drag</b></td><td>Same, on the light under the cursor</td></tr>"
            "<tr><td><b>Double click</b></td><td>Snap the light just off the surface</td></tr>"
            "<tr><td><b>Wheel</b></td><td>Zoom the view</td></tr>"
            "<tr><td><b>Ctrl + wheel</b></td><td>Change the selected light's depth</td></tr>"
            "<tr><td><b>Middle / Shift + drag</b></td><td>Pan</td></tr>"
            "<tr><td><b>F</b></td><td>Reset the view</td></tr>"
            "<tr><td><b>F1 – F7</b></td><td>Switch G-buffer view mode</td></tr>"
            "</table>",
        )

    def show_backends(self) -> None:
        summary = self.controller.backend_summary()
        rows = "".join(
            f"<tr><td><b>{key.title()}</b></td><td>{value}</td></tr>"
            for key, value in summary.items()
        )
        QtWidgets.QMessageBox.information(
            self, "Pipeline backends", f"<table cellpadding='4'>{rows}</table>"
        )

    def show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<h3>{APP_NAME}</h3>"
            "<p>Single-image de-lighting and virtual relighting on a 2.5D "
            "deferred pipeline.</p>"
            "<p>Depth, surface normals and a de-lit albedo are estimated once per "
            "image on a background thread; lighting is then evaluated entirely in "
            "GLSL, with screen-space raymarched cast shadows.</p>",
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _set_has_image(self, has_image: bool) -> None:
        for action in (
            self.action_export_image,
            self.action_export_gbuffer,
            self.action_save_scene,
        ):
            action.setEnabled(has_image)
        self.inspector.setEnabled(has_image)

    def _set_inputs_enabled(self, enabled: bool) -> None:
        self.action_open.setEnabled(enabled)
        self.action_sample.setEnabled(enabled)
        self.inspector.delight_apply.setEnabled(enabled)

    def _restore_geometry(self) -> None:
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        sizes = self.settings.value("splitter_sizes")
        if sizes:
            try:
                self.splitter.setSizes([int(s) for s in sizes])
            except (TypeError, ValueError):
                pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("splitter_sizes", self.splitter.sizes())
        # Order matters: release GL objects while the context still exists,
        # then stop the worker thread before Qt tears the app down.
        self.viewport.cleanup()
        self.controller.shutdown()
        super().closeEvent(event)
