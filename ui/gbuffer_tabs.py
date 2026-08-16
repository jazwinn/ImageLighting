"""G-buffer inspector tabs.

Four tabs mirror the static passes the AI worker produced; the fifth pulls
a live capture out of the GL renderer so the shading mask and specular
response can be inspected while the lights are being moved.
"""

from __future__ import annotations

import numpy as np

from core.gbuffer import GBuffer
from core.qt_compat import QtCore, QtWidgets, Signal
from core.scene import ViewMode
from pipeline.delighting_engine import colorize_depth
from pipeline.normal_engine import encode_normal_for_display
from ui.widgets import ImageCanvas, numpy_to_pixmap


class _PassTab(QtWidgets.QWidget):
    """An image canvas plus a one-line caption describing the pass."""

    def __init__(self, caption: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 6)
        layout.setSpacing(6)

        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas, 1)

        self.caption = QtWidgets.QLabel(caption)
        self.caption.setObjectName("metaLabel")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)

    def set_image(self, image: np.ndarray | None) -> None:
        self.canvas.set_image(None if image is None else numpy_to_pixmap(image))

    def set_caption(self, text: str) -> None:
        self.caption.setText(text)


class LivePassTab(_PassTab):
    """Live shading / specular capture, refreshed from the GL renderer."""

    #: 6 Hz is fast enough to feel live while leaving the viewport the GPU.
    REFRESH_MS = 160

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Live capture from the deferred shader.", parent)
        self._capture = None

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)

        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItem("Shading mask (ambient + diffuse)", int(ViewMode.SHADING))
        self.mode_box.addItem("Specular highlights", int(ViewMode.SPECULAR))
        controls.addWidget(self.mode_box, 1)

        self.auto_check = QtWidgets.QCheckBox("Live")
        self.auto_check.setChecked(True)
        self.auto_check.setToolTip("Continuously re-capture while lights change")
        controls.addWidget(self.auto_check)

        self.refresh_button = QtWidgets.QPushButton("Refresh")
        controls.addWidget(self.refresh_button)

        self.layout().insertLayout(1, controls)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self.REFRESH_MS)
        self._timer.timeout.connect(self.refresh)

        self.refresh_button.clicked.connect(self.refresh)
        self.mode_box.currentIndexChanged.connect(lambda _i: self.refresh())
        self.auto_check.toggled.connect(self._on_auto_toggled)

    def set_capture_callback(self, callback) -> None:
        """``callback(view_mode) -> ndarray`` renders one offscreen pass."""
        self._capture = callback

    def _on_auto_toggled(self, enabled: bool) -> None:
        if enabled and self.isVisible():
            self._timer.start()
        else:
            self._timer.stop()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.auto_check.isChecked():
            self._timer.start()
        self.refresh()

    def hideEvent(self, event) -> None:
        # Stop burning GPU on a tab nobody is looking at.
        self._timer.stop()
        super().hideEvent(event)

    def refresh(self) -> None:
        if self._capture is None:
            return
        mode = ViewMode(self.mode_box.currentData())
        try:
            image = self._capture(mode)
        except Exception:
            # A capture failure (no buffer yet, context not current) must not
            # take the UI down; the next tick will try again.
            return
        if image is not None:
            self.set_image(image)


class GBufferTabs(QtWidgets.QTabWidget):
    """The five-pass inspector."""

    exportPassRequested = Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDocumentMode(True)
        self.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)

        self.original_tab = _PassTab("Source image as loaded, before any processing.")
        self.depth_tab = _PassTab("Estimated depth, Turbo-mapped — warm is near.")
        self.normal_tab = _PassTab("Surface normals, RGB = (Nx, Ny, Nz).")
        self.albedo_tab = _PassTab("De-lit albedo: flat base colour with shading divided out.")
        self.live_tab = LivePassTab()

        self.addTab(self.original_tab, "Original")
        self.addTab(self.depth_tab, "Depth")
        self.addTab(self.normal_tab, "Normals")
        self.addTab(self.albedo_tab, "Albedo")
        self.addTab(self.live_tab, "Shading")

        self._buffer: GBuffer | None = None

    def set_gbuffer(self, buffer: GBuffer | None) -> None:
        self._buffer = buffer
        if buffer is None:
            for tab in (self.original_tab, self.depth_tab, self.normal_tab, self.albedo_tab):
                tab.set_image(None)
            self.live_tab.set_image(None)
            return

        self.original_tab.set_image(buffer.original)
        self.depth_tab.set_image(colorize_depth(buffer.depth))
        self.normal_tab.set_image(encode_normal_for_display(buffer.normal))
        self.albedo_tab.set_image(buffer.albedo)

        near, far = buffer.depth_range()
        units = "m" if buffer.meta.get("depth_is_metric") else " scene units"
        self.depth_tab.set_caption(
            f"Estimated depth via {buffer.meta.get('depth_backend', 'unknown')} — "
            f"range {near:.2f} to {far:.2f}{units}."
        )
        self.normal_tab.set_caption(
            f"Surface normals via {buffer.meta.get('normal_backend', 'unknown')}. "
            "Blue faces the camera."
        )
        self.albedo_tab.set_caption(
            f"De-lit albedo via {buffer.meta.get('albedo_backend', 'unknown')}. "
            "Relighting this instead of the original avoids double shadows."
        )
        self.original_tab.set_caption(
            f"{buffer.meta.get('resolution', '')} — "
            f"{buffer.source_path or 'in-memory image'}"
        )
        self.live_tab.refresh()

    def set_capture_callback(self, callback) -> None:
        self.live_tab.set_capture_callback(callback)
