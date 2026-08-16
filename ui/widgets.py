"""Small reusable controls for the inspector panels."""

from __future__ import annotations

from core.qt_compat import Qt, QtGui, QtWidgets, Signal


class SliderRow(QtWidgets.QWidget):
    """A labelled float slider with a live numeric readout.

    Qt sliders are integer-only, so the value is kept in fixed-point with
    ``steps`` subdivisions across the range.
    """

    valueChanged = Signal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        *,
        decimals: int = 2,
        suffix: str = "",
        steps: int = 1000,
        tooltip: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._min = float(minimum)
        self._max = float(maximum)
        self._steps = int(steps)
        self._decimals = int(decimals)
        self._suffix = suffix
        self._emitting = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(1)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(6)
        self.label = QtWidgets.QLabel(label)
        self.value_label = QtWidgets.QLabel()
        self.value_label.setObjectName("valueLabel")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self.label)
        header.addStretch(1)
        header.addWidget(self.value_label)
        layout.addLayout(header)

        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._steps)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(max(1, self._steps // 20))
        layout.addWidget(self.slider)

        if tooltip:
            self.setToolTip(tooltip)

        self.slider.valueChanged.connect(self._on_slider)
        self.set_value(value)

    def _to_slider(self, value: float) -> int:
        span = max(self._max - self._min, 1e-9)
        t = (float(value) - self._min) / span
        return int(round(min(max(t, 0.0), 1.0) * self._steps))

    def _from_slider(self, position: int) -> float:
        return self._min + (position / float(self._steps)) * (self._max - self._min)

    def _on_slider(self, position: int) -> None:
        value = self._from_slider(position)
        self._refresh_label(value)
        if not self._emitting:
            self.valueChanged.emit(value)

    def _refresh_label(self, value: float) -> None:
        self.value_label.setText(f"{value:.{self._decimals}f}{self._suffix}")

    def value(self) -> float:
        return self._from_slider(self.slider.value())

    def set_value(self, value: float) -> None:
        """Set without emitting, so syncing UI from state cannot feed back."""
        self._emitting = True
        self.slider.setValue(self._to_slider(value))
        self._refresh_label(float(value))
        self._emitting = False


class ColorButton(QtWidgets.QPushButton):
    """A swatch that opens a colour picker; value is RGB floats in [0, 1]."""

    colorChanged = Signal(object)  # tuple[float, float, float]

    def __init__(self, color=(1.0, 1.0, 1.0), parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = tuple(float(c) for c in color)
        self.setFixedHeight(24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self._pick)
        self._refresh()

    def _refresh(self) -> None:
        r, g, b = (int(round(min(max(c, 0.0), 1.0) * 255)) for c in self._color)
        # Flip the caption to dark on pale swatches so it stays readable.
        text_color = "#101014" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#f0f0f4"
        self.setStyleSheet(
            f"QPushButton {{ background-color: rgb({r},{g},{b}); color: {text_color};"
            f" border: 1px solid #3a3a45; border-radius: 4px; font-size: 11px; }}"
            f"QPushButton:hover {{ border-color: #f0a04b; }}"
        )
        self.setText(f"#{r:02X}{g:02X}{b:02X}")

    def _pick(self) -> None:
        r, g, b = (int(round(min(max(c, 0.0), 1.0) * 255)) for c in self._color)
        chosen = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(r, g, b), self, "Light colour"
        )
        if chosen.isValid():
            self.set_color((chosen.redF(), chosen.greenF(), chosen.blueF()))
            self.colorChanged.emit(self._color)

    def color(self) -> tuple[float, float, float]:
        return self._color

    def set_color(self, color) -> None:
        self._color = tuple(float(c) for c in color)
        self._refresh()


class Section(QtWidgets.QGroupBox):
    """Group box with a vertical body layout and convenience adders."""

    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(title.upper(), parent)
        self.body = QtWidgets.QVBoxLayout(self)
        self.body.setContentsMargins(10, 6, 10, 8)
        self.body.setSpacing(4)

    def add(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        self.body.addWidget(widget)
        return widget

    def add_row(self, label: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 2, 0, 2)
        caption = QtWidgets.QLabel(label)
        caption.setMinimumWidth(74)
        row.addWidget(caption)
        row.addWidget(widget, 1)
        self.body.addLayout(row)
        return widget

    def add_hint(self, text: str) -> QtWidgets.QLabel:
        hint = QtWidgets.QLabel(text)
        hint.setObjectName("sectionHint")
        hint.setWordWrap(True)
        self.body.addWidget(hint)
        return hint


class Vector3Row(QtWidgets.QWidget):
    """Three linked spin boxes for editing a position or direction."""

    valueChanged = Signal(object)  # tuple[float, float, float]

    def __init__(
        self,
        value=(0.0, 0.0, 0.0),
        *,
        minimum: float = -100.0,
        maximum: float = 100.0,
        step: float = 0.05,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._boxes: list[QtWidgets.QDoubleSpinBox] = []
        self._emitting = False
        for axis, initial in zip("XYZ", value):
            caption = QtWidgets.QLabel(axis)
            caption.setObjectName("metaLabel")
            caption.setFixedWidth(9)
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(minimum, maximum)
            box.setSingleStep(step)
            box.setDecimals(3)
            # Three of these plus their axis captions have to share one row;
            # the default size hint is far too generous for that.
            box.setMinimumWidth(56)
            box.setValue(float(initial))
            box.setKeyboardTracking(False)
            box.valueChanged.connect(self._on_changed)
            layout.addWidget(caption)
            layout.addWidget(box, 1)
            self._boxes.append(box)

    def _on_changed(self, _value: float) -> None:
        if not self._emitting:
            self.valueChanged.emit(self.value())

    def value(self) -> tuple[float, float, float]:
        return tuple(box.value() for box in self._boxes)  # type: ignore[return-value]

    def set_value(self, value) -> None:
        self._emitting = True
        for box, component in zip(self._boxes, value):
            box.setValue(float(component))
        self._emitting = False


class ImageCanvas(QtWidgets.QLabel):
    """Zoomable, pannable viewer for a static G-buffer pass.

    Scales with smooth transformation on zoom-out so downsampled previews
    of large buffers do not alias into noise.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(160, 120)
        self.setStyleSheet("background-color: #0b0b0e; border-radius: 5px;")
        self.setText("No image loaded")
        self._pixmap: QtGui.QPixmap | None = None

    def set_image(self, pixmap: QtGui.QPixmap | None) -> None:
        self._pixmap = pixmap
        if pixmap is None:
            self.setText("Not available")
        self._rescale()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


def numpy_to_pixmap(image) -> QtGui.QPixmap:
    """Convert an RGB float32 [0, 1] or uint8 array to a QPixmap.

    The QImage is copied because it would otherwise alias the numpy buffer,
    which Python is free to collect the moment this function returns.
    """
    import numpy as np

    if image.dtype != np.uint8:
        data = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    else:
        data = image
    data = np.ascontiguousarray(data)

    if data.ndim == 2:
        data = np.repeat(data[:, :, None], 3, axis=2)

    height, width, _ = data.shape
    qimage = QtGui.QImage(
        data.data, width, height, 3 * width, QtGui.QImage.Format.Format_RGB888
    )
    return QtGui.QPixmap.fromImage(qimage.copy())


class ElidedLabel(QtWidgets.QLabel):
    """Label that middle-elides instead of forcing its parent to grow."""

    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = text
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred
        )
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        self._full_text = text
        self._update_elided()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_elided()

    def _update_elided(self) -> None:
        metrics = QtGui.QFontMetrics(self.font())
        elided = metrics.elidedText(
            self._full_text, Qt.TextElideMode.ElideMiddle, max(self.width() - 4, 20)
        )
        super().setText(elided)
        if elided != self._full_text:
            self.setToolTip(self._full_text)
