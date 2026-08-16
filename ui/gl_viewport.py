"""Interactive OpenGL viewport.

A ``QOpenGLWidget`` that hands its live GL context to ModernGL and drives
:class:`RelightRenderer`.  All AI work happens elsewhere; everything this
widget does per frame is upload a handful of uniforms and issue three draw
calls, which is what keeps light dragging smooth.

Interaction model
-----------------
========================  ==================================================
Left drag on a gizmo      Move that light in the image plane (X/Y at fixed Z)
Left click on the image   Move the selected light to the cursor
Left drag + Ctrl/Alt      Push the light toward or away from the camera (Z)
Double click              Snap the light just in front of the surface hit
Middle drag / Space drag  Pan the view
Wheel                     Zoom;  Ctrl+Wheel adjusts the selected light's Z
========================  ==================================================
"""

from __future__ import annotations

import time

import numpy as np

from core.gbuffer import GBuffer
from core.qt_compat import Qt, QtCore, QtGui, QOpenGLWidget, Signal
from core.scene import Scene
from render.relight_renderer import ImageTransform, RelightRenderer
from ui.theme import VIEWPORT_CLEAR

#: Pixel radius within which a click counts as grabbing a gizmo.
GIZMO_PICK_RADIUS = 18.0


class GLViewport(QOpenGLWidget):
    """Renders the relit scene and turns mouse input into light edits."""

    lightSelected = Signal(int)
    lightMoved = Signal(int)
    cursorMoved = Signal(str)
    fpsUpdated = Signal(float)
    glInitialised = Signal(str)
    glFailed = Signal(str)

    def __init__(self, scene: Scene, parent=None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.buffer: GBuffer | None = None
        self.renderer: RelightRenderer | None = None
        self.ctx = None

        self._pending_buffer: GBuffer | None = None
        self._screen_fbo = None
        self._screen_fbo_glo = -1

        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self._transform = ImageTransform()

        self._drag_mode: str | None = None
        self._drag_light = -1
        self._last_pos = QtCore.QPointF()
        self._press_pos = QtCore.QPointF()

        self._frame_times: list[float] = []
        self._last_frame = time.perf_counter()

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(360, 260)
        self.setCursor(Qt.CursorShape.CrossCursor)

        # A steady repaint keeps the FPS readout meaningful and makes light
        # drags feel continuous rather than event-quantised.
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(8)  # request frames faster than vsync
        self._timer.timeout.connect(self._tick)
        self.continuous = True

    # -- GL lifecycle ------------------------------------------------------
    def initializeGL(self) -> None:
        import moderngl

        try:
            self.ctx = moderngl.create_context()
            self.renderer = RelightRenderer(self.ctx)
        except Exception as exc:
            self.ctx = None
            self.renderer = None
            self.glFailed.emit(str(exc))
            return

        info = f"{self.ctx.info.get('GL_RENDERER', '?')} · GL {self.ctx.version_code / 100:.1f}"
        self.glInitialised.emit(info)

        if self._pending_buffer is not None:
            self.set_gbuffer(self._pending_buffer)
            self._pending_buffer = None

        if self.continuous:
            self._timer.start()

    def _screen_framebuffer(self):
        """Wrap Qt's default FBO, re-detecting only when Qt swaps it out.

        Qt recreates its backing FBO on resize and on some DPI changes;
        re-detecting every frame would allocate a wrapper per frame.
        """
        glo = self.defaultFramebufferObject()
        if self._screen_fbo is None or glo != self._screen_fbo_glo:
            self._screen_fbo = self.ctx.detect_framebuffer(glo)
            self._screen_fbo_glo = glo
        return self._screen_fbo

    def paintGL(self) -> None:
        if self.ctx is None or self.renderer is None:
            return

        now = time.perf_counter()
        self._record_frame(now)

        ratio = self.devicePixelRatioF()
        width = max(1, int(self.width() * ratio))
        height = max(1, int(self.height() * ratio))

        fbo = self._screen_framebuffer()
        fbo.use()
        fbo.viewport = (0, 0, width, height)

        if self.buffer is None:
            self.ctx.clear(*VIEWPORT_CLEAR, 1.0)
            return

        self._transform = ImageTransform.fit(
            width, height, self.buffer.width, self.buffer.height, self._zoom, tuple(self._pan)
        )
        self.renderer.render(fbo, self._transform, self.scene, background=VIEWPORT_CLEAR)

    def resizeGL(self, width: int, height: int) -> None:
        # The transform is recomputed from the live widget size every frame,
        # so there is nothing to cache here; the override exists to stop the
        # base class from installing its own viewport assumptions.
        pass

    def _tick(self) -> None:
        if self.isVisible():
            self.update()

    def set_continuous(self, enabled: bool) -> None:
        self.continuous = bool(enabled)
        if enabled:
            self._timer.start()
        else:
            self._timer.stop()
            self.update()

    def _record_frame(self, now: float) -> None:
        delta = now - self._last_frame
        self._last_frame = now
        if delta <= 0.0:
            return
        self._frame_times.append(delta)
        if len(self._frame_times) >= 30:
            average = sum(self._frame_times) / len(self._frame_times)
            self._frame_times.clear()
            if average > 0.0:
                self.fpsUpdated.emit(1.0 / average)

    def cleanup(self) -> None:
        """Release GL objects while the context is still current."""
        self._timer.stop()
        if self.renderer is None:
            return
        self.makeCurrent()
        try:
            self.renderer.release()
        finally:
            self.renderer = None
            self._screen_fbo = None
            self._screen_fbo_glo = -1
            self.doneCurrent()

    # -- data --------------------------------------------------------------
    def set_gbuffer(self, buffer: GBuffer, reset_view: bool = True) -> None:
        """Swap in a new decomposition, replacing the resident textures.

        ``reset_view`` is off when only the albedo was recomputed: throwing
        away the user's zoom and pan because they nudged a de-lighting
        slider would be maddening.
        """
        self.buffer = buffer
        if self.renderer is None:
            # initializeGL has not run yet; stash it until the context exists.
            self._pending_buffer = buffer
            return
        self.makeCurrent()
        try:
            self.renderer.upload_gbuffer(buffer)
        finally:
            self.doneCurrent()
        if reset_view:
            self.reset_view()
        self.update()

    def clear_gbuffer(self) -> None:
        self.buffer = None
        self._pending_buffer = None
        if self.renderer is not None:
            self.makeCurrent()
            try:
                self.renderer.release_textures()
            finally:
                self.doneCurrent()
        self.update()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self.update()

    def render_to_array(self) -> np.ndarray:
        """Full-resolution offscreen render, for export."""
        if self.renderer is None or self.buffer is None:
            raise RuntimeError("Nothing to render yet")
        self.makeCurrent()
        try:
            return self.renderer.render_to_array(self.scene)
        finally:
            # Qt's FBO stops being bound after the offscreen pass; force a
            # redraw so the widget does not show a stale frame.
            self.doneCurrent()
            self.update()

    # -- coordinate helpers ------------------------------------------------
    def _widget_to_uv(self, pos: QtCore.QPointF) -> tuple[float, float]:
        ratio = self.devicePixelRatioF()
        return self._transform.widget_to_uv(pos.x() * ratio, pos.y() * ratio)

    def _light_screen_pos(self, index: int) -> tuple[float, float] | None:
        """Widget-space pixel position of a light's gizmo, or None if behind."""
        if self.buffer is None or not 0 <= index < len(self.scene.lights):
            return None
        light = self.scene.lights[index]
        p = light.position_np
        if p[2] <= 1e-3:
            return None
        intr = self.buffer.intrinsics
        px = p[0] * intr.fx / p[2] + intr.cx
        py = -p[1] * intr.fy / p[2] + intr.cy
        u = px / self.buffer.width
        v = py / self.buffer.height
        x, y = self._transform.uv_to_widget(u, v)
        ratio = max(self.devicePixelRatioF(), 1e-6)
        return x / ratio, y / ratio

    def _pick_light(self, pos: QtCore.QPointF) -> int:
        """Nearest gizmo within the pick radius, preferring the selected one."""
        best_index = -1
        best_distance = GIZMO_PICK_RADIUS
        for index in range(len(self.scene.lights)):
            screen = self._light_screen_pos(index)
            if screen is None:
                continue
            distance = np.hypot(screen[0] - pos.x(), screen[1] - pos.y())
            # Bias toward the active light so overlapping gizmos stay stable
            # while dragging one across another.
            if index == self.scene.active_index:
                distance -= 4.0
            if distance < best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def _uv_to_camera(self, u: float, v: float, z: float) -> np.ndarray:
        """Unproject an image UV at a chosen depth into camera space."""
        assert self.buffer is not None
        intr = self.buffer.intrinsics
        px = u * self.buffer.width
        py = v * self.buffer.height
        x = (px - intr.cx) * z / intr.fx
        y = -(py - intr.cy) * z / intr.fy
        return np.array([x, y, z], dtype=np.float32)

    def _sample_surface(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Camera-space position and normal of the surface under a UV."""
        if self.buffer is None or not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return None
        col = int(min(max(u * self.buffer.width, 0), self.buffer.width - 1))
        row = int(min(max(v * self.buffer.height, 0), self.buffer.height - 1))
        depth = float(self.buffer.depth[row, col]) * self.scene.render.depth_scale
        position = self._uv_to_camera(u, v, depth)
        normal = self.buffer.normal[row, col].astype(np.float32)
        return position, normal

    # -- mouse -------------------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.buffer is None:
            return
        pos = event.position()
        self._press_pos = pos
        self._last_pos = pos
        button = event.button()
        modifiers = event.modifiers()

        if button == Qt.MouseButton.MiddleButton or (
            button == Qt.MouseButton.LeftButton
            and modifiers & Qt.KeyboardModifier.ShiftModifier
        ):
            self._drag_mode = "pan"
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if button == Qt.MouseButton.LeftButton:
            picked = self._pick_light(pos)
            if picked >= 0 and picked != self.scene.active_index:
                self.scene.active_index = picked
                self.lightSelected.emit(picked)

            if self.scene.active_light() is None:
                return

            depth_drag = bool(
                modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)
            )
            self._drag_mode = "light_z" if depth_drag else "light_xy"
            self._drag_light = self.scene.active_index
            self.setCursor(
                Qt.CursorShape.SizeVerCursor if depth_drag else Qt.CursorShape.SizeAllCursor
            )
            if not depth_drag and picked < 0:
                # Clicking empty canvas teleports the active light there.
                self._move_light_to(pos)
            return

        if button == Qt.MouseButton.RightButton:
            picked = self._pick_light(pos)
            if picked >= 0:
                self.scene.active_index = picked
                self.lightSelected.emit(picked)
                self._drag_mode = "light_z"
                self._drag_light = picked
                self.setCursor(Qt.CursorShape.SizeVerCursor)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.buffer is None:
            return
        pos = event.position()
        delta = pos - self._last_pos
        self._last_pos = pos

        if self._drag_mode == "pan":
            ratio = self.devicePixelRatioF()
            width = max(1.0, self.width() * ratio)
            height = max(1.0, self.height() * ratio)
            self._pan[0] += (delta.x() * ratio) * 2.0 / width
            self._pan[1] -= (delta.y() * ratio) * 2.0 / height
            self.update()
            return

        if self._drag_mode == "light_xy":
            self._move_light_to(pos)
            return

        if self._drag_mode == "light_z":
            light = self.scene.active_light()
            if light is not None:
                span = max(self.buffer.scene_radius(), 0.05)
                # Dragging up moves the light toward the camera.
                light.position[2] = float(
                    max(0.02, light.position[2] + delta.y() * span * 0.004)
                )
                self.lightMoved.emit(self.scene.active_index)
                self.update()
            return

        self._emit_cursor_info(pos)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._drag_mode = None
        self._drag_light = -1
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Snap the active light just off the surface under the cursor."""
        if self.buffer is None or event.button() != Qt.MouseButton.LeftButton:
            return
        light = self.scene.active_light()
        if light is None:
            return
        u, v = self._widget_to_uv(event.position())
        hit = self._sample_surface(u, v)
        if hit is None:
            return
        position, normal = hit
        standoff = max(self.buffer.scene_radius(), 0.05) * 0.35
        # Normals point toward the camera (-Z), so this lifts the light off
        # the surface rather than burying it inside the geometry.
        target = position + normal * standoff
        light.position = [float(v) for v in target]
        self.lightMoved.emit(self.scene.active_index)
        self.update()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        steps = event.angleDelta().y() / 120.0
        if steps == 0.0:
            return

        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            light = self.scene.active_light()
            if light is not None and self.buffer is not None:
                span = max(self.buffer.scene_radius(), 0.05)
                light.position[2] = float(max(0.02, light.position[2] - steps * span * 0.08))
                self.lightMoved.emit(self.scene.active_index)
                self.update()
            return

        self._zoom = float(np.clip(self._zoom * (1.12 ** steps), 0.15, 12.0))
        self.update()

    def _move_light_to(self, pos: QtCore.QPointF) -> None:
        light = self.scene.active_light()
        if light is None or self.buffer is None:
            return
        u, v = self._widget_to_uv(pos)
        # Keep the light's current distance from the camera: dragging across
        # the frame should slide it, not fling it in depth.
        z = max(float(light.position[2]), 0.02)
        target = self._uv_to_camera(u, v, z)
        light.position = [float(c) for c in target]
        self.lightMoved.emit(self.scene.active_index)
        self.update()

    def _emit_cursor_info(self, pos: QtCore.QPointF) -> None:
        if self.buffer is None:
            return
        u, v = self._widget_to_uv(pos)
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            self.cursorMoved.emit("")
            return
        col = int(u * self.buffer.width)
        row = int(v * self.buffer.height)
        col = min(max(col, 0), self.buffer.width - 1)
        row = min(max(row, 0), self.buffer.height - 1)
        depth = float(self.buffer.depth[row, col])
        normal = self.buffer.normal[row, col]
        self.cursorMoved.emit(
            f"px ({col}, {row})   Z {depth:.3f}   "
            f"N ({normal[0]:+.2f}, {normal[1]:+.2f}, {normal[2]:+.2f})"
        )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_F:
            self.reset_view()
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._zoom = float(min(self._zoom * 1.2, 12.0))
            self.update()
        elif key == Qt.Key.Key_Minus:
            self._zoom = float(max(self._zoom / 1.2, 0.15))
            self.update()
        else:
            super().keyPressEvent(event)
