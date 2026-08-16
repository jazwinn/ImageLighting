"""Right-hand inspector: light manager, light/material properties, shadows.

Every control writes straight into the shared :class:`Scene` object and then
emits :attr:`Inspector.sceneChanged`, which the main window turns into a
viewport repaint.  Nothing here touches OpenGL or the AI pipeline, with one
exception: the de-lighting panel emits a request for the worker to redo the
intrinsic decomposition, because those settings change the G-buffer itself
rather than just the shading.

The ``_syncing`` guard matters: :meth:`Inspector.sync_from_scene` pushes
state into the widgets, and without it every programmatic update would echo
back as a user edit.
"""

from __future__ import annotations

from core.qt_compat import Qt, QtWidgets, Signal
from core.scene import (
    MAX_LIGHTS,
    Light,
    LightType,
    Scene,
    ShadingSource,
    SpecularModel,
    ViewMode,
)
from pipeline.delighting_engine import DelightSettings
from pipeline.intrinsic_backend import is_available as intrinsic_available
from pipeline.stablenormal_backend import is_available as stablenormal_available
from ui.widgets import ColorButton, Section, SliderRow, Vector3Row


class Inspector(QtWidgets.QWidget):
    """Scrollable stack of control sections."""

    sceneChanged = Signal()
    lightsChanged = Signal()          # add/remove/select: list must be rebuilt
    activeLightChanged = Signal(int)
    delightRequested = Signal(object)  # DelightSettings
    viewModeChanged = Signal(object)   # ViewMode
    neuralNormalsToggled = Signal(bool)  # needs a full re-import

    def __init__(self, scene: Scene, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self._syncing = False

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        container = QtWidgets.QWidget()
        self.column = QtWidgets.QVBoxLayout(container)
        self.column.setContentsMargins(10, 10, 10, 14)
        self.column.setSpacing(10)
        scroll.setWidget(container)

        self._build_light_manager()
        self._build_light_properties()
        self._build_material()
        self._build_shadows()
        self._build_geometry()
        self._build_delighting()
        self._build_render()
        self.column.addStretch(1)

        # Wide enough for the three-spin-box vector rows plus the scrollbar;
        # below this the Z field and the Remove button get clipped.
        self.setMinimumWidth(408)
        self.sync_from_scene()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_light_manager(self) -> None:
        section = Section("Light manager")

        self.light_list = QtWidgets.QListWidget()
        self.light_list.setMaximumHeight(148)
        self.light_list.currentRowChanged.connect(self._on_light_selected)
        self.light_list.itemChanged.connect(self._on_light_item_changed)
        section.add(self.light_list)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(5)

        self.add_button = QtWidgets.QToolButton()
        self.add_button.setText("Add  ▾")
        self.add_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        add_menu = QtWidgets.QMenu(self.add_button)
        for kind in (LightType.POINT, LightType.SPOT, LightType.DIRECTIONAL):
            action = add_menu.addAction(kind.label)
            action.triggered.connect(lambda _checked=False, k=kind: self.add_light(k))
        self.add_button.setMenu(add_menu)
        buttons.addWidget(self.add_button)

        self.duplicate_button = QtWidgets.QPushButton("Duplicate")
        self.duplicate_button.clicked.connect(self.duplicate_light)
        buttons.addWidget(self.duplicate_button)

        self.remove_button = QtWidgets.QPushButton("Remove")
        self.remove_button.setObjectName("danger")
        self.remove_button.clicked.connect(self.remove_light)
        buttons.addWidget(self.remove_button)

        section.body.addLayout(buttons)
        section.add_hint(
            "Drag a gizmo in the viewport to move a light. "
            "Ctrl-drag or Ctrl-wheel changes its distance from the camera."
        )
        self.column.addWidget(section)

    def _build_light_properties(self) -> None:
        self.light_section = Section("Light properties")
        section = self.light_section

        self.type_combo = QtWidgets.QComboBox()
        for kind in (LightType.POINT, LightType.SPOT, LightType.DIRECTIONAL):
            self.type_combo.addItem(kind.label, int(kind))
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        section.add_row("Type", self.type_combo)

        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.editingFinished.connect(self._on_name_changed)
        section.add_row("Name", self.name_edit)

        self.position_row = Vector3Row()
        self.position_row.valueChanged.connect(self._on_position_changed)
        section.add_row("Position", self.position_row)

        self.direction_row = Vector3Row(minimum=-1.0, maximum=1.0, step=0.05)
        self.direction_row.valueChanged.connect(self._on_direction_changed)
        section.add_row("Aim", self.direction_row)

        self.color_button = ColorButton()
        self.color_button.colorChanged.connect(self._on_color_changed)
        section.add_row("Colour", self.color_button)

        # The range has to reach high: a light placed a scene-radius away
        # needs intensity proportional to distance squared just to register.
        self.intensity_slider = SliderRow(
            "Intensity", 0.0, 100.0, 3.0, decimals=2, steps=2000,
            tooltip="Radiant power before attenuation",
        )
        self.intensity_slider.valueChanged.connect(
            lambda v: self._set_light_attr("intensity", v)
        )
        section.add(self.intensity_slider)

        self.enabled_check = QtWidgets.QCheckBox("Enabled")
        self.enabled_check.toggled.connect(lambda v: self._set_light_attr("enabled", v, True))
        self.shadow_check = QtWidgets.QCheckBox("Casts shadow")
        self.shadow_check.toggled.connect(lambda v: self._set_light_attr("casts_shadow", v))
        toggles = QtWidgets.QHBoxLayout()
        toggles.addWidget(self.enabled_check)
        toggles.addWidget(self.shadow_check)
        toggles.addStretch(1)
        section.body.addLayout(toggles)

        # --- attenuation ---
        self.atten_label = QtWidgets.QLabel("Attenuation  1 / (kc + kl·d + kq·d²)")
        self.atten_label.setObjectName("sectionHint")
        section.add(self.atten_label)

        self.atten_const = SliderRow("Constant kc", 0.0, 4.0, 1.0, decimals=3)
        self.atten_const.valueChanged.connect(
            lambda v: self._set_light_attr("attenuation_constant", v)
        )
        section.add(self.atten_const)

        self.atten_linear = SliderRow("Linear kl", 0.0, 2.0, 0.09, decimals=3)
        self.atten_linear.valueChanged.connect(
            lambda v: self._set_light_attr("attenuation_linear", v)
        )
        section.add(self.atten_linear)

        self.atten_quad = SliderRow("Quadratic kq", 0.0, 4.0, 0.032, decimals=3)
        self.atten_quad.valueChanged.connect(
            lambda v: self._set_light_attr("attenuation_quadratic", v)
        )
        section.add(self.atten_quad)

        # --- spot cone ---
        self.spot_inner = SliderRow("Cone inner", 0.0, 89.0, 18.0, decimals=1, suffix="°")
        self.spot_inner.valueChanged.connect(self._on_spot_inner)
        section.add(self.spot_inner)

        self.spot_outer = SliderRow("Cone outer", 1.0, 90.0, 30.0, decimals=1, suffix="°")
        self.spot_outer.valueChanged.connect(self._on_spot_outer)
        section.add(self.spot_outer)

        self.column.addWidget(section)

    def _build_material(self) -> None:
        section = Section("Material")

        self.ambient_slider = SliderRow("Ambient kₐ", 0.0, 1.5, 0.18)
        self.ambient_slider.valueChanged.connect(lambda v: self._set_material("ambient", v))
        section.add(self.ambient_slider)

        self.ambient_color = ColorButton((0.42, 0.48, 0.6))
        self.ambient_color.colorChanged.connect(self._on_ambient_color)
        section.add_row("Ambient hue", self.ambient_color)

        self.diffuse_slider = SliderRow("Diffuse k_d", 0.0, 3.0, 1.0)
        self.diffuse_slider.valueChanged.connect(lambda v: self._set_material("diffuse", v))
        section.add(self.diffuse_slider)

        self.spec_model_combo = QtWidgets.QComboBox()
        self.spec_model_combo.addItem("GGX (roughness)", int(SpecularModel.GGX))
        self.spec_model_combo.addItem("Blinn-Phong (shininess)", int(SpecularModel.BLINN_PHONG))
        self.spec_model_combo.currentIndexChanged.connect(self._on_spec_model)
        section.add_row("Specular", self.spec_model_combo)

        self.specular_slider = SliderRow("Specular k_s", 0.0, 3.0, 0.35)
        self.specular_slider.valueChanged.connect(lambda v: self._set_material("specular", v))
        section.add(self.specular_slider)

        self.roughness_slider = SliderRow("Roughness", 0.03, 1.0, 0.45)
        self.roughness_slider.valueChanged.connect(lambda v: self._set_material("roughness", v))
        section.add(self.roughness_slider)

        self.shininess_slider = SliderRow("Shininess", 1.0, 256.0, 48.0, decimals=0)
        self.shininess_slider.valueChanged.connect(lambda v: self._set_material("shininess", v))
        section.add(self.shininess_slider)

        self.metallic_slider = SliderRow("Metallic", 0.0, 1.0, 0.0)
        self.metallic_slider.valueChanged.connect(lambda v: self._set_material("metallic", v))
        section.add(self.metallic_slider)

        self.normal_strength_slider = SliderRow(
            "Normal relief", 0.0, 3.0, 1.0,
            tooltip="Scales the tangential part of the normals: below 1 flattens the "
                    "surface, above 1 exaggerates relief.",
        )
        self.normal_strength_slider.valueChanged.connect(
            lambda v: self._set_material("normal_strength", v)
        )
        section.add(self.normal_strength_slider)

        self.base_light_slider = SliderRow(
            "Base bleed", 0.0, 1.0, 0.12,
            tooltip="Adds back a fraction of the unlit base colour so a scene with "
                    "no lights is not pitch black.",
        )
        self.base_light_slider.valueChanged.connect(lambda v: self._set_material("base_light", v))
        section.add(self.base_light_slider)

        self.exposure_slider = SliderRow("Exposure", 0.0, 4.0, 1.0)
        self.exposure_slider.valueChanged.connect(lambda v: self._set_material("exposure", v))
        section.add(self.exposure_slider)

        self.column.addWidget(section)

    def _build_shadows(self) -> None:
        section = Section("Screen-space shadows")

        self.shadow_enabled = QtWidgets.QCheckBox("Enable raymarched cast shadows")
        self.shadow_enabled.toggled.connect(lambda v: self._set_shadow("enabled", v))
        section.add(self.shadow_enabled)

        self.shadow_steps = SliderRow(
            "Steps", 4, 64, 24, decimals=0, steps=60,
            tooltip="Samples per ray. More steps catch thinner occluders at a linear cost.",
        )
        self.shadow_steps.valueChanged.connect(lambda v: self._set_shadow("steps", int(round(v))))
        section.add(self.shadow_steps)

        self.shadow_rays = SliderRow(
            "Rays", 1, 4, 2, decimals=0, steps=3,
            tooltip="Rays per pixel, each starting at a different phase within a step. "
                    "The effective cure for stippled penumbras — raising Steps instead "
                    "barely helps. Costs one full march per ray.",
        )
        self.shadow_rays.valueChanged.connect(lambda v: self._set_shadow("rays", int(round(v))))
        section.add(self.shadow_rays)

        self.shadow_distance = SliderRow("Max distance", 0.1, 12.0, 2.5)
        self.shadow_distance.valueChanged.connect(lambda v: self._set_shadow("max_distance", v))
        section.add(self.shadow_distance)

        self.shadow_softness = SliderRow(
            "Penumbra", 0.0, 1.0, 0.35,
            tooltip="Fades shadows cast by distant occluders.",
        )
        self.shadow_softness.valueChanged.connect(lambda v: self._set_shadow("softness", v))
        section.add(self.shadow_softness)

        self.shadow_bias = SliderRow(
            "Bias", 0.0, 0.12, 0.012, decimals=4,
            tooltip="Raise until surface acne disappears; too high detaches contact shadows.",
        )
        self.shadow_bias.valueChanged.connect(lambda v: self._set_shadow("bias", v))
        section.add(self.shadow_bias)

        self.shadow_strength = SliderRow("Opacity", 0.0, 1.0, 0.85)
        self.shadow_strength.valueChanged.connect(lambda v: self._set_shadow("strength", v))
        section.add(self.shadow_strength)

        self.column.addWidget(section)

    def _build_geometry(self) -> None:
        section = Section("Geometry")

        self.neural_normals_toggle = QtWidgets.QCheckBox(
            "Neural surface normals (StableNormal)"
        )
        ready = stablenormal_available()
        self.neural_normals_toggle.setChecked(ready)
        self.neural_normals_toggle.setEnabled(ready)
        self.neural_normals_toggle.setToolTip(
            "Predict normals straight from the image instead of differentiating "
            "depth.\n\nMarkedly better at object boundaries, and the only way to "
            "get thin structures — chair legs, cables, window bars — which a "
            "depth map cannot resolve at all.\n\nApache-2.0, so no licence "
            "restriction. Costs a few seconds per import."
            if ready else
            "Not installed. To enable:\n    pip install diffusers\n\n"
            "Downloads ~2 GB of Apache-2.0 weights on first use."
        )
        self.neural_normals_toggle.toggled.connect(self._on_neural_normals)
        section.add(self.neural_normals_toggle)
        section.add_hint(
            "Changing this re-runs the import, since normals feed everything "
            "downstream."
        )

        self.column.addWidget(section)

    def _on_neural_normals(self, enabled: bool) -> None:
        if not self._syncing:
            self.neuralNormalsToggled.emit(bool(enabled))

    def _build_delighting(self) -> None:
        section = Section("De-lighting")

        self.delight_toggle = QtWidgets.QCheckBox("Relight the de-lit albedo")
        self.delight_toggle.setToolTip(
            "On: virtual lights are applied to the flat albedo map, so the original "
            "shadows are gone.\nOff: lights are applied to the raw image, which keeps "
            "the source lighting and can produce double shadows."
        )
        self.delight_toggle.toggled.connect(self._on_delight_toggle)
        section.add(self.delight_toggle)

        self.neural_toggle = QtWidgets.QCheckBox("Neural decomposition (compphoto/Intrinsic)")
        neural_ready = intrinsic_available()
        self.neural_toggle.setChecked(neural_ready)
        self.neural_toggle.setEnabled(neural_ready)
        self.neural_toggle.setToolTip(
            "Use the compphoto/Intrinsic ordinal-shading network instead of the "
            "analytic solver. Markedly better: it flattens shading almost "
            "completely and removes most of a hard cast shadow.\n\n"
            "Academic use only — the method is patented, with commercial "
            "licensing through SFU."
            if neural_ready else
            "Not installed. To enable:\n"
            '    pip install "intrinsic @ git+https://github.com/compphoto/Intrinsic@main"\n\n'
            "Academic use only — the method is patented."
        )
        self.neural_toggle.toggled.connect(self._on_neural_toggled)
        section.add(self.neural_toggle)

        self.delight_strength = SliderRow(
            "Strength", 0.0, 1.0, 1.0,
            tooltip="How much of the estimated shading is divided out.",
        )
        section.add(self.delight_strength)

        self.delight_geometry = SliderRow(
            "Geometry shading", 0.0, 1.0, 0.9,
            tooltip="Removes the shading the estimated normals and the recovered light "
                    "direction explain, including traced cast shadows.",
        )
        section.add(self.delight_geometry)

        self.delight_residual = SliderRow(
            "Residual falloff", 0.0, 1.0, 0.5,
            tooltip="Removes the smooth illumination geometry cannot explain — "
                    "vignetting, distance falloff, bounce light. Too high also flattens "
                    "large-scale albedo variation.",
        )
        section.add(self.delight_residual)

        self.delight_radius = SliderRow(
            "Falloff scale", 0.01, 0.25, 0.07, decimals=3,
            tooltip="Low-pass scale for the residual, as a fraction of the image. "
                    "Larger keeps more large-scale variation in the albedo.",
        )
        section.add(self.delight_radius)

        self.delight_shadows = QtWidgets.QCheckBox("Trace the original cast shadows")
        self.delight_shadows.setChecked(True)
        self.delight_shadows.setToolTip(
            "Recovers the photograph's own light direction from the shading, then "
            "raymarches the depth buffer to find where that light was blocked.\n"
            "This is what removes hard cast shadows — a filter cannot tell one from "
            "a dark patch of paint, but geometry can."
        )
        section.add(self.delight_shadows)

        self.delight_cast = SliderRow("Colour cast removal", 0.0, 1.0, 0.6)
        section.add(self.delight_cast)

        self.delight_preserve = SliderRow("Preserve contrast", 0.0, 1.0, 0.0)
        section.add(self.delight_preserve)

        self.delight_apply = QtWidgets.QPushButton("Recompute albedo")
        self.delight_apply.setObjectName("primary")
        self.delight_apply.clicked.connect(self._on_delight_apply)
        section.add(self.delight_apply)
        section.add_hint(
            "Re-runs only the intrinsic decomposition — depth and normals are reused, "
            "so this is much faster than reimporting."
        )

        # The analytic sliders drive the fallback solver only; the neural
        # path ignores every one of them, so leaving them live would be a
        # row of controls that silently do nothing.
        self._analytic_controls = (
            self.delight_strength,
            self.delight_geometry,
            self.delight_residual,
            self.delight_radius,
            self.delight_shadows,
            self.delight_cast,
            self.delight_preserve,
        )
        self._update_delight_controls()

        self.column.addWidget(section)

    def _on_neural_toggled(self, _enabled: bool) -> None:
        self._update_delight_controls()
        if not self._syncing:
            self._on_delight_apply()

    def _update_delight_controls(self) -> None:
        analytic_live = not self.neural_toggle.isChecked()
        for control in self._analytic_controls:
            control.setEnabled(analytic_live)

    def _build_render(self) -> None:
        section = Section("Viewport")

        self.view_combo = QtWidgets.QComboBox()
        for mode, label in (
            (ViewMode.BEAUTY, "Beauty (relit)"),
            (ViewMode.ALBEDO, "Albedo"),
            (ViewMode.ORIGINAL, "Original"),
            (ViewMode.NORMAL, "Normals"),
            (ViewMode.DEPTH, "Depth"),
            (ViewMode.SHADING, "Shading mask"),
            (ViewMode.SPECULAR, "Specular only"),
        ):
            self.view_combo.addItem(label, int(mode))
        self.view_combo.currentIndexChanged.connect(self._on_view_mode)
        section.add_row("Show", self.view_combo)

        self.depth_scale = SliderRow(
            "Depth scale", 0.1, 4.0, 1.0,
            tooltip="Scales the whole depth buffer. Leave at 1.0 with metric depth; "
                    "raise or lower it to correct the absolute scale on images where "
                    "the estimate reads too flat or too deep.",
        )
        self.depth_scale.valueChanged.connect(self._on_depth_scale)
        section.add(self.depth_scale)

        self.tonemap_check = QtWidgets.QCheckBox("ACES tonemapping")
        self.tonemap_check.toggled.connect(self._on_tonemap)
        section.add(self.tonemap_check)

        self.gizmo_check = QtWidgets.QCheckBox("Show light gizmos")
        self.gizmo_check.toggled.connect(self._on_gizmos)
        section.add(self.gizmo_check)

        self.column.addWidget(section)

    # ------------------------------------------------------------------
    # Light list operations
    # ------------------------------------------------------------------
    def add_light(self, kind: LightType) -> None:
        base = self.scene.active_light()
        light = Light(kind=kind)
        if base is not None:
            # Offset from the current light so the new gizmo is not hidden
            # exactly underneath the old one.
            light.position = [base.position[0] + 0.25, base.position[1], base.position[2]]
            light.direction = list(base.direction)
        if kind == LightType.DIRECTIONAL:
            light.attenuation_constant = 1.0
            light.attenuation_linear = 0.0
            light.attenuation_quadratic = 0.0
            light.intensity = 1.4

        if self.scene.add_light(light) is None:
            QtWidgets.QMessageBox.information(
                self,
                "Light limit reached",
                f"The deferred shader evaluates at most {MAX_LIGHTS} lights per pixel.",
            )
            return

        self.rebuild_light_list()
        self.lightsChanged.emit()
        self.sceneChanged.emit()

    def duplicate_light(self) -> None:
        light = self.scene.active_light()
        if light is None:
            return
        clone = Light.from_dict(light.to_dict())
        clone.name = f"{light.name} copy"
        clone.position = [light.position[0] + 0.2, light.position[1], light.position[2]]
        if self.scene.add_light(clone) is None:
            return
        self.rebuild_light_list()
        self.lightsChanged.emit()
        self.sceneChanged.emit()

    def remove_light(self) -> None:
        if self.scene.active_index < 0:
            return
        self.scene.remove_light(self.scene.active_index)
        self.rebuild_light_list()
        self.lightsChanged.emit()
        self.sceneChanged.emit()

    def rebuild_light_list(self) -> None:
        self._syncing = True
        try:
            self.light_list.clear()
            for light in self.scene.lights:
                item = QtWidgets.QListWidgetItem(f"{light.name}   ·   {light.kind.label}")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if light.enabled else Qt.CheckState.Unchecked
                )
                self.light_list.addItem(item)
            if 0 <= self.scene.active_index < self.light_list.count():
                self.light_list.setCurrentRow(self.scene.active_index)
        finally:
            self._syncing = False
        self._sync_light_properties()

    def _on_light_selected(self, row: int) -> None:
        if self._syncing or row < 0:
            return
        self.scene.active_index = row
        self._sync_light_properties()
        self.activeLightChanged.emit(row)
        self.sceneChanged.emit()

    def _on_light_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._syncing:
            return
        row = self.light_list.row(item)
        if 0 <= row < len(self.scene.lights):
            self.scene.lights[row].enabled = item.checkState() == Qt.CheckState.Checked
            self._sync_light_properties()
            self.sceneChanged.emit()

    # ------------------------------------------------------------------
    # Property edits
    # ------------------------------------------------------------------
    def _set_light_attr(self, name: str, value, relabel: bool = False) -> None:
        if self._syncing:
            return
        light = self.scene.active_light()
        if light is None:
            return
        setattr(light, name, value)
        if relabel:
            self.rebuild_light_list()
        self.sceneChanged.emit()

    def _on_type_changed(self, index: int) -> None:
        if self._syncing:
            return
        light = self.scene.active_light()
        if light is None:
            return
        light.kind = LightType(self.type_combo.itemData(index))
        self._update_type_dependent_enabled(light.kind)
        self.rebuild_light_list()
        self.sceneChanged.emit()

    def _on_name_changed(self) -> None:
        if self._syncing:
            return
        light = self.scene.active_light()
        if light is None:
            return
        text = self.name_edit.text().strip()
        if text and text != light.name:
            light.name = text
            self.rebuild_light_list()

    def _on_position_changed(self, value) -> None:
        self._set_light_attr("position", [float(v) for v in value])

    def _on_direction_changed(self, value) -> None:
        self._set_light_attr("direction", [float(v) for v in value])

    def _on_color_changed(self, value) -> None:
        self._set_light_attr("color", [float(v) for v in value])

    def _on_spot_inner(self, value: float) -> None:
        if self._syncing:
            return
        light = self.scene.active_light()
        if light is None:
            return
        light.spot_inner_degrees = float(value)
        if light.spot_inner_degrees >= light.spot_outer_degrees:
            light.spot_outer_degrees = min(90.0, light.spot_inner_degrees + 1.0)
            self.spot_outer.set_value(light.spot_outer_degrees)
        self.sceneChanged.emit()

    def _on_spot_outer(self, value: float) -> None:
        if self._syncing:
            return
        light = self.scene.active_light()
        if light is None:
            return
        light.spot_outer_degrees = float(value)
        if light.spot_inner_degrees >= light.spot_outer_degrees:
            light.spot_inner_degrees = max(0.0, light.spot_outer_degrees - 1.0)
            self.spot_inner.set_value(light.spot_inner_degrees)
        self.sceneChanged.emit()

    def _set_material(self, name: str, value) -> None:
        if self._syncing:
            return
        setattr(self.scene.material, name, value)
        self.sceneChanged.emit()

    def _on_spec_model(self, index: int) -> None:
        if self._syncing:
            return
        model = SpecularModel(self.spec_model_combo.itemData(index))
        self.scene.material.spec_model = model
        # Only one of the two parameters is live at a time; grey out the other
        # rather than leaving a control that silently does nothing.
        self.roughness_slider.setEnabled(model == SpecularModel.GGX)
        self.shininess_slider.setEnabled(model == SpecularModel.BLINN_PHONG)
        self.sceneChanged.emit()

    def _on_ambient_color(self, value) -> None:
        if self._syncing:
            return
        self.scene.render.ambient_color = [float(v) for v in value]
        self.sceneChanged.emit()

    def _set_shadow(self, name: str, value) -> None:
        if self._syncing:
            return
        setattr(self.scene.shadows, name, value)
        self.sceneChanged.emit()

    def _on_delight_toggle(self, enabled: bool) -> None:
        if self._syncing:
            return
        self.scene.render.shading_source = (
            ShadingSource.ALBEDO if enabled else ShadingSource.ORIGINAL
        )
        self.sceneChanged.emit()

    def _on_delight_apply(self) -> None:
        self.delightRequested.emit(self.delight_settings())

    def delight_settings(self) -> DelightSettings:
        return DelightSettings(
            geometry_weight=self.delight_geometry.value(),
            residual_weight=self.delight_residual.value(),
            detail_radius=self.delight_radius.value(),
            strength=self.delight_strength.value(),
            color_cast_removal=self.delight_cast.value(),
            preserve_contrast=self.delight_preserve.value(),
            trace_cast_shadows=self.delight_shadows.isChecked(),
            use_neural=self.neural_toggle.isChecked(),
        )

    def _on_view_mode(self, index: int) -> None:
        if self._syncing:
            return
        mode = ViewMode(self.view_combo.itemData(index))
        self.scene.render.view_mode = mode
        self.viewModeChanged.emit(mode)
        self.sceneChanged.emit()

    def set_view_mode(self, mode: ViewMode) -> None:
        """Programmatic view-mode change (from the View menu)."""
        self._syncing = True
        try:
            index = self.view_combo.findData(int(mode))
            if index >= 0:
                self.view_combo.setCurrentIndex(index)
            self.scene.render.view_mode = mode
        finally:
            self._syncing = False
        self.sceneChanged.emit()

    def _on_depth_scale(self, value: float) -> None:
        if self._syncing:
            return
        self.scene.render.depth_scale = float(value)
        self.sceneChanged.emit()

    def _on_tonemap(self, enabled: bool) -> None:
        if self._syncing:
            return
        self.scene.render.tonemap = bool(enabled)
        self.sceneChanged.emit()

    def _on_gizmos(self, enabled: bool) -> None:
        if self._syncing:
            return
        self.scene.render.show_gizmos = bool(enabled)
        self.sceneChanged.emit()

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------
    def _update_type_dependent_enabled(self, kind: LightType) -> None:
        is_spot = kind == LightType.SPOT
        is_directional = kind == LightType.DIRECTIONAL
        self.spot_inner.setEnabled(is_spot)
        self.spot_outer.setEnabled(is_spot)
        self.direction_row.setEnabled(is_spot or is_directional)
        # A directional light is infinitely far away: position only places
        # its gizmo, and distance attenuation does not apply.
        self.position_row.setEnabled(True)
        for slider in (self.atten_const, self.atten_linear, self.atten_quad):
            slider.setEnabled(not is_directional)
        self.atten_label.setEnabled(not is_directional)

    def _sync_light_properties(self) -> None:
        light = self.scene.active_light()
        has_light = light is not None
        self.light_section.setEnabled(has_light)
        self.duplicate_button.setEnabled(has_light)
        self.remove_button.setEnabled(has_light)
        if light is None:
            return

        self._syncing = True
        try:
            index = self.type_combo.findData(int(light.kind))
            if index >= 0:
                self.type_combo.setCurrentIndex(index)
            self.name_edit.setText(light.name)
            self.position_row.set_value(light.position)
            self.direction_row.set_value(light.direction)
            self.color_button.set_color(light.color)
            self.intensity_slider.set_value(light.intensity)
            self.enabled_check.setChecked(light.enabled)
            self.shadow_check.setChecked(light.casts_shadow)
            self.atten_const.set_value(light.attenuation_constant)
            self.atten_linear.set_value(light.attenuation_linear)
            self.atten_quad.set_value(light.attenuation_quadratic)
            self.spot_inner.set_value(light.spot_inner_degrees)
            self.spot_outer.set_value(light.spot_outer_degrees)
            self._update_type_dependent_enabled(light.kind)
        finally:
            self._syncing = False

    def sync_from_scene(self) -> None:
        """Push the entire scene state into the widgets."""
        material = self.scene.material
        shadows = self.scene.shadows
        render = self.scene.render

        self._syncing = True
        try:
            self.ambient_slider.set_value(material.ambient)
            self.ambient_color.set_color(render.ambient_color)
            self.diffuse_slider.set_value(material.diffuse)
            self.specular_slider.set_value(material.specular)
            self.roughness_slider.set_value(material.roughness)
            self.shininess_slider.set_value(material.shininess)
            self.metallic_slider.set_value(material.metallic)
            self.normal_strength_slider.set_value(material.normal_strength)
            self.base_light_slider.set_value(material.base_light)
            self.exposure_slider.set_value(material.exposure)

            index = self.spec_model_combo.findData(int(material.spec_model))
            if index >= 0:
                self.spec_model_combo.setCurrentIndex(index)
            self.roughness_slider.setEnabled(material.spec_model == SpecularModel.GGX)
            self.shininess_slider.setEnabled(material.spec_model == SpecularModel.BLINN_PHONG)

            self.shadow_enabled.setChecked(shadows.enabled)
            self.shadow_steps.set_value(shadows.steps)
            self.shadow_rays.set_value(shadows.rays)
            self.shadow_distance.set_value(shadows.max_distance)
            self.shadow_softness.set_value(shadows.softness)
            self.shadow_bias.set_value(shadows.bias)
            self.shadow_strength.set_value(shadows.strength)

            self.delight_toggle.setChecked(render.shading_source == ShadingSource.ALBEDO)
            self.depth_scale.set_value(render.depth_scale)
            self.tonemap_check.setChecked(render.tonemap)
            self.gizmo_check.setChecked(render.show_gizmos)

            view_index = self.view_combo.findData(int(render.view_mode))
            if view_index >= 0:
                self.view_combo.setCurrentIndex(view_index)
        finally:
            self._syncing = False

        self.rebuild_light_list()

    def sync_active_light(self) -> None:
        """Refresh only the position/direction fields, e.g. after a drag."""
        light = self.scene.active_light()
        if light is None:
            return
        self._syncing = True
        try:
            self.position_row.set_value(light.position)
            self.direction_row.set_value(light.direction)
        finally:
            self._syncing = False
