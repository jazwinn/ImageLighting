"""ModernGL renderer for the deferred relighting pass.

Deliberately Qt-free: it takes a live ``moderngl.Context`` from whoever owns
the window, a :class:`GBuffer`, and a :class:`Scene`, and draws.  That split
lets the same code render the interactive viewport and the full-resolution
export without duplicating uniform-packing logic.

Resource ownership is explicit -- :meth:`upload_gbuffer` releases the
previous textures before allocating new ones, and :meth:`release` tears
everything down -- because reloading images repeatedly is the normal
workflow and leaked textures would exhaust VRAM within a session.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import moderngl as mgl
import numpy as np

from core.gbuffer import GBuffer
from core.scene import MAX_LIGHTS, LightType, Scene, ShadingSource

SHADER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shaders")


def _read_shader(name: str) -> str:
    with open(os.path.join(SHADER_DIR, name), "r", encoding="utf-8") as handle:
        return handle.read()


@dataclass
class ImageTransform:
    """Maps image space onto the widget, honouring aspect fit, zoom and pan.

    ``scale``/``offset`` go straight into the vertex shaders; the inverse
    mapping in :meth:`widget_to_uv` is what makes click-to-place possible.
    """

    scale: tuple[float, float] = (1.0, 1.0)
    offset: tuple[float, float] = (0.0, 0.0)
    viewport: tuple[int, int] = (1, 1)

    @classmethod
    def fit(
        cls,
        viewport_w: int,
        viewport_h: int,
        image_w: int,
        image_h: int,
        zoom: float = 1.0,
        pan: tuple[float, float] = (0.0, 0.0),
    ) -> "ImageTransform":
        viewport_w = max(1, int(viewport_w))
        viewport_h = max(1, int(viewport_h))
        image_aspect = max(image_w, 1) / float(max(image_h, 1))
        view_aspect = viewport_w / float(viewport_h)

        if image_aspect >= view_aspect:
            sx, sy = 1.0, view_aspect / image_aspect
        else:
            sx, sy = image_aspect / view_aspect, 1.0

        return cls(
            scale=(sx * zoom, sy * zoom),
            offset=(float(pan[0]), float(pan[1])),
            viewport=(viewport_w, viewport_h),
        )

    def widget_to_uv(self, x: float, y: float) -> tuple[float, float]:
        """Widget pixel (origin top-left) to image UV (origin top-left)."""
        w, h = self.viewport
        ndc_x = (x / max(w, 1)) * 2.0 - 1.0
        ndc_y = 1.0 - (y / max(h, 1)) * 2.0
        sx = self.scale[0] if abs(self.scale[0]) > 1e-6 else 1e-6
        sy = self.scale[1] if abs(self.scale[1]) > 1e-6 else 1e-6
        img_x = (ndc_x - self.offset[0]) / sx
        img_y = (ndc_y - self.offset[1]) / sy
        return (img_x + 1.0) * 0.5, (1.0 - img_y) * 0.5

    def uv_to_widget(self, u: float, v: float) -> tuple[float, float]:
        w, h = self.viewport
        ndc_x = (u * 2.0 - 1.0) * self.scale[0] + self.offset[0]
        ndc_y = (1.0 - v * 2.0) * self.scale[1] + self.offset[1]
        return (ndc_x * 0.5 + 0.5) * w, (0.5 - ndc_y * 0.5) * h


class RelightRenderer:
    """Owns the shader programs, the G-buffer textures, and the draw calls."""

    #: Gizmo billboard radius in device pixels.
    GIZMO_RADIUS_PX = 13.0

    def __init__(self, ctx: mgl.Context) -> None:
        self.ctx = ctx
        self.buffer: GBuffer | None = None
        self._frame = 0

        self._textures: dict[str, mgl.Texture] = {}
        self._export_fbo: mgl.Framebuffer | None = None
        self._export_tex: mgl.Texture | None = None

        self.relight_prog = ctx.program(
            vertex_shader=_read_shader("fullscreen.vert"),
            fragment_shader=_read_shader("relight.frag"),
        )
        self.gizmo_prog = ctx.program(
            vertex_shader=_read_shader("gizmo.vert"),
            fragment_shader=_read_shader("gizmo.frag"),
        )
        self.pointer_prog = ctx.program(
            vertex_shader=_read_shader("pointer.vert"),
            fragment_shader=_read_shader("pointer.frag"),
        )

        # All three programs generate their geometry from gl_VertexID, so the
        # vertex arrays carry no buffers at all.
        self.quad_vao = ctx.vertex_array(self.relight_prog, [])
        self.gizmo_vao = ctx.vertex_array(self.gizmo_prog, [])
        self.pointer_vao = ctx.vertex_array(self.pointer_prog, [])

        self._bind_samplers()

    # -- uniforms ----------------------------------------------------------
    @staticmethod
    def _set(program: mgl.Program, name: str, value) -> None:
        """Assign a uniform, tolerating ones the compiler stripped out."""
        try:
            program[name].value = value
        except KeyError:
            pass

    @staticmethod
    def _write(program: mgl.Program, name: str, data: np.ndarray) -> None:
        try:
            program[name].write(np.ascontiguousarray(data, dtype="f4").tobytes())
        except KeyError:
            pass

    def _bind_samplers(self) -> None:
        for name, unit in (
            ("u_albedo", 0),
            ("u_original", 1),
            ("u_normal", 2),
            ("u_depth", 3),
            ("u_shading", 4),
        ):
            self._set(self.relight_prog, name, unit)
        self._set(self.gizmo_prog, "u_depth", 3)

    # -- G-buffer upload ---------------------------------------------------
    def upload_gbuffer(self, buffer: GBuffer) -> None:
        """Replace the resident G-buffer textures with a new decomposition."""
        buffer.validate()
        self.release_textures()

        width, height = buffer.width, buffer.height
        size = (width, height)

        def u8(image: np.ndarray) -> bytes:
            return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).tobytes()

        # sRGB-encoded colour survives 8 bits fine; normals and depth do not.
        self._textures["albedo"] = self.ctx.texture(size, 3, u8(buffer.albedo), alignment=1)
        self._textures["original"] = self.ctx.texture(size, 3, u8(buffer.original), alignment=1)

        encoded_normal = (buffer.normal * 0.5 + 0.5).astype("f2")
        self._textures["normal"] = self.ctx.texture(
            size, 3, encoded_normal.tobytes(), dtype="f2", alignment=1
        )
        self._textures["depth"] = self.ctx.texture(
            size, 1, buffer.depth.astype("f4").tobytes(), dtype="f4"
        )
        self._textures["shading"] = self.ctx.texture(
            size, 1, buffer.shading.astype("f2").tobytes(), dtype="f2"
        )

        for name, tex in self._textures.items():
            # Depth must not be interpolated across silhouettes during the
            # shadow march, so it stays nearest-filtered.
            tex.filter = (
                (mgl.NEAREST, mgl.NEAREST) if name == "depth" else (mgl.LINEAR, mgl.LINEAR)
            )
            tex.repeat_x = False
            tex.repeat_y = False

        self.buffer = buffer

    def release_textures(self) -> None:
        for tex in self._textures.values():
            tex.release()
        self._textures.clear()
        self.buffer = None

    def _bind_textures(self) -> None:
        for name, unit in (
            ("albedo", 0),
            ("original", 1),
            ("normal", 2),
            ("depth", 3),
            ("shading", 4),
        ):
            self._textures[name].use(unit)

    # -- drawing -----------------------------------------------------------
    def render(
        self,
        target: mgl.Framebuffer,
        transform: ImageTransform,
        scene: Scene,
        *,
        draw_gizmos: bool = True,
        background: tuple[float, float, float] = (0.09, 0.09, 0.11),
    ) -> None:
        target.use()
        self.ctx.clear(*background, 1.0)

        if self.buffer is None or not self._textures:
            return

        self._frame += 1
        self._bind_textures()
        self._apply_scene_uniforms(scene, transform)

        self.ctx.disable(mgl.DEPTH_TEST)
        self.ctx.disable(mgl.BLEND)
        self.quad_vao.render(mgl.TRIANGLE_STRIP, vertices=4)

        if draw_gizmos and scene.render.show_gizmos and scene.lights:
            self._render_gizmos(scene, transform)

    def _apply_scene_uniforms(self, scene: Scene, transform: ImageTransform) -> None:
        assert self.buffer is not None
        prog = self.relight_prog
        intr = self.buffer.intrinsics
        material = scene.material
        shadows = scene.shadows
        render = scene.render

        self._set(prog, "u_image_scale", transform.scale)
        self._set(prog, "u_image_offset", transform.offset)
        self._set(prog, "u_intrinsics", (intr.fx, intr.fy, intr.cx, intr.cy))
        self._set(prog, "u_resolution", (float(self.buffer.width), float(self.buffer.height)))
        self._set(prog, "u_depth_scale", float(render.depth_scale))

        self._set(prog, "u_ambient", float(material.ambient))
        self._set(prog, "u_diffuse", float(material.diffuse))
        self._set(prog, "u_specular", float(material.specular))
        self._set(prog, "u_shininess", float(material.shininess))
        self._set(prog, "u_roughness", float(material.roughness))
        self._set(prog, "u_metallic", float(material.metallic))
        self._set(prog, "u_base_light", float(material.base_light))
        self._set(prog, "u_exposure", float(material.exposure))
        self._set(prog, "u_normal_strength", float(material.normal_strength))
        self._set(prog, "u_ambient_color", tuple(float(c) for c in render.ambient_color))
        self._set(prog, "u_spec_model", int(scene.material.spec_model))

        self._set(prog, "u_shadow_enabled", 1 if shadows.enabled else 0)
        self._set(prog, "u_shadow_steps", int(shadows.steps))
        self._set(prog, "u_shadow_rays", int(shadows.rays))
        self._set(prog, "u_shadow_max_distance", float(shadows.max_distance))
        self._set(prog, "u_shadow_bias", float(shadows.bias))
        self._set(prog, "u_shadow_softness", float(shadows.softness))
        self._set(prog, "u_shadow_strength", float(shadows.strength))

        self._set(prog, "u_view_mode", int(render.view_mode))
        self._set(prog, "u_tonemap", 1 if render.tonemap else 0)
        self._set(prog, "u_use_original", 1 if render.shading_source == ShadingSource.ORIGINAL else 0)
        self._set(prog, "u_frame_seed", float(self._frame % 64))

        positions, directions, colors, attenuations = self._pack_lights(scene)
        self._set(prog, "u_light_count", int(min(len(scene.enabled_lights()), MAX_LIGHTS)))
        self._write(prog, "u_light_position", positions)
        self._write(prog, "u_light_direction", directions)
        self._write(prog, "u_light_color", colors)
        self._write(prog, "u_light_atten", attenuations)

    @staticmethod
    def _pack_lights(scene: Scene) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flatten the enabled lights into four fixed-size vec4 arrays.

        Everything is padded to ``vec4`` so the byte layout of a uniform
        array is unambiguous regardless of driver packing rules.
        """
        positions = np.zeros((MAX_LIGHTS, 4), dtype="f4")
        directions = np.zeros((MAX_LIGHTS, 4), dtype="f4")
        colors = np.zeros((MAX_LIGHTS, 4), dtype="f4")
        attenuations = np.zeros((MAX_LIGHTS, 4), dtype="f4")

        for i, light in enumerate(scene.enabled_lights()[:MAX_LIGHTS]):
            positions[i, :3] = light.position_np
            positions[i, 3] = float(int(light.kind))

            directions[i, :3] = light.direction_np
            directions[i, 3] = float(np.cos(np.deg2rad(light.spot_outer_degrees)))

            colors[i, :3] = np.asarray(light.color, dtype="f4") * float(light.intensity)
            # Keep the inner cone inside the outer one, otherwise the cone
            # falloff divides by a negative span and inverts.
            inner = min(light.spot_inner_degrees, light.spot_outer_degrees - 0.5)
            colors[i, 3] = float(np.cos(np.deg2rad(max(inner, 0.0))))

            attenuations[i, :3] = (
                light.attenuation_constant,
                light.attenuation_linear,
                light.attenuation_quadratic,
            )
            attenuations[i, 3] = 1.0 if light.casts_shadow else 0.0

        return positions, directions, colors, attenuations

    def _render_gizmos(self, scene: Scene, transform: ImageTransform) -> None:
        assert self.buffer is not None
        intr = self.buffer.intrinsics
        radius = self.GIZMO_RADIUS_PX

        gizmo_pos = np.zeros((MAX_LIGHTS, 4), dtype="f4")
        gizmo_col = np.zeros((MAX_LIGHTS, 4), dtype="f4")
        pointer_start = np.zeros((MAX_LIGHTS, 4), dtype="f4")
        pointer_dir = np.zeros((MAX_LIGHTS, 4), dtype="f4")

        scene_radius = max(self.buffer.scene_radius(), 1e-3)
        count = min(len(scene.lights), MAX_LIGHTS)

        for i, light in enumerate(scene.lights[:MAX_LIGHTS]):
            selected = i == scene.active_index
            gizmo_pos[i, :3] = light.position_np
            gizmo_pos[i, 3] = radius * (1.25 if selected else 1.0)

            colour = np.asarray(light.color, dtype="f4")
            if not light.enabled:
                colour = colour * 0.28 + 0.06
            gizmo_col[i, :3] = colour
            gizmo_col[i, 3] = 1.0 if selected else 0.0

            pointer_start[i, :3] = light.position_np
            pointer_start[i, 3] = scene_radius * 0.45
            pointer_dir[i, :3] = light.direction_np
            pointer_dir[i, 3] = (
                1.0 if light.kind in (LightType.SPOT, LightType.DIRECTIONAL) else 0.0
            )

        common = {
            "u_intrinsics": (intr.fx, intr.fy, intr.cx, intr.cy),
            "u_resolution": (float(self.buffer.width), float(self.buffer.height)),
            "u_image_scale": transform.scale,
            "u_image_offset": transform.offset,
        }

        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA

        # Pointers first, so the billboards sit on top of their own lines.
        for key, value in common.items():
            self._set(self.pointer_prog, key, value)
        self._write(self.pointer_prog, "u_pointer_start", pointer_start)
        self._write(self.pointer_prog, "u_pointer_dir", pointer_dir)
        self._write(self.pointer_prog, "u_gizmo_color", gizmo_col)
        try:
            self.ctx.line_width = 2.0
        except Exception:
            pass  # core profile drivers often clamp this to 1.0
        self.pointer_vao.render(mgl.LINES, vertices=2, instances=count)

        for key, value in common.items():
            self._set(self.gizmo_prog, key, value)
        self._set(
            self.gizmo_prog,
            "u_viewport",
            (float(transform.viewport[0]), float(transform.viewport[1])),
        )
        self._set(self.gizmo_prog, "u_depth_scale", float(scene.render.depth_scale))
        self._write(self.gizmo_prog, "u_gizmo_position", gizmo_pos)
        self._write(self.gizmo_prog, "u_gizmo_color", gizmo_col)
        self._textures["depth"].use(3)
        self.gizmo_vao.render(mgl.TRIANGLE_STRIP, vertices=4, instances=count)

        self.ctx.disable(mgl.BLEND)

    # -- offscreen ---------------------------------------------------------
    def render_to_array(
        self, scene: Scene, width: int | None = None, height: int | None = None
    ) -> np.ndarray:
        """Render the beauty pass at buffer resolution and read it back.

        Used by the exporter so the saved image is the full-resolution
        result rather than a screengrab of the widget.
        """
        if self.buffer is None:
            raise RuntimeError("No G-buffer loaded")

        width = int(width or self.buffer.width)
        height = int(height or self.buffer.height)
        fbo = self._ensure_export_target(width, height)

        # Identity transform: the image exactly fills the render target.
        transform = ImageTransform(scale=(1.0, 1.0), offset=(0.0, 0.0), viewport=(width, height))
        self.render(fbo, transform, scene, draw_gizmos=False, background=(0.0, 0.0, 0.0))

        data = fbo.read(components=3, alignment=1)
        image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        # OpenGL hands back rows bottom-up.
        return np.ascontiguousarray(image[::-1])

    def _ensure_export_target(self, width: int, height: int) -> mgl.Framebuffer:
        if (
            self._export_fbo is not None
            and self._export_tex is not None
            and self._export_tex.size == (width, height)
        ):
            return self._export_fbo

        self._release_export_target()
        self._export_tex = self.ctx.texture((width, height), 3, dtype="f1")
        self._export_fbo = self.ctx.framebuffer(color_attachments=[self._export_tex])
        return self._export_fbo

    def _release_export_target(self) -> None:
        if self._export_fbo is not None:
            self._export_fbo.release()
            self._export_fbo = None
        if self._export_tex is not None:
            self._export_tex.release()
            self._export_tex = None

    # -- teardown ----------------------------------------------------------
    def release(self) -> None:
        self.release_textures()
        self._release_export_target()
        for vao in (self.quad_vao, self.gizmo_vao, self.pointer_vao):
            vao.release()
        for prog in (self.relight_prog, self.gizmo_prog, self.pointer_prog):
            prog.release()
