"""Scene description: lights, material, and render settings.

These are pure data objects with no Qt or OpenGL dependency, so the whole
scene can be serialised to JSON, diffed, or unit-tested in isolation.  The
viewport reads them once per frame and packs them into uniforms.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Iterator

import numpy as np

#: Hard ceiling shared by the Python side and the GLSL ``MAX_LIGHTS`` constant.
MAX_LIGHTS = 8


class LightType(IntEnum):
    POINT = 0
    SPOT = 1
    DIRECTIONAL = 2

    @property
    def label(self) -> str:
        return {
            LightType.POINT: "Point",
            LightType.SPOT: "Spot",
            LightType.DIRECTIONAL: "Directional",
        }[self]


_light_ids: Iterator[int] = itertools.count(1)


@dataclass
class Light:
    """A single virtual light source in camera space (metres/scene units)."""

    kind: LightType = LightType.POINT
    position: list[float] = field(default_factory=lambda: [0.0, 0.6, 0.6])
    #: Aim direction for spot and directional lights, pointing *away* from the
    #: light toward the scene.  Normalised when uploaded.
    direction: list[float] = field(default_factory=lambda: [0.0, -0.3, 1.0])
    color: list[float] = field(default_factory=lambda: [1.0, 0.96, 0.9])
    intensity: float = 3.0
    # Attenuation: 1 / (kc + kl*d + kq*d^2)
    attenuation_constant: float = 1.0
    attenuation_linear: float = 0.09
    attenuation_quadratic: float = 0.032
    #: Spot cone half-angles in degrees; the outer angle bounds the cone and
    #: the inner angle starts the smooth falloff.
    spot_inner_degrees: float = 18.0
    spot_outer_degrees: float = 30.0
    enabled: bool = True
    casts_shadow: bool = True
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.kind.label} {next(_light_ids)}"

    @property
    def position_np(self) -> np.ndarray:
        return np.asarray(self.position, dtype=np.float32)

    @property
    def direction_np(self) -> np.ndarray:
        d = np.asarray(self.direction, dtype=np.float32)
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = int(self.kind)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Light":
        data = dict(data)
        data["kind"] = LightType(int(data.get("kind", 0)))
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class SpecularModel(IntEnum):
    """Which specular lobe the fragment shader evaluates."""

    BLINN_PHONG = 0
    GGX = 1


@dataclass
class Material:
    """Global surface response applied to every pixel.

    A single image gives us no per-pixel material map, so roughness/metallic
    are scene-wide artistic controls rather than an inferred material buffer.
    """

    ambient: float = 0.18
    diffuse: float = 1.0
    specular: float = 0.35
    #: Blinn-Phong exponent; ignored by the GGX lobe.
    shininess: float = 48.0
    #: GGX roughness; ignored by the Blinn-Phong lobe.
    roughness: float = 0.45
    metallic: float = 0.0
    spec_model: SpecularModel = SpecularModel.GGX
    #: Blend the light-independent original image back in, which keeps the
    #: result recognisable when the de-lighting was aggressive.
    base_light: float = 0.12
    exposure: float = 1.0
    #: Bends surface normals toward the viewer (< 1) or exaggerates relief
    #: (> 1).  Useful when depth is relative rather than metric.
    normal_strength: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["spec_model"] = int(self.spec_model)
        return data


@dataclass
class ShadowSettings:
    enabled: bool = True
    #: Samples per shadow ray.  Too few and an occluder thinner than one
    #: step gets stepped over by some pixels and not others, which shows up
    #: as stipple in the penumbra rather than as a missing shadow.
    steps: int = 32
    #: Rays per pixel, each offset to a different phase within one step.
    #: The cheapest way to turn the march's stipple into a smooth penumbra;
    #: raising steps instead barely helps, because the variance comes from
    #: the start offset rather than the sampling resolution.
    rays: int = 2
    max_distance: float = 2.5
    bias: float = 0.012
    softness: float = 0.35
    strength: float = 0.85

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ShadingSource(IntEnum):
    """Which base texture the deferred shader multiplies lighting into."""

    ALBEDO = 0
    ORIGINAL = 1


class ViewMode(IntEnum):
    """Debug outputs the viewport can render instead of the beauty pass."""

    BEAUTY = 0
    ALBEDO = 1
    NORMAL = 2
    DEPTH = 3
    SHADING = 4
    SPECULAR = 5
    ORIGINAL = 6


@dataclass
class RenderSettings:
    shading_source: ShadingSource = ShadingSource.ALBEDO
    view_mode: ViewMode = ViewMode.BEAUTY
    #: Uniform scale applied to the depth buffer; relative-depth models need
    #: this to place lights at sensible distances.
    depth_scale: float = 1.0
    tonemap: bool = True
    show_gizmos: bool = True
    ambient_color: list[float] = field(default_factory=lambda: [0.42, 0.48, 0.6])

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shading_source"] = int(self.shading_source)
        data["view_mode"] = int(self.view_mode)
        return data


@dataclass
class Scene:
    """Mutable scene state owned by the main window."""

    lights: list[Light] = field(default_factory=list)
    material: Material = field(default_factory=Material)
    shadows: ShadowSettings = field(default_factory=ShadowSettings)
    render: RenderSettings = field(default_factory=RenderSettings)
    active_index: int = -1

    def active_light(self) -> Light | None:
        if 0 <= self.active_index < len(self.lights):
            return self.lights[self.active_index]
        return None

    def add_light(self, light: Light) -> Light | None:
        """Append a light and select it; returns ``None`` when at capacity."""
        if len(self.lights) >= MAX_LIGHTS:
            return None
        self.lights.append(light)
        self.active_index = len(self.lights) - 1
        return light

    def remove_light(self, index: int) -> None:
        if not 0 <= index < len(self.lights):
            return
        self.lights.pop(index)
        self.active_index = min(self.active_index, len(self.lights) - 1)

    def enabled_lights(self) -> list[Light]:
        return [light for light in self.lights if light.enabled]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lights": [light.to_dict() for light in self.lights],
            "material": self.material.to_dict(),
            "shadows": self.shadows.to_dict(),
            "render": self.render.to_dict(),
            "active_index": self.active_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scene":
        scene = cls()
        scene.lights = [Light.from_dict(d) for d in data.get("lights", [])]
        material = dict(data.get("material", {}))
        if "spec_model" in material:
            material["spec_model"] = SpecularModel(int(material["spec_model"]))
        scene.material = Material(**material)
        scene.shadows = ShadowSettings(**data.get("shadows", {}))
        render = dict(data.get("render", {}))
        if "shading_source" in render:
            render["shading_source"] = ShadingSource(int(render["shading_source"]))
        if "view_mode" in render:
            render["view_mode"] = ViewMode(int(render["view_mode"]))
        scene.render = RenderSettings(**render)
        scene.active_index = int(data.get("active_index", -1))
        return scene


def intensity_for_irradiance(light: Light, distance: float, target: float = 1.6) -> float:
    """Intensity that yields ``target`` irradiance at ``distance``.

    Placing a light is a geometric choice, but how bright it needs to be
    depends on where it landed -- inverse-square falloff means a light one
    scene-radius away needs an order of magnitude more power than one at
    arm's length.  Solving for intensity keeps the default exposure sane no
    matter what scale the depth estimate came back at.
    """
    d = max(float(distance), 1e-3)
    attenuation = (
        light.attenuation_constant
        + light.attenuation_linear * d
        + light.attenuation_quadratic * d * d
    )
    return float(target * max(attenuation, 1e-4))


def _place(offset: np.ndarray, center: np.ndarray, radius: float) -> tuple[list, list, float]:
    position = center + offset * max(radius, 1e-3)
    direction = center - position
    return (
        [float(v) for v in position],
        [float(v) for v in direction],
        float(np.linalg.norm(direction)),
    )


#: Offsets are in units of the frame half-width at the subject's depth (see
#: ``GBuffer.scene_radius``).  They are kept under ~1 unit laterally and
#: modest in Z so that both gizmos project inside the visible frame -- a
#: default light the user cannot see or grab is worse than no default.
_KEY_OFFSET = np.array([-0.60, 0.45, -0.35], dtype=np.float32)
_FILL_OFFSET = np.array([0.70, 0.10, -0.20], dtype=np.float32)


def default_key_light(center: np.ndarray, radius: float) -> Light:
    """A three-quarter key light placed relative to the reconstructed scene."""
    position, direction, distance = _place(_KEY_OFFSET, center, radius)
    light = Light(
        kind=LightType.POINT,
        position=position,
        direction=direction,
        color=[1.0, 0.93, 0.82],
        name="Key Light",
    )
    light.intensity = intensity_for_irradiance(light, distance, target=1.7)
    return light


def default_fill_light(center: np.ndarray, radius: float) -> Light:
    position, direction, distance = _place(_FILL_OFFSET, center, radius)
    light = Light(
        kind=LightType.POINT,
        position=position,
        direction=direction,
        color=[0.62, 0.74, 1.0],
        casts_shadow=False,
        name="Fill Light",
    )
    light.intensity = intensity_for_irradiance(light, distance, target=0.55)
    return light
