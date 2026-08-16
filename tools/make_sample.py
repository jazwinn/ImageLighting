"""Generate the bundled sample scene.

A synthetic photograph is a much better smoke test than a real one: we know
the ground truth.  This module ray-traces a small scene with a single warm
key light, hard cast shadows, and strongly textured surfaces, so that

* depth estimation has real occlusion boundaries to find,
* the normal estimator has curved and flat surfaces to distinguish, and
* the de-lighter has an unambiguous shading gradient and cast shadow to
  remove -- if the albedo tab still shows the shadow, the de-lighting is
  under-strength.

Pure numpy, no renderer dependency; roughly 0.2 s at the default size.
"""

from __future__ import annotations

import os
import sys

import numpy as np

if __package__ in (None, ""):  # allow `python tools/make_sample.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.imageio import save_image

ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
)
SAMPLE_PATH = os.path.join(ASSETS_DIR, "sample_scene.png")

# Scene: camera at the origin looking down +Z, Y up.
_SPHERES = (
    # centre, radius, base colour, glossiness
    ((-0.55, -0.52, 3.10), 0.48, (0.82, 0.24, 0.20), 0.35),
    ((0.62, -0.66, 2.55), 0.34, (0.24, 0.46, 0.78), 0.55),
    ((0.10, -0.80, 4.30), 0.20, (0.90, 0.82, 0.30), 0.25),
)
_FLOOR_Y = -1.0
_BACK_Z = 7.5
_LIGHT = np.array([-2.4, 2.9, 0.9], dtype=np.float32)
_LIGHT_COLOR = np.array([1.0, 0.88, 0.72], dtype=np.float32) * 26.0
_AMBIENT = np.array([0.16, 0.19, 0.26], dtype=np.float32)


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def _intersect_spheres(origin: np.ndarray, direction: np.ndarray):
    """Nearest sphere hit per ray. Returns (t, index) with t = inf for misses."""
    best_t = np.full(direction.shape[:-1], np.inf, dtype=np.float32)
    best_i = np.full(direction.shape[:-1], -1, dtype=np.int32)

    for index, (center, radius, _color, _gloss) in enumerate(_SPHERES):
        oc = origin - np.asarray(center, dtype=np.float32)
        b = 2.0 * np.sum(direction * oc, axis=-1)
        c = float(np.dot(oc, oc) - radius * radius) if oc.ndim == 1 else (
            np.sum(oc * oc, axis=-1) - radius * radius
        )
        disc = b * b - 4.0 * c
        hit = disc > 0.0
        if not np.any(hit):
            continue
        root = np.sqrt(np.maximum(disc, 0.0))
        t = (-b - root) * 0.5
        # Take the far root when the near one is behind the ray origin.
        t_far = (-b + root) * 0.5
        t = np.where(t > 1e-3, t, t_far)
        valid = hit & (t > 1e-3) & (t < best_t)
        best_t = np.where(valid, t, best_t)
        best_i = np.where(valid, index, best_i)

    return best_t, best_i


def _shadowed(points: np.ndarray, light_dir: np.ndarray, light_distance: np.ndarray):
    """Hard shadow test against the spheres only."""
    origin = points + light_dir * 1e-3
    occluded = np.zeros(points.shape[:-1], dtype=bool)

    for center, radius, _color, _gloss in _SPHERES:
        oc = origin - np.asarray(center, dtype=np.float32)
        b = 2.0 * np.sum(light_dir * oc, axis=-1)
        c = np.sum(oc * oc, axis=-1) - radius * radius
        disc = b * b - 4.0 * c
        root = np.sqrt(np.maximum(disc, 0.0))
        t = (-b - root) * 0.5
        occluded |= (disc > 0.0) & (t > 1e-3) & (t < light_distance)

    return occluded


def _floor_albedo(points: np.ndarray) -> np.ndarray:
    """Checkerboard with a subtle grain, so albedo detail is unmistakable."""
    check = ((np.floor(points[..., 0] * 2.2) + np.floor(points[..., 2] * 2.2)) % 2.0)
    base = np.where(check > 0.5, 0.62, 0.34)[..., None] * np.array(
        [1.0, 0.97, 0.92], dtype=np.float32
    )
    grain = 0.04 * np.sin(points[..., 0] * 31.0) * np.cos(points[..., 2] * 27.0)
    return np.clip(base + grain[..., None], 0.0, 1.0)


def _wall_albedo(points: np.ndarray) -> np.ndarray:
    stripes = 0.5 + 0.5 * np.sin(points[..., 0] * 4.5)
    tint = np.array([0.42, 0.45, 0.52], dtype=np.float32)
    return np.clip(tint * (0.85 + 0.25 * stripes[..., None]), 0.0, 1.0)


def _sphere_albedo(index: int, normal: np.ndarray) -> np.ndarray:
    base = np.asarray(_SPHERES[index][2], dtype=np.float32)
    # Latitude banding gives the de-lighter texture that must survive.
    band = 0.5 + 0.5 * np.sin(normal[..., 1] * 14.0)
    return np.clip(base[None, :] * (0.78 + 0.34 * band[..., None]), 0.0, 1.0)


def render_sample(width: int = 960, height: int = 640, fov_y: float = 55.0) -> np.ndarray:
    """Ray-trace the sample scene, returning RGB float32 in [0, 1] (sRGB)."""
    fy = (height * 0.5) / np.tan(np.deg2rad(fov_y) * 0.5)
    fx = fy
    cx, cy = width * 0.5, height * 0.5

    us = np.arange(width, dtype=np.float32) + 0.5
    vs = np.arange(height, dtype=np.float32) + 0.5
    uu, vv = np.meshgrid(us, vs)
    direction = _normalize(
        np.stack([(uu - cx) / fx, -(vv - cy) / fy, np.ones_like(uu)], axis=-1)
    )
    origin = np.zeros(3, dtype=np.float32)

    # --- intersect ---
    t_sphere, sphere_index = _intersect_spheres(origin, direction)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(direction[..., 1] < -1e-6, _FLOOR_Y / direction[..., 1], np.inf)
        t_wall = np.where(direction[..., 2] > 1e-6, _BACK_Z / direction[..., 2], np.inf)
    t_floor = np.where(np.isfinite(t_floor) & (t_floor > 0), t_floor, np.inf)
    t_wall = np.where(np.isfinite(t_wall) & (t_wall > 0), t_wall, np.inf)

    t = np.minimum(np.minimum(t_sphere, t_floor), t_wall)
    hit_sphere = np.isfinite(t_sphere) & (t_sphere <= t)
    hit_floor = np.isfinite(t_floor) & (t_floor <= t) & ~hit_sphere
    hit_wall = np.isfinite(t_wall) & (t_wall <= t) & ~hit_sphere & ~hit_floor

    t = np.where(np.isfinite(t), t, _BACK_Z)
    points = origin + direction * t[..., None]

    # --- normals ---
    normal = np.zeros_like(points)
    normal[hit_floor] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    normal[hit_wall] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    for index, (center, _radius, _color, _gloss) in enumerate(_SPHERES):
        mask = hit_sphere & (sphere_index == index)
        if np.any(mask):
            normal[mask] = _normalize(points[mask] - np.asarray(center, dtype=np.float32))

    # --- albedo ---
    albedo = np.zeros_like(points)
    albedo[hit_floor] = _floor_albedo(points)[hit_floor]
    albedo[hit_wall] = _wall_albedo(points)[hit_wall]
    gloss = np.full(t.shape, 0.18, dtype=np.float32)
    for index in range(len(_SPHERES)):
        mask = hit_sphere & (sphere_index == index)
        if np.any(mask):
            albedo[mask] = _sphere_albedo(index, normal)[mask]
            gloss[mask] = _SPHERES[index][3]

    # --- shading ---
    to_light = _LIGHT - points
    light_distance = np.linalg.norm(to_light, axis=-1)
    light_dir = to_light / np.maximum(light_distance[..., None], 1e-6)

    n_dot_l = np.clip(np.sum(normal * light_dir, axis=-1), 0.0, 1.0)
    attenuation = 1.0 / np.maximum(light_distance * light_distance, 1e-4)
    visibility = (~_shadowed(points, light_dir, light_distance)).astype(np.float32)

    view_dir = -direction
    half = _normalize(light_dir + view_dir)
    n_dot_h = np.clip(np.sum(normal * half, axis=-1), 0.0, 1.0)
    shininess = 12.0 + 180.0 * gloss
    specular = np.power(n_dot_h, shininess) * gloss

    radiance = (
        albedo * _AMBIENT[None, None, :]
        + albedo
        * (n_dot_l * attenuation * visibility)[..., None]
        * _LIGHT_COLOR[None, None, :]
        + (specular * attenuation * visibility)[..., None] * _LIGHT_COLOR[None, None, :] * 0.6
    )

    # Distance haze, so the depth estimator sees an aerial-perspective cue too.
    haze = np.clip((t - 2.0) / 8.0, 0.0, 1.0)[..., None]
    radiance = radiance * (1.0 - haze * 0.35) + np.array(
        [0.10, 0.12, 0.16], dtype=np.float32
    ) * haze * 0.35

    # Vignette, then sRGB encode.
    radius = np.sqrt(((uu - cx) / cx) ** 2 + ((vv - cy) / cy) ** 2)
    radiance *= (1.0 - 0.22 * np.clip(radius / 1.6, 0.0, 1.0) ** 2)[..., None]

    linear = np.clip(radiance, 0.0, 1.0)
    srgb = np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1.0 / 2.4) - 0.055
    )
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


def ensure_sample_image(path: str = SAMPLE_PATH, force: bool = False) -> str:
    """Render the sample to disk if it is not already there; return its path."""
    if force or not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_image(path, render_sample())
    return path


if __name__ == "__main__":
    print(ensure_sample_image(force=True))
