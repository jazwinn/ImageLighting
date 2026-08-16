"""G-buffer container and pinhole camera intrinsics.

The whole application passes exactly one of these objects around: the AI
worker produces it, the OpenGL viewport uploads it as textures, and the
exporter writes it to disk.  Keeping it a plain dataclass of numpy arrays
means nothing in the pipeline needs to know about Qt or OpenGL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics ``K`` used for unprojection and reprojection.

    A single image carries no calibration, so we synthesise a plausible
    pinhole camera from a vertical field of view (55 degrees is a decent
    stand-in for a consumer photo) with the principal point at the centre.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, fov_y_degrees: float = 55.0) -> "CameraIntrinsics":
        fov_y = np.deg2rad(float(fov_y_degrees))
        fy = (height * 0.5) / np.tan(fov_y * 0.5)
        return cls(
            width=int(width),
            height=int(height),
            fx=float(fy),  # square pixels
            fy=float(fy),
            cx=float(width) * 0.5,
            cy=float(height) * 0.5,
        )

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def unproject(self, depth: np.ndarray) -> np.ndarray:
        """Unproject a metric depth map into a camera-space position map.

            X = (u - cx) * Z / fx      Y = (v - cy) * Z / fy      Z = Z(u, v)

        Returns an ``(H, W, 3)`` float32 array.  The Y axis points *up* in
        camera space (the image V axis points down), which matches the
        convention used by the GLSL shaders and the light gizmos.
        """
        height, width = depth.shape[:2]
        us = np.arange(width, dtype=np.float32) + 0.5
        vs = np.arange(height, dtype=np.float32) + 0.5
        uu, vv = np.meshgrid(us, vs)
        z = depth.astype(np.float32)
        x = (uu - self.cx) * z / self.fx
        y = -(vv - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=-1).astype(np.float32)

    def scaled(self, width: int, height: int) -> "CameraIntrinsics":
        """Intrinsics for the same camera sampled at a different resolution."""
        sx = width / float(self.width)
        sy = height / float(self.height)
        return CameraIntrinsics(
            width=int(width),
            height=int(height),
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
        )


@dataclass
class GBuffer:
    """Everything the deferred shader needs, plus provenance metadata.

    Array contracts (all ``float32`` unless noted, all shaped ``H x W x C``):

    ``original``  RGB in [0, 1], sRGB-encoded, as loaded from disk.
    ``depth``     ``H x W`` metric-ish depth in scene units, strictly > 0.
    ``normal``    unit-length camera-space normals sharing the position map's
                  axes (X right, Y up, Z away), so viewer-facing normals have
                  a negative Z.
    ``albedo``    RGB in [0, 1], sRGB-encoded, de-lit base colour.
    ``shading``   ``H x W`` scalar shading/irradiance the de-lighter removed.
    """

    original: np.ndarray
    depth: np.ndarray
    normal: np.ndarray
    albedo: np.ndarray
    shading: np.ndarray
    intrinsics: CameraIntrinsics
    source_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.original.shape[1])

    @property
    def height(self) -> int:
        return int(self.original.shape[0])

    @property
    def aspect(self) -> float:
        return self.width / float(self.height)

    def position_map(self) -> np.ndarray:
        """Camera-space XYZ per pixel (computed on demand, not cached)."""
        return self.intrinsics.unproject(self.depth)

    def depth_range(self) -> tuple[float, float]:
        finite = self.depth[np.isfinite(self.depth)]
        if finite.size == 0:
            return 0.1, 1.0
        return float(finite.min()), float(finite.max())

    def subject_depth(self) -> float:
        """Representative depth of the foreground subject.

        A low percentile rather than the median: the median of a typical
        photograph sits on the background, and lights placed around the
        background end up behind everything the user cares about.
        """
        finite = self.depth[np.isfinite(self.depth)]
        if finite.size == 0:
            return 1.0
        return float(max(np.percentile(finite, 35.0), 1e-3))

    def depth_span(self) -> float:
        """Robust front-to-back extent of the scene.

        A 5th-to-95th percentile range rather than min-to-max: with metric
        depth a single patch of distant sky would otherwise stretch this to
        a hundred metres and drag everything scaled from it along with it.
        """
        finite = self.depth[np.isfinite(self.depth)]
        if finite.size == 0:
            return 1.0
        span = float(np.percentile(finite, 95.0) - np.percentile(finite, 5.0))
        return max(span, 1e-3)

    def scene_center(self) -> np.ndarray:
        """Point on the optical axis at the subject's depth.

        Lights are placed relative to this, so they orbit the middle of the
        frame rather than the centroid of a possibly lopsided depth map.
        """
        return np.array([0.0, 0.0, self.subject_depth()], dtype=np.float32)

    def scene_radius(self) -> float:
        """Half-width of the view frustum at the subject's depth.

        This is the natural scale for anything that has to stay on screen:
        an offset of one unit moves a light from the optical axis to the
        edge of the frame, so gizmos placed within about one unit remain
        visible and grabbable.  A bounding-sphere radius would not do --
        it is dominated by however far away the background happens to be,
        which pushes default lights clean out of the picture.
        """
        z = self.subject_depth()
        return float(max(0.5 * self.width * z / self.intrinsics.fx, 1e-3))

    def validate(self) -> None:
        """Fail loudly on a malformed buffer rather than in a GLSL sampler."""
        h, w = self.original.shape[:2]
        checks = {
            "original": (self.original.shape, (h, w, 3)),
            "normal": (self.normal.shape, (h, w, 3)),
            "albedo": (self.albedo.shape, (h, w, 3)),
            "depth": (self.depth.shape, (h, w)),
            "shading": (self.shading.shape, (h, w)),
        }
        for name, (got, want) in checks.items():
            if tuple(got) != tuple(want):
                raise ValueError(f"GBuffer.{name} has shape {got}, expected {want}")

        # Non-finite values do not announce themselves once they are inside a
        # GPU texture -- they surface as black holes or NaN-poisoned shading
        # several stages later.  Catch them at the boundary instead.
        for name in ("original", "depth", "normal", "albedo", "shading"):
            array = getattr(self, name)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"GBuffer.{name} contains NaN or infinite values")
        if self.intrinsics.width != w or self.intrinsics.height != h:
            raise ValueError(
                f"Intrinsics are for {self.intrinsics.width}x{self.intrinsics.height}, "
                f"buffer is {w}x{h}"
            )
