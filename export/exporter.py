"""Export the beauty pass and the full G-buffer.

Everything is encoded in memory and written once, so a failure part-way
through leaves no half-written archive next to the user's images.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import datetime, timezone

import cv2
import numpy as np

from core.gbuffer import GBuffer
from core.imageio import save_image, to_uint8
from core.scene import Scene
from pipeline.delighting_engine import colorize_depth
from pipeline.normal_engine import encode_normal_for_display

GBUFFER_FILTER = "G-buffer archive (*.zip)"


def export_relit_image(path: str, image: np.ndarray, quality: int = 95) -> str:
    """Write the rendered beauty pass to PNG/JPEG/WebP."""
    save_image(path, image, quality=quality)
    return path


def _encode(extension: str, data: np.ndarray, params: list[int] | None = None) -> bytes:
    ok, encoded = cv2.imencode(extension, data, params or [])
    if not ok:
        raise IOError(f"Failed to encode a {extension} buffer")
    return encoded.tobytes()


def _encode_rgb_png(image: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(to_uint8(image), cv2.COLOR_RGB2BGR)
    return _encode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 4])


def _encode_gray16(data: np.ndarray) -> tuple[bytes, float, float]:
    """16-bit PNG plus the range needed to undo the normalisation.

    Depth and shading are unbounded floats; storing them as normalised
    16-bit keeps the archive readable by any image tool while the manifest
    records how to recover the original values.
    """
    finite = data[np.isfinite(data)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 1.0
    span = max(hi - lo, 1e-8)
    normalised = np.clip((data - lo) / span, 0.0, 1.0)
    as_u16 = (normalised * 65535.0 + 0.5).astype(np.uint16)
    return _encode(".png", as_u16), lo, hi


def export_gbuffer_archive(
    path: str,
    buffer: GBuffer,
    scene: Scene | None = None,
    beauty: np.ndarray | None = None,
    *,
    include_raw: bool = True,
) -> str:
    """Write every pass, plus a manifest, to a single ``.zip``.

    ``include_raw`` adds lossless ``.npy`` copies of depth and normals for
    downstream tools; they dominate the archive size, hence the switch.
    """
    buffer.validate()

    depth_png, depth_lo, depth_hi = _encode_gray16(buffer.depth)
    shading_png, shade_lo, shade_hi = _encode_gray16(buffer.shading)
    intr = buffer.intrinsics

    manifest = {
        "generator": "Relighting Studio",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": buffer.source_path,
        "resolution": {"width": buffer.width, "height": buffer.height},
        "intrinsics": {
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.cx,
            "cy": intr.cy,
            "matrix": intr.matrix.tolist(),
        },
        "encodings": {
            "depth.png": {
                "format": "uint16 PNG, normalised",
                "min": depth_lo,
                "max": depth_hi,
                "units": buffer.meta.get("depth_units", "unknown"),
                "metric": bool(buffer.meta.get("depth_is_metric", False)),
                "decode": "depth = min + (pixel / 65535) * (max - min)",
            },
            "shading.png": {
                "format": "uint16 PNG, normalised",
                "min": shade_lo,
                "max": shade_hi,
                "decode": "shading = min + (pixel / 65535) * (max - min)",
            },
            "normal.png": {
                "format": "uint8 RGB",
                "decode": "n = pixel/255*2 - 1, with Z flipped to the viewer-facing "
                          "convention; camera space is X right, Y up, Z away",
            },
            "albedo.png": {"format": "uint8 sRGB"},
            "original.png": {"format": "uint8 sRGB"},
        },
        "pipeline": dict(buffer.meta),
    }
    if scene is not None:
        manifest["scene"] = scene.to_dict()

    entries: list[tuple[str, bytes]] = [
        ("original.png", _encode_rgb_png(buffer.original)),
        ("albedo.png", _encode_rgb_png(buffer.albedo)),
        ("normal.png", _encode_rgb_png(encode_normal_for_display(buffer.normal))),
        ("depth.png", depth_png),
        ("depth_visualised.png", _encode_rgb_png(colorize_depth(buffer.depth))),
        ("shading.png", shading_png),
    ]

    if beauty is not None:
        entries.append(("beauty.png", _encode_rgb_png(beauty)))
        manifest["encodings"]["beauty.png"] = {"format": "uint8 sRGB, relit render"}

    if include_raw:
        for name, array in (("depth.npy", buffer.depth), ("normal.npy", buffer.normal)):
            stream = io.BytesIO()
            np.save(stream, array.astype(np.float32), allow_pickle=False)
            entries.append((name, stream.getvalue()))

    entries.append(("manifest.json", json.dumps(manifest, indent=2).encode("utf-8")))
    entries.append(("README.txt", _archive_readme(buffer).encode("utf-8")))

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)

    return path


def _archive_readme(buffer: GBuffer) -> str:
    meta = buffer.meta
    return (
        "Relighting Studio — G-buffer export\n"
        "===================================\n\n"
        f"Source        : {buffer.source_path or 'in-memory image'}\n"
        f"Resolution    : {buffer.width} x {buffer.height}\n"
        f"Depth model   : {meta.get('depth_backend', 'unknown')}\n"
        f"Depth units   : {meta.get('depth_units', 'unknown')}\n"
        f"Normal model  : {meta.get('normal_backend', 'unknown')}\n"
        f"Albedo model  : {meta.get('albedo_backend', 'unknown')}\n"
        f"Device        : {meta.get('device', 'unknown')}\n\n"
        "Files\n"
        "-----\n"
        "original.png          Source image as loaded.\n"
        "albedo.png            De-lit base colour (sRGB).\n"
        "normal.png            Surface normals, viewer-facing convention.\n"
        "depth.png             Depth, 16-bit normalised — see manifest.json.\n"
        "depth_visualised.png  Colour-mapped depth for quick inspection.\n"
        "shading.png           Shading layer removed by the de-lighter.\n"
        "beauty.png            The relit render, when one was available.\n"
        "depth.npy/normal.npy  Lossless float32 copies.\n"
        "manifest.json         Intrinsics, decode formulas, and scene state.\n\n"
        "Unprojection\n"
        "------------\n"
        "X = (u - cx) * Z / fx,   Y = -(v - cy) * Z / fy,   Z = depth(u, v)\n"
        "with (u, v) in pixels from the top-left and the intrinsics in manifest.json.\n"
    )
