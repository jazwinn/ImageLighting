"""Relighting Studio — application entry point.

Sets up the OpenGL surface format *before* the QApplication exists (Qt
locks the default format in once the first context is created), installs
the dark theme, and opens the main window.

    python main.py [image] [options]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Make the project importable when launched from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.qt_compat import QT_BINDING, Qt, QtCore, QtGui, QtWidgets  # noqa: E402
from pipeline.base import DEFAULT_MAX_SIDE  # noqa: E402
from pipeline.worker import PipelineConfig  # noqa: E402
from ui.main_window import APP_NAME, ORG_NAME, MainWindow  # noqa: E402
from ui.theme import apply_theme  # noqa: E402


def configure_surface_format() -> None:
    """Request an OpenGL 3.3 core profile for every window in the process."""
    fmt = QtGui.QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSwapBehavior(QtGui.QSurfaceFormat.SwapBehavior.DoubleBuffer)
    # Vsync would cap the viewport at the display refresh rate and make the
    # FPS readout meaningless; the render loop is already frame-limited.
    fmt.setSwapInterval(0)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Relighting Studio",
        description="Single-image de-lighting and real-time virtual relighting.",
    )
    parser.add_argument("image", nargs="?", help="image to open on startup")
    parser.add_argument(
        "--sample", action="store_true", help="open the bundled synthetic test scene"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="force CPU inference even if CUDA is present"
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=DEFAULT_MAX_SIDE,
        help=f"longest edge used for AI inference (default {DEFAULT_MAX_SIDE})",
    )
    parser.add_argument(
        "--buffer-max-side",
        type=int,
        default=1600,
        help="longest edge of the G-buffer and its GPU textures (default 1600)",
    )
    parser.add_argument("--depth-model", help="override the Hugging Face depth model id")
    parser.add_argument("--normal-onnx", help="path to an ONNX surface-normal model")
    parser.add_argument(
        "--no-neural-normals",
        action="store_true",
        help="derive normals from depth instead of predicting them with StableNormal",
    )
    parser.add_argument("--albedo-onnx", help="path to an ONNX intrinsic-decomposition model")
    parser.add_argument(
        "--fov", type=float, default=55.0, help="assumed vertical field of view in degrees"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.verbose:
        # -v is for debugging this application, not the HTTP stack the model
        # download happens to sit on; those loggers emit hundreds of lines
        # per launch and bury everything worth reading.
        for noisy in ("httpx", "httpcore", "urllib3", "filelock", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.cpu:
        os.environ["IMAGELIGHTING_FORCE_CPU"] = "1"

    configure_surface_format()

    # Qt 6 handles DPI scaling automatically; this only affects how fractional
    # scale factors round, and PassThrough keeps the viewport pixel-exact.
    QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Only the program name: argparse already consumed our flags, and Qt
    # would otherwise try to interpret them as its own options.
    app = QtWidgets.QApplication([sys.argv[0]])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    apply_theme(app)

    config = PipelineConfig(
        inference_max_side=args.max_side,
        buffer_max_side=args.buffer_max_side,
        camera_fov_y=args.fov,
        depth_model=args.depth_model,
        normal_onnx=args.normal_onnx,
        albedo_onnx=args.albedo_onnx,
        neural_normals=not args.no_neural_normals,
    )

    window = MainWindow(config)
    window.show()

    logging.getLogger("imagelighting").info("Qt binding: %s", QT_BINDING)

    if args.image:
        QtCore.QTimer.singleShot(120, lambda: window.open_image(args.image))
    elif args.sample:
        QtCore.QTimer.singleShot(120, window.load_sample)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
