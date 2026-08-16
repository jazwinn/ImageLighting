"""Qt binding shim.

The app targets PySide6 first (it exposes ``Signal``/``Slot`` and the
``QtOpenGLWidgets`` module) but runs unmodified on PyQt6.  Every UI module
imports Qt symbols from here rather than from a concrete binding so that a
single environment variable (``IMAGELIGHTING_QT=PyQt6``) can flip the whole
application over.
"""

from __future__ import annotations

import os

QT_BINDING: str

_BINDINGS = ("PySide6", "PyQt6")
_preferred = os.environ.get("IMAGELIGHTING_QT", "").strip()
# The preferred binding goes first, but the other stays in the list so a
# typo or a missing install degrades to whatever is actually present.
_order = ([_preferred] if _preferred in _BINDINGS else []) + [
    b for b in _BINDINGS if b != _preferred
]

_last_error: Exception | None = None
for _candidate in _order:
    try:
        if _candidate == "PySide6":
            from PySide6 import QtCore, QtGui, QtWidgets  # noqa: F401
            from PySide6.QtOpenGLWidgets import QOpenGLWidget  # noqa: F401

            Signal = QtCore.Signal
            Slot = QtCore.Slot
            Property = QtCore.Property
            QT_BINDING = "PySide6"
            break
        if _candidate == "PyQt6":
            from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: F401
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget  # noqa: F401

            Signal = QtCore.pyqtSignal
            Slot = QtCore.pyqtSlot
            Property = QtCore.pyqtProperty
            QT_BINDING = "PyQt6"
            break
    except ImportError as exc:  # pragma: no cover - environment dependent
        _last_error = exc
else:  # pragma: no cover - environment dependent
    raise ImportError(
        "Neither PySide6 nor PyQt6 could be imported. Install one of them:\n"
        "    pip install PySide6\n"
        f"Last import error: {_last_error}"
    )

Qt = QtCore.Qt

__all__ = [
    "QtCore",
    "QtGui",
    "QtWidgets",
    "QOpenGLWidget",
    "Qt",
    "Signal",
    "Slot",
    "Property",
    "QT_BINDING",
]
