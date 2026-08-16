"""Dark creative-tool theme.

A single stylesheet string keeps the look consistent across both Qt
bindings without pulling in a theming dependency.  Colours are tuned for
long sessions next to a bright image viewport: low-chroma greys so nothing
competes with the render, and one warm accent for interactive state.
"""

from __future__ import annotations

from core.qt_compat import QtGui

# Palette
BG_DEEP = "#0e0e11"
BG_PANEL = "#16161a"
BG_RAISED = "#1d1d23"
BG_HOVER = "#26262e"
BORDER = "#2c2c35"
TEXT = "#d8d8de"
TEXT_DIM = "#8a8a95"
ACCENT = "#f0a04b"
ACCENT_DIM = "#8a5a26"
DANGER = "#e05252"

VIEWPORT_CLEAR = (0.055, 0.055, 0.067)


STYLESHEET = f"""
QWidget {{
    background-color: {BG_PANEL};
    color: {TEXT};
    font-family: "Segoe UI", "Inter", "SF Pro Text", sans-serif;
    font-size: 12px;
}}

QMainWindow, QDialog {{
    background-color: {BG_DEEP};
}}

QMenuBar {{
    background-color: {BG_DEEP};
    border-bottom: 1px solid {BORDER};
    padding: 2px;
}}
QMenuBar::item {{
    padding: 5px 11px;
    background: transparent;
    border-radius: 4px;
}}
QMenuBar::item:selected {{ background-color: {BG_HOVER}; }}

QMenu {{
    background-color: {BG_RAISED};
    border: 1px solid {BORDER};
    padding: 5px;
}}
QMenu::item {{ padding: 6px 26px 6px 20px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: {ACCENT_DIM}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 5px 8px; }}

QToolBar {{
    background-color: {BG_DEEP};
    border-bottom: 1px solid {BORDER};
    spacing: 6px;
    padding: 5px 8px;
}}

QStatusBar {{
    background-color: {BG_DEEP};
    border-top: 1px solid {BORDER};
    color: {TEXT_DIM};
}}
QStatusBar::item {{ border: none; }}

QGroupBox {{
    background-color: {BG_RAISED};
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 16px;
    padding: 10px 10px 8px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: {TEXT_DIM};
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

QPushButton {{
    background-color: {BG_HOVER};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 13px;
    color: {TEXT};
}}
QPushButton:hover {{ background-color: #30303a; border-color: #3d3d49; }}
QPushButton:pressed {{ background-color: {ACCENT_DIM}; }}
QPushButton:disabled {{ color: #55555f; background-color: #1a1a1f; }}
QPushButton:checked {{
    background-color: {ACCENT_DIM};
    border-color: {ACCENT};
    color: #ffffff;
}}

QPushButton#primary {{
    background-color: {ACCENT_DIM};
    border-color: {ACCENT};
    font-weight: 600;
}}
QPushButton#primary:hover {{ background-color: #a56b2e; }}
QPushButton#danger:hover {{ background-color: {DANGER}; border-color: {DANGER}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {BG_DEEP};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 7px;
    selection-background-color: {ACCENT_DIM};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {BG_RAISED};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_DIM};
    outline: none;
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {BG_DEEP};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT_DIM};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{ background: #ffbc6b; }}
QSlider::groove:horizontal:disabled {{ background: #17171b; }}
QSlider::sub-page:horizontal:disabled {{ background: #33333a; }}
QSlider::handle:horizontal:disabled {{ background: #4a4a54; }}

QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background-color: {BG_DEEP};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_DIM};
    padding: 7px 14px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}}
QTabBar::tab:hover {{ color: {TEXT}; background: {BG_RAISED}; }}
QTabBar::tab:selected {{
    color: {ACCENT};
    background: {BG_DEEP};
    border-color: {BORDER};
    border-bottom-color: {BG_DEEP};
}}

QListWidget {{
    background-color: {BG_DEEP};
    border: 1px solid {BORDER};
    border-radius: 5px;
    outline: none;
    padding: 3px;
}}
QListWidget::item {{ padding: 6px 8px; border-radius: 4px; }}
QListWidget::item:hover {{ background-color: {BG_HOVER}; }}
QListWidget::item:selected {{ background-color: {ACCENT_DIM}; color: #ffffff; }}

QCheckBox, QRadioButton {{ spacing: 7px; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid #43434f;
    border-radius: 3px;
    background: {BG_DEEP};
}}
QRadioButton::indicator {{ border-radius: 7px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #34343e;
    border-radius: 5px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: #45454f; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #34343e; border-radius: 5px; min-width: 28px; }}

QProgressBar {{
    background-color: {BG_DEEP};
    border: 1px solid {BORDER};
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 3px; }}

QSplitter::handle {{ background-color: {BG_DEEP}; }}
QSplitter::handle:horizontal {{ width: 4px; }}
QSplitter::handle:hover {{ background-color: {ACCENT_DIM}; }}

QToolTip {{
    background-color: {BG_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 5px 7px;
}}

QLabel#sectionHint {{ color: {TEXT_DIM}; font-size: 11px; }}
QLabel#valueLabel {{ color: {ACCENT}; font-size: 11px; }}
QLabel#metaLabel {{ color: {TEXT_DIM}; font-size: 11px; }}
"""


def apply_theme(app) -> None:
    """Install the dark palette and stylesheet on a QApplication."""
    app.setStyle("Fusion")

    palette = QtGui.QPalette()
    role = QtGui.QPalette.ColorRole
    palette.setColor(role.Window, QtGui.QColor(BG_PANEL))
    palette.setColor(role.WindowText, QtGui.QColor(TEXT))
    palette.setColor(role.Base, QtGui.QColor(BG_DEEP))
    palette.setColor(role.AlternateBase, QtGui.QColor(BG_RAISED))
    palette.setColor(role.Text, QtGui.QColor(TEXT))
    palette.setColor(role.Button, QtGui.QColor(BG_RAISED))
    palette.setColor(role.ButtonText, QtGui.QColor(TEXT))
    palette.setColor(role.Highlight, QtGui.QColor(ACCENT_DIM))
    palette.setColor(role.HighlightedText, QtGui.QColor("#ffffff"))
    palette.setColor(role.ToolTipBase, QtGui.QColor(BG_RAISED))
    palette.setColor(role.ToolTipText, QtGui.QColor(TEXT))
    app.setPalette(palette)

    app.setStyleSheet(STYLESHEET)
