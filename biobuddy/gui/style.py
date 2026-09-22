"""
Shared visual styling for the BioBuddy desktop GUI.

The editor is a scientific tool with dense forms and marker lists. The style below keeps the interface compact while
making sections, tabs, lists, and primary actions easier to scan.
"""

BIOBUDDY_GUI_STYLESHEET = """
QWidget {
    background: #f3f4f6;
    color: #111827;
    font-size: 12px;
}

QMainWindow, QDialog {
    background: #f3f4f6;
}

QLabel {
    background: transparent;
}

QLabel#WorkflowStatusLabel {
    color: #374151;
    font-weight: 600;
    padding: 4px 0;
}

QLabel#SectionTitleLabel {
    color: #0f172a;
    font-size: 13px;
    font-weight: 700;
    padding: 4px 0 2px 0;
}

QLabel#MutedInfoLabel {
    color: #475569;
    line-height: 125%;
}

QWidget#PreviewWidget {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
}

QScrollArea {
    background: transparent;
    border: none;
}

QWidget:disabled {
    color: #94a3b8;
}

QLineEdit, QTextEdit, QComboBox, QListWidget, QTreeWidget {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    padding: 4px;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}

QListWidget::item:selected, QTreeWidget::item:selected {
    background: #2563eb;
    color: #ffffff;
}

QListWidget::item:alternate:selected, QTreeWidget::item:alternate:selected {
    background: #2563eb;
    color: #ffffff;
}

QComboBox {
    padding-right: 18px;
}

QLineEdit:read-only {
    background: #f8fafc;
    color: #475569;
}

QLineEdit:disabled, QComboBox:disabled, QListWidget:disabled, QTreeWidget:disabled {
    background: #f1f5f9;
    color: #94a3b8;
    border-color: #e2e8f0;
}

QLineEdit#ProblemLineEdit {
    background: #fff1f2;
    border: 2px solid #dc2626;
    color: #7f1d1d;
}

QTextEdit {
    font-family: Menlo, Monaco, Consolas, monospace;
    font-size: 11px;
}

QListWidget::item, QTreeWidget::item {
    min-height: 20px;
    padding: 2px 4px;
}

QListWidget::item:alternate, QTreeWidget::item:alternate {
    background: #f8fafc;
}

QTabWidget::pane {
    border: 1px solid #d1d5db;
    border-radius: 6px;
    background: #ffffff;
    top: -1px;
}

QTabBar::tab {
    background: #ffffff;
    border: 1px solid #d1d5db;
    padding: 6px 12px;
    min-height: 20px;
}

QTabBar::tab:selected {
    background: #2563eb;
    color: #ffffff;
    border-color: #2563eb;
}

QTabBar::tab:first {
    border-top-left-radius: 6px;
    border-bottom-left-radius: 6px;
}

QTabBar::tab:last {
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}

QGroupBox {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 400;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #0f172a;
    font-weight: 700;
}

QPushButton {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 5px 12px;
    min-height: 22px;
}

QPushButton:hover {
    background: #f8fafc;
    border-color: #94a3b8;
}

QPushButton:pressed {
    background: #e5e7eb;
}

QPushButton:disabled {
    background: #f1f5f9;
    color: #94a3b8;
    border-color: #e2e8f0;
}

QPushButton#PrimaryActionButton {
    background: #2563eb;
    color: #ffffff;
    border-color: #2563eb;
    font-weight: 600;
}

QPushButton#PrimaryActionButton:hover {
    background: #1d4ed8;
}

QPushButton#SmallIconButton {
    min-width: 28px;
    max-width: 34px;
    padding: 4px 0;
    font-weight: 700;
}

QPushButton#AddIconButton {
    min-width: 28px;
    max-width: 34px;
    padding: 4px 0;
    font-weight: 700;
    color: #166534;
    background: #dcfce7;
    border-color: #86efac;
}

QPushButton#AddIconButton:hover {
    background: #bbf7d0;
    border-color: #22c55e;
}

QPushButton#RemoveIconButton {
    min-width: 28px;
    max-width: 34px;
    padding: 4px 0;
    font-weight: 700;
    color: #991b1b;
    background: #fee2e2;
    border-color: #fca5a5;
}

QPushButton#RemoveIconButton:hover {
    background: #fecaca;
    border-color: #ef4444;
}

QPushButton#SecondaryActionButton {
    color: #334155;
    background: #f8fafc;
}

QPushButton#SecondaryActionButton:hover {
    background: #eef2f7;
}

QPushButton#DangerActionButton {
    color: #991b1b;
    background: #fff1f2;
    border-color: #fca5a5;
    font-weight: 600;
}

QPushButton#DangerActionButton:hover {
    background: #fee2e2;
    border-color: #ef4444;
}

QCheckBox {
    background: transparent;
    spacing: 6px;
}

QSplitter::handle {
    background: #d1d5db;
}

QSplitter::handle:horizontal {
    width: 1px;
}

QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}

QScrollBar::handle:vertical {
    background: #cbd5e1;
    border-radius: 5px;
    min-height: 24px;
}

QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 2px;
}

QScrollBar::handle:horizontal {
    background: #cbd5e1;
    border-radius: 5px;
    min-width: 24px;
}

QScrollBar::handle:horizontal:hover {
    background: #94a3b8;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

QSlider::groove:horizontal {
    height: 5px;
    background: #d1d5db;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #2563eb;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
"""


def apply_biobuddy_gui_style(application) -> None:
    """
    Apply the shared GUI stylesheet once the QApplication exists.
    """
    application.setStyleSheet(BIOBUDDY_GUI_STYLESHEET)
