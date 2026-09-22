import json
import math
import re
import ast
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

try:
    from scipy.spatial.transform import Rotation as _ScipyRotation
except ImportError:  # pragma: no cover - SciPy is available in the GUI environment, but keep a tiny fallback.
    _ScipyRotation = None

from ..validation import MuscleValidator
from ..model_modifiers.functional_frame_selection import (
    FunctionalFrameSelectionOptions,
    format_functional_frame_report,
    prepare_functional_rt_pair,
    subset_points_by_frame,
)
from .segment_editor import (
    DE_LEVA_MODEL_NAME,
    YEADON_MODEL_NAME,
    SegmentEditorData,
    apply_segment_editor_data,
    available_inertial_models,
    build_inertial_parameters_from_model,
    get_segment_editor_data,
    inertial_model_segment_names,
    load_model,
    validate_parent_name,
)
from .marker_editor import (
    MarkerEditorData,
    add_marker,
    apply_marker_editor_data,
    attach_marker_to_segment,
    get_marker_editor_data,
    remove_marker,
)
from .muscle_editor import (
    MuscleEditorData,
    ViaPointEditorData,
    add_muscle,
    add_muscle_group,
    add_via_point,
    apply_insertion_editor_data,
    apply_muscle_editor_data,
    apply_origin_editor_data,
    apply_via_point_editor_data,
    get_insertion_editor_data,
    get_muscle_editor_data,
    get_origin_editor_data,
    get_via_point_editor_data,
    remove_via_point,
    remove_muscle,
    remove_muscle_group,
)
from .preview_scene import build_preview_scene
from .validation_panel import validate_model_for_editor
from .c3d_model_creation import (
    C3dModelCreationResult,
    C3dModelPreset,
    c3d_model_preset_from_cli_value,
    create_model_from_marker_data,
    create_model_from_c3d_folder,
    supported_c3d_model_presets,
    template_for_c3d_model_preset,
)
from .c3d_creation_workflow import (
    add_axis_to_draft,
    add_segment_to_draft,
    add_virtual_marker_to_draft,
    assign_c3d_file_role_to_draft,
    assign_markers_to_segment,
    c3d_creation_workflow,
    c3d_template_payload_from_draft,
    c3d_workflow_draft,
    c3d_workflow_progress,
    c3d_workflow_summary,
    clear_c3d_file_role_from_draft,
    _expected_marker_names_for_preset,
    remove_axis_from_draft,
    remove_segment_from_draft,
    remove_virtual_marker_from_draft,
    set_segment_marker_technical,
    unassign_markers_from_segment,
    update_segment_parent_in_draft,
    update_segment_settings_in_draft,
    validate_c3d_workflow_draft,
)
from .c3d_workflow_sources import (
    anatomical_axis_source_labels as _anatomical_axis_source_labels,
    axis_source_name_from_list_text as _axis_source_name_from_list_text,
    is_virtual_feature_axis as _is_virtual_feature_axis,
)
from .style import apply_biobuddy_gui_style
from ..utils.marker_data import C3dData

PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS = {
    "hara2016_hip": "Hara 2016 hip",
    "harrington2007_hip": "Harrington 2007 hip",
    "sobral2025_shoulder": "Sobral 2025 shoulder",
    "rab2002_shoulder": "Rab 2002 shoulder",
}

PREVIEW_AXIS_COLORS = {"x": "#dc2626", "y": "#16a34a", "z": "#2563eb"}
PREVIEW_NEUTRAL_COLOR = "#6b7280"
VIRTUAL_MARKER_METHOD_DISPLAY_LABELS = {"axis_projection": "projection on axis"}
SARA_DIRECTION_METHODS = {"sara", "sara_direction"}
WORKFLOW_PLAYBACK_DEFAULT_FPS = 30.0
DEFAULT_SARA_STATIC_AXIS_DEVIATION_LIMIT_DEGREES = 30.0
FUNCTIONAL_RECONSTRUCTION_QLD_MAX_FRAMES = 20


def _is_sara_direction_method(method: str) -> bool:
    return method in SARA_DIRECTION_METHODS


def _c3d_frame_rate(c3d_data) -> float | None:
    """
    Return a finite positive C3D marker frame rate when available.
    """
    if c3d_data is None:
        return None
    frame_rate = getattr(c3d_data, "frame_rate", None)
    if callable(frame_rate):
        frame_rate = frame_rate()
    try:
        frame_rate = float(frame_rate)
    except (TypeError, ValueError):
        return None
    return frame_rate if np.isfinite(frame_rate) and frame_rate > 0 else None


def _workflow_playback_fps(c3d_data) -> float:
    """
    Return the playback fps for a preview C3D, with a conservative fallback.
    """
    return _c3d_frame_rate(c3d_data) or WORKFLOW_PLAYBACK_DEFAULT_FPS


def _workflow_playback_timer_interval_ms(c3d_data) -> int:
    """
    Return the timer interval needed to follow the C3D frame rate.
    """
    return max(1, int(round(1000.0 / _workflow_playback_fps(c3d_data))))


def launch_model_editor(
    *,
    c3d_preset: str | C3dModelPreset | None = None,
    c3d_folder: str | Path | None = None,
    open_c3d_dialog: bool = False,
) -> None:
    """
    Launch the Qt desktop model editor.
    """
    try:
        from PySide6.QtCore import QPointF, Qt, QTimer
        from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
        from PySide6.QtWidgets import (
            QAbstractItemView,
            QApplication,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QCheckBox,
            QLabel,
            QLineEdit,
            QInputDialog,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMenu,
            QMessageBox,
            QProgressDialog,
            QPushButton,
            QScrollArea,
            QSplitter,
            QSlider,
            QStackedWidget,
            QTabWidget,
            QTextEdit,
            QTreeWidget,
            QTreeWidgetItem,
            QVBoxLayout,
            QWidget,
        )

        qt_alignment_center = Qt.AlignmentFlag.AlignCenter
        qt_horizontal = Qt.Orientation.Horizontal
        qt_match_exact = Qt.MatchFlag.MatchExactly
        qt_match_recursive = Qt.MatchFlag.MatchRecursive
        qt_dash_line = Qt.PenStyle.DashLine
        qt_dot_line = Qt.PenStyle.DotLine
        qt_extended_selection = QAbstractItemView.SelectionMode.ExtendedSelection
        qt_open_hand_cursor = Qt.CursorShape.OpenHandCursor
        qt_closed_hand_cursor = Qt.CursorShape.ClosedHandCursor
        qt_right_button = Qt.MouseButton.RightButton
        qt_shift_modifier = Qt.KeyboardModifier.ShiftModifier
        qt_control_modifier = Qt.KeyboardModifier.ControlModifier
        qt_meta_modifier = Qt.KeyboardModifier.MetaModifier
        qt_window_modal = Qt.WindowModality.WindowModal
        qpaint_antialiasing = QPainter.RenderHint.Antialiasing
        get_event_position = lambda event: event.position()
        get_event_global_position = lambda event: event.globalPosition().toPoint()
        get_wheel_delta = lambda event: event.angleDelta().y()
    except ImportError:
        try:
            from PyQt5.QtCore import QPointF, Qt, QTimer
            from PyQt5.QtGui import QColor, QPainter, QPen
            from PyQt5.QtWidgets import (
                QAbstractItemView,
                QApplication,
                QComboBox,
                QDialog,
                QDialogButtonBox,
                QFileDialog,
                QFormLayout,
                QGridLayout,
                QGroupBox,
                QHBoxLayout,
                QCheckBox,
                QLabel,
                QLineEdit,
                QInputDialog,
                QListWidget,
                QListWidgetItem,
                QMainWindow,
                QMenu,
                QMessageBox,
                QProgressDialog,
                QPushButton,
                QScrollArea,
                QSplitter,
                QSlider,
                QStackedWidget,
                QTabWidget,
                QTextEdit,
                QTreeWidget,
                QTreeWidgetItem,
                QVBoxLayout,
                QWidget,
            )

            qt_alignment_center = Qt.AlignCenter
            qt_horizontal = Qt.Horizontal
            qt_match_exact = Qt.MatchExactly
            qt_match_recursive = Qt.MatchRecursive
            qt_dash_line = Qt.DashLine
            qt_dot_line = Qt.DotLine
            qt_extended_selection = QAbstractItemView.ExtendedSelection
            qt_open_hand_cursor = Qt.OpenHandCursor
            qt_closed_hand_cursor = Qt.ClosedHandCursor
            qt_right_button = Qt.RightButton
            qt_shift_modifier = Qt.ShiftModifier
            qt_control_modifier = Qt.ControlModifier
            qt_meta_modifier = Qt.MetaModifier
            qt_window_modal = Qt.WindowModal
            qpaint_antialiasing = QPainter.Antialiasing
            get_event_position = lambda event: event.localPos()
            get_event_global_position = lambda event: event.globalPos()
            get_wheel_delta = lambda event: event.angleDelta().y()
        except ImportError as error:
            raise ImportError(
                "The model editor requires a working Qt binding. Install BioBuddy with `pip install biobuddy[gui]` "
                "or use an environment where PyQt5 is available."
            ) from error
    try:
        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
        except ImportError:
            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
        from matplotlib.figure import Figure
    except ImportError:
        FigureCanvas = None
        Figure = None
    qdialog_accepted = QDialog.DialogCode.Accepted if hasattr(QDialog, "DialogCode") else QDialog.Accepted
    qdialog_ok = (
        QDialogButtonBox.StandardButton.Ok if hasattr(QDialogButtonBox, "StandardButton") else QDialogButtonBox.Ok
    )
    qdialog_cancel = (
        QDialogButtonBox.StandardButton.Cancel
        if hasattr(QDialogButtonBox, "StandardButton")
        else QDialogButtonBox.Cancel
    )

    def _draw_legend_point(painter, center, color, label: str, text_x: int, text_y: int) -> None:
        """
        Draw one point-style legend entry.
        """
        painter.setPen(QPen(color, 1))
        painter.setBrush(color)
        painter.drawEllipse(center, 4, 4)
        painter.setPen(QPen(QColor("#111827"), 1))
        painter.drawText(text_x, text_y, label)

    def _set_preview_label_font(painter) -> None:
        """
        Use a compact font for dense marker labels in 3D previews.
        """
        font = painter.font()
        if font.pointSize() > 0:
            font.setPointSize(max(7, font.pointSize() - 3))
        else:
            font.setPixelSize(10)
        painter.setFont(font)

    def _preview_axis_color(axis_name: str) -> str:
        """
        Return the standard RGB axis color, or a neutral fallback.
        """
        return PREVIEW_AXIS_COLORS.get(axis_name, PREVIEW_NEUTRAL_COLOR)

    def _style_axis_combo(combo) -> None:
        """
        Tint an axis combo to match the RGB axis convention.
        """
        axis_name = combo.currentText().strip().lower()
        styles = {
            "x": ("#dc2626", "#fee2e2"),
            "y": ("#16a34a", "#dcfce7"),
            "z": ("#2563eb", "#dbeafe"),
        }
        border, background = styles.get(axis_name, ("#6b7280", "#f8fafc"))
        combo.setStyleSheet(
            "QComboBox {"
            f"background-color: {background};"
            f"border: 1px solid {border};"
            "border-radius: 4px;"
            "padding: 3px 8px;"
            "}"
        )

    def _construction_axis_pen(axis_name: str, keep_vector: bool):
        """
        Use a subtle dotted pen for marker-to-marker construction vectors.
        """
        color = QColor(_preview_axis_color(axis_name))
        color.setAlpha(175 if keep_vector else 135)
        pen = QPen(color, 2 if keep_vector else 1)
        pen.setStyle(qt_dot_line)
        return pen

    def _small_button(text: str) -> QPushButton:
        """
        Create one compact button used around marker transfer lists.
        """
        button = QPushButton(text)
        if text == "+":
            button.setObjectName("AddIconButton")
        elif text == "-":
            button.setObjectName("RemoveIconButton")
        else:
            button.setObjectName("SmallIconButton")
        button.setMaximumWidth(34)
        button.setMinimumWidth(34)
        return button

    class FunctionalFrameRangeBar(QWidget):
        """
        Lightweight frame-interval paint selector for functional trials.
        """

        def __init__(self):
            super().__init__()
            self.setMinimumHeight(18)
            self.setMaximumHeight(22)
            self.setToolTip(
                "Drag on the timeline to add or remove functional frames. Enable 'Use selected frames' to use the blue zones for SCoRE/SARA calculations."
            )
            self.frame_count = 0
            self.start_frame = 0
            self.end_frame = 0
            self.selection_ranges = ()
            self._dragging = False
            self._drag_anchor = 0
            self._drag_base_ranges = ()
            self._drag_selecting = True
            self.on_frame_dragged = None
            self.on_selection_changed = None

        def set_frame_count(self, frame_count: int) -> None:
            frame_count = max(0, int(frame_count))
            if frame_count == self.frame_count:
                return
            self.frame_count = frame_count
            self.setEnabled(frame_count > 1)
            self.start_frame = 0
            self.end_frame = max(frame_count - 1, 0)
            self.selection_ranges = ((0, frame_count - 1),) if frame_count > 0 else ()
            self.update()

        def selected_indices(self) -> tuple[int, ...]:
            if self.frame_count <= 0:
                return ()
            return tuple(index for start, end in self.selection_ranges for index in range(start, end + 1))

        def set_selection(self, start_frame: int, end_frame: int, *, notify: bool = True) -> None:
            if self.frame_count <= 0:
                return
            start_frame = max(0, min(int(start_frame), self.frame_count - 1))
            end_frame = max(0, min(int(end_frame), self.frame_count - 1))
            self.start_frame = start_frame
            self.end_frame = end_frame
            self._set_selection_ranges(
                self._normalized_ranges(((start_frame, end_frame),)),
                notify=notify,
            )

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            _set_preview_render_hints(self, painter)
            track_rect = self.rect().adjusted(8, 5, -8, -5)
            painter.setPen(QPen(QColor("#cbd5e1"), 1))
            painter.setBrush(QColor("#e2e8f0"))
            painter.drawRect(track_rect)
            if self.frame_count <= 1:
                return
            painter.setPen(QPen(QColor("#2563eb"), 1))
            painter.setBrush(QColor("#2563eb"))
            for start_frame, end_frame in self.selection_ranges:
                left_x = self._x_from_frame(start_frame)
                right_x = self._x_from_frame(end_frame + 1)
                painter.drawRect(
                    left_x,
                    track_rect.top(),
                    max(1, right_x - left_x),
                    track_rect.height(),
                )

        def mousePressEvent(self, event) -> None:
            if self.frame_count <= 1:
                return
            frame = self._frame_from_x(get_event_position(event).x())
            self._dragging = True
            self._drag_anchor = frame
            self._drag_base_ranges = self.selection_ranges
            self._drag_selecting = not self._frame_is_selected(frame)
            self._set_drag_selection(frame, notify=False)
            self._notify_drag_frame(frame)
            event.accept()

        def mouseMoveEvent(self, event) -> None:
            if not self._dragging or self.frame_count <= 1:
                return
            frame = self._frame_from_x(get_event_position(event).x())
            self._set_drag_selection(frame, notify=False)
            self._notify_drag_frame(frame)
            event.accept()

        def mouseReleaseEvent(self, event) -> None:
            if not self._dragging:
                return
            frame = self._frame_from_x(get_event_position(event).x())
            self._set_drag_selection(frame, notify=True)
            self._notify_drag_frame(frame)
            self._dragging = False
            event.accept()

        def mouseDoubleClickEvent(self, event) -> None:
            if self.frame_count <= 1:
                return
            self.set_selection(0, self.frame_count - 1, notify=True)
            event.accept()

        def _set_drag_selection(self, frame: int, *, notify: bool) -> None:
            self.start_frame = self._drag_anchor
            self.end_frame = frame
            self._set_selection_ranges(
                self._ranges_after_paint(
                    self._drag_base_ranges,
                    min(self._drag_anchor, frame),
                    max(self._drag_anchor, frame),
                    self._drag_selecting,
                ),
                notify=notify,
            )

        def _set_selection_ranges(self, ranges: tuple[tuple[int, int], ...], *, notify: bool) -> None:
            normalized_ranges = self._normalized_ranges(ranges)
            if len(normalized_ranges) == 0 and self.frame_count > 0:
                normalized_ranges = self.selection_ranges or ((0, self.frame_count - 1),)
            changed = normalized_ranges != self.selection_ranges
            self.selection_ranges = normalized_ranges
            self.update()
            if changed and notify and self.on_selection_changed is not None:
                self.on_selection_changed()

        def _normalized_ranges(self, ranges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
            normalized = []
            for start_frame, end_frame in ranges:
                if self.frame_count <= 0:
                    continue
                start = max(0, min(int(start_frame), int(end_frame)))
                end = min(self.frame_count - 1, max(int(start_frame), int(end_frame)))
                normalized.append((start, end))
            normalized.sort()
            merged = []
            for start, end in normalized:
                if len(merged) == 0 or start > merged[-1][1] + 1:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
            return tuple((start, end) for start, end in merged)

        def _ranges_after_paint(
            self,
            ranges: tuple[tuple[int, int], ...],
            start_frame: int,
            end_frame: int,
            selecting: bool,
        ) -> tuple[tuple[int, int], ...]:
            if selecting:
                return self._normalized_ranges(ranges + ((start_frame, end_frame),))
            updated_ranges = []
            for start, end in self._normalized_ranges(ranges):
                if end < start_frame or start > end_frame:
                    updated_ranges.append((start, end))
                    continue
                if start < start_frame:
                    updated_ranges.append((start, start_frame - 1))
                if end > end_frame:
                    updated_ranges.append((end_frame + 1, end))
            return self._normalized_ranges(tuple(updated_ranges))

        def _frame_is_selected(self, frame: int) -> bool:
            return any(start <= frame <= end for start, end in self.selection_ranges)

        def _notify_drag_frame(self, frame: int) -> None:
            if self.on_frame_dragged is not None:
                self.on_frame_dragged(frame)

        def _x_from_frame(self, frame: int) -> int:
            rect = self.rect().adjusted(8, 5, -8, -5)
            if self.frame_count <= 1 or rect.width() <= 0:
                return rect.left()
            ratio = max(0.0, min(float(frame) / float(self.frame_count), 1.0))
            return int(round(rect.left() + ratio * rect.width()))

        def _frame_from_x(self, x_position: float) -> int:
            rect = self.rect().adjusted(8, 5, -8, -5)
            if self.frame_count <= 1 or rect.width() <= 0:
                return 0
            ratio = (float(x_position) - rect.left()) / float(rect.width())
            ratio = max(0.0, min(ratio, 1.0))
            return max(0, min(int(ratio * self.frame_count), self.frame_count - 1))

    def _marker_list_widget(
        *,
        min_height: int | None = None,
        max_height: int | None = None,
        min_width: int | None = None,
        max_width: int | None = None,
    ) -> QListWidget:
        """
        Create a multi-selection marker list with optional compact dimensions.
        """
        marker_list = QListWidget()
        marker_list.setSelectionMode(qt_extended_selection)
        marker_list.setAlternatingRowColors(True)
        if min_height is not None:
            marker_list.setMinimumHeight(min_height)
        if max_height is not None:
            marker_list.setMaximumHeight(max_height)
        if min_width is not None:
            marker_list.setMinimumWidth(min_width)
        if max_width is not None:
            marker_list.setMaximumWidth(max_width)
        return marker_list

    def _section_label(text: str) -> QLabel:
        """
        Create a consistent label for section headers inside dense workflow tabs.
        """
        label = QLabel(text)
        label.setObjectName("SectionTitleLabel")
        return label

    def _muted_label(text: str = "") -> QLabel:
        """
        Create a lower-emphasis label for helper text and summaries.
        """
        label = QLabel(text)
        label.setObjectName("MutedInfoLabel")
        label.setWordWrap(True)
        return label

    def _workflow_step_item(step_status) -> QListWidgetItem:
        """
        Create a colored workflow-progress item.
        """
        item = QListWidgetItem(
            f"{step_status.number}. [{step_status.status}] {step_status.name} - {step_status.detail}"
        )
        if step_status.status == "warning":
            item.setForeground(QColor("#92400e"))
            item.setBackground(QColor("#fef3c7"))
        elif step_status.status == "error":
            item.setForeground(QColor("#991b1b"))
            item.setBackground(QColor("#fee2e2"))
        elif step_status.status == "done":
            item.setForeground(QColor("#166534"))
            item.setBackground(QColor("#dcfce7"))
        return item

    def _configure_panel_layout(layout, *, margin: int = 14, spacing: int = 10) -> None:
        """
        Apply consistent spacing to manually assembled Qt layouts.
        """
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(spacing)

    def _style_preview_widget(widget) -> None:
        """
        Style custom paint widgets as explicit preview panels.
        """
        widget.setObjectName("PreviewWidget")

    def _layout_group(title: str, layout) -> QGroupBox:
        """
        Wrap an existing layout in a titled group box.
        """
        group = QGroupBox(title)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        group.setLayout(layout)
        return group

    def _list_with_side_buttons(title: str, item_list, add_button, remove_button):
        """
        Build a compact selector with plus/minus buttons aligned beside the list.
        """
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)
        layout.addWidget(QLabel(title), 0, 0, 1, 2)
        button_column = QVBoxLayout()
        button_column.setContentsMargins(0, 0, 0, 0)
        button_column.setSpacing(6)
        button_column.addStretch(1)
        button_column.addWidget(add_button)
        button_column.addWidget(remove_button)
        button_column.addStretch(1)
        layout.addLayout(button_column, 1, 0)
        layout.addWidget(item_list, 1, 1)
        layout.setColumnStretch(1, 1)
        return layout

    class FunctionalReconstructionPlotWidget(QWidget):
        """
        Matplotlib panel for knee rotation DoF reconstructed from a functional trial.
        """

        def __init__(self):
            super().__init__()
            self.setMinimumHeight(260)
            layout = QVBoxLayout(self)
            _configure_panel_layout(layout, margin=0, spacing=0)
            if FigureCanvas is None or Figure is None:
                self.figure = None
                self.canvas = None
                self.fallback_label = _muted_label("Matplotlib is not available; rotation DoF figure cannot be shown.")
                self.fallback_label.setAlignment(qt_alignment_center)
                layout.addWidget(self.fallback_label)
                return
            self.figure = Figure(figsize=(5.0, 2.7), tight_layout=True)
            self.canvas = FigureCanvas(self.figure)
            self.fallback_label = None
            layout.addWidget(self.canvas)
            self.clear()

        def clear(
            self,
            message: str = "Run a reconstruction diagnostic to plot knee rotations.",
        ) -> None:
            if self.canvas is None or self.figure is None:
                if self.fallback_label is not None:
                    self.fallback_label.setText(message)
                return
            self.figure.clear()
            axes = self.figure.add_subplot(111)
            axes.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                transform=axes.transAxes,
                color="#64748b",
            )
            axes.set_axis_off()
            self.canvas.draw_idle()

        def set_rotation_series(self, plot_data: dict[str, object] | None) -> None:
            if not plot_data:
                self.clear()
                return
            if self.canvas is None or self.figure is None:
                if self.fallback_label is not None:
                    self.fallback_label.setText("Matplotlib is not available; rotation DoF figure cannot be shown.")
                return
            time = np.asarray(plot_data.get("time", ()), dtype=float)
            series = tuple(plot_data.get("series", ()))
            if time.size == 0 or len(series) == 0:
                self.clear()
                return
            self.figure.clear()
            axes = self.figure.add_subplot(111)
            colors = ("#dc2626", "#16a34a", "#2563eb")
            for series_index, series_item in enumerate(series):
                if len(series_item) == 3:
                    label, values, line_style = series_item
                else:
                    label, values = series_item
                    line_style = "-"
                values = np.asarray(values, dtype=float)
                if values.size != time.size:
                    continue
                axes.plot(
                    time,
                    values,
                    color=colors[series_index % len(colors)],
                    linewidth=1.8,
                    linestyle=line_style,
                    label=label,
                )
            axes.axhline(0.0, color="#94a3b8", linewidth=0.8)
            axes.set_title(str(plot_data.get("title", "Rotation DoF")))
            axes.set_xlabel("Time (s)")
            axes.set_ylabel("Rotation (deg)")
            axes.grid(True, color="#e2e8f0", linewidth=0.8)
            axes.legend(loc="best", frameon=False)
            self.canvas.draw_idle()

    def _draw_preview_marker(
        painter,
        center,
        color,
        *,
        is_square: bool,
        size: int,
        pen_width: int = 1,
        marker_shape: str | None = None,
    ) -> None:
        """
        Draw one preview marker with the shared square/circle convention.
        """
        if not _is_finite_qpoint(center):
            return
        painter.setPen(QPen(color, pen_width))
        painter.setBrush(color)
        shape = marker_shape or ("square" if is_square else "circle")
        if shape == "diamond":
            x = center.x()
            y = center.y()
            painter.drawPolygon(
                QPolygonF(
                    (
                        QPointF(x, y - size),
                        QPointF(x + size, y),
                        QPointF(x, y + size),
                        QPointF(x - size, y),
                    )
                )
            )
        elif shape == "square":
            painter.drawRect(int(center.x()) - size // 2, int(center.y()) - size // 2, size, size)
        else:
            painter.drawEllipse(center, size, size)

    def _draw_legend_line(
        painter,
        color,
        width: int,
        start_x: int,
        y: int,
        label: str,
        text_x: int,
        text_y: int,
    ) -> None:
        """
        Draw one line-style legend entry.
        """
        painter.setPen(QPen(color, width))
        painter.drawLine(start_x, y, start_x + 24, y)
        painter.setPen(QPen(QColor("#111827"), 1))
        painter.drawText(text_x, text_y, label)

    def _draw_preview_orientation_axes(painter, width: int, height: int, yaw: float, pitch: float) -> None:
        """
        Draw a small RGB orientation triad in a 3D preview corner.
        """
        origin = QPointF(max(width - 92, 34), max(height - 64, 44))
        length = 34.0
        axes = (
            ("x", (1.0, 0.0, 0.0), QColor(PREVIEW_AXIS_COLORS["x"])),
            ("y", (0.0, 1.0, 0.0), QColor(PREVIEW_AXIS_COLORS["y"])),
            ("z", (0.0, 0.0, 1.0), QColor(PREVIEW_AXIS_COLORS["z"])),
        )
        painter.setBrush(QColor(255, 255, 255, 210))
        painter.setPen(QPen(QColor("#e5e7eb"), 1))
        painter.drawRect(int(origin.x()) - 20, int(origin.y()) - 42, 92, 74)
        for label, vector, color in axes:
            projected = _rotate_preview_point(vector, yaw, pitch)
            norm = math.hypot(projected[0], projected[1])
            if norm == 0:
                continue
            endpoint = QPointF(
                origin.x() + length * projected[0] / norm,
                origin.y() - length * projected[1] / norm,
            )
            painter.setPen(QPen(color, 3))
            painter.drawLine(origin, endpoint)
            painter.setPen(QPen(color, 1))
            painter.drawText(endpoint.x() + 3, endpoint.y() - 3, label)

    def _draw_preview_interaction_hint(painter, width: int) -> None:
        """
        Draw a compact reminder of the interactive 3D preview controls.
        """
        text = "Drag rotate | Wheel zoom | Double-click reset"
        hint_width = 250
        hint_x = max(width - hint_width - 10, 10)
        painter.setPen(QPen(QColor("#64748b"), 1))
        painter.setBrush(QColor(255, 255, 255, 220))
        painter.drawRect(hint_x, 8, hint_width, 22)
        painter.drawText(hint_x + 8, 24, text)

    def _scrollable_widget(widget):
        """
        Wrap a form-heavy tab so all controls remain reachable on smaller screens.
        """
        scroll_area = QScrollArea()
        scroll_area.setWidget(widget)
        scroll_area.setWidgetResizable(True)
        return scroll_area

    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    class InertialModelDialog(QDialog):
        """
        Modal editor for the model-specific parameters needed to compute segment inertia.
        """

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Apply inertial model")
            self.model_parameters = {}

            layout = QVBoxLayout(self)
            form = QFormLayout()
            self.model_name = QComboBox()
            self.model_name.addItems(available_inertial_models())
            self.source_segment_name = QComboBox()
            form.addRow("Model", self.model_name)
            form.addRow("Source segment", self.source_segment_name)
            layout.addLayout(form)

            self.parameter_widget = QWidget()
            self.parameter_layout = QFormLayout(self.parameter_widget)
            layout.addWidget(self.parameter_widget)

            buttons = QDialogButtonBox(qdialog_ok | qdialog_cancel)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

            self.model_name.currentTextChanged.connect(self._refresh_model_parameters)
            self._refresh_model_parameters(self.model_name.currentText())

        def _refresh_model_parameters(self, model_name: str) -> None:
            self.source_segment_name.clear()
            self.source_segment_name.addItems(inertial_model_segment_names(model_name))
            _clear_layout(self.parameter_layout)
            self.model_parameters = {}

            if model_name == DE_LEVA_MODEL_NAME:
                self.model_parameters["total_mass"] = QLineEdit()
                self.model_parameters["total_height"] = QLineEdit()
                self.model_parameters["sex"] = QComboBox()
                self.model_parameters["sex"].addItems(["male", "female"])
                self.parameter_layout.addRow("Total mass (kg)", self.model_parameters["total_mass"])
                self.parameter_layout.addRow("Total height (m)", self.model_parameters["total_height"])
                self.parameter_layout.addRow("Sex", self.model_parameters["sex"])
                return

            if model_name == YEADON_MODEL_NAME:
                self.model_parameters["measurements"] = QLineEdit()
                browse_button = QPushButton("Browse")
                browse_button.clicked.connect(self._browse_yeadon_measurements)
                path_layout = QHBoxLayout()
                path_layout.addWidget(self.model_parameters["measurements"])
                path_layout.addWidget(browse_button)
                path_widget = QWidget()
                path_widget.setLayout(path_layout)
                self.model_parameters["total_mass"] = QLineEdit()
                self.model_parameters["density_set"] = QComboBox()
                self.model_parameters["density_set"].addItems(["Chandler", "Clauser", "Dempster"])
                self.model_parameters["density_set"].setCurrentText("Dempster")
                self.model_parameters["symmetric"] = QCheckBox()
                self.model_parameters["symmetric"].setChecked(True)
                self.parameter_layout.addRow("Measurements file", path_widget)
                self.parameter_layout.addRow("Total mass (kg)", self.model_parameters["total_mass"])
                self.parameter_layout.addRow("Density set", self.model_parameters["density_set"])
                self.parameter_layout.addRow("Symmetric", self.model_parameters["symmetric"])

        def _browse_yeadon_measurements(self) -> None:
            filepath, _ = QFileDialog.getOpenFileName(
                self,
                "Open Yeadon measurements",
                "",
                "Yeadon measurement files (*.txt *.yaml *.yml *.json);;All files (*)",
            )
            if filepath:
                self.model_parameters["measurements"].setText(filepath)

        def parameters(self) -> tuple[str, str, dict[str, object]]:
            model_name = self.model_name.currentText()
            parameters = {}
            for name, widget in self.model_parameters.items():
                if isinstance(widget, QLineEdit):
                    parameters[name] = widget.text().strip()
                elif isinstance(widget, QComboBox):
                    parameters[name] = widget.currentText()
                elif isinstance(widget, QCheckBox):
                    parameters[name] = widget.isChecked()
            return model_name, self.source_segment_name.currentText(), parameters

    def _resize_window_to_available_screen(window, application, preferred_width: int, preferred_height: int) -> None:
        """
        Resize a top-level window so it fits inside the usable area of the screen where it is displayed.
        """
        screen = window.screen() or application.primaryScreen()
        if screen is None:
            window.resize(preferred_width, preferred_height)
            return
        available_geometry = screen.availableGeometry()
        width = min(preferred_width, max(900, int(available_geometry.width() * 0.94)))
        height = min(preferred_height, max(650, int(available_geometry.height() * 0.9)))
        window.resize(width, height)
        window.move(
            available_geometry.x() + max((available_geometry.width() - width) // 2, 0),
            available_geometry.y() + max((available_geometry.height() - height) // 2, 0),
        )

    class _CallbackSignal:
        """
        Minimal signal-like helper used by small composite widgets.
        """

        def __init__(self):
            self._callbacks = []

        def connect(self, callback) -> None:
            self._callbacks.append(callback)

        def emit(self, value: str) -> None:
            for callback in self._callbacks:
                callback(value)

    class _AxisSequencePicker(QWidget):
        """
        Three popup menus that produce a compact axis sequence such as ``xyz`` or ``x``.
        """

        _axes = ("x", "y", "z")
        _empty_label = "-"

        def __init__(self, *, allow_first_third_repeat: bool = False):
            super().__init__()
            self.allow_first_third_repeat = allow_first_third_repeat
            self.textChanged = _CallbackSignal()
            self._syncing = False
            self._combos = [QComboBox(), QComboBox(), QComboBox()]
            layout = QHBoxLayout(self)
            _configure_panel_layout(layout, margin=0, spacing=4)
            for combo in self._combos:
                combo.setMinimumWidth(54)
                combo.setMaximumWidth(64)
                combo.currentTextChanged.connect(self._on_combo_changed)
                layout.addWidget(combo)
            layout.addStretch()
            self.setText("")

        def text(self) -> str:
            sequence = "".join(
                combo.currentText().lower() for combo in self._combos if combo.currentText().lower() in self._axes
            )
            return sequence

        def setText(self, value: str) -> None:
            sequence = [axis for axis in str(value).lower() if axis in self._axes]
            if not self.allow_first_third_repeat:
                deduplicated = []
                for axis in sequence:
                    if axis not in deduplicated:
                        deduplicated.append(axis)
                sequence = deduplicated
            sequence = sequence[:3]
            self._syncing = True
            for index, combo in enumerate(self._combos):
                current = sequence[index] if index < len(sequence) else self._empty_label
                self._set_combo_items(combo, self._items_for_index(index, sequence), current)
            self._syncing = False
            self._refresh_items()
            self.textChanged.emit(self.text())

        def _on_combo_changed(self, _value: str) -> None:
            if self._syncing:
                return
            self._refresh_items()
            self.textChanged.emit(self.text())

        def _refresh_items(self) -> None:
            values = [combo.currentText().lower() for combo in self._combos]
            self._syncing = True
            for index, combo in enumerate(self._combos):
                current = values[index] if values[index] in self._axes else self._empty_label
                self._set_combo_items(combo, self._items_for_index(index, values), current)
            self._syncing = False

        def _items_for_index(self, index: int, values: list[str]) -> list[str]:
            values = (values + [self._empty_label] * len(self._combos))[: len(self._combos)]
            if self.allow_first_third_repeat:
                blocked = {values[1]} if index in {0, 2} and values[1] in self._axes else set()
                if index == 1:
                    blocked = {axis for axis in (values[0], values[2]) if axis in self._axes}
            else:
                blocked = {axis for i, axis in enumerate(values) if i != index and axis in self._axes}
            current = values[index] if index < len(values) and values[index] in self._axes else None
            items = [self._empty_label]
            items.extend(axis for axis in self._axes if axis == current or axis not in blocked)
            return items

        @staticmethod
        def _set_combo_items(combo, items: list[str], current: str) -> None:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            combo.setCurrentText(current if current in items else "-")
            combo.blockSignals(False)

    def _dialog_accepted_value() -> int:
        """
        Return the Qt accepted dialog value for PySide6 and PyQt5.
        """
        return QDialog.DialogCode.Accepted if hasattr(QDialog, "DialogCode") else QDialog.Accepted

    def _dialog_button(button_name: str):
        """
        Return a QDialogButtonBox standard button for PySide6 and PyQt5.
        """
        if hasattr(QDialogButtonBox, "StandardButton"):
            return getattr(QDialogButtonBox.StandardButton, button_name)
        return getattr(QDialogButtonBox, button_name)

    def _exec_dialog(dialog) -> int:
        """
        Execute a dialog across PySide6 and PyQt5.
        """
        return dialog.exec() if hasattr(dialog, "exec") else dialog.exec_()

    def _exec_menu(menu, position):
        """
        Execute a context menu across PySide6 and PyQt5.
        """
        return menu.exec(position) if hasattr(menu, "exec") else menu.exec_(position)

    def _default_preview_camera_matrix() -> np.ndarray:
        """
        Return the default world-to-camera matrix used by lightweight 3D previews.
        """
        return _legacy_preview_camera_matrix(yaw=-0.6, pitch=0.35)

    def _legacy_preview_camera_matrix(yaw: float, pitch: float) -> np.ndarray:
        """
        Convert the former yaw/pitch camera into an explicit 3D rotation matrix.
        """
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        return np.asarray(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_pitch * sin_yaw, sin_pitch * cos_yaw, cos_pitch],
                [cos_pitch * sin_yaw, cos_pitch * cos_yaw, -sin_pitch],
            ],
            dtype=float,
        )

    def _rotation_matrix_from_rotvec(rotation_vector: np.ndarray) -> np.ndarray:
        """
        Build a rotation matrix from a rotation vector using SciPy when available.
        """
        if _ScipyRotation is not None:
            return _ScipyRotation.from_rotvec(rotation_vector).as_matrix()
        angle = float(np.linalg.norm(rotation_vector))
        if angle < 1e-12:
            return np.eye(3)
        axis = rotation_vector / angle
        skew = np.asarray(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=float,
        )
        return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)

    def _initialize_preview_camera(widget) -> None:
        """
        Initialize the shared orbit-camera state used by lightweight 3D previews.
        """
        widget.yaw = _default_preview_camera_matrix()
        widget.pitch = 0.0
        widget.zoom = 1.0
        widget._last_mouse_position = None
        widget._is_preview_dragging = False
        widget._preview_update_pending = False
        widget._preview_update_timer = QTimer(widget)
        widget._preview_update_timer.setSingleShot(True)
        widget._preview_update_timer.timeout.connect(lambda: _flush_preview_camera_update(widget))
        widget.setMouseTracking(True)
        widget.setCursor(qt_open_hand_cursor)
        widget.setToolTip("Drag to rotate, use the mouse wheel to zoom, and double-click to reset the view.")

    def _request_preview_camera_update(widget) -> None:
        """
        Limit camera-driven redraws to roughly one display frame.
        """
        if getattr(widget, "_preview_update_pending", False):
            return
        widget._preview_update_pending = True
        widget._preview_update_timer.start(33)

    def _flush_preview_camera_update(widget) -> None:
        """
        Redraw a preview after the camera throttling timer fires.
        """
        widget._preview_update_pending = False
        widget.update()

    def _set_preview_render_hints(widget, painter) -> None:
        """
        Use cheaper rendering while dragging, then restore antialiasing when the mouse is released.
        """
        painter.setRenderHint(qpaint_antialiasing, not getattr(widget, "_is_preview_dragging", False))

    def _should_draw_preview_labels(widget) -> bool:
        """
        Hide dense text labels during orbit-camera drags to keep interaction responsive.
        """
        return not getattr(widget, "_is_preview_dragging", False)

    def _functional_frame_selection_options(
        enabled: bool,
        manual_frame_indices: tuple[int, ...] = (),
    ) -> FunctionalFrameSelectionOptions:
        """
        Return the default diverse-frame selection options used by the C3D workflow.
        """
        return FunctionalFrameSelectionOptions(
            enabled=enabled and len(manual_frame_indices) == 0,
            manual_frame_indices=tuple(int(index) for index in manual_frame_indices),
        )

    def _uses_selected_functional_frames(report) -> bool:
        options = getattr(report, "options", None)
        if options is None:
            return False
        return bool(getattr(options, "enabled", False)) or len(getattr(options, "manual_frame_indices", ())) != 0

    def _reset_preview_camera(widget) -> None:
        """
        Reset the lightweight 3D preview camera.
        """
        widget.yaw = _default_preview_camera_matrix()
        widget.pitch = 0.0
        widget.zoom = 1.0
        widget._last_mouse_position = None
        widget._is_preview_dragging = False
        widget.setCursor(qt_open_hand_cursor)
        widget.update()

    def _start_preview_camera_drag(widget, event) -> None:
        widget._last_mouse_position = get_event_position(event)
        widget._is_preview_dragging = True
        widget.setCursor(qt_closed_hand_cursor)

    def _drag_preview_camera(widget, event) -> None:
        if not getattr(widget, "_is_preview_dragging", False):
            return
        if widget._last_mouse_position is None:
            _start_preview_camera_drag(widget, event)
            return
        position = get_event_position(event)
        delta_x = position.x() - widget._last_mouse_position.x()
        delta_y = position.y() - widget._last_mouse_position.y()
        rotation_vector = np.asarray([delta_y, delta_x, 0.0], dtype=float) * 0.006
        widget.yaw = _rotation_matrix_from_rotvec(rotation_vector) @ np.asarray(widget.yaw, dtype=float)
        widget._last_mouse_position = position
        _request_preview_camera_update(widget)

    def _end_preview_camera_drag(widget) -> None:
        widget._last_mouse_position = None
        widget._is_preview_dragging = False
        widget.setCursor(qt_open_hand_cursor)
        widget.update()

    def _zoom_preview_camera(widget, event) -> None:
        delta = get_wheel_delta(event)
        if delta == 0:
            return
        widget.zoom = max(0.35, min(8.0, widget.zoom * (1.12 ** (delta / 120.0))))
        event.accept()
        _request_preview_camera_update(widget)

    def _create_axis_vector_controls(index: int) -> dict[str, object]:
        """
        Create the repeated controls used to define one anatomical frame vector.
        """
        start_list = _marker_list_widget(min_height=56, max_height=80, max_width=190)
        end_list = _marker_list_widget(min_height=56, max_height=80, max_width=190)
        start_label = QLabel("Start markers or axis")
        end_label = QLabel("End markers")
        axis_combo = QComboBox()
        axis_combo.setMaximumWidth(160)
        axis_combo.addItems(["x", "y", "z"])
        if index == 1:
            axis_combo.setCurrentText("y")
        _style_axis_combo(axis_combo)
        keep_checkbox = QCheckBox("Keep this vector")
        keep_checkbox.setChecked(index == 0)
        add_start_button = _small_button("+")
        add_end_button = _small_button("+")
        remove_start_button = _small_button("-")
        remove_end_button = _small_button("-")
        swap_button = _small_button("<->")
        swap_button.setToolTip("Swap start and end markers")
        return {
            "start_list": start_list,
            "end_list": end_list,
            "start_label": start_label,
            "end_label": end_label,
            "axis_combo": axis_combo,
            "keep_checkbox": keep_checkbox,
            "add_start_button": add_start_button,
            "add_end_button": add_end_button,
            "remove_start_button": remove_start_button,
            "remove_end_button": remove_end_button,
            "swap_button": swap_button,
        }

    def _axis_vector_layout(index: int, controls: dict[str, object]):
        """
        Build the layout for one repeated anatomical frame vector.
        """
        group = QGroupBox(f"Vector {index + 1}: mean(start markers) -> mean(end markers)")
        group.setMaximumWidth(520)
        layout = QVBoxLayout(group)
        marker_row = QHBoxLayout()
        for title, marker_list, add_button, remove_button in (
            (
                controls["start_label"],
                controls["start_list"],
                controls["add_start_button"],
                controls["remove_start_button"],
            ),
            (
                controls["end_label"],
                controls["end_list"],
                controls["add_end_button"],
                controls["remove_end_button"],
            ),
        ):
            column = QVBoxLayout()
            column.addWidget(title)
            row = QHBoxLayout()
            buttons = QVBoxLayout()
            buttons.addWidget(add_button)
            buttons.addWidget(remove_button)
            buttons.addStretch()
            row.addLayout(buttons)
            row.addWidget(marker_list)
            column.addLayout(row)
            marker_row.addLayout(column)
        swap_column = QVBoxLayout()
        swap_column.addWidget(QLabel(""))
        swap_column.addStretch()
        swap_column.addWidget(controls["swap_button"])
        swap_column.addStretch()
        marker_row.insertLayout(1, swap_column)
        layout.addLayout(marker_row)
        options = QHBoxLayout()
        options.addWidget(QLabel("Axis"))
        options.addWidget(controls["axis_combo"])
        options.addWidget(controls["keep_checkbox"])
        layout.addLayout(options)
        return group

    def _axis_vectors_layout(axis_vector_controls: list[dict[str, object]]):
        """
        Stack vector 1 and vector 2 definitions so their start/end fields read vertically.
        """
        layout = QVBoxLayout()
        for index, controls in enumerate(axis_vector_controls):
            layout.addWidget(_axis_vector_layout(index, controls))
        return layout

    class C3dSegmentAxisPreviewWidget(QWidget):
        """
        Small rotatable preview for marker-defined segment axes in the C3D workflow.
        """

        def __init__(self):
            super().__init__()
            self.setMinimumHeight(220)
            self.c3d_data = None
            self.marker_names = ()
            self.label_marker_names = ()
            self.segment_marker_groups = ()
            self.selected_segment_name = ""
            self.axes = ()
            self.virtual_markers = ()
            self.virtual_feature_c3d_data = {}
            self.use_diverse_functional_frames = False
            self.manual_functional_frame_indices = ()
            self._score_solution_cache = {}
            self.current_vectors = ()
            self.current_origin_markers = ()
            self.frame_index = 0
            _initialize_preview_camera(self)
            self._sara_axis_line_cache = {}
            self._sara_axis_solution_cache = {}
            self._source_position_cache = {}
            self._axis_preview_context_cache_key = None

        def set_context(
            self,
            c3d_data,
            marker_names: tuple[str, ...],
            label_marker_names: tuple[str, ...],
            segment_marker_groups: tuple[object, ...],
            selected_segment_name: str,
            axes: tuple[object, ...],
            current_vectors: tuple[tuple[str, tuple[str, ...], tuple[str, ...], bool], ...],
            virtual_markers: tuple[object, ...] = (),
            virtual_feature_c3d_data: dict[str, object] | None = None,
            current_origin_markers: tuple[str, ...] = (),
            frame_index: int = 0,
            use_diverse_functional_frames: bool = False,
            manual_functional_frame_indices: tuple[int, ...] = (),
        ) -> None:
            manual_functional_frame_indices = tuple(int(index) for index in manual_functional_frame_indices)
            context_cache_key = (
                id(c3d_data),
                id(virtual_feature_c3d_data),
                id(segment_marker_groups),
                id(axes),
                id(virtual_markers),
                use_diverse_functional_frames,
                manual_functional_frame_indices,
            )
            if context_cache_key != self._axis_preview_context_cache_key:
                self._sara_axis_line_cache.clear()
                self._sara_axis_solution_cache.clear()
                self._axis_preview_context_cache_key = context_cache_key
            self.c3d_data = c3d_data
            self.marker_names = marker_names
            self.label_marker_names = label_marker_names
            self.segment_marker_groups = segment_marker_groups
            self.selected_segment_name = selected_segment_name
            self.axes = axes
            self.virtual_markers = virtual_markers
            self.virtual_feature_c3d_data = {} if virtual_feature_c3d_data is None else virtual_feature_c3d_data
            self.use_diverse_functional_frames = use_diverse_functional_frames
            self.manual_functional_frame_indices = manual_functional_frame_indices
            self.current_vectors = current_vectors
            self.current_origin_markers = current_origin_markers
            self.frame_index = frame_index
            self._source_position_cache.clear()
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            _set_preview_render_hints(self, painter)
            painter.fillRect(self.rect(), QColor("white"))
            if self.c3d_data is None:
                painter.drawText(
                    self.rect(),
                    qt_alignment_center,
                    "Choose a C3D to preview marker axes",
                )
                return
            marker_records = []
            seen_records = set()
            for segment_index, group in enumerate(self.segment_marker_groups):
                marker_names = tuple(dict.fromkeys(group.marker_names + group.technical_marker_names))
                for marker_name in marker_names:
                    if marker_name not in self.marker_names or marker_name not in self.c3d_data.marker_names:
                        continue
                    point = _marker_frame_position(self.c3d_data, marker_name, self.frame_index)
                    record_key = (group.segment_name, marker_name)
                    if point is None or record_key in seen_records:
                        continue
                    seen_records.add(record_key)
                    marker_records.append(
                        (
                            marker_name,
                            point,
                            group.segment_name,
                            marker_name in group.technical_marker_names,
                            segment_index,
                        )
                    )
            if not getattr(self, "_is_preview_dragging", False):
                for marker_name in _unassigned_marker_names(self.marker_names, self.segment_marker_groups):
                    if marker_name not in self.c3d_data.marker_names:
                        continue
                    point = _marker_frame_position(self.c3d_data, marker_name, self.frame_index)
                    record_key = ("unassigned", marker_name)
                    if point is None or record_key in seen_records:
                        continue
                    seen_records.add(record_key)
                    marker_records.append((marker_name, point, "", False, -2))
            referenced_virtual_marker_names = self._referenced_virtual_marker_names()
            virtual_marker_records = []
            for marker in self.virtual_markers:
                if getattr(self, "_is_preview_dragging", False) and marker.segment_name != self.selected_segment_name:
                    continue
                if (
                    marker.segment_name != self.selected_segment_name
                    and marker.name not in referenced_virtual_marker_names
                ):
                    continue
                point = self._virtual_marker_position(marker)
                if point is None:
                    continue
                virtual_marker_records.append((marker.name, point, marker.segment_name, False, -1))
            marker_records.extend(virtual_marker_records)
            virtual_axis_segments = []
            for axis in self.axes:
                if not _is_virtual_feature_axis(axis):
                    continue
                if getattr(self, "_is_preview_dragging", False) and axis.segment_name != self.selected_segment_name:
                    continue
                start, end = self._line_points_from_axis_definition(axis)
                if start is not None and end is not None:
                    virtual_axis_segments.append((axis.name, start, end))
            axis_segments = []
            for axis in self.axes:
                if _is_virtual_feature_axis(axis):
                    continue
                start, end = self._line_points_from_axis_sources(axis.start_markers, axis.end_markers)
                if start is not None and end is not None:
                    axis_segments.append((axis.axis, axis.keep_vector, start, end))
            saved_origins = []
            for axis in self.axes:
                origin = self._mean_source_position(axis.origin_markers)
                if origin is not None:
                    saved_origins.append(origin)
            temporary_segments = []
            for (
                axis_name,
                start_markers,
                end_markers,
                keep_vector,
            ) in self.current_vectors:
                start, end = self._line_points_from_axis_sources(start_markers, end_markers)
                if start is not None and end is not None:
                    temporary_segments.append((axis_name, keep_vector, start, end))
            temporary_origin = self._mean_source_position(self.current_origin_markers)
            points = [point for _, point, _, _, _ in marker_records]
            for _, _, start, end in axis_segments:
                points.extend((start, end))
            for _, _, start, end in temporary_segments:
                points.extend((start, end))
            for _, start, end in virtual_axis_segments:
                points.extend((start, end))
            points.extend(saved_origins)
            if temporary_origin is not None:
                points.append(temporary_origin)
            if len(points) == 0:
                painter.drawText(
                    self.rect(),
                    qt_alignment_center,
                    "No visible marker for the selected segment",
                )
                return
            projected_points = [_rotate_preview_point(point, self.yaw, self.pitch) for point in points]
            transform = _fit_projection(projected_points, self.width(), self.height(), QPointF, self.zoom)

            _set_preview_label_font(painter)
            label_marker_names = set(self.label_marker_names)
            for marker_name, point, segment_name, is_technical, segment_index in sorted(
                marker_records,
                key=lambda record: _preview_depth(record[1], self.yaw, self.pitch),
            ):
                is_selected = segment_name == self.selected_segment_name
                color = (
                    QColor("#000000")
                    if segment_index == -2
                    else (QColor("#7c3aed") if segment_index == -1 else QColor(_segment_preview_color(segment_index)))
                )
                center = transform(_rotate_preview_point(point, self.yaw, self.pitch))
                radius = 4 if segment_index == -2 else (7 if segment_index == -1 else (6 if is_selected else 4))
                _draw_preview_marker(
                    painter,
                    center,
                    color,
                    is_square=is_technical,
                    size=2 * radius if is_technical else radius,
                    pen_width=3 if is_selected else 1,
                    marker_shape="diamond" if segment_index == -2 else None,
                )
                if _should_draw_preview_labels(self) and is_selected and marker_name in label_marker_names:
                    painter.drawText(center.x() + 5, center.y() - 5, marker_name)

            for axis_name, keep_vector, start, end in axis_segments:
                painter.setPen(_construction_axis_pen(axis_name, keep_vector))
                painter.drawLine(
                    transform(_rotate_preview_point(start, self.yaw, self.pitch)),
                    transform(_rotate_preview_point(end, self.yaw, self.pitch)),
                )

            for axis_name, keep_vector, start, end in temporary_segments:
                painter.setPen(_construction_axis_pen(axis_name, keep_vector))
                painter.drawLine(
                    transform(_rotate_preview_point(start, self.yaw, self.pitch)),
                    transform(_rotate_preview_point(end, self.yaw, self.pitch)),
                )

            for axis_name, start, end in virtual_axis_segments:
                pen = QPen(QColor("#7c3aed"), 2)
                pen.setStyle(qt_dash_line)
                painter.setPen(pen)
                start_screen = transform(_rotate_preview_point(start, self.yaw, self.pitch))
                end_screen = transform(_rotate_preview_point(end, self.yaw, self.pitch))
                painter.drawLine(start_screen, end_screen)
                painter.setPen(QPen(QColor("#7c3aed"), 1))
                if _should_draw_preview_labels(self):
                    painter.drawText(end_screen.x() + 5, end_screen.y() - 5, axis_name)

            painter.setBrush(QColor("#111827"))
            painter.setPen(QPen(QColor("#111827"), 1))
            for origin in saved_origins:
                center = transform(_rotate_preview_point(origin, self.yaw, self.pitch))
                painter.drawRect(int(center.x()) - 4, int(center.y()) - 4, 8, 8)

            if temporary_origin is not None:
                painter.setBrush(QColor("#f59e0b"))
                painter.setPen(QPen(QColor("#f59e0b"), 2))
                center = transform(_rotate_preview_point(temporary_origin, self.yaw, self.pitch))
                painter.drawRect(int(center.x()) - 5, int(center.y()) - 5, 10, 10)
                self._draw_local_frame(painter, transform, temporary_origin, temporary_segments, points)
            _draw_preview_orientation_axes(painter, self.width(), self.height(), self.yaw, self.pitch)
            _draw_preview_interaction_hint(painter, self.width())

        def _line_points_from_axis_sources(
            self, start_markers: tuple[str, ...], end_markers: tuple[str, ...]
        ) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
            reference_axis = _virtual_axis_from_source_names(start_markers, self.axes)
            if reference_axis is not None:
                return self._line_points_from_axis_definition(reference_axis)
            return (
                self._mean_source_position(start_markers),
                self._mean_source_position(end_markers),
            )

        def _line_points_from_axis_definition(
            self, axis
        ) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
            if _is_virtual_feature_axis(axis):
                sara_line = self._sara_axis_line(axis)
                if sara_line != (None, None):
                    return sara_line
            start = self._mean_source_position(axis.start_markers)
            end = self._mean_source_position(axis.end_markers)
            if start is not None and end is not None:
                return start, end
            origin = self._mean_source_position(axis.origin_markers)
            if origin is None or start is None:
                return start, end
            vector = np.asarray(start, dtype=float) - np.asarray(origin, dtype=float)
            if np.linalg.norm(vector) < 1e-12:
                return start, end
            end = tuple(float(value) for value in np.asarray(origin, dtype=float) + vector)
            return origin, end

        def _mean_source_position(self, source_names: tuple[str, ...]) -> tuple[float, float, float] | None:
            points = [self._source_position(source_name) for source_name in source_names]
            points = [point for point in points if point is not None]
            if len(points) == 0:
                return None
            point = np.nanmean(np.asarray(points, dtype=float), axis=0)
            if np.any(~np.isfinite(point)):
                return None
            return tuple(float(value) for value in point)

        def _source_position(self, source_name: str) -> tuple[float, float, float] | None:
            if source_name in self._source_position_cache:
                return self._source_position_cache[source_name]
            if self.c3d_data is None or source_name == "":
                return None
            if source_name in self.c3d_data.marker_names:
                point = _marker_frame_position(self.c3d_data, source_name, self.frame_index)
                self._source_position_cache[source_name] = point
                return point
            virtual_marker = next(
                (marker for marker in self.virtual_markers if marker.name == source_name),
                None,
            )
            if virtual_marker is None:
                return None
            point = self._virtual_marker_position(virtual_marker)
            self._source_position_cache[source_name] = point
            return point

        def _virtual_marker_position(self, marker) -> tuple[float, float, float] | None:
            if marker.name in self.c3d_data.marker_names:
                return _marker_frame_position(self.c3d_data, marker.name, self.frame_index)
            if marker.method == "marker_mean":
                return self._mean_source_position(_split_marker_names(marker.source))
            if marker.method == "axis_projection":
                return self._axis_projection_marker_position(marker)
            if marker.method == "score":
                return self._score_virtual_marker_position(marker)
            if marker.method == "rab2002_shoulder":
                geometry = _rab2002_geometry(
                    self.c3d_data,
                    marker.source,
                    marker.name,
                    marker.segment_name,
                    self.frame_index,
                )
                return None if geometry is None else _finite_point3d(geometry[2])
            return self._functional_marker_preview_fallback(marker)

        def _axis_projection_marker_position(self, marker) -> tuple[float, float, float] | None:
            point = self._mean_source_position(_axis_projection_point_markers_from_payload(marker.source))
            axis_name, axis_start_markers, axis_end_markers = _axis_projection_axis_from_payload(marker.equation)
            if axis_name:
                axis = next(
                    (candidate for candidate in self.axes if candidate.name == axis_name),
                    None,
                )
                if axis is None:
                    return point
                axis_start, axis_end = self._line_points_from_axis_definition(axis)
            else:
                axis_start = self._mean_source_position(axis_start_markers)
                axis_end = self._mean_source_position(axis_end_markers)
            if point is None or axis_start is None or axis_end is None:
                return point
            axis_start_array = np.asarray(axis_start, dtype=float)
            axis_vector = np.asarray(axis_end, dtype=float) - axis_start_array
            norm = float(np.linalg.norm(axis_vector))
            if norm < 1e-12:
                return point
            axis_unit = axis_vector / norm
            projected = axis_start_array + axis_unit * np.dot(
                np.asarray(point, dtype=float) - axis_start_array, axis_unit
            )
            return _finite_point3d(projected)

        def _functional_marker_preview_fallback(self, marker) -> tuple[float, float, float] | None:
            payload = _key_value_payload(marker.source)
            for key in ("fallback", "point"):
                marker_names = _split_marker_names(payload.get(key, ""))
                point = self._mean_source_position(marker_names)
                if point is not None:
                    return point
            return None

        def _score_virtual_marker_position(self, marker) -> tuple[float, float, float] | None:
            payload = _key_value_payload(marker.source)
            parent_marker_names = _split_marker_names(payload.get("parent markers", ""))
            child_marker_names = _split_marker_names(payload.get("child markers", ""))
            functional_data = self._feature_c3d_data(marker.source)
            preview_data = self.c3d_data
            if functional_data is None or preview_data is None:
                return None
            if len(parent_marker_names) == 0 or len(child_marker_names) == 0:
                return None
            if any(
                marker_name not in functional_data.marker_names
                for marker_name in parent_marker_names + child_marker_names
            ):
                return None
            if any(
                marker_name not in preview_data.marker_names for marker_name in parent_marker_names + child_marker_names
            ):
                return None
            cache_key = (
                id(functional_data),
                id(preview_data),
                parent_marker_names,
                child_marker_names,
                self.use_diverse_functional_frames,
                self.manual_functional_frame_indices,
            )
            if cache_key in self._score_solution_cache:
                (
                    cor_parent_local,
                    cor_child_local,
                    rt_parent_preview,
                    rt_child_preview,
                ) = self._score_solution_cache[cache_key]
            else:
                try:
                    from ..components.generic.rigidbody.segment_coordinate_system import (
                        SegmentCoordinateSystemUtils,
                    )
                    from ..model_modifiers.joint_center_tool import Score

                    parent_functional_marker_data = functional_data.get_partial_dict_data(parent_marker_names)
                    child_functional_marker_data = functional_data.get_partial_dict_data(child_marker_names)
                    parent_preview_marker_data = preview_data.get_partial_dict_data(parent_marker_names)
                    child_preview_marker_data = preview_data.get_partial_dict_data(child_marker_names)
                    rt_parent_func = SegmentCoordinateSystemUtils.rigidify(
                        functional_data=parent_functional_marker_data,
                        static_data=parent_preview_marker_data,
                    )
                    rt_child_func = SegmentCoordinateSystemUtils.rigidify(
                        functional_data=child_functional_marker_data,
                        static_data=child_preview_marker_data,
                    )
                    rt_parent_func, rt_child_func, _ = prepare_functional_rt_pair(
                        rt_parent_func,
                        rt_child_func,
                        _functional_frame_selection_options(
                            self.use_diverse_functional_frames,
                            self.manual_functional_frame_indices,
                        ),
                    )
                    _, cor_parent_local, cor_child_local, _, _ = Score.perform_algorithm(rt_parent_func, rt_child_func)
                    rt_parent_preview = SegmentCoordinateSystemUtils.rigidify(parent_preview_marker_data)
                    rt_child_preview = SegmentCoordinateSystemUtils.rigidify(child_preview_marker_data)
                    self._score_solution_cache[cache_key] = (
                        cor_parent_local,
                        cor_child_local,
                        rt_parent_preview,
                        rt_child_preview,
                    )
                except Exception:
                    return None
            frame_index = max(0, min(self.frame_index, len(rt_parent_preview) - 1))
            parent_cor = (
                rt_parent_preview[frame_index] @ np.hstack((np.asarray(cor_parent_local).reshape(3), 1))
            ).reshape(4)[:3]
            child_cor = (
                rt_child_preview[frame_index] @ np.hstack((np.asarray(cor_child_local).reshape(3), 1))
            ).reshape(4)[:3]
            cor = 0.5 * (parent_cor + child_cor)
            return _finite_point3d(cor)

        def _sara_axis_line(self, axis) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
            cache_key = (
                axis.name,
                id(self.c3d_data),
                id(self._feature_c3d_data(axis.source)),
                self.frame_index,
                self.use_diverse_functional_frames,
                self.manual_functional_frame_indices,
            )
            if cache_key in self._sara_axis_line_cache:
                return self._sara_axis_line_cache[cache_key]
            functional_data = self._feature_c3d_data(axis.source)
            preview_data = self.c3d_data
            if functional_data is None or preview_data is None:
                return (None, None)
            payload = _key_value_payload(axis.source)
            parent_marker_names = _split_marker_names(payload.get("parent markers", ""))
            child_marker_names = _split_marker_names(payload.get("child markers", ""))
            expected_start_markers = (
                tuple(axis.start_markers) or _split_marker_names(payload.get("expected axis", ""))[:1]
            )
            expected_end_markers = tuple(axis.end_markers) or _split_marker_names(payload.get("expected axis", ""))[1:]
            origin_markers = tuple(axis.origin_markers) or _split_marker_names(payload.get("origin markers", ""))
            if len(parent_marker_names) == 0 or len(child_marker_names) == 0:
                return (None, None)
            required_functional_markers = (
                parent_marker_names + child_marker_names + expected_start_markers + expected_end_markers
            )
            if any(marker_name not in functional_data.marker_names for marker_name in required_functional_markers):
                return (None, None)
            if any(
                marker_name not in preview_data.marker_names
                for marker_name in origin_markers + expected_start_markers + expected_end_markers
            ):
                return (None, None)
            solution_cache_key = (
                axis.name,
                id(preview_data),
                id(functional_data),
                parent_marker_names,
                child_marker_names,
                expected_start_markers,
                expected_end_markers,
                origin_markers,
                self.use_diverse_functional_frames,
                self.manual_functional_frame_indices,
            )
            solution = self._sara_axis_solution_cache.get(solution_cache_key)
            if solution is None:
                try:
                    from ..components.generic.rigidbody.segment_coordinate_system import (
                        SegmentCoordinateSystemUtils,
                    )
                    from ..model_modifiers.joint_center_tool import Sara

                    parent_preview_marker_data = preview_data.get_partial_dict_data(parent_marker_names)
                    child_preview_marker_data = preview_data.get_partial_dict_data(child_marker_names)
                    parent_functional_marker_data = functional_data.get_partial_dict_data(parent_marker_names)
                    child_functional_marker_data = functional_data.get_partial_dict_data(child_marker_names)
                    rt_parent_func = SegmentCoordinateSystemUtils.rigidify(
                        functional_data=parent_functional_marker_data,
                        static_data=parent_preview_marker_data,
                    )
                    rt_child_func = SegmentCoordinateSystemUtils.rigidify(
                        functional_data=child_functional_marker_data,
                        static_data=child_preview_marker_data,
                    )
                    rt_parent_func, rt_child_func, frame_selection_report = prepare_functional_rt_pair(
                        rt_parent_func,
                        rt_child_func,
                        _functional_frame_selection_options(
                            self.use_diverse_functional_frames,
                            self.manual_functional_frame_indices,
                        ),
                    )
                    original_axis_global = np.nanmean(
                        _mean_marker_series(
                            functional_data,
                            expected_end_markers,
                        )
                        - _mean_marker_series(functional_data, expected_start_markers),
                        axis=1,
                    )
                    origin_positions_global = (
                        functional_data.markers_center_position(origin_markers) if len(origin_markers) != 0 else None
                    )
                    if _uses_selected_functional_frames(frame_selection_report) and origin_positions_global is not None:
                        origin_positions_global = subset_points_by_frame(
                            origin_positions_global,
                            frame_selection_report.selected_indices,
                        )
                    (
                        _aor_mean_global,
                        aor_parent_local,
                        _aor_child_local,
                        _cor_mean_global,
                        cor_parent_local,
                        _cor_child_local,
                        _rt_parent_valid,
                        _rt_child_valid,
                    ) = Sara.perform_algorithm(
                        rt_parent=rt_parent_func,
                        rt_child=rt_child_func,
                        original_axis_global=original_axis_global,
                        origin_positions_global=origin_positions_global,
                        recursive_outlier_removal=False,
                    )
                    rt_parent_preview = SegmentCoordinateSystemUtils.rigidify(parent_preview_marker_data)
                    solution = (rt_parent_preview, aor_parent_local, cor_parent_local)
                    self._sara_axis_solution_cache[solution_cache_key] = solution
                except Exception:
                    return (None, None)
            rt_parent_preview, aor_parent_local, cor_parent_local = solution
            frame_index = max(0, min(self.frame_index, len(rt_parent_preview) - 1))
            start_global = (
                rt_parent_preview[frame_index] @ np.hstack((np.asarray(cor_parent_local).reshape(3), 1.0))
            ).reshape(4)[:3]
            direction_global = rt_parent_preview[frame_index].rotation_matrix.rotation_matrix @ np.asarray(
                aor_parent_local
            ).reshape(3)
            fallback_line = _sara_static_fallback_axis_line(
                preview_data,
                payload,
                expected_start_markers,
                expected_end_markers,
                direction_global,
                frame_index,
                required_functional_markers,
            )
            if fallback_line is not None:
                self._sara_axis_line_cache[cache_key] = fallback_line
                return fallback_line
            start_point = _finite_point3d(start_global)
            raw_end_point = _finite_point3d(start_global + direction_global)
            if start_point is None or raw_end_point is None:
                self._sara_axis_line_cache[cache_key] = (None, None)
                return (None, None)
            end_point = _scaled_axis_end_point(
                preview_data,
                start_point,
                raw_end_point,
                required_functional_markers,
                frame_index,
            )
            line = (start_point, end_point)
            self._sara_axis_line_cache[cache_key] = line
            return line

        def _feature_c3d_data(self, source: str):
            c3d_name = _c3d_source_name_from_virtual_feature_source(source)
            if c3d_name and c3d_name in self.virtual_feature_c3d_data:
                return self.virtual_feature_c3d_data[c3d_name]
            trial_name = _trial_name_from_virtual_feature_source(source)
            if trial_name and trial_name in self.virtual_feature_c3d_data:
                return self.virtual_feature_c3d_data[trial_name]
            return None

        def _referenced_virtual_marker_names(self) -> set[str]:
            referenced_names = set(self.current_origin_markers)
            for _, start_markers, end_markers, _ in self.current_vectors:
                referenced_names.update(start_markers)
                referenced_names.update(end_markers)
            for axis in self.axes:
                if _is_virtual_feature_axis(axis):
                    continue
                referenced_names.update(axis.origin_markers)
                referenced_names.update(axis.start_markers)
                referenced_names.update(axis.end_markers)
            return {marker.name for marker in self.virtual_markers if marker.name in referenced_names}

        def _draw_local_frame(self, painter, transform, origin, temporary_segments, scene_points) -> None:
            local_axes = _orthonormal_axes_from_vector_segments(temporary_segments)
            if len(local_axes) == 0:
                return
            positions = np.asarray(scene_points, dtype=float)
            scene_span = float(np.nanmax(np.ptp(positions, axis=0))) if positions.size != 0 else 1.0
            axis_length = max(scene_span * 0.18, 1e-6)
            origin_array = np.asarray(origin, dtype=float)
            origin_screen = transform(_rotate_preview_point(tuple(origin_array), self.yaw, self.pitch))
            for axis_name in ("x", "y", "z"):
                if axis_name not in local_axes:
                    continue
                endpoint = origin_array + axis_length * local_axes[axis_name]
                endpoint_screen = transform(_rotate_preview_point(tuple(endpoint), self.yaw, self.pitch))
                painter.setPen(QPen(QColor(255, 255, 255, 220), 7))
                painter.drawLine(origin_screen, endpoint_screen)
                painter.setPen(QPen(QColor(_preview_axis_color(axis_name)), 5))
                painter.drawLine(origin_screen, endpoint_screen)
                if _should_draw_preview_labels(self):
                    painter.drawText(
                        endpoint_screen.x() + 4,
                        endpoint_screen.y() - 4,
                        axis_name.upper(),
                    )

        def mousePressEvent(self, event) -> None:
            _start_preview_camera_drag(self, event)

        def mouseMoveEvent(self, event) -> None:
            _drag_preview_camera(self, event)

        def mouseReleaseEvent(self, event) -> None:
            _end_preview_camera_drag(self)

        def mouseDoubleClickEvent(self, event) -> None:
            _reset_preview_camera(self)

        def wheelEvent(self, event) -> None:
            _zoom_preview_camera(self, event)

    class C3dVirtualMarkerPreviewWidget(QWidget):
        """
        Rotatable C3D preview focused on virtual marker placement.
        """

        def __init__(self):
            super().__init__()
            self.setMinimumHeight(280)
            self.setMinimumWidth(420)
            self.c3d_data = None
            self.solution_c3d_data = None
            self.preview_source_label = ""
            self.groups = ()
            self.axes = ()
            self.virtual_markers = ()
            self._empty_virtual_feature_c3d_data = {}
            self.virtual_feature_c3d_data = self._empty_virtual_feature_c3d_data
            self.selected_marker_name = ""
            self.selected_method = "pointing"
            self.proximal_segment_name = ""
            self.distal_segment_name = ""
            self.frame_index = 0
            self.show_whole_body = False
            self.show_kinematic_chain = False
            self.fast_playback_preview = False
            self.use_diverse_functional_frames = False
            self.manual_functional_frame_indices = ()
            self._score_solution_cache = {}
            self._sara_axis_solution_cache = {}
            self._solution_preview_cache = {}
            self._source_position_cache = {}
            _initialize_preview_camera(self)

        def set_context(
            self,
            c3d_data,
            solution_c3d_data,
            preview_source_label: str,
            groups: tuple[object, ...],
            axes: tuple[object, ...],
            virtual_markers: tuple[object, ...],
            virtual_feature_c3d_data: dict[str, object] | None,
            selected_marker_name: str,
            selected_method: str,
            proximal_segment_name: str,
            distal_segment_name: str,
            frame_index: int,
            show_whole_body: bool,
            use_diverse_functional_frames: bool = False,
            manual_functional_frame_indices: tuple[int, ...] = (),
            show_kinematic_chain: bool = False,
            fast_playback_preview: bool = False,
        ) -> None:
            manual_functional_frame_indices = tuple(int(index) for index in manual_functional_frame_indices)
            normalized_virtual_feature_c3d_data = (
                self._empty_virtual_feature_c3d_data
                if virtual_feature_c3d_data is None or len(virtual_feature_c3d_data) == 0
                else virtual_feature_c3d_data
            )
            context_changed = (
                c3d_data is not self.c3d_data
                or solution_c3d_data is not self.solution_c3d_data
                or normalized_virtual_feature_c3d_data is not self.virtual_feature_c3d_data
                or use_diverse_functional_frames != self.use_diverse_functional_frames
                or manual_functional_frame_indices != self.manual_functional_frame_indices
            )
            frame_changed = frame_index != self.frame_index
            if context_changed:
                self._score_solution_cache.clear()
                self._sara_axis_solution_cache.clear()
                self._solution_preview_cache.clear()
            if context_changed or frame_changed:
                self._source_position_cache.clear()
            self.c3d_data = c3d_data
            self.solution_c3d_data = solution_c3d_data
            self.preview_source_label = preview_source_label
            self.groups = groups
            self.axes = axes
            self.virtual_markers = virtual_markers
            self.virtual_feature_c3d_data = normalized_virtual_feature_c3d_data
            self.selected_marker_name = selected_marker_name
            self.selected_method = selected_method
            self.proximal_segment_name = proximal_segment_name
            self.distal_segment_name = distal_segment_name
            self.frame_index = frame_index
            self.show_whole_body = show_whole_body
            self.show_kinematic_chain = show_kinematic_chain
            self.fast_playback_preview = fast_playback_preview
            self.use_diverse_functional_frames = use_diverse_functional_frames
            self.manual_functional_frame_indices = manual_functional_frame_indices
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            _set_preview_render_hints(self, painter)
            painter.fillRect(self.rect(), QColor("white"))
            if self.c3d_data is None:
                painter.drawText(
                    self.rect(),
                    qt_alignment_center,
                    "Choose a C3D to preview virtual markers",
                )
                return

            highlighted_segments = {
                self.proximal_segment_name,
                self.distal_segment_name,
            }
            highlighted_marker_names = set()
            technical_marker_names = set()
            segment_by_marker = {}
            segment_index_by_name = {group.segment_name: index for index, group in enumerate(self.groups)}
            for group in self.groups:
                if group.segment_name not in highlighted_segments:
                    continue
                marker_names = tuple(dict.fromkeys(group.marker_names + group.technical_marker_names))
                highlighted_marker_names.update(marker_names)
                technical_marker_names.update(group.technical_marker_names)
                for marker_name in marker_names:
                    segment_by_marker[marker_name] = group.segment_name

            marker_records = []
            marker_names_to_draw = _virtual_marker_preview_marker_names_to_draw(
                self.c3d_data.marker_names,
                highlighted_marker_names,
                self.show_whole_body,
                getattr(self, "_is_preview_dragging", False),
            )
            for marker_name in marker_names_to_draw:
                point = _marker_frame_position(self.c3d_data, marker_name, self.frame_index)
                if point is None:
                    continue
                segment_name = segment_by_marker.get(marker_name, "")
                marker_records.append(
                    (
                        marker_name,
                        point,
                        segment_name,
                        marker_name in technical_marker_names,
                        segment_index_by_name.get(segment_name, -1),
                        marker_name in highlighted_marker_names,
                    )
                )

            chain_scene = self._kinematic_chain_scene()
            chain_origins = chain_scene["segment_origins"]
            chain_links = chain_scene["links"]
            chain_frames = chain_scene["segment_frames"]
            chain_frame_length = chain_scene["frame_length"]
            virtual_points = []
            selected_virtual_markers = [
                marker for marker in self.virtual_markers if marker.name == self.selected_marker_name
            ]
            for marker in selected_virtual_markers:
                if self.fast_playback_preview:
                    continue
                if marker.method not in {"marker_mean", "axis_projection"}:
                    continue
                point = (
                    self._axis_projection_marker_position(marker)
                    if marker.method == "axis_projection"
                    else _mean_frame_position(
                        self.c3d_data,
                        _split_marker_names(marker.source),
                        self.frame_index,
                    )
                )
                if point is not None:
                    virtual_points.append((marker.name, point, marker.segment_name))

            solution_points = []
            solution_axes = []
            if self.selected_method == "score":
                if not self.fast_playback_preview:
                    solution_points.extend(
                        self._score_solution_points(self.proximal_segment_name, self.distal_segment_name)
                    )
            elif _is_sara_direction_method(self.selected_method):
                if not self.fast_playback_preview:
                    solution_axes.extend(self._sara_axis_solution_lines())
            elif self.selected_method in set(PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS):
                selected_marker = next(
                    (marker for marker in self.virtual_markers if marker.name == self.selected_marker_name),
                    None,
                )
                if self.selected_method == "rab2002_shoulder" and selected_marker is not None:
                    geometry = _rab2002_geometry(
                        self.c3d_data,
                        selected_marker.source,
                        selected_marker.name,
                        selected_marker.segment_name,
                        self.frame_index,
                    )
                    if geometry is not None:
                        caj, epicondyle_mid, gjc = geometry
                        solution_points.extend((("caj", caj), ("mid", epicondyle_mid), ("mean", gjc)))
                else:
                    proximal_point = self._technical_segment_center(self.proximal_segment_name)
                    distal_point = self._technical_segment_center(self.distal_segment_name)
                    if proximal_point is not None:
                        solution_points.append(("parent", proximal_point))
                    if distal_point is not None:
                        solution_points.append(("segment", distal_point))

            points = [record[1] for record in marker_records]
            points.extend(chain_origins.values())
            for _, _, start_point, end_point in chain_links:
                points.extend((start_point, end_point))
            points.extend(chain_scene["points"])
            points.extend(point for _, point, _ in virtual_points)
            points.extend(point for _, point in solution_points)
            for _, _, start_point, end_point in solution_axes:
                points.extend((start_point, end_point))
            if len(points) == 0:
                painter.drawText(
                    self.rect(),
                    qt_alignment_center,
                    "No visible marker for this virtual marker context",
                )
                return

            focus_points = [record[1] for record in marker_records if record[-1]]
            focus_points.extend(point for _, point, _ in virtual_points)
            focus_points.extend(point for _, point in solution_points)
            for _, _, start_point, end_point in solution_axes:
                focus_points.extend((start_point, end_point))
            points_to_fit = (
                points if self.show_whole_body or self.show_kinematic_chain or len(focus_points) == 0 else focus_points
            )
            projected_points = [_rotate_preview_point(point, self.yaw, self.pitch) for point in points_to_fit]
            transform = _fit_projection(projected_points, self.width(), self.height(), QPointF, self.zoom)

            _set_preview_label_font(painter)
            self._draw_kinematic_chain(
                painter,
                transform,
                chain_origins,
                chain_links,
                chain_frames,
                chain_frame_length,
            )
            for (
                marker_name,
                point,
                segment_name,
                is_technical,
                segment_index,
                is_highlighted,
            ) in sorted(
                marker_records,
                key=lambda record: _preview_depth(record[1], self.yaw, self.pitch),
            ):
                color = QColor(_segment_preview_color(segment_index)) if is_highlighted else QColor("#cbd5e1")
                center = transform(_rotate_preview_point(point, self.yaw, self.pitch))
                if is_technical:
                    size = 8 if is_highlighted else 5
                    _draw_preview_marker(painter, center, color, is_square=True, size=size)
                else:
                    radius = 4 if is_highlighted else 2
                    _draw_preview_marker(painter, center, color, is_square=False, size=radius)
                if (
                    _should_draw_preview_labels(self)
                    and is_highlighted
                    and segment_name in {self.proximal_segment_name, self.distal_segment_name}
                ):
                    painter.drawText(center.x() + 5, center.y() - 5, marker_name)

            painter.setPen(QPen(QColor("#7c3aed"), 2))
            painter.setBrush(QColor("#7c3aed"))
            for marker_name, point, segment_name in sorted(
                virtual_points,
                key=lambda record: _preview_depth(record[1], self.yaw, self.pitch),
            ):
                center = transform(_rotate_preview_point(point, self.yaw, self.pitch))
                painter.drawEllipse(center, 7, 7)
                if _should_draw_preview_labels(self):
                    painter.drawText(center.x() + 8, center.y() - 8, marker_name)

            solution_colors = {
                "parent": "#f97316",
                "segment": "#0891b2",
                "child": "#0891b2",
                "caj": "#f97316",
                "mid": "#0891b2",
                "mean": "#111827",
                "fallback": "#64748b",
            }
            solution_offsets = {
                "parent": (10, -12),
                "segment": (10, 6),
                "caj": (10, -12),
                "mid": (10, 6),
                "mean": (10, 24),
            }
            solution_labels = {
                "caj": "CAJ",
                "mid": "mid epicondyles",
                "mean": "GJC" if self.selected_method == "rab2002_shoulder" else "mean",
            }
            for label, point in solution_points:
                center = transform(_rotate_preview_point(point, self.yaw, self.pitch))
                painter.setPen(QPen(QColor(solution_colors[label]), 3))
                painter.setBrush(QColor("white"))
                painter.drawEllipse(center, 8, 8)
                prefix = (
                    "SCoRE"
                    if self.selected_method == "score"
                    else ("Rab" if self.selected_method == "rab2002_shoulder" else "center")
                )
                offset_x, offset_y = solution_offsets.get(label, (8, -8))
                if _should_draw_preview_labels(self):
                    painter.drawText(
                        center.x() + offset_x,
                        center.y() + offset_y,
                        f"{prefix} {solution_labels.get(label, label)}",
                    )
            if len(solution_points) >= 2:
                painter.setPen(QPen(QColor("#111827"), 1))
                painter.drawLine(
                    transform(_rotate_preview_point(solution_points[0][1], self.yaw, self.pitch)),
                    transform(_rotate_preview_point(solution_points[1][1], self.yaw, self.pitch)),
                )

            for role, axis_name, start_point, end_point in solution_axes:
                start = transform(_rotate_preview_point(start_point, self.yaw, self.pitch))
                end = transform(_rotate_preview_point(end_point, self.yaw, self.pitch))
                line_color = QColor(solution_colors.get(role, "#111827"))
                axis_pen = QPen(line_color, 4 if role == "mean" else 2)
                if role == "fallback":
                    axis_pen.setStyle(qt_dash_line)
                painter.setPen(axis_pen)
                painter.drawLine(start, end)
                painter.setBrush(QColor("white"))
                painter.setPen(QPen(line_color, 3))
                painter.drawEllipse(start, 7, 7)
                painter.setBrush(line_color)
                painter.drawEllipse(end, 4, 4)
                if _should_draw_preview_labels(self):
                    prefix = "Static fallback" if role == "fallback" else "SARA"
                    painter.drawText(end.x() + 8, end.y() - 8, f"{prefix} {role} {axis_name}")

            painter.setPen(QPen(QColor("#111827"), 1))
            if _should_draw_preview_labels(self) and self.preview_source_label:
                painter.drawText(12, self.height() - 12, f"Preview: {self.preview_source_label}")
            _draw_preview_orientation_axes(painter, self.width(), self.height(), self.yaw, self.pitch)
            _draw_preview_interaction_hint(painter, self.width())
            if _should_draw_preview_labels(self):
                self._draw_legend(painter)

        def _kinematic_chain_scene(self) -> dict[str, object]:
            if not self.show_kinematic_chain or self.c3d_data is None:
                return {
                    "segment_origins": {},
                    "segment_frames": {},
                    "links": [],
                    "frame_length": 1.0,
                    "points": [],
                }
            origins: dict[str, tuple[float, float, float]] = {}
            frames: dict[str, dict[str, np.ndarray]] = {}
            points: list[tuple[float, float, float]] = []
            for group in self.groups:
                point = self._segment_origin(group)
                if point is not None:
                    origins[group.segment_name] = point
                    # Keep chain origins anatomical during fast playback; only skip
                    # local frame axes and functional overlays to keep animation fluid.
                    frames[group.segment_name] = (
                        {} if self.fast_playback_preview else self._segment_local_axes(group.segment_name)
                    )
                    points.append(point)
            links = []
            for group in self.groups:
                parent_name = group.parent_name
                if parent_name in {"", "root"}:
                    continue
                start = origins.get(parent_name)
                end = origins.get(group.segment_name)
                if start is None or end is None:
                    continue
                links.append((parent_name, group.segment_name, start, end))
                points.extend((start, end))
            frame_length = max(self._scene_span(points) * 0.12, 1e-6)
            highlighted_segments = {
                self.proximal_segment_name,
                self.distal_segment_name,
            }
            for segment_name, origin in origins.items():
                if segment_name not in highlighted_segments:
                    continue
                origin_array = np.asarray(origin, dtype=float)
                for axis_vector in frames.get(segment_name, {}).values():
                    endpoint = origin_array + frame_length * axis_vector
                    points.append(tuple(float(value) for value in endpoint))
            return {
                "segment_origins": origins,
                "segment_frames": frames,
                "links": links,
                "frame_length": frame_length,
                "points": points,
            }

        def _draw_kinematic_chain(
            self,
            painter,
            transform,
            origins: dict[str, tuple[float, float, float]],
            links: list[tuple[str, str, tuple[float, float, float], tuple[float, float, float]]],
            frames: dict[str, dict[str, np.ndarray]],
            frame_length: float,
        ) -> None:
            if not origins:
                return
            highlighted_segments = {
                self.proximal_segment_name,
                self.distal_segment_name,
            }
            for _parent_name, child_name, start_point, end_point in links:
                highlighted = child_name in highlighted_segments
                painter.setPen(
                    QPen(
                        QColor("#334155" if highlighted else "#94a3b8"),
                        4 if highlighted else 2,
                    )
                )
                painter.drawLine(
                    transform(_rotate_preview_point(start_point, self.yaw, self.pitch)),
                    transform(_rotate_preview_point(end_point, self.yaw, self.pitch)),
                )
            for segment_name, origin in origins.items():
                highlighted = segment_name in highlighted_segments
                center = transform(_rotate_preview_point(origin, self.yaw, self.pitch))
                painter.setBrush(QColor("#ffffff" if highlighted else "#e2e8f0"))
                painter.setPen(QPen(QColor("#0f172a"), 3 if highlighted else 1))
                painter.drawEllipse(center, 6 if highlighted else 4, 6 if highlighted else 4)
                if _should_draw_preview_labels(self):
                    painter.drawText(center.x() + 7, center.y() - 7, segment_name)
                if highlighted:
                    self._draw_chain_segment_frame(
                        painter,
                        transform,
                        origin,
                        frames.get(segment_name, {}),
                        frame_length,
                    )

        def _draw_chain_segment_frame(
            self,
            painter,
            transform,
            origin: tuple[float, float, float],
            axes: dict[str, np.ndarray],
            frame_length: float,
        ) -> None:
            if len(axes) == 0:
                return
            origin_array = np.asarray(origin, dtype=float)
            origin_screen = transform(_rotate_preview_point(origin, self.yaw, self.pitch))
            for axis_name in ("x", "y", "z"):
                axis_vector = axes.get(axis_name)
                if axis_vector is None:
                    continue
                endpoint = origin_array + frame_length * axis_vector
                endpoint_screen = transform(_rotate_preview_point(tuple(endpoint), self.yaw, self.pitch))
                painter.setPen(QPen(QColor(255, 255, 255, 220), 7))
                painter.drawLine(origin_screen, endpoint_screen)
                painter.setPen(QPen(QColor(_preview_axis_color(axis_name)), 4))
                painter.drawLine(origin_screen, endpoint_screen)
                if _should_draw_preview_labels(self):
                    painter.drawText(
                        endpoint_screen.x() + 4,
                        endpoint_screen.y() - 4,
                        axis_name.upper(),
                    )

        def _segment_origin(self, group) -> tuple[float, float, float] | None:
            for axis in self._local_frame_axes(group.segment_name):
                origin = self._mean_source_position(axis.origin_markers)
                if origin is not None:
                    return origin
            marker_names = tuple(dict.fromkeys(group.marker_names + group.technical_marker_names))
            return self._mean_source_position(marker_names)

        def _segment_local_axes(self, segment_name: str) -> dict[str, np.ndarray]:
            vector_segments = []
            for axis in self._local_frame_axes(segment_name)[:2]:
                start, end = self._line_points_from_axis_sources(axis.start_markers, axis.end_markers)
                if start is not None and end is not None:
                    vector_segments.append((axis.axis, axis.keep_vector, start, end))
            return _orthonormal_axes_from_vector_segments(tuple(vector_segments))

        def _local_frame_axes(self, segment_name: str):
            return tuple(
                axis for axis in self.axes if axis.segment_name == segment_name and not _is_virtual_feature_axis(axis)
            )

        def _line_points_from_axis_sources(
            self, start_markers: tuple[str, ...], end_markers: tuple[str, ...]
        ) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
            reference_axis = _virtual_axis_from_source_names(start_markers, self.axes)
            if reference_axis is not None:
                return self._line_points_from_axis_definition(reference_axis)
            return (
                self._mean_source_position(start_markers),
                self._mean_source_position(end_markers),
            )

        def _line_points_from_axis_definition(
            self, axis
        ) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
            if _is_virtual_feature_axis(axis):
                lines = self._sara_axis_solution_lines_for_axis(axis)
                mean_line = next((line for line in lines if line[0] == "mean"), None)
                if mean_line is not None:
                    return mean_line[2], mean_line[3]
                if len(lines) != 0:
                    return lines[0][2], lines[0][3]
            start = self._mean_source_position(axis.start_markers)
            end = self._mean_source_position(axis.end_markers)
            if start is not None and end is not None:
                return start, end
            origin = self._mean_source_position(axis.origin_markers)
            if origin is None or start is None:
                return start, end
            vector = np.asarray(start, dtype=float) - np.asarray(origin, dtype=float)
            if np.linalg.norm(vector) < 1e-12:
                return start, end
            end = tuple(float(value) for value in np.asarray(origin, dtype=float) + vector)
            return origin, end

        def _mean_source_position(self, source_names: tuple[str, ...]) -> tuple[float, float, float] | None:
            points = [self._source_position(source_name) for source_name in source_names]
            points = [point for point in points if point is not None]
            if len(points) == 0:
                return None
            point = np.nanmean(np.asarray(points, dtype=float), axis=0)
            if np.any(~np.isfinite(point)):
                return None
            return tuple(float(value) for value in point)

        def _source_position(self, source_name: str) -> tuple[float, float, float] | None:
            if source_name in self._source_position_cache:
                return self._source_position_cache[source_name]
            if self.c3d_data is None or source_name == "":
                return None
            if source_name in self.c3d_data.marker_names:
                point = _marker_frame_position(self.c3d_data, source_name, self.frame_index)
                self._source_position_cache[source_name] = point
                return point
            virtual_marker = next(
                (marker for marker in self.virtual_markers if marker.name == source_name),
                None,
            )
            if virtual_marker is None:
                return None
            point = self._virtual_marker_position(virtual_marker)
            self._source_position_cache[source_name] = point
            return point

        def _virtual_marker_position(self, marker) -> tuple[float, float, float] | None:
            if marker.name in self.c3d_data.marker_names:
                return _marker_frame_position(self.c3d_data, marker.name, self.frame_index)
            if marker.method == "marker_mean":
                return self._mean_source_position(_split_marker_names(marker.source))
            if marker.method == "axis_projection":
                return self._axis_projection_marker_position(marker)
            if marker.method == "score":
                return self._score_virtual_marker_position(marker)
            if marker.method == "rab2002_shoulder":
                geometry = _rab2002_geometry(
                    self.c3d_data,
                    marker.source,
                    marker.name,
                    marker.segment_name,
                    self.frame_index,
                )
                return None if geometry is None else _finite_point3d(geometry[2])
            return self._functional_marker_preview_fallback(marker)

        @staticmethod
        def _scene_span(points: list[tuple[float, float, float]]) -> float:
            if len(points) == 0:
                return 1.0
            positions = np.asarray(points, dtype=float)
            span = float(np.nanmax(np.ptp(positions, axis=0)))
            return span if np.isfinite(span) and span > 0 else 1.0

        def _functional_marker_preview_fallback(self, marker) -> tuple[float, float, float] | None:
            payload = _key_value_payload(marker.source)
            for key in ("fallback", "point"):
                marker_names = _split_marker_names(payload.get(key, ""))
                point = self._mean_source_position(marker_names)
                if point is not None:
                    return point
            return None

        def _score_virtual_marker_position(self, marker) -> tuple[float, float, float] | None:
            payload = _key_value_payload(marker.source)
            parent_marker_names = _split_marker_names(payload.get("parent markers", ""))
            child_marker_names = _split_marker_names(payload.get("child markers", ""))
            functional_data = self._feature_c3d_data(marker.source)
            preview_data = self.c3d_data
            if functional_data is None or preview_data is None:
                return None
            if len(parent_marker_names) == 0 or len(child_marker_names) == 0:
                return None
            if any(
                marker_name not in functional_data.marker_names
                for marker_name in parent_marker_names + child_marker_names
            ):
                return None
            if any(
                marker_name not in preview_data.marker_names for marker_name in parent_marker_names + child_marker_names
            ):
                return None
            cache_key = (
                id(functional_data),
                id(preview_data),
                parent_marker_names,
                child_marker_names,
                self.use_diverse_functional_frames,
                self.manual_functional_frame_indices,
            )
            if cache_key in self._score_solution_cache:
                (
                    cor_parent_local,
                    cor_child_local,
                    rt_parent_preview,
                    rt_child_preview,
                ) = self._score_solution_cache[cache_key]
            else:
                try:
                    from ..components.generic.rigidbody.segment_coordinate_system import (
                        SegmentCoordinateSystemUtils,
                    )
                    from ..model_modifiers.joint_center_tool import Score

                    parent_functional_marker_data = functional_data.get_partial_dict_data(parent_marker_names)
                    child_functional_marker_data = functional_data.get_partial_dict_data(child_marker_names)
                    parent_preview_marker_data = preview_data.get_partial_dict_data(parent_marker_names)
                    child_preview_marker_data = preview_data.get_partial_dict_data(child_marker_names)
                    rt_parent_func = SegmentCoordinateSystemUtils.rigidify(
                        functional_data=parent_functional_marker_data,
                        static_data=parent_preview_marker_data,
                    )
                    rt_child_func = SegmentCoordinateSystemUtils.rigidify(
                        functional_data=child_functional_marker_data,
                        static_data=child_preview_marker_data,
                    )
                    rt_parent_func, rt_child_func, _ = prepare_functional_rt_pair(
                        rt_parent_func,
                        rt_child_func,
                        _functional_frame_selection_options(
                            self.use_diverse_functional_frames,
                            self.manual_functional_frame_indices,
                        ),
                    )
                    _, cor_parent_local, cor_child_local, _, _ = Score.perform_algorithm(rt_parent_func, rt_child_func)
                    rt_parent_preview = SegmentCoordinateSystemUtils.rigidify(parent_preview_marker_data)
                    rt_child_preview = SegmentCoordinateSystemUtils.rigidify(child_preview_marker_data)
                    self._score_solution_cache[cache_key] = (
                        cor_parent_local,
                        cor_child_local,
                        rt_parent_preview,
                        rt_child_preview,
                    )
                except Exception:
                    return None
            frame_index = max(0, min(self.frame_index, len(rt_parent_preview) - 1))
            parent_cor = (
                rt_parent_preview[frame_index] @ np.hstack((np.asarray(cor_parent_local).reshape(3), 1))
            ).reshape(4)[:3]
            child_cor = (
                rt_child_preview[frame_index] @ np.hstack((np.asarray(cor_child_local).reshape(3), 1))
            ).reshape(4)[:3]
            cor = 0.5 * (parent_cor + child_cor)
            return _finite_point3d(cor)

        def _feature_c3d_data(self, source: str):
            c3d_name = _c3d_source_name_from_virtual_feature_source(source)
            if c3d_name and c3d_name in self.virtual_feature_c3d_data:
                return self.virtual_feature_c3d_data[c3d_name]
            trial_name = _trial_name_from_virtual_feature_source(source)
            if trial_name and trial_name in self.virtual_feature_c3d_data:
                return self.virtual_feature_c3d_data[trial_name]
            if self.solution_c3d_data is not None:
                return self.solution_c3d_data
            return None

        def _technical_segment_center(self, segment_name: str) -> tuple[float, float, float] | None:
            for group in self.groups:
                if group.segment_name == segment_name:
                    marker_names = group.technical_marker_names if group.technical_marker_names else group.marker_names
                    return _mean_frame_position(self.c3d_data, marker_names, self.frame_index)
            return None

        def _technical_markers_for_segment(self, segment_name: str) -> tuple[str, ...]:
            for group in self.groups:
                if group.segment_name == segment_name:
                    return group.technical_marker_names if group.technical_marker_names else group.marker_names
            return ()

        def _axis_projection_marker_position(self, marker) -> tuple[float, float, float] | None:
            point = self._mean_source_position(_axis_projection_point_markers_from_payload(marker.source))
            axis_name, axis_start_markers, axis_end_markers = _axis_projection_axis_from_payload(marker.equation)
            if axis_name:
                axis = next(
                    (candidate for candidate in self.axes if candidate.name == axis_name),
                    None,
                )
                if axis is None:
                    return point
                axis_start, axis_end = self._line_points_from_axis_definition(axis)
            else:
                axis_start = self._mean_source_position(axis_start_markers)
                axis_end = self._mean_source_position(axis_end_markers)
            return _project_point_on_line(point, axis_start, axis_end)

        def _score_solution_points(
            self, parent_segment_name: str, segment_name: str
        ) -> tuple[tuple[str, tuple[float, float, float]], ...]:
            preview_cache_key = (
                "score",
                id(self.c3d_data),
                id(self.solution_c3d_data),
                parent_segment_name,
                segment_name,
                self.frame_index,
                self.use_diverse_functional_frames,
                self.manual_functional_frame_indices,
            )
            if preview_cache_key in self._solution_preview_cache:
                return self._solution_preview_cache[preview_cache_key]
            parent_marker_names = self._technical_markers_for_segment(parent_segment_name)
            child_marker_names = self._technical_markers_for_segment(segment_name)
            if len(parent_marker_names) == 0 or len(child_marker_names) == 0:
                return ()
            marker_names = parent_marker_names + child_marker_names
            functional_data = self.solution_c3d_data
            preview_data = self.c3d_data
            if functional_data is None or preview_data is None:
                return ()
            if any(marker_name not in functional_data.marker_names for marker_name in marker_names):
                return ()
            if any(marker_name not in preview_data.marker_names for marker_name in marker_names):
                return ()
            cache_key = (
                id(functional_data),
                id(preview_data),
                parent_marker_names,
                child_marker_names,
                self.use_diverse_functional_frames,
                self.manual_functional_frame_indices,
            )
            if cache_key in self._score_solution_cache:
                (
                    cor_parent_local,
                    cor_child_local,
                    rt_parent_preview,
                    rt_child_preview,
                ) = self._score_solution_cache[cache_key]
            else:
                try:
                    from ..components.generic.rigidbody.segment_coordinate_system import (
                        SegmentCoordinateSystemUtils,
                    )
                    from ..model_modifiers.joint_center_tool import Score

                    parent_functional_marker_data = functional_data.get_partial_dict_data(parent_marker_names)
                    child_functional_marker_data = functional_data.get_partial_dict_data(child_marker_names)
                    parent_preview_marker_data = preview_data.get_partial_dict_data(parent_marker_names)
                    child_preview_marker_data = preview_data.get_partial_dict_data(child_marker_names)
                    rt_parent_func = SegmentCoordinateSystemUtils.rigidify(
                        functional_data=parent_functional_marker_data,
                        static_data=parent_preview_marker_data,
                    )
                    rt_child_func = SegmentCoordinateSystemUtils.rigidify(
                        functional_data=child_functional_marker_data,
                        static_data=child_preview_marker_data,
                    )
                    rt_parent_func, rt_child_func, _ = prepare_functional_rt_pair(
                        rt_parent_func,
                        rt_child_func,
                        _functional_frame_selection_options(
                            self.use_diverse_functional_frames,
                            self.manual_functional_frame_indices,
                        ),
                    )
                    _, cor_parent_local, cor_child_local, _, _ = Score.perform_algorithm(rt_parent_func, rt_child_func)
                    rt_parent_preview = SegmentCoordinateSystemUtils.rigidify(parent_preview_marker_data)
                    rt_child_preview = SegmentCoordinateSystemUtils.rigidify(child_preview_marker_data)
                    self._score_solution_cache[cache_key] = (
                        cor_parent_local,
                        cor_child_local,
                        rt_parent_preview,
                        rt_child_preview,
                    )
                except Exception:
                    return ()
            frame_index = max(0, min(self.frame_index, len(rt_parent_preview) - 1))
            parent_cor = (rt_parent_preview[frame_index] @ np.hstack((cor_parent_local, 1))).reshape(4)[:3]
            child_cor = (rt_child_preview[frame_index] @ np.hstack((cor_child_local, 1))).reshape(4)[:3]
            mean_cor = 0.5 * (parent_cor + child_cor)
            solution_points = (
                ("parent", tuple(float(value) for value in parent_cor)),
                ("segment", tuple(float(value) for value in child_cor)),
                ("mean", tuple(float(value) for value in mean_cor)),
            )
            self._solution_preview_cache[preview_cache_key] = solution_points
            return solution_points

        def _sara_axis_solution_lines(
            self,
        ) -> tuple[tuple[str, str, tuple[float, float, float], tuple[float, float, float]], ...]:
            axis = self._selected_sara_axis()
            if axis is None:
                return ()
            return self._sara_axis_solution_lines_for_axis(axis)

        def _sara_axis_solution_lines_for_axis(
            self, axis
        ) -> tuple[tuple[str, str, tuple[float, float, float], tuple[float, float, float]], ...]:
            preview_cache_key = (
                "sara",
                id(self.c3d_data),
                id(self.solution_c3d_data),
                axis.name,
                self.frame_index,
                self.use_diverse_functional_frames,
                self.manual_functional_frame_indices,
            )
            if preview_cache_key in self._solution_preview_cache:
                return self._solution_preview_cache[preview_cache_key]
            functional_data = self.solution_c3d_data
            preview_data = self.c3d_data
            if functional_data is None or preview_data is None:
                return ()
            payload = _key_value_payload(axis.source)
            parent_marker_names = _split_marker_names(payload.get("parent markers", ""))
            child_marker_names = _split_marker_names(payload.get("child markers", ""))
            expected_start_markers = (
                tuple(axis.start_markers) or _split_marker_names(payload.get("expected axis", ""))[:1]
            )
            expected_end_markers = tuple(axis.end_markers) or _split_marker_names(payload.get("expected axis", ""))[1:]
            if len(parent_marker_names) == 0:
                parent_marker_names = self._technical_markers_for_segment(self.proximal_segment_name)
            if len(child_marker_names) == 0:
                child_marker_names = self._technical_markers_for_segment(self.distal_segment_name)
            required_markers = parent_marker_names + child_marker_names + expected_start_markers + expected_end_markers
            if any(
                marker_name not in functional_data.marker_names
                for marker_name in parent_marker_names + child_marker_names
            ):
                return ()
            if any(marker_name not in preview_data.marker_names for marker_name in required_markers):
                return ()
            if len(expected_start_markers) == 0 or len(expected_end_markers) == 0:
                return ()
            origin_markers = tuple(axis.origin_markers)
            solution_cache_key = (
                axis.name,
                id(preview_data),
                id(functional_data),
                parent_marker_names,
                child_marker_names,
                expected_start_markers,
                expected_end_markers,
                origin_markers,
                self.use_diverse_functional_frames,
                self.manual_functional_frame_indices,
            )
            solution = self._sara_axis_solution_cache.get(solution_cache_key)
            if solution is None:
                try:
                    from ..components.generic.rigidbody.segment_coordinate_system import (
                        SegmentCoordinateSystemUtils,
                    )
                    from ..model_modifiers.joint_center_tool import Sara

                    parent_preview_marker_data = preview_data.get_partial_dict_data(parent_marker_names)
                    child_preview_marker_data = preview_data.get_partial_dict_data(child_marker_names)
                    parent_functional_marker_data = functional_data.get_partial_dict_data(parent_marker_names)
                    child_functional_marker_data = functional_data.get_partial_dict_data(child_marker_names)
                    rt_parent_func = SegmentCoordinateSystemUtils.rigidify(
                        functional_data=parent_functional_marker_data,
                        static_data=parent_preview_marker_data,
                    )
                    rt_child_func = SegmentCoordinateSystemUtils.rigidify(
                        functional_data=child_functional_marker_data,
                        static_data=child_preview_marker_data,
                    )
                    rt_parent_func, rt_child_func, frame_selection_report = prepare_functional_rt_pair(
                        rt_parent_func,
                        rt_child_func,
                        _functional_frame_selection_options(
                            self.use_diverse_functional_frames,
                            self.manual_functional_frame_indices,
                        ),
                    )
                    original_axis_global = np.nanmean(
                        _mean_marker_series(
                            functional_data,
                            expected_end_markers,
                        )
                        - _mean_marker_series(functional_data, expected_start_markers),
                        axis=1,
                    )
                    origin_positions_global = None
                    if len(origin_markers) != 0:
                        if any(marker_name not in functional_data.marker_names for marker_name in origin_markers):
                            return ()
                        origin_positions_global = functional_data.markers_center_position(origin_markers)
                    if _uses_selected_functional_frames(frame_selection_report) and origin_positions_global is not None:
                        origin_positions_global = subset_points_by_frame(
                            origin_positions_global,
                            frame_selection_report.selected_indices,
                        )
                    (
                        aor_mean_global,
                        aor_parent_local,
                        aor_child_local,
                        cor_mean_global,
                        cor_parent_local,
                        cor_child_local,
                        _rt_parent_valid,
                        _rt_child_valid,
                    ) = Sara.perform_algorithm(
                        rt_parent=rt_parent_func,
                        rt_child=rt_child_func,
                        original_axis_global=original_axis_global,
                        origin_positions_global=origin_positions_global,
                        recursive_outlier_removal=False,
                    )
                    rt_parent_preview = SegmentCoordinateSystemUtils.rigidify(parent_preview_marker_data)
                    rt_child_preview = SegmentCoordinateSystemUtils.rigidify(child_preview_marker_data)
                    solution = (
                        aor_mean_global,
                        aor_parent_local,
                        aor_child_local,
                        cor_mean_global,
                        cor_parent_local,
                        cor_child_local,
                        rt_parent_preview,
                        rt_child_preview,
                    )
                    self._sara_axis_solution_cache[solution_cache_key] = solution
                except Exception:
                    return ()
            (
                aor_mean_global,
                aor_parent_local,
                aor_child_local,
                cor_mean_global,
                cor_parent_local,
                cor_child_local,
                rt_parent_preview,
                rt_child_preview,
            ) = solution
            frame_index = max(
                0,
                min(
                    self.frame_index,
                    min(len(rt_parent_preview), len(rt_child_preview)) - 1,
                ),
            )
            parent_start = (
                rt_parent_preview[frame_index] @ np.hstack((np.asarray(cor_parent_local).reshape(3), 1.0))
            ).reshape(4)[:3]
            parent_direction = rt_parent_preview[frame_index].rotation_matrix.rotation_matrix @ np.asarray(
                aor_parent_local
            ).reshape(3)
            child_start = (
                rt_child_preview[frame_index] @ np.hstack((np.asarray(cor_child_local).reshape(3), 1.0))
            ).reshape(4)[:3]
            child_direction = rt_child_preview[frame_index].rotation_matrix.rotation_matrix @ np.asarray(
                aor_child_local
            ).reshape(3)
            mean_start = 0.5 * (parent_start + child_start)
            shared_start = mean_start
            mean_direction = _average_axis_direction(parent_direction, child_direction)
            if np.linalg.norm(mean_direction) <= 1e-12:
                shared_start = np.asarray(cor_mean_global, dtype=float).reshape(3)
                mean_direction = np.asarray(aor_mean_global, dtype=float).reshape(3)
            fallback_line = _sara_static_fallback_axis_line(
                preview_data,
                payload,
                expected_start_markers,
                expected_end_markers,
                mean_direction,
                frame_index,
                required_markers,
            )
            parent_start_point = tuple(float(value) for value in shared_start)
            parent_raw_end_point = tuple(float(value) for value in shared_start + parent_direction)
            child_start_point = tuple(float(value) for value in shared_start)
            child_raw_end_point = tuple(float(value) for value in shared_start + child_direction)
            mean_start_point = tuple(float(value) for value in shared_start)
            mean_raw_end_point = tuple(float(value) for value in shared_start + mean_direction)
            parent_end_point = _scaled_axis_end_point(
                preview_data,
                parent_start_point,
                parent_raw_end_point,
                required_markers,
                frame_index,
            )
            child_end_point = _scaled_axis_end_point(
                preview_data,
                child_start_point,
                child_raw_end_point,
                required_markers,
                frame_index,
            )
            mean_end_point = _scaled_axis_end_point(
                preview_data,
                mean_start_point,
                mean_raw_end_point,
                required_markers,
                frame_index,
            )
            lines = (
                ("parent", axis.name, parent_start_point, parent_end_point),
                ("child", axis.name, child_start_point, child_end_point),
                ("mean", axis.name, mean_start_point, mean_end_point),
            )
            if fallback_line is not None:
                lines = lines + (
                    (
                        "fallback",
                        f"{axis.name} static fallback",
                        fallback_line[0],
                        fallback_line[1],
                    ),
                )
            self._solution_preview_cache[preview_cache_key] = lines
            return lines

        def _selected_sara_axis(self):
            selected_axis = next(
                (item for item in self.axes if item.name == self.selected_marker_name),
                None,
            )
            if (
                selected_axis is not None
                and _is_virtual_feature_axis(selected_axis)
                and _is_sara_direction_method(selected_axis.method)
            ):
                return selected_axis
            selected_marker = next(
                (marker for marker in self.virtual_markers if marker.name == self.selected_marker_name),
                None,
            )
            if selected_marker is not None and selected_marker.method == "axis_projection":
                axis_name, _, _ = _axis_projection_axis_from_payload(selected_marker.equation)
                selected_axis = next((axis for axis in self.axes if axis.name == axis_name), None)
                if (
                    selected_axis is not None
                    and _is_virtual_feature_axis(selected_axis)
                    and _is_sara_direction_method(selected_axis.method)
                ):
                    return selected_axis
            segment_axes = [
                axis
                for axis in self.axes
                if _is_virtual_feature_axis(axis)
                and _is_sara_direction_method(axis.method)
                and axis.segment_name == self.distal_segment_name
            ]
            if len(segment_axes) == 1:
                return segment_axes[0]
            if len(segment_axes) != 0:
                return segment_axes[0]
            return next(
                (
                    axis
                    for axis in self.axes
                    if _is_virtual_feature_axis(axis) and _is_sara_direction_method(axis.method)
                ),
                None,
            )

        def _draw_legend(self, painter) -> None:
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.setBrush(QColor(255, 255, 255, 225))
            painter.drawRect(8, 8, 300, 176)
            painter.drawText(14, 24, "Legend")
            _draw_legend_point(
                painter,
                QPointF(20, 40),
                QColor("#2563eb"),
                "Selected segment/parent marker",
                36,
                44,
            )
            painter.setPen(QPen(QColor("#2563eb"), 1))
            painter.setBrush(QColor("#2563eb"))
            painter.drawRect(16, 52, 8, 8)
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.drawText(36, 62, "Technical marker")
            _draw_legend_point(painter, QPointF(20, 74), QColor("#7c3aed"), "Virtual marker", 36, 78)
            _draw_legend_point(
                painter,
                QPointF(20, 92),
                QColor("#cbd5e1"),
                "Other functional C3D marker",
                36,
                96,
            )
            _draw_legend_line(painter, QColor("#f97316"), 2, 14, 112, "SARA parent axis", 50, 116)
            _draw_legend_line(painter, QColor("#0891b2"), 2, 14, 130, "SARA child axis", 50, 134)
            _draw_legend_line(painter, QColor("#111827"), 4, 14, 148, "SARA mean axis", 50, 152)
            fallback_pen = QPen(QColor("#64748b"), 2)
            fallback_pen.setStyle(qt_dash_line)
            painter.setPen(fallback_pen)
            painter.drawLine(14, 166, 44, 166)
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.drawText(50, 170, "Static fallback axis")

        def mousePressEvent(self, event) -> None:
            _start_preview_camera_drag(self, event)

        def mouseMoveEvent(self, event) -> None:
            _drag_preview_camera(self, event)

        def mouseReleaseEvent(self, event) -> None:
            _end_preview_camera_drag(self)

        def mouseDoubleClickEvent(self, event) -> None:
            _reset_preview_camera(self)

        def wheelEvent(self, event) -> None:
            _zoom_preview_camera(self, event)

    class FunctionalResidualBoxplotWidget(QWidget):
        """
        Draw compact SCoRE/SARA residual boxplots without adding a plotting dependency.
        """

        def __init__(self):
            super().__init__()
            self.boxplots = ()
            self.setMinimumWidth(860)

        def set_boxplots(self, boxplots: tuple[dict[str, object], ...]) -> None:
            self.boxplots = boxplots
            group_count = len({str(plot.get("unit", "")) for plot in self.boxplots})
            self.setMinimumHeight(max(220, 88 + 32 * len(self.boxplots) + 42 * group_count))
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            _set_preview_render_hints(self, painter)
            painter.fillRect(self.rect(), QColor("white"))
            if len(self.boxplots) == 0:
                painter.setPen(QPen(QColor("#111827"), 1))
                painter.drawText(self.rect(), qt_alignment_center, "No residual values to display")
                return

            label_left = 20
            plot_left = min(max(270, self.width() // 4), 380)
            summary_width = 190 if self.width() >= 820 else 0
            plot_right = self.width() - 28 - summary_width
            if plot_right < plot_left + 160:
                plot_left = 180
                plot_right = self.width() - 28
                summary_width = 0
            plot_width = max(1, plot_right - plot_left)
            top = 28
            row_height = 32
            box_color = QColor("#2563eb")
            median_color = QColor("#111827")
            axis_color = QColor("#334155")
            text_color = QColor("#111827")
            muted_color = QColor("#64748b")
            outlier_color = QColor("#93c5fd")

            unit_order = []
            grouped_boxplots = {}
            for boxplot in self.boxplots:
                unit = str(boxplot.get("unit", ""))
                if unit not in grouped_boxplots:
                    grouped_boxplots[unit] = []
                    unit_order.append(unit)
                grouped_boxplots[unit].append(boxplot)

            def finite_values_for(boxplot):
                values = np.asarray(boxplot.get("values", ()), dtype=float)
                return values[np.isfinite(values)]

            def x_for_value(value, value_min, value_max):
                if value_max <= value_min:
                    return plot_left + plot_width / 2.0
                return plot_left + (float(value) - value_min) / (value_max - value_min) * plot_width

            def compact_title(title: str) -> str:
                marker_match = re.match(r"^([A-Za-z0-9_]+) distance to (.+)$", title)
                if marker_match:
                    return f"{marker_match.group(1)} to {marker_match.group(2)}"
                return title

            y = top
            for unit in unit_order:
                plots = grouped_boxplots[unit]
                all_values = [finite_values_for(boxplot) for boxplot in plots if finite_values_for(boxplot).size > 0]
                if len(all_values) == 0:
                    value_min = 0.0
                    value_max = 1.0
                else:
                    combined = np.concatenate(all_values)
                    value_min = float(np.nanmin(combined))
                    value_max = float(np.nanmax(combined))
                    if np.isclose(value_min, value_max):
                        padding = max(1.0, abs(value_min) * 0.05)
                        value_min -= padding
                        value_max += padding

                unit_label = unit or "value"
                painter.setPen(QPen(text_color, 1))
                painter.drawText(plot_left, y, f"Residual distributions ({unit_label})")
                painter.setPen(QPen(axis_color, 1))
                axis_y = y + 18 + row_height * len(plots)
                painter.drawLine(plot_left, axis_y, plot_right, axis_y)
                painter.drawLine(plot_left, axis_y - 4, plot_left, axis_y + 4)
                painter.drawLine(plot_right, axis_y - 4, plot_right, axis_y + 4)
                painter.setPen(QPen(muted_color, 1))
                painter.drawText(plot_left, axis_y + 18, f"{value_min:.3g} {unit_label}")
                painter.drawText(
                    plot_right - 104,
                    axis_y + 18,
                    f"{value_max:.3g} {unit_label}",
                )

                row_y = y + 28
                for boxplot in plots:
                    values = finite_values_for(boxplot)
                    title = compact_title(str(boxplot.get("title", "Residual")))
                    if len(title) > 34:
                        title = title[:31] + "..."
                    painter.setPen(QPen(text_color, 1))
                    painter.drawText(label_left, row_y + 5, title)
                    if values.size == 0:
                        painter.setPen(QPen(muted_color, 1))
                        painter.drawText(plot_left, row_y + 5, "No finite values")
                        row_y += row_height
                        continue

                    q1, median, q3 = np.nanpercentile(values, [25, 50, 75])
                    iqr = q3 - q1
                    lower_fence = q1 - 1.5 * iqr
                    upper_fence = q3 + 1.5 * iqr
                    finite_inliers = values[(values >= lower_fence) & (values <= upper_fence)]
                    if finite_inliers.size == 0:
                        whisker_min = float(np.nanmin(values))
                        whisker_max = float(np.nanmax(values))
                    else:
                        whisker_min = float(np.nanmin(finite_inliers))
                        whisker_max = float(np.nanmax(finite_inliers))
                    whisker_start = x_for_value(whisker_min, value_min, value_max)
                    q1_x = x_for_value(q1, value_min, value_max)
                    median_x = x_for_value(median, value_min, value_max)
                    q3_x = x_for_value(q3, value_min, value_max)
                    whisker_end = x_for_value(whisker_max, value_min, value_max)
                    center_y = row_y
                    box_top = center_y - 8
                    box_height = 16

                    painter.setPen(QPen(axis_color, 1))
                    painter.drawLine(
                        int(round(whisker_start)),
                        center_y,
                        int(round(whisker_end)),
                        center_y,
                    )
                    painter.drawLine(
                        int(round(whisker_start)),
                        center_y - 5,
                        int(round(whisker_start)),
                        center_y + 5,
                    )
                    painter.drawLine(
                        int(round(whisker_end)),
                        center_y - 5,
                        int(round(whisker_end)),
                        center_y + 5,
                    )
                    painter.setPen(QPen(box_color, 1))
                    painter.setBrush(QColor("#dbeafe"))
                    painter.drawRect(
                        int(round(q1_x)),
                        box_top,
                        max(1, int(round(q3_x - q1_x))),
                        box_height,
                    )
                    painter.setPen(QPen(median_color, 2))
                    painter.drawLine(
                        int(round(median_x)),
                        box_top,
                        int(round(median_x)),
                        box_top + box_height,
                    )

                    outliers = values[(values < lower_fence) | (values > upper_fence)]
                    if outliers.size > 80:
                        indices = np.linspace(0, outliers.size - 1, 80).astype(int)
                        outliers = outliers[indices]
                    painter.setPen(QPen(outlier_color, 1))
                    painter.setBrush(outlier_color)
                    for value in outliers:
                        painter.drawEllipse(
                            QPointF(x_for_value(value, value_min, value_max), center_y),
                            2,
                            2,
                        )

                    if summary_width > 0:
                        summary = f"med={median:.3g}; IQR={(q3 - q1):.3g}; n={values.size}"
                        painter.setPen(QPen(muted_color, 1))
                        painter.drawText(plot_right + 12, row_y + 5, summary)
                    row_y += row_height

                y = axis_y + 46

    class C3dTechnicalSegmentPreviewWidget(QWidget):
        """
        Rotatable frame-by-frame C3D marker preview for technical segment assignment.
        """

        def __init__(self):
            super().__init__()
            self.setMinimumHeight(260)
            self.setMinimumWidth(420)
            self.c3d_data = None
            self.groups = ()
            self.selected_segment_name = ""
            self.selected_marker_names = ()
            self.frame_index = 0
            self._marker_records = ()
            _initialize_preview_camera(self)

        def set_context(
            self,
            c3d_data,
            groups: tuple[object, ...],
            selected_segment_name: str,
            selected_marker_names: tuple[str, ...],
            frame_index: int,
        ) -> None:
            self.c3d_data = c3d_data
            self.groups = groups
            self.selected_segment_name = selected_segment_name
            self.selected_marker_names = selected_marker_names
            self.frame_index = frame_index
            self._marker_records = self._build_marker_records()
            self.update()

        def _build_marker_records(
            self,
        ) -> tuple[tuple[str, tuple[float, float, float], str, bool, int], ...]:
            if self.c3d_data is None:
                return ()
            marker_records = []
            seen_marker_names = set()
            for segment_index, group in enumerate(self.groups):
                for marker_name in group.marker_names:
                    point = _marker_frame_position(self.c3d_data, marker_name, self.frame_index)
                    if point is None:
                        continue
                    marker_records.append(
                        (
                            marker_name,
                            point,
                            group.segment_name,
                            marker_name in group.technical_marker_names,
                            segment_index,
                        )
                    )
                    seen_marker_names.add(marker_name)
            for marker_name in _unassigned_marker_names(tuple(self.c3d_data.marker_names), self.groups):
                if marker_name in seen_marker_names:
                    continue
                point = _marker_frame_position(self.c3d_data, marker_name, self.frame_index)
                if point is None:
                    continue
                marker_records.append((marker_name, point, "", False, -2))
            return tuple(marker_records)

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            _set_preview_render_hints(self, painter)
            painter.fillRect(self.rect(), QColor("white"))
            if self.c3d_data is None:
                painter.drawText(
                    self.rect(),
                    qt_alignment_center,
                    "Choose a C3D to preview technical segments",
                )
                return

            marker_records = self._marker_records
            if len(marker_records) == 0:
                painter.drawText(
                    self.rect(),
                    qt_alignment_center,
                    "No assigned marker is visible in this frame",
                )
                return

            projected_points = [
                _rotate_preview_point(point, self.yaw, self.pitch) for _, point, _, _, _ in marker_records
            ]
            transform = _fit_projection(projected_points, self.width(), self.height(), QPointF, self.zoom)

            _set_preview_label_font(painter)
            for marker_name, point, segment_name, is_technical, segment_index in sorted(
                marker_records,
                key=lambda record: _preview_depth(record[1], self.yaw, self.pitch),
            ):
                is_selected = segment_name == self.selected_segment_name
                is_marker_selected = marker_name in self.selected_marker_names
                color = QColor("#000000") if segment_index == -2 else QColor(_segment_preview_color(segment_index))
                center = transform(_rotate_preview_point(point, self.yaw, self.pitch))
                radius = 6 if is_selected else 4
                if is_marker_selected:
                    painter.setPen(QPen(QColor("#f59e0b"), 3))
                    painter.setBrush(QColor(255, 255, 255, 0))
                    painter.drawEllipse(center, 13, 13)
                _draw_preview_marker(
                    painter,
                    center,
                    color,
                    is_square=is_technical,
                    size=(4 if segment_index == -2 else (2 * radius if is_technical else radius)),
                    pen_width=4 if is_marker_selected else (3 if is_selected else 1),
                    marker_shape="diamond" if segment_index == -2 else None,
                )
                if _should_draw_preview_labels(self) and (is_selected or is_marker_selected):
                    painter.drawText(center.x() + 6, center.y() - 6, marker_name)

            _draw_preview_orientation_axes(painter, self.width(), self.height(), self.yaw, self.pitch)
            _draw_preview_interaction_hint(painter, self.width())
            if _should_draw_preview_labels(self):
                self._draw_legend(painter)

        def _draw_legend(self, painter) -> None:
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.setBrush(QColor(255, 255, 255, 225))
            painter.drawRect(8, 8, 250, 68)
            painter.drawText(14, 24, "Legend")
            _draw_legend_point(
                painter,
                QPointF(20, 40),
                QColor("#2563eb"),
                "Additional/anatomical",
                36,
                44,
            )
            painter.setPen(QPen(QColor("#2563eb"), 1))
            painter.setBrush(QColor("#2563eb"))
            painter.drawRect(16, 52, 8, 8)
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.drawText(36, 62, "Technical")

        def mousePressEvent(self, event) -> None:
            _start_preview_camera_drag(self, event)

        def mouseMoveEvent(self, event) -> None:
            _drag_preview_camera(self, event)

        def mouseReleaseEvent(self, event) -> None:
            _end_preview_camera_drag(self)

        def mouseDoubleClickEvent(self, event) -> None:
            _reset_preview_camera(self)

        def wheelEvent(self, event) -> None:
            _zoom_preview_camera(self, event)

    class C3dSegmentSettingsPreviewWidget(C3dSegmentAxisPreviewWidget):
        """
        Rotatable preview of the kinematic chain and segment DoFs.
        """

        def __init__(self):
            super().__init__()
            self.setMinimumHeight(520)
            self.setMinimumWidth(360)
            self.workflow_draft = None
            self.selected_setting_segment_name = ""
            self.segment_setting_overrides = {}
            self._settings_scene = None
            self._settings_scene_cache_key = None

        def set_context(
            self,
            c3d_data,
            workflow_draft,
            selected_segment_name: str,
            frame_index: int,
            virtual_feature_c3d_data: dict[str, object] | None = None,
            segment_setting_overrides: dict[str, object] | None = None,
            q0_scene: dict[str, object] | None = None,
        ) -> None:
            self.workflow_draft = workflow_draft
            self.selected_setting_segment_name = selected_segment_name
            self.segment_setting_overrides = {} if segment_setting_overrides is None else segment_setting_overrides
            geometry_cache_key = (
                id(c3d_data),
                id(workflow_draft),
                frame_index,
                id(virtual_feature_c3d_data),
                id(q0_scene),
            )
            if geometry_cache_key == self._settings_scene_cache_key:
                self.selected_segment_name = selected_segment_name
                self.frame_index = frame_index
                self.update()
                return
            marker_names = tuple(c3d_data.marker_names) if c3d_data is not None else ()
            super().set_context(
                c3d_data,
                marker_names,
                (),
                workflow_draft.segment_marker_groups,
                selected_segment_name,
                workflow_draft.axes,
                (),
                workflow_draft.virtual_markers,
                virtual_feature_c3d_data,
                (),
                frame_index,
            )
            self._settings_scene = q0_scene if q0_scene is not None else self._build_settings_scene()
            self._settings_scene_cache_key = geometry_cache_key

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            _set_preview_render_hints(self, painter)
            painter.fillRect(self.rect(), QColor("white"))
            if self.c3d_data is None or self.workflow_draft is None:
                painter.drawText(
                    self.rect(),
                    qt_alignment_center,
                    "Choose a C3D to preview the kinematic chain",
                )
                return
            scene = self._settings_scene or self._build_settings_scene()
            message = str(scene.get("message", ""))
            if message:
                painter.drawText(self.rect(), qt_alignment_center, message)
                return
            segment_origins = scene["segment_origins"]
            segment_frames = scene["segment_frames"]
            links = scene["links"]
            frame_length = scene["frame_length"]
            marker_points = scene["marker_points"]
            points = scene["points"]
            if len(points) == 0:
                painter.drawText(
                    self.rect(),
                    qt_alignment_center,
                    "No segment origin can be resolved yet",
                )
                return
            transform = _fit_projection(
                [_rotate_preview_point(point, self.yaw, self.pitch) for point in points],
                self.width(),
                self.height(),
                QPointF,
                self.zoom,
            )

            _set_preview_label_font(painter)
            self._draw_all_c3d_markers(painter, transform, marker_points)
            settings_by_segment = self._settings_by_segment()
            for _parent_name, child_name, start, end in links:
                selected = child_name == self.selected_setting_segment_name
                width = 3 if selected else 2
                color = QColor("#475569") if selected else QColor("#94a3b8")
                painter.setPen(QPen(color, width))
                painter.drawLine(
                    transform(_rotate_preview_point(start, self.yaw, self.pitch)),
                    transform(_rotate_preview_point(end, self.yaw, self.pitch)),
                )

            for segment_index, group in enumerate(self.workflow_draft.segment_marker_groups):
                origin = segment_origins.get(group.segment_name)
                if origin is None:
                    continue
                setting = settings_by_segment.get(group.segment_name)
                axes = segment_frames.get(group.segment_name, {})
                selected = group.segment_name == self.selected_setting_segment_name
                self._draw_segment_frame(
                    painter,
                    transform,
                    group.segment_name,
                    origin,
                    axes,
                    setting,
                    selected,
                    segment_index,
                    frame_length,
                )

            _draw_preview_orientation_axes(painter, self.width(), self.height(), self.yaw, self.pitch)
            _draw_preview_interaction_hint(painter, self.width())
            if _should_draw_preview_labels(self):
                self._draw_settings_legend(painter)

        def _build_settings_scene(self) -> dict[str, object]:
            segment_origins = {}
            segment_frames = {}
            marker_points = self._visible_marker_points()
            points = []
            if self.workflow_draft is None:
                return {
                    "segment_origins": segment_origins,
                    "segment_frames": segment_frames,
                    "links": [],
                    "frame_length": 1.0,
                    "marker_points": marker_points,
                    "points": points,
                }
            points.extend(marker_points)
            for group in self.workflow_draft.segment_marker_groups:
                origin = self._segment_origin(group)
                if origin is None:
                    continue
                axes = self._segment_local_axes(group.segment_name)
                segment_origins[group.segment_name] = origin
                segment_frames[group.segment_name] = axes
                points.append(origin)

            links = []
            for group in self.workflow_draft.segment_marker_groups:
                parent_name = group.parent_name
                if parent_name in {"", "root"}:
                    continue
                start = segment_origins.get(parent_name)
                end = segment_origins.get(group.segment_name)
                if start is None or end is None:
                    continue
                links.append((parent_name, group.segment_name, start, end))
                points.extend((start, end))

            scene_span = self._scene_span(points)
            frame_length = max(scene_span * 0.12, 1e-6)
            for segment_name, origin in segment_origins.items():
                for axis_vector in segment_frames.get(segment_name, {}).values():
                    endpoint = np.asarray(origin, dtype=float) + frame_length * axis_vector
                    points.append(tuple(float(value) for value in endpoint))
            return {
                "segment_origins": segment_origins,
                "segment_frames": segment_frames,
                "links": links,
                "frame_length": frame_length,
                "marker_points": marker_points,
                "points": points,
            }

        @staticmethod
        def _empty_settings_scene(message: str = "") -> dict[str, object]:
            return {
                "segment_origins": {},
                "segment_frames": {},
                "links": [],
                "frame_length": 1.0,
                "marker_points": [],
                "points": [],
                "message": message,
            }

        def _visible_marker_points(self) -> list[tuple[float, float, float]]:
            if self.c3d_data is None:
                return []
            points = []
            for marker_name in self.c3d_data.marker_names:
                point = _marker_frame_position(self.c3d_data, marker_name, self.frame_index)
                if point is not None:
                    points.append(point)
            return points

        def _draw_all_c3d_markers(self, painter, transform, marker_points: list[tuple[float, float, float]]) -> None:
            painter.setPen(QPen(QColor("#cbd5e1"), 1))
            painter.setBrush(QColor("#cbd5e1"))
            for point in sorted(
                marker_points,
                key=lambda marker_point: _preview_depth(marker_point, self.yaw, self.pitch),
            ):
                center = transform(_rotate_preview_point(point, self.yaw, self.pitch))
                painter.drawEllipse(center, 2, 2)

        def _segment_origin(self, group) -> tuple[float, float, float] | None:
            axes = self._local_frame_axes(group.segment_name)
            for axis in axes:
                origin = self._mean_source_position(axis.origin_markers)
                if origin is not None:
                    return origin
            marker_names = tuple(dict.fromkeys(group.marker_names + group.technical_marker_names))
            return self._mean_source_position(marker_names)

        def _segment_local_axes(self, segment_name: str) -> dict[str, np.ndarray]:
            vector_segments = []
            for axis in self._local_frame_axes(segment_name)[:2]:
                start, end = self._line_points_from_axis_sources(axis.start_markers, axis.end_markers)
                if start is not None and end is not None:
                    vector_segments.append((axis.axis, axis.keep_vector, start, end))
            return _orthonormal_axes_from_vector_segments(tuple(vector_segments))

        def _local_frame_axes(self, segment_name: str):
            return tuple(
                axis
                for axis in self.workflow_draft.axes
                if axis.segment_name == segment_name and not _is_virtual_feature_axis(axis)
            )

        def _settings_by_segment(self) -> dict[str, object]:
            settings_by_segment = {setting.segment_name: setting for setting in self.workflow_draft.segment_settings}
            settings_by_segment.update(self.segment_setting_overrides)
            return settings_by_segment

        def _draw_segment_frame(
            self,
            painter,
            transform,
            segment_name: str,
            origin: tuple[float, float, float],
            axes: dict[str, np.ndarray],
            setting,
            selected: bool,
            segment_index: int,
            frame_length: float,
        ) -> None:
            origin_array = np.asarray(origin, dtype=float)
            origin_screen = transform(_rotate_preview_point(origin, self.yaw, self.pitch))
            marker_color = QColor(_segment_preview_color(segment_index))
            painter.setBrush(marker_color if selected else QColor("white"))
            painter.setPen(QPen(marker_color, 3 if selected else 2))
            painter.drawEllipse(origin_screen, 6 if selected else 4, 6 if selected else 4)
            painter.setPen(QPen(QColor("#111827"), 1))
            if _should_draw_preview_labels(self):
                painter.setPen(QPen(QColor("#111827"), 1))
                painter.drawText(origin_screen.x() + 7, origin_screen.y() - 7, segment_name)

            translation_axes = self._translation_dof_axis_names(setting)
            rotation_axes = self._rotation_dof_axis_names(setting)
            for axis_name in ("x", "y", "z"):
                axis_vector = axes.get(axis_name)
                if axis_vector is None:
                    continue
                endpoint = origin_array + frame_length * axis_vector
                endpoint_screen = transform(_rotate_preview_point(tuple(endpoint), self.yaw, self.pitch))
                has_rotation_dof = axis_name in rotation_axes
                has_translation_dof = axis_name in translation_axes
                axis_width = 7 if has_rotation_dof else (4 if has_translation_dof else 2)
                if selected and axis_width < 4:
                    axis_width = 4
                painter.setPen(QPen(QColor(_preview_axis_color(axis_name)), axis_width))
                painter.drawLine(origin_screen, endpoint_screen)
                if has_translation_dof:
                    self._draw_translation_dof_arrow(
                        painter,
                        origin_screen,
                        endpoint_screen,
                        QColor(_preview_axis_color(axis_name)),
                    )
                if _should_draw_preview_labels(self):
                    painter.drawText(
                        endpoint_screen.x() + 4,
                        endpoint_screen.y() - 4,
                        axis_name.upper(),
                    )

        @staticmethod
        def _draw_translation_dof_arrow(painter, start, end, color) -> None:
            dx = float(end.x() - start.x())
            dy = float(end.y() - start.y())
            norm = math.hypot(dx, dy)
            if norm <= 1e-9:
                return
            ux = dx / norm
            uy = dy / norm
            arrow_start = QPointF(start.x() + ux * 9.0, start.y() + uy * 9.0)
            arrow_end = QPointF(start.x() + dx * 0.86, start.y() + dy * 0.86)
            painter.setPen(QPen(color, 5))
            painter.drawLine(arrow_start, arrow_end)
            head_size = 10.0
            left = QPointF(
                arrow_end.x() - ux * head_size - uy * head_size * 0.65,
                arrow_end.y() - uy * head_size + ux * head_size * 0.65,
            )
            right = QPointF(
                arrow_end.x() - ux * head_size + uy * head_size * 0.65,
                arrow_end.y() - uy * head_size - ux * head_size * 0.65,
            )
            painter.drawLine(arrow_end, left)
            painter.drawLine(arrow_end, right)

        def _draw_settings_legend(self, painter) -> None:
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.setBrush(QColor(255, 255, 255, 230))
            painter.drawRect(8, 8, 295, 120)
            painter.drawText(14, 24, "Legend")
            painter.setPen(QPen(QColor("#94a3b8"), 2))
            painter.drawLine(16, 42, 42, 42)
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.drawText(50, 46, "Parent-child translation")
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.drawText(16, 66, "Local axes: X/Y/Z = red/green/blue")
            painter.setPen(QPen(QColor("#dc2626"), 7))
            painter.drawLine(18, 84, 42, 84)
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.drawText(50, 88, "Rotation DoF: thick axis")
            painter.setPen(QPen(QColor("#2563eb"), 5))
            painter.drawLine(18, 108, 42, 108)
            painter.drawLine(QPointF(42, 108), QPointF(35, 104))
            painter.drawLine(QPointF(42, 108), QPointF(35, 112))
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.drawText(50, 112, "Translation DoF: arrow")

        @staticmethod
        def _axis_names_from_sequence(sequence) -> set[str]:
            return {
                axis_name
                for axis_name in str(sequence).strip().replace("-", "").lower()
                if axis_name in {"x", "y", "z"}
            }

        @classmethod
        def _translation_dof_axis_names(cls, setting) -> set[str]:
            if setting is None:
                return set()
            return cls._axis_names_from_sequence(setting.translations)

        @classmethod
        def _rotation_dof_axis_names(cls, setting) -> set[str]:
            if setting is None:
                return set()
            return cls._axis_names_from_sequence(setting.rotations)

        @classmethod
        def _dof_axis_names(cls, setting) -> set[str]:
            return cls._translation_dof_axis_names(setting) | cls._rotation_dof_axis_names(setting)

        @classmethod
        def _has_translation_dof(cls, setting) -> bool:
            return bool(cls._translation_dof_axis_names(setting))

        @classmethod
        def _has_rotation_dof(cls, setting) -> bool:
            return bool(cls._rotation_dof_axis_names(setting))

        @staticmethod
        def _scene_span(points: list[tuple[float, float, float]]) -> float:
            if len(points) == 0:
                return 1.0
            positions = np.asarray(points, dtype=float)
            span = float(np.nanmax(np.ptp(positions, axis=0)))
            return span if np.isfinite(span) and span > 0 else 1.0

    class C3dModelCreationDialog(QDialog):
        """
        Dialog for the C3D-driven model creation workflow.
        """

        def __init__(
            self,
            *,
            initial_preset: str | C3dModelPreset | None = None,
            initial_c3d_folder: str | Path | None = None,
        ):
            super().__init__()
            self.setWindowTitle("New model from C3D")
            self.presets = supported_c3d_model_presets()
            self.c3d_data = None
            self._c3d_data_cache = {}
            self._virtual_feature_c3d_preview_data_cache = None
            self._virtual_feature_c3d_preview_data_cache_signature = None
            self._segment_settings_q0_scene_cache = None
            self._segment_settings_q0_scene_cache_signature = None
            self._is_auto_assigning_c3d_files = False
            self.c3d_folder_path = ""
            self.workflow_draft = c3d_workflow_draft(self.presets[0])
            self.workflow_marker_pool = _marker_pool_from_draft(self.workflow_draft)
            self._suspend_anatomical_preview_updates = False
            self._suspend_segment_settings_preview_updates = False
            self._workflow_playback_slider = None
            self._workflow_playback_fps = WORKFLOW_PLAYBACK_DEFAULT_FPS
            self._workflow_frame_play_buttons = {}
            self._workflow_playback_timer = QTimer(self)
            self._workflow_playback_timer.setInterval(_workflow_playback_timer_interval_ms(None))
            self._workflow_playback_timer.timeout.connect(self._advance_workflow_frame_playback)

            self.preset_combo = QComboBox()
            for preset in self.presets:
                self.preset_combo.addItem(_c3d_preset_label(preset))
            self.preset_combo.currentIndexChanged.connect(self._update_preset_details)

            self.c3d_path = QLineEdit()
            self.c3d_path.setReadOnly(True)
            self.c3d_path.hide()
            self.main_c3d_name_label = QLabel("No main C3D selected")
            self.main_c3d_name_label.setObjectName("MutedInfoLabel")
            self.choose_c3d_button = QPushButton("Choose C3D file")
            self.choose_c3d_button.setObjectName("SecondaryActionButton")
            self.choose_c3d_button.clicked.connect(self._choose_c3d_file)
            self.generate_template_button = QPushButton("Generate template")
            self.generate_template_button.clicked.connect(self._generate_template)
            self.c3d_folder_edit = QLineEdit()
            self.c3d_folder_edit.setReadOnly(True)
            self.choose_c3d_folder_button = QPushButton("Choose C3D folder")
            self.choose_c3d_folder_button.setObjectName("SecondaryActionButton")
            self.choose_c3d_folder_button.clicked.connect(self._choose_c3d_folder)
            self.generate_log_button = QPushButton("Generate log")
            self.generate_log_button.clicked.connect(self._update_generation_log)
            self.generate_python_code_button = QPushButton("Generate Python code")
            self.generate_python_code_button.clicked.connect(self._generate_python_code)
            self.show_summary_button = QPushButton("Show summary")
            self.show_summary_button.clicked.connect(self._show_workflow_summary)
            self.generation_log_edit = QTextEdit()
            self.generation_log_edit.setReadOnly(True)
            self.generation_log_edit.setMinimumHeight(160)

            self.status_label = QLabel()
            self.status_label.setObjectName("WorkflowStatusLabel")
            self.c3d_names_overview_label = QLabel()
            self.c3d_names_overview_label.setObjectName("MutedInfoLabel")
            self.c3d_names_overview_label.setWordWrap(True)
            self.c3d_marker_status_label = QLabel()
            self.c3d_marker_status_label.setObjectName("MutedInfoLabel")
            self.c3d_marker_status_label.setWordWrap(True)
            self.feature_list = QListWidget()
            self.feature_list.setMinimumHeight(180)
            self.feature_list.itemSelectionChanged.connect(self._load_selected_virtual_marker_into_form)
            self.step_list = QListWidget()
            self.marker_list = QListWidget()
            self.marker_list.setSelectionMode(qt_extended_selection)
            self.marker_list.itemSelectionChanged.connect(self._update_technical_segment_preview)
            self.marker_mapping_label = QLabel("Load a C3D to check marker names against the selected template.")
            self.marker_mapping_label.setObjectName("MutedInfoLabel")
            self.marker_mapping_label.setWordWrap(True)
            self.strip_participant_prefix_checkbox = QCheckBox("Remove participant prefix before ':'")
            self.strip_participant_prefix_checkbox.setChecked(True)
            self.strip_participant_prefix_checkbox.stateChanged.connect(self._reload_current_c3d_file)
            self.show_all_markers_checkbox = QCheckBox("Show markers already used by other segments")
            self.show_all_markers_checkbox.setChecked(True)
            self.show_all_markers_checkbox.stateChanged.connect(self._update_available_marker_list)
            self.show_virtual_markers_in_segments_checkbox = QCheckBox("Show virtual markers")
            self.show_virtual_markers_in_segments_checkbox.setChecked(True)
            self.show_virtual_markers_in_segments_checkbox.stateChanged.connect(self._update_available_marker_list)
            self.segment_marker_list = QListWidget()
            self.segment_marker_list.itemSelectionChanged.connect(self._update_assigned_marker_list)
            self.technical_segment_preview = C3dTechnicalSegmentPreviewWidget()
            _style_preview_widget(self.technical_segment_preview)
            self.technical_frame_slider = QSlider(qt_horizontal)
            self.technical_frame_slider.setEnabled(False)
            self.technical_frame_slider.valueChanged.connect(self._update_technical_segment_preview)
            self.technical_frame_label = QLabel("Frame 1/1")
            self.workflow_parent_combo = QComboBox()
            self.workflow_parent_combo.currentTextChanged.connect(self._set_workflow_segment_parent)
            self.assigned_marker_list = QListWidget()
            self.assigned_marker_list.setSelectionMode(qt_extended_selection)
            self.assigned_marker_list.itemSelectionChanged.connect(self._sync_assigned_marker_technical_checkbox)
            self.assigned_marker_technical_checkbox = QCheckBox("Selected markers are technical")
            self.assigned_marker_technical_checkbox.stateChanged.connect(self._set_selected_assigned_markers_technical)
            self.axis_list = QListWidget()
            self.axis_list.setMaximumWidth(720)
            self.axis_list.setMaximumHeight(120)
            self.anatomical_segment_list = QListWidget()
            self.anatomical_segment_list.setMaximumWidth(720)
            self.anatomical_segment_list.setMaximumHeight(110)
            self.anatomical_segment_list.itemSelectionChanged.connect(self._update_anatomical_segment_details)
            self.axis_marker_source_list = _marker_list_widget(min_height=220, min_width=140, max_width=180)
            self.axis_origin_marker_list = _marker_list_widget(min_height=52, max_height=76, max_width=190)
            self.add_axis_origin_marker_button = _small_button("+")
            self.remove_axis_origin_marker_button = _small_button("-")
            self.add_axis_origin_marker_button.clicked.connect(self._add_selected_axis_origin_markers)
            self.remove_axis_origin_marker_button.clicked.connect(self._remove_selected_axis_origin_markers)
            self.axis_vector_controls = [_create_axis_vector_controls(index) for index in range(2)]
            for index, controls in enumerate(self.axis_vector_controls):
                controls["add_start_button"].clicked.connect(
                    lambda checked=False, vector_index=index: self._add_selected_axis_markers(vector_index, "start")
                )
                controls["add_end_button"].clicked.connect(
                    lambda checked=False, vector_index=index: self._add_selected_axis_markers(vector_index, "end")
                )
                controls["remove_start_button"].clicked.connect(
                    lambda checked=False, vector_index=index: self._remove_selected_axis_markers(vector_index, "start")
                )
                controls["remove_end_button"].clicked.connect(
                    lambda checked=False, vector_index=index: self._remove_selected_axis_markers(vector_index, "end")
                )
                controls["swap_button"].clicked.connect(
                    lambda checked=False, vector_index=index: self._swap_axis_vector_endpoints(vector_index)
                )
                controls["axis_combo"].currentTextChanged.connect(
                    lambda text, combo=controls["axis_combo"]: _style_axis_combo(combo)
                )
                controls["axis_combo"].currentTextChanged.connect(self._update_segment_axis_preview)
                controls["keep_checkbox"].stateChanged.connect(
                    lambda checked=False, vector_index=index: self._ensure_single_kept_axis_vector(vector_index)
                )
            self.save_segment_axis_button = QPushButton("Add/update anatomical frame vectors")
            self.save_segment_axis_button.clicked.connect(self._save_segment_axis_from_lists)
            self.segment_axis_preview = C3dSegmentAxisPreviewWidget()
            _style_preview_widget(self.segment_axis_preview)
            self.segment_axis_preview.setMinimumWidth(360)
            self.segment_axis_preview.setMinimumHeight(520)
            self.anatomical_frame_slider = QSlider(qt_horizontal)
            self.anatomical_frame_slider.setEnabled(False)
            self.anatomical_frame_slider.valueChanged.connect(self._update_segment_axis_preview)
            self.anatomical_frame_label = QLabel("Frame 1/1")
            self.virtual_marker_name_edit = QLineEdit()
            self.virtual_marker_suggested_name_label = QLabel()
            self.virtual_marker_suggested_name_label.setObjectName("MutedInfoLabel")
            self.virtual_marker_segment_combo = QComboBox()
            self.virtual_marker_segment_combo.currentTextChanged.connect(self._update_suggested_virtual_marker_name)
            self.virtual_marker_segment_combo.currentTextChanged.connect(self._sync_virtual_marker_segment_context)
            self.virtual_marker_method_combo = QComboBox()
            self.virtual_marker_method_combo.addItems(
                [
                    "pointing",
                    "score",
                    "sara_direction",
                    "marker_mean",
                    _virtual_marker_method_display("axis_projection"),
                    "predictive",
                ]
            )
            self.virtual_marker_method_combo.currentTextChanged.connect(self._sync_virtual_marker_method_fields)
            self.virtual_marker_predictive_method_combo = QComboBox()
            self.virtual_marker_predictive_method_combo.addItems(list(PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS.values()))
            self.virtual_marker_predictive_method_combo.currentTextChanged.connect(
                self._sync_virtual_marker_method_fields
            )
            self.virtual_marker_source_edit = QLineEdit()
            self.virtual_marker_source_edit.textChanged.connect(self._update_virtual_marker_preview)
            self.virtual_marker_source_label = QLabel("Source")
            self.virtual_marker_source_edit.hide()
            self.virtual_marker_source_label.hide()
            self.virtual_marker_projection_available_list = _marker_list_widget(max_height=112, max_width=320)
            self.virtual_marker_projected_marker_list = _marker_list_widget(max_height=70, max_width=240)
            self.virtual_marker_projection_axis_list = _marker_list_widget(max_height=70, max_width=240)
            self.add_projected_marker_button = _small_button("+")
            self.remove_projected_marker_button = _small_button("-")
            self.add_projection_axis_button = _small_button("+")
            self.remove_projection_axis_button = _small_button("-")
            self.add_projected_marker_button.clicked.connect(self._add_projection_source_markers)
            self.remove_projected_marker_button.clicked.connect(self._remove_projection_source_markers)
            self.add_projection_axis_button.clicked.connect(self._add_projection_axis_source)
            self.remove_projection_axis_button.clicked.connect(self._remove_projection_axis_source)
            self.virtual_marker_c3d_file_combo = QComboBox()
            self.virtual_marker_c3d_file_combo.currentTextChanged.connect(self._update_virtual_marker_preview)
            self.virtual_marker_c3d_file_combo.currentTextChanged.connect(self._update_suggested_virtual_marker_name)
            self.browse_virtual_marker_c3d_button = QPushButton("Browse C3D")
            self.browse_virtual_marker_c3d_button.setObjectName("SecondaryActionButton")
            self.browse_virtual_marker_c3d_button.clicked.connect(self._browse_virtual_marker_c3d_source)
            self._virtual_marker_preview_mode = "functional"
            self._segment_settings_preview_mode = "static"
            self.workflow_preview_mode_combo = QComboBox()
            self.workflow_preview_mode_combo.setMinimumWidth(120)
            self.workflow_preview_mode_combo.currentTextChanged.connect(self._set_workflow_preview_mode)
            self.virtual_marker_whole_body_preview_checkbox = QCheckBox("Whole body view")
            self.virtual_marker_whole_body_preview_checkbox.setChecked(False)
            self.virtual_marker_whole_body_preview_checkbox.stateChanged.connect(self._update_virtual_marker_preview)
            self.use_manual_functional_frames_checkbox = QCheckBox("Use selected frames")
            self.use_manual_functional_frames_checkbox.setToolTip(
                "Use the blue frame zones under the preview frame slider for functional SCoRE/SARA calculations."
            )
            self.use_manual_functional_frames_checkbox.stateChanged.connect(self._set_manual_functional_frames_enabled)
            self.use_diverse_functional_frames_checkbox = QCheckBox("Use diverse functional frames")
            self.use_diverse_functional_frames_checkbox.setToolTip(
                "Select a smaller set of valid functional frames with different parent-child rototranslations."
            )
            self.use_diverse_functional_frames_checkbox.stateChanged.connect(
                self._set_diverse_functional_frames_enabled
            )
            self.explore_functional_residuals_button = QPushButton("Explore SCoRE/SARA residuals")
            self.explore_functional_residuals_button.setObjectName("SecondaryActionButton")
            self.explore_functional_residuals_button.setEnabled(False)
            self.explore_functional_residuals_button.clicked.connect(self._show_functional_residual_diagnostics)
            self.functional_reconstruction_feature_combo = QComboBox()
            self.functional_reconstruction_feature_combo.setMinimumWidth(240)
            self.functional_reconstruction_feature_combo.setMinimumContentsLength(28)
            self.functional_reconstruction_feature_combo.currentIndexChanged.connect(
                self._sync_functional_reconstruction_trial_choices
            )
            self.functional_reconstruction_feature_combo.currentIndexChanged.connect(
                self._update_virtual_marker_preview
            )
            self.functional_reconstruction_trial_combo = QComboBox()
            self.functional_reconstruction_trial_combo.setMinimumWidth(240)
            self.functional_reconstruction_trial_combo.setMinimumContentsLength(28)
            self.functional_reconstruction_trial_combo.currentIndexChanged.connect(self._update_virtual_marker_preview)
            self.functional_reconstruction_method_combo = QComboBox()
            self.functional_reconstruction_method_combo.addItems(["QLD", "EKF"])
            self.functional_reconstruction_method_combo.setMaximumWidth(110)
            self.functional_reconstruction_method_combo.setToolTip(
                "EKF uses biorbd Kalman reconstruction when available. QLD uses BioBuddy/biorbd least-squares IK."
            )
            self.run_functional_reconstruction_button = QPushButton("Run diagnostic")
            self.run_functional_reconstruction_button.setMinimumWidth(150)
            self.run_functional_reconstruction_button.setObjectName("SecondaryActionButton")
            self.run_functional_reconstruction_button.clicked.connect(self._run_functional_reconstruction_diagnostic)
            self.functional_reconstruction_output = QTextEdit()
            self.functional_reconstruction_output.setReadOnly(True)
            self.functional_reconstruction_output.setMinimumHeight(220)
            self.functional_reconstruction_plot = FunctionalReconstructionPlotWidget()
            self.virtual_marker_frame_slider = QSlider(qt_horizontal)
            self.virtual_marker_frame_slider.setEnabled(False)
            self.virtual_marker_frame_slider.valueChanged.connect(self._update_virtual_marker_preview)
            self.virtual_marker_frame_label = QLabel("Frame 1/1")
            self.functional_frame_range_bar = FunctionalFrameRangeBar()
            self.functional_frame_range_bar.on_frame_dragged = self._set_virtual_marker_frame_from_range_drag
            self.functional_frame_range_bar.on_selection_changed = self._update_manual_functional_frame_selection
            self.functional_frame_selection_label = QLabel("Selected frames")
            self.functional_frame_selection_label.setObjectName("MutedInfoLabel")
            self.functional_frame_selection_label.setToolTip(
                "Drag this row at any time to paint useful intervals. Enable 'Use selected frames' "
                "to apply the blue zones to SCoRE/SARA calculations."
            )
            self.virtual_marker_equation_edit = QLineEdit()
            self.virtual_marker_equation_edit.hide()
            self.virtual_marker_c3d_role_combo = QComboBox()
            self.virtual_marker_c3d_role_combo.hide()
            self.virtual_marker_parent_label = QLabel("-")
            self.virtual_marker_technical_markers_label = QLabel()
            self.virtual_marker_technical_markers_label.setObjectName("MutedInfoLabel")
            self.virtual_marker_technical_markers_label.setWordWrap(True)
            self.virtual_marker_proximal_combo = QComboBox()
            self.virtual_marker_proximal_combo.currentTextChanged.connect(self._update_virtual_marker_preview)
            self.virtual_marker_proximal_combo.currentTextChanged.connect(self._update_suggested_virtual_marker_name)
            self.virtual_marker_proximal_combo.currentTextChanged.connect(self._update_virtual_marker_technical_markers)
            self.virtual_marker_distal_combo = QComboBox()
            self.virtual_marker_distal_combo.currentTextChanged.connect(self._update_virtual_marker_preview)
            self.virtual_marker_distal_combo.currentTextChanged.connect(self._update_suggested_virtual_marker_name)
            self.virtual_marker_distal_combo.currentTextChanged.connect(self._update_virtual_marker_technical_markers)
            self.virtual_marker_info_label = QLabel()
            self.virtual_marker_info_label.setObjectName("MutedInfoLabel")
            self.virtual_marker_info_label.setWordWrap(True)
            self.save_virtual_marker_button = QPushButton("Save virtual marker/axis")
            self.save_virtual_marker_button.clicked.connect(self._save_workflow_virtual_marker_from_form)
            self.virtual_marker_preview = C3dVirtualMarkerPreviewWidget()
            _style_preview_widget(self.virtual_marker_preview)
            self.segment_settings_list = QListWidget()
            self.segment_settings_list.itemSelectionChanged.connect(self._load_selected_segment_settings_into_form)
            self.segment_settings_preview = C3dSegmentSettingsPreviewWidget()
            _style_preview_widget(self.segment_settings_preview)
            self.workflow_view_button = QPushButton("View")
            self.workflow_view_button.setObjectName("SecondaryActionButton")
            self.workflow_view_button.setMenu(self._create_workflow_view_menu())
            self.segment_settings_frame_slider = QSlider(qt_horizontal)
            self.segment_settings_frame_slider.setEnabled(False)
            self.segment_settings_frame_slider.valueChanged.connect(self._update_segment_settings_preview)
            self.segment_settings_frame_label = QLabel("Frame 1/1")
            self.settings_translations_edit = _AxisSequencePicker(allow_first_third_repeat=False)
            self.settings_translations_edit.setMaximumWidth(220)
            self.settings_translations_edit.setToolTip("Choose translation axes. Each axis can be selected only once.")
            self.settings_rotations_edit = _AxisSequencePicker(allow_first_third_repeat=True)
            self.settings_rotations_edit.setMaximumWidth(220)
            self.settings_rotations_edit.setToolTip(
                "Choose rotation axes. The middle axis cannot be the same as the first or third axis."
            )
            self.settings_q_min_edit = QLineEdit()
            self.settings_q_min_edit.setMaximumWidth(120)
            self.settings_q_min_edit.setToolTip("Leave empty, or enter one q min value per translation/rotation DoF.")
            self.settings_q_max_edit = QLineEdit()
            self.settings_q_max_edit.setMaximumWidth(120)
            self.settings_q_max_edit.setToolTip("Leave empty, or enter one q max value per translation/rotation DoF.")
            self.settings_translations_edit.textChanged.connect(self._sync_segment_settings_validation_style)
            self.settings_rotations_edit.textChanged.connect(self._sync_segment_settings_validation_style)
            self.settings_q_min_edit.textChanged.connect(self._sync_segment_settings_validation_style)
            self.settings_q_max_edit.textChanged.connect(self._sync_segment_settings_validation_style)
            self.settings_child_translation_checkbox = QCheckBox("Allow child translation")
            self.settings_translations_edit.textChanged.connect(self._update_segment_settings_preview)
            self.settings_rotations_edit.textChanged.connect(self._update_segment_settings_preview)
            self.settings_child_translation_checkbox.stateChanged.connect(self._update_segment_settings_preview)
            self.settings_initial_rotation_method_combo = QComboBox()
            self.settings_initial_rotation_method_combo.addItems(["identity", "matrix", "anatomical_c3d"])
            self.settings_initial_rotation_method_combo.setToolTip(
                "identity: use an identity initial RT rotation.\n"
                "matrix: apply the 3x3 rotation matrix entered in Matrix.\n"
                "anatomical_c3d: compute the initial rotation from an anatomical/static C3D.\n"
                "Use non-identity modes only when the next segment must be pre-oriented, typically for an AoR."
            )
            self.settings_initial_rotation_method_combo.currentTextChanged.connect(
                self._sync_initial_rotation_source_fields
            )
            self.settings_initial_rotation_source_edit = QLineEdit()
            self.settings_initial_rotation_source_edit.setMaximumWidth(200)
            self.settings_initial_rotation_source_edit.setPlaceholderText("[[1,0,0], [0,1,0], [0,0,1]]")
            self.settings_initial_rotation_source_edit.setToolTip(
                "Accepted matrix formats:\n"
                "[[a,b,c], [d,e,f], [g,h,i]]\n"
                "a,b,c; d,e,f; g,h,i\n"
                "a b c d e f g h i\n"
                "The values must define exactly 9 finite numbers."
            )
            self.settings_initial_rotation_source_edit.textChanged.connect(self._sync_segment_settings_validation_style)
            self.settings_initial_rotation_c3d_combo = QComboBox()
            self.settings_initial_rotation_c3d_combo.setMaximumWidth(220)
            self.settings_initial_rotation_c3d_combo.currentTextChanged.connect(
                self._mirror_initial_rotation_c3d_source
            )
            self.browse_initial_rotation_c3d_button = QPushButton("Browse C3D")
            self.browse_initial_rotation_c3d_button.setObjectName("SecondaryActionButton")
            self.browse_initial_rotation_c3d_button.clicked.connect(self._browse_initial_rotation_c3d_source)
            self.settings_segment_length_edit = QLineEdit()
            self.settings_segment_length_edit.setReadOnly(True)
            self.settings_segment_length_edit.setMaximumWidth(140)
            self.settings_segment_length_source_label = QLabel("-")
            self.settings_segment_length_source_label.setObjectName("MutedInfoLabel")
            self.settings_segment_length_source_label.setWordWrap(True)
            self.settings_segment_length_source_label.setMaximumWidth(520)
            self.settings_anthropometry_model_combo = QComboBox()
            self.settings_anthropometry_model_combo.setMaximumWidth(140)
            self.settings_anthropometry_model_combo.addItems(["none", "de_leva"])
            self.settings_anthropometry_sex_combo = QComboBox()
            self.settings_anthropometry_sex_combo.setMaximumWidth(140)
            self.settings_anthropometry_sex_combo.addItems(["male", "female"])
            self.settings_anthropometry_mass_edit = QLineEdit()
            self.settings_anthropometry_mass_edit.setPlaceholderText("kg")
            self.settings_anthropometry_mass_edit.setMaximumWidth(140)
            self.apply_anthropometry_button = QPushButton("Apply anthropometry parameters")
            self.apply_anthropometry_button.clicked.connect(self._apply_anthropometry_to_selected_segment)
            self.file_role_list = QListWidget()
            self.issue_list = QListWidget()
            self.add_segment_button = QPushButton("Add segment")
            self.add_segment_button.clicked.connect(self._add_workflow_segment)
            self.remove_segment_button = QPushButton("Remove segment")
            self.remove_segment_button.setObjectName("DangerActionButton")
            self.remove_segment_button.clicked.connect(self._remove_workflow_segment)
            self.assign_marker_button = QPushButton("Assign selected markers")
            self.assign_marker_button.setObjectName("SmallIconButton")
            self.assign_marker_button.clicked.connect(self._assign_workflow_marker)
            self.unassign_marker_button = QPushButton("Remove selected markers")
            self.unassign_marker_button.setObjectName("SmallIconButton")
            self.unassign_marker_button.clicked.connect(self._unassign_workflow_marker)
            self.add_virtual_marker_button = QPushButton("Add/edit virtual marker/axis")
            self.add_virtual_marker_button.clicked.connect(self._add_workflow_virtual_marker)
            self.remove_virtual_marker_button = QPushButton("Remove virtual marker/axis")
            self.remove_virtual_marker_button.setObjectName("DangerActionButton")
            self.remove_virtual_marker_button.clicked.connect(self._remove_workflow_virtual_marker)
            self.edit_segment_settings_button = QPushButton("Apply segment settings")
            self.edit_segment_settings_button.clicked.connect(self._apply_workflow_segment_settings_from_form)
            self.chain_export_format_combo = QComboBox()
            self.chain_export_format_combo.addItems(
                ["BioMod (.bioMod)", "BVH (.bvh)", "OpenSim (.osim)", "URDF (.urdf)"]
            )
            self.chain_export_format_combo.setMaximumWidth(180)
            self.export_chain_model_button = QPushButton("Export model")
            self.export_chain_model_button.clicked.connect(self._export_workflow_chain_model)
            self.assign_c3d_role_button = QPushButton("Assign C3D file")
            self.assign_c3d_role_button.clicked.connect(self._assign_workflow_c3d_role)
            self.clear_c3d_role_button = QPushButton("Clear C3D file")
            self.clear_c3d_role_button.setObjectName("SecondaryActionButton")
            self.clear_c3d_role_button.clicked.connect(self._clear_workflow_c3d_role)
            for list_widget in (
                self.feature_list,
                self.step_list,
                self.marker_list,
                self.segment_marker_list,
                self.assigned_marker_list,
                self.axis_list,
                self.anatomical_segment_list,
                self.segment_settings_list,
                self.file_role_list,
                self.issue_list,
            ):
                list_widget.setAlternatingRowColors(True)
            for primary_button in (
                self.generate_template_button,
                self.save_segment_axis_button,
                self.save_virtual_marker_button,
                self.edit_segment_settings_button,
                self.export_chain_model_button,
            ):
                primary_button.setObjectName("PrimaryActionButton")
            for secondary_button in (
                self.generate_log_button,
                self.generate_python_code_button,
                self.show_summary_button,
                self.apply_anthropometry_button,
                self.add_segment_button,
                self.add_virtual_marker_button,
                self.assign_c3d_role_button,
            ):
                secondary_button.setObjectName("SecondaryActionButton")

            buttons = QDialogButtonBox(_dialog_button("Ok") | _dialog_button("Cancel"))
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)

            layout = QVBoxLayout(self)
            _configure_panel_layout(layout)

            left_pane = QWidget()
            left_layout = QVBoxLayout(left_pane)
            _configure_panel_layout(left_layout, margin=0)

            input_group = QGroupBox("Input data")
            input_layout = QVBoxLayout(input_group)
            _configure_panel_layout(input_layout, margin=10, spacing=8)
            preset_row = QHBoxLayout()
            _configure_panel_layout(preset_row, margin=0, spacing=8)
            preset_row.addWidget(_section_label("Model preset"))
            preset_row.addWidget(self.preset_combo, 1)
            input_layout.addLayout(preset_row)
            folder_row = QHBoxLayout()
            _configure_panel_layout(folder_row, margin=0, spacing=8)
            folder_row.addWidget(QLabel("C3D folder"))
            folder_row.addWidget(self.c3d_folder_edit)
            folder_row.addWidget(self.choose_c3d_folder_button)
            input_layout.addLayout(folder_row)
            c3d_row = QHBoxLayout()
            _configure_panel_layout(c3d_row, margin=0, spacing=8)
            c3d_row.addWidget(QLabel("Main C3D"))
            c3d_row.addWidget(self.main_c3d_name_label)
            c3d_row.addStretch()
            c3d_row.addWidget(self.choose_c3d_button)
            c3d_row.addWidget(self.generate_template_button)
            input_layout.addLayout(c3d_row)
            input_layout.addWidget(self.status_label)
            left_layout.addWidget(input_group)

            self.workflow_tabs = QTabWidget()
            self.workflow_tabs.addTab(self._pipeline_workflow_tab(), "Pipeline")
            self.workflow_tabs.addTab(self._segment_workflow_tab(), "Technical segment")
            self.workflow_tabs.addTab(self._virtual_marker_workflow_tab(), "Virtual markers and axes")
            self.workflow_tabs.addTab(self._anatomical_segment_workflow_tab(), "Anatomical segment")
            self.workflow_tabs.addTab(self._segment_settings_workflow_tab(), "Chain definition")
            self.workflow_tabs.addTab(
                self._functional_reconstruction_workflow_tab(),
                "Functional reconstruction",
            )
            self.workflow_tabs.addTab(self.issue_list, "Checks")
            left_layout.addWidget(self.workflow_tabs, 1)

            workflow_splitter = QSplitter(qt_horizontal)
            workflow_splitter.addWidget(left_pane)
            workflow_splitter.addWidget(self._workflow_preview_panel())
            workflow_splitter.setStretchFactor(0, 2)
            workflow_splitter.setStretchFactor(1, 1)
            workflow_splitter.setCollapsible(0, False)
            workflow_splitter.setCollapsible(1, False)
            workflow_splitter.setSizes([960, 480])
            self.workflow_tabs.currentChanged.connect(self._sync_workflow_preview_panel)
            layout.addWidget(workflow_splitter, 1)
            layout.addWidget(buttons)
            _resize_window_to_available_screen(self, QApplication, 1440, 820)
            self._update_preset_details()
            self.apply_initial_context(
                preset=initial_preset,
                c3d_folder=initial_c3d_folder,
            )
            self._sync_workflow_preview_panel(self.workflow_tabs.currentIndex())

        def selected_preset(self) -> C3dModelPreset:
            """
            Return the preset selected in the dialog.
            """
            return self.presets[self.preset_combo.currentIndex()]

        def selected_c3d_file(self) -> Path | None:
            """
            Return the selected C3D file, if one was chosen.
            """
            text = self.c3d_path.text().strip()
            return None if text == "" else Path(text)

        def selected_c3d_folder(self) -> Path | None:
            """
            Return the selected C3D folder, if one was chosen.
            """
            text = self.c3d_folder_path.strip()
            return None if text == "" else Path(text)

        def apply_initial_context(
            self,
            *,
            preset: str | C3dModelPreset | None = None,
            c3d_folder: str | Path | None = None,
        ) -> None:
            """
            Apply preset/folder values supplied by a launcher or CLI.
            """
            if preset is not None:
                preset_value = c3d_model_preset_from_cli_value(preset)
                if preset_value in self.presets:
                    self.preset_combo.setCurrentIndex(self.presets.index(preset_value))
            if c3d_folder is not None:
                self._set_c3d_folder(str(Path(c3d_folder).expanduser()))

        def _pipeline_workflow_tab(self):
            widget = QWidget()
            layout = QVBoxLayout(widget)
            _configure_panel_layout(layout)
            layout.addWidget(_section_label("Workflow status"))
            layout.addWidget(self.step_list)
            log_row = QHBoxLayout()
            _configure_panel_layout(log_row, margin=0, spacing=8)
            log_row.addWidget(_section_label("Generation log"))
            log_row.addStretch()
            log_row.addWidget(self.show_summary_button)
            log_row.addWidget(self.generate_python_code_button)
            log_row.addWidget(self.generate_log_button)
            layout.addLayout(log_row)
            layout.addWidget(self.generation_log_edit)
            return widget

        def _workflow_preview_panel(self):
            """
            Build the fixed right-hand 3D preview area shared by the workflow tabs.
            """
            panel = QWidget()
            panel.setObjectName("WorkflowPreviewPanel")
            panel.setMinimumWidth(360)
            layout = QVBoxLayout(panel)
            _configure_panel_layout(layout)
            self.workflow_preview_title_label = _section_label("3D preview")
            layout.addWidget(self.workflow_preview_title_label)

            options_row = QHBoxLayout()
            _configure_panel_layout(options_row, margin=0, spacing=8)
            options_row.addWidget(QLabel("Mode"))
            options_row.addWidget(self.workflow_preview_mode_combo)
            options_row.addWidget(self.virtual_marker_whole_body_preview_checkbox)
            options_row.addStretch()
            options_row.addWidget(self.workflow_view_button)
            layout.addLayout(options_row)

            self.workflow_empty_frame_slider = QSlider(qt_horizontal)
            self.workflow_empty_frame_slider.setEnabled(False)
            self.workflow_empty_frame_label = QLabel("Frame 0/0")
            self.workflow_frame_stack = QStackedWidget()
            self.workflow_frame_empty_index = self.workflow_frame_stack.addWidget(
                self._workflow_frame_slider_page(self.workflow_empty_frame_slider, self.workflow_empty_frame_label)
            )
            self.workflow_frame_technical_index = self.workflow_frame_stack.addWidget(
                self._workflow_frame_slider_page(self.technical_frame_slider, self.technical_frame_label)
            )
            self.workflow_frame_virtual_index = self.workflow_frame_stack.addWidget(
                self._workflow_frame_slider_page(
                    self.virtual_marker_frame_slider,
                    self.virtual_marker_frame_label,
                    self.functional_frame_range_bar,
                    self.functional_frame_selection_label,
                )
            )
            self.workflow_frame_anatomical_index = self.workflow_frame_stack.addWidget(
                self._workflow_frame_slider_page(self.anatomical_frame_slider, self.anatomical_frame_label)
            )
            self.workflow_frame_segment_settings_index = self.workflow_frame_stack.addWidget(
                self._workflow_frame_slider_page(
                    self.segment_settings_frame_slider,
                    self.segment_settings_frame_label,
                )
            )
            layout.addWidget(self.workflow_frame_stack)

            self.workflow_canvas_stack = QStackedWidget()
            empty_label = _muted_label(
                "The fixed preview is used while assigning technical markers, defining virtual markers and axes, "
                "or building anatomical segment frames."
            )
            empty_label.setAlignment(qt_alignment_center)
            self.workflow_canvas_empty_index = self.workflow_canvas_stack.addWidget(empty_label)
            self.workflow_canvas_technical_index = self.workflow_canvas_stack.addWidget(self.technical_segment_preview)
            self.workflow_canvas_virtual_index = self.workflow_canvas_stack.addWidget(self.virtual_marker_preview)
            self.workflow_canvas_anatomical_index = self.workflow_canvas_stack.addWidget(self.segment_axis_preview)
            self.workflow_canvas_segment_settings_index = self.workflow_canvas_stack.addWidget(
                self.segment_settings_preview
            )
            layout.addWidget(self.workflow_canvas_stack, 1)
            return panel

        def _workflow_frame_slider_page(self, slider, label, range_bar=None, range_label=None):
            """
            Return one fixed-height frame-control row for the shared preview panel.
            """
            page = QWidget()
            page_layout = QGridLayout(page)
            _configure_panel_layout(page_layout, margin=0, spacing=8)
            page_layout.setVerticalSpacing(2)
            play_button = _small_button(">")
            play_button.setToolTip("Play frames")
            play_button.setEnabled(slider.isEnabled() and slider.maximum() > slider.minimum())
            play_button.clicked.connect(
                lambda checked=False, frame_slider=slider: self._toggle_workflow_frame_playback(frame_slider)
            )
            self._workflow_frame_play_buttons[slider] = play_button
            page_layout.addWidget(QLabel("Frame"), 0, 0)
            page_layout.addWidget(play_button, 0, 1)
            page_layout.addWidget(slider, 0, 2)
            page_layout.addWidget(label, 0, 3)
            page_layout.setColumnStretch(2, 1)
            if range_bar is not None:
                page_layout.addWidget(range_bar, 1, 2)
                if range_label is not None:
                    page_layout.addWidget(range_label, 1, 3)
            return page

        def _toggle_workflow_frame_playback(self, slider) -> None:
            """
            Start or stop playback for one preview frame slider.
            """
            if self._workflow_playback_timer.isActive() and self._workflow_playback_slider is slider:
                self._stop_workflow_frame_playback()
                return
            self._stop_workflow_frame_playback()
            if not slider.isEnabled() or slider.maximum() <= slider.minimum():
                self._sync_workflow_frame_play_button(slider)
                return
            self._workflow_playback_slider = slider
            playback_data = self._workflow_playback_c3d_data_for_slider(slider)
            self._workflow_playback_fps = _workflow_playback_fps(playback_data)
            self._workflow_playback_timer.setInterval(_workflow_playback_timer_interval_ms(playback_data))
            button = self._workflow_frame_play_buttons.get(slider)
            if button is not None:
                button.setText("||")
                button.setToolTip(f"Pause frames ({self._workflow_playback_fps:g} fps)")
            if (
                slider is self.virtual_marker_frame_slider
                and self.workflow_tabs.tabText(self.workflow_tabs.currentIndex()) == "Functional reconstruction"
            ):
                self._update_virtual_marker_preview()
            self._workflow_playback_timer.start()

        def _workflow_playback_c3d_data_for_slider(self, slider):
            """
            Return the C3D data that controls playback speed for a preview slider.
            """
            if slider is self.virtual_marker_frame_slider:
                if self.workflow_tabs.tabText(self.workflow_tabs.currentIndex()) == "Functional reconstruction":
                    axis = self._selected_functional_reconstruction_axis()
                    data, _source_label = self._selected_functional_reconstruction_c3d_data(axis)
                    if data is not None:
                        return data
                return self._selected_virtual_marker_preview_c3d_data()
            if slider in (
                self.technical_frame_slider,
                self.anatomical_frame_slider,
                self.segment_settings_frame_slider,
            ):
                return self.c3d_data
            return None

        def _advance_workflow_frame_playback(self) -> None:
            """
            Advance the currently playing preview slider, looping at the end.
            """
            slider = self._workflow_playback_slider
            if slider is None or not slider.isEnabled() or slider.maximum() <= slider.minimum():
                self._stop_workflow_frame_playback()
                return
            frame_count = slider.maximum() - slider.minimum() + 1
            next_frame = slider.minimum() + ((slider.value() - slider.minimum() + 1) % frame_count)
            slider.setValue(next_frame)

        def _stop_workflow_frame_playback(self) -> None:
            """
            Stop frame playback and restore the play button state.
            """
            if self._workflow_playback_timer.isActive():
                self._workflow_playback_timer.stop()
            slider = self._workflow_playback_slider
            self._workflow_playback_slider = None
            if slider is not None:
                button = self._workflow_frame_play_buttons.get(slider)
                if button is not None:
                    button.setText(">")
                    button.setToolTip("Play frames")
                    button.setEnabled(slider.isEnabled() and slider.maximum() > slider.minimum())

        def _sync_workflow_frame_play_button(self, slider) -> None:
            """
            Enable or disable the playback button associated with a frame slider.
            """
            button = self._workflow_frame_play_buttons.get(slider)
            if button is None:
                return
            can_play = slider.isEnabled() and slider.maximum() > slider.minimum()
            if not can_play and self._workflow_playback_slider is slider:
                self._stop_workflow_frame_playback()
            button.setEnabled(can_play)

        def _show_workflow_view_menu(self) -> None:
            """
            Show visible view choices for the active C3D workflow preview.
            """
            if self.workflow_view_button.menu() is None:
                self.workflow_view_button.setMenu(self._create_workflow_view_menu())
            position = self.workflow_view_button.mapToGlobal(self.workflow_view_button.rect().bottomLeft())
            self.workflow_view_button.menu().popup(position)

        def _create_workflow_view_menu(self):
            """
            Create the native view menu attached to the workflow preview button.
            """
            menu = QMenu(self)
            for plane in ("XY", "YZ", "ZX"):
                action = menu.addAction(f"{plane} plane")
                action.triggered.connect(
                    lambda checked=False, selected_plane=plane: self._apply_workflow_preview_plane(selected_plane)
                )
            menu.addSeparator()
            for label, view in (
                ("Face view", "face"),
                ("Back view", "dos"),
                ("Side view", "cote"),
            ):
                action = menu.addAction(label)
                action.triggered.connect(
                    lambda checked=False, selected_view=view: self._apply_workflow_preview_subject_view(selected_view)
                )
            return menu

        def _apply_workflow_preview_plane(self, plane: str) -> None:
            """
            Apply one standard lab-plane view to the active workflow preview.
            """
            self._apply_workflow_preview_camera(_preview_camera_matrix_for_plane(plane))

        def _apply_workflow_preview_subject_view(self, view: str) -> None:
            """
            Apply one PCA-derived subject view to the active workflow preview.
            """
            try:
                matrix = _preview_camera_matrix_for_subject_view(view, self._active_workflow_preview_marker_positions())
            except ValueError as error:
                QMessageBox.warning(self, "Unavailable view", str(error))
                return
            self._apply_workflow_preview_camera(matrix)

        def _apply_workflow_preview_camera(self, matrix: np.ndarray) -> None:
            """
            Apply one explicit camera matrix to the currently visible workflow preview.
            """
            widget = self.workflow_canvas_stack.currentWidget()
            if not hasattr(widget, "yaw"):
                return
            widget.yaw = matrix
            widget.pitch = 0.0
            widget._last_mouse_position = None
            widget._is_preview_dragging = False
            widget.setCursor(qt_open_hand_cursor)
            widget.update()

        def _active_workflow_preview_marker_positions(self) -> dict[str, np.ndarray]:
            """
            Return marker positions from the active preview for PCA-based subject views.
            """
            widget = self.workflow_canvas_stack.currentWidget()
            scene = getattr(widget, "scene", None)
            if scene is not None:
                return {name: np.asarray(point, dtype=float) for name, point in scene.markers.items()}
            c3d_data = getattr(widget, "c3d_data", None)
            if c3d_data is None:
                return {}
            frame_index = int(getattr(widget, "frame_index", 0))
            positions = {}
            for marker_name in c3d_data.marker_names:
                point = _marker_frame_position(c3d_data, marker_name, frame_index)
                if point is not None:
                    positions[marker_name] = np.asarray(point, dtype=float)
            return positions

        def _sync_workflow_preview_panel(self, tab_index: int) -> None:
            """
            Show the preview that matches the active construction step.
            """
            self._stop_workflow_frame_playback()
            tab_text = self.workflow_tabs.tabText(tab_index)
            title = "Anatomical frame preview"
            frame_index = self.workflow_frame_anatomical_index
            canvas_index = self.workflow_canvas_anatomical_index
            if tab_text == "Pipeline":
                title = "3D preview"
                frame_index = self.workflow_frame_empty_index
                canvas_index = self.workflow_canvas_empty_index
            elif tab_text == "Technical segment":
                title = "Technical segment preview"
                frame_index = self.workflow_frame_technical_index
                canvas_index = self.workflow_canvas_technical_index
            elif tab_text == "Virtual markers and axes":
                title = "Virtual marker/axis preview"
                frame_index = self.workflow_frame_virtual_index
                canvas_index = self.workflow_canvas_virtual_index
            elif tab_text == "Functional reconstruction":
                title = "Functional reconstruction preview"
                frame_index = self.workflow_frame_virtual_index
                canvas_index = self.workflow_canvas_virtual_index
            elif tab_text == "Chain definition":
                title = "Kinematic chain preview"
                frame_index = self.workflow_frame_segment_settings_index
                canvas_index = self.workflow_canvas_segment_settings_index
            self.workflow_preview_title_label.setText(title)
            self.workflow_frame_stack.setCurrentIndex(frame_index)
            self.workflow_canvas_stack.setCurrentIndex(canvas_index)

            virtual_preview_is_active = tab_text in {
                "Virtual markers and axes",
                "Functional reconstruction",
            }
            chain_preview_is_active = tab_text == "Chain definition"
            self._sync_workflow_preview_mode_combo(tab_text)
            self.virtual_marker_whole_body_preview_checkbox.setEnabled(virtual_preview_is_active)
            if tab_text == "Functional reconstruction":
                self._update_virtual_marker_preview()
            elif chain_preview_is_active:
                self._update_segment_settings_preview()

        def _sync_workflow_preview_mode_combo(self, tab_text: str) -> None:
            """
            Configure the preview-mode menu for the currently visible workflow preview.
            """
            if tab_text in {"Virtual markers and axes", "Functional reconstruction"}:
                items = ("functional", "static")
                current = self._virtual_marker_preview_mode
                enabled = True
            elif tab_text == "Chain definition":
                items = ("static", "q0")
                current = self._segment_settings_preview_mode
                enabled = True
            else:
                items = ("static",)
                current = "static"
                enabled = False
            self.workflow_preview_mode_combo.blockSignals(True)
            self.workflow_preview_mode_combo.clear()
            self.workflow_preview_mode_combo.addItems(items)
            index = self.workflow_preview_mode_combo.findText(current)
            self.workflow_preview_mode_combo.setCurrentIndex(max(index, 0))
            self.workflow_preview_mode_combo.setEnabled(enabled)
            self.workflow_preview_mode_combo.blockSignals(False)

        def _set_workflow_preview_mode(self, mode: str) -> None:
            mode = mode.strip()
            tab_text = self.workflow_tabs.tabText(self.workflow_tabs.currentIndex())
            if tab_text in {"Virtual markers and axes", "Functional reconstruction"}:
                self._virtual_marker_preview_mode = "static" if mode == "static" else "functional"
                self._update_virtual_marker_preview()
            elif tab_text == "Chain definition":
                self._segment_settings_preview_mode = "q0" if mode == "q0" else "static"
                self._configure_segment_settings_frame_slider()
                self._update_segment_settings_preview()

        def _functional_reconstruction_workflow_tab(self):
            widget = QWidget()
            layout = QVBoxLayout(widget)
            _configure_panel_layout(layout)
            layout.addWidget(_section_label("Functional trial reconstruction"))
            info = _muted_label(
                "Select a SARA knee AoR, reconstruct the whole functional trial with the current chain, and inspect the knee XYZ rotations plus the optimal axis in proximal/distal frames. QLD uses marker weights 100 on the proximal/distal AoR marker sets and 1 on the other model markers."
            )
            info.setWordWrap(True)
            layout.addWidget(info)

            aor_row = QHBoxLayout()
            _configure_panel_layout(aor_row, margin=0, spacing=8)
            aor_row.addWidget(QLabel("AoR"))
            aor_row.addWidget(self.functional_reconstruction_feature_combo, 1)
            layout.addLayout(aor_row)

            trial_row = QHBoxLayout()
            _configure_panel_layout(trial_row, margin=0, spacing=8)
            trial_row.addWidget(QLabel("Functional C3D"))
            trial_row.addWidget(self.functional_reconstruction_trial_combo, 1)
            layout.addLayout(trial_row)

            action_row = QHBoxLayout()
            _configure_panel_layout(action_row, margin=0, spacing=8)
            action_row.addWidget(QLabel("Method"))
            action_row.addWidget(self.functional_reconstruction_method_combo)
            action_row.addWidget(self.run_functional_reconstruction_button)
            action_row.addStretch()
            layout.addLayout(action_row)
            layout.addWidget(self.functional_reconstruction_plot)
            layout.addWidget(self.functional_reconstruction_output, 1)
            return widget

        def _segment_workflow_tab(self):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            _configure_panel_layout(layout)
            controls_column = QVBoxLayout()
            _configure_panel_layout(controls_column, margin=0)
            row = QHBoxLayout()
            _configure_panel_layout(row, margin=0, spacing=8)
            row.addWidget(self.add_segment_button)
            row.addWidget(self.remove_segment_button)
            row.addStretch()
            controls_column.addLayout(row)
            controls_column.addWidget(_section_label("Segments"))
            controls_column.addWidget(self.segment_marker_list)
            controls_column.addWidget(self.strip_participant_prefix_checkbox)
            controls_column.addWidget(self.marker_mapping_label)
            marker_row = QHBoxLayout()
            _configure_panel_layout(marker_row, margin=0)
            parent_column = QVBoxLayout()
            _configure_panel_layout(parent_column, margin=0, spacing=6)
            parent_column.addWidget(_section_label("Parent segment"))
            parent_column.addWidget(self.workflow_parent_combo)
            parent_column.addStretch()
            left_column = QVBoxLayout()
            _configure_panel_layout(left_column, margin=0, spacing=6)
            left_column.addWidget(_section_label("Available markers"))
            left_column.addWidget(self.show_all_markers_checkbox)
            left_column.addWidget(self.show_virtual_markers_in_segments_checkbox)
            left_column.addWidget(self.marker_list)
            self.marker_list.setMaximumWidth(320)
            transfer_column = QVBoxLayout()
            _configure_panel_layout(transfer_column, margin=0, spacing=8)
            transfer_column.addStretch()
            self.assign_marker_button.setText("->")
            self.unassign_marker_button.setText("<-")
            transfer_column.addWidget(self.assign_marker_button)
            transfer_column.addWidget(self.unassign_marker_button)
            transfer_column.addStretch()
            right_column = QVBoxLayout()
            _configure_panel_layout(right_column, margin=0, spacing=6)
            right_column.addWidget(_section_label("Assigned markers"))
            right_column.addWidget(self.assigned_marker_list)
            right_column.addWidget(self.assigned_marker_technical_checkbox)
            self.assigned_marker_list.setMaximumWidth(320)
            marker_row.addLayout(parent_column, 1)
            marker_row.addLayout(left_column, 2)
            marker_row.addLayout(transfer_column, 0)
            marker_row.addLayout(right_column, 2)
            controls_column.addLayout(marker_row)
            controls_column.addStretch()

            layout.addLayout(controls_column, 1)
            return widget

        def _anatomical_segment_workflow_tab(self):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            _configure_panel_layout(layout)

            controls_column = QVBoxLayout()
            _configure_panel_layout(controls_column, margin=0)
            controls_column.addWidget(_section_label("Anatomical segments"))
            controls_column.addWidget(self.anatomical_segment_list)
            instructions_label = _muted_label(
                "Define two anatomical vectors from marker groups. The kept vector is preserved; the other one "
                "is orthogonalized and the third axis is computed by cross product."
            )
            instructions_label.setMaximumWidth(720)
            controls_column.addWidget(instructions_label)

            controls_column.addWidget(_section_label("Segment system of coordinates"))
            axis_layout = QHBoxLayout()
            _configure_panel_layout(axis_layout, margin=0)
            source_column = QVBoxLayout()
            _configure_panel_layout(source_column, margin=0, spacing=6)
            source_column.addWidget(_section_label("Available markers and axes"))
            source_column.addWidget(self.axis_marker_source_list)
            origin_row = QHBoxLayout()
            _configure_panel_layout(origin_row, margin=0, spacing=8)
            origin_buttons = QVBoxLayout()
            _configure_panel_layout(origin_buttons, margin=0, spacing=8)
            origin_buttons.addWidget(self.add_axis_origin_marker_button)
            origin_buttons.addWidget(self.remove_axis_origin_marker_button)
            origin_buttons.addStretch()
            origin_row.addLayout(origin_buttons)
            origin_row.addWidget(self.axis_origin_marker_list)
            source_column.addWidget(_section_label("Origin markers"))
            source_column.addLayout(origin_row)
            axis_layout.addLayout(source_column, 0)
            axis_layout.addLayout(_axis_vectors_layout(self.axis_vector_controls), 1)
            controls_column.addLayout(axis_layout)
            controls_column.addWidget(self.save_segment_axis_button)
            controls_column.addWidget(_section_label("Saved anatomical frames"))
            controls_column.addWidget(
                _muted_label(
                    "One saved frame recipe per segment: origin + vector 1 + vector 2. "
                    "The third axis is computed automatically."
                )
            )
            controls_column.addWidget(self.axis_list)
            controls_column.addStretch()

            layout.addLayout(controls_column, 1)
            return widget

        def _virtual_marker_workflow_tab(self):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            _configure_panel_layout(layout)

            controls_layout = QVBoxLayout()
            _configure_panel_layout(controls_layout, margin=0)
            list_column = QVBoxLayout()
            _configure_panel_layout(list_column, margin=0, spacing=8)
            list_column.addWidget(_section_label("Virtual markers and axes"))
            self.feature_list.setMaximumHeight(155)
            list_column.addWidget(self.feature_list)
            row = QHBoxLayout()
            _configure_panel_layout(row, margin=0, spacing=8)
            row.addWidget(self.add_virtual_marker_button)
            row.addWidget(self.remove_virtual_marker_button)
            list_column.addLayout(row)
            controls_layout.addLayout(list_column)

            form_column = QVBoxLayout()
            _configure_panel_layout(form_column, margin=0)
            for compact_widget in (
                self.virtual_marker_method_combo,
                self.virtual_marker_predictive_method_combo,
                self.virtual_marker_segment_combo,
                self.virtual_marker_proximal_combo,
                self.virtual_marker_name_edit,
            ):
                compact_widget.setMinimumWidth(150)
                compact_widget.setMaximumWidth(220)
            self.virtual_marker_c3d_file_combo.setMinimumWidth(380)
            self.virtual_marker_c3d_file_combo.setMaximumWidth(560)
            self.virtual_marker_suggested_name_label.setMinimumWidth(260)
            self.virtual_marker_suggested_name_label.setMaximumWidth(360)
            form_grid = QGridLayout()
            form_grid.setHorizontalSpacing(12)
            form_grid.setVerticalSpacing(8)
            form_grid.setColumnMinimumWidth(1, 170)
            form_grid.setColumnMinimumWidth(3, 260)
            form_grid.setColumnStretch(4, 1)
            form_grid.addWidget(QLabel("Method"), 0, 0)
            form_grid.addWidget(self.virtual_marker_method_combo, 0, 1)
            form_grid.addWidget(QLabel("Predictive"), 0, 2)
            form_grid.addWidget(self.virtual_marker_predictive_method_combo, 0, 3)
            form_grid.addWidget(QLabel("Segment"), 1, 0)
            form_grid.addWidget(self.virtual_marker_segment_combo, 1, 1)
            form_grid.addWidget(QLabel("Parent"), 1, 2)
            form_grid.addWidget(self.virtual_marker_proximal_combo, 1, 3)
            form_grid.addWidget(QLabel("Functional C3D"), 2, 0)
            form_grid.addWidget(self.virtual_marker_c3d_file_combo, 2, 1, 1, 3)
            functional_frame_options = QVBoxLayout()
            functional_frame_options.addWidget(self.use_manual_functional_frames_checkbox)
            functional_frame_options.addWidget(self.use_diverse_functional_frames_checkbox)
            form_grid.addLayout(functional_frame_options, 2, 4, 2, 1)
            form_grid.addWidget(QLabel("Name"), 3, 0)
            form_grid.addWidget(self.virtual_marker_name_edit, 3, 1)
            form_grid.addWidget(QLabel("Suggested"), 3, 2)
            form_grid.addWidget(self.virtual_marker_suggested_name_label, 3, 3)
            form_grid.addWidget(self.virtual_marker_source_label, 4, 0)
            form_grid.addWidget(self.virtual_marker_source_edit, 4, 1, 1, 3)
            form_grid.addWidget(self.explore_functional_residuals_button, 4, 4)
            form_column.addWidget(_layout_group("Definition", form_grid))

            self.virtual_marker_projection_group = QGroupBox("Projection on axis: project marker(s) onto an axis")
            projection_layout = QGridLayout(self.virtual_marker_projection_group)
            projection_layout.setHorizontalSpacing(14)
            projection_layout.setVerticalSpacing(6)
            projection_layout.addWidget(QLabel("Available markers and axes"), 0, 0)
            projection_layout.addWidget(self.virtual_marker_projection_available_list, 1, 0)
            projection_layout.addLayout(
                _list_with_side_buttons(
                    "Marker(s) to project",
                    self.virtual_marker_projected_marker_list,
                    self.add_projected_marker_button,
                    self.remove_projected_marker_button,
                ),
                0,
                1,
                2,
                1,
            )
            projection_layout.addLayout(
                _list_with_side_buttons(
                    "Projection axis",
                    self.virtual_marker_projection_axis_list,
                    self.add_projection_axis_button,
                    self.remove_projection_axis_button,
                ),
                0,
                2,
                2,
                1,
            )
            projection_layout.setColumnStretch(0, 1)
            projection_layout.setColumnStretch(1, 1)
            projection_layout.setColumnStretch(2, 1)
            form_column.addWidget(self.virtual_marker_projection_group)

            form_column.addWidget(_section_label("Technical markers used"))
            form_column.addWidget(self.virtual_marker_technical_markers_label)
            form_column.addWidget(self.save_virtual_marker_button)
            form_column.addWidget(_section_label("Selected marker information"))
            form_column.addWidget(self.virtual_marker_info_label)
            form_column.addStretch()
            controls_layout.addLayout(form_column)

            layout.addLayout(controls_layout, 1)
            return widget

        def _segment_settings_workflow_tab(self):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            _configure_panel_layout(layout)
            left_column = QVBoxLayout()
            _configure_panel_layout(left_column, margin=0)
            left_column.addWidget(_section_label("Segments"))
            left_column.addWidget(self.segment_settings_list)
            self.segment_settings_list.setMinimumWidth(280)
            self.segment_settings_list.setMaximumWidth(360)
            layout.addLayout(left_column, 1)

            form_column = QVBoxLayout()
            _configure_panel_layout(form_column, margin=0, spacing=12)

            for compact_widget in (
                self.settings_translations_edit,
                self.settings_rotations_edit,
                self.settings_q_min_edit,
                self.settings_q_max_edit,
                self.settings_initial_rotation_method_combo,
                self.settings_initial_rotation_source_edit,
                self.settings_initial_rotation_c3d_combo,
                self.settings_segment_length_edit,
                self.settings_anthropometry_model_combo,
                self.settings_anthropometry_sex_combo,
                self.settings_anthropometry_mass_edit,
            ):
                compact_widget.setMinimumWidth(130)
                compact_widget.setMaximumWidth(240)

            dof_form = QGridLayout()
            dof_form.setHorizontalSpacing(10)
            dof_form.setVerticalSpacing(8)
            dof_form.addWidget(QLabel("Translations"), 0, 0)
            dof_form.addWidget(self.settings_translations_edit, 0, 1)
            dof_form.addWidget(QLabel("Rotations"), 0, 2)
            dof_form.addWidget(self.settings_rotations_edit, 0, 3)
            dof_form.addWidget(QLabel("q min"), 1, 0)
            dof_form.addWidget(self.settings_q_min_edit, 1, 1)
            dof_form.addWidget(QLabel("q max"), 1, 2)
            dof_form.addWidget(self.settings_q_max_edit, 1, 3)
            dof_form.addWidget(self.settings_child_translation_checkbox, 2, 1, 1, 3)
            dof_form.setColumnStretch(4, 1)
            form_column.addWidget(_layout_group("DoF axes and bounds", dof_form))

            rotation_form = QGridLayout()
            rotation_form.setHorizontalSpacing(10)
            rotation_form.setVerticalSpacing(8)
            rotation_form.addWidget(QLabel("Initial rotation"), 0, 0)
            rotation_form.addWidget(self.settings_initial_rotation_method_combo, 0, 1)
            rotation_form.addWidget(QLabel("Matrix"), 0, 2)
            rotation_form.addWidget(self.settings_initial_rotation_source_edit, 0, 3)
            anatomical_c3d_row = QHBoxLayout()
            _configure_panel_layout(anatomical_c3d_row, margin=0, spacing=8)
            anatomical_c3d_row.addWidget(self.settings_initial_rotation_c3d_combo)
            anatomical_c3d_row.addWidget(self.browse_initial_rotation_c3d_button)
            anatomical_c3d_row.addStretch()
            rotation_form.addWidget(QLabel("Anatomical C3D"), 1, 0)
            rotation_form.addLayout(anatomical_c3d_row, 1, 1, 1, 3)
            form_column.addWidget(_layout_group("Initial rotation", rotation_form))

            anthropometry_form = QGridLayout()
            anthropometry_form.setHorizontalSpacing(10)
            anthropometry_form.setVerticalSpacing(8)
            anthropometry_form.addWidget(QLabel("Segment length"), 0, 0)
            anthropometry_form.addWidget(self.settings_segment_length_edit, 0, 1)
            anthropometry_form.addWidget(QLabel("Anthropometry"), 0, 2)
            anthropometry_form.addWidget(self.settings_anthropometry_model_combo, 0, 3)
            anthropometry_form.addWidget(QLabel("Length source"), 1, 0)
            anthropometry_form.addWidget(self.settings_segment_length_source_label, 1, 1, 1, 3)
            anthropometry_form.addWidget(QLabel("Sex"), 2, 0)
            anthropometry_form.addWidget(self.settings_anthropometry_sex_combo, 2, 1)
            anthropometry_form.addWidget(QLabel("Body mass"), 2, 2)
            anthropometry_form.addWidget(self.settings_anthropometry_mass_edit, 2, 3)
            anthropometry_form.setColumnStretch(4, 1)
            form_column.addWidget(_layout_group("Anthropometry", anthropometry_form))

            action_row = QHBoxLayout()
            _configure_panel_layout(action_row, margin=0, spacing=10)
            self.apply_anthropometry_button.setMinimumWidth(260)
            self.edit_segment_settings_button.setMinimumWidth(260)
            action_row.addWidget(self.apply_anthropometry_button)
            action_row.addWidget(self.edit_segment_settings_button)
            action_row.addStretch()
            form_column.addLayout(action_row)

            export_row = QHBoxLayout()
            _configure_panel_layout(export_row, margin=0, spacing=10)
            export_row.addWidget(QLabel("Export format"))
            export_row.addWidget(self.chain_export_format_combo)
            export_row.addWidget(self.export_chain_model_button)
            export_row.addStretch()
            form_column.addWidget(_layout_group("Export kinematic chain", export_row))
            form_column.addStretch()
            layout.addLayout(form_column, 2)
            return widget

        def _file_role_workflow_tab(self):
            widget = QWidget()
            layout = QVBoxLayout(widget)
            _configure_panel_layout(layout)
            layout.addWidget(_section_label("Files and marker names"))
            layout.addWidget(self.c3d_names_overview_label)
            layout.addWidget(self.c3d_marker_status_label)
            layout.addWidget(_section_label("Assigned C3D files"))
            layout.addWidget(self.file_role_list)
            row = QHBoxLayout()
            _configure_panel_layout(row, margin=0, spacing=8)
            row.addWidget(self.assign_c3d_role_button)
            row.addWidget(self.clear_c3d_role_button)
            layout.addLayout(row)
            return widget

        def _choose_c3d_file(self) -> None:
            filepath, _ = QFileDialog.getOpenFileName(
                self,
                "Choose C3D file",
                self.c3d_folder_path,
                "C3D files (*.c3d)",
            )
            if not filepath:
                return
            try:
                self._load_c3d_file(filepath)
            except Exception as error:
                QMessageBox.critical(self, "Unable to load C3D", str(error))

        def _reload_current_c3d_file(self, *_args) -> None:
            self._c3d_data_cache.clear()
            self._invalidate_virtual_feature_c3d_preview_cache()
            filepath = self.c3d_path.text().strip()
            if filepath == "":
                return
            try:
                self._load_c3d_file(filepath)
            except Exception as error:
                QMessageBox.critical(self, "Unable to reload C3D", str(error))

        def _load_c3d_file(self, filepath: str) -> None:
            self.c3d_data = C3dData(filepath)
            self._invalidate_virtual_feature_c3d_preview_cache()
            if self.strip_participant_prefix_checkbox.isChecked():
                _strip_participant_prefix_from_c3d_data(self.c3d_data)
            self.c3d_path.setText(filepath)
            self._update_main_c3d_name_label()
            marker_mapping = _marker_name_mapping_for_c3d(
                _marker_pool_from_draft(self.workflow_draft),
                tuple(self.c3d_data.marker_names),
            )
            self.workflow_draft = _remap_c3d_workflow_draft_markers(self.workflow_draft, marker_mapping)
            self.marker_mapping_label.setText(_format_marker_mapping_summary(marker_mapping))
            self.workflow_marker_pool = tuple(self.c3d_data.marker_names)
            self._configure_technical_frame_slider()
            self._configure_anatomical_frame_slider()
            self._configure_segment_settings_frame_slider()
            self._sync_virtual_marker_c3d_files()
            self._sync_initial_rotation_c3d_files()
            self._update_preset_details()

        def _update_main_c3d_name_label(self) -> None:
            filepath = self.c3d_path.text().strip()
            self.main_c3d_name_label.setText(Path(filepath).name if filepath else "No main C3D selected")

        def _choose_c3d_folder(self) -> None:
            folder = QFileDialog.getExistingDirectory(self, "Choose folder containing C3D files", self.c3d_folder_path)
            if not folder:
                return
            self._set_c3d_folder(folder)

        def _set_c3d_folder(self, folder: str) -> None:
            """
            Set the C3D folder and auto-assign expected C3D files.
            """
            progress_dialog, progress_callback = self._c3d_folder_selection_progress_reporter()
            try:
                progress_callback(f"Selected folder: {folder}")
                self.c3d_folder_path = folder
                self._invalidate_virtual_feature_c3d_preview_cache()
                self.c3d_folder_edit.setText(folder)
                self._auto_assign_c3d_files_from_folder(progress_callback=progress_callback)
                progress_callback("Refreshing C3D file selectors...")
                self._sync_virtual_marker_c3d_files()
                self._sync_initial_rotation_c3d_files()
                progress_callback("Refreshing preset details...")
                self._update_preset_details()
                self._update_generation_log()
            finally:
                progress_dialog.close()

        def _c3d_folder_selection_progress_reporter(self):
            """
            Create a progress popup while the workflow scans and loads a C3D folder.
            """
            progress_dialog = QProgressDialog(
                "Preparing C3D folder scan...",
                None,
                0,
                0,
                self,
            )
            progress_dialog.setWindowTitle("Loading C3D folder")
            progress_dialog.setWindowModality(qt_window_modal)
            progress_dialog.setCancelButton(None)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.show()

            def progress_callback(message: str) -> None:
                progress_dialog.setLabelText(message)
                progress_dialog.show()
                QApplication.processEvents()

            progress_callback("Preparing C3D folder scan...")
            return progress_dialog, progress_callback

        def _functional_frame_calculation_progress_reporter(self):
            """
            Create a progress popup while diverse functional-frame previews are recomputed.
            """
            progress_dialog = QProgressDialog(
                "Preparing functional-frame calculations...",
                None,
                0,
                0,
                self,
            )
            progress_dialog.setWindowTitle("Updating functional-frame calculations")
            progress_dialog.setWindowModality(qt_window_modal)
            progress_dialog.setCancelButton(None)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.show()

            def progress_callback(message: str) -> None:
                progress_dialog.setLabelText(message)
                progress_dialog.show()
                QApplication.processEvents()

            progress_callback("Preparing functional-frame calculations...")
            return progress_dialog, progress_callback

        def _auto_assign_c3d_files_from_folder(self, load_main: bool = True, progress_callback=None) -> None:
            if not self.c3d_folder_path or self._is_auto_assigning_c3d_files:
                return
            self._is_auto_assigning_c3d_files = True
            try:
                if progress_callback is not None:
                    progress_callback("Scanning folder for expected C3D files...")
                assignments = []
                trial_sources = {}
                for assignment in self.workflow_draft.file_assignments:
                    if progress_callback is not None:
                        progress_callback(f"Matching C3D file: {assignment.generic_name}")
                    matched_file = _matching_c3d_file_for_expected_name(self.c3d_folder_path, assignment.generic_name)
                    source_path = str(matched_file) if matched_file is not None else assignment.source_path
                    assignments.append(replace(assignment, source_path=source_path))
                    if source_path:
                        trial_sources[assignment.role] = source_path

                updated_virtual_markers = []
                if progress_callback is not None:
                    progress_callback("Updating virtual marker C3D sources...")
                for marker in self.workflow_draft.virtual_markers:
                    trial_name = _trial_name_from_virtual_feature_source(
                        marker.source
                    ) or _trial_name_from_virtual_feature_source(marker.equation)
                    source_path = trial_sources.get(trial_name, "")
                    equation = marker.equation
                    if marker.method in {"score", "sara", "sara_direction"} and equation == "":
                        parent_name = _parent_segment_name(self.workflow_draft, marker.segment_name)
                        equation = f"proximal={parent_name}; distal={marker.segment_name}" if parent_name else equation
                    updated_virtual_markers.append(
                        replace(
                            marker,
                            source=_source_with_c3d_assignment(marker.source, source_path),
                            equation=equation,
                        )
                    )

                updated_axes = []
                if progress_callback is not None:
                    progress_callback("Updating virtual axis C3D sources...")
                for axis in self.workflow_draft.axes:
                    trial_name = _trial_name_from_virtual_feature_source(axis.source)
                    source_path = trial_sources.get(trial_name, "")
                    updated_axes.append(
                        replace(
                            axis,
                            source=_source_with_c3d_assignment(axis.source, source_path),
                        )
                    )

                self.workflow_draft = replace(
                    self.workflow_draft,
                    file_assignments=tuple(assignments),
                    virtual_markers=tuple(updated_virtual_markers),
                    axes=tuple(updated_axes),
                )
                main_source = trial_sources.get("main", "")
                if load_main and main_source and not self.c3d_path.text().strip():
                    if progress_callback is not None:
                        progress_callback(f"Loading main C3D: {Path(main_source).name}")
                    self._load_c3d_file(main_source)
            finally:
                self._is_auto_assigning_c3d_files = False

        def _browse_virtual_marker_c3d_source(self) -> None:
            filepath, _ = QFileDialog.getOpenFileName(
                self,
                "Choose virtual marker C3D",
                self.c3d_folder_path,
                "C3D files (*.c3d)",
            )
            if not filepath:
                return
            self.virtual_marker_source_edit.setText(filepath)
            self._set_c3d_combo_to_filepath(
                self.virtual_marker_c3d_file_combo,
                filepath,
                "Choose a C3D folder first",
            )
            self._update_virtual_marker_preview()

        def _browse_initial_rotation_c3d_source(self) -> None:
            filepath, _ = QFileDialog.getOpenFileName(
                self,
                "Choose anatomical C3D",
                self.c3d_folder_path,
                "C3D files (*.c3d)",
            )
            if not filepath:
                return
            self._set_c3d_combo_to_filepath(self.settings_initial_rotation_c3d_combo, filepath, "Choose a C3D first")
            self._sync_initial_rotation_source_fields()

        def _generate_template(self) -> None:
            default_name = f"{self.selected_preset().value}_template.json"
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Generate template",
                default_name,
                "JSON files (*.json)",
            )
            if not filepath:
                return
            try:
                payload = c3d_template_payload_from_draft(self.workflow_draft)
                payload["c3d_folder"] = self.c3d_folder_path
                payload["generation_log"] = list(
                    _c3d_generation_log(
                        self.workflow_draft,
                        self.c3d_data,
                        self.c3d_folder_path,
                        self.workflow_marker_pool,
                    )
                )
                Path(filepath).write_text(json.dumps(payload, indent=2))
            except Exception as error:
                QMessageBox.critical(self, "Unable to generate template", str(error))

        def _export_workflow_chain_model(self) -> None:
            if self.c3d_data is None:
                QMessageBox.critical(
                    self,
                    "Unable to export model",
                    "Load a static/main C3D before exporting.",
                )
                return
            if not self._sync_segment_settings_validation_style():
                QMessageBox.critical(
                    self,
                    "Unable to export model",
                    "Fix the invalid Chain definition fields before exporting.",
                )
                return
            self._apply_workflow_segment_settings_from_form()
            extension = _model_export_extension_from_label(self.chain_export_format_combo.currentText())
            default_folder = self.c3d_folder_path or str(Path.home())
            default_name = str(Path(default_folder) / f"{self.workflow_draft.preset.value}{extension}")
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Export kinematic chain",
                default_name,
                _model_export_filter_for_extension(extension),
            )
            if not filepath:
                return
            filepath = _model_export_filepath_with_extension(
                filepath,
                extension,
                self.workflow_draft.preset.value,
            )
            progress_dialog, progress_callback = self._c3d_folder_generation_progress_reporter()
            try:
                progress_callback("Building kinematic chain from current C3D draft...")
                model = self._model_from_current_chain_settings(progress_callback)
                progress_callback(f"Writing {Path(filepath).name}...")
                _export_model_to_path(model, filepath)
                progress_dialog.close()
                QMessageBox.information(self, "Model exported", f"Wrote:\n{filepath}")
            except Exception as error:
                progress_dialog.close()
                QMessageBox.critical(self, "Unable to export model", str(error))

        def _add_workflow_segment(self) -> None:
            segment_name, accepted = QInputDialog.getText(self, "Add segment", "Segment name")
            if not accepted:
                return
            segment_type, accepted = QInputDialog.getItem(
                self,
                "Segment type",
                "Type",
                ["technical", "anatomical"],
                0,
                False,
            )
            if not accepted:
                return
            parent_name, accepted = QInputDialog.getItem(
                self,
                "Segment parent",
                "Parent",
                _segment_parent_choices(self.workflow_draft),
                0,
                True,
            )
            if not accepted:
                return
            try:
                self.workflow_draft = add_segment_to_draft(
                    self.workflow_draft,
                    segment_name,
                    parent_name=parent_name,
                    segment_type=segment_type,
                )
                self._update_preset_details()
            except Exception as error:
                QMessageBox.critical(self, "Unable to add segment", str(error))

        def _remove_workflow_segment(self) -> None:
            segment_name = self._selected_workflow_segment_name()
            if segment_name is None:
                return
            self.workflow_draft = remove_segment_from_draft(self.workflow_draft, segment_name)
            self._update_preset_details()

        def _assign_workflow_marker(self) -> None:
            segment_name = self._selected_workflow_segment_name()
            marker_names = self._selected_workflow_marker_names()
            if segment_name is None or len(marker_names) == 0:
                return
            try:
                self.workflow_marker_pool = tuple(dict.fromkeys(self.workflow_marker_pool + marker_names))
                self.workflow_draft = assign_markers_to_segment(self.workflow_draft, segment_name, marker_names)
                self._refresh_marker_assignment_details(segment_name)
            except Exception as error:
                QMessageBox.critical(self, "Unable to assign marker", str(error))

        def _unassign_workflow_marker(self) -> None:
            segment_name = self._selected_workflow_segment_name()
            marker_names = self._selected_assigned_marker_names()
            if len(marker_names) == 0:
                marker_names = self._selected_workflow_marker_names()
            if segment_name is None or len(marker_names) == 0:
                return
            self.workflow_draft = unassign_markers_from_segment(self.workflow_draft, segment_name, marker_names)
            self._refresh_marker_assignment_details(segment_name)

        def _add_workflow_virtual_marker(self) -> None:
            self.feature_list.clearSelection()
            self.virtual_marker_name_edit.clear()
            self.virtual_marker_source_edit.clear()
            self.virtual_marker_equation_edit.clear()
            self.virtual_marker_projected_marker_list.clear()
            self.virtual_marker_projection_axis_list.clear()
            if self.virtual_marker_segment_combo.count() != 0:
                self.virtual_marker_segment_combo.setCurrentIndex(0)
            self._set_virtual_marker_method("pointing")
            self.virtual_marker_predictive_method_combo.setCurrentIndex(0)
            self._sync_virtual_marker_method_fields()
            self._update_virtual_marker_info_label(None)
            self._update_virtual_marker_preview()

        def _save_workflow_virtual_marker_from_form(self) -> None:
            name = self.virtual_marker_name_edit.text().strip() or self._suggested_virtual_marker_name()
            segment_name = self.virtual_marker_segment_combo.currentText().strip()
            method = self._selected_virtual_marker_method()
            source = self._virtual_marker_source_from_form(method)
            equation = self._virtual_marker_equation_from_form(method)
            try:
                self.workflow_draft = add_virtual_marker_to_draft(
                    self.workflow_draft,
                    name=name,
                    method=method,
                    segment_name=segment_name,
                    source=source,
                    equation=equation,
                )
                self.workflow_marker_pool = tuple(dict.fromkeys(self.workflow_marker_pool + (name,)))
                self._update_preset_details()
                self._select_virtual_marker_by_name(name)
            except Exception as error:
                QMessageBox.critical(self, "Unable to save virtual marker", str(error))

        def _virtual_marker_source_from_form(self, method: str) -> str:
            if method == "axis_projection":
                return self.virtual_marker_source_edit.text().strip()
            if method == "marker_mean":
                return self._technical_marker_source_from_selected_segments()
            if method == "rab2002_shoulder":
                return self.virtual_marker_source_edit.text().strip() or _default_rab2002_payload(
                    self.virtual_marker_name_edit.text().strip(),
                    self.virtual_marker_segment_combo.currentText().strip(),
                )
            return self._selected_virtual_marker_c3d_file()

        def _virtual_marker_equation_from_form(self, method: str) -> str:
            if method == "axis_projection":
                return self.virtual_marker_equation_edit.text().strip()
            methods_with_segment_context = {"score", "sara", "sara_direction"} | set(
                PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS
            )
            if method not in methods_with_segment_context:
                return ""
            return "; ".join(
                [
                    f"proximal={self.virtual_marker_proximal_combo.currentText().strip()}",
                    f"distal={self.virtual_marker_segment_combo.currentText().strip()}",
                ]
            )

        def _suggested_virtual_marker_name(self) -> str:
            method = self._selected_virtual_marker_method()
            proximal = self.virtual_marker_proximal_combo.currentText().strip()
            segment_name = self.virtual_marker_segment_combo.currentText().strip()
            distal = segment_name
            joint_name = _joint_name_from_segments(proximal, distal) if proximal or distal else segment_name or "Marker"
            method_label = {
                "score": "SCoRE",
                "sara": "SARA",
                "sara_direction": "SARADirection",
                "marker_mean": "Average",
                "axis_projection": "ProjectionOnAxis",
            }.get(
                method,
                method.capitalize() if method else "Virtual",
            )
            if method in {"hara2016_hip", "harrington2007_hip"}:
                method_label = PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS[method].replace(" ", "")
            elif method in {"sobral2025_shoulder", "rab2002_shoulder"}:
                method_label = PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS[method].replace(" ", "")
            local_segment = proximal or segment_name
            if method in {"score", "sara", "sara_direction"} and local_segment:
                return f"CoR_{method_label}_{joint_name}_wrt_{local_segment}"
            if segment_name:
                return f"{method_label}_{joint_name}_wrt_{segment_name}"
            return f"{method_label}_{joint_name}"

        def _update_suggested_virtual_marker_name(self, *_args) -> None:
            suggested_name = self._suggested_virtual_marker_name()
            self.virtual_marker_suggested_name_label.setText(suggested_name)
            self.virtual_marker_suggested_name_label.setToolTip(suggested_name)

        def _remove_workflow_virtual_marker(self) -> None:
            axis = self._selected_virtual_axis()
            if axis is not None:
                self.workflow_draft = remove_axis_from_draft(self.workflow_draft, axis.name)
                self._update_preset_details()
                return
            name = self._selected_virtual_marker_name()
            if name is None:
                return
            self.workflow_draft = remove_virtual_marker_from_draft(self.workflow_draft, name)
            self._update_preset_details()

        def _load_selected_segment_settings_into_form(self) -> None:
            setting = self._selected_segment_setting()
            if setting is None:
                return
            self._suspend_segment_settings_preview_updates = True
            try:
                for widget in (
                    self.settings_translations_edit,
                    self.settings_rotations_edit,
                    self.settings_q_min_edit,
                    self.settings_q_max_edit,
                    self.settings_child_translation_checkbox,
                    self.settings_initial_rotation_method_combo,
                    self.settings_initial_rotation_source_edit,
                    self.settings_initial_rotation_c3d_combo,
                    self.settings_anthropometry_model_combo,
                    self.settings_anthropometry_sex_combo,
                    self.settings_anthropometry_mass_edit,
                ):
                    widget.blockSignals(True)
                self.settings_translations_edit.setText(setting.translations)
                self.settings_rotations_edit.setText(setting.rotations)
                self.settings_q_min_edit.setText(_format_float_list(list(setting.q_min)))
                self.settings_q_max_edit.setText(_format_float_list(list(setting.q_max)))
                self.settings_child_translation_checkbox.setChecked(setting.child_translation)
                self.settings_initial_rotation_method_combo.setCurrentText(setting.initial_rotation_method)
                initial_rotation_source = setting.initial_rotation_source
                if setting.initial_rotation_method == "matrix" and initial_rotation_source == "":
                    initial_rotation_source = _format_rotation_matrix(setting.initial_rotation_matrix)
                self.settings_initial_rotation_source_edit.setText(initial_rotation_source)
                self.settings_anthropometry_model_combo.setCurrentText(setting.anthropometry_model or "none")
                self.settings_anthropometry_sex_combo.setCurrentText(setting.anthropometry_sex or "male")
                self.settings_anthropometry_mass_edit.setText(
                    "" if setting.anthropometry_mass is None else str(setting.anthropometry_mass)
                )
                self._refresh_segment_length_fields(setting)
                self._sync_initial_rotation_c3d_files(setting.initial_rotation_source)
                self._sync_initial_rotation_source_fields()
            finally:
                for widget in (
                    self.settings_translations_edit,
                    self.settings_rotations_edit,
                    self.settings_q_min_edit,
                    self.settings_q_max_edit,
                    self.settings_child_translation_checkbox,
                    self.settings_initial_rotation_method_combo,
                    self.settings_initial_rotation_source_edit,
                    self.settings_initial_rotation_c3d_combo,
                    self.settings_anthropometry_model_combo,
                    self.settings_anthropometry_sex_combo,
                    self.settings_anthropometry_mass_edit,
                ):
                    widget.blockSignals(False)
                self._suspend_segment_settings_preview_updates = False
            self._sync_segment_settings_validation_style()
            self._update_segment_settings_preview()

        def _sync_segment_settings_validation_style(self, *_args) -> bool:
            """
            Highlight q range fields when their value count does not match the selected DoF count.
            """
            translations = self.settings_translations_edit.text().strip().replace("-", "")
            rotations = self.settings_rotations_edit.text().strip().replace("-", "")
            dof_count = len(translations) + len(rotations)
            q_min_count = _safe_float_count(self.settings_q_min_edit.text())
            q_max_count = _safe_float_count(self.settings_q_max_edit.text())
            q_min_valid = q_min_count in {0, dof_count}
            q_max_valid = q_max_count in {0, dof_count}
            initial_rotation_method = self.settings_initial_rotation_method_combo.currentText()
            matrix_valid = True
            if initial_rotation_method == "matrix":
                try:
                    _parse_rotation_matrix_text(self.settings_initial_rotation_source_edit.text())
                except ValueError:
                    matrix_valid = False
            problem_style = "background: #fff1f2; border: 2px solid #dc2626; color: #7f1d1d;"
            self.settings_q_min_edit.setStyleSheet("" if q_min_valid else problem_style)
            self.settings_q_max_edit.setStyleSheet("" if q_max_valid else problem_style)
            self.settings_initial_rotation_source_edit.setStyleSheet("" if matrix_valid else problem_style)
            return q_min_valid and q_max_valid and matrix_valid

        def _apply_workflow_segment_settings_from_form(self) -> None:
            setting = self._selected_segment_setting()
            if setting is None:
                return
            if not self._sync_segment_settings_validation_style():
                QMessageBox.critical(
                    self,
                    "Invalid chain settings",
                    "q min/q max must be empty or contain exactly one value per DoF, "
                    "and Matrix must be a valid 3x3 numeric matrix when initial rotation is set to matrix.",
                )
                return
            initial_rotation_method = self.settings_initial_rotation_method_combo.currentText()
            initial_rotation_source = ""
            initial_rotation_matrix = None
            if initial_rotation_method == "matrix":
                try:
                    initial_rotation_matrix = _parse_rotation_matrix_text(
                        self.settings_initial_rotation_source_edit.text()
                    )
                except ValueError as error:
                    QMessageBox.critical(
                        self,
                        "Invalid initial rotation matrix",
                        f"{error}\n\n"
                        "Use a 3x3 matrix such as [[1,0,0], [0,1,0], [0,0,1]], "
                        "or rows separated by ';' / new lines.",
                    )
                    return
                initial_rotation_source = _format_rotation_matrix(initial_rotation_matrix)
            if initial_rotation_method == "anatomical_c3d":
                initial_rotation_source = self._selected_initial_rotation_c3d_file()
            try:
                self.workflow_draft = update_segment_settings_in_draft(
                    self.workflow_draft,
                    segment_name=setting.segment_name,
                    translations=self.settings_translations_edit.text(),
                    rotations=self.settings_rotations_edit.text(),
                    q_min=tuple(_parse_float_list(self.settings_q_min_edit.text())),
                    q_max=tuple(_parse_float_list(self.settings_q_max_edit.text())),
                    child_translation=self.settings_child_translation_checkbox.isChecked(),
                    initial_rotation_method=initial_rotation_method,
                    initial_rotation_source=initial_rotation_source,
                    initial_rotation_matrix=initial_rotation_matrix,
                    anthropometry_model=_settings_anthropometry_model(self.settings_anthropometry_model_combo),
                    anthropometry_sex=self.settings_anthropometry_sex_combo.currentText(),
                    anthropometry_mass=_parse_optional_float(self.settings_anthropometry_mass_edit.text()),
                    segment_length=_parse_optional_float(self.settings_segment_length_edit.text()),
                    segment_length_source=self.settings_segment_length_source_label.text(),
                )
                self._update_preset_details()
            except Exception as error:
                QMessageBox.critical(self, "Unable to apply segment settings", str(error))

        def _apply_anthropometry_to_selected_segment(self) -> None:
            setting = self._selected_segment_setting()
            if setting is None:
                return
            self.settings_anthropometry_model_combo.setCurrentText("de_leva")
            self._refresh_segment_length_fields(setting, force_recompute=True)
            self._apply_workflow_segment_settings_from_form()

        def _refresh_segment_length_fields(self, setting, force_recompute: bool = False) -> None:
            if self.c3d_data is None:
                self.settings_segment_length_edit.setText(
                    "" if setting.segment_length is None else f"{setting.segment_length:.6g}"
                )
                self.settings_segment_length_source_label.setText(
                    setting.segment_length_source or "No main C3D loaded."
                )
                return
            if setting.segment_length is not None and not force_recompute:
                self.settings_segment_length_edit.setText(f"{setting.segment_length:.6g}")
                self.settings_segment_length_source_label.setText(setting.segment_length_source or "-")
                return
            length, source = _segment_length_from_draft(self.workflow_draft, self.c3d_data, setting.segment_name)
            self.settings_segment_length_edit.setText("" if length is None else f"{length:.6g}")
            self.settings_segment_length_source_label.setText(source)

        def _edit_workflow_segment_settings(self) -> None:
            setting = self._selected_segment_setting()
            if setting is None:
                return
            translations, accepted = QInputDialog.getText(
                self,
                "Segment translations",
                "Translations",
                text=setting.translations,
            )
            if not accepted:
                return
            rotations, accepted = QInputDialog.getText(self, "Segment rotations", "Rotations", text=setting.rotations)
            if not accepted:
                return
            q_min, accepted = QInputDialog.getText(
                self,
                "q min",
                "q min values",
                text=_format_float_list(list(setting.q_min)),
            )
            if not accepted:
                return
            q_max, accepted = QInputDialog.getText(
                self,
                "q max",
                "q max values",
                text=_format_float_list(list(setting.q_max)),
            )
            if not accepted:
                return
            child_translation, accepted = QInputDialog.getItem(
                self,
                "Child translation",
                "Allow child translation",
                ["no", "yes"],
                1 if setting.child_translation else 0,
                False,
            )
            if not accepted:
                return
            initial_rotation_method, accepted = QInputDialog.getItem(
                self,
                "Initial rotation",
                "Method",
                ["identity", "matrix", "anatomical_c3d"],
                (
                    ["identity", "matrix", "anatomical_c3d"].index(setting.initial_rotation_method)
                    if setting.initial_rotation_method in {"identity", "matrix", "anatomical_c3d"}
                    else 0
                ),
                False,
            )
            if not accepted:
                return
            initial_rotation_source, accepted = QInputDialog.getText(
                self,
                "Initial rotation source",
                "Matrix values or anatomical C3D",
                text=setting.initial_rotation_source,
            )
            if not accepted:
                return
            try:
                self.workflow_draft = update_segment_settings_in_draft(
                    self.workflow_draft,
                    segment_name=setting.segment_name,
                    translations=translations,
                    rotations=rotations,
                    q_min=tuple(_parse_float_list(q_min)),
                    q_max=tuple(_parse_float_list(q_max)),
                    child_translation=child_translation == "yes",
                    initial_rotation_method=initial_rotation_method,
                    initial_rotation_source=initial_rotation_source,
                    anthropometry_model=setting.anthropometry_model,
                    anthropometry_sex=setting.anthropometry_sex,
                    anthropometry_mass=setting.anthropometry_mass,
                    segment_length=setting.segment_length,
                    segment_length_source=setting.segment_length_source,
                )
                self._update_preset_details()
            except Exception as error:
                QMessageBox.critical(self, "Unable to edit segment settings", str(error))

        def _assign_workflow_c3d_role(self) -> None:
            role = self._selected_c3d_role()
            if role is None:
                return
            filepath, _ = QFileDialog.getOpenFileName(
                self,
                "Assign C3D file",
                self.c3d_folder_path,
                "C3D files (*.c3d)",
            )
            if not filepath:
                return
            self.workflow_draft = assign_c3d_file_role_to_draft(self.workflow_draft, role, filepath)
            self._update_preset_details()

        def _clear_workflow_c3d_role(self) -> None:
            role = self._selected_c3d_role()
            if role is None:
                return
            self.workflow_draft = clear_c3d_file_role_from_draft(self.workflow_draft, role)
            self._update_preset_details()

        def _set_workflow_segment_parent(self, _parent_name: str | None = None) -> None:
            segment_name = self._selected_workflow_segment_name()
            if segment_name is None:
                return
            try:
                self.workflow_draft = update_segment_parent_in_draft(
                    self.workflow_draft,
                    segment_name,
                    self.workflow_parent_combo.currentText(),
                )
                self._update_preset_details()
            except Exception as error:
                QMessageBox.critical(self, "Unable to update segment parent", str(error))

        def _add_selected_axis_markers(self, vector_index: int, endpoint: str) -> None:
            target_list = self._axis_endpoint_list(vector_index, endpoint)
            self._add_selected_axis_markers_to_list(target_list)
            self._sync_axis_vector_endpoint_state(vector_index)
            self._update_segment_axis_preview()

        def _add_selected_axis_origin_markers(self) -> None:
            self._add_selected_axis_markers_to_list(self.axis_origin_marker_list)
            self._update_segment_axis_preview()

        def _add_selected_axis_markers_to_list(self, target_list) -> None:
            for item in self.axis_marker_source_list.selectedItems():
                source_name = _axis_source_name_from_list_text(item.text())
                if source_name:
                    target_list.addItem(source_name)

        def _remove_selected_axis_markers(self, vector_index: int, endpoint: str) -> None:
            target_list = self._axis_endpoint_list(vector_index, endpoint)
            self._remove_selected_axis_markers_from_list(target_list)
            self._sync_axis_vector_endpoint_state(vector_index)
            self._update_segment_axis_preview()

        def _remove_selected_axis_origin_markers(self) -> None:
            self._remove_selected_axis_markers_from_list(self.axis_origin_marker_list)
            self._update_segment_axis_preview()

        def _remove_selected_axis_markers_from_list(self, target_list) -> None:
            for item in target_list.selectedItems():
                target_list.takeItem(target_list.row(item))

        def _swap_axis_vector_endpoints(self, vector_index: int) -> None:
            controls = self.axis_vector_controls[vector_index]
            start_names = _list_widget_texts(controls["start_list"])
            end_names = _list_widget_texts(controls["end_list"])
            controls["start_list"].clear()
            controls["end_list"].clear()
            controls["start_list"].addItems(end_names)
            controls["end_list"].addItems(start_names)
            self._sync_axis_vector_endpoint_state(vector_index)
            self._update_segment_axis_preview()

        def _axis_endpoint_list(self, vector_index: int, endpoint: str):
            key = "start_list" if endpoint == "start" else "end_list"
            return self.axis_vector_controls[vector_index][key]

        def _sync_axis_vector_endpoint_state(self, vector_index: int) -> None:
            controls = self.axis_vector_controls[vector_index]
            start_markers = _list_widget_texts(controls["start_list"])
            has_axis_source = self._contains_virtual_axis_source(start_markers)
            if has_axis_source:
                controls["end_list"].clear()
            controls["end_list"].setEnabled(not has_axis_source)
            controls["add_end_button"].setEnabled(not has_axis_source)
            controls["remove_end_button"].setEnabled(not has_axis_source)
            controls["end_label"].setEnabled(not has_axis_source)
            controls["swap_button"].setEnabled(not has_axis_source)

        def _contains_virtual_axis_source(self, source_names: tuple[str, ...]) -> bool:
            return _virtual_axis_from_source_names(source_names, self.workflow_draft.axes) is not None

        def _save_segment_axis_from_lists(self) -> None:
            segment_name = self._selected_anatomical_segment_name()
            if segment_name is None:
                return
            vector_specs = self._axis_vector_specs()
            origin_markers = _list_widget_texts(self.axis_origin_marker_list)
            if len(vector_specs) != 2:
                QMessageBox.critical(
                    self,
                    "Unable to save segment axis",
                    "Two complete vectors are required.",
                )
                return
            if len(origin_markers) == 0:
                QMessageBox.critical(
                    self,
                    "Unable to save segment axis",
                    "At least one origin marker is required.",
                )
                return
            if sum(keep_vector for _, _, _, keep_vector in vector_specs) != 1:
                QMessageBox.critical(
                    self,
                    "Unable to save segment axis",
                    "Choose exactly one vector to keep.",
                )
                return
            if len({axis_name for axis_name, _, _, _ in vector_specs}) != 2:
                QMessageBox.critical(
                    self,
                    "Unable to save segment axis",
                    "The two vectors must use two different axes. The third axis is computed by cross product.",
                )
                return
            try:
                # Saving from this tab replaces the whole local-frame recipe for the selected segment.
                # Functional axes such as Axis_LKnee_SARA are kept because they are sources, not the final frame.
                updated_draft = replace(
                    self.workflow_draft,
                    axes=tuple(
                        axis
                        for axis in self.workflow_draft.axes
                        if axis.segment_name != segment_name or _is_virtual_feature_axis(axis)
                    ),
                )
                for index, (
                    axis_name,
                    start_markers,
                    end_markers,
                    keep_vector,
                ) in enumerate(vector_specs, start=1):
                    method = "sara_direction" if self._contains_virtual_axis_source(start_markers) else "markers"
                    updated_draft = add_axis_to_draft(
                        updated_draft,
                        name=f"{segment_name}_frame_vector_{index}",
                        segment_name=segment_name,
                        axis=axis_name,
                        start_markers=start_markers,
                        end_markers=end_markers,
                        origin_markers=origin_markers,
                        method=method,
                        keep_vector=keep_vector,
                    )
                self.workflow_draft = updated_draft
                self._update_preset_details()
            except Exception as error:
                QMessageBox.critical(self, "Unable to save segment axis", str(error))

        def _axis_vector_specs(
            self,
        ) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...], bool], ...]:
            specs = []
            for controls in self.axis_vector_controls:
                axis_name = controls["axis_combo"].currentText()
                start_markers = _list_widget_texts(controls["start_list"])
                end_markers = _list_widget_texts(controls["end_list"])
                if self._contains_virtual_axis_source(start_markers):
                    specs.append(
                        (
                            axis_name,
                            start_markers,
                            (),
                            controls["keep_checkbox"].isChecked(),
                        )
                    )
                elif len(start_markers) != 0 and len(end_markers) != 0:
                    specs.append(
                        (
                            axis_name,
                            start_markers,
                            end_markers,
                            controls["keep_checkbox"].isChecked(),
                        )
                    )
            return tuple(specs)

        def _choose_workflow_segment(self, title: str, current_segment_name: str = "") -> str | None:
            segment_names = [group.segment_name for group in self.workflow_draft.segment_marker_groups]
            if not segment_names:
                return None
            segment_index = segment_names.index(current_segment_name) if current_segment_name in segment_names else 0
            segment_name, accepted = QInputDialog.getItem(self, title, "Segment", segment_names, segment_index, False)
            return segment_name if accepted else None

        def _selected_workflow_segment_name(self) -> str | None:
            if not self.segment_marker_list.selectedItems():
                return None
            return self.segment_marker_list.selectedItems()[0].text().split(":", maxsplit=1)[0]

        def _selected_anatomical_segment_name(self) -> str | None:
            if not self.anatomical_segment_list.selectedItems():
                return None
            return self.anatomical_segment_list.selectedItems()[0].text().split(":", maxsplit=1)[0]

        def _selected_workflow_marker_name(self) -> str | None:
            marker_names = self._selected_workflow_marker_names()
            if len(marker_names) == 0:
                return None
            return marker_names[0]

        def _selected_workflow_marker_names(self) -> tuple[str, ...]:
            marker_names = []
            for item in self.marker_list.selectedItems():
                marker_name = item.text()
                if marker_name.startswith("Choose ") or marker_name.startswith("No available"):
                    continue
                marker_names.append(marker_name)
            return tuple(marker_names)

        def _selected_assigned_marker_names(self) -> tuple[str, ...]:
            marker_names = []
            for item in self.assigned_marker_list.selectedItems():
                marker_name = item.text().split("|", maxsplit=1)[0].strip()
                if marker_name.startswith("Select a segment") or marker_name.startswith("No marker"):
                    continue
                marker_names.append(marker_name)
            return tuple(marker_names)

        def _update_assigned_marker_list(self) -> None:
            self.assigned_marker_list.clear()
            self._sync_assigned_marker_technical_checkbox()
            self._sync_workflow_parent_combo()
            self._update_technical_segment_preview()
            segment_name = self._selected_workflow_segment_name()
            if segment_name is None:
                self.assigned_marker_list.addItem("Select a segment to inspect its markers.")
                return
            for group in self.workflow_draft.segment_marker_groups:
                if group.segment_name != segment_name:
                    continue
                if len(group.marker_names) == 0:
                    self.assigned_marker_list.addItem("No marker assigned to this segment.")
                    return
                for marker_name in group.marker_names:
                    marker_kind = "technical" if marker_name in group.technical_marker_names else "additional"
                    self.assigned_marker_list.addItem(f"{marker_name} | {marker_kind}")
                return

        def _configure_technical_frame_slider(self) -> None:
            frame_count = 0 if self.c3d_data is None else self.c3d_data.nb_frames
            self.technical_frame_slider.blockSignals(True)
            self.technical_frame_slider.setEnabled(frame_count > 1)
            self.technical_frame_slider.setMinimum(0)
            self.technical_frame_slider.setMaximum(max(frame_count - 1, 0))
            self.technical_frame_slider.setValue(0)
            self.technical_frame_slider.blockSignals(False)
            self._sync_workflow_frame_play_button(self.technical_frame_slider)
            self._update_technical_segment_preview()

        def _configure_anatomical_frame_slider(self) -> None:
            frame_count = 0 if self.c3d_data is None else self.c3d_data.nb_frames
            self.anatomical_frame_slider.blockSignals(True)
            self.anatomical_frame_slider.setEnabled(frame_count > 1)
            self.anatomical_frame_slider.setMinimum(0)
            self.anatomical_frame_slider.setMaximum(max(frame_count - 1, 0))
            self.anatomical_frame_slider.setValue(0)
            self.anatomical_frame_slider.blockSignals(False)
            self._sync_workflow_frame_play_button(self.anatomical_frame_slider)
            self._update_segment_axis_preview()

        def _configure_segment_settings_frame_slider(self) -> None:
            frame_count = 0 if self.c3d_data is None else self.c3d_data.nb_frames
            is_q0_mode = self._segment_settings_preview_mode == "q0"
            self.segment_settings_frame_slider.blockSignals(True)
            self.segment_settings_frame_slider.setEnabled(frame_count > 1 and not is_q0_mode)
            self.segment_settings_frame_slider.setMinimum(0)
            self.segment_settings_frame_slider.setMaximum(max(frame_count - 1, 0))
            self.segment_settings_frame_slider.setValue(0)
            self.segment_settings_frame_slider.blockSignals(False)
            self._sync_workflow_frame_play_button(self.segment_settings_frame_slider)
            self._update_segment_settings_preview()

        def _update_technical_segment_preview(self, *_args) -> None:
            frame_index = self.technical_frame_slider.value()
            frame_count = 0 if self.c3d_data is None else self.c3d_data.nb_frames
            if frame_count == 0:
                self.technical_frame_label.setText("Frame 0/0")
            else:
                self.technical_frame_label.setText(f"Frame {frame_index + 1}/{frame_count}")
            self.technical_segment_preview.set_context(
                self.c3d_data,
                self.workflow_draft.segment_marker_groups,
                self._selected_workflow_segment_name() or "",
                self._selected_workflow_marker_names(),
                frame_index,
            )

        def _update_segment_settings_preview(self, *_args) -> None:
            if self._suspend_segment_settings_preview_updates:
                return
            frame_index = self.segment_settings_frame_slider.value()
            frame_count = 0 if self.c3d_data is None else self.c3d_data.nb_frames
            is_q0_mode = self._segment_settings_preview_mode == "q0"
            self.segment_settings_frame_slider.setEnabled(frame_count > 1 and not is_q0_mode)
            self._sync_workflow_frame_play_button(self.segment_settings_frame_slider)
            if is_q0_mode:
                self.segment_settings_frame_label.setText("q0")
                frame_index = 0
            elif frame_count == 0:
                self.segment_settings_frame_label.setText("Frame 0/0")
            else:
                self.segment_settings_frame_label.setText(f"Frame {frame_index + 1}/{frame_count}")
            selected_setting = self._selected_segment_setting()
            overrides = {}
            if selected_setting is not None:
                initial_rotation_method = self.settings_initial_rotation_method_combo.currentText()
                initial_rotation_source = ""
                initial_rotation_matrix = selected_setting.initial_rotation_matrix
                if initial_rotation_method == "matrix":
                    try:
                        initial_rotation_matrix = _parse_rotation_matrix_text(
                            self.settings_initial_rotation_source_edit.text()
                        )
                        initial_rotation_source = _format_rotation_matrix(initial_rotation_matrix)
                    except ValueError:
                        pass
                elif initial_rotation_method == "anatomical_c3d":
                    initial_rotation_source = self._selected_initial_rotation_c3d_file()
                edited_setting = replace(
                    selected_setting,
                    translations=self.settings_translations_edit.text(),
                    rotations=self.settings_rotations_edit.text(),
                    child_translation=self.settings_child_translation_checkbox.isChecked(),
                    initial_rotation_method=initial_rotation_method,
                    initial_rotation_source=initial_rotation_source,
                    initial_rotation_matrix=initial_rotation_matrix,
                )
                if _chain_setting_preview_signature(edited_setting) != _chain_setting_preview_signature(
                    selected_setting
                ):
                    overrides[selected_setting.segment_name] = edited_setting
            q0_scene = self._segment_settings_q0_scene(overrides) if is_q0_mode else None
            self.segment_settings_preview.set_context(
                self.c3d_data,
                self.workflow_draft,
                "" if selected_setting is None else selected_setting.segment_name,
                frame_index,
                self._virtual_feature_c3d_preview_data_map(),
                overrides,
                q0_scene,
            )

        def _segment_settings_q0_scene(self, segment_setting_overrides: dict[str, object]) -> dict[str, object]:
            """
            Return the kinematic-chain scene at q=0 for the current generated model.
            """
            if self.c3d_data is None:
                return C3dSegmentSettingsPreviewWidget._empty_settings_scene("Choose a C3D to preview q0")
            override_signature = tuple(
                sorted(
                    (
                        name,
                        setting.translations,
                        setting.rotations,
                        setting.child_translation,
                    )
                    for name, setting in segment_setting_overrides.items()
                )
            )
            cache_key = (
                id(self.c3d_data),
                id(self.workflow_draft),
                self._virtual_feature_c3d_preview_data_cache_key(),
                override_signature,
            )
            if (
                self._segment_settings_q0_scene_cache is not None
                and self._segment_settings_q0_scene_cache_signature == cache_key
            ):
                return self._segment_settings_q0_scene_cache
            try:
                model = self._diagnostic_model_from_current_preset(
                    "",
                    lambda *_args: None,
                    segment_setting_overrides=segment_setting_overrides,
                )
                q0_scene = self._q0_scene_from_model(model)
            except Exception as error:
                q0_scene = C3dSegmentSettingsPreviewWidget._empty_settings_scene(
                    f"Unable to build q0 preview:\n{error}"
                )
            self._segment_settings_q0_scene_cache = q0_scene
            self._segment_settings_q0_scene_cache_signature = cache_key
            return q0_scene

        def _q0_scene_from_model(self, model) -> dict[str, object]:
            """
            Convert model forward kinematics at q=0 into the chain-preview scene format.
            """
            model.segments_rt_to_local()
            rt_by_segment = model.forward_kinematics(np.zeros((model.nb_q, 1)))
            segment_origins = {}
            segment_frames = {}
            marker_points = []
            points = []
            for group in self.workflow_draft.segment_marker_groups:
                rt_series = rt_by_segment.get(group.segment_name)
                if rt_series is None or len(rt_series) == 0:
                    continue
                rt = rt_series[0]
                origin = _finite_point3d(rt.translation[:3])
                if origin is None:
                    continue
                rotation = rt.rotation_matrix.rotation_matrix
                segment_origins[group.segment_name] = origin
                segment_frames[group.segment_name] = {
                    "x": rotation[:, 0],
                    "y": rotation[:, 1],
                    "z": rotation[:, 2],
                }
                points.append(origin)
            links = []
            for group in self.workflow_draft.segment_marker_groups:
                parent_name = group.parent_name
                if parent_name in {"", "root"}:
                    continue
                start = segment_origins.get(parent_name)
                end = segment_origins.get(group.segment_name)
                if start is None or end is None:
                    continue
                links.append((parent_name, group.segment_name, start, end))
                points.extend((start, end))
            scene_span = C3dSegmentSettingsPreviewWidget._scene_span(points)
            frame_length = max(scene_span * 0.12, 1e-6)
            for segment_name, origin in segment_origins.items():
                origin_array = np.asarray(origin, dtype=float)
                for axis_vector in segment_frames.get(segment_name, {}).values():
                    endpoint = origin_array + frame_length * axis_vector
                    points.append(tuple(float(value) for value in endpoint))
            return {
                "segment_origins": segment_origins,
                "segment_frames": segment_frames,
                "links": links,
                "frame_length": frame_length,
                "marker_points": marker_points,
                "points": points,
            }

        def _update_anatomical_segment_details(self) -> None:
            self._suspend_anatomical_preview_updates = True
            try:
                self._update_axis_marker_source_list()
                self._load_selected_segment_axes_into_controls()
            finally:
                self._suspend_anatomical_preview_updates = False
            self._update_segment_axis_preview()

        def _sync_assigned_marker_technical_checkbox(self) -> None:
            marker_names = self._selected_assigned_marker_names()
            segment_name = self._selected_workflow_segment_name()
            is_checked = False
            is_enabled = segment_name is not None and len(marker_names) != 0
            if is_enabled:
                for group in self.workflow_draft.segment_marker_groups:
                    if group.segment_name == segment_name:
                        is_checked = all(marker_name in group.technical_marker_names for marker_name in marker_names)
                        break
            self.assigned_marker_technical_checkbox.blockSignals(True)
            self.assigned_marker_technical_checkbox.setEnabled(is_enabled)
            self.assigned_marker_technical_checkbox.setChecked(is_checked)
            self.assigned_marker_technical_checkbox.blockSignals(False)

        def _set_selected_assigned_markers_technical(self) -> None:
            segment_name = self._selected_workflow_segment_name()
            marker_names = self._selected_assigned_marker_names()
            if segment_name is None or len(marker_names) == 0:
                return
            try:
                self.workflow_draft = set_segment_marker_technical(
                    self.workflow_draft,
                    segment_name,
                    marker_names,
                    self.assigned_marker_technical_checkbox.isChecked(),
                )
                self._refresh_marker_assignment_details(segment_name)
            except Exception as error:
                QMessageBox.critical(self, "Unable to update marker type", str(error))

        def _sync_workflow_parent_combo(self) -> None:
            segment_name = self._selected_workflow_segment_name()
            self.workflow_parent_combo.blockSignals(True)
            self.workflow_parent_combo.clear()
            if segment_name is None:
                self.workflow_parent_combo.setEnabled(False)
                self.workflow_parent_combo.blockSignals(False)
                return
            choices = _segment_parent_choices(self.workflow_draft, excluded_segment_name=segment_name)
            self.workflow_parent_combo.addItems(choices)
            current_parent = ""
            for group in self.workflow_draft.segment_marker_groups:
                if group.segment_name == segment_name:
                    current_parent = group.parent_name
                    break
            if current_parent not in choices:
                self.workflow_parent_combo.addItem(current_parent)
            self.workflow_parent_combo.setCurrentText(current_parent)
            self.workflow_parent_combo.setEnabled(True)
            self.workflow_parent_combo.blockSignals(False)

        def _update_axis_marker_source_list(self) -> None:
            self.axis_marker_source_list.clear()
            segment_name = self._selected_anatomical_segment_name()
            if segment_name is None:
                self.axis_marker_source_list.addItem("Select a segment to list axis markers.")
                return
            for source_label in _anatomical_axis_source_labels(self.workflow_draft, self.workflow_marker_pool):
                self.axis_marker_source_list.addItem(source_label)

        def _load_selected_segment_axes_into_controls(self) -> None:
            segment_name = self._selected_anatomical_segment_name()
            axes = self._local_frame_axes_for_segment(segment_name)[:2]
            self.axis_origin_marker_list.clear()
            origin_markers = axes[0].origin_markers if len(axes) != 0 else ()
            for marker_name in origin_markers:
                self.axis_origin_marker_list.addItem(marker_name)
            for index, controls in enumerate(self.axis_vector_controls):
                axis_combo = controls["axis_combo"]
                keep_checkbox = controls["keep_checkbox"]
                axis_combo.blockSignals(True)
                keep_checkbox.blockSignals(True)
                controls["start_list"].clear()
                controls["end_list"].clear()
                axis_combo.setCurrentText("x" if index == 0 else "y")
                keep_checkbox.setChecked(index == 0)
                if index >= len(axes):
                    axis_combo.blockSignals(False)
                    keep_checkbox.blockSignals(False)
                    self._sync_axis_vector_endpoint_state(index)
                    continue
                axis = axes[index]
                fallback_axis = "x" if index == 0 else "y"
                axis_combo.setCurrentText(axis.axis if axis.axis in {"x", "y", "z"} else fallback_axis)
                keep_checkbox.setChecked(axis.keep_vector)
                for marker_name in axis.start_markers:
                    controls["start_list"].addItem(marker_name)
                for marker_name in axis.end_markers:
                    controls["end_list"].addItem(marker_name)
                axis_combo.blockSignals(False)
                keep_checkbox.blockSignals(False)
                self._sync_axis_vector_endpoint_state(index)
            self._ensure_single_kept_axis_vector(0)

        def _local_frame_axes_for_segment(self, segment_name: str | None):
            if segment_name is None:
                return ()
            return tuple(
                axis
                for axis in self.workflow_draft.axes
                if axis.segment_name == segment_name and not _is_virtual_feature_axis(axis)
            )

        def _ensure_single_kept_axis_vector(self, selected_index: int) -> None:
            """
            Keep exactly one anatomical vector as the source vector for orthonormalization.
            """
            checked_indices = [
                index
                for index, controls in enumerate(self.axis_vector_controls)
                if controls["keep_checkbox"].isChecked()
            ]
            index_to_keep = (
                selected_index if selected_index in checked_indices else (checked_indices[0] if checked_indices else 0)
            )
            for index, controls in enumerate(self.axis_vector_controls):
                checkbox = controls["keep_checkbox"]
                checkbox.blockSignals(True)
                checkbox.setChecked(index == index_to_keep)
                checkbox.blockSignals(False)
            self._update_segment_axis_preview()

        def _update_segment_axis_preview(self) -> None:
            if self._suspend_anatomical_preview_updates:
                return
            segment_name = self._selected_anatomical_segment_name()
            if segment_name is None:
                self.segment_axis_preview.set_context(self.c3d_data, (), (), (), "", (), ())
                return
            frame_index = self.anatomical_frame_slider.value()
            frame_count = 0 if self.c3d_data is None else self.c3d_data.nb_frames
            if frame_count == 0:
                self.anatomical_frame_label.setText("Frame 0/0")
            else:
                self.anatomical_frame_label.setText(f"Frame {frame_index + 1}/{frame_count}")
            segment_marker_names = tuple(self.workflow_marker_pool)
            label_marker_names = self._marker_names_for_segment(segment_name)
            segment_axes = self._local_frame_axes_for_segment(segment_name)
            current_vector_specs = self._axis_vector_specs()
            referenced_axis_names = {
                source_name
                for axis in segment_axes
                for source_name in axis.start_markers + axis.end_markers
                if self._contains_virtual_axis_source((source_name,))
            }
            referenced_axis_names.update(
                source_name
                for _, start_markers, end_markers, _ in current_vector_specs
                for source_name in start_markers + end_markers
                if self._contains_virtual_axis_source((source_name,))
            )
            referenced_axes = tuple(
                axis
                for axis in self.workflow_draft.axes
                if _is_virtual_feature_axis(axis) and axis.name in referenced_axis_names
            )
            axes = tuple(dict.fromkeys(segment_axes + referenced_axes))
            self.segment_axis_preview.set_context(
                self.c3d_data,
                segment_marker_names,
                label_marker_names,
                self.workflow_draft.segment_marker_groups,
                segment_name,
                axes,
                current_vector_specs,
                self.workflow_draft.virtual_markers,
                self._virtual_feature_c3d_preview_data_map(),
                _list_widget_texts(self.axis_origin_marker_list),
                frame_index,
                self.use_diverse_functional_frames_checkbox.isChecked(),
                self._manual_functional_frame_indices(),
            )

        def _virtual_feature_c3d_preview_data_map(self) -> dict[str, object]:
            cache_key = self._virtual_feature_c3d_preview_data_cache_key()
            if (
                self._virtual_feature_c3d_preview_data_cache is not None
                and self._virtual_feature_c3d_preview_data_cache_signature == cache_key
            ):
                return self._virtual_feature_c3d_preview_data_cache
            c3d_data_by_key = {}
            for feature in tuple(self.workflow_draft.virtual_markers) + tuple(self.workflow_draft.axes):
                sources = (
                    getattr(feature, "source", ""),
                    getattr(feature, "equation", ""),
                )
                for source in sources:
                    c3d_name = _c3d_source_name_from_virtual_feature_source(source)
                    trial_name = _trial_name_from_virtual_feature_source(source)
                    filepath = ""
                    if c3d_name:
                        filepath = str(Path(self.c3d_folder_path) / c3d_name) if self.c3d_folder_path else c3d_name
                    elif trial_name:
                        filepath = _assigned_c3d_source_for_role(self.workflow_draft, trial_name)
                    if not filepath:
                        continue
                    try:
                        data = self._c3d_data_from_path(filepath)
                    except Exception:
                        continue
                    if c3d_name:
                        c3d_data_by_key[c3d_name] = data
                    if trial_name:
                        c3d_data_by_key[trial_name] = data
            self._virtual_feature_c3d_preview_data_cache = c3d_data_by_key
            self._virtual_feature_c3d_preview_data_cache_signature = cache_key
            return c3d_data_by_key

        def _virtual_feature_c3d_preview_data_cache_key(self) -> tuple[object, ...]:
            features = tuple(
                (
                    feature.__class__.__name__,
                    getattr(feature, "name", ""),
                    getattr(feature, "method", ""),
                    getattr(feature, "source", ""),
                    getattr(feature, "equation", ""),
                )
                for feature in tuple(self.workflow_draft.virtual_markers) + tuple(self.workflow_draft.axes)
            )
            assignments = tuple(
                (assignment.role, assignment.generic_name, assignment.source_path)
                for assignment in self.workflow_draft.file_assignments
            )
            return (
                id(self.workflow_draft),
                self.c3d_folder_path,
                self.strip_participant_prefix_checkbox.isChecked(),
                features,
                assignments,
            )

        def _invalidate_virtual_feature_c3d_preview_cache(self) -> None:
            self._virtual_feature_c3d_preview_data_cache = None
            self._virtual_feature_c3d_preview_data_cache_signature = None
            self._segment_settings_q0_scene_cache = None
            self._segment_settings_q0_scene_cache_signature = None

        def _marker_names_for_segment(self, segment_name: str) -> tuple[str, ...]:
            marker_names = []
            for group in self.workflow_draft.segment_marker_groups:
                if group.segment_name == segment_name:
                    marker_names.extend(group.marker_names)
                    marker_names.extend(group.technical_marker_names)
                    break
            marker_names.extend(
                marker.name for marker in self.workflow_draft.virtual_markers if marker.segment_name == segment_name
            )
            return tuple(dict.fromkeys(marker_names))

        def _selected_virtual_marker_name(self) -> str | None:
            if not self.feature_list.selectedItems():
                return None
            text = self.feature_list.selectedItems()[0].text()
            if text.startswith("No additional") or text.startswith("[axis]"):
                return None
            return text.split("|", maxsplit=1)[0].strip()

        def _selected_virtual_axis(self):
            if not self.feature_list.selectedItems():
                return None
            text = self.feature_list.selectedItems()[0].text()
            axis_name = _virtual_axis_name_from_feature_list_text(text)
            if axis_name is None:
                return None
            for axis in self.workflow_draft.axes:
                if axis.name == axis_name:
                    return axis
            return None

        def _selected_virtual_marker(self):
            name = self._selected_virtual_marker_name()
            if name is None:
                return None
            for marker in self.workflow_draft.virtual_markers:
                if marker.name == name:
                    return marker
            return None

        def _select_virtual_marker_by_name(self, marker_name: str) -> None:
            for index in range(self.feature_list.count()):
                item = self.feature_list.item(index)
                if item.text().split("|", maxsplit=1)[0].strip() == marker_name:
                    self.feature_list.setCurrentItem(item)
                    return

        def _load_selected_virtual_marker_into_form(self) -> None:
            self._sync_functional_residual_button()
            marker = self._selected_virtual_marker()
            if marker is None:
                axis = self._selected_virtual_axis()
                if axis is not None:
                    self.virtual_marker_name_edit.setText(axis.name)
                    self.virtual_marker_segment_combo.setCurrentText(axis.segment_name)
                    parent_name = _parent_segment_name(self.workflow_draft, axis.segment_name)
                    if parent_name and self.virtual_marker_proximal_combo.findText(parent_name) >= 0:
                        self.virtual_marker_proximal_combo.setCurrentText(parent_name)
                    if self.virtual_marker_distal_combo.findText(axis.segment_name) >= 0:
                        self.virtual_marker_distal_combo.setCurrentText(axis.segment_name)
                    self._set_virtual_marker_method(axis.method)
                    self.virtual_marker_source_edit.setText(axis.source)
                    self.virtual_marker_equation_edit.setText("")
                    self._select_virtual_marker_c3d_from_source(axis.source)
                    self._sync_virtual_marker_method_fields()
                    self._update_virtual_axis_info_label(axis)
                    self._update_virtual_marker_preview()
                    return
                self._sync_virtual_marker_method_fields()
                self._update_virtual_marker_preview()
                return
            self.virtual_marker_name_edit.setText(marker.name)
            self.virtual_marker_segment_combo.setCurrentText(marker.segment_name)
            if marker.method in PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS:
                self._set_virtual_marker_method("predictive")
                self.virtual_marker_predictive_method_combo.setCurrentText(
                    PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS[marker.method]
                )
            else:
                self._set_virtual_marker_method(marker.method)
            self.virtual_marker_source_edit.setText(marker.source)
            self.virtual_marker_equation_edit.setText(marker.equation if marker.method == "axis_projection" else "")
            self._load_axis_projection_payload_into_lists(marker.source, marker.equation)
            proximal, distal = _score_segments_from_payload(marker.equation)
            if proximal:
                self.virtual_marker_proximal_combo.setCurrentText(proximal)
            if distal:
                self.virtual_marker_distal_combo.setCurrentText(distal)
            self._select_virtual_marker_c3d_from_source("; ".join((marker.source, marker.equation)))
            self._sync_virtual_marker_method_fields()
            self._update_virtual_marker_info_label(marker)
            self._update_virtual_marker_preview()

        def _select_virtual_marker_c3d_from_source(self, source: str) -> None:
            source_name = _c3d_source_name_from_virtual_feature_source(source)
            if not source_name:
                trial_name = _trial_name_from_virtual_feature_source(source)
                source_name = Path(_assigned_c3d_source_for_role(self.workflow_draft, trial_name)).name
            if not source_name:
                return
            if self.virtual_marker_c3d_file_combo.findText(source_name) >= 0:
                self.virtual_marker_c3d_file_combo.setCurrentText(source_name)

        def _sync_virtual_marker_choices(self) -> None:
            current_segment = self.virtual_marker_segment_combo.currentText()
            current_proximal = self.virtual_marker_proximal_combo.currentText()
            current_distal = self.virtual_marker_distal_combo.currentText()

            segment_names = [group.segment_name for group in self.workflow_draft.segment_marker_groups]
            technical_segment_names = [
                group.segment_name
                for group in self.workflow_draft.segment_marker_groups
                if group.segment_type == "technical" or len(group.technical_marker_names) != 0
            ]
            if len(technical_segment_names) == 0:
                technical_segment_names = segment_names

            for combo, values, current_value in (
                (self.virtual_marker_segment_combo, segment_names, current_segment),
                (
                    self.virtual_marker_proximal_combo,
                    technical_segment_names,
                    current_proximal,
                ),
                (
                    self.virtual_marker_distal_combo,
                    technical_segment_names,
                    current_distal,
                ),
            ):
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(values)
                if current_value in values:
                    combo.setCurrentText(current_value)
                combo.blockSignals(False)
            self._sync_virtual_marker_c3d_files()
            self._sync_virtual_marker_segment_context()
            self._update_suggested_virtual_marker_name()

        def _sync_virtual_marker_c3d_files(self) -> None:
            self._sync_c3d_combo(
                self.virtual_marker_c3d_file_combo,
                "Choose a C3D folder first",
                self.virtual_marker_c3d_file_combo.currentText(),
            )

        def _sync_initial_rotation_c3d_files(self, current_source: str = "") -> None:
            self._sync_c3d_combo(
                self.settings_initial_rotation_c3d_combo,
                "Choose a C3D first",
                current_source or self.settings_initial_rotation_c3d_combo.currentText(),
            )
            self._sync_initial_rotation_source_fields()

        def _available_workflow_c3d_files(self) -> tuple[str, ...]:
            files = _c3d_file_names_from_folder(self.c3d_folder_path)
            if self.c3d_path.text().strip():
                files = tuple(dict.fromkeys(files + (Path(self.c3d_path.text().strip()).name,)))
            return files

        def _sync_c3d_combo(self, combo, placeholder: str, current_value: str = "") -> None:
            current_value = current_value.strip()
            files = self._available_workflow_c3d_files()
            selected_value = ""
            if current_value and current_value != placeholder:
                current_path = Path(current_value)
                if current_value in files:
                    selected_value = current_value
                elif current_path.name in files:
                    selected_value = current_path.name
                else:
                    selected_value = current_value
                    files = tuple(dict.fromkeys(files + (current_value,)))

            combo.blockSignals(True)
            combo.clear()
            combo.addItems(files if files else (placeholder,))
            combo.setEnabled(bool(files))
            if selected_value:
                combo.setCurrentText(selected_value)
            combo.blockSignals(False)

        def _set_c3d_combo_to_filepath(self, combo, filepath: str, placeholder: str) -> None:
            filepath = str(Path(filepath))
            selected_value = Path(filepath).name
            if not self.c3d_folder_path or Path(filepath).parent != Path(self.c3d_folder_path):
                selected_value = filepath
            self._sync_c3d_combo(combo, placeholder, selected_value)

        def _sync_virtual_marker_segment_context(self, *_args) -> None:
            segment_name = self.virtual_marker_segment_combo.currentText().strip()
            parent_name = _parent_segment_name(self.workflow_draft, segment_name)
            self.virtual_marker_parent_label.setText(parent_name or "-")
            for combo, value in (
                (self.virtual_marker_proximal_combo, parent_name),
                (self.virtual_marker_distal_combo, segment_name),
            ):
                if value and combo.findText(value) >= 0:
                    combo.blockSignals(True)
                    combo.setCurrentText(value)
                    combo.blockSignals(False)
            self._update_virtual_marker_technical_markers()
            self._update_suggested_virtual_marker_name()

        def _update_virtual_marker_technical_markers(self, *_args) -> None:
            proximal = self.virtual_marker_proximal_combo.currentText().strip()
            distal = self.virtual_marker_segment_combo.currentText().strip()
            lines = []
            for label, segment_name in (("Parent", proximal), ("Segment", distal)):
                technical_markers = _technical_markers_for_segment(self.workflow_draft, segment_name)
                if technical_markers:
                    lines.append(f"{label} {segment_name}: {', '.join(technical_markers)}")
                elif segment_name:
                    lines.append(f"{label} {segment_name}: no technical marker defined")
            self.virtual_marker_technical_markers_label.setText("\n".join(lines) if lines else "-")
            self._sync_projection_available_list()

        def _sync_projection_available_list(self) -> None:
            if not hasattr(self, "virtual_marker_projection_available_list"):
                return
            current_selection = {item.text() for item in self.virtual_marker_projection_available_list.selectedItems()}
            self.virtual_marker_projection_available_list.clear()
            for source_label in _anatomical_axis_source_labels(self.workflow_draft, self.workflow_marker_pool):
                self.virtual_marker_projection_available_list.addItem(source_label)
            for index in range(self.virtual_marker_projection_available_list.count()):
                item = self.virtual_marker_projection_available_list.item(index)
                item.setSelected(item.text() in current_selection)

        def _add_projection_source_markers(self) -> None:
            for item in self.virtual_marker_projection_available_list.selectedItems():
                if item.text().strip().startswith("[axis]"):
                    continue
                marker_name = _axis_source_name_from_list_text(item.text())
                self.virtual_marker_projected_marker_list.addItem(marker_name)
            self._sync_axis_projection_payload_from_lists()

        def _remove_projection_source_markers(self) -> None:
            self._remove_selected_axis_markers_from_list(self.virtual_marker_projected_marker_list)
            self._sync_axis_projection_payload_from_lists()

        def _add_projection_axis_source(self) -> None:
            selected_items = self.virtual_marker_projection_available_list.selectedItems()
            if len(selected_items) == 0:
                return
            self.virtual_marker_projection_axis_list.clear()
            for item in selected_items:
                text = item.text().strip()
                if text.startswith("[axis]"):
                    self.virtual_marker_projection_axis_list.addItem(f"[axis] {_axis_source_name_from_list_text(text)}")
                    break
            else:
                for item in selected_items:
                    marker_name = _axis_source_name_from_list_text(item.text())
                    if marker_name:
                        self.virtual_marker_projection_axis_list.addItem(marker_name)
            self._sync_axis_projection_payload_from_lists()

        def _remove_projection_axis_source(self) -> None:
            self._remove_selected_axis_markers_from_list(self.virtual_marker_projection_axis_list)
            self._sync_axis_projection_payload_from_lists()

        def _load_axis_projection_payload_into_lists(self, source: str, equation: str) -> None:
            self.virtual_marker_projected_marker_list.clear()
            self.virtual_marker_projection_axis_list.clear()
            if self._selected_virtual_marker_method() != "axis_projection":
                return
            for marker_name in _axis_projection_point_markers_from_payload(source):
                self.virtual_marker_projected_marker_list.addItem(marker_name)
            axis_name, axis_start, axis_end = _axis_projection_axis_from_payload(equation)
            if axis_name:
                self.virtual_marker_projection_axis_list.addItem(f"[axis] {axis_name}")
            else:
                for marker_name in axis_start + axis_end:
                    self.virtual_marker_projection_axis_list.addItem(marker_name)

        def _sync_axis_projection_payload_from_lists(self) -> None:
            point_markers = _list_widget_texts(self.virtual_marker_projected_marker_list)
            axis_items = _list_widget_texts(self.virtual_marker_projection_axis_list)
            self.virtual_marker_source_edit.blockSignals(True)
            self.virtual_marker_equation_edit.blockSignals(True)
            self.virtual_marker_source_edit.setText(f"point={','.join(point_markers)}" if point_markers else "")
            axis_name = ""
            axis_markers = []
            for item in axis_items:
                if item.startswith("[axis]"):
                    axis_name = item.removeprefix("[axis]").strip()
                    break
                axis_markers.append(item)
            if axis_name:
                self.virtual_marker_equation_edit.setText(f"axis={axis_name}")
            elif len(axis_markers) >= 2:
                self.virtual_marker_equation_edit.setText(f"axis_start={axis_markers[0]}; axis_end={axis_markers[1]}")
            else:
                self.virtual_marker_equation_edit.setText("")
            self.virtual_marker_equation_edit.blockSignals(False)
            self.virtual_marker_source_edit.blockSignals(False)
            self._update_virtual_marker_preview()

        def _selected_virtual_marker_c3d_file(self) -> str:
            return self._selected_c3d_file_from_combo(self.virtual_marker_c3d_file_combo, "Choose a C3D folder first")

        def _selected_initial_rotation_c3d_file(self) -> str:
            return self._selected_c3d_file_from_combo(self.settings_initial_rotation_c3d_combo, "Choose a C3D first")

        def _selected_c3d_file_from_combo(self, combo, placeholder: str) -> str:
            filename = combo.currentText().strip()
            if not filename or filename == placeholder:
                return ""
            filepath = Path(filename)
            if filepath.is_absolute() or not self.c3d_folder_path:
                return str(filepath)
            return str(Path(self.c3d_folder_path) / filename)

        def _c3d_data_from_path(self, filepath: str):
            if filepath == "":
                return None
            if filepath not in self._c3d_data_cache:
                data = C3dData(filepath)
                if self.strip_participant_prefix_checkbox.isChecked():
                    _strip_participant_prefix_from_c3d_data(data)
                self._c3d_data_cache[filepath] = data
            return self._c3d_data_cache[filepath]

        def _selected_virtual_marker_c3d_data(self):
            filepath = self._selected_virtual_marker_c3d_file()
            try:
                return self._c3d_data_from_path(filepath) if filepath else self.c3d_data
            except Exception:
                return self.c3d_data

        def _selected_virtual_marker_preview_c3d_data(self):
            if self._selected_virtual_marker_method() == "rab2002_shoulder":
                return self.c3d_data
            if self._virtual_marker_preview_mode == "functional":
                return self._selected_virtual_marker_c3d_data()
            return self.c3d_data

        def _selected_virtual_marker_preview_label(self) -> str:
            if self._selected_virtual_marker_method() == "rab2002_shoulder":
                static_file = Path(self.c3d_path.text().strip()).name if self.c3d_path.text().strip() else ""
                return f"static/main C3D ({static_file})" if static_file else "static/main C3D"
            if self._virtual_marker_preview_mode == "functional":
                filename = self.virtual_marker_c3d_file_combo.currentText().strip()
                return f"functional C3D ({filename})" if filename else "functional C3D"
            static_file = Path(self.c3d_path.text().strip()).name if self.c3d_path.text().strip() else ""
            return f"static/main C3D ({static_file})" if static_file else "static/main C3D"

        def _sync_virtual_marker_frame_slider(self, c3d_data) -> None:
            frame_count = 0 if c3d_data is None else c3d_data.nb_frames
            current_frame = min(self.virtual_marker_frame_slider.value(), max(frame_count - 1, 0))
            self.virtual_marker_frame_slider.blockSignals(True)
            self.virtual_marker_frame_slider.setEnabled(frame_count > 1)
            self.virtual_marker_frame_slider.setMinimum(0)
            self.virtual_marker_frame_slider.setMaximum(max(frame_count - 1, 0))
            self.virtual_marker_frame_slider.setValue(current_frame)
            self.virtual_marker_frame_slider.blockSignals(False)
            self._sync_workflow_frame_play_button(self.virtual_marker_frame_slider)
            if frame_count == 0:
                self.virtual_marker_frame_label.setText("Frame 0/0")
            else:
                self.virtual_marker_frame_label.setText(f"Frame {current_frame + 1}/{frame_count}")
            self.functional_frame_range_bar.set_frame_count(frame_count)
            self._sync_functional_frame_range_bar_enabled()

        def _mirror_initial_rotation_c3d_source(self, *_args) -> None:
            self._sync_initial_rotation_source_fields()

        def _sync_initial_rotation_source_fields(self, *args) -> None:
            method = self.settings_initial_rotation_method_combo.currentText()
            is_anatomical_c3d = method == "anatomical_c3d"
            is_matrix = method == "matrix"
            self.settings_initial_rotation_source_edit.setEnabled(is_matrix)
            self.settings_initial_rotation_c3d_combo.setEnabled(
                is_anatomical_c3d and self._has_initial_rotation_c3d_files()
            )
            self.browse_initial_rotation_c3d_button.setEnabled(is_anatomical_c3d)
            if is_anatomical_c3d:
                if args:
                    self._select_static_initial_rotation_c3d()
                elif not self._selected_initial_rotation_c3d_file():
                    self._select_static_initial_rotation_c3d()
                self.settings_initial_rotation_source_edit.setText("")
            elif method == "identity":
                self.settings_initial_rotation_source_edit.setText("")
            self._sync_segment_settings_validation_style()

        def _has_initial_rotation_c3d_files(self) -> bool:
            return self.settings_initial_rotation_c3d_combo.currentText().strip() not in {"", "Choose a C3D first"}

        def _select_static_initial_rotation_c3d(self) -> None:
            if not self.c3d_path.text().strip():
                return
            static_name = Path(self.c3d_path.text().strip()).name
            if self.settings_initial_rotation_c3d_combo.findText(static_name) < 0:
                return
            self.settings_initial_rotation_c3d_combo.blockSignals(True)
            self.settings_initial_rotation_c3d_combo.setCurrentText(static_name)
            self.settings_initial_rotation_c3d_combo.blockSignals(False)

        def _technical_marker_source_from_selected_segments(self) -> str:
            marker_names = []
            for segment_name in (
                self.virtual_marker_proximal_combo.currentText().strip(),
                self.virtual_marker_segment_combo.currentText().strip(),
            ):
                marker_names.extend(_technical_markers_for_segment(self.workflow_draft, segment_name))
            return ",".join(dict.fromkeys(marker_names))

        def _selected_virtual_marker_method(self) -> str:
            method = self.virtual_marker_method_combo.currentText().strip()
            method = _virtual_marker_method_from_display(method)
            if method != "predictive":
                return method
            return _predictive_virtual_marker_method_from_label(
                self.virtual_marker_predictive_method_combo.currentText().strip()
            )

        def _set_virtual_marker_method(self, method: str) -> None:
            self.virtual_marker_method_combo.setCurrentText(_virtual_marker_method_display(method))

        def _sync_virtual_marker_method_fields(self, _method: str | None = None) -> None:
            method = self._selected_virtual_marker_method()
            is_predictive = (
                _virtual_marker_method_from_display(self.virtual_marker_method_combo.currentText().strip())
                == "predictive"
            )
            self.virtual_marker_predictive_method_combo.setEnabled(is_predictive)
            has_c3d_file = self.virtual_marker_c3d_file_combo.currentText().strip() != "Choose a C3D folder first"
            self.virtual_marker_c3d_file_combo.setEnabled(
                method not in {"marker_mean", "axis_projection", "rab2002_shoulder"} and has_c3d_file
            )
            self.virtual_marker_proximal_combo.setEnabled(
                method in {"score", "sara", "sara_direction"} or is_predictive
            )
            self.virtual_marker_distal_combo.setEnabled(method in {"score", "sara", "sara_direction"} or is_predictive)
            is_axis_projection = method == "axis_projection"
            self.virtual_marker_projection_group.setEnabled(is_axis_projection)
            for widget in (
                self.virtual_marker_projection_available_list,
                self.virtual_marker_projected_marker_list,
                self.virtual_marker_projection_axis_list,
                self.add_projected_marker_button,
                self.remove_projected_marker_button,
                self.add_projection_axis_button,
                self.remove_projection_axis_button,
            ):
                widget.setEnabled(is_axis_projection)
            self._sync_projection_available_list()
            if is_axis_projection:
                if self.virtual_marker_source_edit.text().strip() == "":
                    self.virtual_marker_source_edit.setPlaceholderText("point=LKNE,LKNEM")
                if self.virtual_marker_equation_edit.text().strip() == "":
                    self.virtual_marker_equation_edit.setPlaceholderText(
                        "axis=Axis_LKnee_SARA or axis_start=LKNE,LKNEM; axis_end=LANK,LANKM"
                    )
            is_rab2002 = method == "rab2002_shoulder"
            self.virtual_marker_source_label.setVisible(is_rab2002)
            self.virtual_marker_source_edit.setVisible(is_rab2002)
            self.virtual_marker_source_edit.setEnabled(is_rab2002)
            if is_rab2002:
                if self.virtual_marker_source_edit.text().strip() == "":
                    self.virtual_marker_source_edit.setText(
                        _default_rab2002_payload(
                            self.virtual_marker_name_edit.text().strip(),
                            self.virtual_marker_segment_combo.currentText().strip(),
                        )
                    )
                self.virtual_marker_source_label.setText("Rab markers")
                self.virtual_marker_source_edit.setPlaceholderText("point=RCAJ; mid=RHME,RHLE; fraction=0.17")
            hints = {
                "pointing": (
                    "Pointing is not implemented yet: the workflow still needs a dedicated pointing object "
                    "to store the pointed target."
                ),
                "score": "Choose the functional C3D plus proximal and distal technical segments. SCoRE estimates a joint center.",
                "sara": (
                    "Choose the functional C3D plus the segment and parent technical markers. SARA estimates an "
                    "axis direction; its displayed origin comes from the reference point projected onto that axis."
                ),
                "sara_direction": (
                    "Choose the functional C3D plus the segment and parent technical markers. SARA estimates an "
                    "axis direction; its displayed origin comes from the reference point projected onto that axis."
                ),
                "marker_mean": "Average the technical markers shown for the proximal and distal segments.",
                "axis_projection": (
                    "Project one marker, or a marker mean, onto a marker-defined axis or a SARA functional axis."
                ),
                "hara2016_hip": "Predict a hip CoR from the selected C3D and segment pair.",
                "harrington2007_hip": "Predict a hip CoR from the selected C3D and segment pair.",
                "sobral2025_shoulder": "Predict a shoulder CoR from the selected C3D and segment pair.",
                "rab2002_shoulder": (
                    "Static C3D: draw CAJ to the midpoint of HME/HLE, then place GJC at 17% of that vector "
                    "from CAJ. Edit point=... and mid=...,... to choose the three markers."
                ),
            }
            self.virtual_marker_info_label.setText(hints.get(method, ""))
            self._update_suggested_virtual_marker_name()
            self._update_virtual_marker_preview()

        def _update_virtual_marker_info_label(self, marker) -> None:
            if marker is None:
                self.virtual_marker_info_label.setText(
                    "Select a virtual marker to inspect its method, segment, C3D source, and segment pair."
                )
                return
            source = marker.source if marker.source else "-"
            proximal, distal = _score_segments_from_payload(marker.equation)
            segment_pair = f"{proximal or '-'} -> {distal or '-'}"
            if marker.method == "axis_projection":
                point_markers = _axis_projection_point_markers_from_payload(marker.source)
                axis_reference, axis_start, axis_end = _axis_projection_axis_from_payload(marker.equation)
                axis_text = axis_reference or f"{','.join(axis_start) or '-'} -> {','.join(axis_end) or '-'}"
                self.virtual_marker_info_label.setText(
                    f"Name: {marker.name}\nSegment: {marker.segment_name}\n"
                    f"Method: {_virtual_marker_method_display(marker.method)}\n"
                    f"Projected markers: {','.join(point_markers) or '-'}\nProjection axis: {axis_text}"
                )
                return
            if marker.method == "rab2002_shoulder":
                point_marker, mid_markers, fraction = _rab2002_markers_from_payload(marker.source, marker.name)
                self.virtual_marker_info_label.setText(
                    f"Name: {marker.name}\nSegment: {marker.segment_name}\n"
                    "Method: Rab 2002 shoulder\n"
                    f"Static markers: CAJ/acromion={point_marker or '-'}; epicondyles={','.join(mid_markers) or '-'}\n"
                    f"Geometry: GJC = CAJ + {fraction:g} * (mid(epicondyles) - CAJ)\n"
                    "The preview uses the static/main C3D because this predictive marker is anatomical, not functional."
                )
                return
            self.virtual_marker_info_label.setText(
                f"Name: {marker.name}\nSegment: {marker.segment_name}\n"
                f"Method: {_virtual_marker_method_display(marker.method)}\n"
                f"C3D/source: {source}\nSegment pair: {segment_pair}\n"
                "Preview coordinates are computed from the selected C3D; they are not stored in the main C3D "
                "until the virtual feature is generated."
            )

        def _update_virtual_axis_info_label(self, axis) -> None:
            source = axis.source if axis.source else "-"
            origin = ",".join(axis.origin_markers) if len(axis.origin_markers) != 0 else "-"
            start = ",".join(axis.start_markers) if len(axis.start_markers) != 0 else "-"
            end = ",".join(axis.end_markers) if len(axis.end_markers) != 0 else "-"
            self.virtual_marker_info_label.setText(
                f"Name: {axis.name}\nSegment: {axis.segment_name}\nMethod: {axis.method}\n"
                f"Origin markers: {origin}\nAxis fallback/expected orientation: {start} -> {end}\n"
                f"Functional source: {source}\n"
                "SARA returns a functional axis direction. The preview origin is the projected reference point used "
                "to place that axis in the displayed C3D."
            )

        def _update_virtual_marker_preview(self, *_args) -> None:
            preview_c3d_data = self._selected_virtual_marker_preview_c3d_data()
            solution_c3d_data = self._selected_virtual_marker_c3d_data()
            preview_source_label = self._selected_virtual_marker_preview_label()
            functional_preview_active = (
                self.workflow_tabs.tabText(self.workflow_tabs.currentIndex()) == "Functional reconstruction"
            )
            fast_playback_preview = (
                functional_preview_active and self._workflow_playback_slider is self.virtual_marker_frame_slider
            )
            selected_axis = self._selected_virtual_axis()
            if functional_preview_active:
                selected_axis = self._selected_functional_reconstruction_axis()
                solution_c3d_data, source_label = self._selected_functional_reconstruction_c3d_data(selected_axis)
                if solution_c3d_data is not None:
                    preview_c3d_data = solution_c3d_data
                    preview_source_label = f"functional C3D ({source_label})"
            self._sync_virtual_marker_frame_slider(preview_c3d_data)
            selected_name = self.virtual_marker_name_edit.text().strip()
            selected_method = self._selected_virtual_marker_method()
            proximal_segment_name = self.virtual_marker_proximal_combo.currentText().strip()
            distal_segment_name = self.virtual_marker_segment_combo.currentText().strip()
            if selected_axis is not None:
                selected_name = selected_axis.name
                selected_method = selected_axis.method
                proximal_segment_name, distal_segment_name = self._functional_reconstruction_axis_pair(selected_axis)
            virtual_feature_c3d_data = {} if fast_playback_preview else self._virtual_feature_c3d_preview_data_map()
            self.virtual_marker_preview.set_context(
                preview_c3d_data,
                solution_c3d_data,
                preview_source_label,
                self.workflow_draft.segment_marker_groups,
                self.workflow_draft.axes,
                self.workflow_draft.virtual_markers,
                virtual_feature_c3d_data,
                selected_name,
                selected_method,
                proximal_segment_name,
                distal_segment_name,
                self.virtual_marker_frame_slider.value(),
                self.virtual_marker_whole_body_preview_checkbox.isChecked(),
                self.use_diverse_functional_frames_checkbox.isChecked(),
                self._manual_functional_frame_indices(),
                functional_preview_active,
                fast_playback_preview,
            )

        def _update_diverse_functional_frame_calculations(self, *_args) -> None:
            progress_dialog, progress_callback = self._functional_frame_calculation_progress_reporter()
            try:
                if self.use_manual_functional_frames_checkbox.isChecked():
                    mode = "manual range"
                elif self.use_diverse_functional_frames_checkbox.isChecked():
                    mode = "diverse frames"
                else:
                    mode = "all valid frames"
                progress_callback(f"Functional frame selection: {mode}; updating virtual marker preview...")
                self._update_virtual_marker_preview()
                progress_callback("Updating anatomical segment axes preview...")
                self._update_segment_axis_preview()
                progress_callback("Computing functional trial quality metrics...")
                self._update_generation_log()
            finally:
                progress_dialog.close()

        def _set_manual_functional_frames_enabled(self, *_args) -> None:
            if self.use_manual_functional_frames_checkbox.isChecked():
                self.use_diverse_functional_frames_checkbox.blockSignals(True)
                self.use_diverse_functional_frames_checkbox.setChecked(False)
                self.use_diverse_functional_frames_checkbox.blockSignals(False)
            self._sync_functional_frame_range_bar_enabled()
            self._update_diverse_functional_frame_calculations()

        def _set_diverse_functional_frames_enabled(self, *_args) -> None:
            if self.use_diverse_functional_frames_checkbox.isChecked():
                self.use_manual_functional_frames_checkbox.blockSignals(True)
                self.use_manual_functional_frames_checkbox.setChecked(False)
                self.use_manual_functional_frames_checkbox.blockSignals(False)
            self._sync_functional_frame_range_bar_enabled()
            self._update_diverse_functional_frame_calculations()

        def _sync_functional_frame_range_bar_enabled(self) -> None:
            frame_count = self.functional_frame_range_bar.frame_count
            self.functional_frame_range_bar.setEnabled(frame_count > 1)

        def _set_virtual_marker_frame_from_range_drag(self, frame_index: int) -> None:
            self.virtual_marker_frame_slider.setValue(frame_index)
            QApplication.processEvents()

        def _update_manual_functional_frame_selection(self) -> None:
            if not self.use_manual_functional_frames_checkbox.isChecked():
                return
            self._update_virtual_marker_preview()
            self._update_generation_log()

        def _manual_functional_frame_indices(self) -> tuple[int, ...]:
            if not self.use_manual_functional_frames_checkbox.isChecked():
                return ()
            return self.functional_frame_range_bar.selected_indices()

        def _current_functional_frame_selection_options(
            self,
        ) -> FunctionalFrameSelectionOptions:
            return _functional_frame_selection_options(
                self.use_diverse_functional_frames_checkbox.isChecked(),
                self._manual_functional_frame_indices(),
            )

        def _functional_trial_quality_lines(self) -> tuple[str, ...]:
            lines = ["", "Functional trial quality:"]
            feature_count = 0
            features = tuple(self.workflow_draft.virtual_markers) + tuple(
                axis for axis in self.workflow_draft.axes if _is_virtual_feature_axis(axis)
            )
            for feature in features:
                method = getattr(feature, "method", "")
                if method not in {"score", "sara", "sara_direction"}:
                    continue
                source = getattr(feature, "source", "")
                payload = _key_value_payload(source)
                parent_marker_names = _split_marker_names(payload.get("parent markers", ""))
                child_marker_names = _split_marker_names(payload.get("child markers", ""))
                if len(parent_marker_names) == 0:
                    parent_marker_names = _technical_markers_for_segment(
                        self.workflow_draft,
                        _parent_segment_name(self.workflow_draft, feature.segment_name),
                    )
                if len(child_marker_names) == 0:
                    child_marker_names = _technical_markers_for_segment(self.workflow_draft, feature.segment_name)
                functional_data, source_label = self._functional_feature_c3d_data_for_source(source)
                feature_count += 1
                if functional_data is None:
                    lines.append(f"- {feature.name} ({method}): missing functional C3D source.")
                    continue
                marker_names = tuple(dict.fromkeys(parent_marker_names + child_marker_names))
                missing_markers = tuple(marker for marker in marker_names if marker not in functional_data.marker_names)
                if missing_markers:
                    lines.append(
                        f"- {feature.name} ({method}, {source_label}): missing markers "
                        f"{', '.join(missing_markers)}."
                    )
                    continue
                try:
                    from ..components.generic.rigidbody.segment_coordinate_system import (
                        SegmentCoordinateSystemUtils,
                    )

                    parent_functional_marker_data = functional_data.get_partial_dict_data(parent_marker_names)
                    child_functional_marker_data = functional_data.get_partial_dict_data(child_marker_names)
                    rt_parent_func = SegmentCoordinateSystemUtils.rigidify(parent_functional_marker_data)
                    rt_child_func = SegmentCoordinateSystemUtils.rigidify(child_functional_marker_data)
                    rt_parent_used, rt_child_used, report = prepare_functional_rt_pair(
                        rt_parent_func,
                        rt_child_func,
                        self._current_functional_frame_selection_options(),
                    )
                    algorithm_quality = _functional_algorithm_quality_text(method, rt_parent_used, rt_child_used)
                    static_axis_quality = _sara_static_axis_quality_text(
                        method=method,
                        feature=feature,
                        payload=payload,
                        static_data=self.c3d_data,
                        functional_data=functional_data,
                        parent_marker_names=parent_marker_names,
                        child_marker_names=child_marker_names,
                        use_diverse_functional_frames=self.use_diverse_functional_frames_checkbox.isChecked(),
                        manual_functional_frame_indices=self._manual_functional_frame_indices(),
                    )
                except Exception as error:
                    lines.append(f"- {feature.name} ({method}, {source_label}): unable to compute quality ({error}).")
                    continue
                complete_frames = _complete_marker_frame_count(functional_data, marker_names)
                quality_details = algorithm_quality
                if static_axis_quality:
                    quality_details = f"{quality_details}; {static_axis_quality}"
                lines.append(
                    f"- {feature.name} ({method}, {source_label}): {complete_frames} marker-complete frames; "
                    f"{format_functional_frame_report(report)}; {quality_details}."
                )
            if feature_count == 0:
                lines.append("- No functional SCoRE/SARA feature in this draft.")
            return tuple(lines)

        def _show_functional_residual_diagnostics(self) -> None:
            """
            Show residual boxplots for the selected SCoRE/SARA virtual feature.
            """
            feature = self._selected_functional_residual_feature()
            if feature is None:
                QMessageBox.information(
                    self,
                    "SCoRE/SARA residual diagnostics",
                    "Select one SCoRE CoR or SARA AoR in the virtual marker/axis list first.",
                )
                self._sync_functional_residual_button()
                return
            try:
                summary_lines, boxplots = self._functional_feature_residual_plot_data(feature)
            except Exception as error:
                QMessageBox.critical(self, "Unable to compute residual diagnostics", str(error))
                return

            dialog = QDialog(self)
            dialog.setWindowTitle(f"SCoRE/SARA residual diagnostics - {feature.name}")
            layout = QVBoxLayout(dialog)
            _configure_panel_layout(layout)
            summary_label = QLabel("\n".join(summary_lines))
            summary_label.setWordWrap(True)
            summary_label.setObjectName("MutedInfoLabel")
            layout.addWidget(summary_label)
            boxplot_widget = FunctionalResidualBoxplotWidget()
            boxplot_widget.set_boxplots(tuple(boxplots))
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(boxplot_widget)
            layout.addWidget(scroll, 1)
            buttons = QDialogButtonBox(_dialog_button("Ok"))
            buttons.accepted.connect(dialog.accept)
            layout.addWidget(buttons)
            dialog.resize(920, 680)
            _exec_dialog(dialog)

        def _selected_functional_residual_feature(self):
            marker = self._selected_virtual_marker()
            if marker is not None and marker.method in {
                "score",
                "sara",
                "sara_direction",
            }:
                return marker
            axis = self._selected_virtual_axis()
            if axis is not None and _is_sara_direction_method(axis.method):
                return axis
            return None

        def _sync_functional_residual_button(self) -> None:
            feature = self._selected_functional_residual_feature()
            enabled = feature is not None
            self.explore_functional_residuals_button.setEnabled(enabled)
            if enabled:
                self.explore_functional_residuals_button.setToolTip(f"Show residual boxplots for {feature.name} only.")
            else:
                self.explore_functional_residuals_button.setToolTip(
                    "Select one SCoRE CoR or SARA AoR in the virtual marker/axis list."
                )

        def _functional_residual_diagnostics_lines(self) -> tuple[str, ...]:
            lines = [
                "SCoRE/SARA residual diagnostics",
                "",
                "Distances are reported in mm. SARA residual angles are reported in degrees.",
                "Marker boxplots use the selected functional-frame mode before SCoRE/SARA estimation.",
            ]
            feature_count = 0
            features = tuple(self.workflow_draft.virtual_markers) + tuple(
                axis for axis in self.workflow_draft.axes if _is_virtual_feature_axis(axis)
            )
            for feature in features:
                method = getattr(feature, "method", "")
                if method not in {"score", "sara", "sara_direction"}:
                    continue
                feature_count += 1
                lines.extend(("", f"{feature.name} ({method})"))
                try:
                    lines.extend(self._functional_feature_residual_lines(feature))
                except Exception as error:
                    lines.append(f"  unavailable: {error}")
            if feature_count == 0:
                lines.append("")
                lines.append("No SCoRE/SARA feature in this draft.")
            return tuple(lines)

        def _functional_feature_residual_plot_data(
            self, feature
        ) -> tuple[tuple[str, ...], tuple[dict[str, object], ...]]:
            method = getattr(feature, "method", "")
            source = getattr(feature, "source", "")
            payload = _key_value_payload(source)
            parent_marker_names = _split_marker_names(payload.get("parent markers", ""))
            child_marker_names = _split_marker_names(payload.get("child markers", ""))
            if len(parent_marker_names) == 0:
                parent_marker_names = _technical_markers_for_segment(
                    self.workflow_draft,
                    _parent_segment_name(self.workflow_draft, feature.segment_name),
                )
            if len(child_marker_names) == 0:
                child_marker_names = _technical_markers_for_segment(self.workflow_draft, feature.segment_name)
            functional_data, source_label = self._functional_feature_c3d_data_for_source(source)
            if functional_data is None:
                raise RuntimeError(f"Missing functional C3D source ({source_label}).")
            marker_names = tuple(dict.fromkeys(parent_marker_names + child_marker_names))
            missing_markers = tuple(marker for marker in marker_names if marker not in functional_data.marker_names)
            if missing_markers:
                raise RuntimeError(f"Missing markers: {', '.join(missing_markers)}.")

            from ..components.generic.rigidbody.segment_coordinate_system import (
                SegmentCoordinateSystemUtils,
            )

            parent_data = functional_data.get_partial_dict_data(parent_marker_names)
            child_data = functional_data.get_partial_dict_data(child_marker_names)
            rt_parent = SegmentCoordinateSystemUtils.rigidify(parent_data)
            rt_child = SegmentCoordinateSystemUtils.rigidify(child_data)
            rt_parent, rt_child, report = prepare_functional_rt_pair(
                rt_parent,
                rt_child,
                self._current_functional_frame_selection_options(),
            )
            selected_indices = tuple(report.selected_indices)
            summary = [
                f"{feature.name} ({method})",
                f"C3D: {source_label}",
                format_functional_frame_report(report),
            ]
            boxplots = []
            if method == "score":
                from ..model_modifiers.joint_center_tool import Score

                _, cor_parent_local, cor_child_local, _, _ = Score.perform_algorithm(
                    rt_parent,
                    rt_child,
                    recursive_outlier_removal=False,
                )
                residuals = _score_residuals(rt_parent, rt_child, cor_parent_local, cor_child_local)
                boxplots.append(_residual_boxplot_data("CoR parent-child residual", residuals * 1000.0, "mm"))
                parent_cor = _rt_point_series(rt_parent, cor_parent_local)
                child_cor = _rt_point_series(rt_child, cor_child_local)
                for marker_name in parent_marker_names:
                    marker_points = _selected_marker_xyz(functional_data, marker_name, selected_indices)
                    boxplots.append(
                        _residual_boxplot_data(
                            f"{marker_name} distance to parent CoR",
                            np.linalg.norm(marker_points - parent_cor, axis=0) * 1000.0,
                            "mm",
                        )
                    )
                for marker_name in child_marker_names:
                    marker_points = _selected_marker_xyz(functional_data, marker_name, selected_indices)
                    boxplots.append(
                        _residual_boxplot_data(
                            f"{marker_name} distance to child CoR",
                            np.linalg.norm(marker_points - child_cor, axis=0) * 1000.0,
                            "mm",
                        )
                    )
                return tuple(summary), tuple(boxplots)

            from ..model_modifiers.joint_center_tool import Sara, get_svd

            (
                _,
                aor_parent_local,
                aor_child_local,
                _,
                cor_parent_local,
                cor_child_local,
                _,
                _,
            ) = Sara.perform_algorithm(rt_parent, rt_child, recursive_outlier_removal=False)
            residuals = _sara_residual_angles(rt_parent, rt_child, aor_parent_local, aor_child_local)
            _, singular_values, _, _ = get_svd(rt_parent, rt_child)
            summary.append(
                "SVD singular values: "
                + ", ".join(f"{float(value):.4g}" for value in np.asarray(singular_values, dtype=float))
            )
            boxplots.append(_residual_boxplot_data("AoR parent-child angular residual", residuals, "deg"))
            parent_start = _rt_point_series(rt_parent, cor_parent_local)
            parent_direction = _rt_vector_series(rt_parent, aor_parent_local)
            child_start = _rt_point_series(rt_child, cor_child_local)
            child_direction = _rt_vector_series(rt_child, aor_child_local)
            for marker_name in parent_marker_names:
                marker_points = _selected_marker_xyz(functional_data, marker_name, selected_indices)
                boxplots.append(
                    _residual_boxplot_data(
                        f"{marker_name} distance to parent AoR",
                        _point_to_line_distances(marker_points, parent_start, parent_direction) * 1000.0,
                        "mm",
                    )
                )
            for marker_name in child_marker_names:
                marker_points = _selected_marker_xyz(functional_data, marker_name, selected_indices)
                boxplots.append(
                    _residual_boxplot_data(
                        f"{marker_name} distance to child AoR",
                        _point_to_line_distances(marker_points, child_start, child_direction) * 1000.0,
                        "mm",
                    )
                )
            return tuple(summary), tuple(boxplots)

        def _functional_feature_residual_lines(self, feature) -> tuple[str, ...]:
            method = getattr(feature, "method", "")
            source = getattr(feature, "source", "")
            payload = _key_value_payload(source)
            parent_marker_names = _split_marker_names(payload.get("parent markers", ""))
            child_marker_names = _split_marker_names(payload.get("child markers", ""))
            if len(parent_marker_names) == 0:
                parent_marker_names = _technical_markers_for_segment(
                    self.workflow_draft,
                    _parent_segment_name(self.workflow_draft, feature.segment_name),
                )
            if len(child_marker_names) == 0:
                child_marker_names = _technical_markers_for_segment(self.workflow_draft, feature.segment_name)
            functional_data, source_label = self._functional_feature_c3d_data_for_source(source)
            if functional_data is None:
                return (f"  missing functional C3D source ({source_label}).",)
            marker_names = tuple(dict.fromkeys(parent_marker_names + child_marker_names))
            missing_markers = tuple(marker for marker in marker_names if marker not in functional_data.marker_names)
            if missing_markers:
                return (f"  missing markers: {', '.join(missing_markers)}.",)
            from ..components.generic.rigidbody.segment_coordinate_system import (
                SegmentCoordinateSystemUtils,
            )

            parent_data = functional_data.get_partial_dict_data(parent_marker_names)
            child_data = functional_data.get_partial_dict_data(child_marker_names)
            rt_parent = SegmentCoordinateSystemUtils.rigidify(parent_data)
            rt_child = SegmentCoordinateSystemUtils.rigidify(child_data)
            rt_parent, rt_child, report = prepare_functional_rt_pair(
                rt_parent,
                rt_child,
                self._current_functional_frame_selection_options(),
            )
            selected_indices = tuple(report.selected_indices)
            lines = [
                f"  C3D: {source_label}",
                f"  {format_functional_frame_report(report)}",
            ]
            if method == "score":
                from ..model_modifiers.joint_center_tool import Score

                _, cor_parent_local, cor_child_local, _, _ = Score.perform_algorithm(
                    rt_parent,
                    rt_child,
                    recursive_outlier_removal=False,
                )
                residuals = _score_residuals(rt_parent, rt_child, cor_parent_local, cor_child_local)
                lines.extend(_histogram_lines("  CoR parent-child residual", residuals * 1000.0, "mm"))
                parent_cor = _rt_point_series(rt_parent, cor_parent_local)
                child_cor = _rt_point_series(rt_child, cor_child_local)
                for marker_name in parent_marker_names:
                    marker_points = _selected_marker_xyz(functional_data, marker_name, selected_indices)
                    lines.extend(
                        _histogram_lines(
                            f"  {marker_name} distance to parent CoR",
                            np.linalg.norm(marker_points - parent_cor, axis=0) * 1000.0,
                            "mm",
                        )
                    )
                for marker_name in child_marker_names:
                    marker_points = _selected_marker_xyz(functional_data, marker_name, selected_indices)
                    lines.extend(
                        _histogram_lines(
                            f"  {marker_name} distance to child CoR",
                            np.linalg.norm(marker_points - child_cor, axis=0) * 1000.0,
                            "mm",
                        )
                    )
                return tuple(lines)

            from ..model_modifiers.joint_center_tool import Sara, get_svd

            (
                _,
                aor_parent_local,
                aor_child_local,
                _,
                cor_parent_local,
                cor_child_local,
                _,
                _,
            ) = Sara.perform_algorithm(rt_parent, rt_child, recursive_outlier_removal=False)
            residuals = _sara_residual_angles(rt_parent, rt_child, aor_parent_local, aor_child_local)
            _, singular_values, _, _ = get_svd(rt_parent, rt_child)
            lines.extend(_histogram_lines("  AoR parent-child angular residual", residuals, "deg"))
            lines.append(
                "  SVD singular values: "
                + ", ".join(f"{float(value):.4g}" for value in np.asarray(singular_values, dtype=float))
            )
            parent_start = _rt_point_series(rt_parent, cor_parent_local)
            parent_direction = _rt_vector_series(rt_parent, aor_parent_local)
            child_start = _rt_point_series(rt_child, cor_child_local)
            child_direction = _rt_vector_series(rt_child, aor_child_local)
            for marker_name in parent_marker_names:
                marker_points = _selected_marker_xyz(functional_data, marker_name, selected_indices)
                lines.extend(
                    _histogram_lines(
                        f"  {marker_name} distance to parent AoR",
                        _point_to_line_distances(marker_points, parent_start, parent_direction) * 1000.0,
                        "mm",
                    )
                )
            for marker_name in child_marker_names:
                marker_points = _selected_marker_xyz(functional_data, marker_name, selected_indices)
                lines.extend(
                    _histogram_lines(
                        f"  {marker_name} distance to child AoR",
                        _point_to_line_distances(marker_points, child_start, child_direction) * 1000.0,
                        "mm",
                    )
                )
            return tuple(lines)

        def _functional_feature_c3d_data_for_source(self, source: str):
            c3d_name = _c3d_source_name_from_virtual_feature_source(source)
            trial_name = _trial_name_from_virtual_feature_source(source)
            filepath = ""
            if c3d_name:
                filepath = str(Path(self.c3d_folder_path) / c3d_name) if self.c3d_folder_path else c3d_name
            elif trial_name:
                filepath = _assigned_c3d_source_for_role(self.workflow_draft, trial_name)
            if not filepath:
                return None, trial_name or c3d_name or "unassigned"
            try:
                data = self._c3d_data_from_path(filepath)
            except Exception:
                return None, Path(filepath).name
            return data, Path(filepath).name

        def _sync_functional_reconstruction_choices(self) -> None:
            """
            Refresh the SARA AoR choices available in the reconstruction diagnostic tab.
            """
            current_name = self.functional_reconstruction_feature_combo.currentData() or ""
            self.functional_reconstruction_feature_combo.blockSignals(True)
            self.functional_reconstruction_feature_combo.clear()
            for axis in self.workflow_draft.axes:
                if not (_is_virtual_feature_axis(axis) and _is_sara_direction_method(axis.method)):
                    continue
                parent_name, child_name = self._functional_reconstruction_axis_pair(axis)
                label = f"{axis.name} | {parent_name or '-'} -> {axis.segment_name}"
                if child_name != axis.segment_name:
                    label = f"{axis.name} | {parent_name or '-'} -> {child_name or '-'}"
                self.functional_reconstruction_feature_combo.addItem(label, axis.name)
            if self.functional_reconstruction_feature_combo.count() == 0:
                self.functional_reconstruction_feature_combo.addItem("No SARA AoR in this draft", "")
            elif current_name:
                for index in range(self.functional_reconstruction_feature_combo.count()):
                    if self.functional_reconstruction_feature_combo.itemData(index) == current_name:
                        self.functional_reconstruction_feature_combo.setCurrentIndex(index)
                        break
            self.functional_reconstruction_feature_combo.blockSignals(False)
            self.run_functional_reconstruction_button.setEnabled(
                self.functional_reconstruction_feature_combo.currentData() != ""
            )
            self._sync_functional_reconstruction_trial_choices()

        def _sync_functional_reconstruction_trial_choices(self) -> None:
            """
            Refresh the functional C3D choices for the selected reconstruction AoR.
            """
            if not hasattr(self, "functional_reconstruction_trial_combo"):
                return
            current_role = self.functional_reconstruction_trial_combo.currentData() or ""
            axis = self._selected_functional_reconstruction_axis()
            default_role = _trial_name_from_virtual_feature_source(axis.source) if axis is not None else ""
            assignments = tuple(
                assignment
                for assignment in self.workflow_draft.file_assignments
                if assignment.role != "main" and assignment.source_path
            )
            self.functional_reconstruction_trial_combo.blockSignals(True)
            self.functional_reconstruction_trial_combo.clear()
            for assignment in assignments:
                label = f"{assignment.role}: {Path(assignment.source_path).name}"
                self.functional_reconstruction_trial_combo.addItem(label, assignment.role)
            if self.functional_reconstruction_trial_combo.count() == 0:
                self.functional_reconstruction_trial_combo.addItem("No functional C3D assigned", "")
            else:
                wanted_role = default_role or current_role
                if wanted_role:
                    for index in range(self.functional_reconstruction_trial_combo.count()):
                        if self.functional_reconstruction_trial_combo.itemData(index) == wanted_role:
                            self.functional_reconstruction_trial_combo.setCurrentIndex(index)
                            break
            self.functional_reconstruction_trial_combo.blockSignals(False)
            self.functional_reconstruction_trial_combo.setEnabled(
                self.functional_reconstruction_trial_combo.currentData() != ""
            )

        def _selected_functional_reconstruction_axis(self):
            axis_name = self.functional_reconstruction_feature_combo.currentData()
            if not axis_name:
                return None
            for axis in self.workflow_draft.axes:
                if axis.name == axis_name and _is_virtual_feature_axis(axis) and _is_sara_direction_method(axis.method):
                    return axis
            return None

        def _functional_reconstruction_axis_pair(self, axis) -> tuple[str, str]:
            """
            Return the proximal/distal segments for a selected functional AoR.
            """
            if axis is None:
                return "", ""
            child_name = axis.segment_name
            parent_name = _parent_segment_name(self.workflow_draft, child_name)
            payload = _key_value_payload(axis.source)
            payload_parent = payload.get("proximal", "") or payload.get("parent segment", "")
            payload_child = payload.get("distal", "") or payload.get("child segment", "")
            if payload_parent:
                parent_name = payload_parent
            if payload_child:
                child_name = payload_child
            return parent_name, child_name

        def _selected_functional_reconstruction_c3d_data(self, axis):
            role = ""
            if hasattr(self, "functional_reconstruction_trial_combo"):
                role = self.functional_reconstruction_trial_combo.currentData() or ""
            if role:
                filepath = _assigned_c3d_source_for_role(self.workflow_draft, role)
                if filepath:
                    try:
                        return self._c3d_data_from_path(filepath), Path(filepath).name
                    except Exception:
                        return None, Path(filepath).name
            if axis is None:
                return None, "unassigned"
            return self._functional_feature_c3d_data_for_source(axis.source)

        def _run_functional_reconstruction_diagnostic(self) -> None:
            """
            Compute a focused SARA/reconstruction report for one functional knee trial.
            """
            axis = self._selected_functional_reconstruction_axis()
            if axis is None:
                QMessageBox.information(
                    self,
                    "Functional reconstruction",
                    "Select a SARA AoR first.",
                )
                self._sync_functional_reconstruction_choices()
                return
            progress_dialog, progress_callback = self._functional_frame_calculation_progress_reporter()
            try:
                progress_callback("Preparing SARA functional trial data...")
                lines, plot_data = self._functional_reconstruction_diagnostic_lines(
                    axis,
                    self.functional_reconstruction_method_combo.currentText().strip(),
                    progress_callback,
                )
            except Exception as error:
                QMessageBox.critical(self, "Unable to run reconstruction diagnostic", str(error))
                return
            finally:
                progress_dialog.close()
            self.functional_reconstruction_output.setPlainText("\n".join(lines))
            self.functional_reconstruction_plot.set_rotation_series(plot_data)

        def _functional_reconstruction_diagnostic_lines(
            self, axis, method: str, progress_callback
        ) -> tuple[tuple[str, ...], dict[str, object] | None]:
            """
            Return text diagnostics and reconstructed knee rotation plot data.
            """
            payload = _key_value_payload(axis.source)
            parent_segment_name, child_segment_name = self._functional_reconstruction_axis_pair(axis)
            parent_marker_names = _split_marker_names(
                payload.get("parent markers", "")
            ) or _technical_markers_for_segment(self.workflow_draft, parent_segment_name)
            child_marker_names = _split_marker_names(
                payload.get("child markers", "")
            ) or _technical_markers_for_segment(self.workflow_draft, child_segment_name)
            functional_data, source_label = self._selected_functional_reconstruction_c3d_data(axis)
            if functional_data is None:
                raise RuntimeError(f"Missing functional C3D source ({source_label}).")
            required_markers = tuple(dict.fromkeys(parent_marker_names + child_marker_names))
            missing_markers = tuple(marker for marker in required_markers if marker not in functional_data.marker_names)
            if missing_markers:
                raise RuntimeError(f"Missing markers: {', '.join(missing_markers)}.")

            progress_callback("Rigidifying proximal and distal marker clusters...")
            from ..components.generic.rigidbody.segment_coordinate_system import (
                SegmentCoordinateSystemUtils,
            )
            from ..model_modifiers.joint_center_tool import Sara, get_svd

            parent_functional_data = functional_data.get_partial_dict_data(parent_marker_names)
            child_functional_data = functional_data.get_partial_dict_data(child_marker_names)
            parent_static_data = None
            child_static_data = None
            if self.c3d_data is not None and all(
                marker in self.c3d_data.marker_names for marker in parent_marker_names + child_marker_names
            ):
                parent_static_data = self.c3d_data.get_partial_dict_data(parent_marker_names)
                child_static_data = self.c3d_data.get_partial_dict_data(child_marker_names)
                rt_parent = SegmentCoordinateSystemUtils.rigidify(
                    functional_data=parent_functional_data,
                    static_data=parent_static_data,
                )
                rt_child = SegmentCoordinateSystemUtils.rigidify(
                    functional_data=child_functional_data,
                    static_data=child_static_data,
                )
                reference_note = "static marker geometry"
            else:
                rt_parent = SegmentCoordinateSystemUtils.rigidify(parent_functional_data)
                rt_child = SegmentCoordinateSystemUtils.rigidify(child_functional_data)
                reference_note = "functional frame geometry"

            rt_parent, rt_child, report = prepare_functional_rt_pair(
                rt_parent,
                rt_child,
                self._current_functional_frame_selection_options(),
            )
            progress_callback("Running SARA and knee DoF diagnostics...")
            expected_markers = _sara_expected_axis_markers(axis, payload)
            original_axis_global = _sara_expected_global_axis(functional_data, expected_markers)
            origin_markers = _sara_origin_markers(axis, payload, expected_markers)
            origin_positions_global = (
                functional_data.markers_center_position(origin_markers)
                if origin_markers and all(marker_name in functional_data.marker_names for marker_name in origin_markers)
                else None
            )
            if _uses_selected_functional_frames(report) and origin_positions_global is not None:
                origin_positions_global = subset_points_by_frame(origin_positions_global, report.selected_indices)
            (
                aor_mean_global,
                aor_parent_local,
                aor_child_local,
                _cor_mean_global,
                cor_parent_local,
                cor_child_local,
                rt_parent_valid,
                rt_child_valid,
            ) = Sara.perform_algorithm(
                rt_parent,
                rt_child,
                original_axis_global=original_axis_global,
                origin_positions_global=origin_positions_global,
            )
            residuals = _sara_residual_angles(rt_parent_valid, rt_child_valid, aor_parent_local, aor_child_local)
            _, singular_values, _, _ = get_svd(rt_parent_valid, rt_child_valid)
            relative_xyz = _relative_xyz_euler_degrees(rt_parent_valid, rt_child_valid)
            relative_xyz_plot_data = _relative_xyz_plot_data(
                relative_xyz,
                child_segment_name or axis.segment_name,
                _c3d_frame_rate(functional_data),
            )

            lines = [
                f"Functional reconstruction diagnostic: {axis.name}",
                f"C3D: {source_label}",
                f"Segments: {parent_segment_name or '-'} -> {child_segment_name or '-'}",
                f"Markers proximal: {', '.join(parent_marker_names)}",
                f"Markers distal: {', '.join(child_marker_names)}",
                f"Method requested: {method}",
                f"Reference for rigidification: {reference_note}",
                format_functional_frame_report(report),
                "",
                "SARA result",
                f"- AoR mean global: {_format_vector(aor_mean_global)}",
                f"- AoR local proximal: {_format_vector(aor_parent_local)}",
                f"- AoR local distal: {_format_vector(aor_child_local)}",
                f"- CoR local proximal: {_format_vector(cor_parent_local)}",
                f"- CoR local distal: {_format_vector(cor_child_local)}",
                "- SVD singular values: "
                + ", ".join(f"{float(value):.4g}" for value in np.asarray(singular_values, dtype=float)),
                "  SARA has six singular values because the linear system estimates one 3D axis direction "
                "in the child frame and one 3D axis direction in the parent frame. The smallest singular "
                "direction is the coupled AoR solution; the spread of the values indicates conditioning/stability.",
            ]
            lines.extend(_histogram_lines("- SARA parent-child angular residual", residuals, "deg"))
            lines.extend(_local_axis_orientation_lines("AoR proximal frame", aor_parent_local))
            lines.extend(_local_axis_orientation_lines("AoR distal frame", aor_child_local))
            lines.extend(
                self._static_expected_axis_comparison_lines(
                    payload,
                    expected_markers,
                    parent_static_data,
                    child_static_data,
                    aor_parent_local,
                    aor_child_local,
                )
            )
            lines.extend(["", "Knee relative rotations (proximal^-1 * distal, XYZ)"])
            for label, values in zip(("X", "Y", "Z"), relative_xyz):
                values = np.asarray(values, dtype=float)
                lines.append(
                    f"- {label} rotation: mean={np.nanmean(values):.2f} deg; "
                    f"range={np.nanmin(values):.2f}..{np.nanmax(values):.2f} deg"
                )

            progress_callback(f"Running {method} reconstruction probe...")
            reconstruction_lines, plot_data = self._functional_reconstruction_method_probe_lines(
                method,
                axis,
                functional_data,
                parent_marker_names,
                child_marker_names,
                source_label,
                tuple(report.selected_indices),
                progress_callback,
            )
            lines.extend(reconstruction_lines)
            return tuple(lines), plot_data or relative_xyz_plot_data

        def _static_expected_axis_comparison_lines(
            self,
            payload: dict[str, str],
            expected_markers: tuple[str, ...],
            parent_static_data,
            child_static_data,
            aor_parent_local: np.ndarray,
            aor_child_local: np.ndarray,
        ) -> tuple[str, ...]:
            if (
                self.c3d_data is None
                or len(expected_markers) < 2
                or parent_static_data is None
                or child_static_data is None
            ):
                return ("", "Static expected-axis comparison unavailable.")
            expected_start_markers = expected_markers[:1]
            expected_end_markers = expected_markers[1:]
            if any(
                marker not in self.c3d_data.marker_names for marker in expected_start_markers + expected_end_markers
            ):
                return ("", "Static expected-axis comparison unavailable.")
            try:
                from ..components.generic.rigidbody.segment_coordinate_system import (
                    SegmentCoordinateSystemUtils,
                )

                expected_global = np.nanmean(
                    _mean_marker_series(self.c3d_data, expected_end_markers)
                    - _mean_marker_series(self.c3d_data, expected_start_markers),
                    axis=1,
                )
                parent_rt_static = SegmentCoordinateSystemUtils.rigidify(parent_static_data)
                child_rt_static = SegmentCoordinateSystemUtils.rigidify(child_static_data)
                expected_parent_local = _mean_local_vector(parent_rt_static, expected_global)
                expected_child_local = _mean_local_vector(child_rt_static, expected_global)
                parent_deviation = _axis_angle_degrees(aor_parent_local, expected_parent_local)
                child_deviation = _axis_angle_degrees(aor_child_local, expected_child_local)
            except Exception as error:
                return ("", f"Static expected-axis comparison unavailable ({error}).")
            limit = _sara_static_deviation_limit_from_payload(payload)
            warning = ""
            if limit is not None and (parent_deviation > limit or child_deviation > limit):
                warning = f" WARNING > {limit:g} deg"
            label = f"{','.join(expected_start_markers)}->{','.join(expected_end_markers)}"
            return (
                "",
                f"Static expected axis ({label})",
                f"- Expected local proximal: {_format_vector(expected_parent_local)}; SARA deviation={parent_deviation:.1f} deg{warning}",
                f"- Expected local distal: {_format_vector(expected_child_local)}; SARA deviation={child_deviation:.1f} deg{warning}",
            )

        def _functional_reconstruction_method_probe_lines(
            self,
            method: str,
            axis,
            functional_data,
            high_weight_parent_markers: tuple[str, ...],
            high_weight_child_markers: tuple[str, ...],
            source_label: str,
            reconstruction_frame_indices: tuple[int, ...],
            progress_callback,
        ) -> tuple[tuple[str, ...], dict[str, object] | None]:
            """
            Run a best-effort model IK probe with EKF or QLD.
            """
            if self.c3d_data is None:
                return (
                    (
                        "",
                        "Model reconstruction probe skipped: no static C3D is loaded.",
                    ),
                    None,
                )
            _parent_segment_name_for_axis, child_segment_name = self._functional_reconstruction_axis_pair(axis)
            try:
                model = self._diagnostic_model_from_current_preset(child_segment_name, progress_callback)
            except Exception as error:
                return (
                    (
                        "",
                        f"Model reconstruction probe unavailable: {error}",
                        "The SARA section above still uses marker-cluster reconstruction and does not need model IK.",
                    ),
                    None,
                )

            progress_callback(f"Running {method} reconstruction on SARA chain...")
            lines, plot_data = self._run_model_reconstruction_probe(
                method,
                model,
                functional_data,
                source_label,
                child_segment_name,
                high_weight_parent_markers,
                high_weight_child_markers,
                reconstruction_frame_indices,
                "SARA chain",
            )

            progress_callback(f"Building marker fallback chain and running {method} reconstruction...")
            fallback_lines, fallback_plot_data = self._functional_reconstruction_fallback_probe_lines(
                method,
                functional_data,
                source_label,
                child_segment_name,
                high_weight_parent_markers,
                high_weight_child_markers,
                reconstruction_frame_indices,
                progress_callback,
            )
            return tuple(lines) + tuple(fallback_lines), _combine_rotation_plot_data(
                plot_data,
                fallback_plot_data,
                fallback_label_suffix="marker fallback",
            )

        def _functional_reconstruction_fallback_probe_lines(
            self,
            method: str,
            functional_data,
            source_label: str,
            child_segment_name: str,
            high_weight_parent_markers: tuple[str, ...],
            high_weight_child_markers: tuple[str, ...],
            reconstruction_frame_indices: tuple[int, ...],
            progress_callback,
        ) -> tuple[tuple[str, ...], dict[str, object] | None]:
            """
            Reconstruct the same trial with a marker-only fallback chain.
            """
            try:
                fallback_model = self._diagnostic_model_from_current_preset(
                    child_segment_name,
                    progress_callback,
                    use_marker_fallback=True,
                )
            except Exception as error:
                return (
                    (
                        "",
                        f"Marker fallback reconstruction unavailable: {error}",
                    ),
                    None,
                )
            return self._run_model_reconstruction_probe(
                method,
                fallback_model,
                functional_data,
                source_label,
                child_segment_name,
                high_weight_parent_markers,
                high_weight_child_markers,
                reconstruction_frame_indices,
                "marker fallback chain",
            )

        def _run_model_reconstruction_probe(
            self,
            method: str,
            model,
            functional_data,
            source_label: str,
            child_segment_name: str,
            high_weight_parent_markers: tuple[str, ...],
            high_weight_child_markers: tuple[str, ...],
            reconstruction_frame_indices: tuple[int, ...],
            probe_label: str,
        ) -> tuple[tuple[str, ...], dict[str, object] | None]:
            """
            Run EKF or QLD against one diagnostic model.
            """
            model_marker_names = set(getattr(model, "marker_names", ()))
            marker_names = tuple(
                marker_name for marker_name in functional_data.marker_names if marker_name in model_marker_names
            )
            if len(marker_names) == 0:
                return (
                    (
                        "",
                        f"{probe_label} reconstruction unavailable: none of the selected functional markers exists in the generated model.",
                    ),
                    None,
                )
            frame_indices, frame_note = _diagnostic_reconstruction_frame_indices(
                reconstruction_frame_indices,
                functional_data.nb_frames,
                method,
            )
            marker_positions = functional_data.get_position(marker_names)[:3, :, :]
            if len(frame_indices) != 0:
                marker_positions = marker_positions[:, :, frame_indices]
            frame_count = marker_positions.shape[2]
            frame_rate = _c3d_frame_rate(functional_data)
            high_weight_markers = tuple(dict.fromkeys(high_weight_parent_markers + high_weight_child_markers))
            marker_weights = _marker_weights_for_reconstruction(marker_names, high_weight_markers)
            if method == "EKF":
                return self._ekf_reconstruction_probe_lines(
                    model,
                    marker_positions,
                    marker_names,
                    functional_data,
                    child_segment_name,
                    frame_rate,
                    frame_indices,
                    frame_note,
                    probe_label=probe_label,
                )
            return self._qld_reconstruction_probe_lines(
                model,
                marker_positions,
                marker_names,
                marker_weights,
                frame_count,
                source_label,
                child_segment_name,
                high_weight_markers,
                frame_rate,
                frame_indices,
                frame_note,
                probe_label=probe_label,
            )

        def _diagnostic_model_from_current_preset(
            self,
            force_three_rotation_segment: str,
            progress_callback,
            *,
            use_marker_fallback: bool = False,
            segment_setting_overrides: dict[str, object] | None = None,
        ):
            """
            Build the current preset model with assigned C3D files for reconstruction diagnostics.
            """
            template = _diagnostic_template_for_c3d_model_preset(
                self.workflow_draft.preset,
                use_marker_fallback=use_marker_fallback,
            )
            functional_data = {}
            if not use_marker_fallback:
                for assignment in self.workflow_draft.file_assignments:
                    if assignment.role == "main" or not assignment.source_path:
                        continue
                    progress_callback(f"Loading assigned functional C3D: {Path(assignment.source_path).name}")
                    functional_data[assignment.role] = self._c3d_data_from_path(assignment.source_path)
            model = create_model_from_marker_data(
                template=template,
                static_data=self.c3d_data,
                functional_data=functional_data,
                preset=self.workflow_draft.preset,
                static_virtual_points=_static_virtual_point_definitions_from_draft(self.workflow_draft),
                progress_callback=progress_callback,
            ).model
            rotation_segment_name = _rotation_segment_name_for_model(model, force_three_rotation_segment)
            self._apply_chain_settings_to_diagnostic_model(
                model,
                force_three_rotation_segment=rotation_segment_name,
                segment_setting_overrides=segment_setting_overrides,
            )
            return model

        def _model_from_current_chain_settings(self, progress_callback):
            """
            Build a model from the current C3D draft and editable chain settings.
            """
            return self._diagnostic_model_from_current_preset(
                "",
                progress_callback,
                use_marker_fallback=False,
            )

        def _apply_chain_settings_to_diagnostic_model(
            self,
            model,
            force_three_rotation_segment: str = "",
            segment_setting_overrides: dict[str, object] | None = None,
        ) -> None:
            """
            Apply the editable chain settings to the diagnostic model and force the selected knee to 3 rotations.
            """
            from ..utils.enums import Rotations, Translations

            settings_by_segment = {setting.segment_name: setting for setting in self.workflow_draft.segment_settings}
            if segment_setting_overrides:
                settings_by_segment.update(segment_setting_overrides)
            for segment in model.segments:
                setting = settings_by_segment.get(segment.name)
                if setting is not None:
                    if setting.translations:
                        segment.translations = Translations(setting.translations)
                    if setting.rotations:
                        segment.rotations = Rotations(setting.rotations)
                    _apply_initial_rotation_setting_to_segment(segment, setting, self.workflow_draft)
                    segment.q_ranges = None
                    segment.qdot_ranges = None
                    segment.update_dof_names()
                if force_three_rotation_segment and segment.name == force_three_rotation_segment:
                    segment.rotations = Rotations.XYZ
                    segment.q_ranges = None
                    segment.qdot_ranges = None
                    segment.update_dof_names()
            model.validate_dofs()

        def _qld_reconstruction_probe_lines(
            self,
            model,
            marker_positions: np.ndarray,
            marker_names: tuple[str, ...],
            marker_weights,
            frame_count: int,
            source_label: str,
            knee_segment_name: str,
            high_weight_markers: tuple[str, ...],
            frame_rate: float | None,
            frame_indices: tuple[int, ...],
            frame_note: str,
            probe_label: str = "chain",
        ) -> tuple[tuple[str, ...], dict[str, object] | None]:
            try:
                q, residuals = model.inverse_kinematics(
                    marker_positions,
                    list(marker_names),
                    marker_weights=marker_weights,
                    method="lm",
                    compute_residual_distance=True,
                )
            except Exception as error:
                return ("", f"QLD reconstruction unavailable: {error}"), None
            lines = [
                "",
                f"{probe_label} - QLD reconstruction probe ({source_label})",
                f"- Frames used: {frame_count}",
                f"- Frame selection: {frame_note}",
                f"- q shape: {q.shape[0]} DoF x {q.shape[1]} frames",
                f"- Marker weights: 100 for {', '.join(high_weight_markers) or '-'}; 1 for the other model markers in this C3D.",
            ]
            rotation_plot = _rotation_dof_plot_data(model, q, knee_segment_name, frame_rate, frame_indices)
            lines.extend(_rotation_dof_summary_lines(model, q, knee_segment_name))
            if residuals is not None:
                lines.extend(
                    _histogram_lines(
                        "- marker residual",
                        np.asarray(residuals, dtype=float).reshape(-1) * 1000.0,
                        "mm",
                    )
                )
            return tuple(lines), rotation_plot

        def _ekf_reconstruction_probe_lines(
            self,
            model,
            marker_positions: np.ndarray,
            marker_names: tuple[str, ...],
            functional_data,
            knee_segment_name: str,
            frame_rate: float | None,
            frame_indices: tuple[int, ...],
            frame_note: str,
            probe_label: str = "chain",
        ) -> tuple[tuple[str, ...], dict[str, object] | None]:
            try:
                import tempfile

                import biorbd  # type: ignore

                with tempfile.NamedTemporaryFile(suffix=".bioMod", delete=True) as temporary_file:
                    model.to_biomod(temporary_file.name, with_mesh=False)
                    biorbd_model = biorbd.Model(temporary_file.name)
                    biorbd_marker_names = tuple(name.to_string() for name in biorbd_model.markerNames())
                    marker_indices = [marker_names.index(name) for name in biorbd_marker_names if name in marker_names]
                    if len(marker_indices) == 0:
                        return (
                            (
                                "",
                                "EKF reconstruction unavailable: no selected functional marker is present in the generated biorbd model.",
                            ),
                            None,
                        )
                    markers = marker_positions[:, marker_indices, :]
                    frequency = int(round(_c3d_frame_rate(functional_data) or 100.0))
                    params = biorbd.KalmanParam(frequency)
                    kalman = biorbd.KalmanReconsMarkers(biorbd_model, params)
                    q = biorbd.GeneralizedCoordinates(biorbd_model)
                    qdot = biorbd.GeneralizedVelocity(biorbd_model)
                    qddot = biorbd.GeneralizedAcceleration(biorbd_model)
                    q_values = np.zeros((biorbd_model.nbQ(), markers.shape[2]))
                    for frame_index in range(markers.shape[2]):
                        nodes = [
                            biorbd.NodeSegment(markers[:, marker_index, frame_index])
                            for marker_index in range(markers.shape[1])
                        ]
                        kalman.reconstructFrame(biorbd_model, nodes, q, qdot, qddot)
                        q_values[:, frame_index] = q.to_array()
            except Exception as error:
                return (
                    (
                        "",
                        f"EKF reconstruction unavailable: {error}",
                        "Check that biorbd was compiled with Kalman marker reconstruction and that the marker set matches the model technical markers.",
                    ),
                    None,
                )
            lines = [
                "",
                f"{probe_label} - EKF reconstruction probe",
                f"- Frames used: {q_values.shape[1]}",
                f"- Frame selection: {frame_note}",
                f"- q shape: {q_values.shape[0]} DoF x {q_values.shape[1]} frames",
                "- Marker weights: biorbd EKF does not expose per-marker weights here; use QLD for the 100/1 weighted reconstruction.",
            ]
            rotation_plot = _rotation_dof_plot_data(model, q_values, knee_segment_name, frame_rate, frame_indices)
            lines.extend(_rotation_dof_summary_lines(model, q_values, knee_segment_name))
            return tuple(lines), rotation_plot

        def _update_generation_log(self) -> None:
            lines = _c3d_generation_log(
                self.workflow_draft,
                self.c3d_data,
                self.c3d_folder_path,
                self.workflow_marker_pool,
            )
            lines = tuple(lines) + self._functional_trial_quality_lines()
            self.generation_log_edit.setPlainText("\n".join(lines))

        def _show_workflow_summary(self) -> None:
            """
            Show the workflow summary on demand instead of keeping a permanent tab.
            """
            QMessageBox.information(
                self,
                "C3D workflow summary",
                c3d_workflow_summary(self.workflow_draft.preset, self.c3d_data),
            )

        def _update_c3d_names_panel(self) -> None:
            """
            Refresh the compact C3D files and marker names status panel.
            """
            workflow = c3d_creation_workflow(self.workflow_draft.preset)
            required_roles = {role.role for role in workflow.file_roles if role.required}
            assigned_count = sum(1 for assignment in self.workflow_draft.file_assignments if assignment.source_path)
            missing_required = tuple(
                assignment.role
                for assignment in self.workflow_draft.file_assignments
                if assignment.role in required_roles and not assignment.source_path
            )
            folder = self.c3d_folder_path or "not selected"
            main_c3d = Path(self.c3d_path.text().strip()).name if self.c3d_path.text().strip() else "not selected"
            missing_text = ", ".join(missing_required) if missing_required else "none"
            self.c3d_names_overview_label.setText(
                f"Folder: {folder}\n"
                f"Main C3D: {main_c3d}\n"
                f"Assigned C3D files: {assigned_count}/{len(self.workflow_draft.file_assignments)} "
                f"(missing required: {missing_text})"
            )

            expected_markers = _expected_marker_names_for_preset(self.workflow_draft.preset)
            if self.c3d_data is None:
                self.c3d_marker_status_label.setText("Marker names: load a main/static C3D to check template matches.")
                return
            marker_mapping = _marker_name_mapping_for_c3d(expected_markers, tuple(self.c3d_data.marker_names))
            missing_markers = tuple(marker for marker in expected_markers if marker not in marker_mapping)
            unassigned_markers = _unassigned_marker_names(
                self.workflow_marker_pool, self.workflow_draft.segment_marker_groups
            )
            missing_preview = (
                ", ".join(missing_markers[:12]) + ("..." if len(missing_markers) > 12 else "")
                if missing_markers
                else "none"
            )
            unassigned_preview = (
                ", ".join(unassigned_markers[:12]) + ("..." if len(unassigned_markers) > 12 else "")
                if unassigned_markers
                else "none"
            )
            prefix_status = "on" if self.strip_participant_prefix_checkbox.isChecked() else "off"
            self.c3d_marker_status_label.setText(
                f"Marker names: {len(marker_mapping)}/{len(expected_markers)} template markers matched; "
                f"{len(self.c3d_data.marker_names)} markers in main C3D; known pool={len(self.workflow_marker_pool)}.\n"
                f"Participant prefix removal: {prefix_status}. Missing: {missing_preview}. "
                f"Unassigned/available: {unassigned_preview}."
            )

        def _refresh_marker_assignment_details(self, segment_name: str | None) -> None:
            """
            Refresh only the widgets affected by marker assignment edits.
            """
            anatomical_segment_name = self._selected_anatomical_segment_name()
            self._populate_segment_marker_lists()
            self._restore_workflow_segment_selection(segment_name)
            self._restore_anatomical_segment_selection(anatomical_segment_name)
            self._refresh_workflow_progress_and_issues()
            self._update_c3d_names_panel()

        def _refresh_workflow_progress_and_issues(self) -> None:
            self.step_list.clear()
            for step_status in c3d_workflow_progress(self.workflow_draft, self.c3d_data):
                self.step_list.addItem(_workflow_step_item(step_status))
            self.issue_list.clear()
            issues = validate_c3d_workflow_draft(self.workflow_draft, self.c3d_data)
            if len(issues) == 0:
                self.issue_list.addItem("No draft issue detected.")
                return
            for issue in issues:
                item = QListWidgetItem(f"{issue.severity.upper()} | {issue.category} | {issue.message}")
                if issue.severity == "error":
                    item.setForeground(QColor("#991b1b"))
                    item.setBackground(QColor("#fee2e2"))
                elif issue.severity == "warning":
                    item.setForeground(QColor("#92400e"))
                    item.setBackground(QColor("#fef3c7"))
                self.issue_list.addItem(item)

        def _generate_python_code(self) -> None:
            default_folder = self.c3d_folder_path or str(Path.home())
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Save generated Python code",
                str(Path(default_folder) / f"{self.workflow_draft.preset.value}_model_definition.py"),
                "Python files (*.py)",
            )
            if not filepath:
                return
            try:
                Path(filepath).write_text(
                    _python_code_from_c3d_draft(self.workflow_draft, self.c3d_folder_path),
                    encoding="utf-8",
                )
            except Exception as error:
                QMessageBox.critical(self, "Unable to generate Python code", str(error))

        def _selected_segment_setting(self):
            if not self.segment_settings_list.selectedItems():
                return None
            segment_name = self.segment_settings_list.selectedItems()[0].text().split("|", maxsplit=1)[0].strip()
            for setting in self.workflow_draft.segment_settings:
                if setting.segment_name == segment_name:
                    return setting
            return None

        def _selected_c3d_role(self) -> str | None:
            if not self.file_role_list.selectedItems():
                return None
            return self.file_role_list.selectedItems()[0].text().split("|", maxsplit=1)[0].strip()

        def _update_preset_details(self) -> None:
            previously_selected_segment = self._selected_workflow_segment_name()
            previously_selected_anatomical_segment = self._selected_anatomical_segment_name()
            preset = self.selected_preset()
            if self.workflow_draft.preset != preset:
                self.workflow_draft = c3d_workflow_draft(preset)
                self.workflow_marker_pool = (
                    tuple(self.c3d_data.marker_names)
                    if self.c3d_data is not None
                    else _marker_pool_from_draft(self.workflow_draft)
                )
                previously_selected_segment = None
                previously_selected_anatomical_segment = None
                self._auto_assign_c3d_files_from_folder(load_main=not bool(self.c3d_path.text().strip()))
            workflow = c3d_creation_workflow(preset)
            self.step_list.clear()
            self.marker_list.clear()
            self.segment_marker_list.clear()
            self.anatomical_segment_list.clear()
            self.assigned_marker_list.clear()
            self.feature_list.clear()
            self.axis_list.clear()
            self.segment_settings_list.clear()
            self.file_role_list.clear()
            self.issue_list.clear()

            if preset == C3dModelPreset.FROM_SCRATCH:
                self.status_label.setText(
                    "Status: template-free draft; add segments, markers, axes, DoFs, and virtual markers manually."
                )
            elif preset == C3dModelPreset.FULL_BODY:
                self.status_label.setText(
                    "Status: full-body template mapping exists; generation still needs virtual markers."
                )
            elif preset == C3dModelPreset.MOTIVE_57:
                self.status_label.setText(
                    "Status: Motive (57) template mapping exists; GH centers need Rab virtual markers."
                )
            elif preset == C3dModelPreset.MOTIVE_57_ISB:
                self.status_label.setText(
                    "Status: Motive (57) ISB profile; anatomical axes follow static landmarks and GH centers need Rab virtual markers."
                )
            elif preset == C3dModelPreset.UPPER_LIMB:
                self.status_label.setText(
                    "Status: upper-limb template exists; virtual markers/axes must be supplied before generation."
                )
            elif preset == C3dModelPreset.LOWER_LIMBS:
                self.status_label.setText("Status: ready with main marker C3D and functional SCoRE/SARA trials.")
            elif preset == C3dModelPreset.LOWER_LIMBS_ANATOMICAL:
                self.status_label.setText("Status: ready with main marker C3D; segment frames are marker-defined.")
            else:
                self.status_label.setText("Status: ready with main marker C3D and optional functional trials.")

            self._refresh_workflow_progress_and_issues()
            self._populate_segment_marker_lists()
            self._restore_workflow_segment_selection(previously_selected_segment)
            self._restore_anatomical_segment_selection(previously_selected_anatomical_segment)
            self._update_available_marker_list()
            self._sync_virtual_marker_choices()
            self._update_technical_segment_preview()

            for label in _virtual_feature_list_labels(self.workflow_draft):
                self.feature_list.addItem(label)
            if self.feature_list.count() != 0 and self.feature_list.currentItem() is None:
                self.feature_list.setCurrentItem(self.feature_list.item(0))
            self._sync_functional_residual_button()
            self._sync_functional_reconstruction_choices()

            anatomical_frame_lines = _anatomical_frame_instruction_lines(self.workflow_draft)
            if len(anatomical_frame_lines) == 0:
                self.axis_list.addItem("No anatomical frame recipe yet.")
            else:
                for line in anatomical_frame_lines:
                    self.axis_list.addItem(line)

            for setting in self.workflow_draft.segment_settings:
                anthropometry = setting.anthropometry_model or "-"
                length = "-" if setting.segment_length is None else f"{setting.segment_length:.4g}"
                self.segment_settings_list.addItem(
                    f"{setting.segment_name} | translations={setting.translations or '-'} | "
                    f"rotations={setting.rotations or '-'} | child_translation={setting.child_translation} | "
                    f"initial_rotation={setting.initial_rotation_method} | anthropometry={anthropometry} | "
                    f"length={length}"
                )
            if self.segment_settings_list.count() != 0 and self.segment_settings_list.currentItem() is None:
                self.segment_settings_list.setCurrentItem(self.segment_settings_list.item(0))

            for file_role in self.workflow_draft.file_assignments:
                role_definition = next(role for role in workflow.file_roles if role.role == file_role.role)
                required = "required" if role_definition.required else "optional"
                source = Path(file_role.source_path).name if file_role.source_path else "not assigned"
                self.file_role_list.addItem(
                    f"{file_role.role} | {source} | expected={file_role.generic_name} | {required}"
                )

            self._update_c3d_names_panel()
            self._update_virtual_marker_preview()
            self._update_segment_settings_preview()
            self._update_generation_log()

        def _populate_segment_marker_lists(self) -> None:
            self.segment_marker_list.blockSignals(True)
            self.anatomical_segment_list.blockSignals(True)
            self.segment_marker_list.clear()
            self.anatomical_segment_list.clear()
            for group in self.workflow_draft.segment_marker_groups:
                text = _segment_marker_group_label(group)
                self.segment_marker_list.addItem(text)
                self.anatomical_segment_list.addItem(text)
            self.segment_marker_list.blockSignals(False)
            self.anatomical_segment_list.blockSignals(False)

        def _restore_workflow_segment_selection(self, segment_name: str | None) -> None:
            target_index = 0
            if segment_name is not None:
                for index in range(self.segment_marker_list.count()):
                    item_segment_name = self.segment_marker_list.item(index).text().split(":", maxsplit=1)[0]
                    if item_segment_name == segment_name:
                        target_index = index
                        break
            if self.segment_marker_list.count() == 0:
                self._update_assigned_marker_list()
                self._update_available_marker_list()
                return
            self.segment_marker_list.setCurrentItem(self.segment_marker_list.item(target_index))
            self._update_assigned_marker_list()
            self._update_available_marker_list()

        def _restore_anatomical_segment_selection(self, segment_name: str | None) -> None:
            target_index = 0
            if segment_name is not None:
                for index in range(self.anatomical_segment_list.count()):
                    item_segment_name = self.anatomical_segment_list.item(index).text().split(":", maxsplit=1)[0]
                    if item_segment_name == segment_name:
                        target_index = index
                        break
            if self.anatomical_segment_list.count() == 0:
                self._update_anatomical_segment_details()
                return
            self.anatomical_segment_list.setCurrentItem(self.anatomical_segment_list.item(target_index))
            self._update_anatomical_segment_details()

        def _update_available_marker_list(self) -> None:
            self.marker_list.clear()
            if self.c3d_data is None and len(self.workflow_marker_pool) == 0:
                self.marker_list.addItem("Choose the main marker C3D to list markers.")
                return
            marker_names = tuple(self.c3d_data.marker_names) if self.c3d_data is not None else self.workflow_marker_pool
            virtual_marker_names = tuple(marker.name for marker in self.workflow_draft.virtual_markers)
            if self.show_virtual_markers_in_segments_checkbox.isChecked():
                marker_names = tuple(dict.fromkeys(marker_names + virtual_marker_names))
            else:
                virtual_marker_name_set = set(virtual_marker_names)
                marker_names = tuple(
                    marker_name for marker_name in marker_names if marker_name not in virtual_marker_name_set
                )
            selected_segment_name = self._selected_workflow_segment_name()
            selected_segment_marker_names = {
                marker_name
                for group in self.workflow_draft.segment_marker_groups
                if group.segment_name == selected_segment_name
                for marker_name in group.marker_names
            }
            marker_names = tuple(
                marker_name for marker_name in marker_names if marker_name not in selected_segment_marker_names
            )
            if not self.show_all_markers_checkbox.isChecked():
                assigned_marker_names = {
                    marker_name
                    for group in self.workflow_draft.segment_marker_groups
                    if group.segment_name != selected_segment_name
                    for marker_name in group.marker_names
                }
                marker_names = tuple(
                    marker_name for marker_name in marker_names if marker_name not in assigned_marker_names
                )
            if len(marker_names) == 0:
                self.marker_list.addItem("No available marker with the current filter.")
                return
            for marker_name in marker_names:
                self.marker_list.addItem(marker_name)

    class ModelPreviewWidget(QWidget):
        """
        Lightweight 3D-aware preview rendered with an isometric projection.
        """

        def __init__(self):
            super().__init__()
            self.scene = None
            self.selected_segment_name = None
            self.on_segment_selected = None
            self.on_marker_selected = None
            self._projected_joint_positions = {}
            self._projected_marker_positions = {}
            self._press_mouse_position = None
            _initialize_preview_camera(self)

        def set_model(self, model) -> None:
            self.scene = None if model is None else build_preview_scene(model)
            self.update()

        def set_selected_segment(self, segment_name: str | None) -> None:
            self.selected_segment_name = segment_name
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            _set_preview_render_hints(self, painter)
            painter.fillRect(self.rect(), QColor("white"))
            if self.scene is None or not self.scene.joints:
                painter.drawText(self.rect(), qt_alignment_center, "Open a model to preview it")
                return

            projected_joints = {
                name: _rotate_preview_point(point, self.yaw, self.pitch) for name, point in self.scene.joints.items()
            }
            projected_markers = {
                name: _rotate_preview_point(point, self.yaw, self.pitch) for name, point in self.scene.markers.items()
            }
            projected_axes = [
                (
                    axis,
                    _rotate_preview_point(axis.start, self.yaw, self.pitch),
                    _rotate_preview_point(axis.end, self.yaw, self.pitch),
                )
                for axis in self.scene.segment_axes
            ]
            all_points = list(projected_joints.values()) + list(projected_markers.values())
            for path in self.scene.muscles.values():
                all_points.extend(_rotate_preview_point(point, self.yaw, self.pitch) for point in path)
            for _, start, end in projected_axes:
                all_points.extend([start, end])
            transform = _fit_projection(all_points, self.width(), self.height(), QPointF, self.zoom)
            self._projected_joint_positions = {name: transform(point) for name, point in projected_joints.items()}
            self._projected_marker_positions = {name: transform(point) for name, point in projected_markers.items()}

            painter.setPen(QPen(QColor("#6b7280"), 2))
            for parent, child in self.scene.bones:
                painter.drawLine(
                    transform(projected_joints[parent]),
                    transform(projected_joints[child]),
                )

            painter.setPen(QPen(QColor("#dc2626"), 2))
            for path in self.scene.muscles.values():
                for start, end in zip(path, path[1:]):
                    painter.drawLine(
                        transform(_rotate_preview_point(start, self.yaw, self.pitch)),
                        transform(_rotate_preview_point(end, self.yaw, self.pitch)),
                    )

            for axis, start, end in projected_axes:
                painter.setPen(
                    QPen(
                        QColor(_preview_axis_color(axis.axis)),
                        4 if axis.is_rotation_axis else 1,
                    )
                )
                painter.drawLine(transform(start), transform(end))

            painter.setPen(QPen(QColor("#2563eb"), 1))
            painter.setBrush(QColor("#2563eb"))
            for marker_name, marker_point in sorted(
                self.scene.markers.items(),
                key=lambda item: _preview_depth(item[1], self.yaw, self.pitch),
            ):
                center = transform(projected_markers[marker_name])
                painter.drawEllipse(center, 4, 4)

            for name, point in sorted(
                self.scene.joints.items(),
                key=lambda item: _preview_depth(item[1], self.yaw, self.pitch),
            ):
                center = self._projected_joint_positions[name]
                is_selected = name == self.selected_segment_name
                painter.setPen(QPen(QColor("#111827"), 1))
                painter.setBrush(QColor("#f59e0b" if is_selected else "#111827"))
                painter.drawEllipse(center, 5 if is_selected else 3, 5 if is_selected else 3)

            _draw_preview_orientation_axes(painter, self.width(), self.height(), self.yaw, self.pitch)
            _draw_preview_interaction_hint(painter, self.width())
            self._draw_legend(painter)

        def _draw_legend(self, painter) -> None:
            """
            Draw the preview color legend in the top-left corner.
            """
            x = 12
            y = 18
            line_gap = 18
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.setBrush(QColor(255, 255, 255, 225))
            painter.drawRect(8, 8, 230, 148)

            painter.setPen(QPen(QColor("#111827"), 1))
            painter.drawText(x, y, "Legend")
            y += line_gap
            _draw_legend_point(painter, QPointF(x + 6, y - 4), QColor("#2563eb"), "Markers", x + 20, y)
            y += line_gap
            _draw_legend_point(
                painter,
                QPointF(x + 6, y - 4),
                QColor("#111827"),
                "Joint centers",
                x + 20,
                y,
            )
            y += line_gap
            _draw_legend_line(painter, QColor("#6b7280"), 2, x, y - 4, "Bones", x + 36, y)
            y += line_gap
            _draw_legend_line(painter, QColor("#dc2626"), 2, x, y - 4, "Muscles", x + 36, y)
            y += line_gap
            _draw_legend_line(
                painter,
                QColor(PREVIEW_AXIS_COLORS["x"]),
                1,
                x,
                y - 4,
                "x axis",
                x + 36,
                y,
            )
            y += line_gap
            _draw_legend_line(
                painter,
                QColor(PREVIEW_AXIS_COLORS["y"]),
                1,
                x,
                y - 4,
                "y axis",
                x + 36,
                y,
            )
            y += line_gap
            _draw_legend_line(painter, QColor("#2563eb"), 1, x, y - 4, "z axis", x + 36, y)
            y += line_gap
            _draw_legend_line(painter, QColor("#6b7280"), 4, x, y - 4, "Rotational axis", x + 36, y)

        def mousePressEvent(self, event) -> None:
            if event.button() == qt_right_button:
                event.accept()
                return
            self._press_mouse_position = get_event_position(event)
            _start_preview_camera_drag(self, event)

        def contextMenuEvent(self, event) -> None:
            self._show_view_plane_menu(event.globalPos())
            event.accept()

        def _show_view_plane_menu(self, position) -> None:
            """
            Show standard orthographic plane choices for the model preview.
            """
            menu = QMenu(self)
            plane_actions = {menu.addAction(f"{plane} plane"): plane for plane in ("XY", "YZ", "ZX")}
            menu.addSeparator()
            subject_actions = {
                menu.addAction(label): view
                for label, view in (
                    ("Face view", "face"),
                    ("Back view", "dos"),
                    ("Side view", "cote"),
                )
            }
            selected_action = _exec_menu(menu, position)
            plane = plane_actions.get(selected_action)
            if plane is not None:
                self.yaw = _preview_camera_matrix_for_plane(plane)
            else:
                view = subject_actions.get(selected_action)
                if view is None:
                    return
                try:
                    self.yaw = _preview_camera_matrix_for_subject_view(
                        view, self.scene.markers if self.scene is not None else {}
                    )
                except ValueError as error:
                    QMessageBox.warning(self, "Unavailable view", str(error))
                    return
            self.pitch = 0.0
            self._last_mouse_position = None
            self._is_preview_dragging = False
            self.setCursor(qt_open_hand_cursor)
            self.update()

        def mouseMoveEvent(self, event) -> None:
            _drag_preview_camera(self, event)

        def mouseReleaseEvent(self, event) -> None:
            released = get_event_position(event)
            pressed = self._press_mouse_position
            _end_preview_camera_drag(self)
            self._press_mouse_position = None
            if pressed is None:
                return
            if abs(released.x() - pressed.x()) > 4 or abs(released.y() - pressed.y()) > 4:
                return
            clicked = released
            marker_name = _nearest_projected_segment(self._projected_marker_positions, clicked)
            if marker_name is not None and self.on_marker_selected is not None:
                self.on_marker_selected(marker_name)
                return
            segment_name = _nearest_projected_segment(self._projected_joint_positions, clicked)
            if segment_name is not None and self.on_segment_selected is not None:
                self.on_segment_selected(segment_name)

        def mouseDoubleClickEvent(self, event) -> None:
            _reset_preview_camera(self)

        def wheelEvent(self, event) -> None:
            _zoom_preview_camera(self, event)

    class ModelEditorWindow(QMainWindow):
        """
        Minimal desktop editor for inspecting and editing segment properties.
        """

        def __init__(self):
            super().__init__()
            self.setWindowTitle("BioBuddy Model Editor")
            self._resize_to_available_screen(QApplication)
            self.model = None
            self.current_filepath: Path | None = None
            self.current_segment_name: str | None = None
            self.preview = ModelPreviewWidget()
            _style_preview_widget(self.preview)
            self.preview.on_segment_selected = self._select_segment_from_preview
            self.preview.on_marker_selected = self._select_marker_from_preview
            self.validation_messages = QListWidget()
            self.validation_messages.setAlternatingRowColors(True)
            self.validate_button = QPushButton("Validate model")
            self.validate_button.setObjectName("PrimaryActionButton")
            self.validate_button.clicked.connect(self._validate_model)

            self.tree = QTreeWidget()
            self.tree.setHeaderLabel("Segments")
            self.tree.setAlternatingRowColors(True)
            self.tree.itemSelectionChanged.connect(self._on_segment_selection_changed)

            self.parent_name = QLineEdit()
            self.translations = QLineEdit()
            self.rotations = QLineEdit()
            self.q_min = QLineEdit()
            self.q_max = QLineEdit()
            self.mass = QLineEdit()
            self.center_of_mass_fields = {axis: QLineEdit() for axis in ("x", "y", "z")}
            self.inertia_mode = QComboBox()
            self.inertia_mode.addItems(["Principal moments", "Inertia matrix"])
            self.inertia_mode.currentTextChanged.connect(self._update_inertia_component_state)
            self.inertia_fields = {(row, column): QLineEdit() for row in range(3) for column in range(3)}

            self.apply_button = QPushButton("Apply segment changes")
            self.apply_button.setObjectName("PrimaryActionButton")
            self.apply_button.clicked.connect(self._apply_segment_changes)
            self.apply_inertial_model_button = QPushButton("Use inertial model")
            self.apply_inertial_model_button.clicked.connect(self._apply_inertial_model)

            form = QFormLayout()
            form.addRow("Parent", self.parent_name)
            form.addRow("Translations", self.translations)
            form.addRow("Rotations", self.rotations)
            form.addRow("q min", self.q_min)
            form.addRow("q max", self.q_max)
            form.addRow("Mass", self.mass)
            form.addRow("Center of mass", self._component_row(self.center_of_mass_fields))
            form.addRow("Inertia mode", self.inertia_mode)
            form.addRow("Inertia", self._inertia_grid())

            right_panel = QWidget()
            segment_tab = QWidget()
            segment_layout = QVBoxLayout(segment_tab)
            _configure_panel_layout(segment_layout)
            segment_layout.addWidget(_section_label("Segment properties"))
            segment_layout.addLayout(form)
            segment_layout.addWidget(self.apply_inertial_model_button)
            segment_layout.addWidget(self.apply_button)
            segment_layout.addStretch()
            self._update_inertia_component_state()

            self.marker_list = QListWidget()
            self.marker_list.setAlternatingRowColors(True)
            self.marker_list.itemSelectionChanged.connect(self._on_marker_selection_changed)
            self.marker_name = QLineEdit()
            self.marker_name.setPlaceholderText("New or selected marker name")
            self.marker_position = QLineEdit()
            self.marker_position.setPlaceholderText("x y z")
            self.marker_technical = QCheckBox("Technical")
            self.marker_anatomical = QCheckBox("Anatomical")
            self.apply_marker_button = QPushButton("Apply marker changes")
            self.apply_marker_button.setObjectName("PrimaryActionButton")
            self.apply_marker_button.clicked.connect(self._apply_marker_changes)
            self.add_marker_button = QPushButton("Add marker")
            self.add_marker_button.setObjectName("SecondaryActionButton")
            self.add_marker_button.clicked.connect(self._add_marker)
            self.remove_marker_button = QPushButton("Remove marker")
            self.remove_marker_button.setObjectName("DangerActionButton")
            self.remove_marker_button.clicked.connect(self._remove_marker)
            self.marker_target_segment = QLineEdit()
            self.marker_target_segment.setPlaceholderText("Target segment")
            self.attach_marker_button = QPushButton("Attach marker to segment")
            self.attach_marker_button.setObjectName("SecondaryActionButton")
            self.attach_marker_button.clicked.connect(self._attach_marker_to_segment)

            marker_form = QFormLayout()
            marker_form.addRow("Name", self.marker_name)
            marker_form.addRow("Position", self.marker_position)
            marker_form.addRow("", self.marker_technical)
            marker_form.addRow("", self.marker_anatomical)
            marker_form.addRow("Target segment", self.marker_target_segment)

            marker_tab = QWidget()
            marker_layout = QVBoxLayout(marker_tab)
            _configure_panel_layout(marker_layout)
            marker_layout.addWidget(_section_label("Markers on selected segment"))
            marker_layout.addWidget(self.marker_list)
            marker_layout.addLayout(marker_form)
            marker_layout.addWidget(self.apply_marker_button)
            marker_layout.addWidget(self.add_marker_button)
            marker_layout.addWidget(self.remove_marker_button)
            marker_layout.addWidget(self.attach_marker_button)

            self.muscle_tree = QTreeWidget()
            self.muscle_tree.setHeaderLabel("Muscles")
            self.muscle_tree.setAlternatingRowColors(True)
            self.muscle_tree.itemSelectionChanged.connect(self._on_muscle_selection_changed)
            self.optimal_length = QLineEdit()
            self.maximal_force = QLineEdit()
            self.tendon_slack_length = QLineEdit()
            self.pennation_angle = QLineEdit()
            self.maximal_velocity = QLineEdit()
            self.maximal_excitation = QLineEdit()
            self.apply_muscle_button = QPushButton("Apply muscle changes")
            self.apply_muscle_button.setObjectName("PrimaryActionButton")
            self.apply_muscle_button.clicked.connect(self._apply_muscle_changes)
            self.group_name = QLineEdit()
            self.group_origin_parent = QLineEdit()
            self.group_insertion_parent = QLineEdit()
            self.add_group_button = QPushButton("Add muscle group")
            self.add_group_button.setObjectName("SecondaryActionButton")
            self.add_group_button.clicked.connect(self._add_muscle_group)
            self.remove_group_button = QPushButton("Remove selected group")
            self.remove_group_button.setObjectName("DangerActionButton")
            self.remove_group_button.clicked.connect(self._remove_muscle_group)
            self.new_muscle_name = QLineEdit()
            self.add_muscle_button = QPushButton("Add muscle")
            self.add_muscle_button.setObjectName("SecondaryActionButton")
            self.add_muscle_button.clicked.connect(self._add_muscle)
            self.remove_muscle_button = QPushButton("Remove selected muscle")
            self.remove_muscle_button.setObjectName("DangerActionButton")
            self.remove_muscle_button.clicked.connect(self._remove_muscle)
            self.origin_name = QLineEdit()
            self.origin_parent = QLineEdit()
            self.origin_position = QLineEdit()
            self.insertion_name = QLineEdit()
            self.insertion_parent = QLineEdit()
            self.insertion_position = QLineEdit()
            self.apply_path_endpoints_button = QPushButton("Apply origin/insertion changes")
            self.apply_path_endpoints_button.setObjectName("PrimaryActionButton")
            self.apply_path_endpoints_button.clicked.connect(self._apply_path_endpoint_changes)

            muscle_form = QFormLayout()
            muscle_form.addRow("New group name", self.group_name)
            muscle_form.addRow("Group origin parent", self.group_origin_parent)
            muscle_form.addRow("Group insertion parent", self.group_insertion_parent)
            muscle_form.addRow("Optimal length", self.optimal_length)
            muscle_form.addRow("Maximal force", self.maximal_force)
            muscle_form.addRow("Tendon slack length", self.tendon_slack_length)
            muscle_form.addRow("Pennation angle", self.pennation_angle)
            muscle_form.addRow("Maximal velocity", self.maximal_velocity)
            muscle_form.addRow("Maximal excitation", self.maximal_excitation)
            muscle_form.addRow("Origin name", self.origin_name)
            muscle_form.addRow("Origin parent", self.origin_parent)
            muscle_form.addRow("Origin position", self.origin_position)
            muscle_form.addRow("Insertion name", self.insertion_name)
            muscle_form.addRow("Insertion parent", self.insertion_parent)
            muscle_form.addRow("Insertion position", self.insertion_position)

            self.via_point_list = QListWidget()
            self.via_point_list.setAlternatingRowColors(True)
            self.via_point_list.itemSelectionChanged.connect(self._on_via_point_selection_changed)
            self.via_point_name = QLineEdit()
            self.via_point_name.setPlaceholderText("New or selected via-point name")
            self.via_point_parent = QLineEdit()
            self.via_point_parent.setPlaceholderText("Parent segment")
            self.via_point_position = QLineEdit()
            self.via_point_position.setPlaceholderText("x y z")
            self.apply_via_point_button = QPushButton("Apply via-point changes")
            self.apply_via_point_button.setObjectName("PrimaryActionButton")
            self.apply_via_point_button.clicked.connect(self._apply_via_point_changes)
            self.add_via_point_button = QPushButton("Add via point")
            self.add_via_point_button.setObjectName("SecondaryActionButton")
            self.add_via_point_button.clicked.connect(self._add_via_point)
            self.remove_via_point_button = QPushButton("Remove via point")
            self.remove_via_point_button.setObjectName("DangerActionButton")
            self.remove_via_point_button.clicked.connect(self._remove_via_point)

            via_point_form = QFormLayout()
            via_point_form.addRow("Via-point name", self.via_point_name)
            via_point_form.addRow("Parent", self.via_point_parent)
            via_point_form.addRow("Position", self.via_point_position)

            muscle_tab = QWidget()
            muscle_layout = QVBoxLayout(muscle_tab)
            _configure_panel_layout(muscle_layout)
            muscle_layout.addWidget(_section_label("Muscles"))
            muscle_layout.addWidget(self.muscle_tree)
            muscle_layout.addWidget(self.add_group_button)
            muscle_layout.addWidget(self.remove_group_button)
            muscle_layout.addWidget(self.new_muscle_name)
            muscle_layout.addWidget(self.add_muscle_button)
            muscle_layout.addWidget(self.remove_muscle_button)
            muscle_layout.addLayout(muscle_form)
            muscle_layout.addWidget(self.apply_muscle_button)
            muscle_layout.addWidget(self.apply_path_endpoints_button)
            muscle_layout.addWidget(_section_label("Via points"))
            muscle_layout.addWidget(self.via_point_list)
            muscle_layout.addLayout(via_point_form)
            muscle_layout.addWidget(self.apply_via_point_button)
            muscle_layout.addWidget(self.add_via_point_button)
            muscle_layout.addWidget(self.remove_via_point_button)

            tabs = QTabWidget()
            tabs.addTab(_scrollable_widget(segment_tab), "Segment")
            tabs.addTab(_scrollable_widget(marker_tab), "Markers")
            tabs.addTab(_scrollable_widget(muscle_tab), "Muscles")
            tabs.addTab(self.preview, "3D preview")
            validation_tab = QWidget()
            validation_layout = QVBoxLayout(validation_tab)
            _configure_panel_layout(validation_layout)
            validation_layout.addWidget(self.validate_button)
            graph_model_button = QPushButton("Graph model")
            graph_model_button.setObjectName("SecondaryActionButton")
            graph_model_button.clicked.connect(self._write_model_graphviz)
            plot_muscles_button = QPushButton("Plot muscles")
            plot_muscles_button.setObjectName("SecondaryActionButton")
            plot_muscles_button.clicked.connect(self._plot_muscles)
            validation_layout.addWidget(graph_model_button)
            validation_layout.addWidget(plot_muscles_button)
            validation_layout.addWidget(self.validation_messages)
            tabs.addTab(validation_tab, "Validation")

            right_layout = QVBoxLayout(right_panel)
            _configure_panel_layout(right_layout, margin=0)
            right_layout.addWidget(tabs)

            splitter = QSplitter(qt_horizontal)
            splitter.addWidget(self.tree)
            splitter.addWidget(right_panel)
            splitter.setSizes([350, 750])

            open_button = QPushButton("Open model")
            open_button.setObjectName("SecondaryActionButton")
            open_button.clicked.connect(self._open_model)
            new_c3d_model_button = QPushButton("New from C3D")
            new_c3d_model_button.setObjectName("PrimaryActionButton")
            new_c3d_model_button.clicked.connect(lambda: self._new_model_from_c3d())
            save_button = QPushButton("Export model")
            save_button.setObjectName("PrimaryActionButton")
            save_button.clicked.connect(self._save_model)

            toolbar = QHBoxLayout()
            _configure_panel_layout(toolbar, margin=0, spacing=8)
            toolbar.addWidget(open_button)
            toolbar.addWidget(new_c3d_model_button)
            toolbar.addWidget(save_button)
            toolbar.addStretch()

            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
            _configure_panel_layout(layout)
            layout.addLayout(toolbar)
            layout.addWidget(splitter)
            self.setCentralWidget(central_widget)

        def _resize_to_available_screen(self, application) -> None:
            """
            Keep the editor inside the usable screen area.
            """
            screen = self.screen() or application.primaryScreen()
            if screen is None:
                self.resize(1100, 700)
                return
            available_geometry = screen.availableGeometry()
            width = min(1100, int(available_geometry.width() * 0.9))
            height = min(700, int(available_geometry.height() * 0.9))
            self.resize(width, height)
            self.move(
                available_geometry.x() + (available_geometry.width() - width) // 2,
                available_geometry.y() + (available_geometry.height() - height) // 2,
            )

        def _component_row(self, fields: dict[str, QLineEdit]):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            for axis in ("x", "y", "z"):
                layout.addWidget(QLabel(axis))
                fields[axis].setPlaceholderText(axis)
                layout.addWidget(fields[axis])
            return widget

        def _inertia_grid(self):
            labels = (
                ("Ixx", "Ixy", "Ixz"),
                ("Iyx", "Iyy", "Iyz"),
                ("Izx", "Izy", "Izz"),
            )
            widget = QWidget()
            layout = QGridLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            for row in range(3):
                for column in range(3):
                    field = self.inertia_fields[(row, column)]
                    field.setPlaceholderText(labels[row][column])
                    layout.addWidget(QLabel(labels[row][column]), row, column * 2)
                    layout.addWidget(field, row, column * 2 + 1)
            return widget

        def _update_inertia_component_state(self) -> None:
            principal_moments = self.inertia_mode.currentText() == "Principal moments"
            for (row, column), field in self.inertia_fields.items():
                is_diagonal = row == column
                field.setEnabled(is_diagonal or not principal_moments)
                if principal_moments and not is_diagonal:
                    field.setText("0")

        def _set_component_fields(self, fields: dict[str, QLineEdit], values: list[float]) -> None:
            for axis, value in zip(("x", "y", "z"), values):
                fields[axis].setText(str(value))

        def _read_component_fields(self, fields: dict[str, QLineEdit]) -> list[float]:
            return [_parse_optional_float(fields[axis].text()) or 0.0 for axis in ("x", "y", "z")]

        def _set_inertia_matrix_fields(self, inertia_matrix: list[list[float]]) -> None:
            for row in range(3):
                for column in range(3):
                    self.inertia_fields[(row, column)].setText(str(inertia_matrix[row][column]))

        def _read_inertia_matrix_fields(self) -> list[list[float]]:
            return [
                [_parse_optional_float(self.inertia_fields[(row, column)].text()) or 0.0 for column in range(3)]
                for row in range(3)
            ]

        def _open_model(self) -> None:
            filepath, _ = QFileDialog.getOpenFileName(
                self,
                "Open model",
                "",
                "Biomechanical models (*.bioMod *.osim *.urdf *.bvh)",
            )
            if not filepath:
                return
            try:
                self.model = load_model(filepath)
                self.current_filepath = Path(filepath)
                self._refresh_model_views()
            except Exception as error:
                QMessageBox.critical(self, "Unable to open model", str(error))

        def _apply_inertial_model(self) -> None:
            if self.model is None or self.current_segment_name is None:
                QMessageBox.information(self, "No segment", "Open a model and select a segment first.")
                return
            dialog = InertialModelDialog(self)
            result = dialog.exec() if hasattr(dialog, "exec") else dialog.exec_()
            if result != qdialog_accepted:
                return
            model_name, source_segment_name, model_parameters = dialog.parameters()
            try:
                inertia_parameters = build_inertial_parameters_from_model(
                    model_name=model_name,
                    segment_name=source_segment_name,
                    model_parameters=model_parameters,
                )
                self.model.segments[self.current_segment_name].inertia_parameters = inertia_parameters
                data = get_segment_editor_data(self.model.segments[self.current_segment_name])
                self.mass.setText("" if data.mass is None else str(data.mass))
                self._set_component_fields(self.center_of_mass_fields, data.center_of_mass)
                self.inertia_mode.setCurrentText("Inertia matrix")
                self._set_inertia_matrix_fields(data.inertia_matrix)
                self.preview.set_model(self.model)
            except Exception as error:
                QMessageBox.critical(self, "Unable to apply inertial model", str(error))

        def _new_model_from_c3d(
            self,
            *,
            initial_preset: str | C3dModelPreset | None = None,
            initial_c3d_folder: str | Path | None = None,
        ) -> None:
            dialog = C3dModelCreationDialog(
                initial_preset=initial_preset,
                initial_c3d_folder=initial_c3d_folder,
            )
            if _exec_dialog(dialog) != _dialog_accepted_value():
                return
            if dialog.selected_preset() == C3dModelPreset.FROM_SCRATCH:
                QMessageBox.information(
                    self,
                    "Template-free C3D draft",
                    "The template-free C3D workflow is ready for drafting in the dialog. Use 'Generate template' "
                    "after adding segments, markers, axes, DoFs, and virtual markers; direct BioMod generation "
                    "will be enabled once the draft can be converted to a complete model template.",
                )
                return
            selected_c3d_file = dialog.selected_c3d_file()
            if selected_c3d_file is not None:
                self._new_model_from_c3d_file(selected_c3d_file, dialog.selected_preset())
                return
            selected_c3d_folder = dialog.selected_c3d_folder()
            if selected_c3d_folder is None:
                calibration_folder = QFileDialog.getExistingDirectory(
                    self,
                    "Select calibration folder",
                    "",
                )
                if not calibration_folder:
                    return
                folder_path = Path(calibration_folder)
            else:
                folder_path = selected_c3d_folder
            try:
                progress_dialog, progress_callback = self._c3d_folder_generation_progress_reporter()
                result = create_model_from_c3d_folder(
                    calibration_folder=folder_path,
                    preset=dialog.selected_preset(),
                    static_virtual_points=_static_virtual_point_definitions_from_draft(dialog.workflow_draft),
                    progress_callback=progress_callback,
                )
                progress_callback("Refreshing model editor...")
                self.model = result.model
                self.current_filepath = folder_path / result.output_filename
                self._refresh_model_views()
                progress_dialog.close()
                QMessageBox.information(
                    self,
                    "Model generated from C3D",
                    _format_c3d_creation_summary(result, folder_path),
                )
            except Exception as error:
                if "progress_dialog" in locals():
                    progress_dialog.close()
                QMessageBox.critical(self, "Unable to generate model", str(error))

        def _c3d_folder_generation_progress_reporter(self):
            """
            Create a non-cancelable progress popup for C3D folder model generation.
            """
            progress_dialog = QProgressDialog(
                "Preparing C3D model generation...",
                None,
                0,
                0,
                self,
            )
            progress_dialog.setWindowTitle("Generating model from C3D")
            progress_dialog.setWindowModality(qt_window_modal)
            progress_dialog.setCancelButton(None)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.show()

            def progress_callback(message: str) -> None:
                progress_dialog.setLabelText(message)
                progress_dialog.show()
                QApplication.processEvents()

            progress_callback("Preparing C3D model generation...")
            return progress_dialog, progress_callback

        def _new_model_from_c3d_file(self, filepath: Path, preset: C3dModelPreset) -> None:
            try:
                template = template_for_c3d_model_preset(preset)
                result = create_model_from_marker_data(
                    template=template,
                    static_data=C3dData(str(filepath)),
                    preset=preset,
                )
                self.model = result.model
                self.current_filepath = filepath.with_suffix(".bioMod")
                self._refresh_model_views()
                QMessageBox.information(
                    self,
                    "Model generated from C3D",
                    _format_c3d_creation_summary(result, filepath.parent),
                )
            except Exception as error:
                QMessageBox.critical(self, "Unable to generate model", str(error))

        def _save_model(self) -> None:
            if self.model is None:
                QMessageBox.information(self, "No model", "Open a model before saving.")
                return
            default_name = "" if self.current_filepath is None else str(self.current_filepath.with_suffix(".bioMod"))
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Export model",
                default_name,
                "Supported models (*.bioMod *.osim *.urdf *.bvh);;BioMod files (*.bioMod);;OpenSim files (*.osim);;URDF files (*.urdf);;BVH files (*.bvh)",
            )
            if not filepath:
                return
            try:
                _export_model_to_path(self.model, filepath)
            except Exception as error:
                QMessageBox.critical(self, "Unable to export model", str(error))

        def _write_model_graphviz(self) -> None:
            if self.model is None:
                QMessageBox.information(self, "No model", "Open a model before writing a graph.")
                return
            default_name = "" if self.current_filepath is None else str(self.current_filepath.with_suffix(".dot"))
            filepath, _ = QFileDialog.getSaveFileName(self, "Write model graph", default_name, "Graphviz DOT (*.dot)")
            if not filepath:
                return
            try:
                self.model.write_graphviz(filepath)
            except Exception as error:
                QMessageBox.critical(self, "Unable to write model graph", str(error))

        def _plot_muscles(self) -> None:
            if self.model is None:
                QMessageBox.information(self, "No model", "Open a model before plotting muscles.")
                return
            try:
                muscle_validator = MuscleValidator(self.model)
                muscle_validator.plot_force_length()
                muscle_validator.plot_moment_arm()
                muscle_validator.plot_torques()
            except Exception as error:
                QMessageBox.critical(self, "Unable to plot muscles", str(error))

        def _refresh_model_views(self) -> None:
            """
            Refresh all widgets that mirror the current model.
            """
            self.current_segment_name = None
            self._populate_tree()
            self._populate_muscle_tree()
            self._populate_marker_list()
            self.validation_messages.clear()
            self.preview.set_selected_segment(None)
            self.preview.set_model(self.model)

        def _validate_model(self) -> None:
            self.validation_messages.clear()
            if self.model is None:
                self.validation_messages.addItem("Open a model before validation.")
                return
            report = validate_model_for_editor(self.model)
            self.validation_messages.addItem(f"[{report.category}]")
            self.validation_messages.addItems(report.messages)

        def _populate_tree(self) -> None:
            self.tree.clear()
            if self.model is None:
                return

            items = {}
            for segment in self.model.segments:
                item = QTreeWidgetItem([segment.name])
                items[segment.name] = item
                if segment.parent_name in items:
                    items[segment.parent_name].addChild(item)
                else:
                    self.tree.addTopLevelItem(item)
            self.tree.expandAll()

        def _select_segment_from_preview(self, segment_name: str) -> None:
            items = self.tree.findItems(segment_name, qt_match_recursive | qt_match_exact)
            if items:
                self.tree.setCurrentItem(items[0])

        def _select_marker_from_preview(self, marker_name: str) -> None:
            if self.model is None:
                return
            for segment in self.model.segments:
                if marker_name in segment.markers.keys():
                    self._select_segment_from_preview(segment.name)
                    items = self.marker_list.findItems(marker_name, qt_match_exact)
                    if items:
                        self.marker_list.setCurrentItem(items[0])
                    return

        def _on_segment_selection_changed(self) -> None:
            if self.model is None or not self.tree.selectedItems():
                return
            self.current_segment_name = self.tree.selectedItems()[0].text(0)
            segment = self.model.segments[self.current_segment_name]
            data = get_segment_editor_data(segment)
            self.parent_name.setText(data.parent_name)
            self.translations.setText(data.translations)
            self.rotations.setText(data.rotations)
            self.q_min.setText(_format_float_list(data.q_min))
            self.q_max.setText(_format_float_list(data.q_max))
            self.mass.setText("" if data.mass is None else str(data.mass))
            self._set_component_fields(self.center_of_mass_fields, data.center_of_mass)
            if _has_off_diagonal_inertia(data.inertia_matrix):
                self.inertia_mode.setCurrentText("Inertia matrix")
            else:
                self.inertia_mode.setCurrentText("Principal moments")
            self._set_inertia_matrix_fields(data.inertia_matrix)
            self._update_inertia_component_state()
            self._populate_marker_list()
            self.preview.set_selected_segment(self.current_segment_name)

        def _populate_marker_list(self) -> None:
            self.marker_list.clear()
            if self.model is None or self.current_segment_name is None:
                return
            segment = self.model.segments[self.current_segment_name]
            self.marker_list.addItems(list(segment.markers.keys()))

        def _on_marker_selection_changed(self) -> None:
            if self.model is None or self.current_segment_name is None or not self.marker_list.selectedItems():
                return
            marker_name = self.marker_list.selectedItems()[0].text()
            marker = self.model.segments[self.current_segment_name].markers[marker_name]
            data = get_marker_editor_data(marker)
            self.marker_name.setText(data.name)
            self.marker_position.setText(_format_float_list(data.position))
            self.marker_technical.setChecked(data.is_technical)
            self.marker_anatomical.setChecked(data.is_anatomical)

        def _marker_data_from_form(self) -> MarkerEditorData:
            return MarkerEditorData(
                name=self.marker_name.text().strip(),
                position=_parse_vector(self.marker_position.text(), expected_length=3),
                is_technical=self.marker_technical.isChecked(),
                is_anatomical=self.marker_anatomical.isChecked(),
            )

        def _apply_marker_changes(self) -> None:
            if self.model is None or self.current_segment_name is None or not self.marker_list.selectedItems():
                return
            try:
                old_name = self.marker_list.selectedItems()[0].text()
                segment = self.model.segments[self.current_segment_name]
                marker = segment.markers[old_name]
                data = self._marker_data_from_form()
                if old_name != data.name:
                    segment.markers._remove(old_name)
                apply_marker_editor_data(marker, data)
                if old_name != data.name:
                    segment.markers._append(marker)
                self._populate_marker_list()
                self.preview.set_model(self.model)
            except Exception as error:
                QMessageBox.critical(self, "Invalid marker values", str(error))

        def _add_marker(self) -> None:
            if self.model is None or self.current_segment_name is None:
                return
            try:
                add_marker(
                    self.model.segments[self.current_segment_name],
                    self._marker_data_from_form(),
                )
                self._populate_marker_list()
                self.preview.set_model(self.model)
            except Exception as error:
                QMessageBox.critical(self, "Unable to add marker", str(error))

        def _remove_marker(self) -> None:
            if self.model is None or self.current_segment_name is None or not self.marker_list.selectedItems():
                return
            marker_name = self.marker_list.selectedItems()[0].text()
            remove_marker(self.model.segments[self.current_segment_name], marker_name)
            self._populate_marker_list()
            self.preview.set_model(self.model)

        def _attach_marker_to_segment(self) -> None:
            if self.model is None or self.current_segment_name is None or not self.marker_list.selectedItems():
                return
            try:
                marker_name = self.marker_list.selectedItems()[0].text()
                target_segment_name = self.marker_target_segment.text().strip()
                attach_marker_to_segment(
                    model=self.model,
                    source_segment_name=self.current_segment_name,
                    marker_name=marker_name,
                    target_segment_name=target_segment_name,
                )
                self.preview.set_model(self.model)
            except Exception as error:
                QMessageBox.critical(self, "Unable to attach marker", str(error))

        def _populate_muscle_tree(self) -> None:
            self.muscle_tree.clear()
            if self.model is None:
                return
            for muscle_group in self.model.muscle_groups:
                group_item = QTreeWidgetItem([muscle_group.name])
                for muscle in muscle_group.muscles:
                    group_item.addChild(QTreeWidgetItem([muscle.name]))
                self.muscle_tree.addTopLevelItem(group_item)
            self.muscle_tree.expandAll()

        def _selected_muscle(self):
            if self.model is None or not self.muscle_tree.selectedItems():
                return None
            item = self.muscle_tree.selectedItems()[0]
            if item.parent() is None:
                return None
            muscle_group_name = item.parent().text(0)
            muscle_name = item.text(0)
            return self.model.muscle_groups[muscle_group_name].muscles[muscle_name]

        def _selected_muscle_group(self):
            if self.model is None or not self.muscle_tree.selectedItems():
                return None
            item = self.muscle_tree.selectedItems()[0]
            muscle_group_name = item.text(0) if item.parent() is None else item.parent().text(0)
            return self.model.muscle_groups[muscle_group_name]

        def _on_muscle_selection_changed(self) -> None:
            muscle = self._selected_muscle()
            if muscle is None:
                return
            data = get_muscle_editor_data(muscle)
            self.optimal_length.setText(_format_optional_float(data.optimal_length))
            self.maximal_force.setText(_format_optional_float(data.maximal_force))
            self.tendon_slack_length.setText(_format_optional_float(data.tendon_slack_length))
            self.pennation_angle.setText(_format_optional_float(data.pennation_angle))
            self.maximal_velocity.setText(_format_optional_float(data.maximal_velocity))
            self.maximal_excitation.setText(_format_optional_float(data.maximal_excitation))
            origin = get_origin_editor_data(muscle)
            insertion = get_insertion_editor_data(muscle)
            self.origin_name.setText(origin.name)
            self.origin_parent.setText(origin.parent_name)
            self.origin_position.setText(_format_float_list(origin.position))
            self.insertion_name.setText(insertion.name)
            self.insertion_parent.setText(insertion.parent_name)
            self.insertion_position.setText(_format_float_list(insertion.position))
            self._populate_via_point_list()

        def _apply_muscle_changes(self) -> None:
            muscle = self._selected_muscle()
            if muscle is None:
                return
            try:
                apply_muscle_editor_data(
                    muscle,
                    MuscleEditorData(
                        optimal_length=_parse_optional_float(self.optimal_length.text()),
                        maximal_force=_parse_optional_float(self.maximal_force.text()),
                        tendon_slack_length=_parse_optional_float(self.tendon_slack_length.text()),
                        pennation_angle=_parse_optional_float(self.pennation_angle.text()),
                        maximal_velocity=_parse_optional_float(self.maximal_velocity.text()),
                        maximal_excitation=_parse_optional_float(self.maximal_excitation.text()),
                    ),
                )
                self.preview.set_model(self.model)
            except Exception as error:
                QMessageBox.critical(self, "Invalid muscle values", str(error))

        def _add_muscle_group(self) -> None:
            if self.model is None:
                return
            try:
                add_muscle_group(
                    self.model,
                    self.group_name.text().strip(),
                    self.group_origin_parent.text().strip(),
                    self.group_insertion_parent.text().strip(),
                )
                self._populate_muscle_tree()
                self.preview.set_model(self.model)
            except Exception as error:
                QMessageBox.critical(self, "Unable to add muscle group", str(error))

        def _remove_muscle_group(self) -> None:
            muscle_group = self._selected_muscle_group()
            if muscle_group is None:
                return
            remove_muscle_group(self.model, muscle_group.name)
            self._populate_muscle_tree()
            self.preview.set_model(self.model)

        def _add_muscle(self) -> None:
            muscle_group = self._selected_muscle_group()
            if muscle_group is None:
                return
            try:
                add_muscle(muscle_group, self.new_muscle_name.text().strip())
                self._populate_muscle_tree()
                self.preview.set_model(self.model)
            except Exception as error:
                QMessageBox.critical(self, "Unable to add muscle", str(error))

        def _remove_muscle(self) -> None:
            muscle = self._selected_muscle()
            if muscle is None:
                return
            muscle_group = self.model.muscle_groups[muscle.muscle_group]
            remove_muscle(muscle_group, muscle.name)
            self._populate_muscle_tree()
            self.preview.set_model(self.model)

        def _populate_via_point_list(self) -> None:
            self.via_point_list.clear()
            muscle = self._selected_muscle()
            if muscle is None:
                return
            self.via_point_list.addItems(list(muscle.via_points.keys()))

        def _apply_path_endpoint_changes(self) -> None:
            muscle = self._selected_muscle()
            if muscle is None:
                return
            try:
                apply_origin_editor_data(
                    muscle,
                    ViaPointEditorData(
                        name=self.origin_name.text().strip(),
                        parent_name=self.origin_parent.text().strip(),
                        position=_parse_vector(self.origin_position.text(), expected_length=3),
                    ),
                )
                apply_insertion_editor_data(
                    muscle,
                    ViaPointEditorData(
                        name=self.insertion_name.text().strip(),
                        parent_name=self.insertion_parent.text().strip(),
                        position=_parse_vector(self.insertion_position.text(), expected_length=3),
                    ),
                )
                self.preview.set_model(self.model)
            except Exception as error:
                QMessageBox.critical(self, "Invalid origin/insertion values", str(error))

        def _on_via_point_selection_changed(self) -> None:
            muscle = self._selected_muscle()
            if muscle is None or not self.via_point_list.selectedItems():
                return
            via_point_name = self.via_point_list.selectedItems()[0].text()
            data = get_via_point_editor_data(muscle.via_points[via_point_name])
            self.via_point_name.setText(data.name)
            self.via_point_parent.setText(data.parent_name)
            self.via_point_position.setText(_format_float_list(data.position))

        def _via_point_data_from_form(self) -> ViaPointEditorData:
            return ViaPointEditorData(
                name=self.via_point_name.text().strip(),
                parent_name=self.via_point_parent.text().strip(),
                position=_parse_vector(self.via_point_position.text(), expected_length=3),
            )

        def _apply_via_point_changes(self) -> None:
            muscle = self._selected_muscle()
            if muscle is None or not self.via_point_list.selectedItems():
                return
            try:
                old_name = self.via_point_list.selectedItems()[0].text()
                via_point = muscle.via_points[old_name]
                data = self._via_point_data_from_form()
                if old_name != data.name:
                    muscle.via_points._remove(old_name)
                apply_via_point_editor_data(via_point, data)
                if old_name != data.name:
                    muscle.via_points._append(via_point)
                self._populate_via_point_list()
                self.preview.set_model(self.model)
            except Exception as error:
                QMessageBox.critical(self, "Invalid via-point values", str(error))

        def _add_via_point(self) -> None:
            muscle = self._selected_muscle()
            if muscle is None:
                return
            try:
                add_via_point(muscle, self._via_point_data_from_form())
                self._populate_via_point_list()
                self.preview.set_model(self.model)
            except Exception as error:
                QMessageBox.critical(self, "Unable to add via point", str(error))

        def _remove_via_point(self) -> None:
            muscle = self._selected_muscle()
            if muscle is None or not self.via_point_list.selectedItems():
                return
            remove_via_point(muscle, self.via_point_list.selectedItems()[0].text())
            self._populate_via_point_list()
            self.preview.set_model(self.model)

        def _apply_segment_changes(self) -> None:
            if self.model is None or self.current_segment_name is None:
                return
            try:
                data = SegmentEditorData(
                    parent_name=self.parent_name.text().strip(),
                    translations=self.translations.text().strip().lower(),
                    rotations=self.rotations.text().strip().lower(),
                    q_min=_parse_float_list(self.q_min.text()),
                    q_max=_parse_float_list(self.q_max.text()),
                    mass=_parse_optional_float(self.mass.text()),
                    center_of_mass=self._read_component_fields(self.center_of_mass_fields),
                    inertia_matrix=self._read_inertia_matrix_fields(),
                )
                validate_parent_name(
                    model=self.model,
                    segment_name=self.current_segment_name,
                    parent_name=data.parent_name,
                )
                apply_segment_editor_data(self.model.segments[self.current_segment_name], data)
                self._populate_tree()
                self.preview.set_model(self.model)
            except Exception as error:
                QMessageBox.critical(self, "Invalid segment values", str(error))

    app = QApplication.instance() or QApplication([])
    apply_biobuddy_gui_style(app)
    window = ModelEditorWindow()
    window.show()
    if open_c3d_dialog or c3d_preset is not None or c3d_folder is not None:

        def open_startup_c3d_dialog() -> None:
            try:
                window._new_model_from_c3d(
                    initial_preset=c3d_preset,
                    initial_c3d_folder=c3d_folder,
                )
            except Exception as error:
                QMessageBox.critical(window, "Unable to open C3D model workflow", str(error))

        QTimer.singleShot(0, open_startup_c3d_dialog)
    app.exec()


def _c3d_preset_label(preset: C3dModelPreset) -> str:
    """
    Return the label shown in the C3D creation dialog.
    """
    if preset == C3dModelPreset.LOWER_LIMBS:
        return "Lower-limbs & trunk (with functional trials)"
    if preset == C3dModelPreset.LOWER_LIMBS_ANATOMICAL:
        return "Lower-limbs & trunk"
    if preset == C3dModelPreset.FULL_BODY:
        return "Full body"
    if preset == C3dModelPreset.MOTIVE_57:
        return "BioBuddy Motive (57)"
    if preset == C3dModelPreset.MOTIVE_57_ISB:
        return "BioBuddy Motive (57) ISB"
    if preset == C3dModelPreset.UPPER_LIMB:
        return "Upper-limb"
    if preset == C3dModelPreset.FROM_SCRATCH:
        return "From scratch"
    return preset.value


def _c3d_model_preset_from_cli_value(
    value: str | C3dModelPreset | None,
) -> C3dModelPreset | None:
    """
    Resolve a command-line preset value to a C3D model preset.
    """
    if value is None or isinstance(value, C3dModelPreset):
        return value
    normalized_value = value.strip().lower().replace("_", "-")
    aliases = {
        "from-scratch": C3dModelPreset.FROM_SCRATCH,
        "full-body": C3dModelPreset.FULL_BODY,
        "motive-57": C3dModelPreset.MOTIVE_57,
        "biobuddy-motive-57": C3dModelPreset.MOTIVE_57,
        "biomech-motive-57": C3dModelPreset.MOTIVE_57,
        "biomech-motive": C3dModelPreset.MOTIVE_57,
        "motive-57-isb": C3dModelPreset.MOTIVE_57_ISB,
        "biobuddy-motive-57-isb": C3dModelPreset.MOTIVE_57_ISB,
        "biomech-motive-57-isb": C3dModelPreset.MOTIVE_57_ISB,
        "lower-limbs": C3dModelPreset.LOWER_LIMBS_ANATOMICAL,
        "lower-limbs-trunk": C3dModelPreset.LOWER_LIMBS_ANATOMICAL,
        "lower-limbs-and-trunk": C3dModelPreset.LOWER_LIMBS_ANATOMICAL,
        "lower-limbs-functional": C3dModelPreset.LOWER_LIMBS,
        "lower-limbs-trunk-functional": C3dModelPreset.LOWER_LIMBS,
        "lower-limbs-and-trunk-functional": C3dModelPreset.LOWER_LIMBS,
        "lower-limbs-with-functional-trials": C3dModelPreset.LOWER_LIMBS,
        "lower-limbs-with-functional-cor": C3dModelPreset.LOWER_LIMBS,
        "upper-limb": C3dModelPreset.UPPER_LIMB,
    }
    if normalized_value in aliases:
        return aliases[normalized_value]
    for preset in C3dModelPreset:
        if normalized_value == preset.value.lower().replace("_", "-"):
            return preset
    available_values = ", ".join(sorted(aliases))
    raise ValueError(f"Unsupported C3D model preset '{value}'. Available aliases: {available_values}.")


def _format_c3d_creation_summary(result: C3dModelCreationResult, calibration_folder: Path) -> str:
    """
    Build a compact GUI summary for a generated C3D model.
    """
    marker_lines = []
    for trial_name, report in result.marker_reports.items():
        missing = "none" if len(report.missing_markers) == 0 else ", ".join(report.missing_markers)
        marker_lines.append(
            f"{trial_name}: {report.complete_frame_count}/{report.total_frame_count} complete frames, "
            f"missing markers: {missing}"
        )
    quality_lines = [
        f"{name}: raw plane angle {metric.mean_angle_degrees:.1f} deg" for name, metric in result.frame_quality.items()
    ]
    return (
        f"Generated {len(result.model.segments)} segments from '{calibration_folder}'.\n\n"
        "Marker availability\n"
        f"{chr(10).join(marker_lines)}\n\n"
        "Frame quality\n"
        f"{chr(10).join(quality_lines)}"
    )


def _parse_float_list(text: str) -> list[float]:
    """
    Parse a comma- or space-separated float list from a line edit.
    """
    stripped_text = text.strip()
    if stripped_text == "":
        return []
    return [float(value) for value in stripped_text.replace(",", " ").split()]


def _parse_rotation_matrix_text(text: str) -> tuple[tuple[float, float, float], ...]:
    """
    Parse a 3x3 rotation matrix from a compact line edit.
    """
    stripped_text = text.strip()
    if stripped_text == "":
        raise ValueError("The matrix field is empty.")
    try:
        parsed = ast.literal_eval(stripped_text)
    except (SyntaxError, ValueError):
        parsed = _parse_rotation_matrix_rows(stripped_text)
    matrix = _coerce_rotation_matrix(parsed)
    for row in matrix:
        for value in row:
            if not math.isfinite(value):
                raise ValueError("The rotation matrix must contain only finite numbers.")
    return matrix


def _parse_rotation_matrix_rows(text: str) -> list[list[float]]:
    row_texts = [row.strip() for row in re.split(r"[;\n]+", text) if row.strip()]
    if len(row_texts) == 1:
        values = _parse_float_list(row_texts[0])
        if len(values) == 9:
            return [values[0:3], values[3:6], values[6:9]]
        raise ValueError(
            "The rotation matrix must contain exactly 9 numbers, or 3 rows of 3 numbers separated by ';' or new lines."
        )
    return [_parse_float_list(row_text) for row_text in row_texts]


def _coerce_rotation_matrix(value) -> tuple[tuple[float, float, float], ...]:
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError("The rotation matrix must be a 3x3 list, for example [[1,0,0], [0,1,0], [0,0,1]].")
    if len(value) == 9 and all(not isinstance(item, (list, tuple)) for item in value):
        value = [value[0:3], value[3:6], value[6:9]]
    if len(value) != 3:
        raise ValueError("The rotation matrix must have exactly 3 rows.")
    rows = []
    for row in value:
        if isinstance(row, tuple):
            row = list(row)
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("Each rotation matrix row must contain exactly 3 numbers.")
        try:
            rows.append(tuple(float(item) for item in row))
        except (TypeError, ValueError) as error:
            raise ValueError("The rotation matrix must contain only numeric values.") from error
    return tuple(rows)


def _format_rotation_matrix(matrix: tuple[tuple[float, float, float], ...]) -> str:
    return "[" + ", ".join("[" + ", ".join(f"{value:.12g}" for value in row) + "]" for row in matrix) + "]"


def _chain_setting_preview_signature(setting) -> tuple[object, ...]:
    """
    Return the setting fields that affect the q0 chain preview.
    """
    return (
        setting.translations,
        setting.rotations,
        setting.child_translation,
        setting.initial_rotation_method,
        setting.initial_rotation_source,
        tuple(tuple(row) for row in setting.initial_rotation_matrix),
    )


def _q0_initial_rotation_by_segment(workflow_draft) -> dict[str, np.ndarray]:
    """
    Return local initial rotations used by the q0 preview.
    """
    rotations = {}
    for setting in workflow_draft.segment_settings:
        if setting.initial_rotation_method == "identity":
            rotations[setting.segment_name] = np.eye(3)
        else:
            rotations[setting.segment_name] = np.asarray(setting.initial_rotation_matrix, dtype=float).reshape(3, 3)
    return rotations


def _initial_rotation_matrix_from_setting(setting) -> np.ndarray:
    """
    Return the local RT rotation requested by a chain setting.
    """
    if setting.initial_rotation_method == "identity":
        return np.eye(3)
    return np.asarray(setting.initial_rotation_matrix, dtype=float).reshape(3, 3)


def _apply_initial_rotation_setting_to_segment(segment, setting, workflow_draft=None) -> None:
    """
    Apply the requested initial rotation to a generated model segment while keeping its translation.
    """
    if getattr(segment, "segment_coordinate_system", None) is None:
        return
    if setting.initial_rotation_method == "identity" and (
        _segment_uses_aor_axis(workflow_draft, setting.segment_name)
        or _segment_is_anatomical_child_of_joint(workflow_draft, setting.segment_name)
    ):
        return
    segment.segment_coordinate_system.scs.rotation_matrix = _initial_rotation_matrix_from_setting(setting)


def _segment_uses_aor_axis(workflow_draft, segment_name: str) -> bool:
    """
    Return whether a segment frame is constrained by a functional AoR/SARA axis.
    """
    if workflow_draft is None:
        return False
    return any(
        axis.segment_name == segment_name and _is_sara_direction_method(axis.method)
        for axis in getattr(workflow_draft, "axes", ())
    )


def _segment_is_anatomical_child_of_joint(workflow_draft, segment_name: str) -> bool:
    """Return whether a segment stores the fixed anatomical transform after a joint frame."""
    if workflow_draft is None:
        return False
    groups = tuple(getattr(workflow_draft, "segment_marker_groups", ()))
    group_by_name = {group.segment_name: group for group in groups}
    group = group_by_name.get(segment_name)
    parent = group_by_name.get(group.parent_name) if group is not None else None
    return parent is not None and parent.segment_type == "joint"


def _safe_float_count(text: str) -> int:
    """
    Return the number of numeric entries, or a sentinel count when parsing fails.
    """
    try:
        return len(_parse_float_list(text))
    except ValueError:
        return -1


def _list_widget_texts(list_widget) -> tuple[str, ...]:
    """
    Return all marker names shown in a QListWidget, preserving duplicates.
    """
    return tuple(
        list_widget.item(index).text().split("|", maxsplit=1)[0].strip() for index in range(list_widget.count())
    )


def _segment_marker_group_label(group) -> str:
    markers = ", ".join(group.marker_names) if len(group.marker_names) != 0 else "no marker assigned yet"
    parent = group.parent_name if group.parent_name else "-"
    return f"{group.segment_name}: {markers} | type={group.segment_type} | parent={parent}"


def _marker_pool_from_draft(workflow_draft) -> tuple[str, ...]:
    """
    Return the known marker names for a C3D draft, even before a C3D file is loaded.
    """
    marker_names = []
    marker_names.extend(sorted(_expected_marker_names_for_preset(workflow_draft.preset)))
    for group in workflow_draft.segment_marker_groups:
        marker_names.extend(group.marker_names)
    marker_names.extend(marker.name for marker in workflow_draft.virtual_markers)
    return tuple(dict.fromkeys(marker_names))


def _unassigned_marker_names(
    marker_names: tuple[str, ...], segment_marker_groups: tuple[object, ...]
) -> tuple[str, ...]:
    """
    Return known marker names that are intentionally not attached to any segment.
    """
    assigned_marker_names = {
        marker_name
        for group in segment_marker_groups
        for marker_name in group.marker_names + group.technical_marker_names
    }
    return tuple(marker_name for marker_name in marker_names if marker_name not in assigned_marker_names)


def _virtual_marker_preview_marker_names_to_draw(
    marker_names: tuple[str, ...],
    highlighted_marker_names: set[str],
    show_whole_body: bool,
    is_dragging: bool,
) -> set[str]:
    """
    Return marker names for the virtual-marker preview without changing whole-body fit.
    """
    if show_whole_body or not is_dragging:
        return set(marker_names)
    return set(highlighted_marker_names)


def _marker_name_mapping_for_c3d(
    template_marker_names: tuple[str, ...], c3d_marker_names: tuple[str, ...]
) -> dict[str, str]:
    """
    Match template marker names to loaded C3D marker names using exact and normalized names.
    """
    exact_names = set(c3d_marker_names)
    normalized_to_c3d_name = {}
    for marker_name in c3d_marker_names:
        for normalized in _normalized_marker_name_candidates(marker_name):
            if normalized not in normalized_to_c3d_name:
                normalized_to_c3d_name[normalized] = marker_name

    mapping = {}
    for template_name in template_marker_names:
        if template_name in exact_names:
            mapping[template_name] = template_name
            continue
        for normalized in _normalized_marker_name_candidates(template_name):
            if normalized in normalized_to_c3d_name:
                mapping[template_name] = normalized_to_c3d_name[normalized]
                break
    return mapping


def _normalized_marker_name_candidates(marker_name: str) -> tuple[str, ...]:
    """
    Return normalized marker-name variants, including the suffix after a C3D namespace separator.
    """
    candidates = [marker_name]
    if ":" in marker_name:
        candidates.append(marker_name.split(":")[-1])
    if "_" in marker_name:
        candidates.append(marker_name.rsplit("_", maxsplit=1)[-1])
    return tuple(dict.fromkeys(_normalized_marker_name(candidate) for candidate in candidates))


def _normalized_marker_name(marker_name: str) -> str:
    """
    Normalize marker names for template-to-C3D matching.
    """
    return "".join(character for character in marker_name.upper() if character.isalnum())


def _strip_participant_prefix_from_c3d_data(c3d_data) -> None:
    """
    Remove a participant namespace prefix such as 'P01_MH:' from C3D marker names in place.
    """
    stripped_names = _strip_participant_prefix_from_marker_names(tuple(c3d_data.marker_names))
    if len(set(stripped_names)) != len(stripped_names):
        return
    c3d_data.marker_names = list(stripped_names)


def _strip_participant_prefix_from_marker_names(
    marker_names: tuple[str, ...],
) -> tuple[str, ...]:
    """
    Strip the text before ':' from marker names, preserving names without a separator.
    """
    stripped_names = []
    for marker_name in marker_names:
        if ":" in marker_name:
            stripped_names.append(marker_name.split(":", maxsplit=1)[1])
            continue
        stripped_names.append(re.sub(r"^Skeleton_\d+_", "", marker_name))
    return tuple(stripped_names)


def _remap_c3d_workflow_draft_markers(workflow_draft, marker_mapping: dict[str, str]):
    """
    Replace template marker names by their loaded C3D equivalents in editable draft fields.
    """

    def remap_names(marker_names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(marker_mapping.get(marker_name, marker_name) for marker_name in marker_names)

    return replace(
        workflow_draft,
        segment_marker_groups=tuple(
            replace(
                group,
                marker_names=remap_names(group.marker_names),
                technical_marker_names=remap_names(group.technical_marker_names),
            )
            for group in workflow_draft.segment_marker_groups
        ),
        axes=tuple(
            replace(
                axis,
                start_markers=remap_names(axis.start_markers),
                end_markers=remap_names(axis.end_markers),
                origin_markers=remap_names(axis.origin_markers),
            )
            for axis in workflow_draft.axes
        ),
    )


def _format_marker_mapping_summary(marker_mapping: dict[str, str]) -> str:
    """
    Summarize automatic marker-name matching for the technical segment tab.
    """
    remapped = sorted(
        f"{template_name} -> {c3d_name}"
        for template_name, c3d_name in marker_mapping.items()
        if template_name != c3d_name
    )
    if not marker_mapping:
        return "No template marker matched the loaded C3D yet. Assign markers manually or update the template names."
    if not remapped:
        return f"Marker names match the template ({len(marker_mapping)} matched markers)."
    preview = "; ".join(remapped[:8])
    suffix = "" if len(remapped) <= 8 else f"; +{len(remapped) - 8} more"
    return f"Automatic marker name mapping: {preview}{suffix}"


def _load_virtual_marker_joint_names() -> dict:
    """
    Load the editable segment-pair to joint-name mapping used for suggested virtual marker names.
    """
    filepath = Path(__file__).with_name("virtual_marker_joint_names.json")
    try:
        return json.loads(filepath.read_text())
    except (OSError, json.JSONDecodeError):
        return {"default": "Joint", "pairs": []}


def _joint_name_from_segments(proximal_segment_name: str, distal_segment_name: str) -> str:
    """
    Infer a joint name from proximal/distal segment names using an editable JSON mapping.
    """
    mapping = _load_virtual_marker_joint_names()
    proximal = proximal_segment_name.strip()
    distal = distal_segment_name.strip()
    for entry in mapping.get("pairs", []):
        proximal_segments = set(entry.get("proximal_segments", []))
        distal_segments = set(entry.get("distal_segments", []))
        if proximal in proximal_segments and distal in distal_segments:
            return _lateralized_joint_name(entry.get("joint", mapping.get("default", "Joint")), distal)
    if distal:
        return distal
    if proximal:
        return proximal
    return mapping.get("default", "Joint")


def _lateralized_joint_name(joint_name: str, distal_segment_name: str) -> str:
    """
    Prefix left/right joint names when the distal segment name carries side information.
    """
    if distal_segment_name.startswith(("L", "Left", "Gauche")):
        return f"Left_{joint_name}"
    if distal_segment_name.startswith(("R", "Right", "Droit")):
        return f"Right_{joint_name}"
    if distal_segment_name.endswith("G"):
        return f"Left_{joint_name}"
    if distal_segment_name.endswith("D"):
        return f"Right_{joint_name}"
    return joint_name


def _visible_virtual_marker_methods() -> set[str]:
    """
    Return virtual marker methods shown in the GUI.
    """
    return {
        "pointing",
        "score",
        "sara",
        "sara_direction",
        "marker_mean",
        "axis_projection",
    } | set(PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS)


def _virtual_marker_method_display(method: str) -> str:
    """
    Return the user-facing label for a virtual marker method.
    """
    return VIRTUAL_MARKER_METHOD_DISPLAY_LABELS.get(method, method)


def _virtual_marker_method_from_display(label: str) -> str:
    """
    Return the stored method value for a user-facing virtual marker method label.
    """
    for method, display_label in VIRTUAL_MARKER_METHOD_DISPLAY_LABELS.items():
        if label == display_label:
            return method
    return label


def _parent_segment_name(workflow_draft, segment_name: str) -> str:
    """
    Return the parent segment stored in the current C3D workflow draft.
    """
    for group in workflow_draft.segment_marker_groups:
        if group.segment_name == segment_name:
            return group.parent_name
    return ""


def _technical_markers_for_segment(workflow_draft, segment_name: str) -> tuple[str, ...]:
    """
    Return technical markers for one segment, falling back to assigned markers when none are flagged yet.
    """
    for group in workflow_draft.segment_marker_groups:
        if group.segment_name == segment_name:
            return group.technical_marker_names if group.technical_marker_names else group.marker_names
    return ()


def _settings_anthropometry_model(combo) -> str:
    """
    Return the stored anthropometry model key from the GUI combo.
    """
    model = combo.currentText().strip()
    return "" if model == "none" else model


def _segment_length_marker_groups(workflow_draft, segment_name: str) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """
    Return proximal and distal marker groups used to estimate a segment length.

    The proximal point is the anatomical origin of the selected segment. The distal point is the anatomical origin of
    the first child segment found in the kinematic chain, which corresponds to the next joint for usual linked models.
    """
    proximal_markers = _segment_origin_markers(workflow_draft, segment_name)
    child_name = _first_child_segment_name(workflow_draft, segment_name)
    distal_markers = _segment_origin_markers(workflow_draft, child_name) if child_name else ()
    source = (
        f"proximal={','.join(proximal_markers) or '-'}; "
        f"distal={','.join(distal_markers) or '-'}; child={child_name or '-'}"
    )
    return proximal_markers, distal_markers, source


def _segment_origin_markers(workflow_draft, segment_name: str) -> tuple[str, ...]:
    """
    Return the anatomical origin markers associated with one segment.
    """
    for axis in workflow_draft.axes:
        if axis.segment_name == segment_name and len(axis.origin_markers) != 0:
            return tuple(axis.origin_markers)
    return ()


def _first_child_segment_name(workflow_draft, segment_name: str) -> str:
    """
    Return the first child of one segment in the current draft.
    """
    for group in workflow_draft.segment_marker_groups:
        if group.parent_name == segment_name:
            return group.segment_name
    return ""


def _segment_length_from_draft(workflow_draft, c3d_data, segment_name: str) -> tuple[float | None, str]:
    """
    Estimate a segment length from the main C3D and the anatomical origin definitions.
    """
    proximal_markers, distal_markers, source = _segment_length_marker_groups(workflow_draft, segment_name)
    if len(proximal_markers) == 0 or len(distal_markers) == 0:
        return None, f"{source}; missing anatomical origin marker group."
    missing_markers = tuple(
        marker_name
        for marker_name in dict.fromkeys(proximal_markers + distal_markers)
        if marker_name not in c3d_data.marker_names
    )
    if len(missing_markers) != 0:
        return None, f"{source}; missing C3D markers={','.join(missing_markers)}."
    proximal_positions = _mean_marker_group_series(c3d_data, proximal_markers)
    distal_positions = _mean_marker_group_series(c3d_data, distal_markers)
    distances = np.linalg.norm(distal_positions - proximal_positions, axis=0)
    if np.isnan(distances).all():
        return None, f"{source}; all frames are NaN."
    return float(np.nanmean(distances)), source


def _mean_marker_group_series(c3d_data, marker_names: tuple[str, ...]) -> np.ndarray:
    """
    Return the mean marker trajectory for a possibly duplicated marker group.
    """
    positions = [c3d_data.get_position((marker_name,))[:3, 0, :] for marker_name in marker_names]
    return np.nanmean(np.stack(positions, axis=0), axis=0)


def _complete_marker_frame_count(c3d_data, marker_names: tuple[str, ...]) -> int:
    """
    Count frames where every requested marker has finite x/y/z coordinates.
    """
    marker_names = tuple(dict.fromkeys(marker_names))
    if len(marker_names) == 0:
        return 0
    if any(marker_name not in c3d_data.marker_names for marker_name in marker_names):
        return 0
    positions = c3d_data.get_position(marker_names)[:3, :, :]
    return int(np.sum(np.isfinite(positions).all(axis=(0, 1))))


def _functional_algorithm_quality_text(method: str, rt_parent, rt_child) -> str:
    """
    Return final residual metrics after SCoRE/SARA internal outlier filtering.
    """
    try:
        if method == "score":
            from ..model_modifiers.joint_center_tool import Score

            _, cor_parent_local, cor_child_local, rt_parent_valid, rt_child_valid = Score.perform_algorithm(
                rt_parent, rt_child
            )
            residuals = _score_residuals(rt_parent_valid, rt_child_valid, cor_parent_local, cor_child_local)
            return (
                f"SCoRE residual={np.nanmean(residuals) * 1000:.1f} +/- {np.nanstd(residuals) * 1000:.1f} mm "
                f"over {len(rt_parent_valid)} algorithm frames"
            )
        if _is_sara_direction_method(method):
            from ..model_modifiers.joint_center_tool import Sara

            (
                _,
                aor_parent_local,
                aor_child_local,
                _,
                _,
                _,
                rt_parent_valid,
                rt_child_valid,
            ) = Sara.perform_algorithm(rt_parent, rt_child)
            residuals = _sara_residual_angles(rt_parent_valid, rt_child_valid, aor_parent_local, aor_child_local)
            return (
                f"SARA residual={np.nanmean(residuals):.1f} +/- {np.nanstd(residuals):.1f} deg "
                f"over {len(rt_parent_valid)} algorithm frames"
            )
    except Exception as error:
        return f"algorithm residual unavailable ({error})"
    return "algorithm residual unavailable"


def _sara_static_axis_quality_text(
    *,
    method: str,
    feature,
    payload: dict[str, str],
    static_data,
    functional_data,
    parent_marker_names: tuple[str, ...],
    child_marker_names: tuple[str, ...],
    use_diverse_functional_frames: bool,
    manual_functional_frame_indices: tuple[int, ...] = (),
) -> str:
    """
    Return a SARA-vs-static anatomical axis quality message when enough data is available.
    """
    if not _is_sara_direction_method(method) or static_data is None:
        return ""
    expected_markers = _sara_expected_axis_markers(feature, payload)
    if len(expected_markers) < 2:
        return ""
    expected_start_markers = expected_markers[:1]
    expected_end_markers = expected_markers[1:]
    origin_markers = _sara_origin_markers(feature, payload, expected_markers)
    required_static = parent_marker_names + child_marker_names + expected_start_markers + expected_end_markers
    required_functional = parent_marker_names + child_marker_names + expected_start_markers + expected_end_markers
    if any(marker_name not in static_data.marker_names for marker_name in required_static):
        return ""
    if any(marker_name not in functional_data.marker_names for marker_name in required_functional):
        return ""
    if any(marker_name not in functional_data.marker_names for marker_name in origin_markers):
        return ""
    try:
        from ..components.generic.rigidbody.segment_coordinate_system import (
            SegmentCoordinateSystemUtils,
        )
        from ..model_modifiers.joint_center_tool import Sara

        parent_static_data = static_data.get_partial_dict_data(parent_marker_names)
        child_static_data = static_data.get_partial_dict_data(child_marker_names)
        parent_functional_data = functional_data.get_partial_dict_data(parent_marker_names)
        child_functional_data = functional_data.get_partial_dict_data(child_marker_names)
        rt_parent_func = SegmentCoordinateSystemUtils.rigidify(
            functional_data=parent_functional_data,
            static_data=parent_static_data,
        )
        rt_child_func = SegmentCoordinateSystemUtils.rigidify(
            functional_data=child_functional_data,
            static_data=child_static_data,
        )
        rt_parent_func, rt_child_func, report = prepare_functional_rt_pair(
            rt_parent_func,
            rt_child_func,
            _functional_frame_selection_options(
                use_diverse_functional_frames,
                manual_functional_frame_indices,
            ),
        )
        original_axis_global = np.nanmean(
            _mean_marker_series(functional_data, expected_end_markers)
            - _mean_marker_series(functional_data, expected_start_markers),
            axis=1,
        )
        origin_positions_global = (
            functional_data.markers_center_position(origin_markers) if len(origin_markers) != 0 else None
        )
        if _uses_selected_functional_frames(report) and origin_positions_global is not None:
            origin_positions_global = subset_points_by_frame(origin_positions_global, report.selected_indices)
        (
            _aor_mean_global,
            aor_parent_local,
            _aor_child_local,
            _cor_mean_global,
            _cor_parent_local,
            _cor_child_local,
            _rt_parent_valid,
            _rt_child_valid,
        ) = Sara.perform_algorithm(
            rt_parent=rt_parent_func,
            rt_child=rt_child_func,
            original_axis_global=original_axis_global,
            origin_positions_global=origin_positions_global,
        )
        rt_parent_static = SegmentCoordinateSystemUtils.rigidify(parent_static_data)
        sara_static_direction = np.nanmean(_rt_vector_series(rt_parent_static, aor_parent_local), axis=1)
        expected_static_direction = np.nanmean(
            _mean_marker_series(static_data, expected_end_markers)
            - _mean_marker_series(static_data, expected_start_markers),
            axis=1,
        )
        deviation = _axis_angle_degrees(sara_static_direction, expected_static_direction)
    except Exception as error:
        return f"SARA/static axis deviation unavailable ({error})"
    if not np.isfinite(deviation):
        return "SARA/static axis deviation unavailable"
    limit = _sara_static_deviation_limit_from_payload(payload)
    axis_label = f"{','.join(expected_start_markers)}->{','.join(expected_end_markers)}"
    if limit is not None and deviation > limit:
        return (
            f"SARA/static axis deviation={deviation:.1f} deg vs {axis_label}; "
            f"WARNING >{limit:g} deg, anatomical fallback will be used"
        )
    if limit is not None:
        return f"SARA/static axis deviation={deviation:.1f} deg vs {axis_label} (limit {limit:g} deg)"
    return f"SARA/static axis deviation={deviation:.1f} deg vs {axis_label}"


def _sara_expected_axis_markers(feature, payload: dict[str, str]) -> tuple[str, ...]:
    """
    Return the marker pair used as the static expected SARA axis.
    """
    return tuple(getattr(feature, "start_markers", ())) + tuple(getattr(feature, "end_markers", ())) or (
        _split_marker_names(payload.get("expected axis", ""))
    )


def _sara_origin_markers(feature, payload: dict[str, str], expected_markers: tuple[str, ...]) -> tuple[str, ...]:
    """
    Return SARA origin markers from a GUI feature payload.
    """
    origin_markers = tuple(getattr(feature, "origin_markers", ())) or _split_marker_names(
        payload.get("origin markers", "")
    )
    return origin_markers or expected_markers


def _sara_static_deviation_limit_from_payload(payload: dict[str, str]) -> float | None:
    """
    Return the configured SARA/static axis deviation limit, if any.
    """
    text = payload.get("max static deviation", "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return DEFAULT_SARA_STATIC_AXIS_DEVIATION_LIMIT_DEGREES
    return value if np.isfinite(value) and value > 0 else None


def _axis_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """
    Return the oriented angle between two 3D vectors.
    """
    first = np.asarray(first, dtype=float).reshape(3)
    second = np.asarray(second, dtype=float).reshape(3)
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return float("nan")
    cosine = float(np.clip(np.dot(first / first_norm, second / second_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _sara_expected_global_axis(c3d_data, expected_markers: tuple[str, ...]):
    """
    Return the mean expected SARA axis in global coordinates, when markers exist.
    """
    if c3d_data is None or len(expected_markers) < 2:
        return None
    expected_start_markers = expected_markers[:1]
    expected_end_markers = expected_markers[1:]
    if any(marker not in c3d_data.marker_names for marker in expected_start_markers + expected_end_markers):
        return None
    return np.nanmean(
        _mean_marker_series(c3d_data, expected_end_markers) - _mean_marker_series(c3d_data, expected_start_markers),
        axis=1,
    )


def _format_vector(vector: np.ndarray) -> str:
    """
    Format one 3D vector compactly for diagnostic reports.
    """
    vector = np.asarray(vector, dtype=float).reshape(3)
    return "[" + ", ".join(f"{float(value): .4g}" for value in vector) + "]"


def _local_axis_orientation_lines(label: str, vector: np.ndarray) -> list[str]:
    """
    Return local vector components and angles against X/Y/Z unit axes.
    """
    vector = np.asarray(vector, dtype=float).reshape(3)
    axes = {
        "X": np.asarray([1.0, 0.0, 0.0]),
        "Y": np.asarray([0.0, 1.0, 0.0]),
        "Z": np.asarray([0.0, 0.0, 1.0]),
    }
    angle_text = ", ".join(
        f"{axis_name}={_axis_angle_degrees(vector, axis_vector):.1f} deg" for axis_name, axis_vector in axes.items()
    )
    return [f"- {label}: {_format_vector(vector)}; angles to local axes: {angle_text}"]


def _mean_local_vector(rt_series, global_vector: np.ndarray) -> np.ndarray:
    """
    Express one global vector in a RT time series and average it in local coordinates.
    """
    global_vector = np.asarray(global_vector, dtype=float).reshape(3)
    local = np.zeros((3, len(rt_series)))
    for frame_index in range(len(rt_series)):
        local[:, frame_index] = rt_series[frame_index].rotation_matrix.rotation_matrix.T @ global_vector
    return np.nanmean(local, axis=1)


def _relative_xyz_euler_degrees(rt_parent, rt_child) -> np.ndarray:
    """
    Return relative parent-to-child XYZ Euler angles for an RT time series.
    """
    angles = np.zeros((3, len(rt_parent)))
    for frame_index in range(len(rt_parent)):
        relative_rt = rt_parent[frame_index].inverse @ rt_child[frame_index]
        angles[:, frame_index] = np.rad2deg(relative_rt.rotation_matrix.euler_angles("xyz"))
    return angles


def _series_plot_lines(title: str, values: np.ndarray, unit: str, width: int = 64) -> list[str]:
    """
    Return a compact text plot and summary for one finite time series.
    """
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return [f"{title}: no finite values"]
    count = min(width, finite_values.size)
    indices = np.linspace(0, finite_values.size - 1, count).astype(int)
    samples = finite_values[indices]
    minimum = float(np.nanmin(samples))
    maximum = float(np.nanmax(samples))
    span = maximum - minimum
    levels = " .:-=+*#%@"
    if span <= 1e-12:
        plot = levels[0] * count
    else:
        normalized = np.clip((samples - minimum) / span, 0.0, 1.0)
        plot = "".join(levels[int(round(value * (len(levels) - 1)))] for value in normalized)
    return [
        f"{title}: mean={np.nanmean(finite_values):.3g} {unit}; std={np.nanstd(finite_values):.3g} {unit}; "
        f"min={np.nanmin(finite_values):.3g}; max={np.nanmax(finite_values):.3g}; n={finite_values.size}",
        f"    {plot}",
    ]


def _marker_weights_for_reconstruction(marker_names: tuple[str, ...], high_weight_markers: tuple[str, ...]):
    """
    Return BioBuddy marker weights for a reconstruction diagnostic.
    """
    from ..components.real.rigidbody.marker_weight import MarkerWeight
    from ..utils.named_list import NamedList

    high_weight_marker_set = set(high_weight_markers)
    marker_weights = NamedList[MarkerWeight]()
    for marker_name in marker_names:
        weight = 100.0 if marker_name in high_weight_marker_set else 1.0
        marker_weights.append(MarkerWeight(marker_name, weight))
    return marker_weights


def _static_virtual_point_definitions_from_draft(workflow_draft) -> tuple[object, ...]:
    """
    Convert editable draft virtual points that are evaluable on the static C3D.
    """
    from .virtual_points import (
        marker_mean_virtual_point,
        predictive_rab2002_shoulder_cor,
    )

    definitions = []
    for marker in workflow_draft.virtual_markers:
        if marker.method == "marker_mean":
            marker_names = _split_marker_names(marker.source)
            if marker_names:
                definitions.append(marker_mean_virtual_point(marker.name, marker_names))
        elif marker.method == "rab2002_shoulder":
            point_marker, mid_markers, fraction = _rab2002_markers_from_payload(
                marker.source, marker.name, marker.segment_name
            )
            if point_marker and len(mid_markers) == 2:
                definitions.append(
                    predictive_rab2002_shoulder_cor(
                        marker.name,
                        point_marker,
                        mid_markers[0],
                        mid_markers[1],
                        fraction=fraction,
                    )
                )
    return tuple(definitions)


def _diagnostic_template_for_c3d_model_preset(preset: C3dModelPreset, *, use_marker_fallback: bool = False):
    """
    Return the regular diagnostic template or its marker-only fallback variant.
    """
    if not use_marker_fallback:
        return template_for_c3d_model_preset(preset)
    if preset == C3dModelPreset.LOWER_LIMBS:
        from .lower_limb_template import lower_limb_template

        return lower_limb_template(use_functional=False)
    if preset == C3dModelPreset.MOTIVE_57:
        from .motive_57_template import motive_57_template

        return motive_57_template(use_functional=False)
    if preset == C3dModelPreset.MOTIVE_57_ISB:
        from .motive_57_isb_template import motive_57_isb_template

        return motive_57_isb_template(use_functional=False)
    if preset == C3dModelPreset.FULL_BODY:
        from .full_body_model202_template import full_body_model202_template

        return full_body_model202_template(use_functional=False)
    if preset == C3dModelPreset.LOWER_LIMBS_ANATOMICAL:
        return template_for_c3d_model_preset(preset)
    raise NotImplementedError(f"Preset '{preset.value}' does not provide a marker fallback diagnostic template.")


def _diagnostic_reconstruction_frame_indices(
    selected_indices: tuple[int, ...],
    frame_count: int,
    method: str,
) -> tuple[tuple[int, ...], str]:
    """
    Return the frame indices used by the reconstruction diagnostic.

    QLD is intentionally capped because it solves a nonlinear least-squares problem
    frame-by-frame in the GUI thread.
    """
    if frame_count <= 0:
        return (), "no frames"
    indices = tuple(int(index) for index in selected_indices if 0 <= int(index) < int(frame_count))
    if len(indices) == 0:
        indices = tuple(range(frame_count))
    original_count = len(indices)
    if method == "QLD" and len(indices) > FUNCTIONAL_RECONSTRUCTION_QLD_MAX_FRAMES:
        sample_positions = np.linspace(
            0,
            len(indices) - 1,
            FUNCTIONAL_RECONSTRUCTION_QLD_MAX_FRAMES,
        ).astype(int)
        indices = tuple(indices[int(position)] for position in sample_positions)
        return (
            indices,
            f"{len(indices)} / {original_count} selected frames, downsampled for QLD responsiveness",
        )
    return indices, f"{len(indices)} selected frames"


def _rotation_dof_plot_data(
    model,
    q: np.ndarray,
    segment_name: str,
    frame_rate: float | None = None,
    frame_indices: tuple[int, ...] = (),
) -> dict[str, object] | None:
    """
    Return plottable rotation DoF data for one reconstructed segment.
    """
    segment_name = _rotation_segment_name_for_model(model, segment_name)
    rotation_dofs = _segment_rotation_dof_indices(model, segment_name)
    if len(rotation_dofs) != 3:
        return None
    fps = float(frame_rate) if frame_rate is not None and frame_rate > 0 else 1.0
    if len(frame_indices) == q.shape[1]:
        time = np.asarray(frame_indices, dtype=float) / fps
    else:
        time = np.arange(q.shape[1], dtype=float) / fps
    return {
        "title": f"{segment_name} rotation DoFs from chain reconstruction",
        "time": time,
        "series": tuple(
            (dof_name, np.rad2deg(np.asarray(q[dof_index], dtype=float))) for dof_name, dof_index in rotation_dofs
        ),
    }


def _combine_rotation_plot_data(
    primary: dict[str, object] | None,
    fallback: dict[str, object] | None,
    *,
    fallback_label_suffix: str,
) -> dict[str, object] | None:
    """
    Overlay fallback rotation DoFs as dashed lines on the primary reconstruction plot.
    """
    if primary is None:
        if fallback is None:
            return None
        return _styled_rotation_plot_data(fallback, label_suffix=fallback_label_suffix, line_style="--")
    if fallback is None:
        return primary

    primary_time = np.asarray(primary.get("time", ()), dtype=float)
    fallback_time = np.asarray(fallback.get("time", ()), dtype=float)
    frame_count = min(primary_time.size, fallback_time.size)
    if frame_count == 0:
        return primary

    primary_series = _styled_rotation_series(primary.get("series", ()), frame_count=frame_count, line_style="-")
    fallback_series = _styled_rotation_series(
        fallback.get("series", ()),
        frame_count=frame_count,
        label_suffix=fallback_label_suffix,
        line_style="--",
    )
    return {
        "title": str(primary.get("title", "Rotation DoF")) + " vs marker fallback",
        "time": primary_time[:frame_count],
        "series": primary_series + fallback_series,
    }


def _styled_rotation_plot_data(
    plot_data: dict[str, object], *, label_suffix: str, line_style: str
) -> dict[str, object]:
    """
    Return one plot data dict with labels/styles normalized for the plot widget.
    """
    time = np.asarray(plot_data.get("time", ()), dtype=float)
    return {
        "title": str(plot_data.get("title", "Rotation DoF")),
        "time": time,
        "series": _styled_rotation_series(
            plot_data.get("series", ()),
            frame_count=time.size,
            label_suffix=label_suffix,
            line_style=line_style,
        ),
    }


def _styled_rotation_series(
    series,
    *,
    frame_count: int,
    label_suffix: str = "",
    line_style: str,
) -> tuple[tuple[str, np.ndarray, str], ...]:
    """
    Normalize rotation series to the `(label, values, linestyle)` plot format.
    """
    styled = []
    for item in tuple(series):
        if len(item) == 3:
            label, values, _existing_style = item
        else:
            label, values = item
        label = str(label)
        if label_suffix:
            label = f"{label} ({label_suffix})"
        styled.append((label, np.asarray(values, dtype=float)[:frame_count], line_style))
    return tuple(styled)


def _relative_xyz_plot_data(
    relative_xyz: np.ndarray, segment_name: str, frame_rate: float | None = None
) -> dict[str, object] | None:
    """
    Return plottable XYZ relative rotations from proximal/distal rigid clusters.
    """
    values = np.asarray(relative_xyz, dtype=float)
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] == 0:
        return None
    fps = float(frame_rate) if frame_rate is not None and frame_rate > 0 else 1.0
    time = np.arange(values.shape[1], dtype=float) / fps
    return {
        "title": f"{segment_name} relative rotations from functional marker clusters",
        "time": time,
        "series": (
            ("X rotation", values[0]),
            ("Y rotation", values[1]),
            ("Z rotation", values[2]),
        ),
    }


def _rotation_dof_summary_lines(model, q: np.ndarray, segment_name: str) -> list[str]:
    """
    Return compact numeric summaries for reconstructed rotation DoFs.
    """
    segment_name = _rotation_segment_name_for_model(model, segment_name)
    rotation_dofs = _segment_rotation_dof_indices(model, segment_name)
    lines = ["", f"{segment_name} rotation DoFs from chain reconstruction"]
    if len(rotation_dofs) != 3:
        lines.append(
            f"- Expected 3 rotation DoF for {segment_name}, found {len(rotation_dofs)}: "
            + (", ".join(name for name, _ in rotation_dofs) or "-")
        )
        return lines
    for dof_name, dof_index in rotation_dofs:
        values = np.rad2deg(np.asarray(q[dof_index], dtype=float))
        lines.append(
            f"- {dof_name}: mean={np.nanmean(values):.2f} deg; "
            f"range={np.nanmin(values):.2f}..{np.nanmax(values):.2f} deg"
        )
    return lines


def _rotation_segment_name_for_model(model, segment_name: str) -> str:
    """Resolve a physical segment to the separate joint segment that owns its rotational DoFs."""
    if not segment_name:
        return segment_name
    segments_by_name = {segment.name: segment for segment in model.segments}
    segment = segments_by_name.get(segment_name)
    if segment is None:
        return segment_name
    if any("_rot" in name or name.lower().startswith("rot") for name in segment.dof_names):
        return segment_name
    parent = segments_by_name.get(segment.parent_name)
    if parent is not None and any("_rot" in name or name.lower().startswith("rot") for name in parent.dof_names):
        return parent.name
    return segment_name


def _segment_rotation_dof_indices(model, segment_name: str) -> tuple[tuple[str, int], ...]:
    """
    Return global q indices for rotation DoFs of one segment.
    """
    dof_indices = []
    offset = 0
    for segment in model.segments:
        segment_dof_count = segment.nb_q
        if segment.name == segment_name:
            for local_index, dof_name in enumerate(segment.dof_names):
                if "_rot" in dof_name or dof_name.lower().startswith("rot"):
                    dof_indices.append((dof_name, offset + local_index))
            break
        offset += segment_dof_count
    return tuple(dof_indices)


def _score_residuals(rt_parent, rt_child, cor_parent_local: np.ndarray, cor_child_local: np.ndarray) -> np.ndarray:
    """
    Return frame-by-frame distance between parent and child SCoRE center projections.
    """
    residuals = np.zeros((len(rt_parent),))
    parent_local = np.hstack((np.asarray(cor_parent_local).reshape(3), 1.0))
    child_local = np.hstack((np.asarray(cor_child_local).reshape(3), 1.0))
    for frame_index in range(len(rt_parent)):
        parent_global = (rt_parent[frame_index] @ parent_local).reshape(4)[:3]
        child_global = (rt_child[frame_index] @ child_local).reshape(4)[:3]
        residuals[frame_index] = np.linalg.norm(parent_global - child_global)
    return residuals


def _sara_residual_angles(rt_parent, rt_child, aor_parent_local: np.ndarray, aor_child_local: np.ndarray) -> np.ndarray:
    """
    Return frame-by-frame angle between parent and child SARA axis directions, in degrees.
    """
    residuals = np.zeros((len(rt_parent),))
    parent_local = np.asarray(aor_parent_local).reshape(3)
    child_local = np.asarray(aor_child_local).reshape(3)
    for frame_index in range(len(rt_parent)):
        parent_global = rt_parent[frame_index].rotation_matrix.rotation_matrix @ parent_local
        child_global = rt_child[frame_index].rotation_matrix.rotation_matrix @ child_local
        denominator = np.linalg.norm(parent_global) * np.linalg.norm(child_global)
        residuals[frame_index] = np.rad2deg(
            np.arccos(np.clip(np.dot(parent_global, child_global) / denominator, -1.0, 1.0))
        )
    return residuals


def _rt_point_series(rt_series, local_point: np.ndarray) -> np.ndarray:
    """
    Transform one local 3D point through an RT time series.
    """
    local = np.hstack((np.asarray(local_point, dtype=float).reshape(3), 1.0))
    points = np.zeros((3, len(rt_series)))
    for frame_index in range(len(rt_series)):
        points[:, frame_index] = (rt_series[frame_index] @ local).reshape(4)[:3]
    return points


def _rt_vector_series(rt_series, local_vector: np.ndarray) -> np.ndarray:
    """
    Transform one local 3D vector through an RT time series.
    """
    local = np.asarray(local_vector, dtype=float).reshape(3)
    vectors = np.zeros((3, len(rt_series)))
    for frame_index in range(len(rt_series)):
        vectors[:, frame_index] = rt_series[frame_index].rotation_matrix.rotation_matrix @ local
    return vectors


def _selected_marker_xyz(c3d_data, marker_name: str, selected_indices: tuple[int, ...]) -> np.ndarray:
    """
    Return marker xyz positions for the selected functional frames.
    """
    points = c3d_data.get_position((marker_name,))[:3, 0, :]
    if selected_indices:
        return points[:, selected_indices]
    return points


def _point_to_line_distances(points: np.ndarray, line_start: np.ndarray, line_direction: np.ndarray) -> np.ndarray:
    """
    Return framewise distances from points to framewise 3D lines.
    """
    direction_norm = np.linalg.norm(line_direction, axis=0, keepdims=True)
    unit = np.divide(
        line_direction,
        direction_norm,
        out=np.zeros_like(line_direction),
        where=direction_norm > 1e-12,
    )
    delta = points - line_start
    projection = unit * np.sum(delta * unit, axis=0, keepdims=True)
    return np.linalg.norm(delta - projection, axis=0)


def _project_point_on_line(
    point: tuple[float, float, float] | None,
    line_start: tuple[float, float, float] | None,
    line_end: tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
    """
    Project one 3D point onto one 3D line.
    """
    if point is None or line_start is None or line_end is None:
        return point
    start = np.asarray(line_start, dtype=float)
    vector = np.asarray(line_end, dtype=float) - start
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return point
    unit = vector / norm
    projected = start + unit * np.dot(np.asarray(point, dtype=float) - start, unit)
    return _finite_point3d(projected)


def _residual_boxplot_data(title: str, values: np.ndarray, unit: str) -> dict[str, object]:
    """
    Return normalized data for the residual boxplot widget.
    """
    return {
        "title": title,
        "values": np.asarray(values, dtype=float),
        "unit": unit,
    }


def _histogram_lines(title: str, values: np.ndarray, unit: str, bins: int = 10) -> list[str]:
    """
    Return a compact ASCII histogram for finite values.
    """
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return [f"{title}: no finite values"]
    counts, edges = np.histogram(finite_values, bins=min(bins, max(1, finite_values.size)))
    max_count = max(int(np.max(counts)), 1)
    lines = [
        f"{title}: mean={np.nanmean(finite_values):.3g} {unit}; "
        f"std={np.nanstd(finite_values):.3g} {unit}; n={finite_values.size}"
    ]
    for count, start, end in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * max(1, int(round(24 * count / max_count))) if count else ""
        lines.append(f"    {start:8.3g} - {end:8.3g} {unit}: {bar} ({int(count)})")
    return lines


def _c3d_file_names_from_folder(folder_path: str) -> tuple[str, ...]:
    """
    Return C3D filenames available in the selected workflow folder.
    """
    if not folder_path:
        return ()
    folder = Path(folder_path)
    if not folder.exists():
        return ()
    return tuple(sorted(path.name for path in folder.glob("*.c3d")))


def _matching_c3d_file_for_expected_name(folder_path: str, expected_name: str) -> Path | None:
    """
    Return the single C3D file matching an expected template name or participant-independent pattern.
    """
    folder = Path(folder_path)
    if not folder.exists():
        return None
    exact_path = folder / expected_name
    if exact_path.exists():
        return exact_path

    patterns = [expected_name]
    if expected_name.startswith("Test_"):
        patterns.append(f"*{expected_name.removeprefix('Test_')}")
    elif expected_name.lower() == "static.c3d":
        patterns.extend(("*Static.c3d", "*static*.c3d"))
    elif "Func_" in expected_name:
        patterns.append(expected_name.replace("Func_", ""))
    elif "_func_" in expected_name:
        patterns.append(f"*{expected_name.split('_func_', maxsplit=1)[1]}")

    matches = []
    for pattern in dict.fromkeys(patterns):
        matches.extend(folder.glob(pattern))
    unique_matches = tuple(sorted(dict.fromkeys(path for path in matches if path.suffix.lower() == ".c3d")))
    return unique_matches[0] if len(unique_matches) == 1 else None


def _trial_name_from_virtual_feature_source(source: str) -> str:
    """
    Extract the functional trial identifier from a virtual marker or axis source string.
    """
    match = re.search(r"(?:^|;\s*)trial=([^;]+)", source)
    return match.group(1).strip() if match is not None else ""


def _assigned_c3d_source_for_role(workflow_draft, role: str) -> str:
    """
    Return the participant C3D path assigned to one workflow role.
    """
    if not role:
        return ""
    for assignment in workflow_draft.file_assignments:
        if assignment.role == role:
            return assignment.source_path
    return ""


def _source_with_c3d_assignment(source: str, source_path: str) -> str:
    """
    Store a matched C3D next to the existing descriptive source without erasing the source metadata.
    """
    if not source_path:
        return source
    c3d_name = Path(source_path).name
    parts = [part.strip() for part in source.split(";") if part.strip() and not part.strip().startswith("c3d=")]
    return "; ".join((f"c3d={c3d_name}", *parts))


def _c3d_source_name_from_virtual_feature_source(source: str) -> str:
    """
    Return the C3D filename embedded in a virtual marker or axis source string.
    """
    match = re.search(r"(?:^|;\s*)c3d=([^;]+)", source)
    return Path(match.group(1).strip()).name if match is not None else ""


def _predictive_virtual_marker_method_from_label(label_or_key: str) -> str:
    """
    Return the stored predictive method key from the readable GUI label.
    """
    if label_or_key in PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS:
        return label_or_key
    for method_key, method_label in PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS.items():
        if label_or_key == method_label:
            return method_key
    return next(iter(PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS))


def _python_code_from_c3d_draft(workflow_draft, c3d_folder_path: str = "") -> str:
    """
    Generate an editable Python starting point from the current C3D workflow draft.
    """
    payload = _serializable_c3d_draft_payload(workflow_draft)
    return "\n".join(
        [
            '"""Generated BioBuddy C3D model definition.',
            "",
            "Edit the marker names, virtual markers, axes, and DoF settings below, then adapt the final",
            "build section to your participant-specific C3D files.",
            '"""',
            "",
            f"C3D_FOLDER = {c3d_folder_path!r}",
            f"WORKFLOW_DRAFT = {json.dumps(payload, indent=4)}",
            "",
            "# The GUI stores the draft as plain data. Rebuild a C3dWorkflowDraft or adapt this payload",
            "# directly if you want to script project-specific model generation.",
            "template_payload = WORKFLOW_DRAFT",
            "print(template_payload)",
            "",
        ]
    )


def _serializable_c3d_draft_payload(workflow_draft) -> dict:
    """
    Convert the dataclass draft to JSON-friendly Python data.
    """
    payload = asdict(workflow_draft)
    if hasattr(workflow_draft.preset, "value"):
        payload["preset"] = workflow_draft.preset.value
    return payload


def _export_model_to_path(model, filepath: str) -> None:
    """
    Export a model with the writer matching the file extension.
    """
    export_path = Path(filepath)
    suffix = export_path.suffix.lower()
    writer_names = {
        ".biomod": "to_biomod",
        ".osim": "to_osim",
        ".urdf": "to_urdf",
        ".bvh": "to_bvh",
    }
    if suffix not in writer_names:
        raise ValueError("Supported export extensions are .bioMod, .osim, .urdf, and .bvh.")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    getattr(model, writer_names[suffix])(filepath=str(export_path))
    if not export_path.exists():
        raise RuntimeError(f"The model writer finished but did not create:\n{export_path}")
    if export_path.stat().st_size == 0:
        raise RuntimeError(f"The model writer created an empty file:\n{export_path}")


def _model_export_filepath_with_extension(filepath: str, extension: str, default_stem: str) -> str:
    """
    Return the concrete export filepath, accepting either a filename or a selected folder.
    """
    export_path = Path(filepath).expanduser()
    if export_path.exists() and export_path.is_dir():
        export_path = export_path / f"{default_stem}{extension}"
    else:
        export_path = export_path.with_suffix(extension)
    return str(export_path)


def _model_export_extension_from_label(label: str) -> str:
    """
    Return the model export extension selected in the chain definition menu.
    """
    if ".bvh" in label:
        return ".bvh"
    if ".osim" in label:
        return ".osim"
    if ".urdf" in label:
        return ".urdf"
    return ".bioMod"


def _model_export_filter_for_extension(extension: str) -> str:
    """
    Return a save-dialog filter with the selected model format first.
    """
    filters = {
        ".biomod": "BioMod files (*.bioMod)",
        ".bvh": "BVH files (*.bvh)",
        ".osim": "OpenSim files (*.osim)",
        ".urdf": "URDF files (*.urdf)",
    }
    selected_filter = filters.get(extension.lower(), filters[".biomod"])
    return f"{selected_filter};;Supported models (*.bioMod *.osim *.urdf *.bvh)"


def _c3d_generation_log(
    workflow_draft, c3d_data, c3d_folder_path: str, marker_pool: tuple[str, ...]
) -> tuple[str, ...]:
    """
    Build a detailed human-readable generation log for the current C3D model draft.
    """
    lines = [
        f"Preset: {workflow_draft.preset.value}",
        f"C3D folder: {c3d_folder_path or 'not selected'}",
        f"Main C3D markers: {len(c3d_data.marker_names) if c3d_data is not None else 0}",
        f"Known marker pool: {len(marker_pool)}",
        "",
        "C3D file assignments:",
    ]
    for assignment in workflow_draft.file_assignments:
        lines.append(f"- {assignment.role}: {_c3d_assignment_log_name(workflow_draft.preset, assignment)}")
    lines.extend(["", "Segments:"])
    for group in workflow_draft.segment_marker_groups:
        technical = ", ".join(group.technical_marker_names) if group.technical_marker_names else "-"
        markers = ", ".join(group.marker_names) if group.marker_names else "-"
        parent = group.parent_name if group.parent_name else "-"
        lines.append(
            f"- {group.segment_name} ({group.segment_type}, parent={parent}): markers=[{markers}], technical=[{technical}]"
        )
    lines.extend(["", "Virtual markers:"])
    for marker in workflow_draft.virtual_markers:
        source = marker.source if marker.source else "-"
        equation = marker.equation if marker.equation else "-"
        proximal, distal = _score_segments_from_payload(marker.equation)
        local_note = ""
        if marker.method in {"score", "sara", "sara_direction"} | set(PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS):
            local_note = (
                f" | global marker added to marker pool; local offsets reserved for proximal={proximal or '-'} "
                f"and distal={distal or '-'}"
            )
        lines.append(
            f"- {marker.name}: method={marker.method}, segment={marker.segment_name}, source={source}, "
            f"settings={equation}{local_note}"
        )
    lines.extend(["", "Anatomical frames:"])
    lines.extend(_anatomical_frame_instruction_lines(workflow_draft, prefix="- "))
    lines.extend(["", "Chain definition:"])
    for setting in workflow_draft.segment_settings:
        anthropometry = setting.anthropometry_model or "-"
        mass = "-" if setting.anthropometry_mass is None else f"{setting.anthropometry_mass:g} kg"
        length = "-" if setting.segment_length is None else f"{setting.segment_length:g}"
        lines.append(
            f"- {setting.segment_name}: translations={setting.translations or '-'}, rotations={setting.rotations or '-'}, "
            f"child_translation={setting.child_translation}, initial_rotation={setting.initial_rotation_method}, "
            f"anthropometry={anthropometry}, sex={setting.anthropometry_sex or '-'}, mass={mass}, length={length}"
        )
    return tuple(lines)


def _c3d_assignment_log_name(preset: C3dModelPreset, assignment) -> str:
    if preset in {C3dModelPreset.MOTIVE_57, C3dModelPreset.MOTIVE_57_ISB} and assignment.role == "main":
        return assignment.source_path or "*Static.c3d"
    return assignment.source_path or assignment.generic_name


def _anatomical_frame_instruction_lines(workflow_draft, prefix: str = "") -> tuple[str, ...]:
    """
    Return one readable local-frame recipe per segment.

    Internally the draft stores the two user-defined vectors as two axis records because that is what the model
    builder consumes. The GUI presents them as one frame recipe: origin, vector 1, vector 2, and the kept vector.
    """
    lines = []
    for group in workflow_draft.segment_marker_groups:
        axes = tuple(
            axis
            for axis in workflow_draft.axes
            if axis.segment_name == group.segment_name and not _is_virtual_feature_axis(axis)
        )[:2]
        if len(axes) == 0:
            continue
        origin = ",".join(axes[0].origin_markers) if len(axes[0].origin_markers) != 0 else "-"
        vector_lines = []
        for index, axis in enumerate(axes, start=1):
            start = ",".join(axis.start_markers) if len(axis.start_markers) != 0 else "-"
            if len(axis.end_markers) == 0:
                vector_source = start
            else:
                vector_source = f"{start} -> {','.join(axis.end_markers)}"
            keep = ", keep" if axis.keep_vector else ""
            vector_lines.append(f"v{index}={axis.axis}{keep}: {vector_source}")
        if len(axes) < 2:
            vector_lines.append("v2=missing")
        lines.append(f"{prefix}{group.segment_name}: origin={origin} | " + " | ".join(vector_lines))
    return tuple(lines)


def _split_marker_names(text: str) -> tuple[str, ...]:
    """
    Split a comma/semicolon separated marker list while preserving duplicated markers.
    """
    return tuple(marker.strip() for marker in text.replace(";", ",").split(",") if marker.strip())


def _virtual_axis_name_from_feature_list_text(text: str) -> str | None:
    """
    Extract an axis name from the virtual marker/axis list item text.
    """
    if not text.startswith("[axis]"):
        return None
    axis_name = text.removeprefix("[axis]").split("|", maxsplit=1)[0].strip()
    return axis_name or None


def _virtual_feature_list_labels(workflow_draft) -> tuple[str, ...]:
    """
    Return the rows shown in the Virtual markers and axes list.

    Functional SARA axes are listed first because projected knee points and anatomical frames depend on them. With a
    compact list height, this keeps the reusable axes visible without scrolling.
    """
    axis_labels = tuple(
        f"[axis] {axis.name} | {axis.segment_name} | AoR ({axis.method}) | {axis.source or 'functional trial'}"
        for axis in workflow_draft.axes
        if _is_virtual_feature_axis(axis)
    )
    marker_labels = tuple(
        f"{feature.name} | {feature.segment_name} | {_virtual_marker_method_display(feature.method)} | "
        f"{feature.source}"
        for feature in workflow_draft.virtual_markers
    )
    if len(axis_labels) == 0 and len(marker_labels) == 0:
        return ("No additional virtual feature required by this preset.",)
    return axis_labels + marker_labels


def _axis_projection_point_markers_from_payload(text: str) -> tuple[str, ...]:
    """
    Extract markers defining the point to project for an axis-projection virtual marker.
    """
    payload = _key_value_payload(text)
    point_text = payload.get("point", text)
    return _split_marker_names(point_text)


def _axis_projection_axis_from_payload(
    text: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """
    Extract the axis source for an axis-projection virtual marker.

    Returns ``(axis_name, axis_start_markers, axis_end_markers)``. ``axis_name`` is used when projecting onto an
    existing virtual axis such as a SARA AoR; start/end markers are used for marker-defined axes.
    """
    payload = _key_value_payload(text)
    return (
        payload.get("axis", ""),
        _split_marker_names(payload.get("axis_start", "")),
        _split_marker_names(payload.get("axis_end", "")),
    )


def _key_value_payload(text: str) -> dict[str, str]:
    """
    Parse semicolon-separated ``key=value`` snippets.
    """
    values = {}
    for item in text.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", maxsplit=1)
        values[key.strip()] = value.strip()
    return values


def _default_rab2002_payload(marker_name: str = "", segment_name: str = "") -> str:
    """
    Return a side-aware editable Rab 2002 marker payload.
    """
    prefix_source = marker_name.strip() or segment_name.strip()
    side = prefix_source[:1].upper()
    if side not in {"L", "R"}:
        side = "R"
    return f"point={side}CAJ; mid={side}HME,{side}HLE; fraction=0.17"


def _rab2002_markers_from_payload(
    text: str, marker_name: str = "", segment_name: str = ""
) -> tuple[str, tuple[str, ...], float]:
    """
    Extract CAJ/acromion marker, epicondyle markers, and fraction from a Rab 2002 payload.
    """
    payload = _key_value_payload(text)
    defaults = _key_value_payload(_default_rab2002_payload(marker_name, segment_name))
    point_marker = payload.get("point", payload.get("caj", payload.get("acromion", defaults["point"]))).strip()
    mid_markers = _split_marker_names(payload.get("mid", payload.get("epicondyles", defaults["mid"])))
    if len(mid_markers) > 2:
        mid_markers = mid_markers[:2]
    try:
        fraction = float(payload.get("fraction", defaults["fraction"]))
    except ValueError:
        fraction = float(defaults["fraction"])
    return point_marker, mid_markers, fraction


def _rab2002_geometry(c3d_data, source: str, marker_name: str, segment_name: str, frame_index: int) -> (
    tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    | None
):
    """
    Return CAJ, epicondyle midpoint, and Rab 2002 GJC for the requested frame.
    """
    if c3d_data is None:
        return None
    point_marker, mid_markers, fraction = _rab2002_markers_from_payload(source, marker_name, segment_name)
    if point_marker not in c3d_data.marker_names or len(mid_markers) != 2:
        return None
    if any(marker_name not in c3d_data.marker_names for marker_name in mid_markers):
        return None
    caj = _marker_frame_position(c3d_data, point_marker, frame_index)
    epicondyle_mid = _mean_frame_position(c3d_data, mid_markers, frame_index)
    if caj is None or epicondyle_mid is None:
        return None
    caj_array = np.asarray(caj, dtype=float)
    epicondyle_mid_array = np.asarray(epicondyle_mid, dtype=float)
    gjc = caj_array + float(fraction) * (epicondyle_mid_array - caj_array)
    return caj, epicondyle_mid, tuple(float(value) for value in gjc)


def _virtual_axis_from_source_names(source_names: tuple[str, ...], axes) -> object | None:
    """
    Return the virtual SARA axis referenced by a source list, if any.
    """
    source_name_set = set(source_names)
    return next(
        (axis for axis in axes if _is_virtual_feature_axis(axis) and axis.name in source_name_set),
        None,
    )


def _score_segments_from_payload(text: str) -> tuple[str, str]:
    """
    Extract proximal/distal segment names from the compact virtual-marker settings payload.
    """
    values = {"proximal": "", "distal": ""}
    for part in text.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", maxsplit=1)
        key = key.strip()
        if key in values:
            values[key] = value.strip()
    return values["proximal"], values["distal"]


def _strip_score_segment_payload(text: str) -> str:
    """
    Return the user helper/equation part from a compact SCoRE/SARA settings payload.
    """
    helper_parts = []
    for part in text.split(";"):
        stripped = part.strip()
        if stripped.startswith("helper="):
            helper_parts.append(stripped.split("=", maxsplit=1)[1].strip())
        elif stripped and not stripped.startswith("proximal=") and not stripped.startswith("distal="):
            helper_parts.append(stripped)
    return "; ".join(helper_parts)


def _segment_preview_color(segment_index: int) -> str:
    """
    Return a stable color for markers attached to one segment in C3D previews.
    """
    colors = (
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#ca8a04",
        "#7c3aed",
        "#0891b2",
        "#db2777",
        "#4b5563",
    )
    return colors[segment_index % len(colors)]


def _marker_preview_position(c3d_data, marker_name: str) -> tuple[float, float, float] | None:
    """
    Return the mean 3D position of one visible C3D marker.
    """
    if c3d_data is None or marker_name not in c3d_data.marker_names:
        return None
    values = c3d_data.get_position((marker_name,))[:3, 0, :]
    if np.isnan(values).all():
        return None
    point = np.nanmean(values, axis=1)
    return _finite_point3d(point)


def _marker_frame_position(c3d_data, marker_name: str, frame_index: int) -> tuple[float, float, float] | None:
    """
    Return the 3D position of one C3D marker at a given frame.
    """
    if c3d_data is None or marker_name not in c3d_data.marker_names:
        return None
    frame_index = max(0, min(frame_index, c3d_data.nb_frames - 1))
    point = c3d_data.get_position((marker_name,))[:3, 0, frame_index]
    return _finite_point3d(point)


def _mean_preview_position(c3d_data, marker_names: tuple[str, ...]) -> tuple[float, float, float] | None:
    """
    Return the mean point for a possibly duplicated marker list.
    """
    points = [_marker_preview_position(c3d_data, marker_name) for marker_name in marker_names]
    points = [point for point in points if point is not None]
    if len(points) == 0:
        return None
    return tuple(float(value) for value in np.mean(np.asarray(points, dtype=float), axis=0))


def _mean_frame_position(
    c3d_data, marker_names: tuple[str, ...], frame_index: int
) -> tuple[float, float, float] | None:
    """
    Return the mean point for a possibly duplicated marker list at one frame.
    """
    points = [_marker_frame_position(c3d_data, marker_name, frame_index) for marker_name in marker_names]
    points = [point for point in points if point is not None]
    if len(points) == 0:
        return None
    return tuple(float(value) for value in np.mean(np.asarray(points, dtype=float), axis=0))


def _mean_marker_series(c3d_data, marker_names: tuple[str, ...]) -> np.ndarray:
    """
    Return the mean marker-group trajectory over all frames.
    """
    marker_positions = c3d_data.markers_center_position(marker_names)
    return marker_positions[:3, :]


def _scaled_axis_end_point(
    c3d_data,
    start_point: tuple[float, float, float],
    raw_end_point: tuple[float, float, float],
    marker_names: tuple[str, ...],
    frame_index: int,
) -> tuple[float, float, float]:
    """
    Return an axis end point long enough to be readable in a marker preview.
    """
    start = np.asarray(start_point, dtype=float)
    direction = np.asarray(raw_end_point, dtype=float) - start
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0 or not np.isfinite(direction_norm):
        return raw_end_point
    visual_length = _marker_frame_span(c3d_data, marker_names, frame_index) * 0.25
    if visual_length == 0 or not np.isfinite(visual_length):
        visual_length = direction_norm
    end = start + direction / direction_norm * visual_length
    return tuple(float(value) for value in end)


def _sara_static_fallback_axis_line(
    c3d_data,
    payload: dict[str, str],
    expected_start_markers: tuple[str, ...],
    expected_end_markers: tuple[str, ...],
    sara_direction: np.ndarray,
    frame_index: int,
    scale_marker_names: tuple[str, ...],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """
    Return the anatomical fallback line when a SARA direction is too far from the static epicondylar axis.
    """
    limit = _sara_static_deviation_limit_from_payload(payload)
    if limit is None or c3d_data is None or len(expected_start_markers) == 0 or len(expected_end_markers) == 0:
        return None
    required_markers = expected_start_markers + expected_end_markers
    if any(marker_name not in c3d_data.marker_names for marker_name in required_markers):
        return None
    expected_direction = np.nanmean(
        _mean_marker_series(c3d_data, expected_end_markers) - _mean_marker_series(c3d_data, expected_start_markers),
        axis=1,
    )
    deviation = _axis_angle_degrees(sara_direction, expected_direction)
    if not np.isfinite(deviation) or deviation <= limit:
        return None
    start_point = _mean_frame_position(c3d_data, expected_start_markers, frame_index)
    raw_end_point = _mean_frame_position(c3d_data, expected_end_markers, frame_index)
    if start_point is None or raw_end_point is None:
        return None
    return (
        start_point,
        _scaled_axis_end_point(c3d_data, start_point, raw_end_point, scale_marker_names, frame_index),
    )


def _marker_frame_span(c3d_data, marker_names: tuple[str, ...], frame_index: int) -> float:
    """
    Return the largest marker spread at one frame.
    """
    points = [_marker_frame_position(c3d_data, marker_name, frame_index) for marker_name in dict.fromkeys(marker_names)]
    points = [point for point in points if point is not None]
    if len(points) == 0:
        return 0.0
    positions = np.asarray(points, dtype=float)
    return float(np.nanmax(np.ptp(positions, axis=0)))


def _point_series_array(values) -> np.ndarray:
    """
    Normalize one point or a point time series to a ``3-or-4 x frame`` array.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(array.shape[0], 1)
    if array.ndim == 2:
        return array
    return array.reshape(array.shape[0], -1)


def _orthonormal_axes_from_vector_segments(segments) -> dict[str, np.ndarray]:
    """
    Build a local triad from two marker-defined vectors.
    """
    raw_axes = {}
    kept_axis = ""
    for axis_name, keep_vector, start, end in segments[:2]:
        axis_name = axis_name if axis_name in {"x", "y", "z"} else ""
        if not axis_name:
            continue
        vector = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            continue
        raw_axes[axis_name] = vector / norm
        if keep_vector:
            kept_axis = axis_name
    if len(raw_axes) < 2:
        return raw_axes

    axis_names = tuple(raw_axes)
    kept_axis = kept_axis if kept_axis in raw_axes else axis_names[0]
    other_axis = next(axis_name for axis_name in axis_names if axis_name != kept_axis)
    kept_vector = raw_axes[kept_axis]
    other_vector = raw_axes[other_axis] - np.dot(raw_axes[other_axis], kept_vector) * kept_vector
    other_norm = np.linalg.norm(other_vector)
    if other_norm <= 1e-12:
        return {kept_axis: kept_vector}
    axes = {kept_axis: kept_vector, other_axis: other_vector / other_norm}

    missing_axis = next(axis_name for axis_name in ("x", "y", "z") if axis_name not in axes)
    if missing_axis == "x":
        axes["x"] = _normalized_cross(axes["y"], axes["z"])
    elif missing_axis == "y":
        axes["y"] = _normalized_cross(axes["z"], axes["x"])
    else:
        axes["z"] = _normalized_cross(axes["x"], axes["y"])
    return axes


def _normalized_cross(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """
    Return a normalized cross product, or zeros if the vectors are degenerate.
    """
    vector = np.cross(first, second)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-12 else np.zeros(3)


def _average_axis_direction(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """
    Return a sign-consistent mean direction for two axis estimates.
    """
    first = np.asarray(first, dtype=float).reshape(3)
    second = np.asarray(second, dtype=float).reshape(3)
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return np.zeros(3)
    first_unit = first / first_norm
    second_unit = second / second_norm
    if np.dot(first_unit, second_unit) < 0:
        second_unit = -second_unit
    direction = first_unit + second_unit
    direction_norm = np.linalg.norm(direction)
    return direction / direction_norm if direction_norm > 1e-12 else np.zeros(3)


def _rotate_preview_point(
    point: tuple[float, float, float], yaw: float | np.ndarray, pitch: float
) -> tuple[float, float]:
    """
    Project a 3D point with a user-controlled yaw/pitch camera rotation.

    The preview is orthographic: screen coordinates are kept separate from depth. This avoids the oblique shear that
    made rotations look like a mirror flip when the view changed.
    """
    screen_x, screen_y, _depth = _preview_camera_coordinates(point, yaw, pitch)
    return screen_x, screen_y


def _finite_point3d(point) -> tuple[float, float, float] | None:
    """
    Return a finite 3D point, or ``None`` for missing/invalid coordinates.
    """
    array = np.asarray(point, dtype=float).reshape(-1)
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        return None
    return tuple(float(value) for value in array[:3])


def _is_finite_projected_point(point) -> bool:
    """
    Return whether one projected 2D point is safe to pass to Qt painting APIs.
    """
    try:
        values = np.asarray((point[0], point[1]), dtype=float)
    except (TypeError, IndexError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)))


def _is_finite_qpoint(point) -> bool:
    """
    Return whether one QPoint-like object has finite screen coordinates.
    """
    try:
        values = np.asarray((point.x(), point.y()), dtype=float)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)))


def _preview_camera_matrix_for_plane(plane: str) -> np.ndarray:
    """
    Return an orthographic camera matrix aligned with a principal anatomical plane.
    """
    normalized_plane = plane.strip().upper()
    matrices = {
        "XY": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "YZ": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        "ZX": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    }
    if normalized_plane not in matrices:
        raise ValueError(f"Unknown preview plane '{plane}'.")
    return np.asarray(matrices[normalized_plane], dtype=float)


def _preview_camera_matrix_for_subject_view(view: str, markers: dict[str, np.ndarray]) -> np.ndarray:
    """
    Return an orthographic camera matrix aligned with PCA-derived subject views.

    The lab vertical is imposed as the global Z axis. The frontal plane is
    estimated from the marker cloud by finding the dominant horizontal PCA axis.
    """
    vertical = np.asarray((0.0, 0.0, 1.0), dtype=float)
    frontal_horizontal = _subject_frontal_horizontal_axis_from_pca(markers, vertical)
    if frontal_horizontal is None:
        raise ValueError("Face/back/side views require at least three 3D markers.")
    forward = _normalized_cross(vertical, frontal_horizontal)

    normalized_view = view.strip().lower()
    matrices = {
        "face": (frontal_horizontal, vertical, forward),
        "front": (frontal_horizontal, vertical, forward),
        "dos": (-frontal_horizontal, vertical, -forward),
        "back": (-frontal_horizontal, vertical, -forward),
        "cote": (forward, vertical, frontal_horizontal),
        "side": (forward, vertical, frontal_horizontal),
    }
    if normalized_view not in matrices:
        raise ValueError(f"Unknown subject view '{view}'.")
    return np.asarray(matrices[normalized_view], dtype=float)


def _subject_frontal_horizontal_axis_from_pca(
    markers: dict[str, np.ndarray], vertical: np.ndarray
) -> np.ndarray | None:
    """
    Estimate the horizontal axis of the frontal plane from the marker cloud.
    """
    points = np.asarray(
        [point for point in markers.values() if np.all(np.isfinite(point))],
        dtype=float,
    )
    if points.shape[0] < 3:
        return None
    vertical_norm = np.linalg.norm(vertical)
    if vertical_norm <= 1e-12:
        return None
    vertical_unit = vertical / vertical_norm
    centered = points - np.nanmean(points, axis=0)
    horizontal = centered - np.outer(centered @ vertical_unit, vertical_unit)
    if np.linalg.norm(horizontal) <= 1e-12:
        return None
    _u, _s, vh = np.linalg.svd(horizontal, full_matrices=False)
    direction = vh[0]
    direction = direction - np.dot(direction, vertical_unit) * vertical_unit
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return None
    direction = direction / norm
    largest_component = int(np.argmax(np.abs(direction)))
    if direction[largest_component] < 0:
        direction = -direction
    return direction


def _preview_depth(point: tuple[float, float, float], yaw: float | np.ndarray, pitch: float) -> float:
    """
    Return the camera-space depth of a 3D point for painter ordering.
    """
    _screen_x, _screen_y, depth = _preview_camera_coordinates(point, yaw, pitch)
    return depth


def _preview_camera_coordinates(
    point: tuple[float, float, float], yaw: float | np.ndarray, pitch: float
) -> tuple[float, float, float]:
    """
    Rotate a 3D point into a compact camera coordinate system.
    """
    if isinstance(yaw, np.ndarray):
        screen_x, screen_y, depth = np.asarray(yaw, dtype=float) @ np.asarray(point, dtype=float)
        return float(screen_x), float(screen_y), float(depth)
    x, y, z = point
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    yaw_x = cos_yaw * x - sin_yaw * y
    yaw_y = sin_yaw * x + cos_yaw * y
    pitch_y = cos_pitch * yaw_y - sin_pitch * z
    pitch_z = sin_pitch * yaw_y + cos_pitch * z
    return yaw_x, pitch_z, pitch_y


def _segment_parent_choices(workflow_draft, excluded_segment_name: str = "") -> list[str]:
    """
    Return editable parent choices for a new C3D workflow segment.
    """
    return ["", "root", "base"] + [
        group.segment_name
        for group in workflow_draft.segment_marker_groups
        if group.segment_name != excluded_segment_name
    ]


def _parse_optional_float(text: str) -> float | None:
    """
    Parse an optional float from a line edit.
    """
    stripped_text = text.strip()
    return None if stripped_text == "" else float(stripped_text)


def _parse_vector(text: str, expected_length: int) -> list[float]:
    """
    Parse a fixed-length vector from a line edit.
    """
    values = _parse_float_list(text)
    if len(values) != expected_length:
        raise ValueError(f"Expected {expected_length} values, got {len(values)}.")
    return values


def _format_float_list(values: list[float]) -> str:
    """
    Format a float list for display in a line edit.
    """
    return " ".join(str(value) for value in values)


def _format_optional_float(value: float | None) -> str:
    """
    Format an optional float for display in a line edit.
    """
    return "" if value is None else str(value)


def _has_off_diagonal_inertia(inertia_matrix: list[list[float]]) -> bool:
    inertia = np.asarray(inertia_matrix, dtype=float)
    return bool(np.any(inertia - np.diag(np.diag(inertia))))


def _project_point(point) -> tuple[float, float]:
    """
    Project one 3D point into a simple isometric 2D view.
    """
    x, y, z = point[:3]
    return (x - 0.6 * y, z + 0.4 * y)


def _fit_projection(
    points: list[tuple[float, float]],
    width: int,
    height: int,
    point_type,
    zoom: float = 1.0,
):
    """
    Build a transform that fits projected points inside a widget rectangle.
    """
    finite_points = [point for point in points if _is_finite_projected_point(point)]
    fallback_x = width * 0.5
    fallback_y = height * 0.5
    if len(finite_points) == 0:

        def fallback_transform(_point: tuple[float, float]):
            return point_type(fallback_x, fallback_y)

        return fallback_transform
    points = finite_points
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    scale = 0.8 * min(width / span_x, height / span_y) * zoom
    offset_x = (width - scale * (min_x + max_x)) / 2
    offset_y = (height + scale * (min_y + max_y)) / 2

    def transform(point: tuple[float, float]):
        if not _is_finite_projected_point(point):
            return point_type(fallback_x, fallback_y)
        return point_type(offset_x + scale * point[0], offset_y - scale * point[1])

    return transform


def _nearest_projected_segment(projected_positions: dict[str, object], clicked_point, max_distance: float = 12.0):
    """
    Return the nearest projected segment if the click lands close enough.
    """
    nearest_name = None
    nearest_distance = None
    for name, point in projected_positions.items():
        dx = point.x() - clicked_point.x()
        dy = point.y() - clicked_point.y()
        distance = (dx**2 + dy**2) ** 0.5
        if nearest_distance is None or distance < nearest_distance:
            nearest_name = name
            nearest_distance = distance
    if nearest_distance is None or nearest_distance > max_distance:
        return None
    return nearest_name
