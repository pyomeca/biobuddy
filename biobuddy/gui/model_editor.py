import json
import math
import re
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..validation import MuscleValidator
from .segment_editor import (
    SegmentEditorData,
    apply_segment_editor_data,
    get_segment_editor_data,
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
    c3d_virtual_marker_method_examples,
    c3d_workflow_draft,
    c3d_workflow_progress,
    c3d_workflow_summary,
    clear_c3d_file_role_from_draft,
    remove_axis_from_draft,
    remove_segment_from_draft,
    remove_virtual_marker_from_draft,
    set_segment_marker_technical,
    unassign_markers_from_segment,
    update_segment_parent_in_draft,
    update_segment_settings_in_draft,
    validate_c3d_workflow_draft,
)
from ..utils.marker_data import C3dData

PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS = {
    "hara2016_hip": "Hara 2016 hip",
    "harrington2007_hip": "Harrington 2007 hip",
    "sobral2025_shoulder": "Sobral 2025 shoulder",
}


def launch_model_editor() -> None:
    """
    Launch the Qt desktop model editor.
    """
    try:
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QPainter, QPen
        from PySide6.QtWidgets import (
            QAbstractItemView,
            QApplication,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QFileDialog,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QCheckBox,
            QLabel,
            QLineEdit,
            QInputDialog,
            QListWidget,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QSplitter,
            QSlider,
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
        qt_extended_selection = QAbstractItemView.SelectionMode.ExtendedSelection
        qpaint_antialiasing = QPainter.RenderHint.Antialiasing
        get_event_position = lambda event: event.position()
    except ImportError:
        try:
            from PyQt5.QtCore import QPointF, Qt
            from PyQt5.QtGui import QColor, QPainter, QPen
            from PyQt5.QtWidgets import (
                QAbstractItemView,
                QApplication,
                QComboBox,
                QDialog,
                QDialogButtonBox,
                QFileDialog,
                QFormLayout,
                QGroupBox,
                QHBoxLayout,
                QCheckBox,
                QLabel,
                QLineEdit,
                QInputDialog,
                QListWidget,
                QMainWindow,
                QMessageBox,
                QPushButton,
                QScrollArea,
                QSplitter,
                QSlider,
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
            qt_extended_selection = QAbstractItemView.ExtendedSelection
            qpaint_antialiasing = QPainter.Antialiasing
            get_event_position = lambda event: event.localPos()
        except ImportError as error:
            raise ImportError(
                "The model editor requires a working Qt binding. Install BioBuddy with `pip install biobuddy[gui]` "
                "or use an environment where PyQt5 is available."
            ) from error

    def _draw_legend_point(painter, center, color, label: str, text_x: int, text_y: int) -> None:
        """
        Draw one point-style legend entry.
        """
        painter.setPen(QPen(color, 1))
        painter.setBrush(color)
        painter.drawEllipse(center, 4, 4)
        painter.setPen(QPen(QColor("#111827"), 1))
        painter.drawText(text_x, text_y, label)

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
            ("x", (1.0, 0.0, 0.0), QColor("#dc2626")),
            ("y", (0.0, 1.0, 0.0), QColor("#16a34a")),
            ("z", (0.0, 0.0, 1.0), QColor("#2563eb")),
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
                origin.y() + length * projected[1] / norm,
            )
            painter.setPen(QPen(color, 3))
            painter.drawLine(origin, endpoint)
            painter.setPen(QPen(color, 1))
            painter.drawText(endpoint.x() + 3, endpoint.y() - 3, label)

    def _scrollable_widget(widget):
        """
        Wrap a form-heavy tab so all controls remain reachable on smaller screens.
        """
        scroll_area = QScrollArea()
        scroll_area.setWidget(widget)
        scroll_area.setWidgetResizable(True)
        return scroll_area

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

    def _create_axis_vector_controls(index: int) -> dict[str, object]:
        """
        Create the repeated controls used to define one anatomical frame vector.
        """
        start_list = QListWidget()
        start_list.setSelectionMode(qt_extended_selection)
        start_list.setMinimumHeight(56)
        start_list.setMaximumHeight(80)
        start_list.setMaximumWidth(190)
        end_list = QListWidget()
        end_list.setSelectionMode(qt_extended_selection)
        end_list.setMinimumHeight(56)
        end_list.setMaximumHeight(80)
        end_list.setMaximumWidth(190)
        axis_combo = QComboBox()
        axis_combo.setMaximumWidth(160)
        axis_combo.addItems(["x", "y", "z"])
        if index == 1:
            axis_combo.setCurrentText("y")
        keep_checkbox = QCheckBox("Keep this vector")
        keep_checkbox.setChecked(index == 0)
        add_start_button = QPushButton("+")
        add_end_button = QPushButton("+")
        remove_start_button = QPushButton("-")
        remove_end_button = QPushButton("-")
        for button in (add_start_button, add_end_button, remove_start_button, remove_end_button):
            button.setMaximumWidth(34)
            button.setMinimumWidth(34)
        return {
            "start_list": start_list,
            "end_list": end_list,
            "axis_combo": axis_combo,
            "keep_checkbox": keep_checkbox,
            "add_start_button": add_start_button,
            "add_end_button": add_end_button,
            "remove_start_button": remove_start_button,
            "remove_end_button": remove_end_button,
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
            ("Start markers", controls["start_list"], controls["add_start_button"], controls["remove_start_button"]),
            ("End markers", controls["end_list"], controls["add_end_button"], controls["remove_end_button"]),
        ):
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            row = QHBoxLayout()
            buttons = QVBoxLayout()
            buttons.addWidget(add_button)
            buttons.addWidget(remove_button)
            buttons.addStretch()
            row.addLayout(buttons)
            row.addWidget(marker_list)
            column.addLayout(row)
            marker_row.addLayout(column)
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
            self.current_vectors = ()
            self.current_origin_markers = ()
            self.frame_index = 0
            self.yaw = -0.6
            self.pitch = 0.35
            self._last_mouse_position = None

        def set_context(
            self,
            c3d_data,
            marker_names: tuple[str, ...],
            label_marker_names: tuple[str, ...],
            segment_marker_groups: tuple[object, ...],
            selected_segment_name: str,
            axes: tuple[object, ...],
            current_vectors: tuple[tuple[str, tuple[str, ...], tuple[str, ...], bool], ...],
            current_origin_markers: tuple[str, ...] = (),
            frame_index: int = 0,
        ) -> None:
            self.c3d_data = c3d_data
            self.marker_names = marker_names
            self.label_marker_names = label_marker_names
            self.segment_marker_groups = segment_marker_groups
            self.selected_segment_name = selected_segment_name
            self.axes = axes
            self.current_vectors = current_vectors
            self.current_origin_markers = current_origin_markers
            self.frame_index = frame_index
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(qpaint_antialiasing)
            painter.fillRect(self.rect(), QColor("white"))
            if self.c3d_data is None:
                painter.drawText(self.rect(), qt_alignment_center, "Choose a C3D to preview marker axes")
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
            axis_segments = []
            for axis in self.axes:
                start = _mean_frame_position(self.c3d_data, axis.start_markers, self.frame_index)
                end = _mean_frame_position(self.c3d_data, axis.end_markers, self.frame_index)
                if start is not None and end is not None:
                    axis_segments.append((axis.axis, axis.keep_vector, start, end))
            saved_origins = []
            for axis in self.axes:
                origin = _mean_frame_position(self.c3d_data, axis.origin_markers, self.frame_index)
                if origin is not None:
                    saved_origins.append(origin)
            temporary_segments = []
            for axis_name, start_markers, end_markers, keep_vector in self.current_vectors:
                start = _mean_frame_position(self.c3d_data, start_markers, self.frame_index)
                end = _mean_frame_position(self.c3d_data, end_markers, self.frame_index)
                if start is not None and end is not None:
                    temporary_segments.append((axis_name, keep_vector, start, end))
            temporary_origin = _mean_frame_position(self.c3d_data, self.current_origin_markers, self.frame_index)
            points = [point for _, point, _, _, _ in marker_records]
            for _, _, start, end in axis_segments:
                points.extend((start, end))
            for _, _, start, end in temporary_segments:
                points.extend((start, end))
            points.extend(saved_origins)
            if temporary_origin is not None:
                points.append(temporary_origin)
            if len(points) == 0:
                painter.drawText(self.rect(), qt_alignment_center, "No visible marker for the selected segment")
                return
            projected_points = [_rotate_preview_point(point, self.yaw, self.pitch) for point in points]
            transform = _fit_projection(projected_points, self.width(), self.height(), QPointF)

            label_marker_names = set(self.label_marker_names)
            for marker_name, point, segment_name, is_technical, segment_index in marker_records:
                is_selected = segment_name == self.selected_segment_name
                color = QColor(_segment_preview_color(segment_index))
                center = transform(_rotate_preview_point(point, self.yaw, self.pitch))
                painter.setPen(QPen(color, 3 if is_selected else 1))
                painter.setBrush(color)
                radius = 6 if is_selected else 4
                if is_technical:
                    painter.drawRect(int(center.x()) - radius, int(center.y()) - radius, 2 * radius, 2 * radius)
                else:
                    painter.drawEllipse(center, radius, radius)
                if is_selected and marker_name in label_marker_names:
                    painter.drawText(center.x() + 5, center.y() - 5, f"{marker_name} ({segment_name})")

            axis_colors = {"x": "#dc2626", "y": "#16a34a", "z": "#2563eb", "": "#6b7280"}
            for axis_name, keep_vector, start, end in axis_segments:
                painter.setPen(QPen(QColor(axis_colors.get(axis_name, "#6b7280")), 4 if keep_vector else 2))
                painter.drawLine(
                    transform(_rotate_preview_point(start, self.yaw, self.pitch)),
                    transform(_rotate_preview_point(end, self.yaw, self.pitch)),
                )

            for axis_name, keep_vector, start, end in temporary_segments:
                pen = QPen(QColor(axis_colors.get(axis_name, "#6b7280")), 4 if keep_vector else 2)
                pen.setStyle(qt_dash_line)
                painter.setPen(pen)
                painter.drawLine(
                    transform(_rotate_preview_point(start, self.yaw, self.pitch)),
                    transform(_rotate_preview_point(end, self.yaw, self.pitch)),
                )

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

        def _draw_local_frame(self, painter, transform, origin, temporary_segments, scene_points) -> None:
            local_axes = _orthonormal_axes_from_vector_segments(temporary_segments)
            if len(local_axes) == 0:
                return
            positions = np.asarray(scene_points, dtype=float)
            scene_span = float(np.nanmax(np.ptp(positions, axis=0))) if positions.size != 0 else 1.0
            axis_length = max(scene_span * 0.18, 1e-6)
            axis_colors = {"x": "#dc2626", "y": "#16a34a", "z": "#2563eb"}
            origin_array = np.asarray(origin, dtype=float)
            origin_screen = transform(_rotate_preview_point(tuple(origin_array), self.yaw, self.pitch))
            for axis_name in ("x", "y", "z"):
                if axis_name not in local_axes:
                    continue
                endpoint = origin_array + axis_length * local_axes[axis_name]
                endpoint_screen = transform(_rotate_preview_point(tuple(endpoint), self.yaw, self.pitch))
                painter.setPen(QPen(QColor(axis_colors[axis_name]), 3))
                painter.drawLine(origin_screen, endpoint_screen)
                painter.drawText(endpoint_screen.x() + 4, endpoint_screen.y() - 4, axis_name.upper())

        def mousePressEvent(self, event) -> None:
            self._last_mouse_position = get_event_position(event)

        def mouseMoveEvent(self, event) -> None:
            if self._last_mouse_position is None:
                self._last_mouse_position = get_event_position(event)
                return
            position = get_event_position(event)
            self.yaw += (position.x() - self._last_mouse_position.x()) * 0.01
            self.pitch += (position.y() - self._last_mouse_position.y()) * 0.01
            self.pitch = max(-1.4, min(1.4, self.pitch))
            self._last_mouse_position = position
            self.update()

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
            self.virtual_markers = ()
            self.selected_marker_name = ""
            self.selected_method = "pointing"
            self.proximal_segment_name = ""
            self.distal_segment_name = ""
            self.frame_index = 0
            self.show_whole_body = False
            self._score_solution_cache = {}
            self.yaw = -0.6
            self.pitch = 0.35
            self._last_mouse_position = None

        def set_context(
            self,
            c3d_data,
            solution_c3d_data,
            preview_source_label: str,
            groups: tuple[object, ...],
            virtual_markers: tuple[object, ...],
            selected_marker_name: str,
            selected_method: str,
            proximal_segment_name: str,
            distal_segment_name: str,
            frame_index: int,
            show_whole_body: bool,
        ) -> None:
            if c3d_data is not self.c3d_data or solution_c3d_data is not self.solution_c3d_data:
                self._score_solution_cache.clear()
            self.c3d_data = c3d_data
            self.solution_c3d_data = solution_c3d_data
            self.preview_source_label = preview_source_label
            self.groups = groups
            self.virtual_markers = virtual_markers
            self.selected_marker_name = selected_marker_name
            self.selected_method = selected_method
            self.proximal_segment_name = proximal_segment_name
            self.distal_segment_name = distal_segment_name
            self.frame_index = frame_index
            self.show_whole_body = show_whole_body
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(qpaint_antialiasing)
            painter.fillRect(self.rect(), QColor("white"))
            if self.c3d_data is None:
                painter.drawText(self.rect(), qt_alignment_center, "Choose a C3D to preview virtual markers")
                return

            highlighted_segments = {self.proximal_segment_name, self.distal_segment_name}
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
            for marker_name in self.c3d_data.marker_names:
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

            virtual_points = []
            selected_virtual_markers = [
                marker for marker in self.virtual_markers if marker.name == self.selected_marker_name
            ]
            for marker in selected_virtual_markers:
                if marker.method not in {"marker_mean", "axis_projection"}:
                    continue
                point = _mean_frame_position(
                    self.c3d_data,
                    (
                        _axis_projection_point_markers_from_payload(marker.source)
                        if marker.method == "axis_projection"
                        else _split_marker_names(marker.source)
                    ),
                    self.frame_index,
                )
                if point is not None:
                    virtual_points.append((marker.name, point, marker.segment_name))

            solution_points = []
            if self.selected_method == "score":
                solution_points.extend(
                    self._score_solution_points(self.proximal_segment_name, self.distal_segment_name)
                )
            elif self.selected_method in {"sara"} | set(PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS):
                proximal_point = self._technical_segment_center(self.proximal_segment_name)
                distal_point = self._technical_segment_center(self.distal_segment_name)
                if proximal_point is not None:
                    solution_points.append(("parent", proximal_point))
                if distal_point is not None:
                    solution_points.append(("segment", distal_point))

            points = [record[1] for record in marker_records]
            points.extend(point for _, point, _ in virtual_points)
            points.extend(point for _, point in solution_points)
            if len(points) == 0:
                painter.drawText(self.rect(), qt_alignment_center, "No visible marker for this virtual marker context")
                return

            focus_points = [record[1] for record in marker_records if record[-1]]
            focus_points.extend(point for _, point, _ in virtual_points)
            focus_points.extend(point for _, point in solution_points)
            points_to_fit = points if self.show_whole_body or len(focus_points) == 0 else focus_points
            projected_points = [_rotate_preview_point(point, self.yaw, self.pitch) for point in points_to_fit]
            transform = _fit_projection(projected_points, self.width(), self.height(), QPointF)

            for marker_name, point, segment_name, is_technical, segment_index, is_highlighted in marker_records:
                color = QColor(_segment_preview_color(segment_index)) if is_highlighted else QColor("#cbd5e1")
                center = transform(_rotate_preview_point(point, self.yaw, self.pitch))
                painter.setPen(QPen(color, 1))
                painter.setBrush(color)
                if is_technical:
                    size = 8 if is_highlighted else 5
                    painter.drawRect(int(center.x()) - size // 2, int(center.y()) - size // 2, size, size)
                else:
                    radius = 4 if is_highlighted else 2
                    painter.drawEllipse(center, radius, radius)
                if is_highlighted and segment_name == self.distal_segment_name:
                    painter.drawText(center.x() + 5, center.y() - 5, f"{marker_name} ({segment_name})")

            painter.setPen(QPen(QColor("#7c3aed"), 2))
            painter.setBrush(QColor("#7c3aed"))
            for marker_name, point, segment_name in virtual_points:
                center = transform(_rotate_preview_point(point, self.yaw, self.pitch))
                painter.drawEllipse(center, 7, 7)
                painter.drawText(center.x() + 8, center.y() - 8, f"{marker_name} | {segment_name}")

            solution_colors = {"parent": "#f97316", "segment": "#0891b2", "mean": "#111827"}
            solution_offsets = {"parent": (10, -12), "segment": (10, 6), "mean": (10, 24)}
            for label, point in solution_points:
                center = transform(_rotate_preview_point(point, self.yaw, self.pitch))
                painter.setPen(QPen(QColor(solution_colors[label]), 3))
                painter.setBrush(QColor("white"))
                painter.drawEllipse(center, 8, 8)
                prefix = "SCoRE" if self.selected_method == "score" else "center"
                offset_x, offset_y = solution_offsets.get(label, (8, -8))
                painter.drawText(center.x() + offset_x, center.y() + offset_y, f"{prefix} {label}")
            if len(solution_points) >= 2:
                painter.setPen(QPen(QColor("#111827"), 1))
                painter.drawLine(
                    transform(_rotate_preview_point(solution_points[0][1], self.yaw, self.pitch)),
                    transform(_rotate_preview_point(solution_points[1][1], self.yaw, self.pitch)),
                )

            painter.setPen(QPen(QColor("#111827"), 1))
            if self.preview_source_label:
                painter.drawText(12, self.height() - 12, f"Preview: {self.preview_source_label}")
            _draw_preview_orientation_axes(painter, self.width(), self.height(), self.yaw, self.pitch)
            self._draw_legend(painter)

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

        def _score_solution_points(
            self, parent_segment_name: str, segment_name: str
        ) -> tuple[tuple[str, tuple[float, float, float]], ...]:
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
            cache_key = (id(functional_data), parent_marker_names, child_marker_names)
            if cache_key in self._score_solution_cache:
                cor_parent_local, cor_child_local = self._score_solution_cache[cache_key]
            else:
                try:
                    from ..components.generic.rigidbody.segment_coordinate_system import SegmentCoordinateSystemUtils
                    from ..model_modifiers.joint_center_tool import Score

                    parent_functional_marker_data = functional_data.get_partial_dict_data(parent_marker_names)
                    child_functional_marker_data = functional_data.get_partial_dict_data(child_marker_names)
                    rt_parent_func = SegmentCoordinateSystemUtils.rigidify(
                        functional_data=parent_functional_marker_data
                    )
                    rt_child_func = SegmentCoordinateSystemUtils.rigidify(functional_data=child_functional_marker_data)
                    _, cor_parent_local, cor_child_local, _, _ = Score.perform_algorithm(rt_parent_func, rt_child_func)
                    self._score_solution_cache[cache_key] = (cor_parent_local, cor_child_local)
                except Exception:
                    return ()
            try:
                from ..components.generic.rigidbody.segment_coordinate_system import SegmentCoordinateSystemUtils

                parent_preview_marker_data = preview_data.get_partial_dict_data(parent_marker_names)
                child_preview_marker_data = preview_data.get_partial_dict_data(child_marker_names)
                rt_parent_preview = SegmentCoordinateSystemUtils.rigidify(parent_preview_marker_data)
                rt_child_preview = SegmentCoordinateSystemUtils.rigidify(child_preview_marker_data)
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
            return solution_points

        def _draw_legend(self, painter) -> None:
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.setBrush(QColor(255, 255, 255, 225))
            painter.drawRect(8, 8, 275, 100)
            painter.drawText(14, 24, "Legend")
            _draw_legend_point(painter, QPointF(20, 40), QColor("#2563eb"), "Selected segment/parent marker", 36, 44)
            painter.setPen(QPen(QColor("#2563eb"), 1))
            painter.setBrush(QColor("#2563eb"))
            painter.drawRect(16, 52, 8, 8)
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.drawText(36, 62, "Technical marker")
            _draw_legend_point(painter, QPointF(20, 74), QColor("#7c3aed"), "Virtual marker", 36, 78)
            _draw_legend_point(painter, QPointF(20, 92), QColor("#cbd5e1"), "Other functional C3D marker", 36, 96)

        def mousePressEvent(self, event) -> None:
            self._last_mouse_position = get_event_position(event)

        def mouseMoveEvent(self, event) -> None:
            if self._last_mouse_position is None:
                self._last_mouse_position = get_event_position(event)
                return
            position = get_event_position(event)
            self.yaw += (position.x() - self._last_mouse_position.x()) * 0.01
            self.pitch += (position.y() - self._last_mouse_position.y()) * 0.01
            self.pitch = max(-1.4, min(1.4, self.pitch))
            self._last_mouse_position = position
            self.update()

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
            self.frame_index = 0
            self.yaw = -0.6
            self.pitch = 0.35
            self._last_mouse_position = None

        def set_context(
            self, c3d_data, groups: tuple[object, ...], selected_segment_name: str, frame_index: int
        ) -> None:
            self.c3d_data = c3d_data
            self.groups = groups
            self.selected_segment_name = selected_segment_name
            self.frame_index = frame_index
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(qpaint_antialiasing)
            painter.fillRect(self.rect(), QColor("white"))
            if self.c3d_data is None:
                painter.drawText(self.rect(), qt_alignment_center, "Choose a C3D to preview technical segments")
                return

            marker_records = []
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
            if len(marker_records) == 0:
                painter.drawText(self.rect(), qt_alignment_center, "No assigned marker is visible in this frame")
                return

            projected_points = [
                _rotate_preview_point(point, self.yaw, self.pitch) for _, point, _, _, _ in marker_records
            ]
            transform = _fit_projection(projected_points, self.width(), self.height(), QPointF)

            for marker_name, point, segment_name, is_technical, segment_index in marker_records:
                is_selected = segment_name == self.selected_segment_name
                color = QColor(_segment_preview_color(segment_index))
                center = transform(_rotate_preview_point(point, self.yaw, self.pitch))
                painter.setPen(QPen(color, 3 if is_selected else 1))
                painter.setBrush(color)
                radius = 6 if is_selected else 4
                if is_technical:
                    painter.drawRect(int(center.x()) - radius, int(center.y()) - radius, 2 * radius, 2 * radius)
                else:
                    painter.drawEllipse(center, radius, radius)
                if is_selected:
                    painter.drawText(center.x() + 6, center.y() - 6, f"{marker_name} ({segment_name})")

            _draw_preview_orientation_axes(painter, self.width(), self.height(), self.yaw, self.pitch)
            self._draw_legend(painter)

        def _draw_legend(self, painter) -> None:
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.setBrush(QColor(255, 255, 255, 225))
            painter.drawRect(8, 8, 250, 68)
            painter.drawText(14, 24, "Legend")
            _draw_legend_point(painter, QPointF(20, 40), QColor("#2563eb"), "Additional/anatomical", 36, 44)
            painter.setPen(QPen(QColor("#2563eb"), 1))
            painter.setBrush(QColor("#2563eb"))
            painter.drawRect(16, 52, 8, 8)
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.drawText(36, 62, "Technical")

        def mousePressEvent(self, event) -> None:
            self._last_mouse_position = get_event_position(event)

        def mouseMoveEvent(self, event) -> None:
            if self._last_mouse_position is None:
                self._last_mouse_position = get_event_position(event)
                return
            position = get_event_position(event)
            self.yaw += (position.x() - self._last_mouse_position.x()) * 0.01
            self.pitch += (position.y() - self._last_mouse_position.y()) * 0.01
            self.pitch = max(-1.4, min(1.4, self.pitch))
            self._last_mouse_position = position
            self.update()

    class C3dModelCreationDialog(QDialog):
        """
        Dialog for the C3D-driven model creation workflow.
        """

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("New model from C3D")
            self.presets = supported_c3d_model_presets()
            self.c3d_data = None
            self._c3d_data_cache = {}
            self._is_auto_assigning_c3d_files = False
            self.c3d_folder_path = ""
            self.workflow_draft = c3d_workflow_draft(self.presets[0])
            self.workflow_marker_pool = _marker_pool_from_draft(self.workflow_draft)

            self.preset_combo = QComboBox()
            for preset in self.presets:
                self.preset_combo.addItem(_c3d_preset_label(preset))
            self.preset_combo.currentIndexChanged.connect(self._update_preset_details)

            self.c3d_path = QLineEdit()
            self.c3d_path.setReadOnly(True)
            self.c3d_path.hide()
            self.main_c3d_name_label = QLabel("No main C3D selected")
            self.choose_c3d_button = QPushButton("Choose C3D file")
            self.choose_c3d_button.clicked.connect(self._choose_c3d_file)
            self.generate_template_button = QPushButton("Generate template")
            self.generate_template_button.clicked.connect(self._generate_template)
            self.c3d_folder_edit = QLineEdit()
            self.c3d_folder_edit.setReadOnly(True)
            self.choose_c3d_folder_button = QPushButton("Choose C3D folder")
            self.choose_c3d_folder_button.clicked.connect(self._choose_c3d_folder)
            self.generate_log_button = QPushButton("Generate log")
            self.generate_log_button.clicked.connect(self._update_generation_log)
            self.generate_python_code_button = QPushButton("Generate Python code")
            self.generate_python_code_button.clicked.connect(self._generate_python_code)
            self.generation_log_edit = QTextEdit()
            self.generation_log_edit.setReadOnly(True)
            self.generation_log_edit.setMinimumHeight(160)

            self.status_label = QLabel()
            self.summary_label = QLabel()
            self.feature_list = QListWidget()
            self.feature_list.setMinimumHeight(180)
            self.feature_list.itemSelectionChanged.connect(self._load_selected_virtual_marker_into_form)
            self.step_list = QListWidget()
            self.marker_list = QListWidget()
            self.marker_list.setSelectionMode(qt_extended_selection)
            self.marker_mapping_label = QLabel("Load a C3D to check marker names against the selected template.")
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
            self.axis_marker_source_list = QListWidget()
            self.axis_marker_source_list.setSelectionMode(qt_extended_selection)
            self.axis_marker_source_list.setMinimumWidth(140)
            self.axis_marker_source_list.setMaximumWidth(180)
            self.axis_marker_source_list.setMinimumHeight(220)
            self.axis_origin_marker_list = QListWidget()
            self.axis_origin_marker_list.setSelectionMode(qt_extended_selection)
            self.axis_origin_marker_list.setMinimumHeight(52)
            self.axis_origin_marker_list.setMaximumHeight(76)
            self.axis_origin_marker_list.setMaximumWidth(190)
            self.add_axis_origin_marker_button = QPushButton("+")
            self.remove_axis_origin_marker_button = QPushButton("-")
            for button in (self.add_axis_origin_marker_button, self.remove_axis_origin_marker_button):
                button.setMaximumWidth(34)
                button.setMinimumWidth(34)
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
                controls["axis_combo"].currentTextChanged.connect(self._update_segment_axis_preview)
                controls["keep_checkbox"].stateChanged.connect(
                    lambda checked=False, vector_index=index: self._ensure_single_kept_axis_vector(vector_index)
                )
            self.save_segment_axis_button = QPushButton("Add/update anatomical frame vectors")
            self.save_segment_axis_button.clicked.connect(self._save_segment_axis_from_lists)
            self.segment_axis_preview = C3dSegmentAxisPreviewWidget()
            self.segment_axis_preview.setMinimumWidth(520)
            self.segment_axis_preview.setMinimumHeight(520)
            self.anatomical_frame_slider = QSlider(qt_horizontal)
            self.anatomical_frame_slider.setEnabled(False)
            self.anatomical_frame_slider.valueChanged.connect(self._update_segment_axis_preview)
            self.anatomical_frame_label = QLabel("Frame 1/1")
            self.virtual_marker_name_edit = QLineEdit()
            self.virtual_marker_suggested_name_label = QLabel()
            self.use_suggested_virtual_marker_name_button = QPushButton("Use suggested name")
            self.use_suggested_virtual_marker_name_button.clicked.connect(self._use_suggested_virtual_marker_name)
            self.virtual_marker_segment_combo = QComboBox()
            self.virtual_marker_segment_combo.currentTextChanged.connect(self._update_suggested_virtual_marker_name)
            self.virtual_marker_segment_combo.currentTextChanged.connect(self._sync_virtual_marker_segment_context)
            self.virtual_marker_method_combo = QComboBox()
            self.virtual_marker_method_combo.addItems(
                [
                    "pointing",
                    "score",
                    "sara",
                    "marker_mean",
                    "axis_projection",
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
            self.virtual_marker_source_label = QLabel("Source")
            self.virtual_marker_source_edit.textChanged.connect(self._update_virtual_marker_preview)
            self.virtual_marker_source_edit.hide()
            self.virtual_marker_source_label.hide()
            self.virtual_marker_c3d_file_combo = QComboBox()
            self.virtual_marker_c3d_file_combo.currentTextChanged.connect(self._update_virtual_marker_preview)
            self.virtual_marker_c3d_file_combo.currentTextChanged.connect(self._update_suggested_virtual_marker_name)
            self.browse_virtual_marker_c3d_button = QPushButton("Browse C3D")
            self.browse_virtual_marker_c3d_button.clicked.connect(self._browse_virtual_marker_c3d_source)
            self.virtual_marker_show_functional_c3d_checkbox = QCheckBox("Preview functional C3D")
            self.virtual_marker_show_functional_c3d_checkbox.setChecked(True)
            self.virtual_marker_show_functional_c3d_checkbox.stateChanged.connect(self._update_virtual_marker_preview)
            self.virtual_marker_whole_body_preview_checkbox = QCheckBox("Whole body view")
            self.virtual_marker_whole_body_preview_checkbox.setChecked(False)
            self.virtual_marker_whole_body_preview_checkbox.stateChanged.connect(self._update_virtual_marker_preview)
            self.virtual_marker_frame_slider = QSlider(qt_horizontal)
            self.virtual_marker_frame_slider.setEnabled(False)
            self.virtual_marker_frame_slider.valueChanged.connect(self._update_virtual_marker_preview)
            self.virtual_marker_frame_label = QLabel("Frame 1/1")
            self.virtual_marker_equation_edit = QLineEdit()
            self.virtual_marker_equation_label = QLabel("Settings")
            self.virtual_marker_equation_edit.hide()
            self.virtual_marker_equation_label.hide()
            self.virtual_marker_c3d_role_combo = QComboBox()
            self.virtual_marker_c3d_role_combo.hide()
            self.virtual_marker_parent_label = QLabel("-")
            self.virtual_marker_technical_markers_label = QLabel()
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
            self.virtual_marker_info_label.setWordWrap(True)
            self.save_virtual_marker_button = QPushButton("Save virtual marker/axis")
            self.save_virtual_marker_button.clicked.connect(self._save_workflow_virtual_marker_from_form)
            self.virtual_marker_preview = C3dVirtualMarkerPreviewWidget()
            self.segment_settings_list = QListWidget()
            self.segment_settings_list.itemSelectionChanged.connect(self._load_selected_segment_settings_into_form)
            self.settings_translations_edit = QLineEdit()
            self.settings_rotations_edit = QLineEdit()
            self.settings_q_min_edit = QLineEdit()
            self.settings_q_max_edit = QLineEdit()
            self.settings_child_translation_checkbox = QCheckBox("Allow child translation")
            self.settings_initial_rotation_method_combo = QComboBox()
            self.settings_initial_rotation_method_combo.addItems(["identity", "matrix", "anatomical_c3d"])
            self.settings_initial_rotation_method_combo.currentTextChanged.connect(
                self._sync_initial_rotation_source_fields
            )
            self.settings_initial_rotation_source_edit = QLineEdit()
            self.settings_initial_rotation_c3d_combo = QComboBox()
            self.settings_initial_rotation_c3d_combo.currentTextChanged.connect(
                self._mirror_initial_rotation_c3d_source
            )
            self.browse_initial_rotation_c3d_button = QPushButton("Browse C3D")
            self.browse_initial_rotation_c3d_button.clicked.connect(self._browse_initial_rotation_c3d_source)
            self.file_role_list = QListWidget()
            self.issue_list = QListWidget()
            self.example_list = QListWidget()
            self.add_segment_button = QPushButton("Add segment")
            self.add_segment_button.clicked.connect(self._add_workflow_segment)
            self.remove_segment_button = QPushButton("Remove segment")
            self.remove_segment_button.clicked.connect(self._remove_workflow_segment)
            self.assign_marker_button = QPushButton("Assign selected markers")
            self.assign_marker_button.clicked.connect(self._assign_workflow_marker)
            self.unassign_marker_button = QPushButton("Remove selected markers")
            self.unassign_marker_button.clicked.connect(self._unassign_workflow_marker)
            self.add_virtual_marker_button = QPushButton("Add/edit virtual marker/axis")
            self.add_virtual_marker_button.clicked.connect(self._add_workflow_virtual_marker)
            self.remove_virtual_marker_button = QPushButton("Remove virtual marker/axis")
            self.remove_virtual_marker_button.clicked.connect(self._remove_workflow_virtual_marker)
            self.edit_segment_settings_button = QPushButton("Apply segment settings")
            self.edit_segment_settings_button.clicked.connect(self._apply_workflow_segment_settings_from_form)
            self.assign_c3d_role_button = QPushButton("Assign C3D file")
            self.assign_c3d_role_button.clicked.connect(self._assign_workflow_c3d_role)
            self.clear_c3d_role_button = QPushButton("Clear C3D file")
            self.clear_c3d_role_button.clicked.connect(self._clear_workflow_c3d_role)

            buttons = QDialogButtonBox(_dialog_button("Ok") | _dialog_button("Cancel"))
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)

            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Model preset"))
            layout.addWidget(self.preset_combo)
            folder_row = QHBoxLayout()
            folder_row.addWidget(QLabel("C3D folder"))
            folder_row.addWidget(self.c3d_folder_edit)
            folder_row.addWidget(self.choose_c3d_folder_button)
            layout.addLayout(folder_row)
            c3d_row = QHBoxLayout()
            c3d_row.addWidget(QLabel("Main C3D"))
            c3d_row.addWidget(self.main_c3d_name_label)
            c3d_row.addStretch()
            c3d_row.addWidget(self.choose_c3d_button)
            c3d_row.addWidget(self.generate_template_button)
            layout.addLayout(c3d_row)
            layout.addWidget(self.status_label)

            workflow_tabs = QTabWidget()
            workflow_tabs.addTab(self._pipeline_workflow_tab(), "Pipeline")
            workflow_tabs.addTab(self._segment_workflow_tab(), "Technical segment")
            workflow_tabs.addTab(self._virtual_marker_workflow_tab(), "Virtual markers and axes")
            workflow_tabs.addTab(self._anatomical_segment_workflow_tab(), "Anatomical segment")
            workflow_tabs.addTab(self._segment_settings_workflow_tab(), "Segment settings")
            workflow_tabs.addTab(self._file_role_workflow_tab(), "C3D names")
            workflow_tabs.addTab(self.issue_list, "Checks")
            workflow_tabs.addTab(self.example_list, "Examples")
            workflow_tabs.addTab(self.summary_label, "Summary")
            layout.addWidget(workflow_tabs)
            layout.addWidget(buttons)
            self.resize(900, 650)
            self._update_preset_details()

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

        def _pipeline_workflow_tab(self):
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.addWidget(QLabel("Workflow status"))
            layout.addWidget(self.step_list)
            log_row = QHBoxLayout()
            log_row.addWidget(QLabel("Generation log"))
            log_row.addStretch()
            log_row.addWidget(self.generate_python_code_button)
            log_row.addWidget(self.generate_log_button)
            layout.addLayout(log_row)
            layout.addWidget(self.generation_log_edit)
            return widget

        def _segment_workflow_tab(self):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            controls_column = QVBoxLayout()
            row = QHBoxLayout()
            row.addWidget(self.add_segment_button)
            row.addWidget(self.remove_segment_button)
            row.addStretch()
            controls_column.addLayout(row)
            controls_column.addWidget(QLabel("Segments"))
            controls_column.addWidget(self.segment_marker_list)
            controls_column.addWidget(self.strip_participant_prefix_checkbox)
            controls_column.addWidget(self.marker_mapping_label)
            marker_row = QHBoxLayout()
            parent_column = QVBoxLayout()
            parent_column.addWidget(QLabel("Parent segment"))
            parent_column.addWidget(self.workflow_parent_combo)
            parent_column.addStretch()
            left_column = QVBoxLayout()
            left_column.addWidget(QLabel("Available markers in main C3D"))
            left_column.addWidget(self.show_all_markers_checkbox)
            left_column.addWidget(self.show_virtual_markers_in_segments_checkbox)
            left_column.addWidget(self.marker_list)
            self.marker_list.setMaximumWidth(320)
            transfer_column = QVBoxLayout()
            transfer_column.addStretch()
            self.assign_marker_button.setText("->")
            self.unassign_marker_button.setText("<-")
            transfer_column.addWidget(self.assign_marker_button)
            transfer_column.addWidget(self.unassign_marker_button)
            transfer_column.addStretch()
            right_column = QVBoxLayout()
            right_column.addWidget(QLabel("Markers assigned to selected segment"))
            right_column.addWidget(self.assigned_marker_list)
            right_column.addWidget(self.assigned_marker_technical_checkbox)
            self.assigned_marker_list.setMaximumWidth(320)
            marker_row.addLayout(parent_column, 1)
            marker_row.addLayout(left_column, 2)
            marker_row.addLayout(transfer_column, 0)
            marker_row.addLayout(right_column, 2)
            controls_column.addLayout(marker_row)
            controls_column.addStretch()

            preview_column = QVBoxLayout()
            preview_row = QHBoxLayout()
            preview_row.addWidget(QLabel("Frame"))
            preview_row.addWidget(self.technical_frame_slider)
            preview_row.addWidget(self.technical_frame_label)
            preview_column.addLayout(preview_row)
            preview_column.addWidget(self.technical_segment_preview, 1)

            layout.addLayout(controls_column, 2)
            layout.addLayout(preview_column, 1)
            return widget

        def _anatomical_segment_workflow_tab(self):
            widget = QWidget()
            layout = QHBoxLayout(widget)

            controls_column = QVBoxLayout()
            controls_column.addWidget(QLabel("Anatomical segments"))
            controls_column.addWidget(self.anatomical_segment_list)
            instructions_label = QLabel(
                "Define two anatomical vectors from marker groups. The kept vector is preserved; the other one "
                "is orthogonalized and the third axis is computed by cross product."
            )
            instructions_label.setWordWrap(True)
            instructions_label.setMaximumWidth(720)
            controls_column.addWidget(instructions_label)

            controls_column.addWidget(QLabel("Segment system of coordinates"))
            axis_layout = QHBoxLayout()
            source_column = QVBoxLayout()
            source_column.addWidget(QLabel("Available C3D and virtual markers"))
            source_column.addWidget(self.axis_marker_source_list)
            origin_row = QHBoxLayout()
            origin_buttons = QVBoxLayout()
            origin_buttons.addWidget(self.add_axis_origin_marker_button)
            origin_buttons.addWidget(self.remove_axis_origin_marker_button)
            origin_buttons.addStretch()
            origin_row.addLayout(origin_buttons)
            origin_row.addWidget(self.axis_origin_marker_list)
            source_column.addWidget(QLabel("Origin markers"))
            source_column.addLayout(origin_row)
            axis_layout.addLayout(source_column, 0)
            axis_layout.addLayout(_axis_vectors_layout(self.axis_vector_controls), 1)
            controls_column.addLayout(axis_layout)
            controls_column.addWidget(QLabel("Saved anatomical vectors"))
            controls_column.addWidget(self.axis_list)
            controls_column.addStretch()

            preview_column = QVBoxLayout()
            preview_column.addWidget(QLabel("Dynamic frame preview"))
            preview_frame_row = QHBoxLayout()
            preview_frame_row.addWidget(QLabel("Frame"))
            preview_frame_row.addWidget(self.anatomical_frame_slider, 1)
            preview_frame_row.addWidget(self.anatomical_frame_label)
            preview_column.addLayout(preview_frame_row)
            preview_column.addWidget(self.segment_axis_preview, 1)
            preview_column.addWidget(self.save_segment_axis_button)

            layout.addLayout(controls_column, 2)
            layout.addLayout(preview_column, 1)
            return widget

        def _virtual_marker_workflow_tab(self):
            widget = QWidget()
            layout = QHBoxLayout(widget)

            controls_layout = QVBoxLayout()
            list_column = QVBoxLayout()
            list_column.addWidget(QLabel("Virtual markers and axes"))
            self.feature_list.setMaximumHeight(155)
            list_column.addWidget(self.feature_list)
            row = QHBoxLayout()
            row.addWidget(self.add_virtual_marker_button)
            row.addWidget(self.remove_virtual_marker_button)
            list_column.addLayout(row)
            controls_layout.addLayout(list_column)

            form_column = QVBoxLayout()
            form = QFormLayout()
            form.addRow("Segment", self.virtual_marker_segment_combo)
            form.addRow("Parent", self.virtual_marker_proximal_combo)
            form.addRow("Method", self.virtual_marker_method_combo)
            form.addRow("Predictive method", self.virtual_marker_predictive_method_combo)
            form.addRow("Functional C3D", self.virtual_marker_c3d_file_combo)
            form.addRow(self.virtual_marker_source_label, self.virtual_marker_source_edit)
            form.addRow(self.virtual_marker_equation_label, self.virtual_marker_equation_edit)
            form.addRow("Suggested name", self.virtual_marker_suggested_name_label)
            form.addRow("Name", self.virtual_marker_name_edit)
            form_column.addLayout(form)
            form_column.addWidget(QLabel("Technical markers used"))
            form_column.addWidget(self.virtual_marker_technical_markers_label)
            form_column.addWidget(self.use_suggested_virtual_marker_name_button)
            form_column.addWidget(self.save_virtual_marker_button)
            form_column.addWidget(QLabel("Selected marker information"))
            form_column.addWidget(self.virtual_marker_info_label)
            form_column.addStretch()
            controls_layout.addLayout(form_column)

            preview_column = QVBoxLayout()
            preview_column.addWidget(QLabel("3D placement preview"))
            preview_options = QHBoxLayout()
            preview_options.addWidget(self.virtual_marker_show_functional_c3d_checkbox)
            preview_options.addWidget(self.virtual_marker_whole_body_preview_checkbox)
            preview_column.addLayout(preview_options)
            frame_row = QHBoxLayout()
            frame_row.addWidget(self.virtual_marker_frame_label)
            frame_row.addWidget(self.virtual_marker_frame_slider, 1)
            preview_column.addLayout(frame_row)
            preview_column.addWidget(self.virtual_marker_preview, 1)
            layout.addLayout(controls_layout, 2)
            layout.addLayout(preview_column, 1)
            return widget

        def _segment_settings_workflow_tab(self):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            left_column = QVBoxLayout()
            left_column.addWidget(QLabel("Segments"))
            left_column.addWidget(self.segment_settings_list)
            layout.addLayout(left_column, 1)

            form_column = QVBoxLayout()
            form = QFormLayout()
            form.addRow("Translations", self.settings_translations_edit)
            form.addRow("Rotations", self.settings_rotations_edit)
            form.addRow("q min", self.settings_q_min_edit)
            form.addRow("q max", self.settings_q_max_edit)
            form.addRow("", self.settings_child_translation_checkbox)
            form.addRow("Initial rotation", self.settings_initial_rotation_method_combo)
            form.addRow("Matrix source", self.settings_initial_rotation_source_edit)
            anatomical_c3d_row = QHBoxLayout()
            anatomical_c3d_row.addWidget(self.settings_initial_rotation_c3d_combo)
            anatomical_c3d_row.addWidget(self.browse_initial_rotation_c3d_button)
            form.addRow("Anatomical C3D", anatomical_c3d_row)
            form_column.addLayout(form)
            form_column.addWidget(self.edit_segment_settings_button)
            form_column.addStretch()
            layout.addLayout(form_column, 2)
            return widget

        def _file_role_workflow_tab(self):
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.addWidget(self.file_role_list)
            row = QHBoxLayout()
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
            filepath = self.c3d_path.text().strip()
            if filepath == "":
                return
            try:
                self._load_c3d_file(filepath)
            except Exception as error:
                QMessageBox.critical(self, "Unable to reload C3D", str(error))

        def _load_c3d_file(self, filepath: str) -> None:
            self.c3d_data = C3dData(filepath)
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
            self.c3d_folder_path = folder
            self.c3d_folder_edit.setText(folder)
            self._auto_assign_c3d_files_from_folder()
            self._sync_virtual_marker_c3d_files()
            self._sync_initial_rotation_c3d_files()
            self._update_preset_details()
            self._update_generation_log()

        def _auto_assign_c3d_files_from_folder(self, load_main: bool = True) -> None:
            if not self.c3d_folder_path or self._is_auto_assigning_c3d_files:
                return
            self._is_auto_assigning_c3d_files = True
            try:
                assignments = []
                trial_sources = {}
                for assignment in self.workflow_draft.file_assignments:
                    matched_file = _matching_c3d_file_for_expected_name(self.c3d_folder_path, assignment.generic_name)
                    source_path = str(matched_file) if matched_file is not None else assignment.source_path
                    assignments.append(replace(assignment, source_path=source_path))
                    if source_path:
                        trial_sources[assignment.role] = source_path

                updated_virtual_markers = []
                for marker in self.workflow_draft.virtual_markers:
                    trial_name = _trial_name_from_virtual_feature_source(
                        marker.source
                    ) or _trial_name_from_virtual_feature_source(marker.equation)
                    source_path = trial_sources.get(trial_name, "")
                    equation = marker.equation
                    if marker.method in {"score", "sara"} and equation == "":
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
                for axis in self.workflow_draft.axes:
                    trial_name = _trial_name_from_virtual_feature_source(axis.source)
                    source_path = trial_sources.get(trial_name, "")
                    updated_axes.append(replace(axis, source=_source_with_c3d_assignment(axis.source, source_path)))

                self.workflow_draft = replace(
                    self.workflow_draft,
                    file_assignments=tuple(assignments),
                    virtual_markers=tuple(updated_virtual_markers),
                    axes=tuple(updated_axes),
                )
                main_source = trial_sources.get("main", "")
                if load_main and main_source and not self.c3d_path.text().strip():
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
            self._set_c3d_combo_to_filepath(self.virtual_marker_c3d_file_combo, filepath, "Choose a C3D folder first")
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
            self.settings_initial_rotation_source_edit.setText(self._selected_initial_rotation_c3d_file())

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
                self._update_preset_details()
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
            self._update_preset_details()

        def _add_workflow_virtual_marker(self) -> None:
            self.feature_list.clearSelection()
            self.virtual_marker_name_edit.clear()
            self.virtual_marker_source_edit.clear()
            self.virtual_marker_equation_edit.clear()
            if self.virtual_marker_segment_combo.count() != 0:
                self.virtual_marker_segment_combo.setCurrentIndex(0)
            self.virtual_marker_method_combo.setCurrentText("pointing")
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
            return self._selected_virtual_marker_c3d_file()

        def _virtual_marker_equation_from_form(self, method: str) -> str:
            if method == "axis_projection":
                return self.virtual_marker_equation_edit.text().strip()
            methods_with_segment_context = {"score", "sara"} | set(PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS)
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
                "marker_mean": "Average",
                "axis_projection": "Projection",
            }.get(
                method,
                method.capitalize() if method else "Virtual",
            )
            if method in {"hara2016_hip", "harrington2007_hip"}:
                method_label = PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS[method].replace(" ", "")
            elif method == "sobral2025_shoulder":
                method_label = PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS[method].replace(" ", "")
            local_segment = proximal or segment_name
            if method in {"score", "sara"} and local_segment:
                return f"CoR_{method_label}_{joint_name}_in_{local_segment}"
            if segment_name:
                return f"{method_label}_{joint_name}_in_{segment_name}"
            return f"{method_label}_{joint_name}"

        def _update_suggested_virtual_marker_name(self, *_args) -> None:
            self.virtual_marker_suggested_name_label.setText(self._suggested_virtual_marker_name())

        def _use_suggested_virtual_marker_name(self) -> None:
            self.virtual_marker_name_edit.setText(self._suggested_virtual_marker_name())

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
            self.settings_translations_edit.setText(setting.translations)
            self.settings_rotations_edit.setText(setting.rotations)
            self.settings_q_min_edit.setText(_format_float_list(list(setting.q_min)))
            self.settings_q_max_edit.setText(_format_float_list(list(setting.q_max)))
            self.settings_child_translation_checkbox.setChecked(setting.child_translation)
            self.settings_initial_rotation_method_combo.setCurrentText(setting.initial_rotation_method)
            self.settings_initial_rotation_source_edit.setText(setting.initial_rotation_source)
            self._sync_initial_rotation_c3d_files(setting.initial_rotation_source)
            self._sync_initial_rotation_source_fields()

        def _apply_workflow_segment_settings_from_form(self) -> None:
            setting = self._selected_segment_setting()
            if setting is None:
                return
            initial_rotation_method = self.settings_initial_rotation_method_combo.currentText()
            initial_rotation_source = self.settings_initial_rotation_source_edit.text()
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
                )
                self._update_preset_details()
            except Exception as error:
                QMessageBox.critical(self, "Unable to apply segment settings", str(error))

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
            self._update_segment_axis_preview()

        def _add_selected_axis_origin_markers(self) -> None:
            self._add_selected_axis_markers_to_list(self.axis_origin_marker_list)
            self._update_segment_axis_preview()

        def _add_selected_axis_markers_to_list(self, target_list) -> None:
            for item in self.axis_marker_source_list.selectedItems():
                marker_name = item.text().split("|", maxsplit=1)[0].strip()
                if marker_name:
                    target_list.addItem(marker_name)

        def _remove_selected_axis_markers(self, vector_index: int, endpoint: str) -> None:
            target_list = self._axis_endpoint_list(vector_index, endpoint)
            self._remove_selected_axis_markers_from_list(target_list)
            self._update_segment_axis_preview()

        def _remove_selected_axis_origin_markers(self) -> None:
            self._remove_selected_axis_markers_from_list(self.axis_origin_marker_list)
            self._update_segment_axis_preview()

        def _remove_selected_axis_markers_from_list(self, target_list) -> None:
            for item in target_list.selectedItems():
                target_list.takeItem(target_list.row(item))

        def _axis_endpoint_list(self, vector_index: int, endpoint: str):
            key = "start_list" if endpoint == "start" else "end_list"
            return self.axis_vector_controls[vector_index][key]

        def _save_segment_axis_from_lists(self) -> None:
            segment_name = self._selected_anatomical_segment_name()
            if segment_name is None:
                return
            vector_specs = self._axis_vector_specs()
            origin_markers = _list_widget_texts(self.axis_origin_marker_list)
            if len(vector_specs) != 2:
                QMessageBox.critical(self, "Unable to save segment axis", "Two complete vectors are required.")
                return
            if len(origin_markers) == 0:
                QMessageBox.critical(self, "Unable to save segment axis", "At least one origin marker is required.")
                return
            if sum(keep_vector for _, _, _, keep_vector in vector_specs) != 1:
                QMessageBox.critical(self, "Unable to save segment axis", "Choose exactly one vector to keep.")
                return
            if len({axis_name for axis_name, _, _, _ in vector_specs}) != 2:
                QMessageBox.critical(
                    self,
                    "Unable to save segment axis",
                    "The two vectors must use two different axes. The third axis is computed by cross product.",
                )
                return
            try:
                updated_draft = self.workflow_draft
                for index, (axis_name, start_markers, end_markers, keep_vector) in enumerate(vector_specs, start=1):
                    updated_draft = add_axis_to_draft(
                        updated_draft,
                        name=f"{segment_name}_vector_{index}",
                        segment_name=segment_name,
                        axis=axis_name,
                        start_markers=start_markers,
                        end_markers=end_markers,
                        origin_markers=origin_markers,
                        method="markers",
                        keep_vector=keep_vector,
                    )
                self.workflow_draft = updated_draft
                self._update_preset_details()
            except Exception as error:
                QMessageBox.critical(self, "Unable to save segment axis", str(error))

        def _axis_vector_specs(self) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...], bool], ...]:
            specs = []
            for controls in self.axis_vector_controls:
                axis_name = controls["axis_combo"].currentText()
                start_markers = _list_widget_texts(controls["start_list"])
                end_markers = _list_widget_texts(controls["end_list"])
                if len(start_markers) != 0 and len(end_markers) != 0:
                    specs.append((axis_name, start_markers, end_markers, controls["keep_checkbox"].isChecked()))
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
            self._update_technical_segment_preview()

        def _configure_anatomical_frame_slider(self) -> None:
            frame_count = 0 if self.c3d_data is None else self.c3d_data.nb_frames
            self.anatomical_frame_slider.blockSignals(True)
            self.anatomical_frame_slider.setEnabled(frame_count > 1)
            self.anatomical_frame_slider.setMinimum(0)
            self.anatomical_frame_slider.setMaximum(max(frame_count - 1, 0))
            self.anatomical_frame_slider.setValue(0)
            self.anatomical_frame_slider.blockSignals(False)
            self._update_segment_axis_preview()

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
                frame_index,
            )

        def _update_anatomical_segment_details(self) -> None:
            self._update_axis_marker_source_list()
            self._load_selected_segment_axes_into_controls()
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
                self._update_preset_details()
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
            marker_names = tuple(self.workflow_marker_pool)
            virtual_marker_names = tuple(
                marker.name for marker in self.workflow_draft.virtual_markers if marker.segment_name == segment_name
            )
            for marker_name in marker_names:
                self.axis_marker_source_list.addItem(marker_name)
            for marker_name in virtual_marker_names:
                self.axis_marker_source_list.addItem(f"{marker_name} | virtual")

        def _load_selected_segment_axes_into_controls(self) -> None:
            segment_name = self._selected_anatomical_segment_name()
            axes = tuple(axis for axis in self.workflow_draft.axes if axis.segment_name == segment_name)[:2]
            self.axis_origin_marker_list.clear()
            origin_markers = axes[0].origin_markers if len(axes) != 0 else ()
            for marker_name in origin_markers:
                self.axis_origin_marker_list.addItem(marker_name)
            for index, controls in enumerate(self.axis_vector_controls):
                controls["start_list"].clear()
                controls["end_list"].clear()
                controls["axis_combo"].setCurrentText("x" if index == 0 else "y")
                controls["keep_checkbox"].setChecked(index == 0)
                if index >= len(axes):
                    continue
                axis = axes[index]
                fallback_axis = "x" if index == 0 else "y"
                controls["axis_combo"].setCurrentText(axis.axis if axis.axis in {"x", "y", "z"} else fallback_axis)
                controls["keep_checkbox"].setChecked(axis.keep_vector)
                for marker_name in axis.start_markers:
                    controls["start_list"].addItem(marker_name)
                for marker_name in axis.end_markers:
                    controls["end_list"].addItem(marker_name)
            self._ensure_single_kept_axis_vector(0)

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
            axes = tuple(axis for axis in self.workflow_draft.axes if axis.segment_name == segment_name)
            self.segment_axis_preview.set_context(
                self.c3d_data,
                segment_marker_names,
                label_marker_names,
                self.workflow_draft.segment_marker_groups,
                segment_name,
                axes,
                self._axis_vector_specs(),
                _list_widget_texts(self.axis_origin_marker_list),
                frame_index,
            )

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
            marker = self._selected_virtual_marker()
            if marker is None:
                axis = self._selected_virtual_axis()
                if axis is not None:
                    self.virtual_marker_name_edit.setText(axis.name)
                    self.virtual_marker_segment_combo.setCurrentText(axis.segment_name)
                    self.virtual_marker_method_combo.setCurrentText(axis.method)
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
                self.virtual_marker_method_combo.setCurrentText("predictive")
                self.virtual_marker_predictive_method_combo.setCurrentText(
                    PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS[marker.method]
                )
            else:
                self.virtual_marker_method_combo.setCurrentText(marker.method)
            self.virtual_marker_source_edit.setText(marker.source)
            self.virtual_marker_equation_edit.setText(marker.equation if marker.method == "axis_projection" else "")
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
                (self.virtual_marker_proximal_combo, technical_segment_names, current_proximal),
                (self.virtual_marker_distal_combo, technical_segment_names, current_distal),
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
            if self.virtual_marker_show_functional_c3d_checkbox.isChecked():
                return self._selected_virtual_marker_c3d_data()
            return self.c3d_data

        def _selected_virtual_marker_preview_label(self) -> str:
            if self.virtual_marker_show_functional_c3d_checkbox.isChecked():
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
            if frame_count == 0:
                self.virtual_marker_frame_label.setText("Frame 0/0")
            else:
                self.virtual_marker_frame_label.setText(f"Frame {current_frame + 1}/{frame_count}")

        def _mirror_initial_rotation_c3d_source(self, *_args) -> None:
            if self.settings_initial_rotation_method_combo.currentText() == "anatomical_c3d":
                self.settings_initial_rotation_source_edit.setText(self._selected_initial_rotation_c3d_file())

        def _sync_initial_rotation_source_fields(self, *_args) -> None:
            method = self.settings_initial_rotation_method_combo.currentText()
            is_anatomical_c3d = method == "anatomical_c3d"
            is_matrix = method == "matrix"
            self.settings_initial_rotation_source_edit.setEnabled(is_matrix or is_anatomical_c3d)
            self.settings_initial_rotation_c3d_combo.setEnabled(
                is_anatomical_c3d and self._selected_initial_rotation_c3d_file() != ""
            )
            self.browse_initial_rotation_c3d_button.setEnabled(is_anatomical_c3d)
            if is_anatomical_c3d:
                self.settings_initial_rotation_source_edit.setText(self._selected_initial_rotation_c3d_file())

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
            if method != "predictive":
                return method
            return _predictive_virtual_marker_method_from_label(
                self.virtual_marker_predictive_method_combo.currentText().strip()
            )

        def _sync_virtual_marker_method_fields(self, _method: str | None = None) -> None:
            method = self._selected_virtual_marker_method()
            is_predictive = self.virtual_marker_method_combo.currentText().strip() == "predictive"
            self.virtual_marker_predictive_method_combo.setEnabled(is_predictive)
            has_c3d_file = self.virtual_marker_c3d_file_combo.currentText().strip() != "Choose a C3D folder first"
            self.virtual_marker_c3d_file_combo.setEnabled(
                method not in {"marker_mean", "axis_projection"} and has_c3d_file
            )
            self.virtual_marker_proximal_combo.setEnabled(method in {"score", "sara"} or is_predictive)
            self.virtual_marker_distal_combo.setEnabled(method in {"score", "sara"} or is_predictive)
            is_axis_projection = method == "axis_projection"
            self.virtual_marker_source_label.setVisible(is_axis_projection)
            self.virtual_marker_source_edit.setVisible(is_axis_projection)
            self.virtual_marker_equation_label.setVisible(is_axis_projection)
            self.virtual_marker_equation_edit.setVisible(is_axis_projection)
            if is_axis_projection:
                self.virtual_marker_source_label.setText("Source markers")
                self.virtual_marker_equation_label.setText("Projection axis")
                if self.virtual_marker_source_edit.text().strip() == "":
                    self.virtual_marker_source_edit.setPlaceholderText("point=LKNE,LKNEM")
                if self.virtual_marker_equation_edit.text().strip() == "":
                    self.virtual_marker_equation_edit.setPlaceholderText(
                        "axis=Axis_LKnee_SARA or axis_start=LKNE,LKNEM; axis_end=LANK,LANKM"
                    )
            hints = {
                "pointing": (
                    "Pointing is not implemented yet: the workflow still needs a dedicated pointing object "
                    "to store the pointed target."
                ),
                "score": "Choose the functional C3D plus proximal and distal technical segments. SCoRE estimates a joint center.",
                "sara": "Choose the functional C3D plus proximal and distal technical segments. SARA estimates a rotation axis.",
                "marker_mean": "Average the technical markers shown for the proximal and distal segments.",
                "axis_projection": (
                    "Project one marker or a marker mean onto an axis. Use source 'point=A,B' and either "
                    "'axis=Axis_LKnee_SARA' or 'axis_start=A,B; axis_end=C,D'."
                ),
                "hara2016_hip": "Predict a hip CoR from the selected C3D and segment pair.",
                "harrington2007_hip": "Predict a hip CoR from the selected C3D and segment pair.",
                "sobral2025_shoulder": "Predict a shoulder CoR from the selected C3D and segment pair.",
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
                    f"Name: {marker.name}\nSegment: {marker.segment_name}\nMethod: {marker.method}\n"
                    f"Projected markers: {','.join(point_markers) or '-'}\nProjection axis: {axis_text}"
                )
                return
            self.virtual_marker_info_label.setText(
                f"Name: {marker.name}\nSegment: {marker.segment_name}\nMethod: {marker.method}\n"
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
                "Preview axis coordinates are computed from the selected functional C3D."
            )

        def _update_virtual_marker_preview(self, *_args) -> None:
            preview_c3d_data = self._selected_virtual_marker_preview_c3d_data()
            solution_c3d_data = self._selected_virtual_marker_c3d_data()
            self._sync_virtual_marker_frame_slider(preview_c3d_data)
            self.virtual_marker_preview.set_context(
                preview_c3d_data,
                solution_c3d_data,
                self._selected_virtual_marker_preview_label(),
                self.workflow_draft.segment_marker_groups,
                self.workflow_draft.virtual_markers,
                self.virtual_marker_name_edit.text().strip(),
                self._selected_virtual_marker_method(),
                self.virtual_marker_proximal_combo.currentText().strip(),
                self.virtual_marker_segment_combo.currentText().strip(),
                self.virtual_marker_frame_slider.value(),
                self.virtual_marker_whole_body_preview_checkbox.isChecked(),
            )

        def _update_generation_log(self) -> None:
            lines = _c3d_generation_log(
                self.workflow_draft,
                self.c3d_data,
                self.c3d_folder_path,
                self.workflow_marker_pool,
            )
            self.generation_log_edit.setPlainText("\n".join(lines))

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
            self.example_list.clear()

            if preset == C3dModelPreset.FROM_SCRATCH:
                self.status_label.setText(
                    "Status: template-free draft; add segments, markers, axes, DoFs, and virtual markers manually."
                )
            elif preset == C3dModelPreset.FULL_BODY:
                self.status_label.setText(
                    "Status: full-body template mapping exists; generation still needs virtual markers."
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

            for step_status in c3d_workflow_progress(self.workflow_draft, self.c3d_data):
                self.step_list.addItem(
                    f"{step_status.number}. [{step_status.status}] {step_status.name} - {step_status.detail}"
                )

            for group in self.workflow_draft.segment_marker_groups:
                markers = ", ".join(group.marker_names) if len(group.marker_names) != 0 else "no marker assigned yet"
                parent = group.parent_name if group.parent_name else "-"
                self.segment_marker_list.addItem(
                    f"{group.segment_name}: {markers} | type={group.segment_type} | parent={parent}"
                )
                self.anatomical_segment_list.addItem(
                    f"{group.segment_name}: {markers} | type={group.segment_type} | parent={parent}"
                )
            self._restore_workflow_segment_selection(previously_selected_segment)
            self._restore_anatomical_segment_selection(previously_selected_anatomical_segment)
            self._update_available_marker_list()
            self._sync_virtual_marker_choices()
            self._update_technical_segment_preview()

            if len(self.workflow_draft.virtual_markers) == 0:
                if not any(_is_virtual_feature_axis(axis) for axis in self.workflow_draft.axes):
                    self.feature_list.addItem("No additional virtual feature required by this preset.")
            else:
                for feature in self.workflow_draft.virtual_markers:
                    self.feature_list.addItem(
                        f"{feature.name} | {feature.segment_name} | {feature.method} | {feature.source}"
                    )
            for axis in self.workflow_draft.axes:
                if not _is_virtual_feature_axis(axis):
                    continue
                self.feature_list.addItem(
                    f"[axis] {axis.name} | {axis.segment_name} | {axis.method} | {axis.source or 'functional trial'}"
                )
            if self.feature_list.count() != 0 and self.feature_list.currentItem() is None:
                self.feature_list.setCurrentItem(self.feature_list.item(0))

            if len(self.workflow_draft.axes) == 0:
                self.axis_list.addItem("No axis definition yet.")
            else:
                for axis in self.workflow_draft.axes:
                    self.axis_list.addItem(
                        f"{axis.name} | {axis.segment_name} | {axis.axis} | {axis.method} | "
                        f"origin={','.join(axis.origin_markers) or '-'} | "
                        f"{','.join(axis.start_markers)} -> {','.join(axis.end_markers)}"
                    )

            for setting in self.workflow_draft.segment_settings:
                self.segment_settings_list.addItem(
                    f"{setting.segment_name} | translations={setting.translations or '-'} | "
                    f"rotations={setting.rotations or '-'} | child_translation={setting.child_translation} | "
                    f"initial_rotation={setting.initial_rotation_method}"
                )
            if self.segment_settings_list.count() != 0 and self.segment_settings_list.currentItem() is None:
                self.segment_settings_list.setCurrentItem(self.segment_settings_list.item(0))
                self._load_selected_segment_settings_into_form()

            for file_role in self.workflow_draft.file_assignments:
                role_definition = next(role for role in workflow.file_roles if role.role == file_role.role)
                required = "required" if role_definition.required else "optional"
                source = file_role.source_path if file_role.source_path else "not assigned"
                self.file_role_list.addItem(f"{file_role.role} | {file_role.generic_name} | {required} | {source}")

            issues = validate_c3d_workflow_draft(self.workflow_draft, self.c3d_data)
            if len(issues) == 0:
                self.issue_list.addItem("No draft issue detected.")
            else:
                for issue in issues:
                    self.issue_list.addItem(f"{issue.severity.upper()} | {issue.category} | {issue.message}")

            for example in c3d_virtual_marker_method_examples():
                if example.method not in _visible_virtual_marker_methods():
                    continue
                self.example_list.addItem(
                    f"{example.method} | source: {example.source_example} | settings: "
                    f"{example.equation_example or '-'} | {example.description}"
                )

            self.summary_label.setText(c3d_workflow_summary(preset, self.c3d_data))
            self._update_virtual_marker_preview()
            self._update_generation_log()

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
            self.yaw = -0.6
            self.pitch = 0.35
            self._last_mouse_position = None

        def set_model(self, model) -> None:
            self.scene = None if model is None else build_preview_scene(model)
            self.update()

        def set_selected_segment(self, segment_name: str | None) -> None:
            self.selected_segment_name = segment_name
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(qpaint_antialiasing)
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
            transform = _fit_projection(all_points, self.width(), self.height(), QPointF)
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

            axis_colors = {"x": "#dc2626", "y": "#16a34a", "z": "#2563eb"}
            for axis, start, end in projected_axes:
                painter.setPen(QPen(QColor(axis_colors[axis.axis]), 4 if axis.is_rotation_axis else 1))
                painter.drawLine(transform(start), transform(end))

            painter.setPen(QPen(QColor("#2563eb"), 1))
            painter.setBrush(QColor("#2563eb"))
            for marker_point in projected_markers.values():
                center = transform(marker_point)
                painter.drawEllipse(center, 4, 4)

            for name, point in projected_joints.items():
                center = self._projected_joint_positions[name]
                is_selected = name == self.selected_segment_name
                painter.setPen(QPen(QColor("#111827"), 1))
                painter.setBrush(QColor("#f59e0b" if is_selected else "#111827"))
                painter.drawEllipse(center, 5 if is_selected else 3, 5 if is_selected else 3)

            _draw_preview_orientation_axes(painter, self.width(), self.height(), self.yaw, self.pitch)
            self._draw_legend(painter)

        def mousePressEvent(self, event) -> None:
            self._last_mouse_position = get_event_position(event)

        def mouseMoveEvent(self, event) -> None:
            if self._last_mouse_position is None:
                self._last_mouse_position = get_event_position(event)
                return
            position = get_event_position(event)
            self.yaw += (position.x() - self._last_mouse_position.x()) * 0.01
            self.pitch += (position.y() - self._last_mouse_position.y()) * 0.01
            self.pitch = max(-1.4, min(1.4, self.pitch))
            self._last_mouse_position = position
            self.update()

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
            _draw_legend_point(painter, QPointF(x + 6, y - 4), QColor("#111827"), "Joint centers", x + 20, y)
            y += line_gap
            _draw_legend_line(painter, QColor("#6b7280"), 2, x, y - 4, "Bones", x + 36, y)
            y += line_gap
            _draw_legend_line(painter, QColor("#dc2626"), 2, x, y - 4, "Muscles", x + 36, y)
            y += line_gap
            _draw_legend_line(painter, QColor("#dc2626"), 1, x, y - 4, "x axis", x + 36, y)
            y += line_gap
            _draw_legend_line(painter, QColor("#16a34a"), 1, x, y - 4, "y axis", x + 36, y)
            y += line_gap
            _draw_legend_line(painter, QColor("#2563eb"), 4, x, y - 4, "Rotational axis", x + 36, y)

        def mousePressEvent(self, event) -> None:
            clicked = get_event_position(event)
            marker_name = _nearest_projected_segment(self._projected_marker_positions, clicked)
            if marker_name is not None and self.on_marker_selected is not None:
                self.on_marker_selected(marker_name)
                return
            segment_name = _nearest_projected_segment(self._projected_joint_positions, clicked)
            if segment_name is not None and self.on_segment_selected is not None:
                self.on_segment_selected(segment_name)

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
            self.preview.on_segment_selected = self._select_segment_from_preview
            self.preview.on_marker_selected = self._select_marker_from_preview
            self.validation_messages = QListWidget()
            self.validate_button = QPushButton("Validate model")
            self.validate_button.clicked.connect(self._validate_model)

            self.tree = QTreeWidget()
            self.tree.setHeaderLabel("Segments")
            self.tree.itemSelectionChanged.connect(self._on_segment_selection_changed)

            self.parent_name = QLineEdit()
            self.translations = QLineEdit()
            self.rotations = QLineEdit()
            self.q_min = QLineEdit()
            self.q_max = QLineEdit()
            self.mass = QLineEdit()
            self.center_of_mass = QLineEdit()
            self.inertia_diagonal = QLineEdit()

            self.apply_button = QPushButton("Apply segment changes")
            self.apply_button.clicked.connect(self._apply_segment_changes)

            form = QFormLayout()
            form.addRow("Parent", self.parent_name)
            form.addRow("Translations", self.translations)
            form.addRow("Rotations", self.rotations)
            form.addRow("q min", self.q_min)
            form.addRow("q max", self.q_max)
            form.addRow("Mass", self.mass)
            form.addRow("Center of mass", self.center_of_mass)
            form.addRow("Inertia diagonal", self.inertia_diagonal)

            right_panel = QWidget()
            segment_tab = QWidget()
            segment_layout = QVBoxLayout(segment_tab)
            segment_layout.addWidget(QLabel("Segment properties"))
            segment_layout.addLayout(form)
            segment_layout.addWidget(self.apply_button)
            segment_layout.addStretch()

            self.marker_list = QListWidget()
            self.marker_list.itemSelectionChanged.connect(self._on_marker_selection_changed)
            self.marker_name = QLineEdit()
            self.marker_name.setPlaceholderText("New or selected marker name")
            self.marker_position = QLineEdit()
            self.marker_position.setPlaceholderText("x y z")
            self.marker_technical = QCheckBox("Technical")
            self.marker_anatomical = QCheckBox("Anatomical")
            self.apply_marker_button = QPushButton("Apply marker changes")
            self.apply_marker_button.clicked.connect(self._apply_marker_changes)
            self.add_marker_button = QPushButton("Add marker")
            self.add_marker_button.clicked.connect(self._add_marker)
            self.remove_marker_button = QPushButton("Remove marker")
            self.remove_marker_button.clicked.connect(self._remove_marker)
            self.marker_target_segment = QLineEdit()
            self.marker_target_segment.setPlaceholderText("Target segment")
            self.attach_marker_button = QPushButton("Attach marker to segment")
            self.attach_marker_button.clicked.connect(self._attach_marker_to_segment)

            marker_form = QFormLayout()
            marker_form.addRow("Name", self.marker_name)
            marker_form.addRow("Position", self.marker_position)
            marker_form.addRow("", self.marker_technical)
            marker_form.addRow("", self.marker_anatomical)
            marker_form.addRow("Target segment", self.marker_target_segment)

            marker_tab = QWidget()
            marker_layout = QVBoxLayout(marker_tab)
            marker_layout.addWidget(QLabel("Markers on selected segment"))
            marker_layout.addWidget(self.marker_list)
            marker_layout.addLayout(marker_form)
            marker_layout.addWidget(self.apply_marker_button)
            marker_layout.addWidget(self.add_marker_button)
            marker_layout.addWidget(self.remove_marker_button)
            marker_layout.addWidget(self.attach_marker_button)

            self.muscle_tree = QTreeWidget()
            self.muscle_tree.setHeaderLabel("Muscles")
            self.muscle_tree.itemSelectionChanged.connect(self._on_muscle_selection_changed)
            self.optimal_length = QLineEdit()
            self.maximal_force = QLineEdit()
            self.tendon_slack_length = QLineEdit()
            self.pennation_angle = QLineEdit()
            self.maximal_velocity = QLineEdit()
            self.maximal_excitation = QLineEdit()
            self.apply_muscle_button = QPushButton("Apply muscle changes")
            self.apply_muscle_button.clicked.connect(self._apply_muscle_changes)
            self.group_name = QLineEdit()
            self.group_origin_parent = QLineEdit()
            self.group_insertion_parent = QLineEdit()
            self.add_group_button = QPushButton("Add muscle group")
            self.add_group_button.clicked.connect(self._add_muscle_group)
            self.remove_group_button = QPushButton("Remove selected group")
            self.remove_group_button.clicked.connect(self._remove_muscle_group)
            self.new_muscle_name = QLineEdit()
            self.add_muscle_button = QPushButton("Add muscle")
            self.add_muscle_button.clicked.connect(self._add_muscle)
            self.remove_muscle_button = QPushButton("Remove selected muscle")
            self.remove_muscle_button.clicked.connect(self._remove_muscle)
            self.origin_name = QLineEdit()
            self.origin_parent = QLineEdit()
            self.origin_position = QLineEdit()
            self.insertion_name = QLineEdit()
            self.insertion_parent = QLineEdit()
            self.insertion_position = QLineEdit()
            self.apply_path_endpoints_button = QPushButton("Apply origin/insertion changes")
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
            self.via_point_list.itemSelectionChanged.connect(self._on_via_point_selection_changed)
            self.via_point_name = QLineEdit()
            self.via_point_name.setPlaceholderText("New or selected via-point name")
            self.via_point_parent = QLineEdit()
            self.via_point_parent.setPlaceholderText("Parent segment")
            self.via_point_position = QLineEdit()
            self.via_point_position.setPlaceholderText("x y z")
            self.apply_via_point_button = QPushButton("Apply via-point changes")
            self.apply_via_point_button.clicked.connect(self._apply_via_point_changes)
            self.add_via_point_button = QPushButton("Add via point")
            self.add_via_point_button.clicked.connect(self._add_via_point)
            self.remove_via_point_button = QPushButton("Remove via point")
            self.remove_via_point_button.clicked.connect(self._remove_via_point)

            via_point_form = QFormLayout()
            via_point_form.addRow("Via-point name", self.via_point_name)
            via_point_form.addRow("Parent", self.via_point_parent)
            via_point_form.addRow("Position", self.via_point_position)

            muscle_tab = QWidget()
            muscle_layout = QVBoxLayout(muscle_tab)
            muscle_layout.addWidget(self.muscle_tree)
            muscle_layout.addWidget(self.add_group_button)
            muscle_layout.addWidget(self.remove_group_button)
            muscle_layout.addWidget(self.new_muscle_name)
            muscle_layout.addWidget(self.add_muscle_button)
            muscle_layout.addWidget(self.remove_muscle_button)
            muscle_layout.addLayout(muscle_form)
            muscle_layout.addWidget(self.apply_muscle_button)
            muscle_layout.addWidget(self.apply_path_endpoints_button)
            muscle_layout.addWidget(QLabel("Via points"))
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
            validation_layout.addWidget(self.validate_button)
            graph_model_button = QPushButton("Graph model")
            graph_model_button.clicked.connect(self._write_model_graphviz)
            plot_muscles_button = QPushButton("Plot muscles")
            plot_muscles_button.clicked.connect(self._plot_muscles)
            validation_layout.addWidget(graph_model_button)
            validation_layout.addWidget(plot_muscles_button)
            validation_layout.addWidget(self.validation_messages)
            tabs.addTab(validation_tab, "Validation")

            right_layout = QVBoxLayout(right_panel)
            right_layout.addWidget(tabs)

            splitter = QSplitter(qt_horizontal)
            splitter.addWidget(self.tree)
            splitter.addWidget(right_panel)
            splitter.setSizes([350, 750])

            open_button = QPushButton("Open model")
            open_button.clicked.connect(self._open_model)
            new_c3d_model_button = QPushButton("New from C3D")
            new_c3d_model_button.clicked.connect(self._new_model_from_c3d)
            save_button = QPushButton("Export model")
            save_button.clicked.connect(self._save_model)

            toolbar = QHBoxLayout()
            toolbar.addWidget(open_button)
            toolbar.addWidget(new_c3d_model_button)
            toolbar.addWidget(save_button)
            toolbar.addStretch()

            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
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

        def _new_model_from_c3d(self) -> None:
            dialog = C3dModelCreationDialog(self)
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
            calibration_folder = QFileDialog.getExistingDirectory(
                self,
                "Select calibration folder",
                "",
            )
            if not calibration_folder:
                return
            try:
                folder_path = Path(calibration_folder)
                result = create_model_from_c3d_folder(
                    calibration_folder=folder_path,
                    preset=dialog.selected_preset(),
                )
                self.model = result.model
                self.current_filepath = folder_path / result.output_filename
                self._refresh_model_views()
                QMessageBox.information(
                    self,
                    "Model generated from C3D",
                    _format_c3d_creation_summary(result, folder_path),
                )
            except Exception as error:
                QMessageBox.critical(self, "Unable to generate model", str(error))

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
            self.center_of_mass.setText(_format_float_list(data.center_of_mass))
            self.inertia_diagonal.setText(_format_float_list(data.inertia_diagonal))
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
                    center_of_mass=_parse_vector(self.center_of_mass.text(), expected_length=3),
                    inertia_diagonal=_parse_vector(self.inertia_diagonal.text(), expected_length=3),
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
    window = ModelEditorWindow()
    window.show()
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
    if preset == C3dModelPreset.UPPER_LIMB:
        return "Upper-limb"
    if preset == C3dModelPreset.FROM_SCRATCH:
        return "From scratch"
    return preset.value


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


def _list_widget_texts(list_widget) -> tuple[str, ...]:
    """
    Return all marker names shown in a QListWidget, preserving duplicates.
    """
    return tuple(
        list_widget.item(index).text().split("|", maxsplit=1)[0].strip() for index in range(list_widget.count())
    )


def _marker_pool_from_draft(workflow_draft) -> tuple[str, ...]:
    """
    Return the known marker names for a C3D draft, even before a C3D file is loaded.
    """
    marker_names = []
    for group in workflow_draft.segment_marker_groups:
        marker_names.extend(group.marker_names)
    marker_names.extend(marker.name for marker in workflow_draft.virtual_markers)
    return tuple(dict.fromkeys(marker_names))


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


def _strip_participant_prefix_from_marker_names(marker_names: tuple[str, ...]) -> tuple[str, ...]:
    """
    Strip the text before ':' from marker names, preserving names without a separator.
    """
    return tuple(
        marker_name.split(":", maxsplit=1)[1] if ":" in marker_name else marker_name for marker_name in marker_names
    )


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
    return {"pointing", "score", "sara", "marker_mean", "axis_projection"} | set(
        PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS
    )


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


def _is_virtual_feature_axis(axis) -> bool:
    """
    Return whether an axis should appear in the Virtual markers and axes list.
    """
    return axis.method == "sara" and axis.name.startswith("Axis_")


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
    suffix = Path(filepath).suffix.lower()
    writers = {
        ".biomod": model.to_biomod,
        ".osim": model.to_osim,
        ".urdf": model.to_urdf,
        ".bvh": model.to_bvh,
    }
    if suffix not in writers:
        raise ValueError("Supported export extensions are .bioMod, .osim, .urdf, and .bvh.")
    writers[suffix](filepath=filepath)


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
        lines.append(f"- {assignment.role}: {assignment.source_path or assignment.generic_name}")
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
        if marker.method in {"score", "sara"} | set(PREDICTIVE_VIRTUAL_MARKER_METHOD_LABELS):
            local_note = (
                f" | global marker added to marker pool; local offsets reserved for proximal={proximal or '-'} "
                f"and distal={distal or '-'}"
            )
        lines.append(
            f"- {marker.name}: method={marker.method}, segment={marker.segment_name}, source={source}, "
            f"settings={equation}{local_note}"
        )
    lines.extend(["", "Anatomical axes:"])
    for axis in workflow_draft.axes:
        source = f", source={axis.source}" if axis.source else ""
        lines.append(
            f"- {axis.segment_name}/{axis.name}: {axis.axis}, "
            f"origin={','.join(axis.origin_markers) or '-'}, "
            f"{','.join(axis.start_markers)} -> {','.join(axis.end_markers)}, keep={axis.keep_vector}{source}"
        )
    lines.extend(["", "Segment settings:"])
    for setting in workflow_draft.segment_settings:
        lines.append(
            f"- {setting.segment_name}: translations={setting.translations or '-'}, rotations={setting.rotations or '-'}, "
            f"child_translation={setting.child_translation}, initial_rotation={setting.initial_rotation_method}"
        )
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


def _axis_projection_point_markers_from_payload(text: str) -> tuple[str, ...]:
    """
    Extract markers defining the point to project for an axis-projection virtual marker.
    """
    payload = _key_value_payload(text)
    point_text = payload.get("point", text)
    return _split_marker_names(point_text)


def _axis_projection_axis_from_payload(text: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
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
    colors = ("#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#7c3aed", "#0891b2", "#db2777", "#4b5563")
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
    if np.isnan(point).any():
        return None
    return tuple(float(value) for value in point)


def _marker_frame_position(c3d_data, marker_name: str, frame_index: int) -> tuple[float, float, float] | None:
    """
    Return the 3D position of one C3D marker at a given frame.
    """
    if c3d_data is None or marker_name not in c3d_data.marker_names:
        return None
    frame_index = max(0, min(frame_index, c3d_data.nb_frames - 1))
    point = c3d_data.get_position((marker_name,))[:3, 0, frame_index]
    if np.isnan(point).any():
        return None
    return tuple(float(value) for value in point)


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


def _rotate_preview_point(point: tuple[float, float, float], yaw: float, pitch: float) -> tuple[float, float]:
    """
    Project a 3D point with a user-controlled yaw/pitch rotation.
    """
    x, y, z = point
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    yaw_x = cos_yaw * x - sin_yaw * y
    yaw_y = sin_yaw * x + cos_yaw * y
    pitch_y = cos_pitch * yaw_y - sin_pitch * z
    pitch_z = sin_pitch * yaw_y + cos_pitch * z
    return yaw_x + 0.25 * pitch_y, -pitch_z + 0.15 * pitch_y


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


def _project_point(point) -> tuple[float, float]:
    """
    Project one 3D point into a simple isometric 2D view.
    """
    x, y, z = point[:3]
    return (x - 0.6 * y, z + 0.4 * y)


def _fit_projection(points: list[tuple[float, float]], width: int, height: int, point_type):
    """
    Build a transform that fits projected points inside a widget rectangle.
    """
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    scale = 0.8 * min(width / span_x, height / span_y)
    offset_x = (width - scale * (min_x + max_x)) / 2
    offset_y = (height + scale * (min_y + max_y)) / 2

    def transform(point: tuple[float, float]):
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
