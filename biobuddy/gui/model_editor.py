import json
import math
from pathlib import Path

import numpy as np

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
    remove_segment_from_draft,
    remove_virtual_marker_from_draft,
    set_segment_marker_technical,
    unassign_markers_from_segment,
    update_segment_parent_in_draft,
    update_segment_settings_in_draft,
    validate_c3d_workflow_draft,
)
from ..utils.marker_data import C3dData


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
            QTabWidget,
            QTreeWidget,
            QTreeWidgetItem,
            QVBoxLayout,
            QWidget,
        )

        qt_alignment_center = Qt.AlignmentFlag.AlignCenter
        qt_horizontal = Qt.Orientation.Horizontal
        qt_match_exact = Qt.MatchFlag.MatchExactly
        qt_match_recursive = Qt.MatchFlag.MatchRecursive
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
                QTabWidget,
                QTreeWidget,
                QTreeWidgetItem,
                QVBoxLayout,
                QWidget,
            )

            qt_alignment_center = Qt.AlignCenter
            qt_horizontal = Qt.Horizontal
            qt_match_exact = Qt.MatchExactly
            qt_match_recursive = Qt.MatchRecursive
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
        end_list = QListWidget()
        end_list.setSelectionMode(qt_extended_selection)
        axis_combo = QComboBox()
        axis_combo.addItems(["x", "y", "z"])
        if index == 1:
            axis_combo.setCurrentText("y")
        return {
            "start_list": start_list,
            "end_list": end_list,
            "axis_combo": axis_combo,
            "keep_checkbox": QCheckBox("Keep this vector"),
            "add_start_button": QPushButton("Add to start"),
            "add_end_button": QPushButton("Add to end"),
            "remove_start_button": QPushButton("Remove start"),
            "remove_end_button": QPushButton("Remove end"),
        }

    def _axis_vector_layout(index: int, controls: dict[str, object]):
        """
        Build the layout for one repeated anatomical frame vector.
        """
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"Vector {index + 1} start markers"))
        layout.addWidget(controls["start_list"])
        start_buttons = QHBoxLayout()
        start_buttons.addWidget(controls["add_start_button"])
        start_buttons.addWidget(controls["remove_start_button"])
        layout.addLayout(start_buttons)
        layout.addWidget(QLabel(f"Vector {index + 1} end markers"))
        layout.addWidget(controls["end_list"])
        end_buttons = QHBoxLayout()
        end_buttons.addWidget(controls["add_end_button"])
        end_buttons.addWidget(controls["remove_end_button"])
        layout.addLayout(end_buttons)
        layout.addWidget(QLabel(f"Vector {index + 1} axis"))
        layout.addWidget(controls["axis_combo"])
        layout.addWidget(controls["keep_checkbox"])
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
            self.axes = ()
            self.current_vectors = ()
            self.yaw = -0.6
            self.pitch = 0.35
            self._last_mouse_position = None

        def set_context(
            self,
            c3d_data,
            marker_names: tuple[str, ...],
            axes: tuple[object, ...],
            current_vectors: tuple[tuple[str, tuple[str, ...], tuple[str, ...], bool], ...],
        ) -> None:
            self.c3d_data = c3d_data
            self.marker_names = marker_names
            self.axes = axes
            self.current_vectors = current_vectors
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(qpaint_antialiasing)
            painter.fillRect(self.rect(), QColor("white"))
            if self.c3d_data is None:
                painter.drawText(self.rect(), qt_alignment_center, "Choose a C3D to preview marker axes")
                return
            marker_points = {
                marker_name: _marker_preview_position(self.c3d_data, marker_name)
                for marker_name in self.marker_names
                if marker_name in self.c3d_data.marker_names
            }
            marker_points = {name: point for name, point in marker_points.items() if point is not None}
            axis_segments = []
            for axis in self.axes:
                start = _mean_preview_position(self.c3d_data, axis.start_markers)
                end = _mean_preview_position(self.c3d_data, axis.end_markers)
                if start is not None and end is not None:
                    axis_segments.append((axis.axis, axis.keep_vector, start, end))
            temporary_segments = []
            for axis_name, start_markers, end_markers, keep_vector in self.current_vectors:
                start = _mean_preview_position(self.c3d_data, start_markers)
                end = _mean_preview_position(self.c3d_data, end_markers)
                if start is not None and end is not None:
                    temporary_segments.append((axis_name, keep_vector, start, end))
            points = list(marker_points.values())
            for _, _, start, end in axis_segments:
                points.extend((start, end))
            for _, _, start, end in temporary_segments:
                points.extend((start, end))
            if len(points) == 0:
                painter.drawText(self.rect(), qt_alignment_center, "No visible marker for the selected segment")
                return
            projected_points = [_rotate_preview_point(point, self.yaw, self.pitch) for point in points]
            transform = _fit_projection(projected_points, self.width(), self.height(), QPointF)

            painter.setPen(QPen(QColor("#2563eb"), 1))
            painter.setBrush(QColor("#2563eb"))
            for marker_name, point in marker_points.items():
                center = transform(_rotate_preview_point(point, self.yaw, self.pitch))
                painter.drawEllipse(center, 4, 4)
                painter.drawText(center.x() + 5, center.y() - 5, marker_name)

            axis_colors = {"x": "#dc2626", "y": "#16a34a", "z": "#2563eb", "": "#6b7280"}
            for axis_name, keep_vector, start, end in axis_segments:
                painter.setPen(QPen(QColor(axis_colors.get(axis_name, "#6b7280")), 4 if keep_vector else 2))
                painter.drawLine(
                    transform(_rotate_preview_point(start, self.yaw, self.pitch)),
                    transform(_rotate_preview_point(end, self.yaw, self.pitch)),
                )

            for axis_name, keep_vector, start, end in temporary_segments:
                painter.setPen(QPen(QColor("#f59e0b"), 4 if keep_vector else 2))
                painter.drawLine(
                    transform(_rotate_preview_point(start, self.yaw, self.pitch)),
                    transform(_rotate_preview_point(end, self.yaw, self.pitch)),
                )

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
            self.c3d_data = None
            self.groups = ()
            self.virtual_markers = ()
            self.selected_marker_name = ""
            self.selected_method = "pointing"
            self.proximal_segment_name = ""
            self.distal_segment_name = ""
            self.yaw = -0.6
            self.pitch = 0.35
            self._last_mouse_position = None

        def set_context(
            self,
            c3d_data,
            groups: tuple[object, ...],
            virtual_markers: tuple[object, ...],
            selected_marker_name: str,
            selected_method: str,
            proximal_segment_name: str,
            distal_segment_name: str,
        ) -> None:
            self.c3d_data = c3d_data
            self.groups = groups
            self.virtual_markers = virtual_markers
            self.selected_marker_name = selected_marker_name
            self.selected_method = selected_method
            self.proximal_segment_name = proximal_segment_name
            self.distal_segment_name = distal_segment_name
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(qpaint_antialiasing)
            painter.fillRect(self.rect(), QColor("white"))
            if self.c3d_data is None:
                painter.drawText(self.rect(), qt_alignment_center, "Choose a C3D to preview virtual markers")
                return

            marker_records = []
            for segment_index, group in enumerate(self.groups):
                for marker_name in group.marker_names:
                    point = _marker_preview_position(self.c3d_data, marker_name)
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

            virtual_points = []
            for marker in self.virtual_markers:
                if marker.method == "marker_mean":
                    point = _mean_preview_position(self.c3d_data, _split_marker_names(marker.source))
                else:
                    point = _mean_preview_position(self.c3d_data, _split_marker_names(marker.source))
                if point is not None:
                    virtual_points.append((marker.name, point, marker.segment_name))

            solution_points = []
            if self.selected_method in {"score", "sara"}:
                proximal_point = self._technical_segment_center(self.proximal_segment_name)
                distal_point = self._technical_segment_center(self.distal_segment_name)
                if proximal_point is not None:
                    solution_points.append(("proximal", proximal_point))
                if distal_point is not None:
                    solution_points.append(("distal", distal_point))

            points = [record[1] for record in marker_records]
            points.extend(point for _, point, _ in virtual_points)
            points.extend(point for _, point in solution_points)
            if len(points) == 0:
                painter.drawText(self.rect(), qt_alignment_center, "No visible marker for this virtual marker context")
                return

            projected_points = [_rotate_preview_point(point, self.yaw, self.pitch) for point in points]
            transform = _fit_projection(projected_points, self.width(), self.height(), QPointF)

            for marker_name, point, segment_name, is_technical, segment_index in marker_records:
                color = QColor(_segment_preview_color(segment_index))
                center = transform(_rotate_preview_point(point, self.yaw, self.pitch))
                painter.setPen(QPen(color, 1))
                painter.setBrush(color)
                if is_technical:
                    painter.drawRect(int(center.x()) - 4, int(center.y()) - 4, 8, 8)
                else:
                    painter.drawEllipse(center, 4, 4)
                painter.drawText(center.x() + 5, center.y() - 5, f"{marker_name} ({segment_name})")

            painter.setPen(QPen(QColor("#7c3aed"), 2))
            painter.setBrush(QColor("#7c3aed"))
            for marker_name, point, segment_name in virtual_points:
                center = transform(_rotate_preview_point(point, self.yaw, self.pitch))
                painter.drawEllipse(center, 7, 7)
                painter.drawText(center.x() + 8, center.y() - 8, f"{marker_name} | {segment_name}")

            solution_colors = {"proximal": "#f97316", "distal": "#0891b2"}
            for label, point in solution_points:
                center = transform(_rotate_preview_point(point, self.yaw, self.pitch))
                painter.setPen(QPen(QColor(solution_colors[label]), 3))
                painter.setBrush(QColor(solution_colors[label]))
                painter.drawRect(int(center.x()) - 6, int(center.y()) - 6, 12, 12)
                painter.drawText(center.x() + 8, center.y() - 8, f"{label} solution")
            if len(solution_points) == 2:
                painter.setPen(QPen(QColor("#111827"), 1))
                painter.drawLine(
                    transform(_rotate_preview_point(solution_points[0][1], self.yaw, self.pitch)),
                    transform(_rotate_preview_point(solution_points[1][1], self.yaw, self.pitch)),
                )

            self._draw_legend(painter)

        def _technical_segment_center(self, segment_name: str) -> tuple[float, float, float] | None:
            for group in self.groups:
                if group.segment_name == segment_name:
                    marker_names = group.technical_marker_names if group.technical_marker_names else group.marker_names
                    return _mean_preview_position(self.c3d_data, marker_names)
            return None

        def _draw_legend(self, painter) -> None:
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.setBrush(QColor(255, 255, 255, 225))
            painter.drawRect(8, 8, 255, 82)
            painter.drawText(14, 24, "Legend")
            _draw_legend_point(painter, QPointF(20, 40), QColor("#2563eb"), "Anatomical/additional marker", 36, 44)
            painter.setPen(QPen(QColor("#2563eb"), 1))
            painter.setBrush(QColor("#2563eb"))
            painter.drawRect(16, 52, 8, 8)
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.drawText(36, 62, "Technical marker")
            _draw_legend_point(painter, QPointF(20, 74), QColor("#7c3aed"), "Virtual marker", 36, 78)

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
            self.workflow_draft = c3d_workflow_draft(self.presets[0])
            self.workflow_marker_pool = _marker_pool_from_draft(self.workflow_draft)

            self.preset_combo = QComboBox()
            for preset in self.presets:
                self.preset_combo.addItem(_c3d_preset_label(preset))
            self.preset_combo.currentIndexChanged.connect(self._update_preset_details)

            self.c3d_path = QLineEdit()
            self.c3d_path.setReadOnly(True)
            self.choose_c3d_button = QPushButton("Choose C3D file")
            self.choose_c3d_button.clicked.connect(self._choose_c3d_file)
            self.generate_template_button = QPushButton("Generate template")
            self.generate_template_button.clicked.connect(self._generate_template)

            self.status_label = QLabel()
            self.summary_label = QLabel()
            self.feature_list = QListWidget()
            self.feature_list.setMinimumHeight(180)
            self.feature_list.itemSelectionChanged.connect(self._load_selected_virtual_marker_into_form)
            self.step_list = QListWidget()
            self.marker_list = QListWidget()
            self.marker_list.setSelectionMode(qt_extended_selection)
            self.show_all_markers_checkbox = QCheckBox("Show markers already used by other segments")
            self.show_all_markers_checkbox.setChecked(True)
            self.show_all_markers_checkbox.stateChanged.connect(self._update_available_marker_list)
            self.segment_marker_list = QListWidget()
            self.segment_marker_list.itemSelectionChanged.connect(self._update_assigned_marker_list)
            self.workflow_parent_combo = QComboBox()
            self.workflow_parent_combo.currentTextChanged.connect(self._set_workflow_segment_parent)
            self.assigned_marker_list = QListWidget()
            self.assigned_marker_list.setSelectionMode(qt_extended_selection)
            self.assigned_marker_list.itemSelectionChanged.connect(self._sync_assigned_marker_technical_checkbox)
            self.assigned_marker_technical_checkbox = QCheckBox("Selected markers are technical")
            self.assigned_marker_technical_checkbox.stateChanged.connect(self._set_selected_assigned_markers_technical)
            self.axis_list = QListWidget()
            self.anatomical_segment_list = QListWidget()
            self.anatomical_segment_list.itemSelectionChanged.connect(self._update_anatomical_segment_details)
            self.axis_marker_source_list = QListWidget()
            self.axis_marker_source_list.setSelectionMode(qt_extended_selection)
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
                controls["keep_checkbox"].stateChanged.connect(self._update_segment_axis_preview)
            self.save_segment_axis_button = QPushButton("Add/update anatomical frame vectors")
            self.save_segment_axis_button.clicked.connect(self._save_segment_axis_from_lists)
            self.segment_axis_preview = C3dSegmentAxisPreviewWidget()
            self.virtual_marker_name_edit = QLineEdit()
            self.virtual_marker_segment_combo = QComboBox()
            self.virtual_marker_method_combo = QComboBox()
            self.virtual_marker_method_combo.addItems(
                ["pointing", "score", "sara", "marker_mean", "regression", "equation"]
            )
            self.virtual_marker_method_combo.currentTextChanged.connect(self._sync_virtual_marker_method_fields)
            self.virtual_marker_source_edit = QLineEdit()
            self.virtual_marker_source_edit.textChanged.connect(self._update_virtual_marker_preview)
            self.virtual_marker_equation_edit = QLineEdit()
            self.virtual_marker_c3d_role_combo = QComboBox()
            self.virtual_marker_proximal_combo = QComboBox()
            self.virtual_marker_proximal_combo.currentTextChanged.connect(self._update_virtual_marker_preview)
            self.virtual_marker_distal_combo = QComboBox()
            self.virtual_marker_distal_combo.currentTextChanged.connect(self._update_virtual_marker_preview)
            self.virtual_marker_info_label = QLabel()
            self.virtual_marker_info_label.setWordWrap(True)
            self.save_virtual_marker_button = QPushButton("Save virtual marker")
            self.save_virtual_marker_button.clicked.connect(self._save_workflow_virtual_marker_from_form)
            self.virtual_marker_preview = C3dVirtualMarkerPreviewWidget()
            self.segment_settings_list = QListWidget()
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
            self.add_virtual_marker_button = QPushButton("Add/edit virtual marker")
            self.add_virtual_marker_button.clicked.connect(self._add_workflow_virtual_marker)
            self.remove_virtual_marker_button = QPushButton("Remove virtual marker")
            self.remove_virtual_marker_button.clicked.connect(self._remove_workflow_virtual_marker)
            self.edit_segment_settings_button = QPushButton("Edit segment settings")
            self.edit_segment_settings_button.clicked.connect(self._edit_workflow_segment_settings)
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
            c3d_row = QHBoxLayout()
            c3d_row.addWidget(self.c3d_path)
            c3d_row.addWidget(self.choose_c3d_button)
            c3d_row.addWidget(self.generate_template_button)
            layout.addLayout(c3d_row)
            layout.addWidget(self.status_label)

            workflow_tabs = QTabWidget()
            workflow_tabs.addTab(self.step_list, "Pipeline")
            workflow_tabs.addTab(self._segment_workflow_tab(), "Technical segment")
            workflow_tabs.addTab(self._virtual_marker_workflow_tab(), "Virtual markers")
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

        def _segment_workflow_tab(self):
            widget = QWidget()
            layout = QVBoxLayout(widget)
            row = QHBoxLayout()
            row.addWidget(self.add_segment_button)
            row.addWidget(self.remove_segment_button)
            row.addStretch()
            layout.addLayout(row)
            layout.addWidget(QLabel("Segments"))
            layout.addWidget(self.segment_marker_list)
            marker_row = QHBoxLayout()
            parent_column = QVBoxLayout()
            parent_column.addWidget(QLabel("Parent segment"))
            parent_column.addWidget(self.workflow_parent_combo)
            parent_column.addStretch()
            left_column = QVBoxLayout()
            left_column.addWidget(QLabel("Available markers in main C3D"))
            left_column.addWidget(self.show_all_markers_checkbox)
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
            layout.addLayout(marker_row)
            return widget

        def _anatomical_segment_workflow_tab(self):
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.addWidget(QLabel("Anatomical segments"))
            layout.addWidget(self.anatomical_segment_list)
            axis_layout = QHBoxLayout()
            source_column = QVBoxLayout()
            source_column.addWidget(QLabel("Axis marker source"))
            source_column.addWidget(self.axis_marker_source_list)
            axis_layout.addLayout(source_column, 2)
            for index, controls in enumerate(self.axis_vector_controls):
                axis_layout.addLayout(_axis_vector_layout(index, controls), 2)
            save_column = QVBoxLayout()
            save_column.addWidget(QLabel("Frame"))
            save_column.addWidget(self.save_segment_axis_button)
            save_column.addStretch()
            axis_layout.addLayout(save_column, 1)
            axis_layout.addWidget(self.segment_axis_preview, 2)
            layout.addWidget(QLabel("Segment system of coordinates"))
            layout.addLayout(axis_layout)
            layout.addWidget(QLabel("Saved anatomical vectors"))
            layout.addWidget(self.axis_list)
            return widget

        def _virtual_marker_workflow_tab(self):
            widget = QWidget()
            layout = QHBoxLayout(widget)

            list_column = QVBoxLayout()
            list_column.addWidget(QLabel("Virtual markers"))
            list_column.addWidget(self.feature_list)
            row = QHBoxLayout()
            row.addWidget(self.add_virtual_marker_button)
            row.addWidget(self.remove_virtual_marker_button)
            list_column.addLayout(row)
            layout.addLayout(list_column, 1)

            form_column = QVBoxLayout()
            form = QFormLayout()
            form.addRow("Name", self.virtual_marker_name_edit)
            form.addRow("Segment", self.virtual_marker_segment_combo)
            form.addRow("Method", self.virtual_marker_method_combo)
            form.addRow("Source C3D / markers", self.virtual_marker_source_edit)
            form.addRow("C3D role", self.virtual_marker_c3d_role_combo)
            form.addRow("Technical proximal", self.virtual_marker_proximal_combo)
            form.addRow("Technical distal", self.virtual_marker_distal_combo)
            form.addRow("Equation / regression", self.virtual_marker_equation_edit)
            form_column.addLayout(form)
            form_column.addWidget(self.save_virtual_marker_button)
            form_column.addWidget(QLabel("Selected marker information"))
            form_column.addWidget(self.virtual_marker_info_label)
            form_column.addStretch()
            layout.addLayout(form_column, 1)

            preview_column = QVBoxLayout()
            preview_column.addWidget(QLabel("3D placement preview"))
            preview_column.addWidget(self.virtual_marker_preview)
            layout.addLayout(preview_column, 2)
            return widget

        def _segment_settings_workflow_tab(self):
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.addWidget(self.segment_settings_list)
            layout.addWidget(self.edit_segment_settings_button)
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
                "",
                "C3D files (*.c3d)",
            )
            if not filepath:
                return
            try:
                self.c3d_data = C3dData(filepath)
                self.c3d_path.setText(filepath)
                self.workflow_marker_pool = tuple(self.c3d_data.marker_names)
                self._update_preset_details()
            except Exception as error:
                QMessageBox.critical(self, "Unable to load C3D", str(error))

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
                Path(filepath).write_text(json.dumps(c3d_template_payload_from_draft(self.workflow_draft), indent=2))
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
            self._sync_virtual_marker_method_fields()
            self._update_virtual_marker_info_label(None)
            self._update_virtual_marker_preview()

        def _save_workflow_virtual_marker_from_form(self) -> None:
            name = self.virtual_marker_name_edit.text().strip()
            segment_name = self.virtual_marker_segment_combo.currentText().strip()
            method = self.virtual_marker_method_combo.currentText().strip()
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
                self._update_preset_details()
                self._select_virtual_marker_by_name(name)
            except Exception as error:
                QMessageBox.critical(self, "Unable to save virtual marker", str(error))

        def _virtual_marker_source_from_form(self, method: str) -> str:
            source = self.virtual_marker_source_edit.text().strip()
            c3d_role = self.virtual_marker_c3d_role_combo.currentText().strip()
            if method in {"score", "sara", "pointing", "regression", "equation"} and source == "":
                source = c3d_role
            return source

        def _virtual_marker_equation_from_form(self, method: str) -> str:
            equation = self.virtual_marker_equation_edit.text().strip()
            if method not in {"score", "sara"}:
                return equation if method in {"equation", "regression"} else ""
            parts = [
                f"proximal={self.virtual_marker_proximal_combo.currentText().strip()}",
                f"distal={self.virtual_marker_distal_combo.currentText().strip()}",
            ]
            if equation:
                parts.append(f"helper={equation}")
            return "; ".join(parts)

        def _remove_workflow_virtual_marker(self) -> None:
            name = self._selected_virtual_marker_name()
            if name is None:
                return
            self.workflow_draft = remove_virtual_marker_from_draft(self.workflow_draft, name)
            self._update_preset_details()

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
                "",
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
            for item in self.axis_marker_source_list.selectedItems():
                marker_name = item.text().split("|", maxsplit=1)[0].strip()
                if marker_name:
                    target_list.addItem(marker_name)
            self._update_segment_axis_preview()

        def _remove_selected_axis_markers(self, vector_index: int, endpoint: str) -> None:
            target_list = self._axis_endpoint_list(vector_index, endpoint)
            for item in target_list.selectedItems():
                target_list.takeItem(target_list.row(item))
            self._update_segment_axis_preview()

        def _axis_endpoint_list(self, vector_index: int, endpoint: str):
            key = "start_list" if endpoint == "start" else "end_list"
            return self.axis_vector_controls[vector_index][key]

        def _save_segment_axis_from_lists(self) -> None:
            segment_name = self._selected_anatomical_segment_name()
            if segment_name is None:
                return
            vector_specs = self._axis_vector_specs()
            if len(vector_specs) != 2:
                QMessageBox.critical(self, "Unable to save segment axis", "Two complete vectors are required.")
                return
            if sum(keep_vector for _, _, _, keep_vector in vector_specs) != 1:
                QMessageBox.critical(self, "Unable to save segment axis", "Choose exactly one vector to keep.")
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
            for index, controls in enumerate(self.axis_vector_controls):
                controls["start_list"].clear()
                controls["end_list"].clear()
                controls["keep_checkbox"].setChecked(False)
                if index >= len(axes):
                    continue
                axis = axes[index]
                controls["axis_combo"].setCurrentText(axis.axis if axis.axis in {"x", "y", "z"} else "x")
                controls["keep_checkbox"].setChecked(axis.keep_vector)
                for marker_name in axis.start_markers:
                    controls["start_list"].addItem(marker_name)
                for marker_name in axis.end_markers:
                    controls["end_list"].addItem(marker_name)

        def _update_segment_axis_preview(self) -> None:
            segment_name = self._selected_anatomical_segment_name()
            if segment_name is None:
                self.segment_axis_preview.set_context(self.c3d_data, (), (), ())
                return
            segment_marker_names = tuple(self.workflow_marker_pool)
            axes = tuple(axis for axis in self.workflow_draft.axes if axis.segment_name == segment_name)
            self.segment_axis_preview.set_context(
                self.c3d_data,
                segment_marker_names,
                axes,
                self._axis_vector_specs(),
            )

        def _selected_virtual_marker_name(self) -> str | None:
            if not self.feature_list.selectedItems():
                return None
            text = self.feature_list.selectedItems()[0].text()
            return None if text.startswith("No additional") else text.split("|", maxsplit=1)[0].strip()

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
                self._sync_virtual_marker_method_fields()
                self._update_virtual_marker_preview()
                return
            self.virtual_marker_name_edit.setText(marker.name)
            self.virtual_marker_segment_combo.setCurrentText(marker.segment_name)
            self.virtual_marker_method_combo.setCurrentText(marker.method)
            self.virtual_marker_source_edit.setText(marker.source)
            self.virtual_marker_equation_edit.setText(_strip_score_segment_payload(marker.equation))
            proximal, distal = _score_segments_from_payload(marker.equation)
            if proximal:
                self.virtual_marker_proximal_combo.setCurrentText(proximal)
            if distal:
                self.virtual_marker_distal_combo.setCurrentText(distal)
            if marker.source:
                self.virtual_marker_c3d_role_combo.setCurrentText(marker.source)
            self._sync_virtual_marker_method_fields()
            self._update_virtual_marker_info_label(marker)
            self._update_virtual_marker_preview()

        def _sync_virtual_marker_choices(self) -> None:
            current_segment = self.virtual_marker_segment_combo.currentText()
            current_role = self.virtual_marker_c3d_role_combo.currentText()
            current_proximal = self.virtual_marker_proximal_combo.currentText()
            current_distal = self.virtual_marker_distal_combo.currentText()

            segment_names = [group.segment_name for group in self.workflow_draft.segment_marker_groups]
            role_names = [assignment.role for assignment in self.workflow_draft.file_assignments]
            technical_segment_names = [
                group.segment_name
                for group in self.workflow_draft.segment_marker_groups
                if group.segment_type == "technical" or len(group.technical_marker_names) != 0
            ]
            if len(technical_segment_names) == 0:
                technical_segment_names = segment_names

            for combo, values, current_value in (
                (self.virtual_marker_segment_combo, segment_names, current_segment),
                (self.virtual_marker_c3d_role_combo, role_names, current_role),
                (self.virtual_marker_proximal_combo, technical_segment_names, current_proximal),
                (self.virtual_marker_distal_combo, technical_segment_names, current_distal),
            ):
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(values)
                if current_value in values:
                    combo.setCurrentText(current_value)
                combo.blockSignals(False)

        def _sync_virtual_marker_method_fields(self, _method: str | None = None) -> None:
            method = self.virtual_marker_method_combo.currentText()
            self.virtual_marker_source_edit.setEnabled(
                method in {"pointing", "score", "sara", "regression", "equation", "marker_mean"}
            )
            self.virtual_marker_c3d_role_combo.setEnabled(
                method in {"pointing", "score", "sara", "regression", "equation"}
            )
            self.virtual_marker_proximal_combo.setEnabled(method in {"score", "sara"})
            self.virtual_marker_distal_combo.setEnabled(method in {"score", "sara"})
            self.virtual_marker_equation_edit.setEnabled(method in {"equation", "regression", "score", "sara"})
            hints = {
                "pointing": "Choose the pointing C3D/role and optionally the marker or pointer-tip name in source.",
                "score": "Choose the functional C3D plus proximal and distal technical segments. SCoRE estimates a joint center.",
                "sara": "Choose the functional C3D plus proximal and distal technical segments. SARA estimates a rotation axis; use the equation field for orientation markers if needed.",
                "marker_mean": "Write comma-separated marker names in Source C3D / markers. Duplicates are allowed and are averaged.",
                "regression": "Choose the source C3D/role and name the predictive equation, for example example_predictive_hip_cor(D).",
                "equation": "Use a project-specific equation/helper name and the source markers/C3D it needs.",
            }
            self.virtual_marker_info_label.setText(hints.get(method, ""))
            self._update_virtual_marker_preview()

        def _update_virtual_marker_info_label(self, marker) -> None:
            if marker is None:
                self.virtual_marker_info_label.setText(
                    "Select a virtual marker to inspect its method, segment, source, and equation."
                )
                return
            source = marker.source if marker.source else "-"
            equation = marker.equation if marker.equation else "-"
            self.virtual_marker_info_label.setText(
                f"Name: {marker.name}\nSegment: {marker.segment_name}\nMethod: {marker.method}\n"
                f"Source: {source}\nEquation/settings: {equation}"
            )

        def _update_virtual_marker_preview(self, *_args) -> None:
            self.virtual_marker_preview.set_context(
                self.c3d_data,
                self.workflow_draft.segment_marker_groups,
                self.workflow_draft.virtual_markers,
                self.virtual_marker_name_edit.text().strip(),
                self.virtual_marker_method_combo.currentText().strip(),
                self.virtual_marker_proximal_combo.currentText().strip(),
                self.virtual_marker_distal_combo.currentText().strip(),
            )

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

            if len(self.workflow_draft.virtual_markers) == 0:
                self.feature_list.addItem("No additional virtual feature required by this preset.")
            else:
                for feature in self.workflow_draft.virtual_markers:
                    self.feature_list.addItem(
                        f"{feature.name} | {feature.segment_name} | {feature.method} | {feature.source}"
                    )
                if self.feature_list.currentItem() is None:
                    self.feature_list.setCurrentItem(self.feature_list.item(0))

            if len(self.workflow_draft.axes) == 0:
                self.axis_list.addItem("No axis definition yet.")
            else:
                for axis in self.workflow_draft.axes:
                    self.axis_list.addItem(
                        f"{axis.name} | {axis.segment_name} | {axis.axis} | {axis.method} | "
                        f"{','.join(axis.start_markers)} -> {','.join(axis.end_markers)}"
                    )

            for setting in self.workflow_draft.segment_settings:
                self.segment_settings_list.addItem(
                    f"{setting.segment_name} | translations={setting.translations or '-'} | "
                    f"rotations={setting.rotations or '-'} | child_translation={setting.child_translation} | "
                    f"initial_rotation={setting.initial_rotation_method}"
                )

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
                self.example_list.addItem(
                    f"{example.method} | source: {example.source_example} | equation: "
                    f"{example.equation_example or '-'} | {example.description}"
                )

            self.summary_label.setText(c3d_workflow_summary(preset, self.c3d_data))
            self._update_virtual_marker_preview()

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

            projected_joints = {name: _project_point(point) for name, point in self.scene.joints.items()}
            projected_markers = {name: _project_point(point) for name, point in self.scene.markers.items()}
            projected_axes = [
                (axis, _project_point(axis.start), _project_point(axis.end)) for axis in self.scene.segment_axes
            ]
            all_points = list(projected_joints.values()) + list(projected_markers.values())
            for path in self.scene.muscles.values():
                all_points.extend(_project_point(point) for point in path)
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
                    painter.drawLine(transform(_project_point(start)), transform(_project_point(end)))

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
            save_button = QPushButton("Save as .bioMod")
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
            filepath, _ = QFileDialog.getSaveFileName(self, "Save model", default_name, "BioMod files (*.bioMod)")
            if not filepath:
                return
            try:
                self.model.to_biomod(filepath=filepath)
            except Exception as error:
                QMessageBox.critical(self, "Unable to save model", str(error))

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
        return "Lower-limbs"
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


def _split_marker_names(text: str) -> tuple[str, ...]:
    """
    Split a comma/semicolon separated marker list while preserving duplicated markers.
    """
    return tuple(marker.strip() for marker in text.replace(";", ",").split(",") if marker.strip())


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


def _mean_preview_position(c3d_data, marker_names: tuple[str, ...]) -> tuple[float, float, float] | None:
    """
    Return the mean point for a possibly duplicated marker list.
    """
    points = [_marker_preview_position(c3d_data, marker_name) for marker_name in marker_names]
    points = [point for point in points if point is not None]
    if len(points) == 0:
        return None
    return tuple(float(value) for value in np.mean(np.asarray(points, dtype=float), axis=0))


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
