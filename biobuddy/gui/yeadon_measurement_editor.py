from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..characteristics.yeadon import (
    YEADON_MEASUREMENT_NAMES,
    YEADON_MEASUREMENT_SPECS,
    YeadonDensitySet,
    YeadonMeasurementSpec,
    YeadonMeasures,
    YeadonTable,
)
from ..utils.enums import LengthUnits

_METERS_PER_UNIT = {
    LengthUnits.M: 1.0,
    LengthUnits.CM: 0.01,
    LengthUnits.MM: 0.001,
}
_UNIT_LABELS = {
    LengthUnits.M: "m",
    LengthUnits.CM: "cm",
    LengthUnits.MM: "mm",
}


@dataclass(frozen=True)
class YeadonIllustrationPrimitive:
    kind: str
    points: tuple[tuple[float, float], ...]
    label: str = ""
    highlight: bool = False


def parse_yeadon_measurement_values(
    values: Mapping[str, str | float | int],
) -> dict[str, float]:
    """
    Convert GUI-friendly measurement values to the dictionary expected by yeadon.
    """
    measurements: dict[str, float] = {}
    errors = []
    for name in YEADON_MEASUREMENT_NAMES:
        raw_value = values.get(name, "")
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
        if raw_value == "":
            errors.append(f"{name} is missing")
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"{name} is not numeric")
            continue
        if value <= 0:
            errors.append(f"{name} must be positive")
            continue
        measurements[name] = value
    if errors:
        raise ValueError("; ".join(errors))
    return measurements


def yeadon_measurement_illustration(
    name: str,
) -> tuple[YeadonIllustrationPrimitive, ...]:
    """
    Return normalized drawing primitives for one Yeadon measurement.
    """
    spec = _measurement_spec(name)
    base = (
        YeadonIllustrationPrimitive("line", ((0.50, 0.08), (0.50, 0.36)), "torso"),
        YeadonIllustrationPrimitive("line", ((0.30, 0.20), (0.70, 0.20)), "shoulders"),
        YeadonIllustrationPrimitive("line", ((0.30, 0.20), (0.20, 0.62)), "left arm"),
        YeadonIllustrationPrimitive("line", ((0.70, 0.20), (0.80, 0.62)), "right arm"),
        YeadonIllustrationPrimitive("line", ((0.43, 0.36), (0.38, 0.86)), "left leg"),
        YeadonIllustrationPrimitive("line", ((0.57, 0.36), (0.62, 0.86)), "right leg"),
        YeadonIllustrationPrimitive("ellipse", ((0.43, 0.02), (0.57, 0.14)), "head"),
    )
    return base + (_highlight_for_spec(spec),)


def save_yeadon_model(table: YeadonTable, filepath: str | Path) -> None:
    """
    Save the simple Yeadon segment model as a bioMod file.
    """
    table.to_simple_model().to_biomod(filepath=str(filepath), with_mesh=False)


def launch_yeadon_measurement_editor(
    yeadon_table: YeadonTable | None = None,
) -> None:
    """
    Launch a Qt desktop editor for entering Yeadon anthropometric measurements.

    Parameters
    ----------
    yeadon_table : YeadonTable, optional
        An optional YeadonTable object to pre-fill the editor with existing measurements.
    """
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QDoubleValidator, QPainter, QPen
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QSplitter,
            QVBoxLayout,
            QWidget,
        )

        qt_alignment_center = Qt.AlignmentFlag.AlignCenter
        qt_horizontal = Qt.Orientation.Horizontal
        qpaint_antialiasing = QPainter.RenderHint.Antialiasing
    except ImportError:
        try:
            from PyQt5.QtCore import Qt
            from PyQt5.QtGui import QColor, QDoubleValidator, QPainter, QPen
            from PyQt5.QtWidgets import (
                QApplication,
                QCheckBox,
                QComboBox,
                QFileDialog,
                QFormLayout,
                QGridLayout,
                QGroupBox,
                QHBoxLayout,
                QLabel,
                QLineEdit,
                QListWidget,
                QMainWindow,
                QMessageBox,
                QPushButton,
                QScrollArea,
                QSplitter,
                QVBoxLayout,
                QWidget,
            )

            qt_alignment_center = Qt.AlignCenter
            qt_horizontal = Qt.Horizontal
            qpaint_antialiasing = QPainter.Antialiasing
        except ImportError as error:
            raise ImportError(
                "The Yeadon measurement editor requires a working Qt binding. "
                "Install BioBuddy with `pip install biobuddy[gui]` or use an environment where PyQt5 is available."
            ) from error

    class IllustrationWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.measurement_name = YEADON_MEASUREMENT_NAMES[0]
            self.setMinimumSize(360, 440)

        def set_measurement(self, measurement_name: str) -> None:
            self.measurement_name = measurement_name
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(qpaint_antialiasing)
            painter.fillRect(self.rect(), QColor("white"))
            primitives = yeadon_measurement_illustration(self.measurement_name)
            width = self.width()
            height = self.height()

            def xy(point):
                return int(point[0] * width), int(point[1] * height)

            for primitive in primitives:
                color = QColor("#dc2626" if primitive.highlight else "#374151")
                painter.setPen(QPen(color, 5 if primitive.highlight else 2))
                if primitive.kind == "line":
                    start, end = primitive.points
                    painter.drawLine(*xy(start), *xy(end))
                elif primitive.kind == "ellipse":
                    top_left, bottom_right = primitive.points
                    x0, y0 = xy(top_left)
                    x1, y1 = xy(bottom_right)
                    painter.drawEllipse(x0, y0, x1 - x0, y1 - y0)
            painter.setPen(QPen(QColor("#111827"), 1))
            painter.drawText(self.rect(), qt_alignment_center, self.measurement_name)

    class YeadonMeasurementEditor(QMainWindow):
        def __init__(self, yeadon_table: YeadonTable | None = None):
            super().__init__()
            self.setWindowTitle("Yeadon measurements")
            self.measurement_fields: dict[str, QLineEdit] = {}
            self.yeadon_table: YeadonTable | None = None
            self._highlighted_field: QLineEdit | None = None

            root = QWidget()
            root_layout = QVBoxLayout(root)
            splitter = QSplitter(qt_horizontal)
            root_layout.addWidget(splitter)
            self.setCentralWidget(root)

            self.measurement_list = QListWidget()
            for spec in YEADON_MEASUREMENT_SPECS:
                self.measurement_list.addItem(f"{spec.name} - {spec.group}")
            self.measurement_list.currentRowChanged.connect(self._select_measurement)
            splitter.addWidget(self.measurement_list)

            form_widget = QWidget()
            form_layout = QVBoxLayout(form_widget)
            settings = QGroupBox("Subject")
            settings_layout = QFormLayout(settings)
            self.total_mass_field = QLineEdit()
            self.total_mass_field.setValidator(QDoubleValidator(0.0, 1000.0, 6))
            self.density_field = QComboBox()
            for density_set in YeadonDensitySet:
                self.density_field.addItem(density_set.value)
            self.symmetric_field = QCheckBox()
            self.symmetric_field.setChecked(True)
            self.units_field = QComboBox()
            for units in LengthUnits:
                self.units_field.addItem(f"{units.value} ({_UNIT_LABELS[units]})", units)
            self._current_units = self.units_field.currentData()
            self.units_field.currentIndexChanged.connect(self._on_units_changed)
            settings_layout.addRow("Total mass (kg)", self.total_mass_field)
            settings_layout.addRow("Density set", self.density_field)
            settings_layout.addRow("Symmetric", self.symmetric_field)
            settings_layout.addRow("Measurement units", self.units_field)
            form_layout.addWidget(settings)

            self.measurements_group = QGroupBox(f"Measurements ({_UNIT_LABELS[self._current_units]})")
            measurements_layout = QGridLayout(self.measurements_group)
            validator = QDoubleValidator(0.0, 100000.0, 6)
            for index, spec in enumerate(YEADON_MEASUREMENT_SPECS):
                label = QLabel(spec.name)
                field = QLineEdit()
                field.setValidator(validator)
                self.measurement_fields[spec.name] = field
                row = index // 3
                column = (index % 3) * 2
                measurements_layout.addWidget(label, row, column)
                measurements_layout.addWidget(field, row, column + 1)
            self.measurements_scroll = QScrollArea()
            self.measurements_scroll.setWidget(self.measurements_group)
            self.measurements_scroll.setWidgetResizable(True)
            form_layout.addWidget(self.measurements_scroll)

            action_layout = QHBoxLayout()
            self.save_button = QPushButton("Save as .bioMod")
            self.save_button.clicked.connect(self._save_model)
            action_layout.addWidget(self.save_button)
            self.export_button = QPushButton("Export measurements")
            self.export_button.clicked.connect(self._export_measurements)
            action_layout.addWidget(self.export_button)
            self.import_button = QPushButton("Import measurements")
            self.import_button.clicked.connect(self._import_measurements)
            action_layout.addWidget(self.import_button)
            self.result_label = QLabel("")
            action_layout.addWidget(self.result_label)
            form_layout.addLayout(action_layout)
            splitter.addWidget(form_widget)

            preview_widget = QWidget()
            preview_layout = QVBoxLayout(preview_widget)
            self.selected_label = QLabel("")
            self.selected_description = QLabel("")
            self.selected_description.setWordWrap(True)
            self.illustration = IllustrationWidget()
            preview_layout.addWidget(self.selected_label)
            preview_layout.addWidget(self.selected_description)
            preview_layout.addWidget(self.illustration)
            splitter.addWidget(preview_widget)

            splitter.setSizes([180, 680, 360])
            self.measurement_list.setCurrentRow(0)

            if yeadon_table is not None:
                self._load_table_into_fields(yeadon_table)

        def _select_measurement(self, row: int) -> None:
            if row < 0:
                return
            spec = YEADON_MEASUREMENT_SPECS[row]
            self.selected_label.setText(spec.label)
            self.selected_description.setText(spec.description)
            self.illustration.set_measurement(spec.name)

            if self._highlighted_field is not None:
                self._highlighted_field.setStyleSheet("")
            field = self.measurement_fields[spec.name]
            field.setStyleSheet("background-color: #fef08a; border: 1px solid #ca8a04;")
            self._highlighted_field = field
            self.measurements_scroll.ensureWidgetVisible(field)

        def _on_units_changed(self, index: int) -> None:
            new_units = self.units_field.itemData(index)
            if new_units is None or new_units == self._current_units:
                return
            old_multiplier = _METERS_PER_UNIT[self._current_units]
            new_multiplier = _METERS_PER_UNIT[new_units]
            for field in self.measurement_fields.values():
                text = field.text().strip()
                if not text:
                    continue
                try:
                    value = float(text)
                except ValueError:
                    continue
                field.setText(f"{value * old_multiplier / new_multiplier:.6g}")
            self._current_units = new_units
            self.measurements_group.setTitle(f"Measurements ({_UNIT_LABELS[new_units]})")

        def _build_table_from_fields(self) -> YeadonTable:
            measurements = parse_yeadon_measurement_values(
                {name: field.text() for name, field in self.measurement_fields.items()}
            )
            measures = YeadonMeasures(units=self._current_units, **measurements)
            total_mass = self.total_mass_field.text().strip()
            table = YeadonTable(
                symmetric=self.symmetric_field.isChecked(),
                density_set=self.density_field.currentText(),
                total_mass=float(total_mass) if total_mass else None,
            )
            table.from_measurements(measures)
            return table

        def _load_table_into_fields(self, table: YeadonTable) -> None:
            self.yeadon_table = table
            self.symmetric_field.setChecked(bool(table.symmetric))
            density_index = self.density_field.findText(table.density_set)
            if density_index >= 0:
                self.density_field.setCurrentIndex(density_index)
            if table.total_mass is not None:
                self.total_mass_field.setText(f"{table.total_mass:g}")
            if table.measures is not None:
                multiplier = _METERS_PER_UNIT[self._current_units]
                for name in YEADON_MEASUREMENT_NAMES:
                    value_in_meters = getattr(table.measures, name)
                    self.measurement_fields[name].setText(f"{value_in_meters / multiplier:.6g}")
            if table.human is not None:
                self.result_label.setText(f"Mass: {table.mass:0.3f} kg")

        def _build_table_or_show_error(self) -> YeadonTable | None:
            """
            Build a YeadonTable from the current fields, computing it on the fly. Used by every action
            that needs an up-to-date table (saving the model, exporting measurements, ...) so there is no
            separate "Compute" step to remember to run first.
            """
            try:
                table = self._build_table_from_fields()
            except Exception as error:
                QMessageBox.critical(self, "Yeadon measurements", str(error))
                return None
            self.yeadon_table = table
            self.result_label.setText(f"Mass: {table.mass:0.3f} kg")
            return table

        def _save_model(self) -> None:
            table = self._build_table_or_show_error()
            if table is None:
                return
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Save Yeadon model",
                "yeadon_model.bioMod",
                "BioMod files (*.bioMod)",
            )
            if not filepath:
                return
            try:
                save_yeadon_model(table, filepath)
            except Exception as error:
                QMessageBox.critical(self, "Unable to save Yeadon model", str(error))

        def _export_measurements(self) -> None:
            table = self._build_table_or_show_error()
            if table is None:
                return
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Export Yeadon measurements",
                "yeadon_measures.txt",
                "Text files (*.txt)",
            )
            if not filepath:
                return
            try:
                table.to_file(filepath)
            except Exception as error:
                QMessageBox.critical(self, "Unable to export measurements", str(error))

        def _import_measurements(self) -> None:
            filepath, _ = QFileDialog.getOpenFileName(
                self,
                "Import Yeadon measurements",
                "",
                "Text files (*.txt);;All files (*)",
            )
            if not filepath:
                return
            table = YeadonTable(
                symmetric=self.symmetric_field.isChecked(),
                density_set=self.density_field.currentText(),
            )
            try:
                table.from_file(filepath)
            except Exception as error:
                QMessageBox.critical(self, "Unable to import measurements", str(error))
                return
            self._load_table_into_fields(table)

    app = QApplication.instance() or QApplication([])
    window = YeadonMeasurementEditor(yeadon_table)
    window.resize(1280, 760)
    window.show()
    app.exec()


def _measurement_spec(name: str) -> YeadonMeasurementSpec:
    for spec in YEADON_MEASUREMENT_SPECS:
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown Yeadon measurement '{name}'.")


def _highlight_for_spec(spec: YeadonMeasurementSpec) -> YeadonIllustrationPrimitive:
    region_x = {
        "Torso": 0.50,
        "Left arm": 0.25,
        "Right arm": 0.75,
        "Left leg": 0.40,
        "Right leg": 0.60,
    }[spec.group]
    level = _measurement_level(spec.name)
    y = 0.10 + min(level, 9) * 0.075

    if spec.kind == "length":
        return YeadonIllustrationPrimitive(
            "line",
            ((region_x, max(0.05, y - 0.08)), (region_x, y + 0.08)),
            spec.name,
            True,
        )
    if spec.kind == "perimeter":
        return YeadonIllustrationPrimitive(
            "ellipse",
            ((region_x - 0.08, y - 0.035), (region_x + 0.08, y + 0.035)),
            spec.name,
            True,
        )
    if spec.kind == "width":
        return YeadonIllustrationPrimitive(
            "line",
            ((region_x - 0.10, y), (region_x + 0.10, y)),
            spec.name,
            True,
        )
    return YeadonIllustrationPrimitive(
        "line",
        ((region_x - 0.07, y + 0.05), (region_x + 0.07, y - 0.05)),
        spec.name,
        True,
    )


def _measurement_level(name: str) -> int:
    for character in reversed(name):
        if character.isdigit():
            return int(character)
    return 0
