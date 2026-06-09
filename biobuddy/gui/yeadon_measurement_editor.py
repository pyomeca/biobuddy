from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..characteristics.yeadon import (
    YEADON_MEASUREMENT_NAMES,
    YEADON_MEASUREMENT_SPECS,
    YeadonDensitySet,
    YeadonMeasurementSpec,
    YeadonTable,
)


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


def launch_yeadon_measurement_editor() -> None:
    """
    Launch a Qt desktop editor for entering Yeadon anthropometric measurements.
    """
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QDoubleValidator, QPainter, QPen
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
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
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Yeadon measurements")
            self.measurement_fields: dict[str, QLineEdit] = {}
            self.computed_table = None

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
            settings_layout.addRow("Total mass (kg)", self.total_mass_field)
            settings_layout.addRow("Density set", self.density_field)
            settings_layout.addRow("Symmetric", self.symmetric_field)
            form_layout.addWidget(settings)

            measurements_group = QGroupBox("Measurements (m)")
            measurements_layout = QGridLayout(measurements_group)
            validator = QDoubleValidator(0.0, 10.0, 6)
            for index, spec in enumerate(YEADON_MEASUREMENT_SPECS):
                label = QLabel(spec.name)
                field = QLineEdit()
                field.setValidator(validator)
                self.measurement_fields[spec.name] = field
                row = index // 3
                column = (index % 3) * 2
                measurements_layout.addWidget(label, row, column)
                measurements_layout.addWidget(field, row, column + 1)
            scroll = QScrollArea()
            scroll.setWidget(measurements_group)
            scroll.setWidgetResizable(True)
            form_layout.addWidget(scroll)

            action_layout = QHBoxLayout()
            compute_button = QPushButton("Compute")
            compute_button.clicked.connect(self._compute)
            action_layout.addWidget(compute_button)
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

        def _select_measurement(self, row: int) -> None:
            if row < 0:
                return
            spec = YEADON_MEASUREMENT_SPECS[row]
            self.selected_label.setText(spec.label)
            self.selected_description.setText(spec.description)
            self.illustration.set_measurement(spec.name)

        def _compute(self) -> None:
            try:
                measurements = parse_yeadon_measurement_values(
                    {name: field.text() for name, field in self.measurement_fields.items()}
                )
                total_mass = self.total_mass_field.text().strip()
                table = YeadonTable(
                    measurements,
                    symmetric=self.symmetric_field.isChecked(),
                    density_set=self.density_field.currentText(),
                    total_mass=float(total_mass) if total_mass else None,
                )
            except Exception as error:
                QMessageBox.critical(self, "Yeadon measurements", str(error))
                return
            self.computed_table = table
            self.result_label.setText(f"Mass: {table.mass:0.3f} kg")

    app = QApplication.instance() or QApplication([])
    window = YeadonMeasurementEditor()
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
