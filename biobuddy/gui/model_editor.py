from pathlib import Path

from .segment_editor import (
    SegmentEditorData,
    apply_segment_editor_data,
    get_segment_editor_data,
    load_model,
    validate_parent_name,
)


def launch_model_editor() -> None:
    """
    Launch the PySide6 desktop model editor.
    """
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QApplication,
            QFileDialog,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSplitter,
            QTreeWidget,
            QTreeWidgetItem,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as error:
        raise ImportError("The model editor requires PySide6. Install BioBuddy with `pip install biobuddy[gui]`.") from error

    class ModelEditorWindow(QMainWindow):
        """
        Minimal desktop editor for inspecting and editing segment properties.
        """

        def __init__(self):
            super().__init__()
            self.setWindowTitle("BioBuddy Model Editor")
            self.resize(1100, 700)
            self.model = None
            self.current_filepath: Path | None = None
            self.current_segment_name: str | None = None

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
            right_layout = QVBoxLayout(right_panel)
            right_layout.addWidget(QLabel("Segment properties"))
            right_layout.addLayout(form)
            right_layout.addWidget(self.apply_button)
            right_layout.addStretch()

            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.addWidget(self.tree)
            splitter.addWidget(right_panel)
            splitter.setSizes([350, 750])

            open_button = QPushButton("Open model")
            open_button.clicked.connect(self._open_model)
            save_button = QPushButton("Save as .bioMod")
            save_button.clicked.connect(self._save_model)

            toolbar = QHBoxLayout()
            toolbar.addWidget(open_button)
            toolbar.addWidget(save_button)
            toolbar.addStretch()

            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
            layout.addLayout(toolbar)
            layout.addWidget(splitter)
            self.setCentralWidget(central_widget)

        def _open_model(self) -> None:
            filepath, _ = QFileDialog.getOpenFileName(
                self,
                "Open model",
                "",
                "Biomechanical models (*.bioMod *.osim *.urdf)",
            )
            if not filepath:
                return
            try:
                self.model = load_model(filepath)
                self.current_filepath = Path(filepath)
                self._populate_tree()
            except Exception as error:
                QMessageBox.critical(self, "Unable to open model", str(error))

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
            except Exception as error:
                QMessageBox.critical(self, "Invalid segment values", str(error))

    app = QApplication.instance() or QApplication([])
    window = ModelEditorWindow()
    window.show()
    app.exec()


def _parse_float_list(text: str) -> list[float]:
    """
    Parse a comma- or space-separated float list from a line edit.
    """
    stripped_text = text.strip()
    if stripped_text == "":
        return []
    return [float(value) for value in stripped_text.replace(",", " ").split()]


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
