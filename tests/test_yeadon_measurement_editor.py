import pytest

from biobuddy import YEADON_MEASUREMENT_NAMES
from biobuddy.gui.yeadon_measurement_editor import (
    parse_yeadon_measurement_values,
    save_yeadon_model,
    yeadon_measurement_illustration,
)


def test_parse_yeadon_measurement_values():
    values = {name: "1.25" for name in YEADON_MEASUREMENT_NAMES}

    parsed = parse_yeadon_measurement_values(values)

    assert parsed == {name: 1.25 for name in YEADON_MEASUREMENT_NAMES}


def test_parse_yeadon_measurement_values_reports_missing_invalid_and_non_positive():
    values = {name: "1.0" for name in YEADON_MEASUREMENT_NAMES}
    values["Ls1L"] = ""
    values["La2L"] = "abc"
    values["Lj1L"] = "0"

    with pytest.raises(ValueError) as error:
        parse_yeadon_measurement_values(values)

    message = str(error.value)
    assert "Ls1L is missing" in message
    assert "La2L is not numeric" in message
    assert "Lj1L must be positive" in message


@pytest.mark.parametrize("name", ["Ls4d", "La4w", "Lb3p", "Lj5L", "Lk6d"])
def test_yeadon_measurement_illustration_has_one_highlight(name):
    primitives = yeadon_measurement_illustration(name)

    highlights = [primitive for primitive in primitives if primitive.highlight]
    assert len(highlights) == 1
    assert highlights[0].label == name
    assert len(primitives) > 1


def test_save_yeadon_model(tmp_path):
    class FakeModel:
        def __init__(self):
            self.saved = []

        def to_biomod(self, filepath, with_mesh=True):
            self.saved.append((filepath, with_mesh))

    class FakeTable:
        def __init__(self):
            self.model = FakeModel()

        def to_simple_model(self):
            return self.model

    table = FakeTable()
    filepath = tmp_path / "yeadon.bioMod"

    save_yeadon_model(table, filepath)

    assert table.model.saved == [(str(filepath), False)]
