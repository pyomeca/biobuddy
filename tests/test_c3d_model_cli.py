from pathlib import Path
from types import SimpleNamespace

import pytest

from biobuddy.gui import build_c3d_model, model_editor_cli
from biobuddy.gui.build_c3d_model import (
    DEFAULT_MOTIVE_57_C3D_FOLDER,
    DEFAULT_P6_MOTIVE_C3D_FOLDER,
    build_arg_parser,
    build_model_from_c3d_cli,
)
from biobuddy.gui.model_editor_cli import build_arg_parser as build_gui_arg_parser
from biobuddy.gui.c3d_model_creation import (
    C3dModelPreset,
    DEFAULT_STATIC_C3D_PATTERNS,
    c3d_model_preset_from_cli_value,
    default_static_virtual_points_for_c3d_model_preset,
    find_static_c3d_file,
)


@pytest.mark.parametrize(
    ("value", "preset"),
    (
        ("motive_57", C3dModelPreset.MOTIVE_57),
        ("motive-57", C3dModelPreset.MOTIVE_57),
        ("biomech-motive-57", C3dModelPreset.MOTIVE_57),
        ("motive_57_isb", C3dModelPreset.MOTIVE_57_ISB),
        ("motive-57-isb", C3dModelPreset.MOTIVE_57_ISB),
        ("biomech-motive-57-isb", C3dModelPreset.MOTIVE_57_ISB),
        ("full_body", C3dModelPreset.FULL_BODY),
        ("model202", C3dModelPreset.FULL_BODY),
        ("lower_limbs", C3dModelPreset.LOWER_LIMBS),
        ("lower-limbs-anatomical", C3dModelPreset.LOWER_LIMBS_ANATOMICAL),
    ),
)
def test_c3d_model_preset_from_cli_value_accepts_aliases(value, preset):
    assert c3d_model_preset_from_cli_value(value) == preset


def test_default_static_virtual_points_for_motive_57_adds_rab_gjc_points():
    definitions = default_static_virtual_points_for_c3d_model_preset(C3dModelPreset.MOTIVE_57)

    assert tuple(definition.name for definition in definitions) == ("LGJC", "RGJC")
    assert definitions[0].required_markers == ("LCAJ", "LHME", "LHLE")
    assert definitions[1].required_markers == ("RCAJ", "RHME", "RHLE")


def test_default_static_virtual_points_for_motive_57_isb_adds_rab_gjc_points():
    definitions = default_static_virtual_points_for_c3d_model_preset(C3dModelPreset.MOTIVE_57_ISB)

    assert tuple(definition.name for definition in definitions) == ("LGJC", "RGJC")


def test_build_c3d_model_parser_defaults_to_motive_57_data_folder():
    args = build_arg_parser().parse_args([])

    assert Path(args.c3d_folder) == DEFAULT_MOTIVE_57_C3D_FOLDER
    assert args.preset == C3dModelPreset.MOTIVE_57.value
    assert args.output is None
    assert args.no_default_virtual_points is False


def test_build_c3d_model_parser_accepts_p6_motive_shortcut():
    args = build_arg_parser().parse_args(["--p6-motive"])

    assert args.p6_motive is True


def test_build_c3d_model_cli_p6_shortcut_uses_p6_folder(monkeypatch):
    calls = []

    def fake_build_model_from_c3d_cli(*args, **kwargs):
        calls.append((args, kwargs))
        return DEFAULT_P6_MOTIVE_C3D_FOLDER / "motive_57.bioMod"

    monkeypatch.setattr(
        build_c3d_model,
        "build_model_from_c3d_cli",
        fake_build_model_from_c3d_cli,
    )

    assert build_c3d_model.main(["--p6-motive", "--quiet"]) == 0

    assert calls == [
        (
            (DEFAULT_P6_MOTIVE_C3D_FOLDER,),
            {
                "preset": C3dModelPreset.MOTIVE_57,
                "output": None,
                "add_default_virtual_points": True,
                "with_mesh": True,
                "marker_name_prefixes_to_strip": (),
                "quiet": True,
            },
        )
    ]


def test_default_static_c3d_patterns_include_case_sensitive_static_glob():
    assert "*Static*.c3d" in DEFAULT_STATIC_C3D_PATTERNS
    assert find_static_c3d_file.__defaults__[0] is DEFAULT_STATIC_C3D_PATTERNS


def test_model_editor_cli_accepts_motive_57_shortcut():
    args = build_gui_arg_parser().parse_args(["--motive-57"])

    assert args.motive_57 is True
    assert args.new_from_c3d is False
    assert args.preset is None
    assert args.c3d_folder is None


def test_model_editor_cli_accepts_motive_57_isb_shortcut():
    args = build_gui_arg_parser().parse_args(["--motive-57-isb"])

    assert args.motive_57_isb is True
    assert args.new_from_c3d is False
    assert args.preset is None
    assert args.c3d_folder is None


def test_model_editor_cli_accepts_p6_motive_shortcut():
    args = build_gui_arg_parser().parse_args(["--p6-motive"])

    assert args.p6_motive is True
    assert args.new_from_c3d is False
    assert args.preset is None
    assert args.c3d_folder is None


def test_model_editor_cli_accepts_explicit_c3d_context():
    args = build_gui_arg_parser().parse_args(
        [
            "--new-from-c3d",
            "--preset",
            "motive_57",
            "--c3d-folder",
            str(DEFAULT_MOTIVE_57_C3D_FOLDER),
        ]
    )

    assert args.new_from_c3d is True
    assert args.preset == "motive_57"
    assert args.c3d_folder == DEFAULT_MOTIVE_57_C3D_FOLDER


def test_model_editor_cli_motive_57_shortcut_launches_c3d_workflow(monkeypatch):
    calls = []

    def fake_launch_model_editor(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(model_editor_cli, "launch_model_editor", fake_launch_model_editor)

    assert model_editor_cli.main(["--motive-57"]) == 0

    assert calls == [
        {
            "c3d_preset": C3dModelPreset.MOTIVE_57,
            "c3d_folder": DEFAULT_MOTIVE_57_C3D_FOLDER,
            "open_c3d_dialog": True,
        }
    ]


def test_model_editor_cli_resolves_motive_57_shortcut_without_qt():
    args = build_gui_arg_parser().parse_args(["--motive-57"])

    assert model_editor_cli.resolve_launch_options(args) == {
        "c3d_preset": C3dModelPreset.MOTIVE_57,
        "c3d_folder": DEFAULT_MOTIVE_57_C3D_FOLDER,
        "open_c3d_dialog": True,
    }


def test_model_editor_cli_resolves_p6_motive_shortcut_without_qt():
    args = build_gui_arg_parser().parse_args(["--p6-motive"])

    assert model_editor_cli.resolve_launch_options(args) == {
        "c3d_preset": C3dModelPreset.MOTIVE_57,
        "c3d_folder": DEFAULT_P6_MOTIVE_C3D_FOLDER,
        "open_c3d_dialog": True,
    }


def test_model_editor_cli_explicit_context_opens_c3d_workflow(monkeypatch, tmp_path):
    calls = []

    def fake_launch_model_editor(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(model_editor_cli, "launch_model_editor", fake_launch_model_editor)

    assert model_editor_cli.main(["--preset", "motive_57", "--c3d-folder", str(tmp_path)]) == 0

    assert calls == [
        {
            "c3d_preset": "motive_57",
            "c3d_folder": tmp_path,
            "open_c3d_dialog": True,
        }
    ]


def test_build_model_from_c3d_cli_passes_normalized_options(monkeypatch, tmp_path):
    calls = []
    written = []

    class FakeModel:
        def to_biomod(self, filepath, with_mesh):
            written.append((Path(filepath), with_mesh))

    def fake_create_model_from_c3d_folder(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model=FakeModel(),
            preset=C3dModelPreset.MOTIVE_57,
            output_filename="motive_57.bioMod",
            static_data=SimpleNamespace(marker_names=("A", "B")),
            functional_data={"left_knee_sara": object()},
        )

    monkeypatch.setattr(
        build_c3d_model,
        "create_model_from_c3d_folder",
        fake_create_model_from_c3d_folder,
    )

    output_path = build_model_from_c3d_cli(
        tmp_path,
        preset="motive-57",
        add_default_virtual_points=False,
        with_mesh=False,
        quiet=True,
    )

    assert output_path == tmp_path / "motive_57.bioMod"
    assert written == [(output_path, False)]
    assert calls[0]["calibration_folder"] == tmp_path
    assert calls[0]["preset"] == C3dModelPreset.MOTIVE_57
    assert calls[0]["static_virtual_points"] == ()
    assert calls[0]["marker_name_prefixes_to_strip"] == ()
    assert calls[0]["progress_callback"] is not None
