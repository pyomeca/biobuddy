from __future__ import annotations

import argparse
from pathlib import Path

from .build_c3d_model import DEFAULT_MOTIVE_57_C3D_FOLDER, DEFAULT_P6_MOTIVE_C3D_FOLDER
from .c3d_model_creation import C3dModelPreset
from .model_editor import launch_model_editor


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for launching the model editor GUI.
    """
    parser = argparse.ArgumentParser(description="Launch the BioBuddy model editor GUI.")
    parser.add_argument(
        "--new-from-c3d",
        action="store_true",
        help="Open the 'New model from C3D' workflow at startup.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help=("Preset to select in the C3D workflow. Examples: motive_57, motive_57_isb, " "full_body, lower_limbs."),
    )
    parser.add_argument(
        "--c3d-folder",
        type=Path,
        default=None,
        help="C3D folder to load in the startup C3D workflow.",
    )
    parser.add_argument(
        "--motive-57",
        action="store_true",
        help=("Shortcut for --new-from-c3d --preset motive_57 --c3d-folder " f"{DEFAULT_MOTIVE_57_C3D_FOLDER}."),
    )
    parser.add_argument(
        "--motive-57-isb",
        action="store_true",
        help=("Shortcut for --new-from-c3d --preset motive_57_isb --c3d-folder " f"{DEFAULT_MOTIVE_57_C3D_FOLDER}."),
    )
    parser.add_argument(
        "--p6-motive",
        action="store_true",
        help=("Shortcut for --new-from-c3d --preset motive_57 --c3d-folder " f"{DEFAULT_P6_MOTIVE_C3D_FOLDER}."),
    )
    return parser


def resolve_launch_options(args) -> dict[str, object]:
    """
    Resolve parsed CLI arguments into ``launch_model_editor`` keyword arguments.
    """
    preset = args.preset
    c3d_folder = args.c3d_folder
    open_c3d_dialog = args.new_from_c3d
    if args.p6_motive:
        preset = C3dModelPreset.MOTIVE_57
        c3d_folder = DEFAULT_P6_MOTIVE_C3D_FOLDER
        open_c3d_dialog = True
    elif args.motive_57_isb:
        preset = C3dModelPreset.MOTIVE_57_ISB
        c3d_folder = DEFAULT_MOTIVE_57_C3D_FOLDER
        open_c3d_dialog = True
    elif args.motive_57:
        preset = C3dModelPreset.MOTIVE_57
        c3d_folder = DEFAULT_MOTIVE_57_C3D_FOLDER
        open_c3d_dialog = True
    elif preset is not None or c3d_folder is not None:
        open_c3d_dialog = True
    return {
        "c3d_preset": preset,
        "c3d_folder": c3d_folder,
        "open_c3d_dialog": open_c3d_dialog,
    }


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    launch_model_editor(**resolve_launch_options(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
