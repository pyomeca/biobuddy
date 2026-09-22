from __future__ import annotations

import argparse
from pathlib import Path

from .c3d_model_creation import (
    C3dModelPreset,
    c3d_model_preset_from_cli_value,
    create_model_from_c3d_folder,
    default_static_virtual_points_for_c3d_model_preset,
)

DEFAULT_MOTIVE_57_C3D_FOLDER = Path("/Users/mickaelbegon/Downloads/data/Motive")
DEFAULT_EXAMPLES_DATA_FOLDER = Path(__file__).resolve().parents[2] / "examples" / "data"
DEFAULT_P6_MOTIVE_C3D_FOLDER = DEFAULT_EXAMPLES_DATA_FOLDER / "motive_57_p6"
DEFAULT_P6_MOTIVE_MARKER_PREFIXES = ()


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for C3D-driven model creation.
    """
    parser = argparse.ArgumentParser(
        description=("Build a BioBuddy biomechanical model directly from calibration C3D files.")
    )
    parser.add_argument(
        "c3d_folder",
        nargs="?",
        default=str(DEFAULT_MOTIVE_57_C3D_FOLDER),
        help=(
            "Folder containing the static/main and functional C3D files. " f"Default: {DEFAULT_MOTIVE_57_C3D_FOLDER}"
        ),
    )
    parser.add_argument(
        "--preset",
        default=C3dModelPreset.MOTIVE_57.value,
        help=(
            "Template preset to use. Examples: motive_57, motive_57_isb, full_body, "
            "lower_limbs, lower_limbs_anatomical. Default: motive_57."
        ),
    )
    parser.add_argument(
        "--p6-motive",
        action="store_true",
        help=(
            "Shortcut for --preset motive_57 with the bundled lightweight P6 Motive C3D example "
            f"({DEFAULT_P6_MOTIVE_C3D_FOLDER})."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=("Output .bioMod path. Defaults to the generated preset filename inside " "the C3D folder."),
    )
    parser.add_argument(
        "--no-default-virtual-points",
        action="store_true",
        help=(
            "Do not add preset static virtual points automatically. For Motive 57, "
            "the default adds LGJC/RGJC with the Rab 2002 rule."
        ),
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="Write the .bioMod without mesh entries.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print progress messages.",
    )
    return parser


def build_model_from_c3d_cli(
    c3d_folder: str | Path = DEFAULT_MOTIVE_57_C3D_FOLDER,
    *,
    preset: str | C3dModelPreset = C3dModelPreset.MOTIVE_57,
    output: str | Path | None = None,
    add_default_virtual_points: bool = True,
    with_mesh: bool = True,
    marker_name_prefixes_to_strip: tuple[str, ...] = (),
    quiet: bool = False,
) -> Path:
    """
    Build and write a model from a C3D folder using CLI-style options.
    """
    preset_value = c3d_model_preset_from_cli_value(preset)
    folder_path = Path(c3d_folder).expanduser()
    if not folder_path.exists():
        raise FileNotFoundError(f"C3D folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"C3D folder is not a directory: {folder_path}")

    static_virtual_points = (
        default_static_virtual_points_for_c3d_model_preset(preset_value) if add_default_virtual_points else ()
    )

    def progress(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    result = create_model_from_c3d_folder(
        calibration_folder=folder_path,
        preset=preset_value,
        static_virtual_points=static_virtual_points,
        marker_name_prefixes_to_strip=marker_name_prefixes_to_strip,
        progress_callback=progress,
    )
    output_path = Path(output).expanduser() if output is not None else folder_path / result.output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.model.to_biomod(filepath=str(output_path), with_mesh=with_mesh)

    if not quiet:
        print(f"Preset: {result.preset.value}", flush=True)
        print(f"Static/main markers: {len(result.static_data.marker_names)}", flush=True)
        print(
            f"Functional trials: {', '.join(sorted(result.functional_data))}",
            flush=True,
        )
        print(f"Wrote: {output_path}", flush=True)
    return output_path


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    c3d_folder = DEFAULT_P6_MOTIVE_C3D_FOLDER if args.p6_motive else args.c3d_folder
    preset = C3dModelPreset.MOTIVE_57 if args.p6_motive else args.preset
    marker_name_prefixes_to_strip = DEFAULT_P6_MOTIVE_MARKER_PREFIXES if args.p6_motive else ()
    try:
        build_model_from_c3d_cli(
            c3d_folder,
            preset=preset,
            output=args.output,
            add_default_virtual_points=not args.no_default_virtual_points,
            with_mesh=not args.no_mesh,
            marker_name_prefixes_to_strip=marker_name_prefixes_to_strip,
            quiet=args.quiet,
        )
    except Exception as error:
        parser.exit(status=1, message=f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
