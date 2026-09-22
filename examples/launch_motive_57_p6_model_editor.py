"""
Launch the model editor directly on the bundled Motive (57) P6 C3D example.
"""

from pathlib import Path

from biobuddy.gui import C3dModelPreset, launch_model_editor

EXAMPLE_C3D_FOLDER = Path(__file__).parent / "data" / "motive_57_p6"


def main() -> None:
    launch_model_editor(
        c3d_preset=C3dModelPreset.MOTIVE_57,
        c3d_folder=EXAMPLE_C3D_FOLDER,
        open_c3d_dialog=True,
    )


if __name__ == "__main__":
    main()
