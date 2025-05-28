"""
This example shows how to read and write models.
"""

import logging
from pathlib import Path

from biobuddy import (
    MuscleType,
    MuscleStateType,
    MeshParser,
    MeshFormat,
    BiomechanicalModelReal,
)

_logger = logging.getLogger(__name__)


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # logging.FileHandler("app.log"),  # Log to a file
            logging.StreamHandler()  # Log to the console
        ],
    )
    visualization_flag = True

    # Paths
    current_path_file = Path(__file__).parent
    biomod_filepath = f"{current_path_file}/models/wholebody.bioMod"
    osim_filepath = f"{current_path_file}/models/wholebody.osim"
    geometry_path = f"{current_path_file}/../external/opensim-models/Geometry"
    geometry_cleaned_path = f"{current_path_file}/models/Geometry_cleaned"

    # Convert the vtp files
    mesh = MeshParser(geometry_folder=geometry_path)
    mesh.process_meshes(fail_on_error=False)
    mesh.write(geometry_cleaned_path, format=MeshFormat.VTP)

    # --- Reading an .osim model and translating it to a .bioMod model --- #
    # Read an .osim file
    model = BiomechanicalModelReal.from_osim(
        filepath=osim_filepath,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir="Geometry_cleaned",
    )

    # And convert it to a .bioMod file
    model.to_biomod(biomod_filepath, with_mesh=visualization_flag)

    # Test that the model created is valid
    try:
        import biorbd

        biorbd.Model(biomod_filepath)
    except ImportError:
        _logger.warning("You must install biorbd to load the model with biorbd")

    if visualization_flag:
        # Compare the result visually
        import numpy as np

        try:
            import pyorerun
        except ImportError:

            raise ImportError("You must install pyorerun to visualize the model (visualization_flag=True)")

        # Visualization
        t = np.linspace(0, 1, 10)
        viz = pyorerun.PhaseRerun(t)

        # Model output
        model = pyorerun.BiorbdModel(biomod_filepath)
        model.options.transparent_mesh = False
        model.options.show_gravity = True
        model.options.show_marker_labels = False
        model.options.show_center_of_mass_labels = False
        q = np.zeros((model.nb_q, 10))
        viz.add_animated_model(model, q)

        # Model reference
        reference_model = pyorerun.BiorbdModel(biomod_filepath.replace(".bioMod", "_reference.bioMod"))
        reference_model.options.transparent_mesh = False
        reference_model.options.show_gravity = True
        reference_model.options.show_marker_labels = False
        reference_model.options.show_center_of_mass_labels = False
        q_ref = np.zeros((reference_model.nb_q, 10))
        q_ref[0, :] = 0.5
        viz.add_animated_model(reference_model, q_ref)

        # Animate
        viz.rerun_by_frame("Model output")

    # --- Reading an .bioMod model and translating it to a .osim model --- #
    # Read a .bioMod file
    model = BiomechanicalModelReal().from_biomod(filepath=biomod_filepath)

    # TODO
    # And convert it to an .osim file
    # model.to_osim(osim_filepath, with_mesh=visualization_flag)
    #
    # # Test that the model created is valid
    # try:
    #     import opensim as osim
    #
    #     osim.Model(osim_filepath)
    # except ImportError:
    #     _logger.warning("You must install opensim to load the model with opensim")


if __name__ == "__main__":
    main()
