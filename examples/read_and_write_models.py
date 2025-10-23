"""
This example shows how to read and write models.
"""

import logging
from pathlib import Path
import numpy as np

from biobuddy import (
    MuscleType,
    MuscleStateType,
    MeshParser,
    MeshFormat,
    BiomechanicalModelReal,
    ViewAs,
)

_logger = logging.getLogger(__name__)


def osim_biomod_convertion():
    visualization_flag = True

    # Paths
    current_path_file = Path(__file__).parent
    biomod_filepath = f"{current_path_file}/models/wholebody.bioMod"
    osim_filepath = f"{current_path_file}/models/wholebody.osim"
    geometry_path = f"{current_path_file}/../external/opensim-models/Geometry"
    geometry_cleaned_path = f"{current_path_file}/models/Geometry_cleaned"

    # # Convert the vtp files
    # mesh = MeshParser(geometry_folder=geometry_path)
    # mesh.process_meshes(fail_on_error=False)
    # mesh.write(geometry_cleaned_path, format=MeshFormat.VTP)

    # --- Reading an .osim model and translating it to a .bioMod model --- #
    # Read an .osim file
    model = BiomechanicalModelReal().from_osim(
        filepath=osim_filepath,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir="Geometry_cleaned",
    )
    # Fix the via points before translating to biomod as there are some conditional and moving via points
    model.fix_via_points(q=np.zeros((model.nb_q,)))

    # And convert it to a .bioMod file
    model.to_biomod(biomod_filepath, with_mesh=visualization_flag)

    # Test that the model created is valid
    try:
        import biorbd
    except:
        raise ImportError("You must install biorbd to load the model with biorbd")
    biorbd.Model(biomod_filepath)

    # --- Reading the .osim model and translate it to a .bioMod model --- #
    # Read a .biomod file
    model = BiomechanicalModelReal().from_biomod(filepath=biomod_filepath)

    # And convert it to an .osim file
    model.to_osim(osim_filepath.replace(".osim", "_from_biomod.osim"), with_mesh=visualization_flag)

    # Test that the model created is valid
    try:
        import opensim as osim
    except:
        raise ImportError("You must install opensim to load the model with opensim")
    osim.Model(osim_filepath)

    if visualization_flag:
        # Compare the result visually
        try:
            import pyorerun
        except:
            raise ImportError("You must install pyorerun to visualize the model (visualization_flag=True)")

        # Visualization
        t = np.linspace(0, 1, 10)
        viz = pyorerun.PhaseRerun(t)

        # Biomod model output
        model = pyorerun.BiorbdModel(biomod_filepath)
        model.options.transparent_mesh = False
        model.options.show_gravity = True
        model.options.show_marker_labels = False
        model.options.show_center_of_mass_labels = False
        q = np.zeros((model.nb_q, 10))
        viz.add_animated_model(model, q)

        # Biomod model reference
        reference_model = pyorerun.BiorbdModel(biomod_filepath.replace(".bioMod", "_reference.bioMod"))
        reference_model.options.transparent_mesh = False
        reference_model.options.show_gravity = True
        reference_model.options.show_marker_labels = False
        reference_model.options.show_center_of_mass_labels = False
        q_ref = np.zeros((reference_model.nb_q, 10))
        q_ref[0, :] = 0.5
        viz.add_animated_model(reference_model, q_ref)

        # TODO: see with aceglia why I get the error:
        # "RuntimeError: std::exception in 'OpenSim::Model::Model(std::string const &)': Joint::getMotionType() given an invalid CoordinateIndex
	    # Thrown at Joint.cpp:224 in getMotionType()." although `osim.Model(osim_filepath)` works fine
        # Osim model reference
        # display_options = pyorerun.DisplayModelOptions()
        # display_options.mesh_path = f"{current_path_file}/models/Geometry_cleaned"
        # model = pyorerun.OsimModel(osim_filepath, options=display_options)
        model_translated = BiomechanicalModelReal().from_osim(
            filepath=osim_filepath.replace(".osim", "_from_biomod.osim"),
            muscle_type=MuscleType.HILL_DE_GROOTE,
            muscle_state_type=MuscleStateType.DEGROOTE,
            mesh_dir="Geometry_cleaned",
            skip_virtual=True,
        )
        model_translated.to_biomod(
            biomod_filepath.replace(".bioMod", "_from_osim_translated.bioMod"),
            with_mesh=True,
        )
        model = pyorerun.BiorbdModel(biomod_filepath.replace(".bioMod", "_from_osim_translated.bioMod"))
        model.options.transparent_mesh = False
        model.options.show_gravity = True
        model.options.show_marker_labels = False
        model.options.show_center_of_mass_labels = False
        q = np.zeros((model.nb_q, 10))
        viz.add_animated_model(model, q)

        # # Osim model output
        # reference_model = pyorerun.OsimModel(
        #     osim_filepath.replace(".osim", "_from_biomod.osim"),
        #     options=display_options,
        # )
        # reference_model.options.transparent_mesh = False
        # reference_model.options.show_gravity = True
        # reference_model.options.show_marker_labels = False
        # reference_model.options.show_center_of_mass_labels = False
        # q_ref = np.zeros((reference_model.nb_q, 10))
        # q_ref[0, :] = 0.5
        # viz.add_animated_model(reference_model, q_ref)

        # Animate
        viz.rerun_by_frame("Translated models")


def urdf_biomod_convertion():

    visualization_flag = True

    # Paths
    current_path_file = Path(__file__).parent
    # biomod_filepath = f"{current_path_file}/models/kuka_lwr.bioMod"
    # urdf_filepath = f"{current_path_file}/models/kuka_lwr.urdf"
    biomod_filepath = f"{current_path_file}/models/flexiv_Rizon10s_kinematics.bioMod"
    urdf_filepath = f"{current_path_file}/models/flexiv_Rizon10s_kinematics.urdf"

    # --- Reading an .urdf model and translating it to a .bioMod model --- #
    # Read an .urdf file
    model = BiomechanicalModelReal().from_urdf(
        filepath=urdf_filepath,
    )

    # And convert it to a .bioMod file
    model.to_biomod(biomod_filepath, with_mesh=visualization_flag)

    # Test that the model created is valid
    try:
        import biorbd
    except ImportError:
        _logger.warning("You must install biorbd to load the model with biorbd")
    biorbd.Model(biomod_filepath)

    if visualization_flag:
        model.animate(view_as=ViewAs.BIORBD, model_path=biomod_filepath)

    # --- Reading an .bioMod model and translating it to an .urdf model --- #
    # Read a .bioMod file
    model = BiomechanicalModelReal().from_biomod(filepath=biomod_filepath)

    # And convert it to an .urdf file
    model.to_urdf(urdf_filepath.replace(".urdf", "_translated.urdf"), with_mesh=visualization_flag)


if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # logging.FileHandler("app.log"),  # Log to a file
            logging.StreamHandler()  # Log to the console
        ],
    )

    osim_biomod_convertion()
    urdf_biomod_convertion()
