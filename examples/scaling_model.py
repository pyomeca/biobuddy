"""
This example shows how to scale a model based on a generic model and a static trial.
"""

import logging
from pathlib import Path
import numpy as np
import biorbd

from biobuddy import BiomechanicalModelReal, MuscleType, MuscleStateType, MeshParser, MeshFormat, ScaleTool


def main(visualization):

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # logging.FileHandler("app.log"),  # Log to a file
            logging.StreamHandler()  # Log to the console
        ],
    )

    # Paths
    current_path_file = Path(__file__).parent
    osim_filepath = f"{current_path_file}/models/wholebody.osim"
    biomod_filepath = f"{current_path_file}/models/wholebody.bioMod"
    geometry_path = f"{current_path_file}/../external/opensim-models/Geometry"
    geometry_cleaned_path = f"{current_path_file}/models/Geometry_cleaned"
    xml_filepath = f"{current_path_file}/models/wholebody.xml"
    scaled_biomod_filepath = f"{current_path_file}/models/wholebody_scaled.bioMod"
    static_filepath = f"{current_path_file}/data/static.c3d"

    # # Convert the vtp files
    # mesh = MeshParser(geometry_folder=geometry_path)
    # mesh.process_meshes(fail_on_error=False)
    # mesh.write(geometry_cleaned_path, format=MeshFormat.VTP)

    # Read an .osim file
    model = BiomechanicalModelReal().from_osim(
        filepath=osim_filepath,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir="Geometry_cleaned",
    )
    # Fix the via points before translating to biomod as there are some conditional and moving via points
    model.fix_via_points(q=np.zeros((model.nb_q,)))

    # Translate into biomod
    model.to_biomod(biomod_filepath)

    # Setup the scaling configuration (which markers to use)
    scale_tool = ScaleTool(original_model=model).from_xml(filepath=xml_filepath)

    # # Note that it is also possible to initialize the ScaleTool from scratch like this
    # from biobuddy import SegmentScaling, BodyWiseScaling, SegmentWiseScaling, AxisWiseScaling, Translations
    #
    # # BodyWise scaling
    # scale_tool.add_scaling_segment(
    #     SegmentScaling(
    #         name="pelvis",
    #         scaling_type=BodyWiseScaling(
    #             subject_height=1.70,
    #         ),
    #     )
    # )
    # # SegmentWise scaling
    # scale_tool.add_scaling_segment(
    #     SegmentScaling(
    #         name="pelvis",
    #         scaling_type=SegmentWiseScaling(
    #             axis=Translations.XYZ,
    #             marker_pairs=[
    #                 ["RASIS", "LASIS"],
    #                 ["RPSIS", "LPSIS"],
    #                 ["RASIS", "RPSIS"],
    #                 ["LASIS", "LPSIS"],
    #             ],
    #         ),
    #     )
    # )
    # # AxisWise scaling
    # scale_tool.add_scaling_segment(
    #     SegmentScaling(
    #         name="pelvis",
    #         scaling_type=AxisWiseScaling(
    #             marker_pairs={
    #                 Translations.X: [["RASIS", "LASIS"], ["RPSIS", "LPSIS"]],
    #                 Translations.Y: [["RASIS", "RPSIS"], ["LASIS", "LPSIS"]],
    #             },
    #         ),
    #     )
    # )

    # Scale the model
    scaled_model = scale_tool.scale(
        filepath=static_filepath,
        first_frame=100,
        last_frame=200,
        mass=80,
        q_regularization_weight=0.01,
        make_static_pose_the_models_zero=True,
        visualize_optimal_static_pose=False,
    )

    # Write the scaled model to a .bioMod file
    scaled_model.to_biomod(scaled_biomod_filepath, with_mesh=True)

    # Test that the model created is valid
    biorbd.Model(scaled_biomod_filepath)

    if visualization:
        import pyorerun

        # Compare the result visually
        t = np.linspace(0, 1, 10)
        viz = pyorerun.PhaseRerun(t)
        q = np.zeros((42, 10))

        # Biorbd model translated from .osim
        viz_biomod_model = pyorerun.BiorbdModel(biomod_filepath)
        viz_biomod_model.options.transparent_mesh = False
        viz_biomod_model.options.show_gravity = True
        viz_biomod_model.options.show_marker_labels = False
        viz_biomod_model.options.show_center_of_mass_labels = False
        viz.add_animated_model(viz_biomod_model, q)

        # Add the experimental markers from the static trial
        fake_exp_markers = np.repeat(scale_tool.mean_experimental_markers[:, :, np.newaxis], 10, axis=2)
        pyomarkers = pyorerun.PyoMarkers(data=fake_exp_markers, channels=scaled_model.marker_names, show_labels=False)

        # Model output
        viz_scaled_model = pyorerun.BiorbdModel(scaled_biomod_filepath)
        viz_scaled_model.options.transparent_mesh = False
        viz_scaled_model.options.show_gravity = True
        viz_scaled_model.options.show_marker_labels = False
        viz_scaled_model.options.show_center_of_mass_labels = False
        viz.add_animated_model(viz_scaled_model, q, tracked_markers=pyomarkers)

        # Animate
        viz.rerun_by_frame("Model output")


if __name__ == "__main__":
    try:
        import pyorerun

        visualization = True
    except:
        visualization = False

    main(visualization)
