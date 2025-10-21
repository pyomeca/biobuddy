"""
TODO: add MuJoCo muscle implementation
see https://github.com/MyoHub/myo_sim/blob/main/elbow/assets/myoelbow_2dof6muscles_body.xml
"""

import os
import numpy as np

from biobuddy import BiomechanicalModelReal
from test_utils import compare_models


def test_translation_urdf_to_biomod():
    """Test comprehensive URDF to BioMod translation"""

    np.random.seed(42)

    # Paths
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_filepaths = [
        parent_path + f"/examples/models/flexiv_Rizon10s_kinematics.urdf",
        parent_path + f"/examples/models/kuka_lwr.urdf",
    ]

    for urdf_filepath in urdf_filepaths:
        biomod_filepath = urdf_filepath.replace(".urdf", ".bioMod")

        # Delete the biomod file so we are sure to create it
        if os.path.exists(biomod_filepath):
            os.remove(biomod_filepath)

        print(f" ******** Converting {urdf_filepath} ******** ")

        # Convert URDF to biomod and check that they are the same
        model_from_urdf = BiomechanicalModelReal().from_urdf(
            filepath=urdf_filepath,
        )
        model_from_biomod = BiomechanicalModelReal().from_biomod(
            filepath=biomod_filepath.replace(".bioMod", "_reference.bioMod"),
        )
        compare_models(model_from_urdf, model_from_biomod, decimal=5)

        # Test that the model created can be exported into .biomod
        model_from_urdf.to_biomod(biomod_filepath, with_mesh=True)

        # Test that the .biomod can be reconverted into .urdf
        model_from_biomod_2 = BiomechanicalModelReal().from_biomod(
            filepath=biomod_filepath,
        )
        model_from_biomod_2.to_urdf(filepath=urdf_filepath.replace(".urdf", "_translated.urdf"), with_mesh=True)
        model_from_urdf_2 = BiomechanicalModelReal().from_urdf(
            filepath=urdf_filepath.replace(".urdf", "_translated.urdf")
        )
        compare_models(model_from_urdf, model_from_urdf_2, decimal=5)

        if os.path.exists(biomod_filepath):
            os.remove(biomod_filepath)

        if os.path.exists(urdf_filepath.replace(".urdf", "_translated.urdf")):
            os.remove(urdf_filepath.replace(".urdf", "_translated.urdf"))
