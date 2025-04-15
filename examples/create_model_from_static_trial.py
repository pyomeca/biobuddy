"""
Example taken from biorbd.model_creation
# TODO: This example should be replace with an actual biomechanics example.
"""

import os

import numpy as np
import biorbd
from biobuddy import (
    BiomechanicalModelReal,
    MarkerReal,
    MeshReal,
    SegmentReal,
    SegmentCoordinateSystemReal,
    Translations,
    Rotations,
)


def model_creation_from_static_trial(remove_temporary: bool = True):
    """
    We define a new model by feeding in the actual dimension and position of the model
    Please note that a bunch of useless markers are defined, this is for the other model creation below which needs them
    to define the SegmentCoordinateSystem matrices
    """

    kinematic_model_file_path = "temporary.bioMod"

    # Create a model holder
    bio_model = BiomechanicalModelReal()

    # The trunk segment
    bio_model.segments.append(
        SegmentReal(
            name="TRUNK",
            translations=Translations.YZ,
            rotations=Rotations.X,
            mesh=MeshReal(((0, 0, 0), (0, 0, 0.53))),
        )
    )
    bio_model.segments["TRUNK"].add_marker(MarkerReal(name="PELVIS", parent_name="TRUNK"))

    # The head segment
    bio_model.segments.append(
        SegmentReal(
            name="HEAD",
            parent_name="TRUNK",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                (0, 0, 0), "xyz", (0, 0, 0.53), is_scs_local=True
            ),
            mesh=MeshReal(((0, 0, 0), (0, 0, 0.24))),
        )
    )
    bio_model.segments["HEAD"].add_marker(MarkerReal(name="BOTTOM_HEAD", parent_name="HEAD", position=(0, 0, 0)))
    bio_model.segments["HEAD"].add_marker(MarkerReal(name="TOP_HEAD", parent_name="HEAD", position=(0, 0, 0.24)))
    bio_model.segments["HEAD"].add_marker(MarkerReal(name="HEAD_Z", parent_name="HEAD", position=(0, 0, 0.24)))
    bio_model.segments["HEAD"].add_marker(MarkerReal(name="HEAD_XZ", parent_name="HEAD", position=(0.24, 0, 0.24)))

    # The arm segment
    bio_model.segments.append(
        SegmentReal(
            name="UPPER_ARM",
            parent_name="TRUNK",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                (0, 0, 0), "xyz", (0, 0, 0.53), is_scs_local=True
            ),
            rotations=Rotations.X,
            mesh=MeshReal(((0, 0, 0), (0, 0, -0.28))),
        )
    )
    bio_model.segments["UPPER_ARM"].add_marker(MarkerReal(name="SHOULDER", parent_name="UPPER_ARM", position=(0, 0, 0)))
    bio_model.segments["UPPER_ARM"].add_marker(
        MarkerReal(name="SHOULDER_X", parent_name="UPPER_ARM", position=(1, 0, 0))
    )
    bio_model.segments["UPPER_ARM"].add_marker(
        MarkerReal(name="SHOULDER_XY", parent_name="UPPER_ARM", position=(1, 1, 0))
    )

    bio_model.segments.append(
        SegmentReal(
            name="LOWER_ARM",
            parent_name="UPPER_ARM",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                (0, 0, 0), "xyz", (0, 0, -0.28), is_scs_local=True
            ),
            mesh=MeshReal(((0, 0, 0), (0, 0, -0.27))),
        )
    )
    bio_model.segments["LOWER_ARM"].add_marker(MarkerReal(name="ELBOW", parent_name="LOWER_ARM", position=(0, 0, 0)))
    bio_model.segments["LOWER_ARM"].add_marker(MarkerReal(name="ELBOW_Y", parent_name="LOWER_ARM", position=(0, 1, 0)))
    bio_model.segments["LOWER_ARM"].add_marker(MarkerReal(name="ELBOW_XY", parent_name="LOWER_ARM", position=(1, 1, 0)))

    bio_model.segments.append(
        SegmentReal(
            name="HAND",
            parent_name="LOWER_ARM",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                (0, 0, 0), "xyz", (0, 0, -0.27), is_scs_local=True
            ),
            mesh=MeshReal(((0, 0, 0), (0, 0, -0.19))),
        )
    )
    bio_model.segments["HAND"].add_marker(MarkerReal(name="WRIST", parent_name="HAND", position=(0, 0, 0)))
    bio_model.segments["HAND"].add_marker(MarkerReal(name="FINGER", parent_name="HAND", position=(0, 0, -0.19)))
    bio_model.segments["HAND"].add_marker(MarkerReal(name="HAND_Y", parent_name="HAND", position=(0, 1, 0)))
    bio_model.segments["HAND"].add_marker(MarkerReal(name="HAND_YZ", parent_name="HAND", position=(0, 1, 1)))

    # The thigh segment
    bio_model.segments.append(
        SegmentReal(
            name="THIGH",
            parent_name="TRUNK",
            rotations=Rotations.X,
            mesh=MeshReal(((0, 0, 0), (0, 0, -0.42))),
        )
    )
    bio_model.segments["THIGH"].add_marker(MarkerReal(name="THIGH_ORIGIN", parent_name="THIGH", position=(0, 0, 0)))
    bio_model.segments["THIGH"].add_marker(MarkerReal(name="THIGH_X", parent_name="THIGH", position=(1, 0, 0)))
    bio_model.segments["THIGH"].add_marker(MarkerReal(name="THIGH_Y", parent_name="THIGH", position=(0, 1, 0)))

    # The shank segment
    bio_model.segments.append(
        SegmentReal(
            name="SHANK",
            parent_name="THIGH",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                (0, 0, 0), "xyz", (0, 0, -0.42), is_scs_local=True
            ),
            rotations=Rotations.X,
            mesh=MeshReal(((0, 0, 0), (0, 0, -0.43))),
        )
    )
    bio_model.segments["SHANK"].add_marker(MarkerReal(name="KNEE", parent_name="SHANK", position=(0, 0, 0)))
    bio_model.segments["SHANK"].add_marker(MarkerReal(name="KNEE_Z", parent_name="SHANK", position=(0, 0, 1)))
    bio_model.segments["SHANK"].add_marker(MarkerReal(name="KNEE_XZ", parent_name="SHANK", position=(1, 0, 1)))

    # The foot segment
    bio_model.segments.append(
        SegmentReal(
            name="FOOT",
            parent_name="SHANK",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                (-np.pi / 2, 0, 0), "xyz", (0, 0, -0.43), is_scs_local=True
            ),
            rotations=Rotations.X,
            mesh=MeshReal(((0, 0, 0), (0, 0, 0.25))),
        )
    )
    bio_model.segments["FOOT"].add_marker(MarkerReal(name="ANKLE", parent_name="FOOT", position=(0, 0, 0)))
    bio_model.segments["FOOT"].add_marker(MarkerReal(name="TOE", parent_name="FOOT", position=(0, 0, 0.25)))
    bio_model.segments["FOOT"].add_marker(MarkerReal(name="ANKLE_Z", parent_name="FOOT", position=(0, 0, 1)))
    bio_model.segments["FOOT"].add_marker(MarkerReal(name="ANKLE_YZ", parent_name="FOOT", position=(0, 1, 1)))

    # Put the model together, print it and print it to a bioMod file
    bio_model.to_biomod(kinematic_model_file_path)

    model = biorbd.Model(kinematic_model_file_path)
    assert model.nbQ() == 7
    assert model.nbSegment() == 8
    assert model.nbMarkers() == 25
    np.testing.assert_almost_equal(model.markers(np.zeros((model.nbQ(),)))[-3].to_array(), [0, 0.25, -0.85], decimal=4)

    if remove_temporary:
        os.remove(kinematic_model_file_path)


def main():
    # Create the model from user defined dimensions
    model_creation_from_static_trial(remove_temporary=False)


if __name__ == "__main__":
    main()
