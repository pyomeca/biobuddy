"""
Example taken from biorbd.model_creation
# TODO: This example should be replace with an actual biomechanics example.
"""

import os

import numpy as np
import biorbd
from biobuddy import (
    Axis,
    BiomechanicalModel,
    C3dData,
    Marker,
    Mesh,
    Segment,
    SegmentCoordinateSystem,
    Translations,
    Rotations,
    DeLevaTable,
)
import ezc3d

def model_creation_from_measured_data(remove_temporary: bool = True):
    """
    We are using the previous model to define a new model based on the position of the markers. This is solely so we
    have realistic data to use. Typically, the 'write_markers' function would be some actual data collection
    """

    def write_markers_to_c3d(save_path: str, model: biorbd.Model):
        """
        Write data to a c3d file
        """
        q = np.zeros(model.nbQ())
        marker_names = tuple(name.to_string() for name in model.markerNames())
        marker_positions = np.array(tuple(m.to_array() for m in model.markers(q))).T[:, :, np.newaxis]
        c3d = ezc3d.c3d()

        # Fill it with random data
        c3d["parameters"]["POINT"]["RATE"]["value"] = [100]
        c3d["parameters"]["POINT"]["LABELS"]["value"] = marker_names
        c3d["data"]["points"] = marker_positions

        # Write the data
        c3d.write(save_path)

    kinematic_model_file_path = "temporary.bioMod"
    c3d_file_path = "temporary.c3d"

    # Prepare a fake model and a fake static from the previous test
    model = biorbd.Model(kinematic_model_file_path)
    write_markers_to_c3d(c3d_file_path, model)
    os.remove(kinematic_model_file_path)

    # Fill the kinematic chain model
    model = BiomechanicalModel()
    de_leva = DeLevaTable(total_mass=100, sex="female")

    model.segments["TRUNK"] = Segment(
        name="TRUNK",
        translations=Translations.YZ,
        rotations=Rotations.X,
        inertia_parameters=de_leva["TRUNK"],
    )
    model.segments["TRUNK"].add_marker(Marker("PELVIS"))

    model.segments["HEAD"] = Segment(
        name="HEAD",
        parent_name="TRUNK",
        segment_coordinate_system=SegmentCoordinateSystem(
            "BOTTOM_HEAD",
            first_axis=Axis(name=Axis.Name.Z, start="BOTTOM_HEAD", end="HEAD_Z"),
            second_axis=Axis(name=Axis.Name.X, start="BOTTOM_HEAD", end="HEAD_XZ"),
            axis_to_keep=Axis.Name.Z,
        ),
        mesh=Mesh(("BOTTOM_HEAD", "TOP_HEAD", "HEAD_Z", "HEAD_XZ", "BOTTOM_HEAD")),
        inertia_parameters=de_leva["HEAD"],
    )
    model.segments["HEAD"].add_marker(Marker("BOTTOM_HEAD"))
    model.segments["HEAD"].add_marker(Marker("TOP_HEAD"))
    model.segments["HEAD"].add_marker(Marker("HEAD_Z"))
    model.segments["HEAD"].add_marker(Marker("HEAD_XZ"))

    model.segments["UPPER_ARM"] = Segment(
        name="UPPER_ARM",
        parent_name="TRUNK",
        rotations=Rotations.X,
        segment_coordinate_system=SegmentCoordinateSystem(
            origin="SHOULDER",
            first_axis=Axis(name=Axis.Name.X, start="SHOULDER", end="SHOULDER_X"),
            second_axis=Axis(name=Axis.Name.Y, start="SHOULDER", end="SHOULDER_XY"),
            axis_to_keep=Axis.Name.X,
        ),
        inertia_parameters=de_leva["UPPER_ARM"],
    )
    model.segments["UPPER_ARM"].add_marker(Marker("SHOULDER"))
    model.segments["UPPER_ARM"].add_marker(Marker("SHOULDER_X"))
    model.segments["UPPER_ARM"].add_marker(Marker("SHOULDER_XY"))

    model.segments["LOWER_ARM"] = Segment(
        name="LOWER_ARM",
        parent_name="UPPER_ARM",
        segment_coordinate_system=SegmentCoordinateSystem(
            origin="ELBOW",
            first_axis=Axis(name=Axis.Name.Y, start="ELBOW", end="ELBOW_Y"),
            second_axis=Axis(name=Axis.Name.X, start="ELBOW", end="ELBOW_XY"),
            axis_to_keep=Axis.Name.Y,
        ),
        inertia_parameters=de_leva["LOWER_ARM"],
    )
    model.segments["LOWER_ARM"].add_marker(Marker("ELBOW"))
    model.segments["LOWER_ARM"].add_marker(Marker("ELBOW_Y"))
    model.segments["LOWER_ARM"].add_marker(Marker("ELBOW_XY"))

    model.segments["HAND"] = Segment(
        name="HAND",
        parent_name="LOWER_ARM",
        segment_coordinate_system=SegmentCoordinateSystem(
            origin="WRIST",
            first_axis=Axis(name=Axis.Name.Y, start="WRIST", end="HAND_Y"),
            second_axis=Axis(name=Axis.Name.Z, start="WRIST", end="HAND_YZ"),
            axis_to_keep=Axis.Name.Y,
        ),
        inertia_parameters=de_leva["HAND"],
    )
    model.segments["HAND"].add_marker(Marker("WRIST"))
    model.segments["HAND"].add_marker(Marker("FINGER"))
    model.segments["HAND"].add_marker(Marker("HAND_Y"))
    model.segments["HAND"].add_marker(Marker("HAND_YZ"))

    model.segments["THIGH"] = Segment(
        name="THIGH",
        parent_name="TRUNK",
        rotations=Rotations.X,
        segment_coordinate_system=SegmentCoordinateSystem(
            origin="THIGH_ORIGIN",
            first_axis=Axis(name=Axis.Name.X, start="THIGH_ORIGIN", end="THIGH_X"),
            second_axis=Axis(name=Axis.Name.Y, start="THIGH_ORIGIN", end="THIGH_Y"),
            axis_to_keep=Axis.Name.X,
        ),
        inertia_parameters=de_leva["THIGH"],
    )
    model.segments["THIGH"].add_marker(Marker("THIGH_ORIGIN"))
    model.segments["THIGH"].add_marker(Marker("THIGH_X"))
    model.segments["THIGH"].add_marker(Marker("THIGH_Y"))

    model.segments["SHANK"] = Segment(
        name="SHANK",
        parent_name="THIGH",
        rotations=Rotations.X,
        segment_coordinate_system=SegmentCoordinateSystem(
            origin="KNEE",
            first_axis=Axis(name=Axis.Name.Z, start="KNEE", end="KNEE_Z"),
            second_axis=Axis(name=Axis.Name.X, start="KNEE", end="KNEE_XZ"),
            axis_to_keep=Axis.Name.Z,
        ),
        inertia_parameters=de_leva["SHANK"],
    )
    model.segments["SHANK"].add_marker(Marker("KNEE"))
    model.segments["SHANK"].add_marker(Marker("KNEE_Z"))
    model.segments["SHANK"].add_marker(Marker("KNEE_XZ"))

    model.segments["FOOT"] = Segment(
        name="FOOT",
        parent_name="SHANK",
        rotations=Rotations.X,
        segment_coordinate_system=SegmentCoordinateSystem(
            origin="ANKLE",
            first_axis=Axis(name=Axis.Name.Z, start="ANKLE", end="ANKLE_Z"),
            second_axis=Axis(name=Axis.Name.Y, start="ANKLE", end="ANKLE_YZ"),
            axis_to_keep=Axis.Name.Z,
        ),
        inertia_parameters=de_leva["FOOT"],
    )
    model.segments["FOOT"].add_marker(Marker("ANKLE"))
    model.segments["FOOT"].add_marker(Marker("TOE"))
    model.segments["FOOT"].add_marker(Marker("ANKLE_Z"))
    model.segments["FOOT"].add_marker(Marker("ANKLE_YZ"))

    # Put the model together, print it and print it to a bioMod file
    model.write(kinematic_model_file_path, C3dData(c3d_file_path))

    model = biorbd.Model(kinematic_model_file_path)
    assert model.nbQ() == 7
    assert model.nbSegment() == 8
    assert model.nbMarkers() == 25
    np.testing.assert_almost_equal(model.markers(np.zeros((model.nbQ(),)))[-3].to_array(), [0, 0.25, -0.85], decimal=4)

    if remove_temporary:
        os.remove(kinematic_model_file_path)
        os.remove(c3d_file_path)


if __name__ == "__main__":
    # Create the model from a data file and markers as template
    model_creation_from_measured_data()
