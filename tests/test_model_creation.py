import os
import pytest
from pathlib import Path

import numpy as np
import numpy.testing as npt
from biorbd import Model

from biobuddy import (
    BiomechanicalModelReal,
    SegmentReal,
    Translations,
    Rotations,
    MeshReal,
    MarkerReal,
    SegmentCoordinateSystemReal,
    DeLevaTable,
    BiomechanicalModel,
    Segment,
    Marker,
    SegmentCoordinateSystem,
    Axis,
    Mesh,
    MuscleType,
    MuscleStateType,
    Sex,
    SegmentName,
    C3dData,
)
from test_utils import destroy_model


def test_model_creation_from_static(remove_temporary: bool = True):
    """
    Produces a model from real data
    """

    kinematic_model_filepath = "temporary.bioMod"

    # Create a model holder
    bio_model = BiomechanicalModelReal()

    # The trunk segment
    bio_model.add_segment(
        SegmentReal(
            name="TRUNK",
            parent_name="base",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                np.array([0, 0, 0]),
                "xyz",
                np.array([0, 0, 0]),
            ),
            translations=Translations.YZ,
            rotations=Rotations.X,
            mesh=MeshReal([np.array([0, 0, 0]), np.array([0, 0, 0.53])]),
        )
    )
    bio_model.segments["TRUNK"].add_marker(MarkerReal(name="PELVIS", parent_name="TRUNK", position=np.array([0, 0, 0])))

    # The head segment
    bio_model.add_segment(
        SegmentReal(
            name="HEAD",
            parent_name="TRUNK",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                np.array([0, 0, 0]), "xyz", np.array([0, 0, 0.53]), is_scs_local=True
            ),
            mesh=MeshReal([np.array([0, 0, 0]), np.array([0, 0, 0.24])]),
        )
    )
    bio_model.segments["HEAD"].add_marker(
        MarkerReal(name="BOTTOM_HEAD", parent_name="HEAD", position=np.array([0, 0, 0]))
    )
    bio_model.segments["HEAD"].add_marker(
        MarkerReal(name="TOP_HEAD", parent_name="HEAD", position=np.array([0, 0, 0.24]))
    )
    bio_model.segments["HEAD"].add_marker(
        MarkerReal(name="HEAD_Z", parent_name="HEAD", position=np.array([0, 0, 0.24]))
    )
    bio_model.segments["HEAD"].add_marker(
        MarkerReal(name="HEAD_XZ", parent_name="HEAD", position=np.array([0.24, 0, 0.24]))
    )

    # The arm segment
    bio_model.add_segment(
        SegmentReal(
            name="UPPER_ARM",
            parent_name="TRUNK",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                np.array([0, 0, 0]),
                "xyz",
                np.array([0, 0, 0.53]),
                is_scs_local=True,
            ),
            rotations=Rotations.X,
            mesh=MeshReal([np.array([0, 0, 0]), np.array([0, 0, -0.28])]),
        )
    )
    bio_model.segments["UPPER_ARM"].add_marker(
        MarkerReal(name="SHOULDER", parent_name="UPPER_ARM", position=np.array([0, 0, 0]))
    )
    bio_model.segments["UPPER_ARM"].add_marker(
        MarkerReal(name="SHOULDER_X", parent_name="UPPER_ARM", position=np.array([1, 0, 0]))
    )
    bio_model.segments["UPPER_ARM"].add_marker(
        MarkerReal(name="SHOULDER_XY", parent_name="UPPER_ARM", position=np.array([1, 1, 0]))
    )

    bio_model.add_segment(
        SegmentReal(
            name="LOWER_ARM",
            parent_name="UPPER_ARM",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                np.array([0, 0, 0]), "xyz", np.array([0, 0, -0.28]), is_scs_local=True
            ),
            mesh=MeshReal([np.array([0, 0, 0]), np.array([0, 0, -0.27])]),
        )
    )
    bio_model.segments["LOWER_ARM"].add_marker(
        MarkerReal(name="ELBOW", parent_name="LOWER_ARM", position=np.array([0, 0, 0]))
    )
    bio_model.segments["LOWER_ARM"].add_marker(
        MarkerReal(name="ELBOW_Y", parent_name="LOWER_ARM", position=np.array([0, 1, 0]))
    )
    bio_model.segments["LOWER_ARM"].add_marker(
        MarkerReal(name="ELBOW_XY", parent_name="LOWER_ARM", position=np.array([1, 1, 0]))
    )

    bio_model.add_segment(
        SegmentReal(
            name="HAND",
            parent_name="LOWER_ARM",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                np.array([0, 0, 0]), "xyz", np.array([0, 0, -0.27]), is_scs_local=True
            ),
            mesh=MeshReal([np.array([0, 0, 0]), np.array([0, 0, -0.19])]),
        )
    )
    bio_model.segments["HAND"].add_marker(MarkerReal(name="WRIST", parent_name="HAND", position=np.array([0, 0, 0])))
    bio_model.segments["HAND"].add_marker(
        MarkerReal(name="FINGER", parent_name="HAND", position=np.array([0, 0, -0.19]))
    )
    bio_model.segments["HAND"].add_marker(MarkerReal(name="HAND_Y", parent_name="HAND", position=np.array([0, 1, 0])))
    bio_model.segments["HAND"].add_marker(MarkerReal(name="HAND_YZ", parent_name="HAND", position=np.array([0, 1, 1])))

    # The thigh segment
    bio_model.add_segment(
        SegmentReal(
            name="THIGH",
            parent_name="TRUNK",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                np.array([0, 0, 0]), "xyz", np.array([0, 0, 0]), is_scs_local=True
            ),
            rotations=Rotations.X,
            mesh=MeshReal([np.array([0, 0, 0]), np.array([0, 0, -0.42])]),
        )
    )
    bio_model.segments["THIGH"].add_marker(
        MarkerReal(name="THIGH_ORIGIN", parent_name="THIGH", position=np.array([0, 0, 0]))
    )
    bio_model.segments["THIGH"].add_marker(
        MarkerReal(name="THIGH_X", parent_name="THIGH", position=np.array([1, 0, 0]))
    )
    bio_model.segments["THIGH"].add_marker(
        MarkerReal(name="THIGH_Y", parent_name="THIGH", position=np.array([0, 1, 0]))
    )

    # The shank segment
    bio_model.add_segment(
        SegmentReal(
            name="SHANK",
            parent_name="THIGH",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                np.array([0, 0, 0]), "xyz", np.array([0, 0, -0.42]), is_scs_local=True
            ),
            rotations=Rotations.X,
            mesh=MeshReal([np.array([0, 0, 0]), np.array([0, 0, -0.43])]),
        )
    )
    bio_model.segments["SHANK"].add_marker(MarkerReal(name="KNEE", parent_name="SHANK", position=np.array([0, 0, 0])))
    bio_model.segments["SHANK"].add_marker(MarkerReal(name="KNEE_Z", parent_name="SHANK", position=np.array([0, 0, 1])))
    bio_model.segments["SHANK"].add_marker(
        MarkerReal(name="KNEE_XZ", parent_name="SHANK", position=np.array([1, 0, 1]))
    )

    # The foot segment
    bio_model.add_segment(
        SegmentReal(
            name="FOOT",
            parent_name="SHANK",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                np.array([-np.pi / 2, 0, 0]), "xyz", np.array([0, 0, -0.43]), is_scs_local=True
            ),
            rotations=Rotations.X,
            mesh=MeshReal([np.array([0, 0, 0]), np.array([0, 0, 0.25])]),
        )
    )
    bio_model.segments["FOOT"].add_marker(MarkerReal(name="ANKLE", parent_name="FOOT", position=np.array([0, 0, 0])))
    bio_model.segments["FOOT"].add_marker(MarkerReal(name="HEEL", parent_name="FOOT", position=np.array([0, 0, -0.01])))
    bio_model.segments["FOOT"].add_marker(MarkerReal(name="TOE", parent_name="FOOT", position=np.array([0, 0, 0.25])))
    bio_model.segments["FOOT"].add_marker(MarkerReal(name="ANKLE_Z", parent_name="FOOT", position=np.array([0, 0, 1])))
    bio_model.segments["FOOT"].add_marker(MarkerReal(name="ANKLE_YZ", parent_name="FOOT", position=np.array([0, 1, 1])))

    # Put the model together, print it and print it to a bioMod file
    bio_model.to_biomod(kinematic_model_filepath, with_mesh=False)

    model = Model(kinematic_model_filepath)
    assert model.nbQ() == 7
    assert bio_model.nb_q == 7
    assert model.nbSegment() == 9
    assert bio_model.nb_segments == 9
    assert model.nbMarkers() == 26
    assert bio_model.nb_markers == 26
    value = model.markers(np.zeros((model.nbQ(),)))[-3].to_array()
    np.testing.assert_almost_equal(value, [0, 0.25, -0.85], decimal=4)

    # Test the attributes of the model
    assert all(
        segment_name
        in [
            "root",
            "TRUNK",
            "HEAD",
            "UPPER_ARM",
            "LOWER_ARM",
            "HAND",
            "THIGH",
            "SHANK",
            "FOOT",
        ]
        for segment_name in bio_model.segment_names
    )
    assert len(bio_model.segment_names) == 9

    assert all(
        marker_name
        in [
            "PELVIS",
            "BOTTOM_HEAD",
            "TOP_HEAD",
            "HEAD_Z",
            "HEAD_XZ",
            "SHOULDER",
            "SHOULDER_X",
            "SHOULDER_XY",
            "ELBOW",
            "ELBOW_Y",
            "ELBOW_XY",
            "WRIST",
            "FINGER",
            "HAND_Y",
            "HAND_YZ",
            "THIGH_ORIGIN",
            "THIGH_X",
            "THIGH_Y",
            "KNEE",
            "KNEE_Z",
            "KNEE_XZ",
            "ANKLE",
            "TOE",
            "HEEL",
            "ANKLE_Z",
            "ANKLE_YZ",
        ]
        for marker_name in bio_model.marker_names
    )
    assert len(bio_model.marker_names) == 26

    destroy_model(bio_model)

    # Test the attributes of the model
    assert bio_model.segment_names == []
    assert bio_model.marker_names == []

    if os.path.exists(kinematic_model_filepath) and remove_temporary:
        os.remove(kinematic_model_filepath)


class FakeData:
    def __init__(self, model):
        q = np.zeros(model.nbQ())
        marker_positions = np.array(tuple(m.to_array() for m in model.markers(q))).T[:, :, np.newaxis]
        self.values = {m.to_string(): marker_positions[:, i, :] for i, m in enumerate(model.markerNames())}
        self.nb_frames = 1


def test_model_creation_from_data():

    kinematic_model_filepath = "temporary.bioMod"
    test_model_creation_from_static(remove_temporary=False)

    # Prepare a fake model and a fake static from the previous test
    fake_data = FakeData(Model(kinematic_model_filepath))

    # Fill the kinematic chain model
    model = BiomechanicalModel()
    de_leva = DeLevaTable(total_mass=100, sex=Sex.FEMALE)
    de_leva.from_data(fake_data)

    model.add_segment(
        Segment(
            name="TRUNK",
            segment_coordinate_system=SegmentCoordinateSystem(
                lambda m, k: np.array([0, 0, 0]),
                first_axis=Axis(
                    name=Axis.Name.Z,
                    start=lambda m, model: np.array([0, 0, 0]),
                    end=lambda m, model: np.array([0, 0, 1]),
                ),
                second_axis=Axis(
                    name=Axis.Name.X,
                    start=lambda m, model: np.array([0, 0, 0]),
                    end=lambda m, model: np.array([1, 0, 0]),
                ),
                axis_to_keep=Axis.Name.Z,
            ),
            translations=Translations.YZ,
            rotations=Rotations.X,
            inertia_parameters=de_leva[SegmentName.TRUNK],
        )
    )
    model.segments["TRUNK"].add_marker(Marker("PELVIS"))

    model.add_segment(
        Segment(
            name="HEAD",
            parent_name="TRUNK",
            segment_coordinate_system=SegmentCoordinateSystem(
                "BOTTOM_HEAD",
                first_axis=Axis(name=Axis.Name.Z, start="BOTTOM_HEAD", end="HEAD_Z"),
                second_axis=Axis(name=Axis.Name.X, start="BOTTOM_HEAD", end="HEAD_XZ"),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(("BOTTOM_HEAD", "TOP_HEAD", "HEAD_Z", "HEAD_XZ", "BOTTOM_HEAD"), is_local=False),
            inertia_parameters=de_leva[SegmentName.HEAD],
        )
    )
    model.segments["HEAD"].add_marker(Marker("BOTTOM_HEAD"))
    model.segments["HEAD"].add_marker(Marker("TOP_HEAD"))
    model.segments["HEAD"].add_marker(Marker("HEAD_Z"))
    model.segments["HEAD"].add_marker(Marker("HEAD_XZ"))

    model.add_segment(
        Segment(
            name="UPPER_ARM",
            parent_name="TRUNK",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="SHOULDER",
                first_axis=Axis(name=Axis.Name.X, start="SHOULDER", end="SHOULDER_X"),
                second_axis=Axis(name=Axis.Name.Y, start="SHOULDER", end="SHOULDER_XY"),
                axis_to_keep=Axis.Name.X,
            ),
            inertia_parameters=de_leva[SegmentName.UPPER_ARM],
        )
    )
    model.segments["UPPER_ARM"].add_marker(Marker("SHOULDER"))
    model.segments["UPPER_ARM"].add_marker(Marker("SHOULDER_X"))
    model.segments["UPPER_ARM"].add_marker(Marker("SHOULDER_XY"))

    model.add_segment(
        Segment(
            name="LOWER_ARM",
            parent_name="UPPER_ARM",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="ELBOW",
                first_axis=Axis(name=Axis.Name.Y, start="ELBOW", end="ELBOW_Y"),
                second_axis=Axis(name=Axis.Name.X, start="ELBOW", end="ELBOW_XY"),
                axis_to_keep=Axis.Name.Y,
            ),
            inertia_parameters=de_leva[SegmentName.LOWER_ARM],
        )
    )
    model.segments["LOWER_ARM"].add_marker(Marker("ELBOW"))
    model.segments["LOWER_ARM"].add_marker(Marker("ELBOW_Y"))
    model.segments["LOWER_ARM"].add_marker(Marker("ELBOW_XY"))

    model.add_segment(
        Segment(
            name="HAND",
            parent_name="LOWER_ARM",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="WRIST",
                first_axis=Axis(name=Axis.Name.Y, start="WRIST", end="HAND_Y"),
                second_axis=Axis(name=Axis.Name.Z, start="WRIST", end="HAND_YZ"),
                axis_to_keep=Axis.Name.Y,
            ),
            inertia_parameters=de_leva[SegmentName.HAND],
        )
    )
    model.segments["HAND"].add_marker(Marker("WRIST"))
    model.segments["HAND"].add_marker(Marker("FINGER"))
    model.segments["HAND"].add_marker(Marker("HAND_Y"))
    model.segments["HAND"].add_marker(Marker("HAND_YZ"))

    model.add_segment(
        Segment(
            name="THIGH",
            parent_name="TRUNK",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="THIGH_ORIGIN",
                first_axis=Axis(name=Axis.Name.X, start="THIGH_ORIGIN", end="THIGH_X"),
                second_axis=Axis(name=Axis.Name.Y, start="THIGH_ORIGIN", end="THIGH_Y"),
                axis_to_keep=Axis.Name.X,
            ),
            inertia_parameters=de_leva[SegmentName.THIGH],
        )
    )
    model.segments["THIGH"].add_marker(Marker("THIGH_ORIGIN"))
    model.segments["THIGH"].add_marker(Marker("THIGH_X"))
    model.segments["THIGH"].add_marker(Marker("THIGH_Y"))

    model.add_segment(
        Segment(
            name="SHANK",
            parent_name="THIGH",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="KNEE",
                first_axis=Axis(name=Axis.Name.Z, start="KNEE", end="KNEE_Z"),
                second_axis=Axis(name=Axis.Name.X, start="KNEE", end="KNEE_XZ"),
                axis_to_keep=Axis.Name.Z,
            ),
            inertia_parameters=de_leva[SegmentName.SHANK],
        )
    )
    model.segments["SHANK"].add_marker(Marker("KNEE"))
    model.segments["SHANK"].add_marker(Marker("KNEE_Z"))
    model.segments["SHANK"].add_marker(Marker("KNEE_XZ"))

    model.add_segment(
        Segment(
            name="FOOT",
            parent_name="SHANK",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="ANKLE",
                first_axis=Axis(name=Axis.Name.Z, start="ANKLE", end="ANKLE_Z"),
                second_axis=Axis(name=Axis.Name.Y, start="ANKLE", end="ANKLE_YZ"),
                axis_to_keep=Axis.Name.Z,
            ),
            inertia_parameters=de_leva[SegmentName.FOOT],
        )
    )
    model.segments["FOOT"].add_marker(Marker("ANKLE"))
    model.segments["FOOT"].add_marker(Marker("TOE"))
    model.segments["FOOT"].add_marker(Marker("HEEL"))
    model.segments["FOOT"].add_marker(Marker("ANKLE_Z"))
    model.segments["FOOT"].add_marker(Marker("ANKLE_YZ"))

    real_model = model.to_real(fake_data)
    if os.path.exists(kinematic_model_filepath):
        os.remove(kinematic_model_filepath)

    # print it to a bioMod file
    real_model.to_biomod(kinematic_model_filepath, with_mesh=False)

    biorbd_model = Model(kinematic_model_filepath)
    assert biorbd_model.nbQ() == 7
    assert real_model.nb_q == 7
    assert model.nb_q == 7
    assert biorbd_model.nbSegment() == 9
    assert real_model.nb_segments == 9
    assert model.nb_segments == 9
    assert biorbd_model.nbMarkers() == 26
    assert real_model.nb_markers == 26
    assert model.nb_markers == 26
    biorbd_markers = biorbd_model.markers(np.zeros((biorbd_model.nbQ(),)))[-3].to_array()
    np.testing.assert_almost_equal(biorbd_markers, [0, -0.01, -0.85], decimal=4)
    biobuddy_markers = real_model.markers_in_global(np.zeros((real_model.nb_q,)))[:3, -3, 0]
    np.testing.assert_almost_equal(biobuddy_markers, [0, -0.01, -0.85], decimal=4)

    destroy_model(model)
    destroy_model(real_model)

    if os.path.exists(kinematic_model_filepath):
        os.remove(kinematic_model_filepath)


def test_model_creation_from_data_lower_body():

    from examples.create_model_from_c3d import model_creation_from_measured_data

    # Load the static trial
    current_path_file = Path(__file__).parent
    static_trial = C3dData(f"{current_path_file}/../examples/data/static_lower_body.c3d")

    # Create the model
    model_real = model_creation_from_measured_data(
        static_trial=static_trial, remove_temporary=True, animate_model=False
    )

    # Check some values of the model
    assert model_real.nb_q == 14
    assert model_real.nb_segments == 9
    assert model_real.nb_markers == 16

    # Test segment
    segment = model_real.segments["Pelvis"]
    assert segment.name == "Pelvis"
    assert segment.parent_name == "Ground"
    assert segment.translations == Translations.XYZ
    assert segment.rotations == Rotations.XYZ
    assert segment.nb_q == 6
    npt.assert_almost_equal(segment.inertia_parameters.mass, 28.096200000000003)
    npt.assert_almost_equal(
        segment.inertia_parameters.center_of_mass.reshape(
            4,
        ),
        np.array([0.0, 0.0, 0.22295421, 1.0]),
    )
    npt.assert_almost_equal(
        np.diag(segment.inertia_parameters.inertia)[:3], np.array([0.51902017, 0.46954064, 0.11899868])
    )
    npt.assert_almost_equal(
        segment.segment_coordinate_system.scs.rt_matrix,
        np.array(
            [
                [-0.06364053, 0.99797289, 0.0, 0.24594278],
                [-0.99797289, -0.06364053, 0.0, 0.53419547],
                [0.0, 0.0, 1.0, 0.97607442],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    npt.assert_almost_equal(
        segment.mesh.positions,
        np.array(
            [
                [-0.04585975, 0.04703374, 0.10769938, -0.10887336, -0.04585975],
                [-0.10007909, -0.09855995, 0.09855995, 0.10007909, -0.10007909],
                [0.01530446, 0.01463813, -0.00959209, -0.02035051, 0.01530446],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    )

    # Test markers
    npt.assert_almost_equal(
        segment.markers["LPSIS"].position.reshape(
            4,
        ),
        np.array([-0.04585975, -0.10007909, 0.01530446, 1.0]),
    )
    npt.assert_almost_equal(
        segment.markers["RPSIS"].position.reshape(
            4,
        ),
        np.array([0.04703374, -0.09855995, 0.01463813, 1.0]),
    )
    npt.assert_almost_equal(
        segment.markers["LASIS"].position.reshape(
            4,
        ),
        np.array([-0.10887336, 0.10007909, -0.02035051, 1.0]),
    )
    npt.assert_almost_equal(
        segment.markers["RA"].position.reshape(
            4,
        ),
        np.array([0.17473068, -0.00946185, 0.44572759, 1.0]),
    )

    # Test segment
    segment = model_real.segments["RFoot"]
    assert segment.name == "RFoot"
    assert segment.parent_name == "RTibia"
    assert segment.translations == Translations.NONE
    assert segment.rotations == Rotations.X
    assert segment.nb_q == 1
    assert segment.nb_markers == 1
    npt.assert_almost_equal(segment.inertia_parameters.mass, 0.8514)
    npt.assert_almost_equal(
        segment.inertia_parameters.center_of_mass.reshape(
            4,
        ),
        np.array([0.01104557, 0.00422068, 0.04311145, 1.0]),
    )
    npt.assert_almost_equal(
        np.diag(segment.inertia_parameters.inertia)[:3], np.array([0.00879901, 0.00766125, 0.00151333])
    )
    npt.assert_almost_equal(
        segment.segment_coordinate_system.scs.rt_matrix,
        np.array(
            [
                [9.79470459e-01, -1.88107239e-01, 7.24795638e-02, 5.55111512e-17],
                [-9.89117223e-02, -1.35164447e-01, 9.85873746e-01, -2.77555756e-17],
                [-1.75653328e-01, -9.72803289e-01, -1.50995594e-01, -3.93526241e-01],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        ),
    )
    npt.assert_almost_equal(
        segment.mesh.positions,
        np.array(
            [
                [4.05993515e-02, -4.16333634e-17, -4.05993515e-02, 4.05993515e-02],
                [1.38777878e-17, 0.00000000e00, 1.38777878e-17, 1.38777878e-17],
                [4.07329215e-03, 1.35427279e-01, -4.07329215e-03, 4.07329215e-03],
                [1.00000000e00, 1.00000000e00, 1.00000000e00, 1.00000000e00],
            ]
        ),
    )

    # Test markers
    npt.assert_almost_equal(
        segment.markers["RTT2"].position.reshape(
            4,
        ),
        np.array([-4.16333634e-17, 0.00000000e00, 1.35427279e-01, 1.00000000e00]),
    )


def test_complex_model():
    from examples.create_model import complex_model_from_scratch

    current_path_folder = os.path.dirname(os.path.realpath(__file__))
    kinematic_model_filepath = f"{current_path_folder}/../examples/models/temporary_complex.bioMod"
    mesh_path = f"{current_path_folder}/../examples/models/meshes"

    # Create the model
    real_model = complex_model_from_scratch(mesh_path=mesh_path, remove_temporary=False)

    # Print it to a bioMod file
    real_model.to_biomod(kinematic_model_filepath, with_mesh=False)

    model = Model(kinematic_model_filepath)
    assert model.nbQ() == 4
    assert real_model.nb_q == 4
    assert model.nbSegment() == 3
    assert real_model.nb_segments == 3
    assert model.nbMarkers() == 0
    assert real_model.nb_markers == 0
    assert model.nbMuscles() == 1
    assert real_model.nb_muscles == 1
    assert model.nbMuscleGroups() == 1
    assert real_model.nb_muscle_groups == 1
    assert model.nbContacts() == 3  # Number of rigid contact axis
    assert real_model.nb_contacts == 1  # Number of rigid contact points

    # Test the position of elements
    assert real_model.segments["PENDULUM"].q_ranges.min_bound == [-1, -1, -1, -np.pi]
    assert real_model.segments["PENDULUM"].q_ranges.max_bound == [1, 1, 1, np.pi]
    assert real_model.segments["PENDULUM"].qdot_ranges.min_bound == [-10, -10, -10, -np.pi * 10]
    assert real_model.segments["PENDULUM"].qdot_ranges.max_bound == [10, 10, 10, np.pi * 10]
    npt.assert_almost_equal(
        real_model.segments["PENDULUM"].mesh_file.mesh_scale.reshape(
            4,
        ),
        np.array([1.0, 1.0, 10.0, 1.0]),
    )
    npt.assert_almost_equal(
        real_model.segments["PENDULUM"].mesh_file.mesh_rotation.reshape(
            4,
        ),
        np.array([np.pi / 2, 0.0, 0.0, 1.0]),
    )
    npt.assert_almost_equal(
        real_model.segments["PENDULUM"].mesh_file.mesh_translation.reshape(
            4,
        ),
        np.array([0.1, 0.0, 0.0, 1.0]),
    )

    npt.assert_almost_equal(
        real_model.segments["PENDULUM"]
        .contacts["PENDULUM_CONTACT"]
        .position.reshape(
            4,
        ),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    assert real_model.segments["PENDULUM"].contacts["PENDULUM_CONTACT"].axis == Translations.XYZ

    assert real_model.muscle_groups["PENDULUM_MUSCLE_GROUP"].origin_parent_name == "GROUND"
    assert real_model.muscle_groups["PENDULUM_MUSCLE_GROUP"].insertion_parent_name == "PENDULUM"

    assert (
        real_model.muscle_groups["PENDULUM_MUSCLE_GROUP"].muscles["PENDULUM_MUSCLE"].muscle_type
        == MuscleType.HILL_THELEN
    )
    assert (
        real_model.muscle_groups["PENDULUM_MUSCLE_GROUP"].muscles["PENDULUM_MUSCLE"].state_type
        == MuscleStateType.DEGROOTE
    )
    npt.assert_almost_equal(
        real_model.muscle_groups["PENDULUM_MUSCLE_GROUP"]
        .muscles["PENDULUM_MUSCLE"]
        .origin_position.position.reshape(
            4,
        ),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    npt.assert_almost_equal(
        real_model.muscle_groups["PENDULUM_MUSCLE_GROUP"]
        .muscles["PENDULUM_MUSCLE"]
        .insertion_position.position.reshape(
            4,
        ),
        np.array([0.0, 0.0, 1.0, 1.0]),
    )
    assert real_model.muscle_groups["PENDULUM_MUSCLE_GROUP"].muscles["PENDULUM_MUSCLE"].optimal_length == 0.1
    assert real_model.muscle_groups["PENDULUM_MUSCLE_GROUP"].muscles["PENDULUM_MUSCLE"].maximal_force == 100.0
    assert real_model.muscle_groups["PENDULUM_MUSCLE_GROUP"].muscles["PENDULUM_MUSCLE"].tendon_slack_length == 0.05
    assert real_model.muscle_groups["PENDULUM_MUSCLE_GROUP"].muscles["PENDULUM_MUSCLE"].pennation_angle == 0.05

    npt.assert_almost_equal(
        real_model.muscle_groups["PENDULUM_MUSCLE_GROUP"]
        .muscles["PENDULUM_MUSCLE"]
        .via_points["PENDULUM_MUSCLE"]
        .position.reshape(
            4,
        ),
        np.array([0.0, 0.0, 0.5, 1.0]),
    )

    destroy_model(real_model)

    if os.path.exists(kinematic_model_filepath):
        os.remove(kinematic_model_filepath)
