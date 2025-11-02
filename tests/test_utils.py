import os
import numpy as np
import numpy.testing as npt
from lxml import etree

from biobuddy import (
    BiomechanicalModelReal,
    BiomechanicalModel,
    SegmentReal,
    MarkerReal,
    ContactReal,
    InertiaParametersReal,
    SegmentCoordinateSystemReal,
    C3dData,
    RotoTransMatrix,
    MuscleGroupReal,
    MuscleReal,
    ViaPointReal,
    MuscleType,
    MuscleStateType,
    Translations,
    Rotations,
    RangeOfMotion,
    Ranges,
    MeshReal,
)


def destroy_model(bio_model: BiomechanicalModelReal | BiomechanicalModel):
    """
    Let's test the remove functions and make sure that there is nothing left in the model.
    """

    # Remove segments
    for segment_name in bio_model.segment_names:

        # Remove markers
        marker_names = bio_model.segments[segment_name].marker_names
        for marker_name in marker_names:
            bio_model.segments[segment_name].remove_marker(marker_name)
        assert bio_model.segments[segment_name].nb_markers == 0

        # Remove contacts
        contact_names = bio_model.segments[segment_name].contact_names
        for contact_name in contact_names:
            bio_model.segments[segment_name].remove_contact(contact_name)
        assert bio_model.segments[segment_name].nb_contacts == 0

        # Remove imus
        imu_names = bio_model.segments[segment_name].imu_names
        for imu_name in imu_names:
            bio_model.segments[segment_name].remove_imu(imu_name)
        assert bio_model.segments[segment_name].nb_imus == 0

        # Remove segment
        bio_model.remove_segment(segment_name)
    assert bio_model.nb_segments == 0
    assert bio_model.segment_names == []

    # Remove muscle groups
    for muscle_group_name in bio_model.muscle_group_names:
        bio_model.remove_muscle_group(muscle_group_name)

        # Remove muscles
        for muscle_name in bio_model.muscle_names:
            bio_model.muscle_groups[muscle_group_name].remove_muscle(muscle_name)

            # Remove via points
            for via_point_name in bio_model.via_point_names:
                bio_model.muscle_groups[muscle_group_name].muscles[muscle_name].remove_via_point(via_point_name)
            assert bio_model.nb_via_points == 0
            assert bio_model.via_point_names == []

        assert bio_model.nb_muscles == 0
        assert bio_model.muscle_names == []

    assert bio_model.nb_muscle_groups == 0
    assert bio_model.muscle_group_names == []


def compare_models(model1: BiomechanicalModelReal, model2: BiomechanicalModelReal, decimal: int = 5):
    """
    Compare two biomechanical models for equality.
    """

    # Compare segments
    assert model1.nb_segments == model2.nb_segments
    assert set(model1.segment_names) == set(model2.segment_names)

    assert model1.nb_markers == model2.nb_markers
    assert set(model1.marker_names) == set(model2.marker_names)
    assert model1.nb_contacts == model2.nb_contacts
    assert set(model1.contact_names) == set(model2.contact_names)
    assert model1.nb_imus == model2.nb_imus
    assert set(model1.imu_names) == set(model2.imu_names)

    for segment_name in model1.segment_names:
        assert model1.segments[segment_name].name == model2.segments[segment_name].name
        assert model1.segments[segment_name].parent_name == model2.segments[segment_name].parent_name
        npt.assert_almost_equal(
            model1.segments[segment_name].segment_coordinate_system.scs.rt_matrix,
            model2.segments[segment_name].segment_coordinate_system.scs.rt_matrix,
            decimal=decimal,
        )
        assert np.all(model1.segments[segment_name].translations == model2.segments[segment_name].translations)
        assert np.all(model1.segments[segment_name].rotations == model2.segments[segment_name].rotations)
        if model1.segments[segment_name].q_ranges is not None:
            npt.assert_almost_equal(
                model1.segments[segment_name].q_ranges.min_bound,
                model2.segments[segment_name].q_ranges.min_bound,
                decimal=decimal,
            )
            npt.assert_almost_equal(
                model1.segments[segment_name].q_ranges.max_bound,
                model2.segments[segment_name].q_ranges.max_bound,
                decimal=decimal,
            )
        else:
            assert model2.segments[segment_name].q_ranges is None
        if model1.segments[segment_name].inertia_parameters is not None:
            assert np.all(
                model1.segments[segment_name].inertia_parameters.mass
                == model2.segments[segment_name].inertia_parameters.mass
            )
            npt.assert_almost_equal(
                model1.segments[segment_name].inertia_parameters.inertia,
                model2.segments[segment_name].inertia_parameters.inertia,
                decimal=decimal,
            )
            npt.assert_almost_equal(
                model1.segments[segment_name].inertia_parameters.center_of_mass,
                model2.segments[segment_name].inertia_parameters.center_of_mass,
                decimal=decimal,
            )
        else:
            assert model2.segments[segment_name].inertia_parameters is None
        if model1.segments[segment_name].mesh is not None:
            npt.assert_almost_equal(
                model1.segments[segment_name].mesh.positions,
                model2.segments[segment_name].mesh.positions,
                decimal=decimal,
            )
        else:
            assert model2.segments[segment_name].mesh is None
        if model1.segments[segment_name].mesh_file is not None:
            assert np.all(
                model1.segments[segment_name].mesh_file.mesh_scale == model2.segments[segment_name].mesh_file.mesh_scale
            )
            assert np.all(
                model1.segments[segment_name].mesh_file.mesh_rotation
                == model2.segments[segment_name].mesh_file.mesh_rotation
            )
            assert np.all(
                model1.segments[segment_name].mesh_file.mesh_translation
                == model2.segments[segment_name].mesh_file.mesh_translation
            )
            assert np.all(
                model1.segments[segment_name].mesh_file.mesh_color == model2.segments[segment_name].mesh_file.mesh_color
            )
        else:
            assert model2.segments[segment_name].mesh_file is None

        # Compare markers
        assert model1.segments[segment_name].nb_markers == model2.segments[segment_name].nb_markers
        assert set(model1.segments[segment_name].marker_names) == set(model2.segments[segment_name].marker_names)
        for marker_name in model1.segments[segment_name].marker_names:
            npt.assert_almost_equal(
                model1.segments[segment_name].markers[marker_name].position,
                model2.segments[segment_name].markers[marker_name].position,
                decimal=decimal,
            )
            assert (
                model1.segments[segment_name].markers[marker_name].parent_name
                == model2.segments[segment_name].markers[marker_name].parent_name
            )
            assert (
                model1.segments[segment_name].markers[marker_name].is_technical
                == model2.segments[segment_name].markers[marker_name].is_technical
            )
            assert (
                model1.segments[segment_name].markers[marker_name].is_anatomical
                == model2.segments[segment_name].markers[marker_name].is_anatomical
            )

        # Compare contacts
        assert model1.segments[segment_name].nb_contacts == model2.segments[segment_name].nb_contacts
        assert set(model1.segments[segment_name].contact_names) == set(model2.segments[segment_name].contact_names)
        for contact_name in model1.segments[segment_name].contact_names:
            npt.assert_almost_equal(
                model1.segments[segment_name].contacts[contact_name].position,
                model2.segments[segment_name].contacts[contact_name].position,
                decimal=decimal,
            )
            assert (
                model1.segments[segment_name].contacts[contact_name].parent_name
                == model2.segments[segment_name].contacts[contact_name].parent_name
            )
            assert (
                model1.segments[segment_name].contacts[contact_name].is_technical
                == model2.segments[segment_name].contacts[contact_name].is_technical
            )
            assert (
                model1.segments[segment_name].contacts[contact_name].is_anatomical
                == model2.segments[segment_name].contacts[contact_name].is_anatomical
            )

        # Compare imus
        assert model1.segments[segment_name].nb_imus == model2.segments[segment_name].nb_imus
        assert set(model1.segments[segment_name].imu_names) == set(model2.segments[segment_name].imu_names)
        for imu_name in model1.segments[segment_name].imu_names:
            npt.assert_almost_equal(
                model1.segments[segment_name].imus[imu_name].scs.rt_matrix,
                model2.segments[segment_name].imus[imu_name].scs.rt_matrix,
                decimal=decimal,
            )
            assert (
                model1.segments[segment_name].imus[imu_name].parent_name
                == model2.segments[segment_name].imus[imu_name].parent_name
            )
            assert (
                model1.segments[segment_name].imus[imu_name].is_technical
                == model2.segments[segment_name].imus[imu_name].is_technical
            )
            assert (
                model1.segments[segment_name].imus[imu_name].is_anatomical
                == model2.segments[segment_name].imus[imu_name].is_anatomical
            )

    # Compare muscle groups
    assert model1.nb_muscle_groups == model2.nb_muscle_groups
    assert set(model1.muscle_group_names) == set(model2.muscle_group_names)
    assert model1.nb_muscles == model2.nb_muscles
    assert set(model1.muscle_names) == set(model2.muscle_names)
    assert model1.nb_via_points == model2.nb_via_points
    assert set(model1.via_point_names) == set(model2.via_point_names)
    for muscle_group_name in model1.muscle_group_names:
        assert (
            model1.muscle_groups[muscle_group_name].insertion_parent_name
            == model2.muscle_groups[muscle_group_name].insertion_parent_name
        )
        assert (
            model1.muscle_groups[muscle_group_name].origin_parent_name
            == model2.muscle_groups[muscle_group_name].origin_parent_name
        )

        # Compare muscles
        for muscle_name in model1.muscle_groups[muscle_group_name].muscle_names:
            assert (
                model1.muscle_groups[muscle_group_name].muscles[muscle_name].muscle_type
                == model2.muscle_groups[muscle_group_name].muscles[muscle_name].muscle_type
            )
            assert (
                model1.muscle_groups[muscle_group_name].muscles[muscle_name].state_type
                == model2.muscle_groups[muscle_group_name].muscles[muscle_name].state_type
            )
            assert (
                model1.muscle_groups[muscle_group_name].muscles[muscle_name].muscle_group
                == model2.muscle_groups[muscle_group_name].muscles[muscle_name].muscle_group
            )
            npt.assert_almost_equal(
                model1.muscle_groups[muscle_group_name].muscles[muscle_name].origin_position.position,
                model2.muscle_groups[muscle_group_name].muscles[muscle_name].origin_position.position,
                decimal=decimal,
            )
            npt.assert_almost_equal(
                model1.muscle_groups[muscle_group_name].muscles[muscle_name].insertion_position.position,
                model2.muscle_groups[muscle_group_name].muscles[muscle_name].insertion_position.position,
                decimal=decimal,
            )
            assert (
                model1.muscle_groups[muscle_group_name].muscles[muscle_name].optimal_length
                == model2.muscle_groups[muscle_group_name].muscles[muscle_name].optimal_length
            )
            assert (
                model1.muscle_groups[muscle_group_name].muscles[muscle_name].maximal_force
                == model2.muscle_groups[muscle_group_name].muscles[muscle_name].maximal_force
            )
            assert (
                model1.muscle_groups[muscle_group_name].muscles[muscle_name].tendon_slack_length
                == model2.muscle_groups[muscle_group_name].muscles[muscle_name].tendon_slack_length
            )
            assert (
                model1.muscle_groups[muscle_group_name].muscles[muscle_name].pennation_angle
                == model2.muscle_groups[muscle_group_name].muscles[muscle_name].pennation_angle
            )
            assert (
                model1.muscle_groups[muscle_group_name].muscles[muscle_name].maximal_velocity
                == model2.muscle_groups[muscle_group_name].muscles[muscle_name].maximal_velocity
            )
            assert (
                model1.muscle_groups[muscle_group_name].muscles[muscle_name].maximal_excitation
                == model2.muscle_groups[muscle_group_name].muscles[muscle_name].maximal_excitation
            )

            # Compare via points
            for via_point_name in model1.muscle_groups[muscle_group_name].muscles[muscle_name].via_point_names:
                assert (
                    model1.muscle_groups[muscle_group_name].muscles[muscle_name].via_points[via_point_name].parent_name
                    == model2.muscle_groups[muscle_group_name]
                    .muscles[muscle_name]
                    .via_points[via_point_name]
                    .parent_name
                )
                assert (
                    model1.muscle_groups[muscle_group_name].muscles[muscle_name].via_points[via_point_name].muscle_name
                    == model2.muscle_groups[muscle_group_name]
                    .muscles[muscle_name]
                    .via_points[via_point_name]
                    .muscle_name
                )
                assert (
                    model1.muscle_groups[muscle_group_name].muscles[muscle_name].via_points[via_point_name].muscle_group
                    == model2.muscle_groups[muscle_group_name]
                    .muscles[muscle_name]
                    .via_points[via_point_name]
                    .muscle_group
                )
                npt.assert_almost_equal(
                    model1.muscle_groups[muscle_group_name].muscles[muscle_name].via_points[via_point_name].position,
                    model2.muscle_groups[muscle_group_name].muscles[muscle_name].via_points[via_point_name].position,
                    decimal=decimal,
                )
                if (
                    model1.muscle_groups[muscle_group_name].muscles[muscle_name].via_points[via_point_name].condition
                    is not None
                ):
                    assert (
                        model1.muscle_groups[muscle_group_name]
                        .muscles[muscle_name]
                        .via_points[via_point_name]
                        .condition.dof_name
                        == model2.muscle_groups[muscle_group_name]
                        .muscles[muscle_name]
                        .via_points[via_point_name]
                        .condition.dof_name
                    )
                    assert (
                        model1.muscle_groups[muscle_group_name]
                        .muscles[muscle_name]
                        .via_points[via_point_name]
                        .condition.range_min
                        == model2.muscle_groups[muscle_group_name]
                        .muscles[muscle_name]
                        .via_points[via_point_name]
                        .condition.range_min
                    )
                    assert (
                        model1.muscle_groups[muscle_group_name]
                        .muscles[muscle_name]
                        .via_points[via_point_name]
                        .condition.range_max
                        == model2.muscle_groups[muscle_group_name]
                        .muscles[muscle_name]
                        .via_points[via_point_name]
                        .condition.range_max
                    )
                else:
                    assert (
                        model2.muscle_groups[muscle_group_name]
                        .muscles[muscle_name]
                        .via_points[via_point_name]
                        .condition
                        is None
                    )
                if (
                    model1.muscle_groups[muscle_group_name].muscles[muscle_name].via_points[via_point_name].movement
                    is not None
                ):
                    assert set(
                        model1.muscle_groups[muscle_group_name]
                        .muscles[muscle_name]
                        .via_points[via_point_name]
                        .movement.dof_names
                    ) == set(
                        model2.muscle_groups[muscle_group_name]
                        .muscles[muscle_name]
                        .via_points[via_point_name]
                        .movement.dof_names
                    )
                    for i in range(3):
                        assert (
                            model1.muscle_groups[muscle_group_name]
                            .muscles[muscle_name]
                            .via_points[via_point_name]
                            .movement.locations[i]
                            .x_points
                            == model2.muscle_groups[muscle_group_name]
                            .muscles[muscle_name]
                            .via_points[via_point_name]
                            .movement.locations[i]
                            .x_points
                        )
                        assert (
                            model1.muscle_groups[muscle_group_name]
                            .muscles[muscle_name]
                            .via_points[via_point_name]
                            .movement.locations[i]
                            .y_points
                            == model2.muscle_groups[muscle_group_name]
                            .muscles[muscle_name]
                            .via_points[via_point_name]
                            .movement.locations[i]
                            .y_points
                        )
                else:
                    assert (
                        model2.muscle_groups[muscle_group_name].muscles[muscle_name].via_points[via_point_name].movement
                        is None
                    )


def remove_temporary_biomods():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_names_to_remove = ["temporary.bioMod", "temporary_rt.bioMod"]

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in file_names_to_remove:
                full_path = os.path.join(dirpath, filename)
                try:
                    os.remove(full_path)
                except:
                    print(f"File {full_path} could not be deleted.")


def create_simple_model():
    """Create a simple model for testing"""
    model = BiomechanicalModelReal()

    # Add a parent segment
    model.add_segment(
        SegmentReal(
            name="parent",
            translations=Translations.XYZ,
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystemReal(scs=RotoTransMatrix(), is_scs_local=True),
            inertia_parameters=InertiaParametersReal(
                mass=10.0, center_of_mass=np.array([0.0, 0.0, 0.5, 1.0]), inertia=np.eye(3) * 0.3
            ),
            q_ranges=RangeOfMotion(range_type=Ranges.Q, min_bound=[-np.pi] * 6, max_bound=[np.pi] * 6),
            mesh=MeshReal(np.array([[0, 0, 0, 1], [0.2, 0.1, 0.3, 1]]).T),
        )
    )

    # Add a child segment
    segment_coordinate_system_child = SegmentCoordinateSystemReal().from_euler_and_translation(
        angles=np.zeros((3,)),
        angle_sequence="xyz",
        translation=np.array([0.0, 0.0, 1.0, 1.0]),
    )
    segment_coordinate_system_child.is_in_local = True
    model.add_segment(
        SegmentReal(
            name="child",
            parent_name="parent",
            rotations=Rotations.X,
            segment_coordinate_system=segment_coordinate_system_child,
            inertia_parameters=InertiaParametersReal(
                mass=5.0, center_of_mass=np.array([0.0, 0.1, 0.0, 1.0]), inertia=np.eye(3) * 0.01
            ),
            mesh=MeshReal(np.array([[0, 0, 0, 1], [0.2, 0.1, 0.3, 1]]).T),
        )
    )

    # Add markers to segments
    model.segments["parent"].add_marker(
        MarkerReal(
            name="parent_marker",
            parent_name="parent",
            position=np.array([0.1, 0.2, 0.3, 1.0]),
            is_technical=True,
            is_anatomical=False,
        )
    )
    model.segments["parent"].add_marker(
        MarkerReal(
            name="parent_marker2",
            parent_name="parent",
            position=np.array([0.2, 0.2, 0.1, 1.0]),
            is_technical=True,
            is_anatomical=False,
        )
    )

    model.segments["child"].add_marker(
        MarkerReal(
            name="child_marker",
            parent_name="child",
            position=np.array([0.4, 0.5, 0.6, 1.0]),
            is_technical=True,
            is_anatomical=False,
        )
    )
    model.segments["child"].add_marker(
        MarkerReal(
            name="child_marker2",
            parent_name="child",
            position=np.array([0.1, 0.3, 0.5, 1.0]),
            is_technical=True,
            is_anatomical=False,
        )
    )

    model.segments["parent"].add_contact(
        ContactReal(
            name="parent_contact1",
            parent_name="parent",
            position=np.array([0.05, 0.2, 0.15, 1.0]),
            axis=Translations.XYZ,
        )
    )

    model.segments["child"].add_contact(
        ContactReal(
            name="child_contact1",
            parent_name="child",
            position=np.array([-0.05, 0.5, 0.35, 1.0]),
            axis=Translations.Z,
        )
    )

    model.add_muscle_group(
        MuscleGroupReal(name="parent_to_child", origin_parent_name="parent", insertion_parent_name="child")
    )
    model.muscle_groups["parent_to_child"].add_muscle(
        MuscleReal(
            name="muscle1",
            muscle_type=MuscleType.HILL_DE_GROOTE,
            state_type=MuscleStateType.DEGROOTE,
            muscle_group="parent_to_child",
            origin_position=ViaPointReal(
                name=f"origin_muscle1", parent_name="parent", position=np.array([0.0, 0.1, 0.0, 1.0])
            ),
            insertion_position=ViaPointReal(
                name=f"insertion_muscle1", parent_name="child", position=np.array([0.5, 0.4, 0.3, 1.0])
            ),
            optimal_length=0.5,
            maximal_force=1000,
            tendon_slack_length=0.2,
            pennation_angle=0.1,
            maximal_excitation=1.0,
        )
    )

    model.muscle_groups["parent_to_child"].muscles["muscle1"].add_via_point(
        ViaPointReal(
            name="via_point1",
            parent_name="child",
            position=np.array([0.2, 0.3, 0.4, 1.0]),
        )
    )

    return model


class MockEmptyC3dData(C3dData):
    def __init__(self):

        parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        knee_functional_trial_path = parent_path + "/examples/data/functional_trials/right_knee.c3d"
        C3dData.__init__(self, c3d_path=knee_functional_trial_path, first_frame=0, last_frame=0)

    @property
    def all_marker_positions(self) -> np.ndarray:
        return self.get_position(marker_names=self.marker_names)

    @all_marker_positions.setter
    def all_marker_positions(self, value: np.ndarray):
        # Removing the check for shape to allow empty data
        self.ezc3d_data["data"]["points"][:, :, self.first_frame : self.last_frame] = value


class MockC3dData(C3dData):
    def __init__(self, c3d_path: str = None):

        if c3d_path is None:
            parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            c3d_path = parent_path + "/examples/data/functional_trials/right_knee.c3d"

        C3dData.__init__(self, c3d_path, first_frame=0, last_frame=10)

        # Fix the seed for reproducibility
        np.random.seed(42)

        # Create marker positions for 10 frames
        self.all_marker_positions = np.random.rand(4, 73, 10)


def read_xml_str():
    # Read the content
    with open("temporary.xml", "r") as file:
        xml_content = file.read()

    # Remove the temporary file
    os.remove("temporary.xml")
    return xml_content


def get_xml_str(fake_xml_model: etree.Element) -> str:
    # Write to a temporary file
    tree = etree.ElementTree(fake_xml_model)
    tree.write("temporary.xml", pretty_print=True, xml_declaration=True, encoding="utf-8")

    xml_content = read_xml_str()

    return xml_content
