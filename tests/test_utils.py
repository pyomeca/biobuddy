import os
import numpy as np

from biobuddy import (
    BiomechanicalModelReal,
    SegmentReal,
    MarkerReal,
    InertiaParametersReal,
    SegmentCoordinateSystemReal,
    C3dData,
)


def destroy_model(bio_model: BiomechanicalModelReal):
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
    assert bio_model.nb_muscle_groups == 0
    assert bio_model.muscle_group_names == []

    # Remove muscles
    for muscle_name in bio_model.muscle_names:
        bio_model.remove_muscle(muscle_name)
    assert bio_model.nb_muscles == 0
    assert bio_model.muscle_names == []

    # Remove via points
    for via_point_name in bio_model.via_point_names:
        bio_model.remove_via_point(via_point_name)
    assert bio_model.nb_via_points == 0
    assert bio_model.via_point_names == []


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

    # Add a root segment
    model.add_segment(
        SegmentReal(
            name="root",
            segment_coordinate_system=SegmentCoordinateSystemReal(scs=np.eye(4), is_scs_local=True),
            inertia_parameters=InertiaParametersReal(
                mass=10.0, center_of_mass=np.array([0.0, 0.0, 0.5, 1.0]), inertia=np.eye(3) * 0.3
            ),
        )
    )

    # Add a child segment
    segment_coordinate_system_child = SegmentCoordinateSystemReal()
    segment_coordinate_system_child.from_euler_and_translation(np.zeros((3,)), "xyz", np.array([0.0, 0.0, 1.0, 1.0]))
    segment_coordinate_system_child.is_in_local = True
    model.add_segment(
        SegmentReal(
            name="child",
            parent_name="root",
            segment_coordinate_system=segment_coordinate_system_child,
            inertia_parameters=InertiaParametersReal(
                mass=5.0, center_of_mass=np.array([0.0, 0.1, 0.0, 1.0]), inertia=np.eye(3) * 0.01
            ),
        )
    )

    # Add markers to segments
    model.segments["root"].add_marker(
        MarkerReal(
            name="root_marker",
            parent_name="root",
            position=np.array([0.1, 0.2, 0.3, 1.0]),
            is_technical=True,
            is_anatomical=False,
        )
    )
    model.segments["root"].add_marker(
        MarkerReal(
            name="root_marker2",
            parent_name="root",
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
    def __init__(self, c3d_path):

        super().__init__(self, c3d_path)

        # Fix the seed for reproducibility
        np.random.seed(42)

        self.marker_names = ["marker1", "marker2", "marker3", "marker4", "marker5"]
        # Create marker positions for 10 frames
        self.all_marker_positions = np.random.rand(4, 5, 10)
