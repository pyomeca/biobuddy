import numpy as np
import numpy.testing as npt

from biobuddy import BiomechanicalModelReal, build_preview_scene
from biobuddy.components.muscle_utils import MuscleStateType, MuscleType
from biobuddy.components.real.force.muscle_group_real import MuscleGroupReal
from biobuddy.components.real.force.muscle_real import MuscleReal
from biobuddy.components.real.force.via_point_real import ViaPointReal
from biobuddy.components.real.rigidbody.marker_real import MarkerReal
from biobuddy.components.real.rigidbody.segment_coordinate_system_real import (
    SegmentCoordinateSystemReal,
)
from biobuddy.components.real.rigidbody.segment_real import SegmentReal
from biobuddy.utils.enums import Rotations


def test_build_preview_scene_collects_joints_markers_and_muscles():
    """
    Build renderable geometry from a simple biomechanical model.
    """
    model = BiomechanicalModelReal()
    model.add_segment(SegmentReal(name="Pelvis"))
    model.add_segment(
        SegmentReal(
            name="Thigh",
            parent_name="Pelvis",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                angles=np.zeros(3),
                angle_sequence="xyz",
                translation=np.array([0.0, -1.0, 0.0]),
                is_scs_local=True,
            ),
            rotations=Rotations.X,
        )
    )
    model.segments["Pelvis"].add_marker(MarkerReal("LASI", "Pelvis", np.array([1.0, 0.0, 0.0])))

    muscle_group = MuscleGroupReal("Hip", "Pelvis", "Thigh")
    muscle_group.add_muscle(
        MuscleReal(
            name="Flexor",
            muscle_type=MuscleType.HILL_DE_GROOTE,
            state_type=MuscleStateType.DEGROOTE,
            muscle_group="Hip",
            origin_position=ViaPointReal("origin", "Pelvis", position=np.array([0.0, 0.0, 0.0])),
            insertion_position=ViaPointReal("insertion", "Thigh", position=np.array([0.0, 0.0, 0.0])),
            maximal_force=100.0,
        )
    )
    model.add_muscle_group(muscle_group)

    scene = build_preview_scene(model)

    assert scene.bones == [("root", "Pelvis"), ("Pelvis", "Thigh")]
    npt.assert_array_equal(scene.joints["Thigh"], np.array([0.0, -1.0, 0.0]))
    npt.assert_array_equal(scene.markers["LASI"], np.array([1.0, 0.0, 0.0]))
    assert list(scene.muscles.keys()) == ["Flexor"]

    thigh_axes = {axis.axis: axis for axis in scene.segment_axes if axis.segment_name == "Thigh"}
    assert set(thigh_axes) == {"x", "y", "z"}
    assert thigh_axes["x"].is_rotation_axis is True
    assert thigh_axes["y"].is_rotation_axis is False
    npt.assert_array_equal(thigh_axes["x"].start, scene.joints["Thigh"])
    assert np.linalg.norm(thigh_axes["x"].end - thigh_axes["x"].start) > 0.0
