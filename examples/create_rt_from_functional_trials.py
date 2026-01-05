"""
This example shows how to build a RT from functional trials.
See https://github.com/s2mLab/momentum_health_walking_reconstruction/blob/main/momentum_health_walking_reconstruction/models/lower_body.py for a more complete example.
"""

import logging
from pathlib import Path

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
    SegmentCoordinateSystemUtils,
    BiomechanicalModelReal,
)

_logger = logging.getLogger(__name__)

# TODO: The SCoRE/SARA seems weird ! This should be checked, but in the mean time this example is used ofr tests


def generate_lower_body_model(visualize: bool = True) -> BiomechanicalModelReal:

    # --- Load the c3d data --- #
    current_dir = Path(__file__).parent
    static_c3d_path = f"{current_dir}/data/anat_pose_ECH.c3d"
    static_data = C3dData(static_c3d_path)
    right_hip_functional_c3d_path = f"{current_dir}/data/functional_trials/right_hip.c3d"
    right_hip_data = C3dData(right_hip_functional_c3d_path)
    right_knee_functional_c3d_path = f"{current_dir}/data/functional_trials/right_knee.c3d"
    right_knee_data = C3dData(right_knee_functional_c3d_path)

    # --- Generate the personalized kinematic model --- #
    model = BiomechanicalModel()

    # Pelvis
    model.add_segment(
        Segment(
            name="Pelvis",
            parent_name="Ground",
            translations=Translations.XYZ,
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=SegmentCoordinateSystemUtils.mean_markers(["RASIS", "LASIS", "RPSIS", "LPSIS"]),
                first_axis=Axis(
                    name=Axis.Name.X,
                    start=SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "LASIS"]),
                    end=SegmentCoordinateSystemUtils.mean_markers(["RPSIS", "RASIS"]),
                ),
                second_axis=Axis(
                    name=Axis.Name.Y,
                    start=SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "RPSIS"]),
                    end=SegmentCoordinateSystemUtils.mean_markers(["LASIS", "RASIS"]),
                ),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("LPSIS", "RPSIS", "RASIS", "LASIS", "LPSIS"), is_local=False),
        )
    )
    model.segments["Pelvis"].add_marker(Marker("LPSIS", is_technical=True, is_anatomical=True))
    model.segments["Pelvis"].add_marker(Marker("RPSIS", is_technical=True, is_anatomical=True))
    model.segments["Pelvis"].add_marker(Marker("LASIS", is_technical=True, is_anatomical=True))
    model.segments["Pelvis"].add_marker(Marker("RASIS", is_technical=True, is_anatomical=True))

    # Hip
    right_knee_mid = SegmentCoordinateSystemUtils.mean_markers(["RLFE", "RMFE"])
    right_hip_origin = SegmentCoordinateSystemUtils.score(
        functional_data=right_hip_data,
        parent_marker_names=["LPSIS", "RPSIS", "LASIS", "RASIS"],
        child_marker_names=["RLFE", "RMFE", "RTHI1", "RTHI2", "RTHI3"],
        visualize=visualize,
    )
    model.add_segment(
        Segment(
            name="RThigh",
            parent_name="Pelvis",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=right_hip_origin,
                first_axis=Axis(name=Axis.Name.Z, start=right_knee_mid, end=right_hip_origin),
                second_axis=Axis(name=Axis.Name.X, start="LASIS", end="RASIS"),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(
                (
                    right_hip_origin,
                    "RTHI1",
                    "RTHI3",
                    "RTHI2",
                    "RTHI1",
                    "RLFE",
                    right_knee_mid,
                    "RMFE",
                    right_hip_origin,
                ),
                is_local=False,
            ),
        )
    )
    model.segments["RThigh"].add_marker(Marker("RTHI1", is_technical=True, is_anatomical=False))
    model.segments["RThigh"].add_marker(Marker("RTHI2", is_technical=True, is_anatomical=False))
    model.segments["RThigh"].add_marker(Marker("RTHI3", is_technical=True, is_anatomical=False))
    model.segments["RThigh"].add_marker(Marker("RLFE", is_technical=False, is_anatomical=True))
    model.segments["RThigh"].add_marker(Marker("RMFE", is_technical=False, is_anatomical=True))

    # Knee
    right_tibia_axis = SegmentCoordinateSystemUtils.sara(
        name=Axis.Name.X,
        functional_data=right_knee_data,
        parent_marker_names=["RTHI1", "RTHI2", "RTHI3"],
        child_marker_names=["RLEG1", "RLEG2", "RLEG3", "RATT", "RLM", "RSPH"],
        visualize=visualize,
    )
    right_ankle_mid = SegmentCoordinateSystemUtils.mean_markers(["RLM", "RSPH"])
    model.add_segment(
        Segment(
            name="RShank",
            parent_name="RThigh",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=right_tibia_axis.start,
                first_axis=Axis(name=Axis.Name.Z, start=right_ankle_mid, end=right_tibia_axis.start),
                second_axis=right_tibia_axis,
                axis_to_keep=Axis.Name.X,
            ),
            mesh=Mesh(
                (
                    right_tibia_axis.start,
                    "RLEG1",
                    "RLEG2",
                    "RLEG3",
                    "RLEG1",
                    right_tibia_axis.start,
                    right_ankle_mid,
                    "RLM",
                    "RSPH",
                    right_tibia_axis.start,
                ),
                is_local=False,
            ),
        )
    )
    model.segments["RShank"].add_marker(Marker("RLEG1", is_technical=True, is_anatomical=False))
    model.segments["RShank"].add_marker(Marker("RLEG2", is_technical=True, is_anatomical=False))
    model.segments["RShank"].add_marker(Marker("RLEG3", is_technical=True, is_anatomical=False))
    model.segments["RShank"].add_marker(Marker("RLM", is_technical=False, is_anatomical=True))
    model.segments["RShank"].add_marker(Marker("RSPH", is_technical=False, is_anatomical=True))
    model.segments["RShank"].add_marker(Marker("RATT", is_technical=False, is_anatomical=True))

    _logger.info("Collapsing the model to real...")
    model_real = model.to_real(static_data)

    if visualize:
        model_real.animate()

    return model_real


if __name__ == "__main__":
    generate_lower_body_model()
