"""
This example shows how to create a personalized kinematic model from a C3D file containing a static trial.
Here, we generate a simple lower-body model with only a trunk segment.
The marker position and names are taken from Maldonado & al., 2018 (https://hal.science/hal-01841355/)
"""

import os

import numpy as np
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
    Sex,
    SegmentName,
    ViewAs,
    SegmentCoordinateSystemUtils,
    RotoTransMatrix,
)


def model_creation_from_measured_data(static_trial: C3dData,
                                    remove_temporary: bool = True,
                                      animate_model: bool = True):

    total_mass = 66
    total_height = 1.70

    output_model_filepath = f"lower_body.bioMod"
    de_leva = DeLevaTable(total_mass=total_mass, sex=Sex.FEMALE)
    de_leva.from_measurements(
        total_height=total_height,
        ankle_height=SegmentCoordinateSystemUtils.mean_markers(["RSPH", "RLM", "LLM", "LSPH"])(
            static_trial.values, None
        )[2],
        knee_height=SegmentCoordinateSystemUtils.mean_markers(["RLFE", "RMFE", "LLFE", "LMFE"])(
            static_trial.values, None
        )[2],
        pelvis_height=SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "RPSIS", "LASIS", "RASIS"])(
            static_trial.values, None
        )[2],
        shoulder_height=SegmentCoordinateSystemUtils.mean_markers(["LA", "RA"])(static_trial.values, None)[2],
        finger_span=total_height,
        wrist_span=total_height * 0.9,  # TODO: find data from literature for these % to set default values
        elbow_span=total_height * 0.5,
        shoulder_span=total_height * 0.2,
        foot_length=total_height * 0.2,
    )

    # Generate the personalized kinematic model
    reduced_model = BiomechanicalModel()

    reduced_model.add_segment(Segment(name="Ground"))

    reduced_model.add_segment(
        Segment(
            name="Pelvis",
            parent_name="Ground",
            translations=Translations.XYZ,
            rotations=Rotations.XYZ,
            inertia_parameters=de_leva[SegmentName.TRUNK],  # TODO: merge
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "RPSIS", "LASIS", "RASIS"]),
                first_axis=Axis(
                    name=Axis.Name.X,
                    start=SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "LASIS"]),
                    end=SegmentCoordinateSystemUtils.mean_markers(["RPSIS", "RASIS"]),
                ),
                second_axis=Axis(name=Axis.Name.Z),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(("LPSIS", "RPSIS", "RASIS", "LASIS", "LPSIS")),
        )
    )
    reduced_model.segments["Pelvis"].add_marker(Marker("LPSIS", is_technical=True, is_anatomical=True))
    reduced_model.segments["Pelvis"].add_marker(Marker("RPSIS", is_technical=True, is_anatomical=True))
    reduced_model.segments["Pelvis"].add_marker(Marker("LASIS", is_technical=True, is_anatomical=True))
    reduced_model.segments["Pelvis"].add_marker(Marker("RASIS", is_technical=True, is_anatomical=True))
    reduced_model.segments["Pelvis"].add_marker(Marker("RA", is_technical=True, is_anatomical=True))
    reduced_model.segments["Pelvis"].add_marker(Marker("LA", is_technical=True, is_anatomical=True))

    reduced_model.add_segment(
        Segment(
            name="RFemur",
            parent_name="Pelvis",
            rotations=Rotations.XY,
            inertia_parameters=de_leva[SegmentName.THIGH],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: SegmentCoordinateSystemUtils.mean_markers(["RPSIS", "RASIS"])(
                    static_trial.values, None
                )
                - np.array([0.0, 0.0, 0.05 * total_height, 0.0]),
                first_axis=Axis(name=Axis.Name.X, start="RMFE", end="RLFE"),
                second_axis=Axis(
                    name=Axis.Name.Z,
                    start=SegmentCoordinateSystemUtils.mean_markers(["RMFE", "RLFE"]),
                    end=SegmentCoordinateSystemUtils.mean_markers(["RPSIS", "RASIS"]),
                ),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: SegmentCoordinateSystemUtils.mean_markers(["RPSIS", "RASIS"])(
                        static_trial.values, None
                    )
                    - np.array([0.0, 0.0, 0.05 * total_height, 0.0]),
                    "RMFE",
                    "RLFE",
                    lambda m, bio: SegmentCoordinateSystemUtils.mean_markers(["RPSIS", "RASIS"])(
                        static_trial.values, None
                    )
                    - np.array([0.0, 0.0, 0.05 * total_height, 0.0]),
                )
            ),
        )
    )
    reduced_model.segments["RFemur"].add_marker(Marker("RLFE", is_technical=True, is_anatomical=True))
    reduced_model.segments["RFemur"].add_marker(Marker("RMFE", is_technical=True, is_anatomical=True))

    reduced_model.add_segment(
        Segment(
            name="RTibia",
            parent_name="RFemur",
            rotations=Rotations.X,
            inertia_parameters=de_leva[SegmentName.SHANK],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=SegmentCoordinateSystemUtils.mean_markers(["RMFE", "RLFE"]),
                first_axis=Axis(name=Axis.Name.X, start="RSPH", end="RLM"),
                second_axis=Axis(
                    name=Axis.Name.Z,
                    start=SegmentCoordinateSystemUtils.mean_markers(["RSPH", "RLM"]),
                    end=SegmentCoordinateSystemUtils.mean_markers(["RMFE", "RLFE"]),
                ),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(("RMFE", "RSPH", "RLM", "RLFE")),
        )
    )
    reduced_model.segments["RTibia"].add_marker(Marker("RLM", is_technical=True, is_anatomical=True))
    reduced_model.segments["RTibia"].add_marker(Marker("RSPH", is_technical=True, is_anatomical=True))

    # The foot is a special case since the position of the ankle relatively to the foot length is not given in De Leva
    # So here we assume that the foot com is in the middle of the three foot markers
    foot_inertia_parameters = de_leva[SegmentName.FOOT]
    rt_matrix = RotoTransMatrix()
    rt_matrix.from_euler_angles_and_translation(
        angle_sequence="y",
        angles=np.array([-np.pi/2]),
        translation=np.array([0., 0., 0.]),
    )
    foot_inertia_parameters.center_of_mass = lambda m, bio: rt_matrix.rt_matrix @ np.nanmean(
        np.nanmean(np.array([m[name] for name in ["LSPH", "LLM", "LTT2"]]), axis=0) -
        np.nanmean(np.array([m[name] for name in ["LSPH", "LLM"]]), axis=0),
        axis=1)

    reduced_model.add_segment(
        Segment(
            name="RFoot",
            parent_name="RTibia",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=SegmentCoordinateSystemUtils.mean_markers(["RSPH", "RLM"]),
                first_axis=Axis(
                    Axis.Name.Z, start=SegmentCoordinateSystemUtils.mean_markers(["RSPH", "RLM"]), end="RTT2"
                ),
                second_axis=Axis(Axis.Name.X, start="RSPH", end="RLM"),
                axis_to_keep=Axis.Name.Z,
            ),
            inertia_parameters=foot_inertia_parameters,
            mesh=Mesh(("RLM", "RTT2", "RSPH", "RLM")),
        )
    )
    reduced_model.segments["RFoot"].add_marker(Marker("RTT2", is_technical=True, is_anatomical=True))

    reduced_model.add_segment(
        Segment(
            name="LFemur",
            parent_name="Pelvis",
            rotations=Rotations.XY,
            inertia_parameters=de_leva[SegmentName.THIGH],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "LASIS"])(
                    static_trial.values, None
                )
                - np.array([0.0, 0.0, 0.05 * total_height, 0.0]),
                first_axis=Axis(name=Axis.Name.X, start="LLFE", end="LMFE"),
                second_axis=Axis(
                    name=Axis.Name.Z,
                    start=SegmentCoordinateSystemUtils.mean_markers(["LMFE", "LLFE"]),
                    end=SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "LASIS"]),
                ),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "LASIS"])(
                        static_trial.values, None
                    )
                    - np.array([0.0, 0.0, 0.05 * total_height, 0.0]),
                    "LMFE",
                    "LLFE",
                    lambda m, bio: SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "LASIS"])(
                        static_trial.values, None
                    )
                    - np.array([0.0, 0.0, 0.05 * total_height, 0.0]),
                )
            ),
        )
    )
    reduced_model.segments["LFemur"].add_marker(Marker("LLFE", is_technical=True, is_anatomical=True))
    reduced_model.segments["LFemur"].add_marker(Marker("LMFE", is_technical=True, is_anatomical=True))

    reduced_model.add_segment(
        Segment(
            name="LTibia",
            parent_name="LFemur",
            rotations=Rotations.X,
            inertia_parameters=de_leva[SegmentName.SHANK],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=SegmentCoordinateSystemUtils.mean_markers(["LMFE", "LLFE"]),
                first_axis=Axis(name=Axis.Name.X, start="LLM", end="LSPH"),
                second_axis=Axis(
                    name=Axis.Name.Z,
                    start=SegmentCoordinateSystemUtils.mean_markers(["LSPH", "LLM"]),
                    end=SegmentCoordinateSystemUtils.mean_markers(["LMFE", "LLFE"]),
                ),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(("LMFE", "LSPH", "LLM", "LLFE")),
        )
    )
    reduced_model.segments["LTibia"].add_marker(Marker("LLM", is_technical=True, is_anatomical=True))
    reduced_model.segments["LTibia"].add_marker(Marker("LSPH", is_technical=True, is_anatomical=True))

    foot_inertia_parameters = de_leva[SegmentName.FOOT]
    rt_matrix = RotoTransMatrix()
    rt_matrix.from_euler_angles_and_translation(
        angle_sequence="y",
        angles=np.array([-np.pi/2]),
        translation=np.array([0., 0., 0.]),
    )
    foot_inertia_parameters.center_of_mass = lambda m, bio: rt_matrix.rt_matrix @ np.nanmean(
        np.nanmean(np.array([m[name] for name in ["LSPH", "LLM", "LTT2"]]), axis=0) -
        np.nanmean(np.array([m[name] for name in ["LSPH", "LLM"]]), axis=0),
        axis=1)

    reduced_model.add_segment(
        Segment(
            name="LFoot",
            parent_name="LTibia",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=SegmentCoordinateSystemUtils.mean_markers(["LSPH", "LLM"]),
                first_axis=Axis(
                    Axis.Name.Z, start=SegmentCoordinateSystemUtils.mean_markers(["LLM", "LSPH"]), end="LTT2"
                ),
                second_axis=Axis(Axis.Name.X, start="LLM", end="LSPH"),
                axis_to_keep=Axis.Name.Z,
            ),
            inertia_parameters=foot_inertia_parameters,
            mesh=Mesh(("LLM", "LTT2", "LSPH", "LLM")),
        )
    )
    reduced_model.segments["LFoot"].add_marker(Marker("LTT2", is_technical=True, is_anatomical=True))

    # Put the model together, print it and print it to a bioMod file
    model_real = reduced_model.to_real(static_trial)
    model_real.to_biomod(output_model_filepath)

    if animate_model:
        model_real.animate(view_as=ViewAs.BIORBD, model_path=output_model_filepath)

    if remove_temporary:
        os.remove(output_model_filepath)

    return model_real


def main():

    # Load the static trial
    static_trial = C3dData(f"data/static_lower_body.c3d")

    model_creation_from_measured_data(static_trial)


if __name__ == "__main__":
    main()
