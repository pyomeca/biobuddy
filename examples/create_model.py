"""
Example taken from biorbd.model_creation
# TODO: This example should be replace with an actual biomechanics example.
"""

import os
from pathlib import Path

import numpy as np
import biorbd
from biorbd.model_creation import (
    Axis,
    BiomechanicalModel,
    BiomechanicalModelReal,
    C3dData,
    Marker,
    MarkerReal,
    Mesh,
    MeshReal,
    MeshFile,
    Segment,
    SegmentReal,
    SegmentCoordinateSystemReal,
    SegmentCoordinateSystem,
    Contact,
    MuscleGroup,
    Muscle,
    MuscleType,
    MuscleStateType,
    Translations,
    Rotations,
    RangeOfMotion,
    Ranges,
    ViaPoint,
)
import ezc3d

from de_leva import DeLevaTable


def complex_model_from_scratch(mesh_path, remove_temporary: bool = True):
    """
    We define a new model by feeding in the actual dimension and position of the model.
    Please note that this model is not a human, it is only used to show the functionalities of the model creation module.
    """

    kinematic_model_file_path = "temporary_complex.bioMod"

    # Create a model holder
    bio_model = BiomechanicalModel()

    # The ground segment
    bio_model.segments["GROUND"] = Segment(name="GROUND")

    # The pendulum segment
    bio_model.segments["PENDULUM"] = Segment(
        name="PENDULUM",
        translations=Translations.XYZ,
        rotations=Rotations.X,
        q_ranges=RangeOfMotion(range_type=Ranges.Q, min_bound=[-1, -1, -1, -np.pi], max_bound=[1, 1, 1, np.pi]),
        qdot_ranges=RangeOfMotion(range_type=Ranges.Qdot, min_bound=[-10, -10, -10, -np.pi*10], max_bound=[10, 10, 10, np.pi*10]),
        mesh_file=MeshFile(mesh_file_name=mesh_path,
                            mesh_color=np.array([0, 0, 1]),
                            scaling_function=lambda m: np.array([1, 1, 10]),
                            rotation_function=lambda m: np.array([np.pi/2, 0, 0]),
                            translation_function=lambda m: np.array([0.1, 0, 0])),
    )
    # The pendulum segment contact point
    bio_model.segments["PENDULUM"].add_contact(Contact(name="PENDULUM_CONTACT",
                                                        function=lambda m: np.array([0, 0, 0]),
                                                        parent_name="PENDULUM",
                                                        axis=Translations.XYZ))

    # The pendulum muscle group
    bio_model.muscle_groups["PENDULUM_MUSCLE_GROUP"] = MuscleGroup(name="PENDULUM_MUSCLE_GROUP",
                                                                    origin_parent_name="GROUND",
                                                                    insertion_parent_name="PENDULUM")

    # The pendulum muscle
    bio_model.muscles["PENDULUM_MUSCLE"] = Muscle("PENDULUM_MUSCLE",
                                                muscle_type=MuscleType.HILLTHELEN,
                                                state_type=MuscleStateType.DEGROOTE,
                                                muscle_group="PENDULUM_MUSCLE_GROUP",
                                                origin_position_function=lambda m: np.array([0, 0, 0]),
                                                insertion_position_function=lambda m: np.array([0, 0, 1]),
                                                optimal_length_function=lambda model, m: 0.1,
                                                maximal_force_function=lambda m: 100.0,
                                                tendon_slack_length_function=lambda model, m: 0.05,
                                                pennation_angle_function=lambda model, m: 0.05,
                                                maximal_excitation=1)
    bio_model.via_points["PENDULUM_MUSCLE"] = ViaPoint("PENDULUM_MUSCLE",
                                                        position_function=lambda m: np.array([0, 0, 0.5]),
                                                        parent_name="PENDULUM",
                                                        muscle_name="PENDULUM_MUSCLE",
                                                        muscle_group="PENDULUM_MUSCLE_GROUP",
                                                        )


    # Put the model together, print it and print it to a bioMod file
    bio_model.write(kinematic_model_file_path, {})

    model = biorbd.Model(kinematic_model_file_path)
    assert model.nbQ() == 4
    assert model.nbSegment() == 2
    assert model.nbMarkers() == 0
    assert model.nbMuscles() == 1
    assert model.nbMuscleGroups() == 1
    assert model.nbContacts() == 3

    if remove_temporary:
        os.remove(kinematic_model_file_path)


if __name__ == "__main__":
    current_path_file = Path(__file__).parent
    complex_model_from_scratch(mesh_path=f"{current_path_file}/meshFiles/pendulum.STL")
