from copy import deepcopy
from typing import Callable

import numpy as np

from ...components.real.biomechanical_model_real import BiomechanicalModelReal
from ...components.real.rigidbody.segment_real import (
    SegmentReal,
    InertialMeasurementUnitReal,
    InertiaParametersReal,
    MeshReal,
    SegmentCoordinateSystemReal,
    MarkerReal,
)
from ...components.real.muscle.muscle_real import MuscleReal, MuscleType, MuscleStateType
from ...components.generic.muscle.muscle_group import MuscleGroup
from ...components.real.muscle.via_point_real import ViaPointReal
from ...components.real.rigidbody.segment_scaling import SegmentScaling
from ...utils.named_list import NamedList
from .utils import (
    tokenize_biomod,
    check_if_version_defined,
    read_str,
    read_int,
    read_float,
    read_bool,
    read_float_vector,
)


class EndOfFileReached(Exception):
    pass


class BiomodModelParser:
    def __init__(self, filepath: str):
        tokens = tokenize_biomod(filepath=filepath)

        # Prepare the internal structure to hold the model
        self.gravity = None
        self.segments = NamedList[SegmentReal]()
        self.muscle_groups = NamedList[MuscleGroup]()
        self.muscles = NamedList[MuscleReal]()
        self.via_points = NamedList[ViaPointReal]()
        self.warnings = ""

        def next_token():
            nonlocal token_index
            token_index += 1
            if token_index >= len(tokens):
                raise EndOfFileReached()
            return tokens[token_index]

        # Parse the model
        biorbd_version = None
        gravity = None
        current_component = None
        token_index = -1
        try:
            while True:
                token = read_str(next_token=next_token).lower()

                if current_component is None:
                    if token == "version":
                        if biorbd_version is not None:
                            raise ValueError("Version already defined")
                        biomod_version = read_int(next_token=next_token)
                        # True for version 3 or less, False for version 4 or more
                        rt_in_matrix_default = biomod_version < 4
                    elif token == "gravity":
                        check_if_version_defined(biomod_version)
                        if gravity is not None:
                            raise ValueError("Gravity already defined")
                        gravity = read_float_vector(next_token=next_token, length=3)
                    elif token == "segment":
                        check_if_version_defined(biomod_version)
                        current_component = SegmentReal(name=read_str(next_token=next_token))
                        current_rt_in_matrix = rt_in_matrix_default
                    elif token == "imu":
                        check_if_version_defined(biomod_version)
                        current_component = InertialMeasurementUnitReal(
                            name=read_str(next_token=next_token), parent_name=""
                        )
                        current_rt_in_matrix = rt_in_matrix_default
                    elif token == "marker":
                        check_if_version_defined(biomod_version)
                        current_component = MarkerReal(name=read_str(next_token=next_token), parent_name="")
                    elif token == "musclegroup":
                        check_if_version_defined(biomod_version)
                        current_component = MuscleGroup(
                            name=read_str(next_token=next_token), origin_parent_name="", insertion_parent_name=""
                        )
                    elif token == "muscle":
                        check_if_version_defined(biomod_version)
                        current_component = MuscleReal(
                            name=read_str(next_token=next_token),
                            muscle_type=MuscleType.HILL_DE_GROOTE,
                            state_type=MuscleStateType.DEGROOTE,
                            muscle_group="",
                            origin_position=None,
                            insertion_position=None,
                            optimal_length=None,
                            maximal_force=None,
                            tendon_slack_length=None,
                            pennation_angle=None,
                            maximal_excitation=None,
                        )
                    elif token == "viapoint":
                        check_if_version_defined(biomod_version)
                        current_component = ViaPointReal(
                            name=read_str(next_token=next_token),
                            parent_name="",
                            muscle_name="",
                            muscle_group="",
                            position=None,
                        )
                    elif token in ["mass", "scalingsegment"]:
                        continue
                    else:
                        raise ValueError(f"Unknown component {token}")

                elif isinstance(current_component, SegmentReal):
                    if token == "endsegment":
                        self.segments.append(current_component)
                        current_component = None
                    elif token == "parent":
                        current_component.parent_name = read_str(next_token=next_token)
                    elif token == "rtinmatrix":
                        current_rt_in_matrix = read_bool(next_token=next_token)
                    elif token == "rt":
                        scs = _get_rt_matrix(next_token=next_token, current_rt_in_matrix=current_rt_in_matrix)
                        current_component.segment_coordinate_system = SegmentCoordinateSystemReal(
                            scs=scs, is_scs_local=True
                        )
                    elif token == "translations":
                        current_component.translations = read_str(next_token=next_token)
                    elif token == "rotations":
                        current_component.rotations = read_str(next_token=next_token)
                    elif token in ("mass", "com", "centerofmass", "inertia", "inertia_xxyyzz"):
                        if current_component.inertia_parameters is None:
                            current_component.inertia_parameters = InertiaParametersReal()

                        if token == "mass":
                            current_component.inertia_parameters.mass = read_float(next_token=next_token)
                        elif token == "com" or token == "centerofmass":
                            com = read_float_vector(next_token=next_token, length=3)
                            current_component.inertia_parameters.center_of_mass = com
                        elif token == "inertia":
                            inertia = read_float_vector(next_token=next_token, length=9).reshape((3, 3))
                            current_component.inertia_parameters.inertia = inertia
                        elif token == "inertia_xxyyzz":
                            inertia = read_float_vector(next_token=next_token, length=3)
                            current_component.inertia_parameters.inertia = np.diag(inertia)
                    elif token == "mesh":
                        if current_component.mesh is None:
                            current_component.mesh = MeshReal()
                        position = read_float_vector(next_token=next_token, length=3).T
                        current_component.mesh.add_positions(position)
                    elif token == "mesh_file":
                        raise NotImplementedError()
                    else:
                        raise ValueError(f"Unknown information in segment")

                elif isinstance(current_component, InertialMeasurementUnitReal):
                    if token == "endimu":
                        if not current_component.parent_name:
                            raise ValueError(f"Parent name not found in imu {current_component.name}")
                        self.segments[current_component.parent_name].imus.append(current_component)
                        current_component = None
                    elif token == "parent":
                        current_component.parent_name = read_str(next_token=next_token)
                    elif token == "rtinmatrix":
                        current_rt_in_matrix = read_bool(next_token=next_token)
                    elif token == "rt":
                        scs = _get_rt_matrix(next_token=next_token, current_rt_in_matrix=current_rt_in_matrix)
                        current_component.scs = scs
                    elif token == "technical":
                        current_component.is_technical = read_bool(next_token=next_token)
                    elif token == "anatomical":
                        current_component.is_anatomical = read_bool(next_token=next_token)

                elif isinstance(current_component, MarkerReal):
                    if token == "endmarker":
                        if not current_component.parent_name:
                            raise ValueError(f"Parent name not found in marker {current_component.name}")
                        self.segments[current_component.parent_name].markers.append(current_component)
                        current_component = None
                    elif token == "parent":
                        current_component.parent_name = read_str(next_token=next_token)
                    elif token == "position":
                        current_component.position = read_float_vector(next_token=next_token, length=3)
                    elif token == "technical":
                        current_component.is_technical = read_bool(next_token=next_token)
                    elif token == "anatomical":
                        current_component.is_anatomical = read_bool(next_token=next_token)

                elif isinstance(current_component, MuscleGroup):
                    if token == "endmusclegroup":
                        if not current_component.insertion_parent_name:
                            raise ValueError(f"Insertion parent name not found in musclegroup {current_component.name}")
                        if not current_component.origin_parent_name:
                            raise ValueError(f"Origin parent name not found in musclegroup {current_component.name}")
                        self.muscle_groups.append(current_component)
                        current_component = None
                    elif token == "insertionparent":
                        current_component.insertion_parent_name = read_str(next_token=next_token)
                    elif token == "originparent":
                        current_component.origin_parent_name = read_str(next_token=next_token)

                elif isinstance(current_component, MuscleReal):
                    if token == "endmuscle":
                        if not current_component.muscle_type:
                            raise ValueError(f"Muscle type not found in muscle {current_component.name}")
                        if not current_component.state_type:
                            raise ValueError(f"Muscle state type not found in muscle {current_component.name}")
                        if not current_component.muscle_group:
                            raise ValueError(f"Muscle group not found in muscle {current_component.name}")
                        if current_component.origin_position is None:
                            raise ValueError(f"Origin position not found in muscle {current_component.name}")
                        if current_component.insertion_position is None:
                            raise ValueError(f"Insertion position not found in muscle {current_component.name}")
                        if current_component.optimal_length is None:
                            raise ValueError(f"Optimal length not found in muscle {current_component.name}")
                        if current_component.maximal_force is None:
                            raise ValueError(f"Maximal force not found in muscle {current_component.name}")
                        if current_component.tendon_slack_length is None:
                            raise ValueError(f"Tendon slack length not found in muscle {current_component.name}")
                        if current_component.pennation_angle is None:
                            raise ValueError(f"Pennation angle not found in muscle {current_component.name}")
                        self.muscles.append(current_component)
                        current_component = None
                    elif token == "type":
                        current_component.muscle_type = MuscleType(read_str(next_token=next_token))
                    elif token == "statetype":
                        current_component.state_type = MuscleStateType(read_str(next_token=next_token))
                    elif token == "musclegroup":
                        current_component.muscle_group = read_str(next_token=next_token)
                    elif token == "originposition":
                        current_component.origin_position = read_float_vector(next_token=next_token, length=3)
                    elif token == "insertionposition":
                        current_component.insertion_position = read_float_vector(next_token=next_token, length=3)
                    elif token == "optimallength":
                        current_component.optimal_length = read_float(next_token=next_token)
                    elif token == "maximalforce":
                        current_component.maximal_force = read_float(next_token=next_token)
                    elif token == "tendonslacklength":
                        current_component.tendon_slack_length = read_float(next_token=next_token)
                    elif token == "pennationangle":
                        current_component.pennation_angle = read_float(next_token=next_token)
                    elif token == "maximal_excitation":
                        current_component.maximal_excitation = read_float(next_token=next_token)

                elif isinstance(current_component, ViaPointReal):
                    if token == "endviapoint":
                        if not current_component.parent_name:
                            raise ValueError(f"Parent name not found in via point {current_component.name}")
                        if not current_component.muscle_name:
                            raise ValueError(f"Muscle name type not found in via point {current_component.name}")
                        if not current_component.muscle_group:
                            raise ValueError(f"Muscle group not found in muscle {current_component.name}")
                        self.via_points.append(current_component)
                        current_component = None
                    elif token == "parent":
                        current_component.parent_name = read_str(next_token=next_token)
                    elif token == "muscle":
                        current_component.muscle_name = read_str(next_token=next_token)
                    elif token == "musclegroup":
                        current_component.muscle_group = read_str(next_token=next_token)
                    elif token == "position":
                        current_component.position = read_float_vector(next_token=next_token, length=3)

                elif isinstance(current_component, SegmentScaling):
                    # Segment scaling is read by biomod_configuration_parser
                    continue

                else:
                    raise ValueError(f"Unknown component {type(current_component)}")
        except EndOfFileReached:
            pass

    def to_real(self) -> BiomechanicalModelReal:
        model = BiomechanicalModelReal(gravity=self.gravity)

        # Add the segments
        for segment in self.segments:
            model.add_segment(deepcopy(segment))

        # Add the muscle groups
        for muscle_group in self.muscle_groups:
            model.add_muscle_group(deepcopy(muscle_group))

        # Add the muscles
        for muscle in self.muscles:
            model.add_muscle(deepcopy(muscle))

        # Add the via points
        for via_point in self.via_points:
            model.add_via_point(deepcopy(via_point))

        model.warnings = self.warnings

        return model


def _get_rt_matrix(next_token: Callable, current_rt_in_matrix: bool) -> np.ndarray:
    if current_rt_in_matrix:
        scs = SegmentCoordinateSystemReal.from_rt_matrix(
            rt_matrix=read_float_vector(next_token=next_token, length=16).reshape((4, 4)),
            is_scs_local=True,
        )
    else:
        angles = read_float_vector(next_token=next_token, length=3)
        angle_sequence = read_str(next_token=next_token)
        translations = read_float_vector(next_token=next_token, length=3)
        scs = SegmentCoordinateSystemReal.from_euler_and_translation(
            angles=angles, angle_sequence=angle_sequence, translations=translations, is_scs_local=True
        )
    return scs.scs[:, :, 0]
