from copy import deepcopy
from typing import Callable

import numpy as np

from ..abstract_model_parser import AbstractModelParser
from ... import MeshFileReal
from ...components.real.biomechanical_model_real import BiomechanicalModelReal
from ...components.real.rigidbody.segment_real import (
    SegmentReal,
    InertialMeasurementUnitReal,
    InertiaParametersReal,
    MeshReal,
    SegmentCoordinateSystemReal,
    MarkerReal,
    ContactReal,
)
from ...components.real.muscle.muscle_real import MuscleReal
from ...components.muscle_utils import MuscleType, MuscleStateType
from ...components.real.muscle.muscle_group_real import MuscleGroupReal
from ...components.generic.rigidbody.range_of_motion import Ranges, RangeOfMotion
from ...components.real.muscle.via_point_real import ViaPointReal
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
from ...utils.enums import Translations


# TODO: when we update to biorbd=1.12.0, we need to parse the mesh file dir and ubdate the mesh_file_directory
#  in MeshFileReal


TOKENS_TO_IGNORE_NO_COMPONENTS = ["endscalingsegment"]
TOKENS_TO_IGNORE_ONE_COMPONENTS = ["scalingsegment", "scalingtype", "axis"]
TOKENS_TO_IGNORE_TWO_COMPONENTS = ["markerpair", "xmarkerpair", "ymarkerpair", "zmarkerpair", "markerweight"]


class EndOfFileReached(Exception):
    pass


class BiomodModelParser(AbstractModelParser):
    def __init__(self, filepath: str):

        super().__init__(filepath)

        tokens = tokenize_biomod(filepath=filepath)

        # Prepare the internal structure to hold the model
        self.gravity = None
        self.segments = NamedList[SegmentReal]()
        self.muscle_groups = NamedList[MuscleGroupReal]()
        self.warnings = ""

        def next_token():
            nonlocal token_index
            token_index += 1
            if token_index >= len(tokens):
                raise EndOfFileReached()
            return tokens[token_index]

        def nb_float_tokens_until_next_str() -> int:
            """
            Count the number of float tokens until the next str token.
            """
            nonlocal token_index
            count = 1
            while True:
                try:
                    float(tokens[token_index + count])
                except ValueError:
                    break
                count += 1
            return count - 1

        # Parse the model
        biorbd_version = None
        gravity = None
        current_component = None
        token_index = -1
        try:
            while True:
                token = read_str(next_token=next_token)

                if current_component is None:
                    if token.lower() == "version":
                        if biorbd_version is not None:
                            raise ValueError("Version already defined")
                        biomod_version = read_int(next_token=next_token)
                        # True for version 3 or less, False for version 4 or more
                        rt_in_matrix_default = biomod_version < 4
                    elif token.lower() == "gravity":
                        check_if_version_defined(biomod_version)
                        if gravity is not None:
                            raise ValueError("Gravity already defined")
                        self.gravity = read_float_vector(next_token=next_token, length=3)
                    elif token.lower() == "segment":
                        check_if_version_defined(biomod_version)
                        current_component = SegmentReal(name=read_str(next_token=next_token))
                        current_rt_in_matrix = rt_in_matrix_default
                    elif token.lower() == "imu":
                        check_if_version_defined(biomod_version)
                        current_component = InertialMeasurementUnitReal(
                            name=read_str(next_token=next_token), parent_name=""
                        )
                        current_rt_in_matrix = rt_in_matrix_default
                    elif token.lower() == "marker":
                        check_if_version_defined(biomod_version)
                        current_component = MarkerReal(name=read_str(next_token=next_token), parent_name="")
                    elif token.lower() == "contact":
                        check_if_version_defined(biomod_version)
                        current_component = ContactReal(name=read_str(next_token=next_token), parent_name="")
                    elif token.lower() == "musclegroup":
                        check_if_version_defined(biomod_version)
                        current_component = MuscleGroupReal(
                            name=read_str(next_token=next_token), origin_parent_name="", insertion_parent_name=""
                        )
                    elif token.lower() == "muscle":
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
                    elif token.lower() == "viapoint":
                        check_if_version_defined(biomod_version)
                        current_component = ViaPointReal(
                            name=read_str(next_token=next_token),
                            parent_name="",
                            muscle_name="",
                            muscle_group="",
                            position=None,
                        )
                    elif token in TOKENS_TO_IGNORE_NO_COMPONENTS:
                        continue
                    elif token in TOKENS_TO_IGNORE_ONE_COMPONENTS:
                        token_index += 1
                    elif token in TOKENS_TO_IGNORE_TWO_COMPONENTS:
                        token_index += 2
                    else:
                        raise ValueError(f"Unknown component {token}")

                elif isinstance(current_component, SegmentReal):
                    if token.lower() == "endsegment":
                        current_component.update_dof_names()
                        self.segments.append(current_component)
                        current_component = None
                    elif token.lower() == "parent":
                        current_component.parent_name = read_str(next_token=next_token)
                    elif token.lower() == "rtinmatrix":
                        current_rt_in_matrix = read_bool(next_token=next_token)
                    elif token.lower() == "rt":
                        scs = _get_rt_matrix(next_token=next_token, current_rt_in_matrix=current_rt_in_matrix)
                        current_component.segment_coordinate_system = SegmentCoordinateSystemReal(
                            scs=scs, is_scs_local=True
                        )
                    elif token.lower() == "translations":
                        current_component.translations = read_str(next_token=next_token)
                    elif token.lower() == "rotations":
                        current_component.rotations = read_str(next_token=next_token)
                    elif token.lower() == "rangesq" or token.lower() == "ranges" or token.lower() == "rangesqdot":
                        length = nb_float_tokens_until_next_str()
                        if length % 2 != 0:
                            raise ValueError(f"Length of range_q is not even: {length}")
                        min_max = read_float_vector(next_token=next_token, length=length)
                        min_bound = min_max[0::2]
                        max_bound = min_max[1::2]
                        if token.lower() == "rangesq" or token.lower() == "ranges":
                            current_component.q_ranges = RangeOfMotion(
                                range_type=Ranges.Q, min_bound=min_bound, max_bound=max_bound
                            )
                        else:
                            current_component.qdot_ranges = RangeOfMotion(
                                range_type=Ranges.Qdot, min_bound=min_bound, max_bound=max_bound
                            )
                    elif token.lower() in ("mass", "com", "centerofmass", "inertia", "inertia_xxyyzz"):
                        if current_component.inertia_parameters is None:
                            current_component.inertia_parameters = InertiaParametersReal()

                        if token.lower() == "mass":
                            current_component.inertia_parameters.mass = read_float(next_token=next_token)
                        elif token.lower() == "com" or token.lower() == "centerofmass":
                            com = read_float_vector(next_token=next_token, length=3)
                            current_component.inertia_parameters.center_of_mass = com
                        elif token.lower() == "inertia":
                            inertia = read_float_vector(next_token=next_token, length=9).reshape((3, 3))
                            current_component.inertia_parameters.inertia = inertia
                        elif token.lower() == "inertia_xxyyzz":
                            inertia = read_float_vector(next_token=next_token, length=3)
                            current_component.inertia_parameters.inertia = np.diag(inertia)
                    elif token.lower() == "mesh":
                        if current_component.mesh is None:
                            current_component.mesh = MeshReal()
                        position = read_float_vector(next_token=next_token, length=3).T
                        current_component.mesh.add_positions(position)
                    elif token.lower() == "meshfile":
                        mesh_file = read_str(next_token=next_token)
                        if current_component.mesh_file is not None:
                            raise RuntimeError(
                                f"The mesh file {mesh_file} is the second mesh defined for this segment."
                            )
                        split_name = mesh_file.split("/")
                        mesh_file_name = split_name[-1]
                        if len(split_name) > 1:
                            mesh_file_directory = "/".join(split_name[0:-1])
                        else:
                            mesh_file_directory = "."
                        current_component.mesh_file = MeshFileReal(
                            mesh_file_name=mesh_file_name, mesh_file_directory=mesh_file_directory
                        )
                    elif token.lower() == "meshcolor":
                        if current_component.mesh_file is None:
                            raise RuntimeError("The mesh file must be defined before the mesh color.")
                        current_component.mesh_file.mesh_color = read_float_vector(next_token=next_token, length=3)
                    elif token.lower() == "meshscale":
                        if current_component.mesh_file is None:
                            raise RuntimeError("The mesh file must be defined before the mesh scale.")
                        current_component.mesh_file.mesh_scale = read_float_vector(next_token=next_token, length=3)
                    elif token.lower() == "meshrt":
                        if current_component.mesh_file is None:
                            raise RuntimeError("The mesh file must be defined before the mesh rt.")
                        angles = read_float_vector(next_token=next_token, length=3)
                        angle_sequence = read_str(next_token=next_token)
                        translations = read_float_vector(next_token=next_token, length=3)
                        current_component.mesh_file.mesh_rotation = angles
                        current_component.mesh_file.mesh_translation = translations
                    else:
                        raise ValueError(f"Unknown information in segment: {token.lower()}")

                elif isinstance(current_component, InertialMeasurementUnitReal):
                    if token.lower() == "endimu":
                        if not current_component.parent_name:
                            raise ValueError(f"Parent name not found in imu {current_component.name}")
                        self.segments[current_component.parent_name].imus.append(current_component)
                        current_component = None
                    elif token.lower() == "parent":
                        current_component.parent_name = read_str(next_token=next_token)
                    elif token.lower() == "rtinmatrix":
                        current_rt_in_matrix = read_bool(next_token=next_token)
                    elif token.lower() == "rt":
                        scs = _get_rt_matrix(next_token=next_token, current_rt_in_matrix=current_rt_in_matrix)
                        current_component.scs = scs
                    elif token.lower() == "technical":
                        current_component.is_technical = read_bool(next_token=next_token)
                    elif token.lower() == "anatomical":
                        current_component.is_anatomical = read_bool(next_token=next_token)

                elif isinstance(current_component, MarkerReal):
                    if token.lower() == "endmarker":
                        if not current_component.parent_name:
                            raise ValueError(f"Parent name not found in marker {current_component.name}")
                        self.segments[current_component.parent_name].markers.append(current_component)
                        current_component = None
                    elif token.lower() == "parent":
                        current_component.parent_name = read_str(next_token=next_token)
                    elif token.lower() == "position":
                        current_component.position = read_float_vector(next_token=next_token, length=3)
                    elif token.lower() == "technical":
                        current_component.is_technical = read_bool(next_token=next_token)
                    elif token.lower() == "anatomical":
                        current_component.is_anatomical = read_bool(next_token=next_token)

                elif isinstance(current_component, ContactReal):
                    if token.lower() == "endcontact":
                        if not current_component.parent_name:
                            raise ValueError(f"Parent name not found in contact {current_component.name}")
                        self.segments[current_component.parent_name].contacts.append(current_component)
                        current_component = None
                    elif token.lower() == "parent":
                        current_component.parent_name = read_str(next_token=next_token)
                    elif token.lower() == "position":
                        current_component.position = read_float_vector(next_token=next_token, length=3)
                    elif token.lower() == "axis":
                        current_component.axis = Translations(read_str(next_token=next_token))

                elif isinstance(current_component, MuscleGroupReal):
                    if token.lower() == "endmusclegroup":
                        if not current_component.insertion_parent_name:
                            raise ValueError(f"Insertion parent name not found in musclegroup {current_component.name}")
                        if not current_component.origin_parent_name:
                            raise ValueError(f"Origin parent name not found in musclegroup {current_component.name}")
                        self.muscle_groups.append(current_component)
                        current_component = None
                    elif token.lower() == "insertionparent":
                        current_component.insertion_parent_name = read_str(next_token=next_token)
                    elif token.lower() == "originparent":
                        current_component.origin_parent_name = read_str(next_token=next_token)

                elif isinstance(current_component, MuscleReal):
                    if token.lower() == "endmuscle":
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
                        self.muscle_groups[current_component.muscle_group].add_muscle(current_component)
                        current_component = None
                    elif token.lower() == "type":
                        current_component.muscle_type = MuscleType(read_str(next_token=next_token))
                    elif token.lower() == "statetype":
                        current_component.state_type = MuscleStateType(read_str(next_token=next_token))
                    elif token.lower() == "musclegroup":
                        current_component.muscle_group = read_str(next_token=next_token)
                    elif token.lower() == "originposition":
                        current_component.origin_position = ViaPointReal(
                            name=f"origin_{current_component.name}",
                            parent_name=self.muscle_groups[current_component.muscle_group].origin_parent_name,
                            muscle_name=current_component.name,
                            muscle_group=current_component.muscle_group,
                            position=read_float_vector(next_token=next_token, length=3),
                        )
                    elif token.lower() == "insertionposition":
                        current_component.insertion_position = ViaPointReal(
                            name=f"insertion_{current_component.name}",
                            parent_name=self.muscle_groups[current_component.muscle_group].insertion_parent_name,
                            muscle_name=current_component.name,
                            muscle_group=current_component.muscle_group,
                            position=read_float_vector(next_token=next_token, length=3),
                        )
                    elif token.lower() == "optimallength":
                        current_component.optimal_length = read_float(next_token=next_token)
                    elif token.lower() == "maximalforce":
                        current_component.maximal_force = read_float(next_token=next_token)
                    elif token.lower() == "tendonslacklength":
                        current_component.tendon_slack_length = read_float(next_token=next_token)
                    elif token.lower() == "pennationangle":
                        current_component.pennation_angle = read_float(next_token=next_token)
                    elif token.lower() == "maximal_excitation":
                        current_component.maximal_excitation = read_float(next_token=next_token)

                elif isinstance(current_component, ViaPointReal):
                    if token.lower() == "endviapoint":
                        if not current_component.parent_name:
                            raise ValueError(f"Parent name not found in via point {current_component.name}")
                        if not current_component.muscle_name:
                            raise ValueError(f"Muscle name type not found in via point {current_component.name}")
                        if not current_component.muscle_group:
                            raise ValueError(f"Muscle group not found in muscle {current_component.name}")
                        self.muscle_groups[current_component.muscle_group].muscles[
                            current_component.muscle_name
                        ].add_via_point(current_component)
                        current_component = None
                    elif token.lower() == "parent":
                        current_component.parent_name = read_str(next_token=next_token)
                    elif token.lower() == "muscle":
                        current_component.muscle_name = read_str(next_token=next_token)
                    elif token.lower() == "musclegroup":
                        current_component.muscle_group = read_str(next_token=next_token)
                    elif token.lower() == "position":
                        current_component.position = read_float_vector(next_token=next_token, length=3)
                else:
                    raise ValueError(f"Unknown component : {type(current_component)}")
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
            angles=angles, angle_sequence=angle_sequence, translation=translations, is_scs_local=True
        )
    return scs.scs
