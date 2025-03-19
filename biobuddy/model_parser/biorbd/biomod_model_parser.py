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
)
from ...utils.named_list import NamedList


class EndOfFileReached(Exception):
    pass


class BiomodModelParser:
    def __init__(self, filepath: str):
        tokens = _tokenize_biomod(filepath=filepath)

        # Prepare the internal structure to hold the model
        self.segments = NamedList[SegmentReal]()

        def next_token():
            nonlocal token_index
            token_index += 1
            if token_index >= len(tokens):
                raise EndOfFileReached()
            return tokens[token_index]

        # Parse the model
        biorbd_version = None
        current_component = None
        token_index = -1
        try:
            while True:
                token = _read_str(next_token=next_token).lower()

                if current_component is None:
                    if token == "version":
                        if biorbd_version is not None:
                            raise ValueError("Version already defined")
                        biomod_version = _read_int(next_token=next_token)
                        # True for version 3 or less, False for version 4 or more
                        rt_in_matrix_default = biomod_version < 4
                    elif token == "segment":
                        if biomod_version is None:
                            raise ValueError("Version not defined")
                        current_component = SegmentReal(name=_read_str(next_token=next_token))
                        current_rt_in_matrix = rt_in_matrix_default
                    elif token == "imu":
                        if biomod_version is None:
                            raise ValueError("Version not defined")
                        current_component = InertialMeasurementUnitReal(
                            name=_read_str(next_token=next_token), parent_name=""
                        )
                        current_rt_in_matrix = rt_in_matrix_default
                    else:
                        raise ValueError(f"Unknown component {token}")

                elif isinstance(current_component, SegmentReal):
                    if token == "endsegment":
                        self.segments.append(current_component)
                        current_component = None
                    elif token == "parent":
                        current_component.parent_name = _read_str(next_token=next_token)
                    elif token == "rtinmatrix":
                        current_rt_in_matrix = _read_bool(next_token=next_token)
                    elif token == "rt":
                        scs = _get_rt_matrix(next_token=next_token, current_rt_in_matrix=current_rt_in_matrix)
                        current_component.segment_coordinate_system = SegmentCoordinateSystemReal(
                            scs=scs, is_scs_local=True
                        )
                    elif token == "translations":
                        current_component.translations = _read_str(next_token=next_token)
                    elif token == "rotations":
                        current_component.rotations = _read_str(next_token=next_token)
                    elif token in ("mass", "com", "inertia"):
                        if current_component.inertia_parameters is None:
                            current_component.inertia_parameters = InertiaParametersReal()

                        if token == "mass":
                            current_component.inertia_parameters.mass = _read_float(next_token=next_token)
                        elif token == "com":
                            com = _read_float_vector(next_token=next_token, length=3)
                            current_component.inertia_parameters.center_of_mass = com
                        elif token == "inertia":
                            inertia = _read_float_vector(next_token=next_token, length=9).reshape((3, 3))
                            current_component.inertia_parameters.inertia = inertia
                    elif token == "mesh":
                        if current_component.mesh is None:
                            current_component.mesh = MeshReal()
                        position = _read_float_vector(next_token=next_token, length=3).T
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
                        current_component.parent_name = _read_str(next_token=next_token)
                    elif token == "rtinmatrix":
                        current_rt_in_matrix = _read_bool(next_token=next_token)
                    elif token == "rt":
                        scs = _get_rt_matrix(next_token=next_token, current_rt_in_matrix=current_rt_in_matrix)
                        current_component.scs = scs
                    elif token == "technical":
                        current_component.is_technical = _read_bool(next_token=next_token)
                    elif token == "anatomical":
                        current_component.is_anatomical = _read_bool(next_token=next_token)

                else:
                    raise ValueError(f"Unknown component {type(current_component)}")
        except EndOfFileReached:
            pass

    def to_real(self) -> BiomechanicalModelReal:
        model = BiomechanicalModelReal()

        # Add the segments
        for segment in self.segments:
            model.segments.append(deepcopy(segment))

        return model


def _tokenize_biomod(filepath: str) -> list[str]:
    # Load the model from the filepath
    with open(filepath) as f:
        content = f.read()
    lines = content.split("\n")

    # Do a first pass to remove every commented content
    is_block_commenting = False
    line_index = 0
    for line_index in range(len(lines)):
        line = lines[line_index]
        # Remove everything after // or between /* */ (comments)
        if "/*" in line and "*/" in line:
            # Deal with the case where the block comment is on the same line
            is_block_commenting = False
            line = (line.split("/*")[0] + "" + line.split("*/")[1]).strip()
        if not is_block_commenting and "/*" in line:
            is_block_commenting = True
            line = line.split("/*")[0]
        if is_block_commenting and "*/" in line:
            is_block_commenting = False
            line = line.split("*/")[1]
        line = line.split("//")[0]
        line = line.strip()
        lines[line_index] = line

    # Make spaces also a separator
    tokens = []
    for line in lines:
        tokens.extend(line.split(" "))

    # Remove empty lines
    return [token for token in tokens if token]


def _read_str(next_token: Callable) -> str:
    return next_token()


def _read_int(next_token: Callable) -> int:
    return int(next_token())


def _read_float(next_token: Callable) -> float:
    return float(next_token())


def _read_bool(next_token: Callable) -> bool:
    return next_token() == "1"


def _read_float_vector(next_token: Callable, length: int) -> np.ndarray:
    return np.array([_read_float(next_token=next_token) for _ in range(length)])


def _get_rt_matrix(next_token: Callable, current_rt_in_matrix: bool) -> np.ndarray:
    if current_rt_in_matrix:
        scs = SegmentCoordinateSystemReal.from_rt_matrix(
            rt_matrix=_read_float_vector(next_token=next_token, length=16).reshape((4, 4))
        )
    else:
        angles = _read_float_vector(next_token=next_token, length=3)
        angle_sequence = _read_str(next_token=next_token)
        translations = _read_float_vector(next_token=next_token, length=3)
        scs = SegmentCoordinateSystemReal.from_euler_and_translation(
            angles=angles, angle_sequence=angle_sequence, translations=translations
        )
    return scs.scs[:, :, 0]
