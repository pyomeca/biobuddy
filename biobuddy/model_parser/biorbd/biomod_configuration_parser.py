from copy import deepcopy
from typing import Callable

import numpy as np

from .utils import _tokenize_biomod, _read_str, _read_int, _read_float, _read_bool, _read_float_vector
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
from ...utils.named_list import NamedList


class EndOfFileReached(Exception):
    pass


class BiomodConfigurationParser:
    def __init__(self, filepath: str):
        tokens = _tokenize_biomod(filepath=filepath)

        # Prepare the internal structure to hold the model
        self.segment_scaling_config = NamedList[SegmentScalingConfig]()

        def next_token():
            nonlocal token_index
            token_index += 1
            if token_index >= len(tokens):
                raise EndOfFileReached()
            return tokens[token_index]

        # Parse the model
        current_component = None
        token_index = -1
        try:
            while True:
                token = _read_str(next_token=next_token).lower()

                if current_component is None:
                    if token == "scalingsegment":
                        current_component = SegmentScalingConfig(name=_read_str(next_token=next_token))
                    else:
                        raise ValueError(f"Unknown component {token}")

                elif isinstance(current_component, SegmentScalingConfiguration):
                    if token == "endscalingsegment":
                        self.segment_scaling_config.append(current_component)
                        current_component = None
                    elif token == "xmarkers":
                        current_component.x_marker_names.append(next_token=next_token)
                    else:
                        raise ValueError(f"Unknown information in segmentscaling")

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
