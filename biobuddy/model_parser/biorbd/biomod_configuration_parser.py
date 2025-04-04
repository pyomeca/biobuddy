from time import strftime

from .utils import tokenize_biomod, read_str, read_str_list, read_float
from ...components.real.rigidbody.segment_scaling import (
    SegmentScaling,
    SegmentWiseScaling,
    AxisWiseScaling,
)
from ...model_modifiers.scale_tool import ScaleTool
from ...components.real.rigidbody.segment_real import (
    SegmentReal,
    InertialMeasurementUnitReal,
    MarkerReal,
)
from ...components.real.muscle.muscle_real import MuscleReal
from ...components.generic.muscle.muscle_group import MuscleGroup
from ...components.real.muscle.via_point_real import ViaPointReal
from ...utils.translations import Translations


class EndOfFileReached(Exception):
    pass


class BiomodConfigurationParser:
    def __init__(self, filepath: str):
        tokens = tokenize_biomod(filepath=filepath)

        # Initial attributes
        self.filepath = filepath

        # Initialize the scaling configuration
        self.scale_tool = ScaleTool()
        self.header = (
            "This scaling configuration was created by BioBuddy on "
            + strftime("%Y-%m-%d %H:%M:%S")
            + f"\nIt is based on the original file {filepath}.\n"
        )
        self.warnings = ""

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
                token = read_str(next_token=next_token).lower()

                if current_component is None:

                    if token == "scalingsegment":
                        current_component = SegmentScaling(name=read_str(next_token=next_token), scaling_type=None)

                    elif token == "mass":
                        if self.scale_tool.mass is not None:
                            raise ValueError("Mass already defined")
                        self.scale_tool.mass = read_float(next_token=next_token)

                    elif token in ["version", "gravity", "segment", "imu", "marker", "musclegroup", "muscle", "viapoint"]:
                        continue

                    else:
                        raise ValueError(f"Unknown component {token}")

                elif isinstance(current_component, SegmentScaling):
                    if token == "endscalingsegment":
                        self.scale_tool.scaling_segments.append(current_component)
                        current_component = None

                    elif token == "scalingtype":
                        scaling_type = read_str(next_token=next_token)
                        if scaling_type.lower() == "segmentwisescaling":
                            current_component.scaling_type = SegmentWiseScaling(axis=None, marker_pairs=[])
                        elif scaling_type.lower() == "axiswisescaling":
                            current_component.scaling_type = AxisWiseScaling(
                                axis=None, x_marker_pairs=[], y_marker_pairs=[], z_marker_pairs=[]
                            )
                        else:
                            raise NotImplementedError(f"Scaling type {scaling_type} not implemented yet.")

                    elif token == "axis":
                        if current_component.scaling_type is None:
                            raise RuntimeError(
                                f"The segments scaling type was not defined for scalingsegment {current_component.name}"
                            )
                        current_component.axis = Translations(read_str(next_token=next_token))

                    elif token == "markerpair":
                        if current_component.scaling_type is None:
                            raise RuntimeError(
                                f"The segments scaling type was not defined for scalingsegment {current_component.name}"
                            )
                        elif not isinstance(current_component.scaling_type, SegmentWiseScaling):
                            raise NotImplementedError(
                                f"markerpairs can only be used for scalingtype SegmentWiseScaling"
                            )
                        marker_pair = read_str_list(next_token=next_token)
                        if len(marker_pair) != 2:
                            raise RuntimeError(
                                f"There was a problem reading markerpair {read_str(next_token=next_token)}"
                            )
                        current_component.marker_pairs += [marker_pair]

                    elif token == "xmarkerpair":
                        marker_pair = read_str_list(next_token=next_token)
                        if len(marker_pair) != 2:
                            raise RuntimeError(
                                f"There was a problem reading xmarkerpair {read_str(next_token=next_token)}"
                            )
                        current_component.x_marker_pairs += [marker_pair]

                    elif token == "ymarkerpair":
                        marker_pair = read_str_list(next_token=next_token)
                        if len(marker_pair) != 2:
                            raise RuntimeError(
                                f"There was a problem reading ymarkerpair {read_str(next_token=next_token)}"
                            )
                        current_component.y_marker_pairs += [marker_pair]

                    elif token == "zmarkerpair":
                        marker_pair = read_str_list(next_token=next_token)
                        if len(marker_pair) != 2:
                            raise RuntimeError(
                                f"There was a problem reading zmarkerpair {read_str(next_token=next_token)}"
                            )
                        current_component.z_marker_pairs += [marker_pair]

                    else:
                        raise ValueError(f"Unknown information type {token} in segmentscaling")

                elif (
                    isinstance(current_component, ViaPointReal)
                    or isinstance(current_component, MuscleReal)
                    or isinstance(current_component, MuscleGroup)
                    or isinstance(current_component, MarkerReal)
                    or isinstance(current_component, InertialMeasurementUnitReal)
                    or isinstance(current_component, SegmentReal)
                ):
                    # These components are read by biomod_model_parser
                    continue

                else:
                    raise ValueError(f"Unknown component {type(current_component)}")

        except EndOfFileReached:
            pass
