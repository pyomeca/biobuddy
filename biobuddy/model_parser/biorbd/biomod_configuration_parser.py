from time import strftime

from .utils import tokenize_biomod, read_str, read_str_list, read_float
from ...components.real.rigidbody.segment_scaling import (
    SegmentScaling,
    SegmentWiseScaling,
    AxisWiseScaling,
)
from ...components.real.rigidbody.marker_weight import MarkerWeight
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


TOKENS_TO_IGNORE_NO_COMPONENTS = ["endsegment",
                                  "endimu",
                                  "endmarker",
                                  "endmusclegroup",
                                  "endmuscle",
                                  "endviapoint",
                                  ]
TOKENS_TO_IGNORE_ONE_COMPONENTS = ["version",
                                   "segment",
                                   "imu",
                                   "marker",
                                   "musclegroup",
                                   "muscle",
                                   "viapoint",
                                   "parent",
                                   "rtinmatrix",
                                   "translations",
                                   "rotations",
                                   "meshfile",
                                   "technical",
                                   "anatomical",
                                   "insertionparent",
                                   "originparent",
                                   "type",
                                   "statetype"]
TOKENS_TO_IGNORE_THREE_COMPONENTS = []
TOKENS_TO_IGNORE_NUMBERS = ["gravity",
                            "rt",
                            "rangesq",
                            "rangesqdot",
                            "mass",
                            "com",
                            "centerofmass",
                            "inertia",
                            "inertia_xxyyzz",
                            "mesh",
                            "meshcolor",
                            "meshscale",
                            "meshrt",
                            "position",
                            "originposition",
                            "insertionposition",
                            "optimallength",
                            "maximalforce",
                            "tendonslacklength",
                            "pennationangle",
                            "maximal_excitation",
                            "maxvelocity"
                            ]


class EndOfFileReached(Exception):
    pass


class BiomodConfigurationParser:
    def __init__(self, filepath: str, original_model: "BiomechanicalModelReal"):

        # Initial attributes
        self.filepath = filepath

        # Initialize the scaling configuration
        self.scale_tool = ScaleTool(original_model)  # TODO: this is weird !
        self._read()

    def _read(self):
        self.header = (
            "This scaling configuration was created by BioBuddy on "
            + strftime("%Y-%m-%d %H:%M:%S")
            + f"\nIt is based on the original file {self.filepath}.\n"
        )
        self.warnings = ""

        tokens = tokenize_biomod(filepath=self.filepath)

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
        current_component = None
        token_index = -1
        try:
            while True:
                token = read_str(next_token=next_token).lower()

                if current_component is None:

                    if token == "scalingsegment":
                        current_component = SegmentScaling(name=read_str(next_token=next_token), scaling_type=None)
                    elif token == "markerweight":
                        marker_name = read_str(next_token=next_token)
                        marker_wight = read_float(next_token=next_token)
                        current_component = MarkerWeight(name=marker_name, weight=marker_wight)
                        self.scale_tool.add_marker_weight(current_component)
                        current_component = None

                    elif token in TOKENS_TO_IGNORE_NO_COMPONENTS:
                        continue
                    elif token in TOKENS_TO_IGNORE_ONE_COMPONENTS:
                        token_index += 1
                    elif token in TOKENS_TO_IGNORE_THREE_COMPONENTS:
                        token_index += 3
                    elif token in TOKENS_TO_IGNORE_NUMBERS:
                        number_of_tokens = nb_float_tokens_until_next_str()
                        token_index += number_of_tokens
                    else:
                        raise ValueError(f"Unknown component {token}")

                elif isinstance(current_component, SegmentScaling):
                    if token == "endscalingsegment":
                        self.scale_tool.add_scaling_segment(current_component)
                        current_component = None

                    elif token == "scalingtype":
                        scaling_type = read_str(next_token=next_token)
                        if scaling_type.lower() == "segmentwisescaling":
                            current_component.scaling_type = SegmentWiseScaling(
                                segment_name="", axis=None, marker_pairs=[]
                            )
                        elif scaling_type.lower() == "axiswisescaling":
                            current_component.scaling_type = AxisWiseScaling(
                                segment_name="", axis=None,  marker_pairs=[]
                            )
                        else:
                            raise NotImplementedError(f"Scaling type {scaling_type} not implemented yet.")

                    elif token == "axis":
                        if current_component.scaling_type is None:
                            raise RuntimeError(
                                f"The segments scaling type was not defined for scalingsegment {current_component.name}"
                            )
                        current_component.scaling_type.axis = Translations(read_str(next_token=next_token))

                    elif token == "markerpair":
                        if current_component.scaling_type is None:
                            raise RuntimeError(
                                f"The segments scaling type was not defined for scalingsegment {current_component.name}"
                            )
                        elif not isinstance(current_component.scaling_type, SegmentWiseScaling):
                            raise NotImplementedError(
                                f"markerpairs can only be used for scalingtype SegmentWiseScaling"
                            )
                        marker_pair = read_str_list(next_token=next_token, length=2)
                        if len(marker_pair) != 2:
                            raise RuntimeError(
                                f"There was a problem reading markerpair {read_str(next_token=next_token)}"
                            )
                        current_component.scaling_type.marker_pairs += [marker_pair]

                    elif token == "xmarkerpair":
                        marker_pair = read_str_list(next_token=next_token, length=2)
                        if len(marker_pair) != 2:
                            raise RuntimeError(
                                f"There was a problem reading xmarkerpair {read_str(next_token=next_token)}"
                            )
                        current_component.scaling_type.marker_pairs += [marker_pair]

                    elif token == "ymarkerpair":
                        marker_pair = read_str_list(next_token=next_token, length=2)
                        if len(marker_pair) != 2:
                            raise RuntimeError(
                                f"There was a problem reading ymarkerpair {read_str(next_token=next_token)}"
                            )
                        current_component.scaling_type.marker_pairs += [marker_pair]

                    elif token == "zmarkerpair":
                        marker_pair = read_str_list(next_token=next_token, length=2)
                        if len(marker_pair) != 2:
                            raise RuntimeError(
                                f"There was a problem reading zmarkerpair {read_str(next_token=next_token)}"
                            )
                        current_component.scaling_type.marker_pairs += [marker_pair]

                    else:
                        raise ValueError(f"Unknown information type {token} in scalingsegment")

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


"""
example:

scalingsegment pelvis
	scalingtype	segmentwisescaling
	axis	xyz
	markerpair	RASIS LASIS
	markerpair	LPSIS RPSIS
	markerpair	RASIS RPSIS
	markerpair	LASIS LPSIS
	markerpair	LASIS RPSIS
	markerpair	RASIS LPSIS
endscalingsegment
"""
