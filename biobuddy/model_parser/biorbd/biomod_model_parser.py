from ...components.real.biomechanical_model_real import BiomechanicalModelReal
from ...components.real.rigidbody.segment_real import SegmentReal


class BiomodModelParser:
    def __init__(self, filepath: str):
        # Load the model from the filepath
        with open(filepath) as f:
            content = f.read()

        # Parse the elements of the model
        self.segments: dict[str, SegmentReal] = {}

    def to_real(self) -> BiomechanicalModelReal:
        raise NotImplementedError()
