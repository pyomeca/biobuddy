
from ..utils.named_list import NamedList

class ScalingConfiguration:
    def __init__(self):
        self.scaling_segments = NamedList[SegmentReal]()
