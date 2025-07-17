from typing import Protocol

import numpy as np
from ....utils.linear_algebra import RotoTransMatrix


class CoordinateSystemRealProtocol(Protocol):
    """
    This is use to evaluate a "real" coordinate system (mostly SegmentCoordinateSystemReal).
    It is declare to prevent circular imports of SegmentCoordinateSystemReal
    """

    def __init__(self):
        self.scs: RotoTransMatrix = RotoTransMatrix()

    @property
    def inverse(self) -> np.ndarray:
        """
        Get the transpose of the coordinate system
        """
