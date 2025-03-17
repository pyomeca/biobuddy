from typing import Protocol

import numpy as np


class CoordinateSystemRealProtocol(Protocol):
    """
    This is use to evaluate a "real" coordinate system (mostly SegmentCoordinateSystemReal).
    It is declare to prevent circular imports of SegmentCoordinateSystemReal
    """

    @property
    def transpose(self) -> np.ndarray:
        """
        Get the transpose of the coordinate system
        """
