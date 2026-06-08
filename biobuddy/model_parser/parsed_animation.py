from dataclasses import dataclass

import numpy as np


@dataclass
class ParsedAnimation:
    """
    Animation samples mapped to biorbd-compatible generalized coordinates.

    Parameters
    ----------
    q
        The generalized coordinates with shape ``(nb_q, nb_frames)``.
    time
        The sample times in seconds with shape ``(nb_frames,)``.
    dof_names
        The DoF names associated with the rows of ``q``.
    """

    q: np.ndarray
    time: np.ndarray
    dof_names: list[str]

    @property
    def frame_count(self) -> int:
        """
        Return the number of animation frames.
        """
        return int(self.time.shape[0])
