import numpy as np


class Kinematics:
    """
    Generalized-coordinate samples associated with a biomechanical model.

    from_bvh and from_fbx constructors extract kinematics independently from model parsing.

    Parameters
    ----------
    q
        The generalized coordinates in radians with shape (nb_q, nb_frames).
    time
        The sample times in seconds with shape (nb_frames,).
    dof_names
        The DoF names associated with the rows of q.
    """
    def __init__(
            self,
            q: np.ndarray = None,
            time: np.ndarray = None,
            dof_names: list[str] = None,
    ):
        if q is None:
            self.q = np.empty((0, 0))
        else:
            self.q = q

        if time is None:
            self.time = np.empty((0,))
        else:
            self.time = time

        if dof_names is None:
            self.dof_names = []
        else:
            self.dof_names = dof_names


    @property
    def frame_count(self) -> int:
        """
        Return the number of kinematic frames.
        """
        return int(self.time.shape[0])

    @classmethod
    def from_bvh(cls, filepath: str) -> "Kinematics":
        """
        Extract generalized-coordinate samples from a BVH file.

        Parameters
        ----------
        filepath
            The path to the BVH file to parse.
        """
        from ..model_parser.bvh import BvhModelParser

        return BvhModelParser(filepath=filepath).to_kinematics()

    @classmethod
    def from_fbx(cls, filepath: str) -> "Kinematics":
        """
        Extract generalized-coordinate samples from an FBX file.

        Parameters
        ----------
        filepath
            The path to the FBX file to parse.
        """
        from ..model_parser.fbx import FbxModelParser

        return FbxModelParser(filepath=filepath).to_kinematics()

