import numpy as np

from .functions import Functions


class PathPointMovement:
    def __init__(
        self,
        dof_names: list[str],
        locations: list[Functions],
    ):
        if len(dof_names) != 3:
            raise RuntimeError("dof_names must be a list of 3 dof_names (x, y, x).")
        if len(locations) != 3:
            raise RuntimeError("locations must be a list of 3 Functions (x, y, x).")
        if not all(isinstance(loc, Functions) for loc in locations):
            raise RuntimeError("All locations must be instances of Functions.")

        self.dof_names = dof_names
        self.locations = locations

    def evaluate(self, angles: np.ndarray) -> np.ndarray:
        """Evaluate the condition based on the current joint angles."""
        position = np.zeros((angles.shape[0],))
        for i_angle, angle in enumerate(angles):
            position[i_angle] = self.locations[i_angle].evaluate(angle)
        return position


class PathPointCondition:
    def __init__(
        self,
        dof_name: str,
        range_min: float,
        range_max: float,
    ):

        self.dof_name = dof_name
        self.range_min = float(range_min)
        self.range_max = float(range_max)

    def evaluate(self, angle: float) -> bool:
        """Evaluate the condition based on the current joint angles."""
        if self.range_min <= angle <= self.range_max:
            return True
        else:
            return False
