import numpy as np
from enum import Enum


class Ranges(Enum):
    Q = "Q"
    Qdot = "Qdot"


class RangeOfMotion:
    def __init__(self, range_type: Ranges, min_bound: list[float] | np.ndarray, max_bound: list[float] | np.ndarray):

        self.range_type = range_type
        self.min_bound = min_bound
        self.max_bound = max_bound

    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        if self.range_type == Ranges.Q:
            out_string = f"\trangesQ \n"
        elif self.range_type == Ranges.Qdot:
            out_string = f"\trangesQdot \n"
        else:
            raise RuntimeError("RangeOfMotion's range_type must be Range.Q or Ranges.Qdot")

        for i_dof in range(len(self.min_bound)):
            out_string += f"\t\t{self.min_bound[i_dof]:0.6f}\t{self.max_bound[i_dof]:0.6f}\n"
        out_string += "\n"

        return out_string
