import numpy as np
from enum import Enum


class Ranges(Enum):
    Q = "Q"
    Qdot = "Qdot"


class RangeOfMotion:
    def __init__(self, range_type: Ranges, min_bound: list[float] | np.ndarray, max_bound: list[float] | np.ndarray):

        # Sanity check
        for min_bound_i, max_bound_i in zip(min_bound, max_bound):
            if min_bound_i > max_bound_i:
                raise ValueError(
                    f"The min_bound must be smaller than the max_bound for each degree of freedom, got {min_bound_i} > {max_bound_i}."
                )

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

    def to_osim(self):
        """
        Generate OpenSim XML representation of range of motion.
        Note: In OpenSim, ranges are specified per coordinate in the joint definition,
        so this method returns the bounds as a tuple for use by the coordinate writer.
        """
        # OpenSim handles ranges at the coordinate level, not as a separate element
        # This method is here for consistency but the actual range writing happens
        # in the joint/coordinate creation in opensim_model_writer.py
        return (self.min_bound, self.max_bound)
