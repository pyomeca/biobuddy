import numpy as np
from enum import Enum


class Ranges(Enum):
    Q = "Q"
    Qdot = "Qdot"


class RangeOfMotion:
    def __init__(self, range_type: Ranges, min_bound: list[float] | np.ndarray, max_bound: list[float] | np.ndarray):

        # Sanity check
        if len(min_bound) != len(max_bound):
            raise ValueError(
                f"The min_bound and max_bound must have the same length, got {len(min_bound)} and {len(max_bound)}."
            )
        for min_bound_i, max_bound_i in zip(min_bound, max_bound):
            if min_bound_i > max_bound_i:
                raise ValueError(
                    f"The min_bound must be smaller than the max_bound for each degree of freedom, got {min_bound_i} > {max_bound_i}."
                )

        self.range_type = range_type
        self.min_bound = min_bound
        self.max_bound = max_bound

    @property
    def range_type(self):
        return self._range_type

    @range_type.setter
    def range_type(self, value):
        if not isinstance(value, Ranges):
            raise TypeError(f"range_type must be an instance of Ranges Enum, got {type(value)}")
        self._range_type = value

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

    def to_urdf(self, limit_elt):
        if self.range_type == Ranges.Q:
            limit_elt.set("lower", str(self.min_bound[0]))
            limit_elt.set("upper", str(self.max_bound[0]))
        else:
            raise NotImplementedError("URDF only supports Ranges.Q limits.")

    def to_osim(self):
        """
        Generate OpenSim XML representation of range of motion.
        Note: In OpenSim, ranges are specified per coordinate in the joint definition,
        so this method returns the bounds as a tuple for use by the coordinate writer.
        """
        # OpenSim handles ranges at the coordinate level, not as a separate element
        # This method is here for consistency but the actual range writing happens
        # in the joint/coordinate creation in opensim_model_writer.py
        if self.range_type == Ranges.Q:
            return (self.min_bound, self.max_bound)
        else:
            raise NotImplementedError("OpenSim only supports Ranges.Q limits.")
