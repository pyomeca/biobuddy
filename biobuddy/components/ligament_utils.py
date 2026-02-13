from enum import Enum


class LigamentType(Enum):
    CONSTANT = "constant"
    LINEAR_SPRING = "linear_spring"
    QUADRATIC_SPRING = "quadratic_spring"
    # TODO: add Osim ligament types (force-length function)
