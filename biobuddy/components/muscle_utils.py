from enum import Enum


class MuscleType(Enum):
    HILL = "hill"
    HILL_THELEN = "hillthelen"
    HILL_DE_GROOTE = "hilldegroote"


class MuscleStateType(Enum):
    DEGROOTE = "degroote"
    DEFAULT = "default"
    BUCHANAN = "buchanan"
