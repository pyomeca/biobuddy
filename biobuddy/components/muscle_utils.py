from enum import Enum


class MuscleType(Enum):
    HILL = "hill"
    HILL_THELEN = "hillthelen"
    HILL_DE_GROOTE = "hilldegroote"


class MuscleStateType(Enum):
    DEGROOTE = "degroote"
    DEFAULT = "default"
    BUCHANAN = "buchanan"


class MuscleUtils:
    def __init__(self):
        pass

    @property
    def nb_via_points(self) -> int:
        return len(self.via_points)

    @property
    def via_point_names(self) -> list[str]:
        name_list = []
        for via_point in self.via_points:
            name_list += [via_point.name]
        return name_list
