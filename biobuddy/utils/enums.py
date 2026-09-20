from enum import Enum


class Rotations(Enum):
    NONE = None
    X = "x"
    Y = "y"
    Z = "z"
    XY = "xy"
    XZ = "xz"
    YX = "yx"
    YZ = "yz"
    ZX = "zx"
    ZY = "zy"
    XYZ = "xyz"
    XZY = "xzy"
    YXZ = "yxz"
    YZX = "yzx"
    ZXY = "zxy"
    ZYX = "zyx"


class Translations(Enum):
    NONE = None
    X = "x"
    Y = "y"
    Z = "z"
    XY = "xy"
    XZ = "xz"
    YX = "yx"
    YZ = "yz"
    ZX = "zx"
    ZY = "zy"
    XYZ = "xyz"
    XZY = "xzy"
    YXZ = "yxz"
    YZX = "yzx"
    ZXY = "zxy"
    ZYX = "zyx"


class Sign(Enum):
    POSITIVE = 1
    ZERO = 0
    NEGATIVE = -1


class ViewAs(Enum):
    BIORBD = "biorbd"
    # OPENSIM = "opensim"  # TODO


class ViewerType(Enum):
    PYORERUN = "pyorerun"
    BIOVIZ = "bioviz"

class LengthUnits(Enum):
    M = "meter"
    CM = "centimeter"
    MM = "milimeter"
