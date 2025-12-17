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


class ViewAs(Enum):
    # TODO @charbie Split Backend model and Backend visualizer
    BIORBD = "biorbd"
    BIORBD_BIOVIZ = "biorbd_bioviz"
    # OPENSIM = "opensim"
