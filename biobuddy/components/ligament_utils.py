from enum import Enum
import numpy as np


class LigamentType(Enum):
    # I am leaving out on purpose the "constant" type, since it is not really a ligament, but rather a force generator.
    LINEAR_SPRING = "linearspring"
    SECOND_ORDER_SPRING = "secondorderspring"
    FUNCTION = "function"


LIGAMENT_FUNCTION = {
    LigamentType.LINEAR_SPRING: lambda length, slack_length, stiffness: stiffness * (length - slack_length),
    LigamentType.SECOND_ORDER_SPRING: lambda length, slack_length, stiffness: (stiffness / 2) * ((length - slack_length) + np.sqrt((length - slack_length) * (length - slack_length) + 1e-6)),
}