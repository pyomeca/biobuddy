from enum import Enum
import numpy as np


class LigamentType(Enum):
    CONSTANT = "constant"
    LINEAR_SPRING = "linear_spring"
    SECOND_ORDER_SPRING = "second_order_spring"
    FUNCTION = "function"


LIGAMENT_FUNCTION = {
    LigamentType.CONSTANT: lambda length, slack_length, force: force,
    LigamentType.LINEAR_SPRING: lambda length, slack_length, stiffness: stiffness * (length - slack_length),
    LigamentType.SECOND_ORDER_SPRING: lambda length, slack_length, stiffness, epsilon: (stiffness / 2) * ((length - slack_length) + np.sqrt((lenght - slack_length) * (length - slack_length) + epsilon * epsilon)),
}