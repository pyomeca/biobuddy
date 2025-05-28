class MarkerWeight:
    def __init__(self, name: str, weight: float):
        """
        Initialize a marker weight.

        Parameters
        ----------
        name
            The name of the marker
        weight
            The weight of the marker in the inverse kinematics. A higher weight means that the marker will have more influence on the optimal pose.
        """
        self.name = name
        self.weight = weight
