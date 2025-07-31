class FromOriginalModel:
    def __init__(self, total_mass: float):
        """
        Gets the inertial characteristics form the original model to be used in the scaling
        Parameters
        ----------
        total_mass
            The mass of the subject
        """

        raise NotImplementedError("The scaling based on a specific model is not implemented yet.")
        self.inertial_table = {}

    def __getitem__(self, segment_name: str):
        """
        The inertia paremeters for a particular segment
        Parameters
        ----------
        segment_name
            The name of the segment
        """
        return self.inertial_table[segment_name]
