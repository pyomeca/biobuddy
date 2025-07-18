from ....utils.named_list import NamedList
from ..muscle.muscle_real import MuscleReal


class MuscleGroupReal:
    def __init__(
        self,
        name: str,
        origin_parent_name: str,
        insertion_parent_name: str,
    ):
        """
        Parameters
        ----------
        name
            The name of the new muscle group
        origin_parent_name
            The name of the parent segment for this muscle group
        insertion_parent_name
            The name of the insertion segment for this muscle group
        """
        # Sanity checks
        if not isinstance(name, str):
            raise ValueError("The name of the muscle group must be a string.")
        if not isinstance(origin_parent_name, str):
            raise ValueError("The name of the origin parent segment must be a string.")
        if not isinstance(insertion_parent_name, str):
            raise ValueError("The name of the insertion parent segment must be a string.")
        if origin_parent_name == insertion_parent_name and origin_parent_name != "":
            raise ValueError("The origin and insertion parent names cannot be the same.")

        self.name = name
        self.origin_parent_name = origin_parent_name
        self.insertion_parent_name = insertion_parent_name
        self.muscles = NamedList[MuscleReal]()

    def add_muscle(self, muscle: MuscleReal) -> None:
        """
        Add a muscle to the model

        Parameters
        ----------
        muscle
            The muscle to add
        """
        if muscle.muscle_group is not None and muscle.muscle_group != self.name:
            raise ValueError(
                "The muscle's muscle_group should be the same as the 'key'. Alternatively, muscle.muscle_group can be left undefined"
            )

        muscle.muscle_group = self.name
        self.muscles._append(muscle)

    def remove_muscle(self, muscle_name: str) -> None:
        """
        Remove a muscle from the model

        Parameters
        ----------
        muscle_name
            The name of the muscle to remove
        """
        self.muscles._remove(muscle_name)

    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        out_string = f"musclegroup\t{self.name}\n"
        out_string += f"\tOriginParent\t{self.origin_parent_name}\n"
        out_string += f"\tInsertionParent\t{self.insertion_parent_name}\n"
        out_string += "endmusclegroup\n"

        out_string += "\n // ------ MUSCLES ------\n"
        for muscle in self.muscles:
            out_string += muscle.to_biomod()
        return out_string
