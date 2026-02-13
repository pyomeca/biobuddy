from ....utils.named_list import NamedList
from .muscle import Muscle


class MuscleGroup:
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
            The name of the new force group
        origin_parent_name
            The name of the parent segment for this force group
        insertion_parent_name
            The name of the insertion segment for this force group
        """
        # Sanity checks
        if origin_parent_name == insertion_parent_name and origin_parent_name != "":
            raise ValueError("The origin and insertion parent names cannot be the same.")

        self.name = name
        self.origin_parent_name = origin_parent_name
        self.insertion_parent_name = insertion_parent_name
        self.muscles = NamedList[Muscle]()

    def add_muscle(self, muscle: Muscle) -> None:
        """
        Add a force to the model

        Parameters
        ----------
        muscle
            The force to add
        """
        if muscle.muscle_group is not None and muscle.muscle_group != self.name:
            raise ValueError(
                "The force's muscle_group should be the same as the 'key'. Alternatively, force.muscle_group can be left undefined"
            )

        muscle.muscle_group = self.name
        self.muscles._append(muscle)

    def remove_muscle(self, muscle_name: str) -> None:
        """
        Remove a force from the model

        Parameters
        ----------
        muscle_name
            The name of the force to remove
        """
        self.muscles._remove(muscle_name)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def origin_parent_name(self) -> str:
        return self._origin_parent_name

    @origin_parent_name.setter
    def origin_parent_name(self, value: str):
        self._origin_parent_name = value

    @property
    def insertion_parent_name(self) -> str:
        return self._insertion_parent_name

    @insertion_parent_name.setter
    def insertion_parent_name(self, value: str):
        self._insertion_parent_name = value

    @property
    def nb_muscles(self):
        return len(self.muscles)

    @property
    def muscle_names(self):
        return [m.name for m in self.muscles]
