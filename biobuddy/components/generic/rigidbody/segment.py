from .contact import Contact
from .inertia_parameters import InertiaParameters
from .marker import Marker
from .mesh import Mesh
from .mesh_file import MeshFile
from .range_of_motion import RangeOfMotion, Ranges
from .segment_coordinate_system import SegmentCoordinateSystem
from ....utils.named_list import NamedList
from ....utils.enums import Rotations
from ....utils.enums import Translations
from ...segment_utils import SegmentUtils
from ....utils.checks import check_name


class Segment(SegmentUtils):
    def __init__(
        self,
        name,
        parent_name: str = "base",
        translations: Translations = Translations.NONE,
        rotations: Rotations = Rotations.NONE,
        dof_names: list[str] = None,
        q_ranges: RangeOfMotion = None,
        qdot_ranges: RangeOfMotion = None,
        segment_coordinate_system: SegmentCoordinateSystem = None,
        inertia_parameters: InertiaParameters = None,
        mesh: Mesh = None,
        mesh_file: MeshFile = None,
    ):
        """
        Create a new generic segment.

        Parameters
        ----------
        name
            The name of the segment
        parent_name
            The name of the segment the current segment is attached to
        translations
            The sequence of translation
        rotations
            The sequence of rotation
        dof_names
            The names of the degrees of freedom of the segment
            If None, it will be automatically generated based on translations and rotations (like "segment_transX" or "segment_rotY")
        q_ranges
            The range of motion of the segment
        qdot_ranges
            The range of motion of the segment
        segment_coordinate_system
            The coordinate system of the segment
        inertia_parameters
            The inertia parameters of the segment
        mesh
            The mesh points of the segment
        mesh_file
            The mesh file of the segment
        """

        super().__init__()
        self.name = check_name(name)
        self.parent_name = check_name(parent_name)
        self.translations = translations
        self.rotations = rotations
        self.dof_names = dof_names
        self.q_ranges = q_ranges
        self.qdot_ranges = qdot_ranges
        self.markers = NamedList[Marker]()
        self.contacts = NamedList[Contact]()
        self.segment_coordinate_system = segment_coordinate_system
        self.inertia_parameters = inertia_parameters
        self.mesh = mesh
        self.mesh_file = mesh_file

    @property
    def dof_names(self) -> list[str]:
        return self._dof_names

    @dof_names.setter
    def dof_names(self, value: list[str]):
        if value is None:
            value = []
            if self.translations != Translations.NONE:
                for trans in self.translations.value:
                    value += [f"{self.name}_trans{trans.upper()}"]
            if self.rotations != Rotations.NONE:
                for rot in self.rotations.value:
                    value += [f"{self.name}_rot{rot.upper()}"]
        if len(value) != self.nb_q:
            raise RuntimeError(
                f"The number of DoF names ({len(value)}) does not match the number of DoFs ({self.nb_q}) in segment {self.name}."
            )
        self._dof_names = value

    @property
    def markers(self) -> NamedList[Marker]:
        return self._markers

    @markers.setter
    def markers(self, value: NamedList[Marker]):
        if isinstance(value, list) and not isinstance(value, NamedList):
            value = NamedList.from_list(value)
        self._markers = value

    @property
    def contacts(self) -> NamedList[Contact]:
        return self._contacts

    @contacts.setter
    def contacts(self, value: NamedList[Contact]):
        if isinstance(value, list) and not isinstance(value, NamedList):
            value = NamedList.from_list(value)
        self._contacts = value

    @property
    def segment_coordinate_system(self) -> SegmentCoordinateSystem:
        return self._segment_coordinate_system

    @segment_coordinate_system.setter
    def segment_coordinate_system(self, value: SegmentCoordinateSystem):
        self._segment_coordinate_system = value

    @property
    def inertia_parameters(self) -> InertiaParameters:
        return self._inertia_parameters

    @inertia_parameters.setter
    def inertia_parameters(self, value: InertiaParameters):
        self._inertia_parameters = value

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @mesh.setter
    def mesh(self, value: Mesh):
        self._mesh = value

    def add_marker(self, marker: Marker):
        """
        Add a new marker to the segment

        Parameters
        ----------
        marker
            The marker to add
        """
        if marker.parent_name is not None and marker.parent_name != self.name:
            raise ValueError(
                "The marker name should be the same as the 'key'. Alternatively, marker.name can be left undefined"
            )

        marker.parent_name = self.name
        self.markers._append(marker)

    def remove_marker(self, marker: str):
        self.markers._remove(marker)

    def add_contact(self, contact: Contact):
        """
        Add a new contact to the segment

        Parameters
        ----------
        contact
            The contact to add
        """
        if contact.parent_name is not None and contact.parent_name != self.name:
            raise ValueError(
                "The contact name should be the same as the 'key'. Alternatively, contact.name can be left undefined"
            )
        contact.parent_name = self.name
        self.contacts._append(contact)

    def remove_contact(self, contact: str):
        self.contacts._remove(contact)
