import numpy as np

from .contact_real import ContactReal
from .inertial_measurement_unit_real import InertialMeasurementUnitReal
from .inertia_parameters_real import InertiaParametersReal
from .marker_real import MarkerReal
from .mesh_file_real import MeshFileReal
from .mesh_real import MeshReal
from .segment_coordinate_system_real import SegmentCoordinateSystemReal
from ...generic.rigidbody.range_of_motion import RangeOfMotion
from ....utils.linear_algebra import RotoTransMatrix
from ....utils.rotations import Rotations
from ....utils.translations import Translations
from ....utils.named_list import NamedList
from ...segment_utils import SegmentUtils
from ....utils.checks import check_name


class SegmentReal(SegmentUtils):
    def __init__(
        self,
        name: str,
        parent_name: str = "base",
        segment_coordinate_system: SegmentCoordinateSystemReal = SegmentCoordinateSystemReal(
            scs=RotoTransMatrix(), is_scs_local=True
        ),
        translations: Translations = Translations.NONE,
        rotations: Rotations = Rotations.NONE,
        dof_names: list[str] = None,
        q_ranges: RangeOfMotion = None,
        qdot_ranges: RangeOfMotion = None,
        inertia_parameters: InertiaParametersReal = None,
        mesh: MeshReal = None,
        mesh_file: MeshFileReal = None,
    ):
        """
        Create a new real segment.

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
        self.segment_coordinate_system = segment_coordinate_system
        self.translations = translations
        self.rotations = rotations
        self.dof_names = dof_names
        self.q_ranges = q_ranges
        self.qdot_ranges = qdot_ranges
        self.markers = NamedList[MarkerReal]()
        self.contacts = NamedList[ContactReal]()
        self.imus = NamedList[InertialMeasurementUnitReal]()
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
    def markers(self) -> NamedList[MarkerReal]:
        return self._markers

    @markers.setter
    def markers(self, value: NamedList[MarkerReal]):
        if isinstance(value, list) and not isinstance(value, NamedList):
            value = NamedList.from_list(value)
        self._markers = value

    @property
    def contacts(self) -> NamedList[ContactReal]:
        return self._contacts

    @contacts.setter
    def contacts(self, value: NamedList[ContactReal]):
        if isinstance(value, list) and not isinstance(value, NamedList):
            value = NamedList.from_list(value)
        self._contacts = value

    @property
    def imus(self) -> NamedList[InertialMeasurementUnitReal]:
        return self._imus

    @imus.setter
    def imus(self, value: NamedList[InertialMeasurementUnitReal]):
        if isinstance(value, list) and not isinstance(value, NamedList):
            value = NamedList.from_list(value)
        self._imus = value

    @property
    def segment_coordinate_system(self) -> SegmentCoordinateSystemReal:
        return self._segment_coordinate_system

    @segment_coordinate_system.setter
    def segment_coordinate_system(self, value: SegmentCoordinateSystemReal):
        self._segment_coordinate_system = value

    @property
    def inertia_parameters(self) -> InertiaParametersReal:
        return self._inertia_parameters

    @inertia_parameters.setter
    def inertia_parameters(self, value: InertiaParametersReal):
        self._inertia_parameters = value

    @property
    def mesh(self) -> MeshReal:
        return self._mesh

    @mesh.setter
    def mesh(self, value: MeshReal):
        self._mesh = value

    def add_marker(self, marker: MarkerReal):
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

    def add_contact(self, contact: ContactReal):
        """
        Add a new contact to the segment

        Parameters
        ----------
        contact
            The contact to add
        """
        if contact.parent_name is None:
            raise ValueError(f"Contacts must have parents. Contact {contact.name} does not have a parent.")
        elif contact.parent_name != self.name:
            raise ValueError("The contact name should be the same as the 'key'.")
        contact.parent_name = self.name
        self.contacts._append(contact)

    def remove_contact(self, contact: str):
        self.contacts._remove(contact)

    def add_imu(self, imu: InertialMeasurementUnitReal):
        if imu.parent_name is None:
            raise RuntimeError(f"IMUs must have parents, but {imu.name} does not.")
        self.imus._append(imu)

    def remove_imu(self, imu: str):
        self.imus._remove(imu)

    def rt_from_local_q(self, local_q: np.ndarray) -> RotoTransMatrix:

        if local_q.shape[0] != self.nb_q:
            raise RuntimeError(
                f"The shape of the q vector is not correct: got local_q of size {local_q.shape} for the segment {self.name} with {self.nb_q} Dofs."
            )
        rt = RotoTransMatrix()

        if self.nb_q != 0:
            q_counter = 0
            translations = np.zeros((3,))
            rotations = np.zeros((3,))
            angle_sequence = "xyz"
            if self.translations != Translations.NONE:
                for i_trans, trans in enumerate(["X", "Y", "Z"]):
                    if trans in self.translations.value.upper():
                        translations[i_trans] = local_q[q_counter]
                        q_counter += 1

            if self.rotations != Rotations.NONE:
                rotations = local_q[q_counter:]
                angle_sequence = self.rotations.value

            rt.from_euler_angles_and_translation(
                angle_sequence=angle_sequence, angles=rotations, translation=translations
            )

        return rt

    def to_biomod(self, with_mesh):
        # Define the print function, so it automatically formats things in the file properly
        out_string = f"segment\t{self.name}\n"
        if self.parent_name:
            out_string += f"\tparent\t{self.parent_name}\n"
        if self.segment_coordinate_system:
            out_string += f"{self.segment_coordinate_system.to_biomod()}"
        if self.translations != Translations.NONE:
            out_string += f"\ttranslations\t{self.translations.value}\n"
        if self.rotations != Rotations.NONE:
            out_string += f"\trotations\t{self.rotations.value}\n"
        if self.q_ranges is not None:
            out_string += self.q_ranges.to_biomod()
        if self.qdot_ranges is not None:
            out_string += self.qdot_ranges.to_biomod()
        if self.inertia_parameters:
            out_string += self.inertia_parameters.to_biomod()
        if self.mesh and with_mesh:
            out_string += self.mesh.to_biomod()
        if self.mesh_file and with_mesh:
            out_string += self.mesh_file.to_biomod()
        out_string += "endsegment\n"

        # Also print the markers attached to the segment
        if self.markers:
            out_string += "\n"
            for marker in self.markers:
                marker.parent_name = marker.parent_name if marker.parent_name is not None else self.name
                out_string += marker.to_biomod()

        # Also print the contacts attached to the segment
        if self.contacts:
            out_string += "\n"
            for contact in self.contacts:
                contact.parent_name = contact.parent_name
                out_string += contact.to_biomod()

        if self.imus:
            out_string += "\n"
            for imu in self.imus:
                out_string += imu.to_biomod()

        return out_string
