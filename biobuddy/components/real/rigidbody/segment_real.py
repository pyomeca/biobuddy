import numpy as np

from .contact_real import ContactReal
from .inertial_measurement_unit_real import InertialMeasurementUnitReal
from .inertia_parameters_real import InertiaParametersReal
from .marker_real import MarkerReal
from .mesh_file_real import MeshFileReal
from .mesh_real import MeshReal
from .segment_coordinate_system_real import SegmentCoordinateSystemReal
from ...generic.rigidbody.range_of_motion import RangeOfMotion
from ....utils.linear_algebra import RotoTransMatrix, euler_and_translation_to_matrix
from ....utils.rotations import Rotations
from ....utils.translations import Translations
from ....utils.named_list import NamedList


class SegmentReal:
    def __init__(
        self,
        name: str,
        parent_name: str = "base",
        segment_coordinate_system: SegmentCoordinateSystemReal = None,
        translations: Translations = Translations.NONE,
        rotations: Rotations = Rotations.NONE,
        q_ranges: RangeOfMotion = None,
        qdot_ranges: RangeOfMotion = None,
        inertia_parameters: InertiaParametersReal = None,
        mesh: MeshReal = None,
        mesh_file: MeshFileReal = None,
    ):
        self.name = name
        self.parent_name = parent_name
        self.segment_coordinate_system = segment_coordinate_system
        self.translations = translations
        self.rotations = rotations
        self.q_ranges = q_ranges
        self.qdot_ranges = qdot_ranges
        self.markers = NamedList[MarkerReal]()
        self.contacts = NamedList[ContactReal]()
        self.imus = NamedList[InertialMeasurementUnitReal]()
        self.inertia_parameters = inertia_parameters
        self.mesh = mesh
        self.mesh_file = mesh_file

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def parent_name(self) -> str:
        return self._parent_name

    @parent_name.setter
    def parent_name(self, value: str):
        self._parent_name = value

    @property
    def translations(self) -> Translations:
        return self._translations

    @translations.setter
    def translations(self, value: Translations | str | None):
        if value is None or isinstance(value, str):
            value = Translations(value)
        self._translations = value

    @property
    def rotations(self) -> Rotations:
        return self._rotations

    @rotations.setter
    def rotations(self, value: Rotations | str | None):
        if value is None or isinstance(value, str):
            value = Rotations(value)
        self._rotations = value

    @property
    def q_ranges(self) -> RangeOfMotion:
        return self._q_ranges

    @q_ranges.setter
    def q_ranges(self, value: RangeOfMotion):
        self._q_ranges = value

    @property
    def qdot_ranges(self) -> RangeOfMotion:
        return self._qdot_ranges

    @qdot_ranges.setter
    def qdot_ranges(self, value: RangeOfMotion):
        self._qdot_ranges = value

    @property
    def marker_names(self):
        return [marker.name for marker in self.markers]

    @property
    def contact_names(self):
        return [contact.name for contact in self.contacts]

    @property
    def imu_names(self):
        return [imu.name for imu in self.imus]

    @property
    def nb_q(self):
        nb_translations = 0
        if self.translations is not None and self.translations != Translations.NONE:
            nb_translations = len(self.translations.value)
        nb_rotations = 0
        if self.rotations is not None and self.rotations != Rotations.NONE:
            nb_rotations = len(self.rotations.value)
        return nb_translations + nb_rotations

    @property
    def markers(self) -> NamedList[MarkerReal]:
        return self._markers

    @markers.setter
    def markers(self, value: NamedList[MarkerReal]):
        if isinstance(value, list) and not isinstance(value, NamedList):
            value = NamedList.from_list(value)
        self._markers = value

    @property
    def nb_markers(self):
        return len(self.markers)

    @property
    def contacts(self) -> NamedList[ContactReal]:
        return self._contacts

    @contacts.setter
    def contacts(self, value: NamedList[ContactReal]):
        if isinstance(value, list) and not isinstance(value, NamedList):
            value = NamedList.from_list(value)
        self._contacts = value

    @property
    def nb_contacts(self):
        return len(self.contacts)

    @property
    def imus(self) -> NamedList[InertialMeasurementUnitReal]:
        return self._imus

    @imus.setter
    def imus(self, value: NamedList[InertialMeasurementUnitReal]):
        if isinstance(value, list) and not isinstance(value, NamedList):
            value = NamedList.from_list(value)
        self._imus = value

    @property
    def nb_imus(self):
        return len(self.imus)

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
        self.markers.append(marker)

    def remove_marker(self, marker: MarkerReal):
        self.markers.remove(marker)

    def add_contact(self, contact: ContactReal):
        if contact.parent_name is None:
            raise RuntimeError(f"Contacts must have parents, but {contact.name} does not.")
        self.contacts.append(contact)

    def remove_contact(self, contact: ContactReal):
        self.contacts.remove(contact)

    def add_imu(self, imu: InertialMeasurementUnitReal):
        if imu.parent_name is None:
            raise RuntimeError(f"IMUs must have parents, but {imu.name} does not.")
        self.imus.append(imu)

    def remove_imu(self, imu: InertialMeasurementUnitReal):
        self.imus.remove(imu)

    def rt_from_local_q(self, local_q: np.ndarray) -> np.ndarray:

        if local_q.shape[0] != self.nb_q:
            raise RuntimeError(
                f"The shape of the q vector is not correct: got local_q of size {local_q.shape} for the segment {self.name} with {self.nb_q} Dofs."
            )
        rt = np.eye(4)

        if self.nb_q != 0:
            # @pariterre: is it possible to add translations, then rotations, then again translations in biorbd ?
            # TODO: make the order trans, rot dynamic
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


            rt = euler_and_translation_to_matrix(angles=rotations,
                                                angle_sequence=angle_sequence,
                                                translations=translations)

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
