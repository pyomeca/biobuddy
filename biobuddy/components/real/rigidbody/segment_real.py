from .contact_real import ContactReal
from .inertia_parameters_real import InertiaParametersReal
from .marker_real import MarkerReal
from .mesh_file_real import MeshFileReal
from .mesh_real import MeshReal
from .segment_coordinate_system_real import SegmentCoordinateSystemReal
from ...generic.rigidbody.range_of_motion import RangeOfMotion
from ....utils.rotations import Rotations
from ....utils.translations import Translations


class SegmentReal:
    def __init__(
        self,
        name: str,
        parent_name: str = "",
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
        self.translations = translations
        self.rotations = rotations
        self.q_ranges = q_ranges
        self.qdot_ranges = qdot_ranges
        self.markers: list[MarkerReal] = []
        self.contacts: list[ContactReal] = []
        self.segment_coordinate_system = segment_coordinate_system
        self.inertia_parameters = inertia_parameters
        self.mesh = mesh
        self.mesh_file = mesh_file

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

    def to_biomod(self, with_mesh):
        # Define the print function, so it automatically formats things in the file properly
        out_string = f"segment\t{self.name}\n"
        if self.parent_name:
            out_string += f"\tparent\t{self.parent_name}\n"
        if self.segment_coordinate_system:
            out_string += f"{self.segment_coordinate_system.to_biomod}\n"
        if self.translations != Translations.NONE:
            out_string += f"\ttranslations\t{self.translations.value}\n"
        if self.rotations != Rotations.NONE:
            out_string += f"\trotations\t{self.rotations.value}\n"
        if self.q_ranges is not None:
            out_string += self.q_ranges.to_biomod
        if self.qdot_ranges is not None:
            out_string += self.qdot_ranges.to_biomod
        if self.inertia_parameters:
            out_string += self.inertia_parameters.to_biomod
        if self.mesh and with_mesh:
            out_string += self.mesh.to_biomod
        if self.mesh_file and with_mesh:
            out_string += self.mesh_file.to_biomod
        out_string += "endsegment\n"

        # Also print the markers attached to the segment
        if self.markers:
            for marker in self.markers:
                marker.parent_name = marker.parent_name if marker.parent_name is not None else self.name
                out_string += marker.to_biomod

        # Also print the contacts attached to the segment
        if self.contacts:
            for contact in self.contacts:
                contact.parent_name = contact.parent_name
                out_string += contact.to_biomod

        return out_string
