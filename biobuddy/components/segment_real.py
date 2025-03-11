from .inertia_parameters_real import InertiaParametersReal
from .marker_real import MarkerReal
from .contact import Contact
from .mesh_real import MeshReal
from .mesh_file_real import MeshFileReal
from .rotations import Rotations
from .range_of_motion import RangeOfMotion
from .segment_coordinate_system_real import SegmentCoordinateSystemReal
from .translations import Translations


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
        if not isinstance(name, str):
            raise RuntimeError(f"The name must be a str, not {type(name)}")
        if not isinstance(parent_name, str):
            raise RuntimeError(f"The parent_name must be a str, not {type(parent_name)}")
        if not isinstance(segment_coordinate_system, SegmentCoordinateSystemReal):
            raise RuntimeError(
                f"The segment_coordinate_system must be a SegmentCoordinateSystemReal, not {type(segment_coordinate_system)}"
            )
        if not isinstance(translations, Translations):
            raise RuntimeError(f"The translations must be a Translations, not {type(translations)}")
        if not isinstance(rotations, Rotations):
            raise RuntimeError(f"The rotations must be a Rotations, not {type(rotations)}")
        if q_ranges is not None and not isinstance(q_ranges, RangeOfMotion):
            raise RuntimeError(f"The q_ranges must be a RangeOfMotion, not {type(q_ranges)}")
        if qdot_ranges is not None and not isinstance(qdot_ranges, RangeOfMotion):
            raise RuntimeError(f"The qdot_ranges must be a RangeOfMotion, not {type(qdot_ranges)}")
        if inertia_parameters is not None and not isinstance(inertia_parameters, InertiaParametersReal):
            raise RuntimeError(
                f"The inertia_parameters must be a InertiaParametersReal, not {type(inertia_parameters)}"
            )
        if mesh is not None and not isinstance(mesh, MeshReal):
            raise RuntimeError(f"The mesh must be a MeshReal, not {type(mesh)}")
        if mesh_file is not None and not isinstance(mesh_file, MeshFileReal):
            raise RuntimeError(f"The mesh_file must be a MeshFileReal, not {type(mesh_file)}")

        self.name = name
        self.parent_name = parent_name
        self.translations = translations
        self.rotations = rotations
        self.q_ranges = q_ranges
        self.qdot_ranges = qdot_ranges
        self.markers = []
        self.contacts = []
        self.segment_coordinate_system = segment_coordinate_system
        self.inertia_parameters = inertia_parameters
        self.mesh = mesh
        self.mesh_file = mesh_file

    def add_marker(self, marker: MarkerReal):
        self.markers.append(marker)

    def remove_marker(self, marker: MarkerReal):
        self.markers.remove(marker)

    def add_contact(self, contact: Contact):
        if contact.parent_name is None:
            raise RuntimeError(f"Contacts must have parents. Contact {contact.name} does not have a parent.")
        self.contacts.append(contact)

    def remove_contact(self, contact: Contact):
        self.contacts.remove(contact)

    @property
    def to_biomod(self):
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
        if self.mesh:
            out_string += self.mesh.to_biomod
        if self.mesh_file:
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
