import numpy as np
from lxml import etree

from .contact_real import ContactReal
from .inertial_measurement_unit_real import InertialMeasurementUnitReal
from .inertia_parameters_real import InertiaParametersReal
from .marker_real import MarkerReal
from .mesh_file_real import MeshFileReal
from .mesh_real import MeshReal
from .segment_coordinate_system_real import SegmentCoordinateSystemReal
from ...generic.rigidbody.range_of_motion import RangeOfMotion
from ....utils.linear_algebra import RotoTransMatrix
from ....utils.enums import Rotations
from ....utils.enums import Translations
from ....utils.named_list import NamedList
from ...segment_utils import SegmentUtils
from ....utils.checks import check_name
from ....utils.linear_algebra import get_vector_from_sequence


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
        if value is None or value == []:
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
        if contact.parent_name is not None and contact.parent_name != self.name:
            raise ValueError(
                "The contact name should be the same as the 'key'. Alternatively, contact.name can be left undefined"
            )
        contact.parent_name = self.name
        self.contacts._append(contact)

    def remove_contact(self, contact: str):
        self.contacts._remove(contact)

    def add_imu(self, imu: InertialMeasurementUnitReal):
        if imu.parent_name is not None and imu.parent_name != self.name:
            raise ValueError(
                "The imu name should be the same as the 'key'. Alternatively, imu.name can be left undefined"
            )
        imu.parent_name = self.name
        self.imus._append(imu)

    def remove_imu(self, imu: str):
        self.imus._remove(imu)

    def remove_dof(self, dof_name: str) -> None:
        """
        Remove a degree of freedom from the segment

        Parameters
        ----------
        dof_name
            The name of the degree of freedom to remove
        """
        if dof_name not in self.dof_names:
            raise RuntimeError(f"The dof {dof_name} is not part of the segment {self.name}.")
        dof_index = self.dof_names.index(dof_name)
        nb_translations = 0 if self.translations == Translations.NONE else len(self.translations.value)
        nb_rotations = 0 if self.rotations == Rotations.NONE else len(self.rotations.value)

        # Remove the dof type
        if nb_translations == 1:
            self.translations = Translations.NONE
        elif dof_index < nb_translations:
            new_dof_str = self.translations.value[:dof_index] + self.translations.value[dof_index + 1 :]
            self.translations = Translations(new_dof_str)
        elif nb_rotations == 1:
            self.rotations = Rotations.NONE
        else:
            new_dof_str = (
                self.rotations.value[: dof_index - nb_translations]
                + self.rotations.value[dof_index - nb_translations + 1 :]
            )
            self.rotations = Rotations(new_dof_str)

        # Remove the dof ranges
        if self.q_ranges is not None:
            if len(self.q_ranges.min_bound) == 1:
                self.q_ranges = None
            else:
                self.q_ranges = RangeOfMotion(
                    range_type=self.q_ranges.range_type,
                    min_bound=[m for i, m in enumerate(self.q_ranges.min_bound) if i != dof_index],
                    max_bound=[m for i, m in enumerate(self.q_ranges.max_bound) if i != dof_index],
                )
        if self.qdot_ranges is not None:
            if len(self.qdot_ranges.min_bound) == 1:
                self.qdot_ranges = None
            else:
                self.qdot_ranges = RangeOfMotion(
                    range_type=self.qdot_ranges.range_type,
                    min_bound=[m for i, m in enumerate(self.qdot_ranges.min_bound) if i != dof_index],
                    max_bound=[m for i, m in enumerate(self.qdot_ranges.max_bound) if i != dof_index],
                )

        # Remove dof names (must be done last to avoid messing up with nb_q)
        if self.dof_names is not None:
            if len(self.dof_names) == 1:
                self.dof_names = None
            else:
                self.dof_names = [m for m in self.dof_names if m != dof_name]

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

    def to_biomod(self, with_mesh: bool) -> str:
        """
        Define the print function, so it automatically formats things in the file properly
        """
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

    def to_urdf(self, urdf_model: etree.Element, with_mesh: bool):
        """
        Define the print function, so it automatically formats things in the file properly
        """
        link = etree.SubElement(urdf_model, "link", name=self.name)
        if self.inertia_parameters is not None:
            self.inertia_parameters.to_urdf(link)
        if self.mesh_file is not None and with_mesh:
            self.mesh_file.to_urdf(urdf_model, link)
        if self.mesh is not None and with_mesh:
            raise NotImplementedError("Mesh points are not yet implemented in URDF export")

        # Get the joint type
        if self.nb_q == 0:
            joint_type = "fixed"
            joint = etree.SubElement(urdf_model, "joint", name=f"{self.name}_fixed")
        elif self.nb_q == 1:
            joint_type = "revolute"
            joint = etree.SubElement(urdf_model, "joint", name=f"{self.dof_names[0]}")
        else:
            joint_type = "continuous"
            raise NotImplementedError("Joints with more than one DoF are not yet implemented in URDF export")

        # Write the parent/child
        if joint is not None:
            joint.set("type", joint_type)
            parent = etree.SubElement(joint, "parent", link=self.parent_name)
            child = etree.SubElement(joint, "child", link=self.name)
            origin = etree.SubElement(joint, "origin")
            self.segment_coordinate_system.to_urdf(origin)

        # Add dof specifications
        if self.nb_q == 1:
            if self.q_ranges is not None:
                limit = etree.SubElement(joint, "limit")
                self.q_ranges.to_urdf(limit)
            rotation_array = get_vector_from_sequence(self.rotations.value)
            axis = etree.SubElement(joint, "axis", xyz=f"{rotation_array[0]} {rotation_array[1]} {rotation_array[2]}")

        if self.nb_markers > 0:
            raise NotImplementedError("Markers are not implemented yet for URDF export")
        if self.nb_contacts > 0:
            raise NotImplementedError("Contacts are not implemented yet for URDF export")
        if self.nb_imus > 0:
            raise NotImplementedError("IMUs are not implemented yet for URDF export")

        return

    def to_osim(self, with_mesh: bool = False):
        """Generate OpenSim XML representation of the segment (Body element)"""

        body_elem = etree.Element("Body", name=self.name)

        # frame_geometry = etree.SubElement(body_elem, "FrameGeometry")

        if self.inertia_parameters is not None:
            mass_elem = etree.SubElement(body_elem, "mass")
            mass_elem.text = f"{self.inertia_parameters.mass:.8f}"

            com = np.nanmean(self.inertia_parameters.center_of_mass, axis=1)[:3]
            mass_center_elem = etree.SubElement(body_elem, "mass_center")
            mass_center_elem.text = f"{com[0]:.8f} {com[1]:.8f} {com[2]:.8f}"

            inertia_elem = etree.SubElement(body_elem, "inertia")
            i = self.inertia_parameters.inertia
            inertia_elem.text = f"{i[0,0]:.8f} {i[1,1]:.8f} {i[2,2]:.8f} {i[0,1]:.8f} {i[0,2]:.8f} {i[1,2]:.8f}"
        else:
            mass_elem = etree.SubElement(body_elem, "mass")
            mass_elem.text = f"0.00000000"

            mass_center_elem = etree.SubElement(body_elem, "mass_center")
            mass_center_elem.text = f"0.00000000 0.00000000 0.00000000"

            inertia_elem = etree.SubElement(body_elem, "inertia")
            inertia_elem.text = f"0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000"

        frame_geometry = etree.SubElement(body_elem, "FrameGeometry")
        etree.SubElement(frame_geometry, "socket_frame").text = ".."

        if with_mesh and self.mesh_file is not None:

            attached_geometry = etree.SubElement(body_elem, "attached_geometry")
            socket_frame = etree.SubElement(attached_geometry, "socket_frame")
            socket_frame.text = ".."

            mesh_elem = etree.SubElement(attached_geometry, "Mesh", name=f"{self.name}_mesh")

            mesh_file_elem = etree.SubElement(mesh_elem, "mesh_file")
            mesh_file_elem.text = self.mesh_file.mesh_file_name

            if self.mesh_file.mesh_scale is not None:
                scale_factors = etree.SubElement(mesh_elem, "scale_factors")
                s = self.mesh_file.mesh_scale
                scale_factors.text = f"{s[0,0]:.8f} {s[1,0]:.8f} {s[2,0]:.8f}"

            if self.mesh_file.mesh_color is not None:
                # TODO: add opacity
                appearance = etree.SubElement(mesh_elem, "Appearance")
                color_elem = etree.SubElement(appearance, "color")
                c = self.mesh_file.mesh_color
                color_elem.text = f"{c[0]:.8f} {c[1]:.8f} {c[2]:.8f}"

        return body_elem
