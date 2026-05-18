from typing import TYPE_CHECKING

import numpy as np

from ..abstract_model_writer import AbstractModelWriter

if TYPE_CHECKING:
    from ...components.real.biomechanical_model_real import BiomechanicalModelReal
    from ...components.real.rigidbody.segment_real import SegmentReal


class BvhModelWriter(AbstractModelWriter):
    """
    Write a :class:`BiomechanicalModelReal` into the BVH format.

    The current BVH export focuses on kinematic hierarchies only. Segment
    offsets and degree-of-freedom orders are preserved, while model features
    that BVH cannot represent directly (muscles, markers, contacts, IMUs,
    inertial parameters, meshes, and range limits) are intentionally rejected.
    """

    @staticmethod
    def _channel_names(segment: "SegmentReal") -> list[str]:
        """
        Convert biobuddy translation and rotation sequences into BVH channels.

        Parameters
        ----------
        segment
            The segment whose channels should be exported.
        """
        channels = []
        if segment.translations is not None and segment.translations.value is not None:
            channels.extend(f"{axis.upper()}position" for axis in segment.translations.value)
        if segment.rotations is not None and segment.rotations.value is not None:
            channels.extend(f"{axis.upper()}rotation" for axis in segment.rotations.value)
        return channels

    @staticmethod
    def _offset_string(segment: "SegmentReal") -> str:
        """
        Format the local segment translation into a BVH ``OFFSET`` line.

        Parameters
        ----------
        segment
            The segment whose local translation should be exported.
        """
        translation = segment.segment_coordinate_system.scs.translation
        return f"OFFSET {translation[0]:0.6f} {translation[1]:0.6f} {translation[2]:0.6f}"

    def _validate_segment(self, segment: "SegmentReal") -> None:
        """
        Ensure the segment can be represented in BVH.

        Parameters
        ----------
        segment
            The segment to validate.
        """
        if segment.segment_coordinate_system.is_in_global:
            raise RuntimeError(
                f"Something went wrong, the segment coordinate system of segment {segment.name} is expressed in the global."
            )

        rotation = segment.segment_coordinate_system.scs.rotation_matrix.rotation_matrix
        if not np.allclose(rotation, np.eye(3)):
            raise NotImplementedError(
                f"BVH export currently only supports identity local segment rotations. Segment {segment.name} is rotated."
            )

        unsupported_fields = {
            "markers": segment.nb_markers,
            "contacts": segment.nb_contacts,
            "imus": segment.nb_imus,
            "inertia parameters": segment.inertia_parameters is not None,
            "mesh": segment.mesh is not None,
            "mesh file": segment.mesh_file is not None,
            "q ranges": segment.q_ranges is not None,
            "qdot ranges": segment.qdot_ranges is not None,
        }
        for field_name, is_present in unsupported_fields.items():
            if is_present:
                raise NotImplementedError(
                    f"BVH export does not support segment {field_name}. Segment {segment.name} cannot be exported."
                )

    def _validate_model(self, model: "BiomechanicalModelReal") -> None:
        """
        Ensure the model can be represented in BVH.

        Parameters
        ----------
        model
            The model to validate before export.
        """
        if model.muscle_groups:
            raise NotImplementedError("BVH export does not support muscles.")
        if model.ligaments:
            raise NotImplementedError("BVH export does not support ligaments.")
        if model.gravity is not None:
            raise NotImplementedError("BVH export does not support gravity metadata.")

        for segment in model.segments:
            self._validate_segment(segment)

    @staticmethod
    def _children(model: "BiomechanicalModelReal", parent_name: str) -> list["SegmentReal"]:
        """
        Get the direct child segments of a parent segment.

        Parameters
        ----------
        model
            The model containing the hierarchy.
        parent_name
            The name of the parent segment.
        """
        return [segment for segment in model.segments if segment.parent_name == parent_name]

    def _write_joint(self, model: "BiomechanicalModelReal", segment: "SegmentReal", level: int, joint_type: str) -> str:
        """
        Write one BVH joint block recursively.

        Parameters
        ----------
        model
            The model containing the hierarchy.
        segment
            The segment to export.
        level
            The indentation level.
        joint_type
            The BVH joint type, either ``ROOT`` or ``JOINT``.
        """
        indent = "    " * level
        channels = self._channel_names(segment)
        children = self._children(model=model, parent_name=segment.name)

        lines = [
            f"{indent}{joint_type} {segment.name}",
            f"{indent}{{",
            f"{indent}    {self._offset_string(segment)}",
            f"{indent}    CHANNELS {len(channels)} {' '.join(channels)}" if channels else f"{indent}    CHANNELS 0",
        ]

        if children:
            for child in children:
                lines.append(self._write_joint(model=model, segment=child, level=level + 1, joint_type="JOINT"))
        else:
            lines.extend(
                [
                    f"{indent}    End Site",
                    f"{indent}    {{",
                    f"{indent}        OFFSET 0.000000 0.000000 0.000000",
                    f"{indent}    }}",
                ]
            )

        lines.append(f"{indent}}}")
        return "\n".join(lines)

    def _motion_line(self, model: "BiomechanicalModelReal") -> str:
        """
        Create one neutral BVH motion sample matching the exported channels.

        Parameters
        ----------
        model
            The model being exported.
        """
        values = []
        for segment in model.segments:
            values.extend(["0.000000"] * len(self._channel_names(segment)))
        return " ".join(values)

    def write(self, model: "BiomechanicalModelReal") -> None:
        """
        Write the model into a ``.bvh`` file.

        Parameters
        ----------
        model
            The model to export.
        """
        self._validate_model(model)

        root_candidates = self._children(model=model, parent_name="base")
        if len(root_candidates) != 1:
            raise RuntimeError("BVH export requires exactly one root segment attached to base.")

        root_segment = root_candidates[0]
        out_string = "HIERARCHY\n"
        out_string += self._write_joint(model=model, segment=root_segment, level=0, joint_type="ROOT")
        out_string += "\nMOTION\n"
        out_string += "Frames: 1\n"
        out_string += "Frame Time: 0.0333333\n"
        out_string += self._motion_line(model) + "\n"

        with open(self.filepath, "w") as file:
            file.write(out_string)
