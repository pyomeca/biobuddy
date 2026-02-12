from typing import TYPE_CHECKING

import numpy as np
from pathlib import Path
import subprocess

from ..utils.enums import Translations, Rotations

if TYPE_CHECKING:
    from .real.rigidbody.segment_real import SegmentReal


class ModelUtils:
    def __init__(self):

        # Attributes that will be filled by BiomechanicalModelReal
        self.segments = None
        self.muscle_groups = None

    @property
    def segment_names(self) -> list[str]:
        """
        Get the names of the segments in the model
        """
        return list(self.segments.keys())

    @property
    def dof_names(self) -> list[str]:
        names = []
        for segment in self.segments:
            names += segment.dof_names
        return names

    @property
    def marker_names(self) -> list[str]:
        list_marker_names = []
        for segment in self.segments:
            for marker in segment.markers:
                list_marker_names += [marker.name]
        return list_marker_names

    @property
    def contact_names(self) -> list[str]:
        list_contact_names = []
        for segment in self.segments:
            for contact in segment.contacts:
                list_contact_names += [contact.name]
        return list_contact_names

    @property
    def imu_names(self) -> list[str]:
        list_imu_names = []
        for segment in self.segments:
            for imu in segment.imus:
                list_imu_names += [imu.name]
        return list_imu_names

    @property
    def muscle_group_names(self) -> list[str]:
        """
        Get the names of the muscle groups in the model
        """
        return list(self.muscle_groups.keys())

    @property
    def muscle_names(self) -> list[str]:
        """
        Get the names of the muscles in the model
        """
        names = []
        for muscle_group in self.muscle_groups:
            for muscle in muscle_group.muscles:
                names.append(muscle.name)
        return names

    @property
    def via_point_names(self) -> list[str]:
        """
        Get the names of the via points in the model
        """
        names = []
        for muscle_group in self.muscle_groups:
            for muscle in muscle_group.muscles:
                for via_point in muscle.via_points:
                    names.append(via_point.name)
        return names

    def has_parent_offset(self, segment_name: str) -> bool:
        """True if the segment segment_name has an offset parent."""
        return segment_name + "_parent_offset" in self.segment_names

    @property
    def has_meshes(self) -> bool:
        """True if at least one segment has a mesh."""
        for segment in self.segments:
            if segment.mesh is not None:
                return True
        return False

    @property
    def has_mesh_files(self) -> bool:
        """True if at least one segment has a mesh file."""
        for segment in self.segments:
            if segment.mesh_file is not None:
                return True
        return False

    def segment_has_ghost_parents(self, segment_name: str) -> bool:
        """
        Check if the segment has ghost parents.
        A ghost parent is a segment that does not hold inertia, but is used to define the segment's coordinate system.
        """
        ghost_keys = [
            "_parent_offset",
            "_translation",
            "_rotation_transform",
            "_reset_axis",
        ]
        for key in ghost_keys:
            if segment_name + key in self.segments.keys():
                return True
        return False

    def children_segment_names(self, parent_name: str):
        children = []
        for segment_name in self.segments.keys():
            if self.segments[segment_name].parent_name == parent_name:
                children.append(segment_name)
        return children

    def get_chain_between_segments(self, first_segment_name: str, last_segment_name: str) -> list[str]:
        """
        Get the name of the segments in the kinematic chain between first_segment_name and last_segment_name.
        WARNING: This list does not include brother/sister segments.
        So for example, if s_geom_1 and s_geom_2 both have s_offset_parent as parent, and only s_geom_1 has a child,
        only s_geom_1 will  be included in the chain.
        """
        chain = []
        this_segment = last_segment_name
        while this_segment != first_segment_name:
            chain.append(this_segment)
            this_segment = self.segments[this_segment].parent_name
        chain.append(first_segment_name)
        chain.reverse()
        return chain

    def get_full_segment_chain(self, segment_name: str) -> list[str]:
        """
        Get the name of the segments in the complete fake kinematic chain derived from this segment.
        This includes all ghost segments if they exist, including brother/sister segments.
        """
        if segment_name + "_parent_offset" in self.segment_names:
            first_segment_name = segment_name + "_parent_offset"
        else:
            first_segment_name = segment_name
        last_segment_name = segment_name
        first_segment_index = self.segment_index(first_segment_name)
        last_segment_index = self.segment_index(last_segment_name)
        segment_list = self.segment_names[first_segment_index : last_segment_index + 1]

        # Check that the segments were in "order" (meaning that only parent-child relationships were used)
        for this_segment_name in segment_list.copy()[::-1]:
            if this_segment_name == first_segment_name:
                break
            if self.segments[this_segment_name].parent_name not in segment_list:
                raise NotImplementedError(
                    f"The segments in the model are not in the correct order to get the full segment chain for {segment_name}."
                )
        return segment_list

    def get_real_parent_name(self, segment_name: str) -> str:
        current_parent = self.segments[segment_name].parent_name
        while self.segment_has_ghost_parents(current_parent):
            current_parent = self.segments[current_parent].parent_name
        return current_parent

    @property
    def nb_segments(self) -> int:
        return len(self.segments)

    @property
    def nb_markers(self) -> int:
        return sum(segment.nb_markers for segment in self.segments)

    @property
    def nb_contacts(self) -> int:
        return sum(segment.nb_contacts for segment in self.segments)

    @property
    def nb_imus(self) -> int:
        return sum(segment.nb_imus for segment in self.segments)

    @property
    def nb_muscle_groups(self) -> int:
        return len(self.muscle_groups)

    @property
    def nb_muscles(self) -> int:
        nb = 0
        for muscle_group in self.muscle_groups:
            nb += len(muscle_group.muscles)
        return nb

    @property
    def nb_via_points(self) -> int:
        nb = 0
        for muscle_group in self.muscle_groups:
            for muscle in muscle_group.muscles:
                nb += len(muscle.via_points)
        return nb

    @property
    def nb_q(self) -> int:
        return sum(segment.nb_q for segment in self.segments)

    def segment_index(self, segment_name: str) -> int:
        return list(self.segments.keys()).index(segment_name)

    def dof_indices(self, segment_name: str) -> list[int]:
        """
        Get the indices of the degrees of freedom from the model

        Parameters
        ----------
        segment_name
            The name of the segment to get the indices for
        """
        nb_dof = 0
        for segment in self.segments:
            if segment.name != segment_name:
                if segment.translations != Translations.NONE:
                    nb_dof += len(segment.translations.value)
                if segment.rotations != Rotations.NONE:
                    nb_dof += len(segment.rotations.value)
            else:
                nb_translations = len(segment.translations.value) if segment.translations != Translations.NONE else 0
                nb_rotations = len(segment.rotations.value) if segment.rotations != Rotations.NONE else 0
                return list(range(nb_dof, nb_dof + nb_translations + nb_rotations))
        raise ValueError(f"Segment {segment_name} not found in the model")

    def dof_index(self, dof_name: str) -> int:
        """
        Get the index of a degree of freedom from the model.

        Parameters
        ----------
        dof_name
            The name of the degree of freedom to get the index for
        """
        return self.dof_names.index(dof_name)

    def dof_parent_segment_name(self, dof_name: str) -> str:
        """
        Get the name of the segment to which a degree of freedom belongs.

        Parameters
        ----------
        dof_name
            The name of the degree of freedom to get the parent segment for
        """
        for segment in self.segments:
            if dof_name in segment.dof_names:
                return segment.name
        raise ValueError(f"Degree of freedom {dof_name} not found in the model")

    def markers_indices(self, marker_names: list[str]) -> list[int]:
        """
        Get the indices of the markers of the model

        Parameters
        ----------
        marker_names
            The name of the markers to get the indices for
        """
        return [self.marker_names.index(marker) for marker in marker_names]

    def contact_indices(self, contact_names: list[str]) -> list[int]:
        """
        Get the indices of the contacts of the model

        Parameters
        ----------
        contact_names
            The name of the contacts to get the indices for
        """
        return [self.contact_names.index(contact) for contact in contact_names]

    def imu_indices(self, imu_names: list[str]) -> list[int]:
        """
        Get the indices of the imus of the model

        Parameters
        ----------
        imu_names
            The name of the imu to get the indices for
        """
        return [self.imu_names.index(imu) for imu in imu_names]

    @property
    def root_segment(self) -> "SegmentReal":
        """
        Get the root segment of the model, which is the segment with no parent.
        """
        for segment in self.segments:
            if segment.name == "root":
                return segment
        # TODO: make sure that the base segment is always defined
        # raise ValueError("No root segment found in the model. Please check your model.")

    @property
    def dofs(self) -> list[Translations | Rotations]:
        dofs = []
        for segment in self.segments:
            if segment.translations != Translations.NONE:
                dofs.append(segment.translations)
            if segment.rotations != Rotations.NONE:
                dofs.append(segment.rotations)
        return dofs

    @property
    def muscle_group_origin_parent_names(self):
        muscle_group_origins = []
        for mg in self.muscle_groups:
            muscle_group_origins.append(mg.origin_parent_name)
        return muscle_group_origins

    @property
    def muscle_group_insertion_parent_names(self):
        muscle_group_origins = []
        for mg in self.muscle_groups:
            muscle_group_origins.append(mg.insertion_parent_name)
        return muscle_group_origins

    def remove_dofs(self, dofs_to_remove: list[str]):
        """
        Remove the degrees of freedom from the model

        Parameters
        ----------
        dofs_to_remove: A list of the names of the degrees of freedom to remove
        """
        for dof_name in dofs_to_remove:
            for segment in self.segments:
                if dof_name in segment.dof_names:
                    segment.remove_dof(dof_name)

    def remove_muscles(self, muscles_to_remove: list[str]):
        """
        Remove the muscles from the model

        Parameters
        ----------
        muscles_to_remove: A list of the names of the muscles to remove
        """
        for muscle_group in self.muscle_groups.copy():
            for muscle in muscle_group.muscles.copy():
                if muscle.name in muscles_to_remove:
                    self.muscle_groups[muscle_group.name].remove_muscle(muscle.name)

    def update_muscle_groups(self):
        """
        Update the muscle groups to remove any empty muscle groups
        """
        original_muscle_groups = self.muscle_groups.copy()
        for muscle_group in original_muscle_groups:
            if len(muscle_group.muscles) == 0:
                self.remove_muscle_group(muscle_group.name)

    def update_segments(self):
        """
        Update the segments to remove empty segments.
        If a segment has no markers, no contacts and no imus, no dof, no mesh, no mesh file, no inertia parameters,
        it is removed from the model and the kinematic chain is updated rto reflect this change.
        TODO: I removed only segments that also do not have any RT, but this should not be a criteria (I should carry the RT).
        """
        original_segments = self.segments.copy()
        for segment in original_segments:
            no_inertia = segment.inertia_parameters is None or segment.inertia_parameters.mass <= 0.0001
            if (
                segment.nb_markers == 0
                and segment.nb_contacts == 0
                and segment.nb_imus == 0
                and segment.nb_q == 0
                and segment.mesh is None
                and segment.mesh_file is None
                and no_inertia
                and segment.segment_coordinate_system.scs.is_identity
            ):
                # Except the root segment
                if segment.name == "root":
                    continue
                # Update the kinematic chain
                child_segments = self.children_segment_names(segment.name)
                for child_name in child_segments:
                    self.segments[child_name].parent_name = segment.parent_name
                # Remove the segment
                self.remove_segment(segment.name)

    def modify_model_static_pose(self, q_static: np.ndarray):
        from .real.rigidbody.segment_coordinate_system_real import (
            SegmentCoordinateSystemReal,
        )

        if q_static.shape != (self.nb_q,):
            raise RuntimeError(f"The shape of q_static must be (nb_q, ), you have {q_static.shape}.")

        # Find the joint coordinate systems in the global frame in the new static pose
        jcs_in_global = self.forward_kinematics(q_static)
        for i_segment, segment_name in enumerate(self.segments.keys()):
            self.segments[segment_name].segment_coordinate_system = SegmentCoordinateSystemReal(
                scs=jcs_in_global[segment_name][0],  # We can that the 0th since there is just one frame in q_original
                is_scs_local=(
                    segment_name == "base"
                ),  # joint coordinate system is now expressed in the global except for the base because it does not have a parent
            )

        # Replace the jsc in local reference frames
        self.segments_rt_to_local()

    def get_dof_ranges(self) -> np.ndarray:
        """
        Returns the min_bound and max_bound of the degrees of freedom of the model (2 x nb_q).
        """
        ranges = np.empty((2, 0))
        for segment in self.segments:
            if segment.nb_q > 0 and segment.q_ranges is not None:
                min_bound = segment.q_ranges.min_bound
                max_bound = segment.q_ranges.max_bound
                bound = np.vstack((min_bound, max_bound))
                ranges = np.hstack((ranges, bound))
        return ranges

    def change_mesh_directories(self, new_directory: str):
        """
        Change the mesh file directory for all segments in the model.

        Parameters
        ----------
        new_directory
            The new directory to set for the mesh files.
        """
        for segment in self.segments:
            if segment.mesh_file is not None:
                segment.mesh_file.mesh_file_directory = new_directory

    def write_graphviz(
        self,
        out_path: str,
        include_ghost_segments: bool = True,
        include_dof_segments: bool = True,
        include_via_points: bool = True,
        include_markers: bool = True,
        include_legend: bool = True,
    ) -> None:
        """
        Write a graphviz .dot file representing the biomechanical model structure

        Parameters
        ----------
        out_path
            path to save the .dot file
        include_ghost_segments
            whether to include segments without inertia parameters or markers as "ghost" segments (empty boxes)
        include_dof_segments
            whether to include segments with DOFs as "dof" segments (boxes with diagonals)
        include_via_points
            whether to include via points in the graph
        include_markers
            whether to include markers in the graph
        include_legend
            whether to include legend in the graph
        """

        def convert_dot_to_png(path_dot_file: Path | str) -> None:
            path_dot_file = Path(path_dot_file)

            if path_dot_file.suffix != ".dot":
                path_dot_file = path_dot_file.with_suffix(".dot")

            if not path_dot_file.exists():
                raise FileNotFoundError(f"DOT file not found: {path_dot_file}")

            png_file = path_dot_file.with_suffix(".png")

            subprocess.run(["dot", "-Tpng", str(path_dot_file), "-o", str(png_file)], check=True)

            print(f"A graph has been created here: {png_file}")

            return

        out_path = Path(out_path)

        if out_path.suffix != ".dot":
            out_path = out_path.with_suffix(".dot")

        with open(out_path, "w") as f:
            f.write("digraph biomech {\n")
            f.write("  rankdir=TB;\n")
            f.write('  node [fontname="Helvetica"];\n\n')

            # title
            title = Path(out_path).stem
            f.write('  labelloc="t";\n')
            f.write(f'  label=<<B><FONT POINT-SIZE="28">BioMod Model Graph - {title}</FONT></B>>;\n\n')

            # -------- SEGMENTS --------
            # TODO : add specific box for segments with contact points
            f.write("  // Segments\n")
            true_segments = []
            for s in self.segments:
                if include_dof_segments and s.dof_names:
                    label = f"{s.name}\\n(DOF T: {s.translations.name.lower()} | R: {s.rotations.name.lower()})"
                    f.write(
                        f'  "{s.name}" [shape=box, style="filled,diagonals", fillcolor=lightyellow, label="{label}"];\n'
                    )
                    true_segments.append(s.name)
                elif (
                    s.inertia_parameters
                    or (s.markers and include_markers)
                    or s.name in self.muscle_group_origin_parent_names
                    or s.name in self.muscle_group_insertion_parent_names
                ):
                    label = s.name
                    if s.inertia_parameters:
                        label += r"\n(inertia)"

                    f.write(f'  "{s.name}" [shape=box, style="filled,bold", fillcolor=lightgray, label="{label}"];\n')
                    true_segments.append(s.name)

                elif include_ghost_segments:
                    f.write(f'  "{s.name}" [shape=box, style=rounded];\n')

                else:
                    continue

            f.write("\n  // Segment hierarchy\n")

            if include_ghost_segments:
                for s_name in self.segment_names:
                    for children_name in self.children_segment_names(s_name):
                        f.write(f'  "{s_name}" -> "{children_name}";\n')
            else:
                for s_name in true_segments[::-1]:
                    s_parent_name = self.segments[s_name].parent_name
                    if s_parent_name == "root":
                        break
                    while True:
                        if s_parent_name in true_segments:
                            f.write(f'  "{s_parent_name}" -> "{s_name}";\n')
                            break
                        else:
                            if s_parent_name == "base":
                                break
                            s_parent_name = self.segments[s_parent_name].parent_name
                            if s_parent_name == "root":
                                break

            # -------- MARKERS --------
            if include_markers:
                f.write("\n  // Markers\n")
                for marker_name in self.marker_names:
                    f.write(f'  "{marker_name}" [shape=octagon, style=filled, fillcolor=lightgreen];\n')

                for s in self.segments:
                    for marker in s.markers:
                        f.write(f'  "{s.name}" -> "{marker.name}";\n')

            # -------- MUSCLE GROUPS --------
            f.write("\n  // Muscle groups\n")
            for mg in self.muscle_groups:
                origin = mg.origin_parent_name
                insertion = mg.insertion_parent_name

                label = f"{mg.name}\\n({origin} → {insertion})"
                f.write(f'  "{mg.name}" [shape=parallelogram, style=filled, fillcolor=lightblue, label="{label}"];\n')
                # origin ---> insertion
                if origin:
                    f.write(f'  "{origin}" -> "{mg.name}" [label="origin"];\n')
                if insertion:
                    f.write(f'  "{mg.name}" -> "{insertion}" [label="insertion"];\n')

            # -------- MUSCLES --------
            f.write("\n  // Muscles\n")
            for mg in self.muscle_groups:
                for m_name in mg.muscles.keys():
                    f.write(f'  "{m_name}" [shape=diamond, style=filled, fillcolor=lightcoral];\n')
                    f.write(f'  "{mg.name}" -> "{m_name}";\n')

            if include_via_points:
                # -------- VIAPOINTS --------
                f.write("\n  // Via points\n")
                for vp_name in self.via_point_names:
                    f.write(f'  "{vp_name}" [shape=ellipse, style=filled, fillcolor=orange];\n')

                # -------- MUSCLE --> VIAPOINT CHAINS --------
                f.write("\n  // Muscle paths\n")

                for mg in self.muscle_groups:
                    for m in mg.muscles:
                        if not m.via_points:
                            continue
                        # muscle -> first VP
                        f.write(f'  "{m.name}" -> "{m.via_points[0].name}" [label="via"];\n')

                        # other VP
                        for i in range(len(m.via_points) - 1):
                            f.write(f'  "{m.via_points[i].name}" -> "{m.via_points[i+1].name}" [label="via"];\n')

            if include_legend:
                # -------- LEGENDS --------
                f.write("\n  // Legend\n")
                keys = []
                f.write(
                    '  subgraph cluster_legend {\n    label="Legend";\n    fontsize=16;\n    style="rounded,dashed";'
                )

                f.write('\n    key_segment [label="Segment", shape=box, style="filled,bold", fillcolor=lightgray]')
                keys.append("key_segment")

                if include_dof_segments:
                    f.write(
                        '\n    key_segment_dof   [label="DOF segment \\n(DOF T: (x,y,z) | R: (x,y,z))", shape=box, style="filled,diagonals", fillcolor=lightyellow];'
                    )
                    keys.append("key_segment_dof")

                if include_ghost_segments:
                    f.write('\n    key_segment_ghost [label="Ghost segment", shape=box, style=rounded]')
                    keys.append("key_segment_ghost")

                if include_markers:
                    f.write('\n    key_marker    [label="Marker", shape=octagon, style=filled, fillcolor=lightgreen];')
                    keys.append("key_marker")

                f.write(
                    '\n    key_muscle_groups   [label="Muscle group\\n(origin → insertion)", shape=parallelogram, style=filled, fillcolor=lightblue];'
                )
                keys.append("key_muscle_groups")
                nb_column = len(keys) - 1

                f.write('\n    key_muscles   [label="Muscle", shape=diamond, style=filled, fillcolor=lightcoral];')
                keys.append("key_muscles")
                if include_via_points:
                    f.write(
                        '\n    key_via_point   [label="Via point", shape=ellipse, style=filled, fillcolor=orange]; '
                    )
                    keys.append("key_via_point")

                # -----------------
                f.write("\n    { rank=same; " + " ".join(keys[:nb_column]) + " }")
                f.write("\n    { rank=same; " + " ".join(keys[nb_column:]) + " }")

                f.write(f"\n    key_segment -> key_muscle_groups [style=invis];")

                f.write("\n  }")

            f.write("\n}\n")

        print(f"A graph have been created for the model here : {out_path}")

        convert_dot_to_png(out_path)
