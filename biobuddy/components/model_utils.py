from ..utils.enums import Translations, Rotations


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

    def children_segment_names(self, parent_name: str):
        children = []
        for segment_name in self.segments.keys():
            if self.segments[segment_name].parent_name == parent_name:
                children.append(segment_name)
        return children

    def get_chain_between_segments(self, first_segment_name: str, last_segment_name: str) -> list[str]:
        """
        Get the name of the segments in the kinematic chain between first_segment_name and last_segment_name
        """
        chain = []
        this_segment = last_segment_name
        while this_segment != first_segment_name:
            chain.append(this_segment)
            this_segment = self.segments[this_segment].parent_name
        chain.append(first_segment_name)
        chain.reverse()
        return chain

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

    @property
    def dof_names(self) -> list[str]:
        names = []
        for segment in self.segments:
            names += segment.dof_names
        return names

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

    def degrees_of_freedom(self) -> list[Translations | Rotations]:
        dofs = []
        for segment in self.segments:
            if segment.translations != Translations.NONE:
                dofs.append(segment.translations)
            if segment.rotations != Rotations.NONE:
                dofs.append(segment.rotations)
        return dofs

    def update_muscle_groups(self):
        """
        Update the muscle groups to make sure there are no empty muscle groups
        """
        original_muscle_groups = self.muscle_groups.copy()
        for muscle_group in original_muscle_groups:
            if len(muscle_group.muscles) == 0:
                self.remove_muscle_group(muscle_group.name)






