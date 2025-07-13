import os
from biobuddy import BiomechanicalModelReal, BiomechanicalModel


def destroy_model(bio_model: BiomechanicalModelReal | BiomechanicalModel):
    """
    Let's test the remove functions and make sure that there is nothing left in the model.
    """

    # Remove segments
    for segment_name in bio_model.segment_names:

        # Remove markers
        marker_names = bio_model.segments[segment_name].marker_names
        for marker_name in marker_names:
            bio_model.segments[segment_name].remove_marker(marker_name)
        assert bio_model.segments[segment_name].nb_markers == 0

        # Remove contacts
        contact_names = bio_model.segments[segment_name].contact_names
        for contact_name in contact_names:
            bio_model.segments[segment_name].remove_contact(contact_name)
        assert bio_model.segments[segment_name].nb_contacts == 0

        # Remove imus
        imu_names = bio_model.segments[segment_name].imu_names
        for imu_name in imu_names:
            bio_model.segments[segment_name].remove_imu(imu_name)
        assert bio_model.segments[segment_name].nb_imus == 0

        # Remove segment
        bio_model.remove_segment(segment_name)
    assert bio_model.nb_segments == 0
    assert bio_model.segment_names == []

    # Remove muscle groups
    for muscle_group_name in bio_model.muscle_group_names:
        bio_model.remove_muscle_group(muscle_group_name)
    assert bio_model.nb_muscle_groups == 0
    assert bio_model.muscle_group_names == []

    # Remove muscles
    for muscle_name in bio_model.muscle_names:
        bio_model.remove_muscle(muscle_name)
    assert bio_model.nb_muscles == 0
    assert bio_model.muscle_names == []

    # Remove via points
    for via_point_name in bio_model.via_point_names:
        bio_model.remove_via_point(via_point_name)
    assert bio_model.nb_via_points == 0
    assert bio_model.via_point_names == []


def remove_temporary_biomods():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_names_to_remove = ["temporary.bioMod", "temporary_rt.bioMod"]

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in file_names_to_remove:
                full_path = os.path.join(dirpath, filename)
                try:
                    os.remove(full_path)
                except:
                    print(f"File {full_path} could not be deleted.")
