from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from biobuddy import (
    BiomechanicalModelReal,
    DictData,
)
from biobuddy.components.generic.rigidbody.axis import Axis
from biobuddy.components.real.rigidbody.segment_coordinate_system_real import (
    SegmentCoordinateSystemReal,
)
from biobuddy.gui.c3d_creation_workflow import (
    C3dSegmentSettingsDraft,
    add_axis_to_draft,
    add_segment_to_draft,
    assign_c3d_file_role_to_draft,
    c3d_workflow_draft,
    c3d_workflow_progress,
    remove_axis_from_draft,
    update_segment_settings_in_draft,
)
from biobuddy.gui.c3d_model_creation import C3dModelPreset
from biobuddy.gui.model_editor import (
    _complete_marker_frame_count,
    _c3d_frame_rate,
    _c3d_file_names_from_folder,
    _c3d_generation_log,
    _export_model_to_path,
    _joint_name_from_segments,
    _marker_frame_position,
    _matching_c3d_file_for_expected_name,
    _mean_marker_series,
    _marker_name_mapping_for_c3d,
    _is_virtual_feature_axis,
    _anatomical_axis_source_labels,
    _apply_initial_rotation_setting_to_segment,
    _axis_source_name_from_list_text,
    _orthonormal_axes_from_vector_segments,
    _predictive_virtual_marker_method_from_label,
    _preview_camera_coordinates,
    _preview_camera_matrix_for_plane,
    _preview_camera_matrix_for_subject_view,
    _python_code_from_c3d_draft,
    _fit_projection,
    _rab2002_geometry,
    _rab2002_markers_from_payload,
    _remap_c3d_workflow_draft_markers,
    _axis_projection_axis_from_payload,
    _axis_projection_point_markers_from_payload,
    _strip_participant_prefix_from_c3d_data,
    _strip_participant_prefix_from_marker_names,
    _score_segments_from_payload,
    _segment_length_from_draft,
    _segment_length_marker_groups,
    _split_marker_names,
    _strip_score_segment_payload,
    _source_with_c3d_assignment,
    _c3d_source_name_from_virtual_feature_source,
    _trial_name_from_virtual_feature_source,
    _virtual_axis_name_from_feature_list_text,
    _virtual_feature_list_labels,
    _virtual_marker_preview_marker_names_to_draw,
    _workflow_playback_timer_interval_ms,
    _marker_pool_from_draft,
    _model_export_extension_from_label,
    _model_export_filepath_with_extension,
    _model_export_filter_for_extension,
    _parse_rotation_matrix_text,
    _q0_initial_rotation_by_segment,
    _rotation_segment_name_for_model,
    _unassigned_marker_names,
)
from biobuddy.gui.lower_limb_template import lower_limb_template
from biobuddy.gui.model_builder import build_generic_model
from biobuddy.gui.motive_57_isb_template import motive_57_isb_template
from biobuddy.model_modifiers.functional_frame_selection import (
    FunctionalFrameSelectionOptions,
    functional_frame_selection_report,
    prepare_functional_rt_pair,
)
from biobuddy.gui.segment_editor import load_model
from biobuddy.utils.linear_algebra import RotoTransMatrix, RotoTransMatrixTimeSeries


def test_parse_rotation_matrix_text_accepts_common_3x3_formats():
    expected = ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))

    assert _parse_rotation_matrix_text("[[1, 0, 0], [0, 0, -1], [0, 1, 0]]") == expected
    assert _parse_rotation_matrix_text("1,0,0; 0,0,-1; 0,1,0") == expected
    assert _parse_rotation_matrix_text("1 0 0 0 0 -1 0 1 0") == expected


def test_parse_rotation_matrix_text_rejects_invalid_shapes_and_values():
    for text in (
        "",
        "[[1, 0], [0, 1]]",
        "1 2 3 4",
        "[[1, 0, 'x'], [0, 1, 0], [0, 0, 1]]",
    ):
        with pytest.raises(ValueError):
            _parse_rotation_matrix_text(text)


def test_chain_export_format_helpers_return_selected_extension_and_filter():
    assert _model_export_extension_from_label("BioMod (.bioMod)") == ".bioMod"
    assert _model_export_extension_from_label("BVH (.bvh)") == ".bvh"
    assert _model_export_extension_from_label("OpenSim (.osim)") == ".osim"
    assert _model_export_extension_from_label("URDF (.urdf)") == ".urdf"
    assert _model_export_filter_for_extension(".bvh").startswith("BVH files")


def test_chain_export_filepath_accepts_file_or_folder(tmp_path):
    assert _model_export_filepath_with_extension(str(tmp_path / "custom_name"), ".bioMod", "motive_57") == str(
        tmp_path / "custom_name.bioMod"
    )
    assert _model_export_filepath_with_extension(str(tmp_path), ".bioMod", "motive_57") == str(
        tmp_path / "motive_57.bioMod"
    )


def test_q0_initial_rotation_uses_identity_unless_a_matrix_is_requested():
    draft = c3d_workflow_draft(C3dModelPreset.MOTIVE_57)
    requested_rotation = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    draft = update_segment_settings_in_draft(
        draft,
        "Thorax",
        translations="",
        rotations="xyz",
        initial_rotation_method="identity",
    )
    draft = update_segment_settings_in_draft(
        draft,
        "LShank",
        translations="",
        rotations="z",
        initial_rotation_method="matrix",
        initial_rotation_source="[[0,-1,0], [1,0,0], [0,0,1]]",
        initial_rotation_matrix=requested_rotation,
    )

    rotations = _q0_initial_rotation_by_segment(draft)

    np.testing.assert_allclose(rotations["Thorax"], np.eye(3))
    np.testing.assert_allclose(rotations["LShank"], np.asarray(requested_rotation))


def test_identity_initial_rotation_replaces_exported_segment_rt_rotation_but_keeps_translation():
    original_rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    translation = np.array([1.0, 2.0, 3.0])
    segment = SimpleNamespace(
        segment_coordinate_system=SegmentCoordinateSystemReal(
            scs=RotoTransMatrix.from_rotation_matrix_and_translation(original_rotation, translation),
            is_scs_local=True,
        )
    )
    setting = C3dSegmentSettingsDraft(segment_name="Thorax", initial_rotation_method="identity")

    _apply_initial_rotation_setting_to_segment(segment, setting)

    np.testing.assert_allclose(segment.segment_coordinate_system.scs.rotation_matrix.rotation_matrix, np.eye(3))
    np.testing.assert_allclose(segment.segment_coordinate_system.scs.translation, translation)


def test_identity_initial_rotation_preserves_exported_segment_rt_rotation_for_aor_segments():
    original_rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    translation = np.array([1.0, 2.0, 3.0])
    segment = SimpleNamespace(
        segment_coordinate_system=SegmentCoordinateSystemReal(
            scs=RotoTransMatrix.from_rotation_matrix_and_translation(original_rotation, translation),
            is_scs_local=True,
        )
    )
    setting = C3dSegmentSettingsDraft(segment_name="LShank", initial_rotation_method="identity")
    draft = c3d_workflow_draft(C3dModelPreset.MOTIVE_57)

    _apply_initial_rotation_setting_to_segment(segment, setting, draft)

    np.testing.assert_allclose(
        segment.segment_coordinate_system.scs.rotation_matrix.rotation_matrix,
        original_rotation,
    )
    np.testing.assert_allclose(segment.segment_coordinate_system.scs.translation, translation)


def test_identity_initial_rotation_preserves_fixed_anatomical_frame_after_joint_segment():
    original_rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    translation = np.array([1.0, 2.0, 3.0])
    segment = SimpleNamespace(
        segment_coordinate_system=SegmentCoordinateSystemReal(
            scs=RotoTransMatrix.from_rotation_matrix_and_translation(original_rotation, translation),
            is_scs_local=True,
        )
    )
    setting = C3dSegmentSettingsDraft(segment_name="LShank", initial_rotation_method="identity")
    draft = c3d_workflow_draft(C3dModelPreset.MOTIVE_57_ISB)

    _apply_initial_rotation_setting_to_segment(segment, setting, draft)

    np.testing.assert_allclose(
        segment.segment_coordinate_system.scs.rotation_matrix.rotation_matrix,
        original_rotation,
    )
    np.testing.assert_allclose(segment.segment_coordinate_system.scs.translation, translation)


def test_reconstruction_resolves_physical_segment_to_joint_dofs():
    model = build_generic_model(motive_57_isb_template())

    assert _rotation_segment_name_for_model(model, "LShank") == "LKneeJoint"
    assert _rotation_segment_name_for_model(model, "RFoot") == "RAnkleJoint"
    assert _rotation_segment_name_for_model(model, "LUpperArm") == "LShoulderJoint"
    assert _rotation_segment_name_for_model(model, "RForearm") == "RElbowJoint"
    assert _rotation_segment_name_for_model(model, "LHand") == "LWristJoint"
    assert _rotation_segment_name_for_model(model, "Thorax") == "Thorax"


def test_load_model_supports_bvh(tmp_path):
    """
    Load a minimal BVH file through the GUI-facing loader.
    """
    filepath = tmp_path / "minimal.bvh"
    filepath.write_text(
        "\n".join(
            [
                "HIERARCHY",
                "ROOT Hips",
                "{",
                "    OFFSET 0 0 0",
                "    CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation",
                "}",
                "MOTION",
                "Frames: 1",
                "Frame Time: 0.01",
                "0 0 0 0 0 0",
            ]
        )
    )

    model = load_model(str(filepath))

    assert isinstance(model, BiomechanicalModelReal)
    assert "Hips" in model.segment_names


def test_virtual_marker_editor_payload_helpers_preserve_score_settings():
    """
    Parse the compact GUI payload used for SCoRE/SARA proximal and distal segment settings.
    """
    payload = "proximal=PelvisTech; distal=ThighTech; helper=condyles=ME,LE"

    assert _score_segments_from_payload(payload) == ("PelvisTech", "ThighTech")
    assert _strip_score_segment_payload(payload) == "condyles=ME,LE"
    assert _split_marker_names("LASI, RASI; LPSI") == ("LASI", "RASI", "LPSI")


def test_workflow_playback_interval_uses_c3d_frame_rate():
    class FakeC3dData:
        frame_rate = 100.0

    assert _c3d_frame_rate(FakeC3dData()) == 100.0
    assert _workflow_playback_timer_interval_ms(FakeC3dData()) == 10
    assert _workflow_playback_timer_interval_ms(None) == 33


def test_matching_c3d_file_accepts_short_motive_functional_names(tmp_path):
    p6_lhip = tmp_path / "P6_LHip.c3d"
    p6_lhip.write_text("", encoding="utf-8")

    assert _matching_c3d_file_for_expected_name(tmp_path, "*Func_LHip.c3d") == p6_lhip


def test_segment_length_uses_selected_segment_and_child_origins():
    """
    Estimate segment length from the anatomical origin of a segment and its child.
    """
    draft = c3d_workflow_draft(C3dModelPreset.FROM_SCRATCH)
    draft = add_segment_to_draft(draft, "Thigh")
    draft = add_segment_to_draft(draft, "Shank", parent_name="Thigh")
    draft = add_axis_to_draft(
        draft,
        name="Thigh_axis",
        segment_name="Thigh",
        axis="z",
        start_markers=("Hip",),
        end_markers=("Knee",),
        origin_markers=("Hip",),
    )
    draft = add_axis_to_draft(
        draft,
        name="Shank_axis",
        segment_name="Shank",
        axis="z",
        start_markers=("Knee",),
        end_markers=("Ankle",),
        origin_markers=("Knee",),
    )
    c3d_data = DictData(
        {
            "Hip": np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]),
            "Knee": np.array([[0.0, 0.0], [3.0, 3.0], [4.0, 4.0], [1.0, 1.0]]),
            "Ankle": np.array([[0.0, 0.0], [6.0, 6.0], [8.0, 8.0], [1.0, 1.0]]),
        }
    )

    proximal_markers, distal_markers, source = _segment_length_marker_groups(draft, "Thigh")
    length, length_source = _segment_length_from_draft(draft, c3d_data, "Thigh")

    assert proximal_markers == ("Hip",)
    assert distal_markers == ("Knee",)
    assert source == "proximal=Hip; distal=Knee; child=Shank"
    assert length == 5.0
    assert length_source == source


def test_virtual_marker_editor_suggests_joint_names_from_segment_pairs():
    """
    Infer understandable virtual marker names from proximal/distal technical segment names.
    """
    assert _joint_name_from_segments("Pelvis", "LThigh") == "Left_Hip"
    assert _joint_name_from_segments("LThigh", "LShank") == "Left_Knee"


def test_lower_limb_functional_template_uses_correct_anatomical_axes():
    """
    Keep the lower-limb functional template aligned with the intended anatomical frame definitions.
    """
    template = lower_limb_template(use_functional=True)
    segments = {segment.name: segment for segment in template.segments}

    for side in ("L", "R"):
        thigh_frame = segments[f"{side}Thigh"].frame
        shank_frame = segments[f"{side}Shank"].frame
        foot_frame = segments[f"{side}Foot"].frame

        assert thigh_frame.first_axis.name == Axis.Name.Z
        assert thigh_frame.second_axis.fallback.name == Axis.Name.X
        assert thigh_frame.axis_to_keep == Axis.Name.Z

        assert shank_frame.first_axis.name == Axis.Name.Z
        assert shank_frame.second_axis.fallback.name == Axis.Name.X
        assert shank_frame.axis_to_keep == Axis.Name.X

        assert foot_frame.first_axis.name == Axis.Name.Y
        assert foot_frame.second_axis.name == Axis.Name.X
        assert foot_frame.axis_to_keep == Axis.Name.Y


def test_virtual_marker_axis_list_text_can_remove_axis_from_draft():
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)
    axis_list_text = "[axis] Axis_LKnee_SARA | LShank | sara | trial=left_knee_sara"

    axis_name = _virtual_axis_name_from_feature_list_text(axis_list_text)
    updated_draft = remove_axis_from_draft(draft, axis_name)

    assert axis_name == "Axis_LKnee_SARA"
    assert _virtual_axis_name_from_feature_list_text("CoR_LThigh_in_Pelvis | LThigh | score") is None
    assert any(axis.name == "Axis_LKnee_SARA" for axis in draft.axes)
    assert all(axis.name != "Axis_LKnee_SARA" for axis in updated_draft.axes)


def test_virtual_marker_list_shows_only_named_sara_virtual_axes():
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)
    named_sara_axis = next(axis for axis in draft.axes if axis.name == "Axis_LKnee_SARA")
    anatomical_sara_axis = next(axis for axis in draft.axes if axis.name == "LShank_second_axis")

    assert _is_virtual_feature_axis(named_sara_axis)
    assert not _is_virtual_feature_axis(anatomical_sara_axis)


def test_virtual_feature_list_shows_sara_axes_first():
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)
    labels = _virtual_feature_list_labels(draft)

    assert labels[0].startswith("[axis] Axis_LKnee_SARA")
    assert "| AoR (sara_direction) |" in labels[0]
    assert labels[1].startswith("[axis] Axis_RKnee_SARA")
    assert "| AoR (sara_direction) |" in labels[1]
    assert any(label.startswith("Proj_LKnee_on_Axis_LKnee_SARA") for label in labels)


def test_motive_57_virtual_feature_list_shows_knee_aors():
    draft = c3d_workflow_draft(C3dModelPreset.MOTIVE_57)
    labels = _virtual_feature_list_labels(draft)

    assert any(
        label.startswith("[axis] Axis_LKnee_SARA") and "| LShank | AoR (sara_direction) |" in label for label in labels
    )
    assert any(
        label.startswith("[axis] Axis_RKnee_SARA") and "| RShank | AoR (sara_direction) |" in label for label in labels
    )


def test_motive_57_unassigned_markers_remain_available_to_gui():
    draft = c3d_workflow_draft(C3dModelPreset.MOTIVE_57)

    marker_pool = _marker_pool_from_draft(draft)
    unassigned_markers = _unassigned_marker_names(marker_pool, draft.segment_marker_groups)

    assert {"RCAJ", "LCAJ", "RHGT", "LHGT", "RDP1", "LDP1"} <= set(marker_pool)
    assert {"RCAJ", "LCAJ", "RHGT", "LHGT", "RDP1", "LDP1"} <= set(unassigned_markers)


def test_preview_projection_keeps_positive_z_visually_up():
    _x, projected_low_y, _depth = _preview_camera_coordinates((0.0, 0.0, 0.0), 0.0, 0.0)
    _x, projected_high_y, _depth = _preview_camera_coordinates((0.0, 0.0, 1.0), 0.0, 0.0)
    transform = _fit_projection(
        [(0.0, projected_low_y), (0.0, projected_high_y)],
        100,
        100,
        lambda x, y: (x, y),
    )

    assert projected_high_y > projected_low_y
    assert transform((0.0, projected_high_y))[1] < transform((0.0, projected_low_y))[1]


def test_preview_projection_ignores_non_finite_points():
    transform = _fit_projection(
        [(0.0, 0.0), (np.nan, 1.0), (np.inf, 2.0)],
        100,
        80,
        lambda x, y: (x, y),
    )

    assert np.all(np.isfinite(transform((0.0, 0.0))))
    assert np.all(np.isfinite(transform((np.nan, np.inf))))


def test_preview_camera_matrix_for_standard_planes():
    point = (1.0, 2.0, 3.0)

    assert _preview_camera_coordinates(point, _preview_camera_matrix_for_plane("XY"), 0.0) == (1.0, 2.0, 3.0)
    assert _preview_camera_coordinates(point, _preview_camera_matrix_for_plane("YZ"), 0.0) == (2.0, 3.0, 1.0)
    assert _preview_camera_coordinates(point, _preview_camera_matrix_for_plane("ZX"), 0.0) == (3.0, 1.0, 2.0)


def test_preview_camera_matrix_for_subject_views_uses_global_vertical_and_pca():
    markers = {
        "A": np.asarray((-3.0, 0.0, 0.0)),
        "B": np.asarray((3.0, 0.0, 0.0)),
        "C": np.asarray((-3.0, 0.0, 2.0)),
        "D": np.asarray((3.0, 0.0, 2.0)),
        "E": np.asarray((0.0, 0.8, 1.0)),
        "F": np.asarray((0.0, -0.8, 1.0)),
    }
    point = (1.0, 2.0, 3.0)

    assert _preview_camera_coordinates(point, _preview_camera_matrix_for_subject_view("face", markers), 0.0) == (
        1.0,
        3.0,
        2.0,
    )
    assert _preview_camera_coordinates(point, _preview_camera_matrix_for_subject_view("dos", markers), 0.0) == (
        -1.0,
        3.0,
        -2.0,
    )
    assert _preview_camera_coordinates(point, _preview_camera_matrix_for_subject_view("cote", markers), 0.0) == (
        2.0,
        3.0,
        1.0,
    )


def test_mean_marker_series_preserves_frame_axis_for_sara_preview():
    class FakeC3dData:
        def markers_center_position(self, marker_names):
            assert marker_names == ("A", "B")
            return np.asarray(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [1.0, 1.0, 1.0],
                ]
            )

    series = _mean_marker_series(FakeC3dData(), ("A", "B"))

    assert series.shape == (3, 3)
    np.testing.assert_allclose(series[:, 0], (1.0, 4.0, 7.0))
    np.testing.assert_allclose(series[:, 2], (3.0, 6.0, 9.0))


def test_virtual_marker_whole_body_preview_keeps_all_markers_while_dragging():
    marker_names = ("Pelvis", "Thorax", "Head", "Foot")
    highlighted_marker_names = {"Pelvis", "Foot"}

    assert _virtual_marker_preview_marker_names_to_draw(
        marker_names, highlighted_marker_names, show_whole_body=True, is_dragging=True
    ) == set(marker_names)
    assert (
        _virtual_marker_preview_marker_names_to_draw(
            marker_names,
            highlighted_marker_names,
            show_whole_body=False,
            is_dragging=True,
        )
        == highlighted_marker_names
    )


def test_lower_limb_functional_sara_axes_do_not_trigger_missing_xyz_warning():
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)

    coordinate_system_step = next(step for step in c3d_workflow_progress(draft) if step.number == 6)

    assert coordinate_system_step.status == "done"
    assert "functional reference" in coordinate_system_step.detail
    assert "Axis_LKnee_SARA" in coordinate_system_step.detail
    assert "Axis_RKnee_SARA" in coordinate_system_step.detail


def test_anatomical_axis_preview_builds_rgb_local_frame_from_two_vectors():
    axes = _orthonormal_axes_from_vector_segments(
        (
            ("x", True, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            ("y", False, (0.0, 0.0, 0.0), (1.0, 2.0, 0.0)),
        )
    )

    assert np.allclose(axes["x"], (1.0, 0.0, 0.0))
    assert np.allclose(axes["y"], (0.0, 1.0, 0.0))
    assert np.allclose(axes["z"], (0.0, 0.0, 1.0))


def test_axis_projection_payload_parses_point_and_axis_sources():
    point_markers = _axis_projection_point_markers_from_payload("point=LKNE,LKNEM")
    axis_reference, axis_start, axis_end = _axis_projection_axis_from_payload("axis=Axis_LKnee_SARA")
    marker_axis_reference, marker_axis_start, marker_axis_end = _axis_projection_axis_from_payload(
        "axis_start=LKNE,LKNEM; axis_end=LANK,LANKM"
    )

    assert point_markers == ("LKNE", "LKNEM")
    assert axis_reference == "Axis_LKnee_SARA"
    assert axis_start == ()
    assert axis_end == ()
    assert marker_axis_reference == ""
    assert marker_axis_start == ("LKNE", "LKNEM")
    assert marker_axis_end == ("LANK", "LANKM")


def test_anatomical_axis_source_labels_include_all_virtual_markers_and_axes():
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)
    labels = _anatomical_axis_source_labels(draft, ("LASI", "RASI"))

    assert labels[:2] == ("LASI", "RASI")
    assert "CoR_LThigh_wrt_Pelvis | virtual marker | LThigh" in labels
    assert "CoR_LFoot_wrt_LShank | virtual marker | LFoot" in labels
    assert "[axis] Axis_LKnee_SARA | AoR virtual axis | LShank" in labels
    assert "[axis] LShank_second_axis | AoR virtual axis | LShank" not in labels
    assert _axis_source_name_from_list_text("[axis] Axis_LKnee_SARA | AoR virtual axis | LShank") == "Axis_LKnee_SARA"
    assert (
        _axis_source_name_from_list_text("CoR_LThigh_wrt_Pelvis | virtual marker | LThigh") == "CoR_LThigh_wrt_Pelvis"
    )


def test_c3d_file_names_from_folder_lists_available_c3d_files(tmp_path):
    """
    The virtual marker GUI should offer C3D files from the selected pipeline folder.
    """
    (tmp_path / "left_hip_functional.c3d").write_text("")
    (tmp_path / "notes.txt").write_text("")
    (tmp_path / "right_hip_functional.c3d").write_text("")

    assert _c3d_file_names_from_folder(str(tmp_path)) == (
        "left_hip_functional.c3d",
        "right_hip_functional.c3d",
    )


def test_c3d_file_matching_accepts_template_names_and_functional_patterns(tmp_path):
    (tmp_path / "Test_func_anat.c3d").write_text("")
    (tmp_path / "Test_func_lknee.c3d").write_text("")

    assert _matching_c3d_file_for_expected_name(str(tmp_path), "Test_func_anat.c3d").name == "Test_func_anat.c3d"
    assert _matching_c3d_file_for_expected_name(str(tmp_path), "*func_lknee.c3d").name == "Test_func_lknee.c3d"
    assert _matching_c3d_file_for_expected_name(str(tmp_path), "missing.c3d") is None


def test_virtual_feature_source_keeps_trial_metadata_with_c3d_assignment():
    source = "trial=left_knee_sara; parent markers=LTIBD,LTIB,LTIBF"
    assigned = _source_with_c3d_assignment(source, "/tmp/Test_func_lknee.c3d")

    assert _trial_name_from_virtual_feature_source(assigned) == "left_knee_sara"
    assert _c3d_source_name_from_virtual_feature_source(assigned) == "Test_func_lknee.c3d"
    assert "parent markers=LTIBD,LTIB,LTIBF" in assigned


def test_marker_name_mapping_matches_normalized_c3d_names():
    """
    Template marker names can be automatically matched to participant-specific C3D naming.
    """
    mapping = _marker_name_mapping_for_c3d(("LASI", "RASI", "LTHIB"), ("L_ASI", "rasi", "L-THIB", "extra"))

    assert mapping == {"LASI": "L_ASI", "RASI": "rasi", "LTHIB": "L-THIB"}


def test_marker_name_mapping_matches_participant_prefixed_c3d_names():
    """
    C3D participant namespaces should not prevent matching markers to a template.
    """
    mapping = _marker_name_mapping_for_c3d(("S3", "T6", "C2"), ("P01_MH:S3", "P01_MH:T6", "P01_MH:C2"))

    assert mapping == {"S3": "P01_MH:S3", "T6": "P01_MH:T6", "C2": "P01_MH:C2"}


def test_marker_name_mapping_matches_motive_skeleton_prefix():
    mapping = _marker_name_mapping_for_c3d(
        ("LIAS", "RFM5"),
        ("Skeleton_001_LIAS", "Skeleton_001_RFM5"),
    )

    assert mapping == {
        "LIAS": "Skeleton_001_LIAS",
        "RFM5": "Skeleton_001_RFM5",
    }


def test_strip_participant_prefix_from_marker_names():
    """
    Users can remove C3D participant prefixes such as P01_MH: from marker names.
    """
    assert _strip_participant_prefix_from_marker_names(("P01_MH:S3", "P01_MH:T6", "Skeleton_001_LIAS", "LASI")) == (
        "S3",
        "T6",
        "LIAS",
        "LASI",
    )


def test_strip_participant_prefix_from_c3d_data_changes_marker_names_in_place():
    """
    The GUI strips marker names on loaded C3D data while preserving marker order.
    """

    class FakeC3dData:
        marker_names = ["P01_MH:S3", "P01_MH:T6"]

    data = FakeC3dData()
    _strip_participant_prefix_from_c3d_data(data)

    assert data.marker_names == ["S3", "T6"]


def test_remap_c3d_workflow_draft_markers_updates_segment_groups():
    """
    Loading a C3D should update template marker references before the user edits segment assignments.
    """
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)
    mapping = {"LASI": "L_ASI", "RASI": "R_ASI"}
    updated = _remap_c3d_workflow_draft_markers(draft, mapping)
    pelvis = next(group for group in updated.segment_marker_groups if group.segment_name == "Pelvis")

    assert pelvis.marker_names[:2] == ("LPSI", "RPSI")
    assert "L_ASI" in pelvis.marker_names
    assert "R_ASI" in pelvis.marker_names


def test_marker_frame_position_reads_selected_frame():
    """
    The technical segment preview should display marker coordinates at the slider frame.
    """

    class FakeC3dData:
        marker_names = ["LASI"]
        nb_frames = 2

        def get_position(self, marker_names):
            values = np.ones((4, 1, 2))
            values[:3, 0, 0] = (1.0, 2.0, 3.0)
            values[:3, 0, 1] = (4.0, 5.0, 6.0)
            return values

    assert _marker_frame_position(FakeC3dData(), "LASI", 1) == (4.0, 5.0, 6.0)


def test_marker_frame_position_rejects_non_finite_coordinates():
    class FakeC3dData:
        marker_names = ["LASI"]
        nb_frames = 1

        def get_position(self, marker_names):
            values = np.ones((4, 1, 1))
            values[:3, 0, 0] = (1.0, np.inf, 3.0)
            return values

    assert _marker_frame_position(FakeC3dData(), "LASI", 0) is None


def test_rab2002_geometry_uses_static_markers_and_editable_fraction():
    class FakeC3dData:
        marker_names = ["RCAJ", "RHME", "RHLE"]
        nb_frames = 1

        def get_position(self, marker_names):
            positions = {
                "RCAJ": (0.0, 0.0, 0.0),
                "RHME": (100.0, -20.0, 0.0),
                "RHLE": (100.0, 20.0, 0.0),
            }
            values = np.ones((4, len(marker_names), 1))
            for index, marker_name in enumerate(marker_names):
                values[:3, index, 0] = positions[marker_name]
            return values

    assert _rab2002_markers_from_payload("point=RCAJ; mid=RHME,RHLE; fraction=0.25") == (
        "RCAJ",
        ("RHME", "RHLE"),
        0.25,
    )
    geometry = _rab2002_geometry(
        FakeC3dData(),
        "point=RCAJ; mid=RHME,RHLE; fraction=0.17",
        "RGJC",
        "RUpperArm",
        0,
    )

    assert geometry == ((0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (17.0, 0.0, 0.0))


def test_complete_marker_frame_count_requires_all_marker_coordinates():
    class FakeC3dData:
        marker_names = ["A", "B"]

        def get_position(self, marker_names):
            values = np.ones((4, len(marker_names), 3))
            values[:3, 1, 1] = np.nan
            return values

    assert _complete_marker_frame_count(FakeC3dData(), ("A", "B")) == 2
    assert _complete_marker_frame_count(FakeC3dData(), ("A", "Missing")) == 0


def test_functional_frame_selection_keeps_diverse_relative_rotations():
    rotations = np.zeros((3, 3, 4))
    translations = np.zeros((3, 4))
    parent = RotoTransMatrixTimeSeries.from_rotation_matrix_and_translation(
        np.repeat(np.eye(3)[:, :, None], 4, axis=2), translations
    )
    for index, angle in enumerate((0.0, 0.001, 0.002, 0.5)):
        rotations[:, :, index] = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    child = RotoTransMatrixTimeSeries.from_rotation_matrix_and_translation(rotations, translations)

    report = functional_frame_selection_report(
        parent,
        child,
        FunctionalFrameSelectionOptions(enabled=True, max_frames=10, min_rotation_degrees=2.0),
    )

    assert report.total_frames == 4
    assert report.valid_rt_frames == 4
    assert report.selected_indices == (0, 3)
    assert report.rotation_range_degrees > 25


def test_functional_frame_selection_keeps_manual_indices_without_diverse_mode():
    rotations = np.repeat(np.eye(3)[:, :, None], 5, axis=2)
    translations = np.zeros((3, 5))
    translations[0, :] = np.arange(5, dtype=float)
    parent = RotoTransMatrixTimeSeries.from_rotation_matrix_and_translation(rotations, np.zeros((3, 5)))
    child = RotoTransMatrixTimeSeries.from_rotation_matrix_and_translation(rotations, translations)

    parent_subset, child_subset, report = prepare_functional_rt_pair(
        parent,
        child,
        FunctionalFrameSelectionOptions(enabled=False, manual_frame_indices=(1, 3, 99)),
    )

    assert report.selected_indices == (1, 3)
    assert len(parent_subset) == 2
    assert len(child_subset) == 2
    np.testing.assert_allclose(child_subset.to_numpy()[0, 3, :], [1.0, 3.0])


def test_c3d_generation_log_reports_virtual_marker_local_offset_context():
    """
    Keep a trace that SCoRE/SARA markers are global markers with local offsets reserved for model construction.
    """
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)
    lines = _c3d_generation_log(draft, None, "/tmp/c3d", ("LASI", "RASI"))

    assert any("Preset: lower_limbs" in line for line in lines)
    assert any("Virtual markers:" in line for line in lines)
    assert any("global marker added to marker pool" in line for line in lines)


def test_motive_57_generation_log_uses_generic_static_name():
    draft = c3d_workflow_draft(C3dModelPreset.MOTIVE_57)
    lines = _c3d_generation_log(
        draft,
        None,
        "/tmp/c3d",
        ("LIAS", "RIAS", "LIPS", "RIPS"),
    )

    assert "- main: *Static.c3d" in lines
    assert all("- main: Static.c3d" != line for line in lines)
    assert all("P5_Calib_Static.c3d" not in line for line in lines)


def test_motive_57_generation_log_uses_resolved_static_assignment():
    draft = assign_c3d_file_role_to_draft(
        c3d_workflow_draft(C3dModelPreset.MOTIVE_57),
        "main",
        "/tmp/P5_Calib_Static.c3d",
    )
    lines = _c3d_generation_log(
        draft,
        None,
        "/tmp/c3d",
        ("LIAS", "RIAS", "LIPS", "RIPS"),
    )

    assert "- main: /tmp/P5_Calib_Static.c3d" in lines
    assert "- main: *Static.c3d" not in lines


def test_static_generic_name_matches_participant_prefixed_static_c3d(tmp_path):
    static_file = tmp_path / "P5_Calib_Static.c3d"
    static_file.write_text("", encoding="utf-8")

    assert _matching_c3d_file_for_expected_name(str(tmp_path), "*Static.c3d") == static_file
    assert _matching_c3d_file_for_expected_name(str(tmp_path), "Static.c3d") == static_file


def test_predictive_virtual_marker_method_label_maps_to_internal_key():
    """
    Keep readable predictive labels in the GUI while storing explicit method names in the draft.
    """
    assert _predictive_virtual_marker_method_from_label("Hara 2016 hip") == "hara2016_hip"
    assert _predictive_virtual_marker_method_from_label("harrington2007_hip") == "harrington2007_hip"
    assert _predictive_virtual_marker_method_from_label("Rab 2002 shoulder") == "rab2002_shoulder"


def test_motive_57_gjc_virtual_markers_use_rab2002_predictive_method():
    draft = c3d_workflow_draft(C3dModelPreset.MOTIVE_57)
    markers_by_name = {marker.name: marker for marker in draft.virtual_markers}

    assert markers_by_name["LGJC"].method == "rab2002_shoulder"
    assert "point=LCAJ" in markers_by_name["LGJC"].source
    assert "mid=LHME,LHLE" in markers_by_name["LGJC"].source
    assert markers_by_name["RGJC"].method == "rab2002_shoulder"
    assert "point=RCAJ" in markers_by_name["RGJC"].source
    assert "mid=RHME,RHLE" in markers_by_name["RGJC"].source


def test_python_code_from_c3d_draft_serializes_preset_value():
    """
    The generated script must contain editable JSON-like data, not Python enum reprs.
    """
    code = _python_code_from_c3d_draft(c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS), "/tmp/c3d")

    assert "C3dModelPreset" not in code
    assert '"preset": "lower_limbs"' in code
    assert "C3D_FOLDER = '/tmp/c3d'" in code


def test_export_model_to_path_dispatches_by_extension(tmp_path):
    """
    The model editor export button should use every writer supported by BioBuddy.
    """
    calls = []

    class FakeModel:
        def to_biomod(self, filepath):
            calls.append(("biomod", filepath))
            Path(filepath).write_text("biomod")

        def to_osim(self, filepath):
            calls.append(("osim", filepath))
            Path(filepath).write_text("osim")

        def to_urdf(self, filepath):
            calls.append(("urdf", filepath))
            Path(filepath).write_text("urdf")

        def to_bvh(self, filepath):
            calls.append(("bvh", filepath))
            Path(filepath).write_text("bvh")

    model = FakeModel()
    for extension in (".bioMod", ".osim", ".urdf", ".bvh"):
        _export_model_to_path(model, str(tmp_path / f"model{extension}"))

    assert [call[0] for call in calls] == ["biomod", "osim", "urdf", "bvh"]


def test_export_model_to_path_reports_missing_writer_output(tmp_path):
    class SilentModel:
        def to_biomod(self, filepath):
            return None

    with pytest.raises(RuntimeError, match="did not create"):
        _export_model_to_path(SilentModel(), str(tmp_path / "missing.bioMod"))
