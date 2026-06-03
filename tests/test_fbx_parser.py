import json
from pathlib import Path

import numpy as np
import pytest

from biobuddy import BiomechanicalModelReal
from biobuddy.model_parser.bvh import BvhModelParser
from biobuddy.model_parser.fbx import FbxModelParser
from biobuddy.model_parser.fbx.fbx_model_parser import _FbxSkinCluster


def test_fbx_parser_extracts_the_fullbody_skeleton():
    """
    Parse the real FBX example and extract the expected skeleton roots.
    """
    parent_path = Path(__file__).resolve().parent.parent
    filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"

    parser = FbxModelParser(filepath=str(filepath))

    assert parser.version == 7400
    assert len(parser.root_ids) == 1
    root_node = parser.skeleton_nodes[parser.root_ids[0]]
    assert root_node.name == "Hips"
    assert root_node.node_type == "Root"
    assert "HeadEE" not in [node.name for node in parser.skeleton_nodes.values()]


def test_fbx_and_bvh_share_the_same_kinematic_topology():
    """
    Build models from FBX and BVH and compare their segment hierarchy.
    """
    parent_path = Path(__file__).resolve().parent.parent
    fbx_filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"
    bvh_filepath = parent_path / "examples" / "models" / "fullbody_model.bvh"

    model_from_fbx = BiomechanicalModelReal().from_fbx(filepath=str(fbx_filepath))
    model_from_bvh = BiomechanicalModelReal().from_bvh(filepath=str(bvh_filepath))

    fbx_topology = [(segment.name, segment.parent_name) for segment in model_from_fbx.segments]
    bvh_topology = [(segment.name, segment.parent_name) for segment in model_from_bvh.segments]

    assert fbx_topology == bvh_topology


def test_fbx_model_can_be_exported_to_biomod(tmp_path: Path):
    """
    Export a converted FBX hierarchy to a biorbd-compatible ``.bioMod`` file.
    """
    parent_path = Path(__file__).resolve().parent.parent
    fbx_filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"
    biomod_filepath = tmp_path / "fullbody_model_from_fbx.bioMod"

    model = BiomechanicalModelReal().from_fbx(filepath=str(fbx_filepath))
    model.to_biomod(filepath=str(biomod_filepath), with_mesh=False)

    content = biomod_filepath.read_text()
    assert "segment\tHips" in content
    assert "\ttranslations\txyz" in content
    assert "\trotations\tzyx" in content


def test_fbx_parser_maps_animation_to_biorbd_q():
    """
    Map the FBX animation curves to biorbd-compatible generalized coordinates.
    """
    parent_path = Path(__file__).resolve().parent.parent
    filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"

    animation = FbxModelParser(filepath=str(filepath)).to_q()

    assert animation.q.shape == (165, 1977)
    assert animation.time.shape == (1977,)
    assert animation.time[0] == pytest.approx(0.0)
    assert animation.time[1] == pytest.approx(1 / 26, abs=1e-6)
    assert animation.dof_names[:6] == [
        "Hips_transX",
        "Hips_transY",
        "Hips_transZ",
        "Hips_rotZ",
        "Hips_rotY",
        "Hips_rotX",
    ]
    np.testing.assert_allclose(
        animation.q[:6, 0],
        np.array(
            [
                1425.986572265625,
                557.1002197265625,
                1308.45849609375,
                np.deg2rad(-45.29030990600586),
                np.deg2rad(-50.993621826171875),
                np.deg2rad(64.83574676513672),
            ]
        ),
        atol=1e-8,
    )


def test_fbx_parser_reports_animation_diagnostics():
    """
    Report whether FBX animation curves, DoFs and generated meshes are complete.
    """
    parent_path = Path(__file__).resolve().parent.parent
    filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"

    diagnostics = FbxModelParser(filepath=str(filepath)).animation_diagnostics()

    assert diagnostics.frame_count == 1977
    assert diagnostics.dof_count == 165
    assert diagnostics.mapped_dof_count == 165
    assert diagnostics.missing_dof_names == []
    assert "LeftHandIndex2_rotX" in diagnostics.zero_dof_names
    assert "LeftHandIndex2_rotX" in diagnostics.constant_dof_names
    assert "Spine3" in diagnostics.segments_without_visual_meshes
    assert "LeftShoulder" in diagnostics.segments_without_visual_meshes
    assert "RightShoulder" in diagnostics.segments_without_visual_meshes
    assert {
        "name": "HeadEE",
        "node_type": "Root",
        "animated_properties": ["Lcl Rotation"],
    } in diagnostics.ignored_animated_model_nodes


def test_fbx_and_bvh_animation_reconstruct_the_same_joint_positions():
    """
    Compare joint positions reconstructed from coherent BVH and FBX animations.

    The comparison is expressed relative to ``Hips`` so the test validates the
    articulated pose, not the global root trajectory. This validates both the
    FBX rest pose convention and the conversion from FBX Euler animation curves
    to BioBuddy generalized coordinates.
    """
    parent_path = Path(__file__).resolve().parent.parent
    fbx_filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"
    bvh_filepath = parent_path / "examples" / "models" / "fullbody_model.bvh"

    model_from_fbx = BiomechanicalModelReal().from_fbx(filepath=str(fbx_filepath))
    model_from_bvh = BiomechanicalModelReal().from_bvh(filepath=str(bvh_filepath))
    animation_from_fbx = FbxModelParser(filepath=str(fbx_filepath)).to_q()
    animation_from_bvh = BvhModelParser(filepath=str(bvh_filepath)).to_q()

    frame_indices = np.asarray([0, 250, 500, 1000, 1500, 1976], dtype=int)
    fbx_kinematics = model_from_fbx.forward_kinematics(animation_from_fbx.q[:, frame_indices])
    bvh_kinematics = model_from_bvh.forward_kinematics(animation_from_bvh.q[:, frame_indices])

    shared_segments = [
        segment_name
        for segment_name in model_from_bvh.segment_names
        if segment_name in model_from_fbx.segment_names and segment_name != "root"
    ]
    fbx_hips = np.column_stack(
        [fbx_kinematics["Hips"][frame_index].rt_matrix[:3, 3] for frame_index in range(frame_indices.shape[0])]
    )
    bvh_hips = np.column_stack(
        [bvh_kinematics["Hips"][frame_index].rt_matrix[:3, 3] for frame_index in range(frame_indices.shape[0])]
    )

    errors = []
    for segment_name in shared_segments:
        fbx_positions = np.column_stack(
            [
                fbx_kinematics[segment_name][frame_index].rt_matrix[:3, 3]
                for frame_index in range(frame_indices.shape[0])
            ]
        )
        bvh_positions = np.column_stack(
            [
                bvh_kinematics[segment_name][frame_index].rt_matrix[:3, 3]
                for frame_index in range(frame_indices.shape[0])
            ]
        )
        errors.extend(
            np.linalg.norm(
                (fbx_positions - fbx_hips) - (bvh_positions - bvh_hips),
                axis=0,
            )
        )

    errors = np.asarray(errors)
    assert np.mean(errors) < 50.0
    assert np.percentile(errors, 95) < 150.0
    assert np.max(errors) < 250.0


def test_fbx_visual_mesh_can_be_split_per_segment(tmp_path: Path):
    """
    Split the skinned FBX mesh into one generated mesh file per segment.
    """
    parent_path = Path(__file__).resolve().parent.parent
    fbx_filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"
    mesh_output_dir = tmp_path / "segment_meshes"

    model = BiomechanicalModelReal().from_fbx(
        filepath=str(fbx_filepath),
        load_visual_meshes=True,
        mesh_output_dir=str(mesh_output_dir),
    )

    segments_with_mesh = [
        segment for segment in model.segments if segment.name != "root" and segment.mesh_file is not None
    ]
    assert len(segments_with_mesh) >= 20
    assert (mesh_output_dir / "hips.ply").exists()
    assert (mesh_output_dir / "leftarm.ply").exists()

    hips_mesh = model.segments["Hips"].mesh_file
    assert hips_mesh.mesh_file_name == "hips.ply"
    assert Path(hips_mesh.mesh_file_directory) == mesh_output_dir.resolve()

    hips_content = (mesh_output_dir / "hips.ply").read_text()
    assert "element vertex " in hips_content
    assert "element face " in hips_content
    assert "end_header" in hips_content


def test_fbx_visual_mesh_is_written_in_biomod(tmp_path: Path):
    """
    Export the generated segment meshes through the biorbd writer.
    """
    parent_path = Path(__file__).resolve().parent.parent
    fbx_filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"
    biomod_filepath = tmp_path / "fullbody_model_with_meshes.bioMod"
    mesh_output_dir = tmp_path / "segment_meshes"

    model = BiomechanicalModelReal().from_fbx(
        filepath=str(fbx_filepath),
        load_visual_meshes=True,
        mesh_output_dir=str(mesh_output_dir),
    )
    model.to_biomod(filepath=str(biomod_filepath), with_mesh=True)

    content = biomod_filepath.read_text()
    assert "\tmeshfile\t" in content
    assert "segment_meshes/hips.ply" in content.replace("\\", "/")


def test_fbx_package_export_creates_a_portable_biomod_bundle(tmp_path: Path):
    """
    Export an FBX conversion package with bioMod, meshes, animation and source copy.
    """
    parent_path = Path(__file__).resolve().parent.parent
    fbx_filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"

    package_directory = BiomechanicalModelReal.package_from_fbx(
        filepath=str(fbx_filepath),
        output_directory=str(tmp_path),
        package_name="fullbody_bundle",
        with_animation=True,
    )

    assert package_directory == (tmp_path / "fullbody_bundle").resolve()
    assert (package_directory / "fullbody_bundle.bioMod").exists()
    assert (package_directory / "meshes" / "hips.ply").exists()
    assert (package_directory / "source" / "fullbody_model.fbx").exists()
    assert (package_directory / "animations" / "fullbody_bundle_q.npz").exists()
    assert (package_directory / "animations" / "metadata.json").exists()

    animation_npz = np.load(
        package_directory / "animations" / "fullbody_bundle_q.npz",
        allow_pickle=True,
    )
    assert animation_npz["q"].shape == (165, 1977)
    assert animation_npz["time"].shape == (1977,)
    assert animation_npz["dof_names"][0] == "Hips_transX"

    biomod_content = (package_directory / "fullbody_bundle.bioMod").read_text()
    assert "meshes/hips.ply" in biomod_content.replace("\\", "/")

    metadata = json.loads((package_directory / "animations" / "metadata.json").read_text())
    assert metadata["mapped_dof_count"] == 165
    assert metadata["missing_dof_names"] == []
    assert "Spine3" in metadata["segments_without_visual_meshes"]


def test_fbx_shared_faces_are_kept_on_boundary_segments():
    """
    Duplicate shared boundary faces across the segments involved in the skinning.
    """
    parent_path = Path(__file__).resolve().parent.parent
    fbx_filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"
    parser = FbxModelParser(filepath=str(fbx_filepath))

    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=int)
    clusters = [
        _FbxSkinCluster(
            segment_name="Pelvis",
            control_point_indices=np.asarray([0, 1, 3, 4, 5], dtype=int),
            weights=np.asarray([1.0, 0.7, 1.0, 1.0, 1.0]),
        ),
        _FbxSkinCluster(
            segment_name="Trunk",
            control_point_indices=np.asarray([1, 2], dtype=int),
            weights=np.asarray([0.3, 1.0]),
        ),
    ]

    segment_faces = parser._segment_faces_from_skin(faces=faces, clusters=clusters)

    np.testing.assert_array_equal(segment_faces["Pelvis"], np.asarray([[0, 1, 2], [3, 4, 5]]))
    np.testing.assert_array_equal(segment_faces["Trunk"], np.asarray([[0, 1, 2]]))
