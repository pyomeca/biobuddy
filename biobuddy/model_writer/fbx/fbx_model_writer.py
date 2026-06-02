from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from typing import TYPE_CHECKING

import numpy as np

from ..abstract_model_writer import AbstractModelWriter

if TYPE_CHECKING:
    from ...components.real.biomechanical_model_real import BiomechanicalModelReal


class FbxModelWriter(AbstractModelWriter):
    """
    Write a :class:`BiomechanicalModelReal` into FBX using Blender's exporter.

    BioBuddy deliberately delegates the actual FBX serialization to Blender
    instead of hand-writing the proprietary FBX binary format. Blender is an
    optional runtime dependency for this writer only; the rest of BioBuddy can
    still parse FBX files without it.
    """

    def __init__(self, filepath: str, with_mesh: bool = False, blender_executable: str = "blender"):
        """
        Parameters
        ----------
        filepath
            The path to the FBX file to write.
        with_mesh
            Whether meshes should be exported. Mesh export is not implemented
            yet in the Blender-backed FBX writer.
        blender_executable
            Path or command name used to launch Blender in background mode.
        """
        super().__init__(filepath=filepath, with_mesh=with_mesh)
        self.blender_executable = blender_executable

    @staticmethod
    def _matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
        """
        Convert a numpy matrix into a JSON-friendly nested list.
        """
        return [[float(value) for value in row] for row in matrix]

    def _segments_payload(self, model: "BiomechanicalModelReal") -> list[dict]:
        """
        Serialize the model hierarchy and segment frames for the Blender script.
        """
        segments = []
        for segment in model.segments:
            local_scs = model.segment_coordinate_system_in_local(segment.name)
            global_scs = model.segment_coordinate_system_in_global(segment.name)
            segments.append(
                {
                    "name": segment.name,
                    "parent_name": segment.parent_name,
                    "local_translation": [float(value) for value in local_scs.translation],
                    "local_rotation": self._matrix_to_list(local_scs.rotation_matrix.rotation_matrix),
                    "global_translation": [float(value) for value in global_scs.translation],
                    "global_rotation": self._matrix_to_list(global_scs.rotation_matrix.rotation_matrix),
                }
            )
        return segments

    def _payload(self, model: "BiomechanicalModelReal") -> dict:
        """
        Build the complete JSON payload consumed by Blender.
        """
        return {
            "armature_name": Path(self.filepath).stem,
            "segments": self._segments_payload(model),
        }

    @staticmethod
    def _blender_script() -> str:
        """
        Return the Python script executed by Blender to create and export an armature.
        """
        return r"""
import json
import sys

import bpy
from mathutils import Matrix, Vector


def _argv_after_double_dash():
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def _bone_length(segment, children):
    head = Vector(segment["global_translation"])
    distances = [
        (Vector(child["global_translation"]) - head).length
        for child in children
        if child["parent_name"] == segment["name"]
    ]
    distances = [distance for distance in distances if distance > 1e-8]
    return min(distances) if distances else 1.0


def _rotation_matrix(segment):
    return Matrix(segment["global_rotation"]).to_3x3()


payload_path, output_path = _argv_after_double_dash()
with open(payload_path, "r", encoding="utf-8") as payload_file:
    payload = json.load(payload_file)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

bpy.ops.object.armature_add(enter_editmode=True, location=(0.0, 0.0, 0.0))
armature_object = bpy.context.object
armature_object.name = payload["armature_name"]
armature_object.data.name = payload["armature_name"] + "_Armature"

edit_bones = armature_object.data.edit_bones
for bone in list(edit_bones):
    edit_bones.remove(bone)

created_bones = {}
segments = payload["segments"]
for segment in segments:
    head = Vector(segment["global_translation"])
    rotation = _rotation_matrix(segment)
    length = _bone_length(segment, segments)

    bone = edit_bones.new(segment["name"])
    bone.head = head
    bone.tail = head + rotation @ Vector((0.0, length, 0.0))
    bone.align_roll(rotation @ Vector((0.0, 0.0, 1.0)))

    parent_name = segment["parent_name"]
    if parent_name in created_bones:
        bone.parent = created_bones[parent_name]
        bone.use_connect = False
    created_bones[segment["name"]] = bone

bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.object.select_all(action="DESELECT")
armature_object.select_set(True)
bpy.context.view_layer.objects.active = armature_object

bpy.ops.export_scene.fbx(
    filepath=output_path,
    use_selection=True,
    object_types={"ARMATURE"},
    add_leaf_bones=False,
    bake_anim=False,
    armature_nodetype="ROOT",
)
"""

    def _validate_model(self, model: "BiomechanicalModelReal") -> None:
        """
        Ensure the model can be represented by the current writer.
        """
        if self.with_mesh:
            raise NotImplementedError("FBX mesh export is not implemented yet.")
        if model.muscle_groups:
            raise NotImplementedError("FBX export does not support muscles.")
        if model.ligaments:
            raise NotImplementedError("FBX export does not support ligaments.")

    def write(self, model: "BiomechanicalModelReal") -> None:
        """
        Write the model into an FBX file.

        Blender is called in background mode and receives a small JSON payload
        describing the BioBuddy segment hierarchy and segment coordinate systems.
        """
        self._validate_model(model)
        output_path = Path(self.filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            payload_path = tmp_path / "biobuddy_fbx_payload.json"
            script_path = tmp_path / "biobuddy_export_fbx.py"
            payload_path.write_text(json.dumps(self._payload(model)), encoding="utf-8")
            script_path.write_text(self._blender_script(), encoding="utf-8")

            command = [
                self.blender_executable,
                "--background",
                "--python",
                str(script_path),
                "--",
                str(payload_path),
                str(output_path),
            ]
            try:
                completed_process = subprocess.run(command, capture_output=True, text=True, check=False)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "FBX export requires Blender. Install Blender or pass blender_executable to to_fbx()."
                ) from exc

        if completed_process.returncode != 0:
            raise RuntimeError(
                "Blender failed to export the FBX file.\n"
                f"Command: {' '.join(command)}\n"
                f"stdout:\n{completed_process.stdout}\n"
                f"stderr:\n{completed_process.stderr}"
            )
