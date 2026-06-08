from dataclasses import dataclass, field
from pathlib import Path
import struct
import zlib

import numpy as np
from scipy.spatial.transform import Rotation

from ...components.real.biomechanical_model_real import BiomechanicalModelReal
from ...components.real.rigidbody.mesh_file_real import MeshFileReal
from ...components.real.rigidbody.segment_coordinate_system_real import (
    SegmentCoordinateSystemReal,
)
from ...components.real.rigidbody.segment_real import SegmentReal
from ...utils.enums import Rotations, Translations
from ..abstract_model_parser import AbstractModelParser
from ..parsed_animation import ParsedAnimation


@dataclass
class _FbxNodeRecord:
    """
    Minimal representation of a binary FBX node record.

    Parameters
    ----------
    name
        The FBX record name.
    properties
        The decoded properties attached to the record.
    children
        The direct child records.
    """

    name: str
    properties: list = field(default_factory=list)
    children: list["_FbxNodeRecord"] = field(default_factory=list)


@dataclass
class _FbxSkeletonNode:
    """
    Internal representation of one FBX skeleton node.

    Parameters
    ----------
    node_id
        The FBX object identifier.
    name
        The cleaned node name.
    node_type
        The FBX node type, eg. ``Root`` or ``LimbNode``.
    translation
        The local translation with respect to the parent node.
    rotation
        The local rest rotation used to orient the segment coordinate system,
        expressed in FBX extrinsic XYZ Euler angles in degrees.
    children_ids
        The direct child skeleton node identifiers.
    """

    node_id: int
    name: str
    node_type: str
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    children_ids: list[int] = field(default_factory=list)


@dataclass
class _FbxSkinCluster:
    """
    Internal representation of one FBX skin cluster.

    Parameters
    ----------
    segment_name
        The skeleton segment driven by this cluster.
    control_point_indices
        The control-point indices influenced by the cluster.
    weights
        The influence weights associated with ``control_point_indices``.
    """

    segment_name: str
    control_point_indices: np.ndarray
    weights: np.ndarray


@dataclass
class FbxAnimationDiagnostics:
    """
    Summary of the FBX animation data mapped by :class:`FbxModelParser`.

    Parameters
    ----------
    frame_count
        The number of animation samples extracted from the FBX curves.
    duration_seconds
        The extracted animation duration in seconds.
    dof_count
        The number of DoFs expected by the parsed biomechanical model.
    mapped_dof_count
        The number of DoFs backed by an FBX animation curve.
    missing_dof_names
        The DoFs without a matching FBX animation curve.
    zero_dof_names
        The DoFs whose extracted values are zero for all frames.
    constant_dof_names
        The DoFs whose extracted values are constant for all frames.
    ignored_animated_model_nodes
        FBX model nodes carrying animation curve nodes but not included in the
        parsed biomechanical skeleton.
    segments_without_visual_meshes
        Parsed skeleton segments that do not receive a generated visual mesh.
    """

    frame_count: int
    duration_seconds: float
    dof_count: int
    mapped_dof_count: int
    missing_dof_names: list[str]
    zero_dof_names: list[str]
    constant_dof_names: list[str]
    ignored_animated_model_nodes: list[dict[str, object]]
    segments_without_visual_meshes: list[str]


class FbxModelParser(AbstractModelParser):
    """
    Parse a binary FBX skeleton into a :class:`BiomechanicalModelReal`.

    The current implementation focuses on the scene-graph information needed to
    rebuild a kinematic chain:

    - ``Objects/Model`` nodes of type ``Root`` and ``LimbNode``
    - local translations from ``Lcl Translation``
    - rest orientations from ``PreRotation`` and ``Lcl Rotation``
    - hierarchy from ``Connections/C: OO``

    This keeps the implementation aligned with the existing BVH parser while
    staying dependency-free.
    """

    _FBX_TIME_UNIT = 46186158000.0

    def __init__(
        self,
        filepath: str,
        split_meshes_per_segment: bool = False,
        mesh_output_dir: str = None,
    ):
        """
        Load the FBX skeleton hierarchy from a file.

        Parameters
        ----------
        filepath
            The path to the FBX file to parse.
        split_meshes_per_segment
            Whether the skinned FBX mesh should be split into per-segment mesh files.
        mesh_output_dir
            The directory where the generated per-segment mesh files should be written.
            If ``None`` and ``split_meshes_per_segment`` is ``True``, a ``<fbx_stem>_meshes``
            directory is created next to the FBX file.
        """
        super().__init__(filepath)
        self.split_meshes_per_segment = split_meshes_per_segment
        self.mesh_output_dir = mesh_output_dir
        self.version: int | None = None
        self._top_level_records: list[_FbxNodeRecord] = []
        self.skeleton_nodes: dict[int, _FbxSkeletonNode] = {}
        self.root_ids: list[int] = []
        self._read()

    def _read(self) -> None:
        """
        Read the FBX binary structure and extract the skeleton scene graph.
        """
        with open(self.filepath, "rb") as file:
            data = file.read()

        if not data.startswith(b"Kaydara FBX Binary"):
            raise ValueError("Only binary FBX files are supported.")

        self.version = struct.unpack_from("<I", data, 23)[0]
        offset = 27
        self._top_level_records = []
        while offset < len(data):
            record, offset = self._parse_record(data=data, start_offset=offset)
            if record is None:
                break
            self._top_level_records.append(record)

        self._extract_skeleton()

    def _parse_record(self, data: bytes, start_offset: int) -> tuple[_FbxNodeRecord | None, int]:
        """
        Parse one binary FBX record recursively.

        Parameters
        ----------
        data
            The raw FBX file bytes.
        start_offset
            The offset where the record starts.

        Returns
        -------
        tuple[_FbxNodeRecord | None, int]
            The parsed record and the next unread offset.
        """
        if self.version is None:
            raise RuntimeError("The FBX version must be known before parsing records.")

        if self.version >= 7500:
            end_offset, prop_count, prop_list_length = struct.unpack_from("<QQQ", data, start_offset)
            cursor = start_offset + 24
            null_record_size = 25
        else:
            end_offset, prop_count, prop_list_length = struct.unpack_from("<III", data, start_offset)
            cursor = start_offset + 12
            null_record_size = 13

        name_length = data[cursor]
        cursor += 1
        if end_offset == 0:
            return None, start_offset + null_record_size

        name = data[cursor : cursor + name_length].decode("utf-8", errors="ignore")
        cursor += name_length

        properties = []
        for _ in range(prop_count):
            property_value, cursor = self._parse_property(data=data, start_offset=cursor)
            properties.append(property_value)

        children = []
        while cursor < end_offset - null_record_size:
            child_record, cursor = self._parse_record(data=data, start_offset=cursor)
            if child_record is not None:
                children.append(child_record)

        return (
            _FbxNodeRecord(name=name, properties=properties, children=children),
            end_offset,
        )

    @staticmethod
    def _parse_property(data: bytes, start_offset: int) -> tuple[object, int]:
        """
        Parse one FBX property.

        Parameters
        ----------
        data
            The raw FBX file bytes.
        start_offset
            The offset where the property starts.
        """
        property_type = chr(data[start_offset])
        cursor = start_offset + 1

        if property_type == "S":
            length = struct.unpack_from("<I", data, cursor)[0]
            cursor += 4
            value = data[cursor : cursor + length].decode("utf-8", errors="ignore")
            cursor += length
            return value, cursor
        if property_type == "L":
            return struct.unpack_from("<q", data, cursor)[0], cursor + 8
        if property_type == "I":
            return struct.unpack_from("<i", data, cursor)[0], cursor + 4
        if property_type == "D":
            return struct.unpack_from("<d", data, cursor)[0], cursor + 8
        if property_type == "F":
            return struct.unpack_from("<f", data, cursor)[0], cursor + 4
        if property_type == "C":
            return bool(data[cursor]), cursor + 1
        if property_type == "Y":
            return struct.unpack_from("<h", data, cursor)[0], cursor + 2
        if property_type == "R":
            length = struct.unpack_from("<I", data, cursor)[0]
            cursor += 4
            return {"raw_length": length}, cursor + length
        if property_type in "fdibc":
            array_length, encoding, compressed_length = struct.unpack_from("<III", data, cursor)
            cursor += 12
            array_bytes = data[cursor : cursor + compressed_length]
            cursor += compressed_length
            if encoding == 1:
                array_bytes = zlib.decompress(array_bytes)

            format_characters = {
                "f": "f",
                "d": "d",
                "i": "i",
                "l": "q",
                "b": "?",
                "c": "b",
            }
            format_character = format_characters[property_type]
            item_size = struct.calcsize(f"<{format_character}")
            expected_size = array_length * item_size
            values = struct.unpack(
                f"<{array_length}{format_character}",
                array_bytes[:expected_size],
            )
            return {
                "array_length": array_length,
                "encoding": encoding,
                "values": values,
            }, cursor

        if property_type == "l":
            array_length, encoding, compressed_length = struct.unpack_from("<III", data, cursor)
            cursor += 12
            array_bytes = data[cursor : cursor + compressed_length]
            cursor += compressed_length
            if encoding == 1:
                array_bytes = zlib.decompress(array_bytes)
            values = struct.unpack(f"<{array_length}q", array_bytes[: array_length * 8])
            return {
                "array_length": array_length,
                "encoding": encoding,
                "values": values,
            }, cursor

        raise NotImplementedError(f"Unsupported FBX property type '{property_type}'.")

    @staticmethod
    def _clean_name(raw_name: str) -> str:
        """
        Normalize an FBX object name.

        Parameters
        ----------
        raw_name
            The raw FBX object name, potentially including namespaces and
            trailing metadata bytes decoded from the binary stream.
        """
        return str(raw_name).split("\x00", maxsplit=1)[0].split(":", maxsplit=1)[-1]

    @staticmethod
    def _properties70_dict(model_record: _FbxNodeRecord) -> dict[str, list[float]]:
        """
        Convert an FBX ``Properties70`` block into a dictionary.

        Parameters
        ----------
        model_record
            The ``Model`` record containing the property block.
        """
        properties = {}
        properties_record = next(
            (child for child in model_record.children if child.name == "Properties70"),
            None,
        )
        if properties_record is None:
            return properties

        for child in properties_record.children:
            if child.name != "P" or len(child.properties) < 5:
                continue
            properties[child.properties[0]] = child.properties[4:]

        return properties

    @staticmethod
    def _vector3(properties: dict[str, list[float]], property_name: str) -> np.ndarray:
        """
        Extract a three-component vector from an FBX property dictionary.

        Parameters
        ----------
        properties
            The parsed ``Properties70`` dictionary.
        property_name
            The name of the property to retrieve.
        """
        values = properties.get(property_name, [0.0, 0.0, 0.0])
        return np.array([float(values[0]), float(values[1]), float(values[2])], dtype=float)

    @staticmethod
    def _fbx_euler_xyz_rt_matrix(angles: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """
        Convert an FBX XYZ Euler transform into a homogeneous matrix.

        FBX stores the skeleton rest pose in ``PreRotation``/``Lcl Rotation``
        properties using extrinsic XYZ Euler angles in degrees. BioBuddy's
        generic Euler helper uses the internal generalized-coordinate
        convention instead, so the FBX rest transform must be materialized as a
        matrix before it is attached to the segment coordinate system.

        Parameters
        ----------
        angles
            FBX Euler angles in degrees.
        translation
            Local FBX translation.

        Returns
        -------
        numpy.ndarray
            The 4x4 homogeneous transform represented by the FBX properties.
        """
        rt_matrix = np.eye(4)
        rt_matrix[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        rt_matrix[:3, 3] = translation[:3]
        return rt_matrix

    def _extract_skeleton(self) -> None:
        """
        Build the skeleton hierarchy from ``Objects`` and ``Connections``.
        """
        objects_record = next(
            (record for record in self._top_level_records if record.name == "Objects"),
            None,
        )
        connections_record = next(
            (record for record in self._top_level_records if record.name == "Connections"),
            None,
        )
        if objects_record is None or connections_record is None:
            raise ValueError("The FBX file must contain both Objects and Connections sections.")

        raw_models = [child for child in objects_record.children if child.name == "Model"]
        skeleton_types = {"Root", "LimbNode"}
        scene_models = {}
        for record in raw_models:
            if len(record.properties) < 3:
                continue
            node_id = int(record.properties[0])
            node_name = self._clean_name(record.properties[1])
            node_type = str(record.properties[2])
            properties = self._properties70_dict(record)
            scene_models[node_id] = _FbxSkeletonNode(
                node_id=node_id,
                name=node_name,
                node_type=node_type,
                translation=self._vector3(properties, "Lcl Translation"),
                rotation=self._vector3(properties, "PreRotation") + self._vector3(properties, "Lcl Rotation"),
            )

        parent_lookup = {}
        children_lookup = {}
        for connection in connections_record.children:
            if connection.name != "C" or len(connection.properties) < 3 or connection.properties[0] != "OO":
                continue
            child_id = int(connection.properties[1])
            parent_id = int(connection.properties[2])
            if child_id not in scene_models:
                continue
            parent_lookup[child_id] = parent_id
            children_lookup.setdefault(parent_id, []).append(child_id)

        def first_skeleton_descendants(node_id: int) -> list[int]:
            if node_id in scene_models and scene_models[node_id].node_type in skeleton_types:
                return [node_id]

            descendants = []
            for child_id in children_lookup.get(node_id, []):
                descendants.extend(first_skeleton_descendants(child_id))
            return descendants

        root_candidates = []
        for scene_root_id in children_lookup.get(0, []):
            root_candidates.extend(first_skeleton_descendants(scene_root_id))

        self.root_ids = list(dict.fromkeys(root_candidates))
        if not self.root_ids:
            raise ValueError("No FBX skeleton root could be identified in the file.")

        included_node_ids = set()

        def collect_skeleton(node_id: int) -> None:
            if node_id in included_node_ids:
                return
            if node_id not in scene_models or scene_models[node_id].node_type not in skeleton_types:
                return
            included_node_ids.add(node_id)
            for child_id in children_lookup.get(node_id, []):
                collect_skeleton(child_id)

        for root_id in self.root_ids:
            collect_skeleton(root_id)

        self.skeleton_nodes = {}
        for node_id in included_node_ids:
            node = scene_models[node_id]
            node.children_ids = [
                child_id
                for child_id in children_lookup.get(node_id, [])
                if child_id in included_node_ids
                and not (scene_models[child_id].node_type == "Root" and len(children_lookup.get(child_id, [])) == 0)
            ]
            if node.node_type == "Root" and node_id not in self.root_ids and len(node.children_ids) == 0:
                continue
            self.skeleton_nodes[node_id] = node

    @staticmethod
    def _array_values(record: _FbxNodeRecord) -> tuple:
        """
        Extract the decoded values from an FBX array record.

        Parameters
        ----------
        record
            The record storing one array-valued property.
        """
        if (
            len(record.properties) != 1
            or not isinstance(record.properties[0], dict)
            or "values" not in record.properties[0]
        ):
            raise ValueError(f"Record '{record.name}' does not contain a decoded FBX array.")
        return record.properties[0]["values"]

    @staticmethod
    def _polygon_indices_to_faces(polygon_indices: tuple[int, ...]) -> np.ndarray:
        """
        Convert FBX polygon vertex indices to triangle faces.

        Parameters
        ----------
        polygon_indices
            The FBX ``PolygonVertexIndex`` sequence.
        """
        faces = []
        current_face = []
        for index in polygon_indices:
            if index < 0:
                current_face.append(-index - 1)
                if len(current_face) < 3:
                    raise ValueError("FBX faces must contain at least three vertices.")
                for face_index in range(1, len(current_face) - 1):
                    faces.append(
                        [
                            current_face[0],
                            current_face[face_index],
                            current_face[face_index + 1],
                        ]
                    )
                current_face = []
            else:
                current_face.append(index)

        if current_face:
            raise ValueError("The FBX PolygonVertexIndex array ended before closing the last polygon.")

        return np.asarray(faces, dtype=int)

    def _visual_output_directory(self) -> Path:
        """
        Resolve the output directory used for generated segment meshes.
        """
        if self.mesh_output_dir is not None:
            return Path(self.mesh_output_dir).resolve()
        filepath = Path(self.filepath)
        return (filepath.parent / f"{filepath.stem}_meshes").resolve()

    def _visual_geometry_record(self) -> _FbxNodeRecord | None:
        """
        Retrieve the first mesh geometry record from the FBX objects block.
        """
        objects_record = next(
            (record for record in self._top_level_records if record.name == "Objects"),
            None,
        )
        if objects_record is None:
            return None

        for child in objects_record.children:
            if child.name == "Geometry" and len(child.properties) >= 3 and child.properties[2] == "Mesh":
                return child
        return None

    def _cluster_records(self) -> tuple[dict[int, _FbxNodeRecord], dict[int, str]]:
        """
        Extract cluster records and their associated segment names.
        """
        objects_record = next(
            (record for record in self._top_level_records if record.name == "Objects"),
            None,
        )
        connections_record = next(
            (record for record in self._top_level_records if record.name == "Connections"),
            None,
        )
        if objects_record is None or connections_record is None:
            return {}, {}

        object_lookup = {}
        cluster_records = {}
        for child in objects_record.children:
            if len(child.properties) >= 3 and isinstance(child.properties[0], int):
                object_lookup[int(child.properties[0])] = child
                if child.name == "Deformer" and child.properties[2] == "Cluster":
                    cluster_records[int(child.properties[0])] = child

        cluster_to_segment = {}
        for connection in connections_record.children:
            if connection.name != "C" or len(connection.properties) < 3 or connection.properties[0] != "OO":
                continue
            child_id = int(connection.properties[1])
            parent_id = int(connection.properties[2])
            if parent_id not in cluster_records or child_id not in object_lookup:
                continue
            child_record = object_lookup[child_id]
            if len(child_record.properties) < 3 or child_record.properties[2] not in {
                "Root",
                "LimbNode",
            }:
                continue
            cluster_to_segment[parent_id] = self._clean_name(child_record.properties[1])

        return cluster_records, cluster_to_segment

    def _object_records_by_id(self) -> dict[int, _FbxNodeRecord]:
        """
        Return the FBX object records indexed by their object identifier.
        """
        objects_record = next(
            (record for record in self._top_level_records if record.name == "Objects"),
            None,
        )
        if objects_record is None:
            return {}

        records_by_id = {}
        for child in objects_record.children:
            if len(child.properties) >= 1 and isinstance(child.properties[0], int):
                records_by_id[int(child.properties[0])] = child
        return records_by_id

    def _skin_clusters(self) -> list[_FbxSkinCluster]:
        """
        Build the list of FBX skin clusters attached to skeleton segments.
        """
        cluster_records, cluster_to_segment = self._cluster_records()
        skin_clusters = []
        for cluster_id, cluster_record in cluster_records.items():
            segment_name = cluster_to_segment.get(cluster_id)
            if segment_name is None or segment_name not in self.segment_names():
                continue
            indices_record = next(
                (child for child in cluster_record.children if child.name == "Indexes"),
                None,
            )
            weights_record = next(
                (child for child in cluster_record.children if child.name == "Weights"),
                None,
            )
            if indices_record is None or weights_record is None:
                continue
            control_point_indices = np.asarray(self._array_values(indices_record), dtype=int)
            weights = np.asarray(self._array_values(weights_record), dtype=float)
            skin_clusters.append(
                _FbxSkinCluster(
                    segment_name=segment_name,
                    control_point_indices=control_point_indices,
                    weights=weights,
                )
            )
        return skin_clusters

    def _ordered_skeleton_nodes(self) -> list[_FbxSkeletonNode]:
        """
        Return the parsed skeleton nodes in the same traversal order as the model.
        """
        ordered_nodes = []

        def append_descendants(node_id: int) -> None:
            node = self.skeleton_nodes[node_id]
            ordered_nodes.append(node)
            for child_id in node.children_ids:
                append_descendants(child_id)

        for root_id in self.root_ids:
            append_descendants(root_id)
        return ordered_nodes

    def _q_dof_names(self) -> list[str]:
        """
        Return the biorbd-compatible DoF names implied by the FBX skeleton.
        """
        dof_names = []
        root_ids = set(self.root_ids)
        for node in self._ordered_skeleton_nodes():
            if node.node_id in root_ids:
                dof_names.extend(
                    [
                        f"{node.name}_transX",
                        f"{node.name}_transY",
                        f"{node.name}_transZ",
                    ]
                )
            dof_names.extend([f"{node.name}_rot{axis.upper()}" for axis in Rotations.ZYX.value])
        return dof_names

    @staticmethod
    def _parse_dof_name(dof_name: str) -> tuple[str, str, str]:
        """
        Split a DoF name into segment name, quantity kind and axis.
        """
        segment_name, quantity = dof_name.rsplit("_", maxsplit=1)
        if quantity.startswith("trans"):
            return segment_name, "Lcl Translation", quantity[-1].lower()
        if quantity.startswith("rot"):
            return segment_name, "Lcl Rotation", quantity[-1].lower()
        raise ValueError(f"Unsupported DoF name '{dof_name}'.")

    def _curve_key_values(self, curve_record: _FbxNodeRecord) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract the key times and values stored in one FBX animation curve.
        """
        key_time_record = next(
            (child for child in curve_record.children if child.name == "KeyTime"),
            None,
        )
        key_value_record = next(
            (child for child in curve_record.children if child.name == "KeyValueFloat"),
            None,
        )
        if key_time_record is None or key_value_record is None:
            raise ValueError("An FBX animation curve must contain KeyTime and KeyValueFloat arrays.")

        key_times = np.asarray(self._array_values(key_time_record), dtype=np.int64)
        key_values = np.asarray(self._array_values(key_value_record), dtype=float)
        if key_times.shape[0] != key_values.shape[0]:
            raise ValueError("FBX animation curve times and values must have the same length.")
        return key_times, key_values

    def _animation_curves(
        self,
    ) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
        """
        Extract the animation curves attached to the parsed skeleton.

        Returns
        -------
        dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]
            A mapping from ``(segment_name, property_name, axis)`` to the raw
            FBX key times and values.
        """
        object_records_by_id = self._object_records_by_id()
        connections_record = next(
            (record for record in self._top_level_records if record.name == "Connections"),
            None,
        )
        if connections_record is None:
            return {}

        node_id_to_name = {node.node_id: node.name for node in self.skeleton_nodes.values()}

        curve_node_targets = {}
        for connection in connections_record.children:
            if connection.name != "C" or len(connection.properties) < 4 or connection.properties[0] != "OP":
                continue
            child_id = int(connection.properties[1])
            parent_id = int(connection.properties[2])
            property_name = str(connection.properties[3])
            child_record = object_records_by_id.get(child_id)
            if (
                child_record is None
                or child_record.name != "AnimationCurveNode"
                or parent_id not in node_id_to_name
                or property_name not in {"Lcl Translation", "Lcl Rotation"}
            ):
                continue
            curve_node_targets[child_id] = (node_id_to_name[parent_id], property_name)

        animation_curves = {}
        for connection in connections_record.children:
            if connection.name != "C" or len(connection.properties) < 4 or connection.properties[0] != "OP":
                continue
            curve_id = int(connection.properties[1])
            curve_node_id = int(connection.properties[2])
            axis_property = str(connection.properties[3])
            curve_record = object_records_by_id.get(curve_id)
            if (
                curve_record is None
                or curve_record.name != "AnimationCurve"
                or curve_node_id not in curve_node_targets
                or not axis_property.startswith("d|")
            ):
                continue

            segment_name, property_name = curve_node_targets[curve_node_id]
            axis = axis_property[-1].lower()
            animation_curves[(segment_name, property_name, axis)] = self._curve_key_values(curve_record)

        return animation_curves

    def _ignored_animated_model_nodes(self) -> list[dict[str, object]]:
        """
        Return animated FBX model nodes ignored by the skeleton parser.
        """
        object_records_by_id = self._object_records_by_id()
        connections_record = next(
            (record for record in self._top_level_records if record.name == "Connections"),
            None,
        )
        if connections_record is None:
            return []

        parsed_segment_names = self.segment_names()
        ignored_nodes = {}
        for connection in connections_record.children:
            if connection.name != "C" or len(connection.properties) < 4 or connection.properties[0] != "OP":
                continue
            child_record = object_records_by_id.get(int(connection.properties[1]))
            parent_record = object_records_by_id.get(int(connection.properties[2]))
            property_name = str(connection.properties[3])
            if (
                child_record is None
                or parent_record is None
                or child_record.name != "AnimationCurveNode"
                or parent_record.name != "Model"
                or property_name not in {"Lcl Translation", "Lcl Rotation"}
            ):
                continue

            model_name = self._clean_name(parent_record.properties[1])
            if model_name in parsed_segment_names:
                continue

            model_type = str(parent_record.properties[2])
            ignored_node = ignored_nodes.setdefault(
                model_name,
                {
                    "name": model_name,
                    "node_type": model_type,
                    "animated_properties": set(),
                },
            )
            ignored_node["animated_properties"].add(property_name)

        return [
            {
                "name": node["name"],
                "node_type": node["node_type"],
                "animated_properties": sorted(node["animated_properties"]),
            }
            for node in sorted(ignored_nodes.values(), key=lambda value: value["name"])
        ]

    def segment_names(self) -> set[str]:
        """
        Return the parsed skeleton segment names.
        """
        return {node.name for node in self.skeleton_nodes.values()}

    def _segment_names_with_visual_meshes(self) -> set[str]:
        """
        Return the parsed segment names that would receive generated visual meshes.
        """
        geometry_record = self._visual_geometry_record()
        if geometry_record is None:
            return set()

        polygon_record = next(
            (child for child in geometry_record.children if child.name == "PolygonVertexIndex"),
            None,
        )
        if polygon_record is None:
            return set()

        faces = self._polygon_indices_to_faces(self._array_values(polygon_record))
        skin_clusters = self._skin_clusters()
        if not skin_clusters:
            return set()

        segment_faces = self._segment_faces_from_skin(faces=faces, clusters=skin_clusters)
        return set(segment_faces.keys())

    def _segment_faces_from_skin(self, faces: np.ndarray, clusters: list[_FbxSkinCluster]) -> dict[str, np.ndarray]:
        """
        Assign each face of the skinned mesh to one or more segments.

        Parameters
        ----------
        faces
            The triangle faces built from ``PolygonVertexIndex``.
        clusters
            The per-segment skin clusters.

        Notes
        -----
        Faces fully owned by one segment are assigned only to that segment.
        Faces spanning several segments are duplicated across the involved
        segments so the exported per-segment meshes remain visually closed
        around inter-segment boundaries.
        """
        weights_by_control_point = {}
        for cluster in clusters:
            for control_point_index, weight in zip(cluster.control_point_indices, cluster.weights):
                if control_point_index not in weights_by_control_point:
                    weights_by_control_point[control_point_index] = {}
                previous_weight = weights_by_control_point[control_point_index].get(cluster.segment_name, 0.0)
                weights_by_control_point[control_point_index][cluster.segment_name] = previous_weight + float(weight)

        segment_faces = {cluster.segment_name: [] for cluster in clusters}
        for face in faces:
            segment_scores = {}
            for control_point_index in face:
                for segment_name, weight in weights_by_control_point.get(int(control_point_index), {}).items():
                    segment_scores[segment_name] = segment_scores.get(segment_name, 0.0) + weight
            if not segment_scores:
                continue

            if len(segment_scores) == 1:
                owner_segments = [next(iter(segment_scores.keys()))]
            else:
                owner_segments = sorted(segment_scores.keys())

            for owner_segment in owner_segments:
                segment_faces.setdefault(owner_segment, []).append(face.tolist())

        return {
            segment_name: np.asarray(face_values, dtype=int)
            for segment_name, face_values in segment_faces.items()
            if face_values
        }

    @staticmethod
    def _write_ascii_ply(filepath: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
        """
        Write one triangle mesh as an ASCII PLY file.

        Parameters
        ----------
        filepath
            The path where the PLY file should be written.
        vertices
            The mesh vertices with shape ``(n_vertices, 3)``.
        faces
            The triangle faces with shape ``(n_faces, 3)``.
        """
        with open(filepath, "w", encoding="utf-8") as file:
            file.write("ply\n")
            file.write("format ascii 1.0\n")
            file.write(f"element vertex {vertices.shape[0]}\n")
            file.write("property float x\n")
            file.write("property float y\n")
            file.write("property float z\n")
            file.write(f"element face {faces.shape[0]}\n")
            file.write("property list uchar int vertex_indices\n")
            file.write("end_header\n")
            for vertex in vertices:
                file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
            for face in faces:
                file.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    def _attach_visual_meshes(self, model: BiomechanicalModelReal) -> None:
        """
        Extract the FBX skinned mesh, split it by segment and attach per-segment
        mesh files to the biomechanical model.

        Parameters
        ----------
        model
            The biomechanical model being built from the FBX skeleton.
        """
        geometry_record = self._visual_geometry_record()
        if geometry_record is None:
            return

        vertices_record = next(
            (child for child in geometry_record.children if child.name == "Vertices"),
            None,
        )
        polygon_record = next(
            (child for child in geometry_record.children if child.name == "PolygonVertexIndex"),
            None,
        )
        if vertices_record is None or polygon_record is None:
            return

        control_points = np.asarray(self._array_values(vertices_record), dtype=float).reshape((-1, 3))
        faces = self._polygon_indices_to_faces(self._array_values(polygon_record))
        skin_clusters = self._skin_clusters()
        if not skin_clusters:
            return

        segment_faces = self._segment_faces_from_skin(faces=faces, clusters=skin_clusters)
        if not segment_faces:
            return

        output_directory = self._visual_output_directory()
        output_directory.mkdir(parents=True, exist_ok=True)

        for segment_name, segment_mesh_faces in segment_faces.items():
            used_indices, inverse_indices = np.unique(segment_mesh_faces.reshape(-1), return_inverse=True)
            segment_faces_local = inverse_indices.reshape((-1, 3))
            segment_vertices_global = control_points[used_indices]

            segment_rt_global = model.segment_coordinate_system_in_global(segment_name=segment_name)
            homogeneous_vertices = np.hstack(
                (
                    segment_vertices_global,
                    np.ones((segment_vertices_global.shape[0], 1)),
                )
            )
            segment_vertices_local = (segment_rt_global.inverse.rt_matrix @ homogeneous_vertices.T).T[:, :3]

            mesh_filename = f"{segment_name.lower()}.ply"
            mesh_filepath = output_directory / mesh_filename
            self._write_ascii_ply(
                filepath=mesh_filepath,
                vertices=segment_vertices_local,
                faces=segment_faces_local,
            )
            model.segments[segment_name].mesh_file = MeshFileReal(
                mesh_file_name=mesh_filename,
                mesh_file_directory=str(output_directory),
            )

    @staticmethod
    def _translations_for_root(is_root: bool) -> Translations:
        """
        Return the translation sequence used for a node.

        Parameters
        ----------
        is_root
            Whether the node is a skeleton root.
        """
        return Translations.XYZ if is_root else Translations.NONE

    def _append_node(
        self,
        model: BiomechanicalModelReal,
        node_id: int,
        parent_name: str,
        is_root: bool,
    ) -> None:
        """
        Append one FBX skeleton node and its descendants to the biomechanical model.

        Parameters
        ----------
        model
            The model being built.
        node_id
            The identifier of the skeleton node to append.
        parent_name
            The name of the parent segment.
        is_root
            Whether the node is a skeleton root.
        """
        node = self.skeleton_nodes[node_id]
        rt_matrix = self._fbx_euler_xyz_rt_matrix(angles=node.rotation, translation=node.translation)
        model.add_segment(
            SegmentReal(
                name=node.name,
                parent_name=parent_name,
                segment_coordinate_system=SegmentCoordinateSystemReal.from_rt_matrix(
                    rt_matrix=rt_matrix,
                    is_scs_local=True,
                ),
                translations=self._translations_for_root(is_root=is_root),
                rotations=Rotations.ZYX,
            )
        )

        for child_id in node.children_ids:
            self._append_node(model=model, node_id=child_id, parent_name=node.name, is_root=False)

    def to_q(self) -> ParsedAnimation:
        """
        Convert the FBX animation curves into biorbd-compatible generalized coordinates.

        Returns
        -------
        ParsedAnimation
            The extracted generalized coordinates. Rotational DoFs are converted
            from degrees to radians to match biorbd conventions.
        """
        animation_curves = self._animation_curves()
        if not animation_curves:
            raise RuntimeError("The FBX file does not contain any animation curve attached to the skeleton.")

        raw_time_arrays = [curve_data[0] for curve_data in animation_curves.values()]
        start_time = min(int(times[0]) for times in raw_time_arrays if times.size > 0)
        time = np.unique(
            np.concatenate(
                [
                    (times.astype(float) - start_time) / self._FBX_TIME_UNIT
                    for times in raw_time_arrays
                    if times.size > 0
                ]
            )
        )

        dof_names = self._q_dof_names()
        q = np.zeros((len(dof_names), time.shape[0]), dtype=float)
        for dof_index, dof_name in enumerate(dof_names):
            segment_name, property_name, axis = self._parse_dof_name(dof_name)
            curve_data = animation_curves.get((segment_name, property_name, axis))
            if curve_data is None:
                continue

            key_times, key_values = curve_data
            normalized_times = (key_times.astype(float) - start_time) / self._FBX_TIME_UNIT
            if normalized_times.shape[0] == 1:
                q[dof_index, :] = key_values[0]
            else:
                q[dof_index, :] = np.interp(
                    x=time,
                    xp=normalized_times,
                    fp=key_values,
                    left=key_values[0],
                    right=key_values[-1],
                )

            if property_name == "Lcl Rotation":
                q[dof_index, :] = np.deg2rad(q[dof_index, :])

        return ParsedAnimation(q=q, time=time, dof_names=dof_names)

    def animation_diagnostics(self, tolerance: float = 1e-12) -> FbxAnimationDiagnostics:
        """
        Inspect the completeness of the FBX animation import.

        Parameters
        ----------
        tolerance
            Numerical tolerance used to detect zero and constant DoF trajectories.

        Returns
        -------
        FbxAnimationDiagnostics
            A compact diagnostic report for the animation and visual mesh coverage.
        """
        animation_curves = self._animation_curves()
        dof_names = self._q_dof_names()
        mapped_dof_names = []
        for dof_name in dof_names:
            segment_name, property_name, axis = self._parse_dof_name(dof_name)
            if (segment_name, property_name, axis) in animation_curves:
                mapped_dof_names.append(dof_name)

        missing_dof_names = [dof_name for dof_name in dof_names if dof_name not in mapped_dof_names]

        animation = self.to_q()
        zero_dof_names = []
        constant_dof_names = []
        for dof_index, dof_name in enumerate(animation.dof_names):
            values = animation.q[dof_index, :]
            if np.all(np.abs(values) <= tolerance):
                zero_dof_names.append(dof_name)
            if np.all(np.abs(values - values[0]) <= tolerance):
                constant_dof_names.append(dof_name)

        segment_names_with_meshes = self._segment_names_with_visual_meshes()
        segments_without_visual_meshes = [
            node.name for node in self._ordered_skeleton_nodes() if node.name not in segment_names_with_meshes
        ]

        return FbxAnimationDiagnostics(
            frame_count=animation.frame_count,
            duration_seconds=float(animation.time[-1]) if animation.time.size else 0.0,
            dof_count=len(dof_names),
            mapped_dof_count=len(mapped_dof_names),
            missing_dof_names=missing_dof_names,
            zero_dof_names=zero_dof_names,
            constant_dof_names=constant_dof_names,
            ignored_animated_model_nodes=self._ignored_animated_model_nodes(),
            segments_without_visual_meshes=segments_without_visual_meshes,
        )

    def to_real(self) -> BiomechanicalModelReal:
        """
        Convert the parsed FBX skeleton into a real biomechanical model.
        """
        if not self.skeleton_nodes:
            raise RuntimeError("The FBX skeleton has not been parsed.")

        model = BiomechanicalModelReal()
        for root_id in self.root_ids:
            self._append_node(model=model, node_id=root_id, parent_name="base", is_root=True)
        if self.split_meshes_per_segment:
            self._attach_visual_meshes(model=model)
        return model