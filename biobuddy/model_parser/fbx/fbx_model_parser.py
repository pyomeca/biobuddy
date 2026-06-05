from dataclasses import dataclass, field
import struct

import numpy as np

from ...components.real.biomechanical_model_real import BiomechanicalModelReal
from ...components.real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ...components.real.rigidbody.segment_real import SegmentReal
from ...utils.enums import Rotations, Translations
from ..abstract_model_parser import AbstractModelParser


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
        The local rest rotation used to orient the segment coordinate system.
    children_ids
        The direct child skeleton node identifiers.
    """

    node_id: int
    name: str
    node_type: str
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    children_ids: list[int] = field(default_factory=list)


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

    def __init__(self, filepath: str):
        """
        Load the FBX skeleton hierarchy from a file.

        Parameters
        ----------
        filepath
            The path to the FBX file to parse.
        """
        super().__init__(filepath)
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

        return _FbxNodeRecord(name=name, properties=properties, children=children), end_offset

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
        if property_type in "fdiilbc":
            array_length, encoding, compressed_length = struct.unpack_from("<III", data, cursor)
            cursor += 12
            return {"array_length": array_length, "encoding": encoding}, cursor + compressed_length

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
        properties_record = next((child for child in model_record.children if child.name == "Properties70"), None)
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

    def _extract_skeleton(self) -> None:
        """
        Build the skeleton hierarchy from ``Objects`` and ``Connections``.
        """
        objects_record = next((record for record in self._top_level_records if record.name == "Objects"), None)
        connections_record = next((record for record in self._top_level_records if record.name == "Connections"), None)
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
    def _translations_for_root(is_root: bool) -> Translations:
        """
        Return the translation sequence used for a node.

        Parameters
        ----------
        is_root
            Whether the node is a skeleton root.
        """
        return Translations.XYZ if is_root else Translations.NONE

    def _append_node(self, model: BiomechanicalModelReal, node_id: int, parent_name: str, is_root: bool) -> None:
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
        model.add_segment(
            SegmentReal(
                name=node.name,
                parent_name=parent_name,
                segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                    angles=np.deg2rad(node.rotation),
                    angle_sequence="xyz",
                    translation=node.translation,
                    is_scs_local=True,
                ),
                translations=self._translations_for_root(is_root=is_root),
                rotations=Rotations.XYZ,
            )
        )

        for child_id in node.children_ids:
            self._append_node(model=model, node_id=child_id, parent_name=node.name, is_root=False)

    def to_real(self) -> BiomechanicalModelReal:
        """
        Convert the parsed FBX skeleton into a real biomechanical model.
        """
        if not self.skeleton_nodes:
            raise RuntimeError("The FBX skeleton has not been parsed.")

        model = BiomechanicalModelReal()
        for root_id in self.root_ids:
            self._append_node(model=model, node_id=root_id, parent_name="base", is_root=True)
        return model
