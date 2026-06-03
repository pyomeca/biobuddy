from dataclasses import dataclass, field

import numpy as np

from ...components.real.biomechanical_model_real import BiomechanicalModelReal
from ...components.real.rigidbody.segment_coordinate_system_real import (
    SegmentCoordinateSystemReal,
)
from ...components.real.rigidbody.segment_real import SegmentReal
from ...utils.enums import Rotations, Translations
from ..abstract_model_parser import AbstractModelParser
from ..parsed_animation import ParsedAnimation


@dataclass
class _BvhJoint:
    """
    Internal representation of a BVH joint.

    Parameters
    ----------
    name
        The BVH joint name.
    offset
        The local translation from the parent joint to this joint.
    channels
        The BVH channels attached to the joint.
    children
        The direct child joints.
    """

    name: str
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    channels: list[str] = field(default_factory=list)
    children: list["_BvhJoint"] = field(default_factory=list)


class BvhModelParser(AbstractModelParser):
    """
    Parse a BVH file into a :class:`BiomechanicalModelReal`.

    BVH files describe a hierarchy with local offsets and one ordered list of
    channels per joint. The parser converts each BVH joint into one biobuddy
    segment, preserving the channel order as the corresponding translation and
    rotation sequences expected by biorbd-compatible models.
    """

    def __init__(self, filepath: str):
        """
        Load the BVH hierarchy and motion metadata from a file.

        Parameters
        ----------
        filepath
            The path to the BVH file to parse.
        """
        super().__init__(filepath)
        self.root: _BvhJoint | None = None
        self.frame_count: int | None = None
        self.frame_time: float | None = None
        self.motion_data: np.ndarray | None = None
        self._read()

    def _read(self) -> None:
        """
        Read the BVH hierarchy and optional motion block from disk.
        """
        with open(self.filepath, "r") as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]

        if not lines or lines[0] != "HIERARCHY":
            raise ValueError("A BVH file must start with a HIERARCHY block.")

        self.root, next_index = self._parse_joint(lines, 1)

        if next_index >= len(lines):
            return
        if lines[next_index] != "MOTION":
            raise ValueError("Expected a MOTION block after the BVH hierarchy.")

        self._parse_motion(lines[next_index + 1 :])

    def _parse_joint(self, lines: list[str], start_index: int) -> tuple[_BvhJoint, int]:
        """
        Parse one BVH joint block recursively.

        Parameters
        ----------
        lines
            The cleaned BVH lines.
        start_index
            The index where the joint declaration starts.

        Returns
        -------
        tuple[_BvhJoint, int]
            The parsed joint and the next unread line index.
        """
        declaration = lines[start_index].split()
        if len(declaration) < 2 or declaration[0] not in {"ROOT", "JOINT"}:
            raise ValueError(f"Expected a ROOT or JOINT declaration, got: {lines[start_index]}")

        joint = _BvhJoint(name=declaration[1])
        index = start_index + 1
        if lines[index] != "{":
            raise ValueError(f"Expected '{{' after joint declaration {joint.name}.")
        index += 1

        while index < len(lines):
            line = lines[index]
            if line == "}":
                return joint, index + 1
            if line.startswith("OFFSET"):
                joint.offset = self._parse_vector(line, expected_token="OFFSET")
                index += 1
            elif line.startswith("CHANNELS"):
                tokens = line.split()
                expected_channel_count = int(tokens[1])
                joint.channels = tokens[2:]
                if len(joint.channels) != expected_channel_count:
                    raise ValueError(
                        f"Joint {joint.name} declares {expected_channel_count} channels "
                        f"but provides {len(joint.channels)}."
                    )
                index += 1
            elif line.startswith("JOINT"):
                child_joint, index = self._parse_joint(lines, index)
                joint.children.append(child_joint)
            elif line == "End Site":
                index = self._skip_end_site(lines, index)
            else:
                raise ValueError(f"Unsupported BVH hierarchy line: {line}")

        raise ValueError(f"Joint {joint.name} is missing a closing brace.")

    @staticmethod
    def _parse_vector(line: str, expected_token: str) -> np.ndarray:
        """
        Parse a three-component BVH vector line.

        Parameters
        ----------
        line
            The line to parse.
        expected_token
            The token expected at the beginning of the line.
        """
        tokens = line.split()
        if len(tokens) != 4 or tokens[0] != expected_token:
            raise ValueError(f"Expected '{expected_token} x y z', got: {line}")
        return np.array([float(token) for token in tokens[1:]], dtype=float)

    @staticmethod
    def _skip_end_site(lines: list[str], start_index: int) -> int:
        """
        Skip a BVH ``End Site`` block.

        End sites do not correspond to movable segments, so they are ignored
        while parsing the biomechanical hierarchy.
        """
        if lines[start_index + 1] != "{":
            raise ValueError("Expected '{' after End Site.")
        if not lines[start_index + 2].startswith("OFFSET"):
            raise ValueError("Expected an OFFSET line inside End Site.")
        if lines[start_index + 3] != "}":
            raise ValueError("Expected '}' after End Site OFFSET.")
        return start_index + 4

    def _parse_motion(self, lines: list[str]) -> None:
        """
        Parse the BVH motion metadata and samples.
        """
        if len(lines) < 2 or not lines[0].startswith("Frames:") or not lines[1].startswith("Frame Time:"):
            raise ValueError("A BVH MOTION block must define Frames and Frame Time.")

        self.frame_count = int(lines[0].split(":", maxsplit=1)[1].strip())
        self.frame_time = float(lines[1].split(":", maxsplit=1)[1].strip())

        motion_rows = [[float(value) for value in line.split()] for line in lines[2:]]
        if len(motion_rows) != self.frame_count:
            raise ValueError(f"Expected {self.frame_count} BVH motion rows, got {len(motion_rows)}.")

        expected_channels = self._count_channels(self.root)
        if any(len(row) != expected_channels for row in motion_rows):
            raise ValueError(f"Each BVH motion row must contain {expected_channels} channel values.")

        self.motion_data = np.array(motion_rows, dtype=float)

    def _count_channels(self, joint: _BvhJoint | None) -> int:
        """
        Count all BVH channels in a hierarchy.
        """
        if joint is None:
            return 0
        return len(joint.channels) + sum(self._count_channels(child) for child in joint.children)

    @staticmethod
    def _get_translations(channels: list[str]) -> Translations:
        """
        Convert BVH translation channels into a biobuddy translation sequence.
        """
        translations = "".join(channel[0].lower() for channel in channels if channel.endswith("position"))
        return Translations(translations) if translations else Translations.NONE

    @staticmethod
    def _get_rotations(channels: list[str]) -> Rotations:
        """
        Convert BVH rotation channels into a biobuddy rotation sequence.
        """
        rotations = "".join(channel[0].lower() for channel in channels if channel.endswith("rotation"))
        return Rotations(rotations) if rotations else Rotations.NONE

    def _append_joint(self, model: BiomechanicalModelReal, joint: _BvhJoint, parent_name: str) -> None:
        """
        Append one BVH joint and all its descendants to a biomechanical model.
        """
        model.add_segment(
            SegmentReal(
                name=joint.name,
                parent_name=parent_name,
                segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                    angles=np.zeros(3),
                    angle_sequence="xyz",
                    translation=joint.offset,
                    is_scs_local=True,
                ),
                translations=self._get_translations(joint.channels),
                rotations=self._get_rotations(joint.channels),
            )
        )

        for child in joint.children:
            self._append_joint(model=model, joint=child, parent_name=joint.name)

    def _motion_channel_names(self, joint: _BvhJoint | None) -> list[str]:
        """
        Return the BVH channel names in file order.

        Parameters
        ----------
        joint
            The hierarchy node from which to collect channels.
        """
        if joint is None:
            return []

        channel_names = []
        for channel in joint.channels:
            if channel.endswith("position"):
                channel_names.append(f"{joint.name}_trans{channel[0].upper()}")
            elif channel.endswith("rotation"):
                channel_names.append(f"{joint.name}_rot{channel[0].upper()}")
            else:
                raise ValueError(f"Unsupported BVH channel '{channel}' in joint {joint.name}.")

        for child in joint.children:
            channel_names.extend(self._motion_channel_names(child))
        return channel_names

    def _q_dof_names(self, joint: _BvhJoint | None) -> list[str]:
        """
        Return the biorbd-compatible DoF names implied by the BVH hierarchy.

        Parameters
        ----------
        joint
            The hierarchy node from which to collect DoFs.
        """
        if joint is None:
            return []

        dof_names = []
        for channel in joint.channels:
            if channel.endswith("position"):
                dof_names.append(f"{joint.name}_trans{channel[0].upper()}")
        for channel in joint.channels:
            if channel.endswith("rotation"):
                dof_names.append(f"{joint.name}_rot{channel[0].upper()}")

        for child in joint.children:
            dof_names.extend(self._q_dof_names(child))
        return dof_names

    def to_q(self) -> ParsedAnimation:
        """
        Convert the BVH motion block into biorbd-compatible generalized coordinates.

        Returns
        -------
        ParsedAnimation
            The extracted generalized coordinates. Rotational DoFs are converted
            from degrees to radians to match biorbd conventions.
        """
        if self.root is None:
            raise RuntimeError("The BVH hierarchy has not been parsed.")
        if self.motion_data is None or self.frame_time is None:
            raise RuntimeError("The BVH file does not contain motion samples.")

        source_dof_names = self._motion_channel_names(self.root)
        target_dof_names = self._q_dof_names(self.root)
        source_index_by_name = {dof_name: dof_index for dof_index, dof_name in enumerate(source_dof_names)}
        reordered_indices = [source_index_by_name[dof_name] for dof_name in target_dof_names]

        q = self.motion_data[:, reordered_indices].T.astype(float)
        for dof_index, dof_name in enumerate(target_dof_names):
            if "_rot" in dof_name:
                q[dof_index, :] = np.deg2rad(q[dof_index, :])

        time = np.arange(self.frame_count, dtype=float) * self.frame_time
        return ParsedAnimation(q=q, time=time, dof_names=target_dof_names)

    def to_real(self) -> BiomechanicalModelReal:
        """
        Convert the parsed BVH hierarchy into a real biomechanical model.
        """
        if self.root is None:
            raise RuntimeError("The BVH hierarchy has not been parsed.")

        model = BiomechanicalModelReal()
        self._append_joint(model=model, joint=self.root, parent_name="base")
        return model
