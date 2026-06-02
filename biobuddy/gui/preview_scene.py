from dataclasses import dataclass

import numpy as np

from ..components.real.biomechanical_model_real import BiomechanicalModelReal


@dataclass
class PreviewScene:
    """
    Lightweight geometric representation used by the desktop preview widget.
    """

    joints: dict[str, np.ndarray]
    bones: list[tuple[str, str]]
    markers: dict[str, np.ndarray]
    muscles: dict[str, list[np.ndarray]]


def build_preview_scene(model: BiomechanicalModelReal) -> PreviewScene:
    """
    Build a static zero-pose preview scene from a biomechanical model.
    """
    global_jcs = model.forward_kinematics()
    joints = {name: global_jcs[name][0].translation for name in model.segment_names}
    bones = [
        (segment.parent_name, segment.name)
        for segment in model.segments
        if segment.parent_name != "base" and segment.parent_name in joints
    ]

    markers = {}
    for segment in model.segments:
        rt = global_jcs[segment.name][0]
        for marker in segment.markers:
            markers[marker.name] = np.asarray((rt @ marker.mean_position)[:3]).reshape(3)

    muscles = {}
    for muscle_group in model.muscle_groups:
        for muscle in muscle_group.muscles:
            path = []
            origin_rt = global_jcs[muscle.origin_position.parent_name][0]
            path.append((origin_rt @ muscle.origin_position.position)[:3, 0])
            for via_point in muscle.via_points:
                if via_point.condition is None and via_point.movement is None:
                    via_rt = global_jcs[via_point.parent_name][0]
                    path.append((via_rt @ via_point.position)[:3, 0])
            insertion_rt = global_jcs[muscle.insertion_position.parent_name][0]
            path.append((insertion_rt @ muscle.insertion_position.position)[:3, 0])
            muscles[muscle.name] = path

    return PreviewScene(joints=joints, bones=bones, markers=markers, muscles=muscles)
