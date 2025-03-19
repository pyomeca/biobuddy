import numpy as np

from ...components.real.biomechanical_model_real import BiomechanicalModelReal
from ...components.real.rigidbody.segment_real import SegmentReal, InertiaParametersReal, MeshReal


class BiomodModelParser:
    def __init__(self, filepath: str):
        # Load the model from the filepath
        with open(filepath) as f:
            content = f.read()
        lines = content.split("\n")

        # Prepare the internal structure to hold the model
        self.segments: dict[str, SegmentReal] = {}

        # Do a first pass to remove every commented content
        is_block_commenting = False
        line_index = 0
        for line_index in range(len(lines)):
            line = lines[line_index]
            # Remove everything after // or between /* */ (comments)
            if "/*" in line and "*/" in line:
                # Deal with the case where the block comment is on the same line
                is_block_commenting = False
                line = (line.split("/*")[0] + "" + line.split("*/")[1]).strip()
            if not is_block_commenting and "/*" in line:
                is_block_commenting = True
                line = line.split("/*")[0]
            if is_block_commenting and "*/" in line:
                is_block_commenting = False
                line = line.split("*/")[1]
            line = line.split("//")[0]
            line = line.strip()
            lines[line_index] = line

        # Make spaces also a separator
        lines_tp = []
        for line in lines:
            lines_tp.extend(line.split(" "))
        elements = [line for line in lines_tp if line]

        # Parse the model
        current_component = None
        element_index = 0
        while element_index < len(elements):
            element = elements[element_index]
            if current_component is None:
                if element == "version":
                    element_index += 1
                    # Do nothing with the version number
                elif element == "segment":
                    element_index += 1
                    element = elements[element_index]
                    current_component = SegmentReal(name=element)
                else:
                    raise ValueError(f"Unknown component {element}")

            elif isinstance(current_component, SegmentReal):
                if element == "endsegment":
                    self.segments[current_component.name] = current_component
                    current_component = None
                elif element == "parent":
                    element_index += 1
                    element = elements[element_index]
                    current_component.parent_name = element
                elif element == "translations":
                    element_index += 1
                    element = elements[element_index]
                    current_component.translations = element
                elif element == "rotations":
                    element_index += 1
                    element = elements[element_index]
                    current_component.rotations = element
                elif element == "mass":
                    element_index += 1
                    element = elements[element_index]
                    if current_component.inertia_parameters is None:
                        current_component.inertia_parameters = InertiaParametersReal()
                    current_component.inertia_parameters.mass = float(element)
                elif element == "com":
                    if current_component.inertia_parameters is None:
                        current_component.inertia_parameters = InertiaParametersReal()
                    com = []
                    for _ in range(3):
                        element_index += 1
                        element = elements[element_index]
                        com.append(float(element))
                    current_component.inertia_parameters.center_of_mass = np.array(com)
                elif element == "inertia":
                    if current_component.inertia_parameters is None:
                        current_component.inertia_parameters = InertiaParametersReal()
                    inertia = []
                    for _ in range(9):
                        element_index += 1
                        element = elements[element_index]
                        inertia.append(float(element))
                    current_component.inertia_parameters.inertia = np.array(inertia).reshape((3, 3))
                elif element == "mesh":
                    if current_component.mesh is None:
                        current_component.mesh = MeshReal()
                    position = []
                    for _ in range(3):
                        element_index += 1
                        element = elements[element_index]
                        position.append(float(element))
                    current_component.mesh.positions = np.concatenate(
                        (current_component.mesh.positions, np.array([position]).T), axis=1
                    )
                elif element == "mesh_file":
                    raise NotImplementedError()
                else:
                    raise ValueError(f"Unknown information in segment")
            else:
                raise ValueError(f"Unknown component {type(current_component)}")

            element_index += 1

    def to_real(self) -> BiomechanicalModelReal:
        raise NotImplementedError()
