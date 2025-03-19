from ...components.real.biomechanical_model_real import BiomechanicalModelReal
from ...components.real.rigidbody.segment_real import SegmentReal, InertiaParametersReal


class BiomodModelParser:
    def __init__(self, filepath: str):
        # Load the model from the filepath
        with open(filepath) as f:
            content = f.read()
        lines = content.split("\n")

        # Prepare the internal structure to hold the model
        self.segments: dict[str, SegmentReal] = {}

        # Parse the model
        is_block_commenting = False
        current_component = None
        for line in lines:
            # Remove everything after // or between /* */ (comments)
            if not is_block_commenting and "/*" in line:
                is_block_commenting = True
                line = line.split("/*")[0]
            if is_block_commenting and "*/" in line:
                is_block_commenting = False
                line = line.split("*/")[1]
            line = line.split("//")[0]
            line = line.strip()
            if not line:
                continue

            if current_component is None:
                if line.startswith("version"):
                    pass
                elif line.startswith("segment"):
                    current_component = SegmentReal(name=line[len("segment") :].strip())

                else:
                    raise ValueError(f"Unknown component {line}")
            elif isinstance(current_component, SegmentReal):
                if line.startswith("endsegment"):
                    self.segments[current_component.name] = current_component
                    current_component = None
                elif line.startswith("parent"):
                    current_component.parent_name = line[len("parent") :].strip()
                elif line.startswith("translations"):
                    current_component.translations = line[len("translations") :].strip()
                elif line.startswith("rotations"):
                    current_component.rotations = line[len("rotations") :].strip()
                elif line.startswith("mass"):
                    if current_component.inertia_parameters is None:
                        current_component.inertia_parameters = InertiaParametersReal()
                    current_component.mass = line[len("mass") :].strip()
                elif line.startswith("com"):
                    if current_component.inertia_parameters is None:
                        current_component.inertia_parameters = InertiaParametersReal()
                    current_component.com = line[len("com") :].strip()
                elif line.startswith("inertia"):
                    if current_component.inertia_parameters is None:
                        current_component.inertia_parameters = InertiaParametersReal()
                    current_component.inertia_parameters = line[len("inertia") :].strip()
                elif line.startswith("mesh"):
                    current_component.mesh = line[len("mesh") :].strip()
                elif line.startswith("mesh_file"):
                    raise NotImplementedError()
                else:
                    raise ValueError(f"Unknown information in segment")
            else:
                raise ValueError(f"Unknown component {type(current_component)}")

    def to_real(self) -> BiomechanicalModelReal:
        raise NotImplementedError()
