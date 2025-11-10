from lxml import etree

from ..abstract_model_writer import AbstractModelWriter


class UrdfModelWriter(AbstractModelWriter):

    def write(self, model: "BiomechanicalModelReal") -> None:
        """
        Writes the BiomechanicalModelReal into a text file of format .urdf
        """

        urdf_model = etree.Element("robot", name="model")

        # Write each segment
        for segment in model.segments:
            if segment.segment_coordinate_system.is_in_global:
                raise RuntimeError(
                    f"Something went wrong, the segment coordinate system of segment {segment.name} is expressed in the global."
                )
            if segment.name != "root":
                if segment.parent_name == "base":
                    # First segment, written as a simple link without a joint
                    link = etree.SubElement(urdf_model, "link", name=segment.name)
                    if segment.inertia_parameters is not None:
                        segment.inertia_parameters.to_urdf(link)
                else:
                    # Regular segment with joint
                    segment.to_urdf(urdf_model, with_mesh=self.with_mesh)

        # No muscles yet
        if len(model.muscle_groups) != 0:
            raise NotImplementedError("Muscles are not implemented yet for URDF export")

        # Write it to the .urdf file
        tree = etree.ElementTree(urdf_model)
        tree.write(self.filepath, pretty_print=True, xml_declaration=True, encoding="utf-8")
