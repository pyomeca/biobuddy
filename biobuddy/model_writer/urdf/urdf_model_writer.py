from lxml import etree

from ..abstract_model_writer import AbstractModelWriter


class UrdfModelWriter(AbstractModelWriter):

    def _collect_segments(self) -> str:
        out_string = ""
        for segment in self.model.segments:
            out_string += segment.to_urdf(with_mesh=self.with_mesh)
            out_string += "\n\n\n"  # Give some space between segments
        return out_string

    def write(self, model: "BiomechanicalModelReal") -> None:
        """
        Writes the BiomechanicalModelReal into a text file of format .urdf
        """

        # Initialize the model
        model_elts = etree.Element("robot", name="model")
        material_elts = etree.SubElement(model_elts, "material", name="default")
        links_elts = etree.SubElement(model_elts, "links", name="default")
        joints_elts = etree.SubElement(model_elts, "joints", name="default")

        # Write each segment
        for segment in model.segments:
            if segment.segment_coordinate_system.is_in_global:
                raise RuntimeError(
                    f"Something went wrong, the segment coordinate system of segment {segment.name} is expressed in the global."
                )
            segment.to_urdf(material_elts, links_elts, joints_elts, with_mesh=self.with_mesh)

        # Write it to the .urdf file
        tree = etree.ElementTree(model_elts)
        tree.write(self.filepath, pretty_print=True, xml_declaration=True, encoding="utf-8")
