from typing import Iterable
import numpy as np
import logging

from .mesh import Mesh

_logger = logging.getLogger(__name__)


def read_vtp(filename: str) -> Mesh:
    mesh: Mesh = None

    with open(filename, "r") as file:
        content = file.readlines()

    line_type = None
    i = 0

    # First declare the mesh properly
    for line in content:
        if "<Piece" in line:
            num_points = _extract_number_from_line(line, 'NumberOfPoints="')
            mesh = Mesh(polygons=np.zeros((0, 3)), nodes=np.zeros((num_points, 3)), normals=np.zeros((num_points, 3)))
            break
    if mesh is None:
        raise ValueError("The file is not a valid vtp file.")

    # Extract the polygons data
    polygons = []
    offsets = [0.0]
    for line in content:
        if '<PointData Normals="Normals">' in line:
            line_type = "normals"
            i = 0
        elif "<Points>" in line:
            line_type = "nodes"
            i = 0
        elif "<Polys>" in line:
            line_type = "polygons"
            i = 0
        elif 'Name="offsets"' in line:
            line_type = "offsets"

        elif "<" not in line and line_type is not None:
            i += 1
            list_line = line.replace("\t", " ").replace("\n", " ").split(" ")
            tmp = [float(item) for item in list_line if item != ""]

            if line_type == "polygons":
                polygons += tmp

            elif line_type == "nodes":
                if len(tmp) == 6:
                    mesh.nodes[i - 1, :] = tmp[0:3]
                    i += 1
                    mesh.nodes[i - 1, :] = tmp[3:6]
                else:
                    mesh.nodes[i - 1, :] = tmp

            elif line_type == "normals":
                mesh.normals[i - 1, :] = tmp

            elif line_type == "offsets":
                offsets += tmp

            else:
                raise ValueError("The line type is not valid.")

    if offsets == [0.0]:
        raise RuntimeError("The 'offset' field must be declared in the vtp file.")

    # Fill the mesh with the right polygons
    for i in range(len(offsets) - 1):

        # Identify the index of the points in the polygon
        range_start = int(offsets[i])
        range_end = int(offsets[i + 1])

        # Transform the polygons
        polygon = polygons[range_start:range_end]
        polygon_apex_idx = _handle_polygons_shape(polygon_apex_idx=polygon)
        for poly in polygon_apex_idx:
            mesh.polygons = np.vstack((mesh.polygons, np.array(poly)))

    return mesh


def _format_row_data(fid, data: Iterable[float], format_string: str, indent_level: int = 0):
    for row in data:
        fid.write("\t" * indent_level + (format_string % tuple(row)) + "\n")


def write_vtp(filepath: str, mesh: Mesh) -> None:
    """
    Write a mesh to a vtp file.

    Parameters
    ----------
    filepath: str
        The path to the file to write
    mesh: Mesh
        The mesh to write
    """
    with open(filepath, "w") as fid:
        fid.write('<?xml version="1.0"?>\n')
        fid.write(
            '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">\n'
        )
        fid.write("\t<PolyData>\n")

        nb_points = mesh.nodes.shape[0]
        nb_polys = mesh.polygons.shape[0]
        nb_nodes_polys = mesh.polygons.shape[1]

        fid.write(
            f'\t\t<Piece NumberOfPoints="{nb_points}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{nb_polys}">\n'
        )

        fid.write('\t\t\t<PointData Normals="Normals">\n')
        fid.write('\t\t\t\t<DataArray type="Float32" Name="Normals" NumberOfComponents="3" format="ascii">\n')
        _format_row_data(fid, mesh.normals, "%8.6f %8.6f %8.6f", 4)
        fid.write("\t\t\t\t</DataArray>\n")
        fid.write("\t\t\t</PointData>\n")

        fid.write("\t\t\t<Points>\n")
        fid.write('\t\t\t\t<DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        _format_row_data(fid, mesh.nodes, "%8.6f %8.6f %8.6f", 4)
        fid.write("\t\t\t\t</DataArray>\n")
        fid.write("\t\t\t</Points>\n")

        fid.write("\t\t\t<Polys>\n")
        fid.write('\t\t\t\t<DataArray type="Int32" Name="connectivity" format="ascii">\n')
        format_chain = " ".join(["%i"] * nb_nodes_polys)
        _format_row_data(fid, mesh.polygons, format_chain, 5)
        fid.write("\t\t\t\t</DataArray>\n")

        fid.write('\t\t\t\t<DataArray type="Int32" Name="offsets" format="ascii">\n')
        fid.write("\t\t\t\t\t")
        poly_list = np.arange(1, len(mesh.polygons) + 1) * nb_nodes_polys
        fid.write(" ".join(map(str, poly_list)))
        fid.write("\n")
        fid.write("\t\t\t\t</DataArray>\n")
        fid.write("\t\t\t</Polys>\n")

        fid.write("\t\t</Piece>\n")
        fid.write("\t</PolyData>\n")
        fid.write("</VTKFile>\n")


def _extract_number_from_line(line: str, pattern: str) -> int:
    """Extracts the number from a given pattern in a line."""
    start_index = line.find(pattern) + len(pattern)
    end_index = line[start_index:].find('"')
    return int(line[start_index : start_index + end_index])


def _handle_polygons_shape(polygon_apex_idx: list) -> list[list[float]]:
    """Handles the shape of the polygons array."""

    if len(polygon_apex_idx) > 3:
        polygon_apex_list = []

        # Append with triangles starting at the first point
        for i in range(len(polygon_apex_idx) - 2):
            polygon_apex_list += [[polygon_apex_idx[0], polygon_apex_idx[i + 1], polygon_apex_idx[i + 2]]]

        return polygon_apex_list

    elif len(polygon_apex_idx) == 3:  # Already a triangle
        return [polygon_apex_idx]

    else:
        _logger.warning("Something went wrong: a polygon formed of two apex was detected.")
