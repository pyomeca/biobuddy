from typing import Iterable
import numpy as np

from .mesh import Mesh
from ..utils import norm2


def read_vtp(filename: str) -> Mesh:
    mesh: Mesh = None

    with open(filename, "r") as file:
        content = file.readlines()

    line_type = None
    i = 0

    # First declare the mesh properly
    for line in content:
        if "<Piece" in line:
            num_polys = _extract_number_from_line(line, 'NumberOfPolys="')
            num_points = _extract_number_from_line(line, 'NumberOfPoints="')
            mesh = Mesh(
                polygons=np.zeros((num_polys, 3)), nodes=np.zeros((num_points, 3)), normals=np.zeros((num_points, 3))
            )
            break
    if mesh is None:
        raise ValueError("The file is not a valid vtp file.")

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
            line_type = None

        if "<" not in line and line_type is not None:
            i += 1
            tmp = np.fromstring(line, sep=" ")

            if line_type == "polygons":
                tmp = _handle_polygons_shape(mesh=mesh, polygon_apex_idx=tmp)

            elif line_type == "nodes":
                if tmp.shape[0] == 6:
                    mesh.nodes[i - 1, :] = tmp[0:3]
                    i += 1
                    mesh.nodes[i - 1, :] = tmp[3:6]
                else:
                    mesh.nodes[i - 1, :] = tmp

            elif line_type == "normals":
                mesh.normals[i - 1, :] = tmp

            else:
                raise ValueError("The line type is not valid.")

    if mesh.polygons.shape[1] == 3:  # it means it doesn't need to be converted into triangles
        return mesh
    else:
        return _transform_polygon_to_mesh(mesh)


def _format_row_data(fid, data: Iterable[float], format_string: str, indent_level: int = 0):
    for row in data:
        fid.write(f"{"\t"* indent_level}{format_string % tuple(row)}\n")


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

        nb_polys = mesh.polygons.shape[0]
        nb_nodes_polys = mesh.polygons.shape[1]
        nb_points = mesh.nodes.shape[0]

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


def _handle_polygons_shape(mesh: Mesh, polygon_apex_idx: np.ndarray) -> np.ndarray:
    """Handles the shape of the polygons array."""
    if polygon_apex_idx.size > mesh.polygons.shape[1]:
        tp: np.ndarray = np.zeros((mesh.polygons.shape[0], polygon_apex_idx.size))
        tp[:, : mesh.polygons.shape[1]] = mesh.polygons
        diff = polygon_apex_idx.size - mesh.polygons.shape[1]
        tp[:, mesh.polygons.shape[1] :] = np.repeat(mesh.polygons[:, -1].reshape(-1, 1), diff, axis=1)
        mesh.polygons = tp
    elif polygon_apex_idx.size < mesh.polygons.shape[1]:
        diff = mesh.polygons.shape[1] - polygon_apex_idx.size
        polygon_apex_idx = np.hstack([polygon_apex_idx, np.repeat(None, diff)])
    return polygon_apex_idx


def _convert_polygon_to_mesh(mesh: Mesh) -> Mesh:
    """
    Transform any polygons with more than 3 edges into polygons with 3 edges (triangles).
    """

    # Search for polygons with more than 3 edges
    polygons_with_more_than_3_edges = np.where((mesh.polygons[:, 3] != 0) & (~np.isnan(mesh.polygons[:, 3])))[0]
    polygons_with_3_edges = np.where(np.isnan(mesh.polygons[:, 3]))[0]

    triangles = []
    new_normals = []  # TODO Why isn't this used?
    for j, poly_idx in enumerate(polygons_with_more_than_3_edges):
        # get only the non-nan values
        current_polygon = mesh.polygons[poly_idx, np.isnan(mesh.polygons[poly_idx]) == False]
        # Split the polygons into triangles
        # For simplicity, we'll use vertex 0 as the common vertex and form triangles:
        # (0, 1, 2), (0, 2, 3), (0, 3, 4), ..., (0, n-2, n-1)

        for i in range(1, current_polygon.shape[0] - 1):
            triangles.append(
                np.column_stack(
                    [mesh.polygons[poly_idx, 0], mesh.polygons[poly_idx, i], mesh.polygons[poly_idx, i + 1]]
                )
            )

    return Mesh(
        polygons=np.vstack([mesh.polygons[polygons_with_3_edges, :3], *triangles]),
        nodes=mesh.nodes,
        normals=mesh.normals,
    )


def _convert_quadrangles_to_mesh(mesh: Mesh) -> Mesh:
    """Transform polygons with 4 edges (quadrangles) into polygons with 3 edges (triangles)."""
    # 1. Search for quadrangles
    quadrangles_idx = np.where((mesh.polygons[:, 3] != 0) & (~np.isnan(mesh.polygons[:, 3])))[0]
    triangles_idx = np.where(np.isnan(mesh.polygons[:, 3]))[0]

    # transform polygons[quadrangles, X] as a list of int
    polygons_0 = mesh.polygons[quadrangles_idx, 0].astype(int)
    polygons_1 = mesh.polygons[quadrangles_idx, 1].astype(int)
    polygons_2 = mesh.polygons[quadrangles_idx, 2].astype(int)
    polygons_3 = mesh.polygons[quadrangles_idx, 3].astype(int)

    # 2. Determine triangles to be made
    mH = 0.5 * (mesh.nodes[polygons_0] + mesh.nodes[polygons_2])  # Barycentres AC
    mK = 0.5 * (mesh.nodes[polygons_1] + mesh.nodes[polygons_3])  # Barycentres BD
    KH = mH - mK
    AC = -mesh.nodes[polygons_0] + mesh.nodes[polygons_2]  # Vector AC
    BD = -mesh.nodes[polygons_1] + mesh.nodes[polygons_3]  # Vector BD
    # Search for the optimal segment for the quadrangle cut
    type_ = np.sign((np.sum(KH * BD, axis=1) / norm2(BD)) ** 2 - (np.sum(KH * AC, axis=1) / norm2(AC)) ** 2)

    # 3. Creation of new triangles
    tBD = np.where(type_ >= 0)[0]
    tAC = np.where(type_ < 0)[0]
    # For BD
    PBD_1 = np.column_stack(
        [
            mesh.polygons[quadrangles_idx[tBD], 0],
            mesh.polygons[quadrangles_idx[tBD], 1],
            mesh.polygons[quadrangles_idx[tBD], 3],
        ]
    )
    PBD_2 = np.column_stack(
        [
            mesh.polygons[quadrangles_idx[tBD], 1],
            mesh.polygons[quadrangles_idx[tBD], 2],
            mesh.polygons[quadrangles_idx[tBD], 3],
        ]
    )
    # For AC
    PAC_1 = np.column_stack(
        [
            mesh.polygons[quadrangles_idx[tAC], 0],
            mesh.polygons[quadrangles_idx[tAC], 1],
            mesh.polygons[quadrangles_idx[tAC], 2],
        ]
    )
    PAC_2 = np.column_stack(
        [
            mesh.polygons[quadrangles_idx[tAC], 2],
            mesh.polygons[quadrangles_idx[tAC], 3],
            mesh.polygons[quadrangles_idx[tAC], 0],
        ]
    )

    # 4. Matrix of final polygons
    new_polygons = np.vstack([mesh.polygons[triangles_idx, :3], PBD_1, PBD_2, PAC_1, PAC_2])

    return Mesh(polygons=new_polygons, nodes=mesh.nodes, normals=mesh.normals)


def _transform_polygon_to_mesh(mesh: Mesh) -> Mesh:
    """Transform any polygons with more than 3 edges into polygons with 3 edges (triangles)."""

    if mesh.polygons.shape[1] == 3:
        return mesh

    elif mesh.polygons.shape[1] == 4:
        return _convert_quadrangles_to_mesh(mesh)

    elif mesh.polygons.shape[1] > 4:
        return _convert_polygon_to_mesh(mesh)

    else:
        RuntimeError("The polygons array must have at least 3 columns.")
