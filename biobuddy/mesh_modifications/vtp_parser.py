import os
import numpy as np


def extract_number_from_line(line, pattern):
    """Extracts the number from a given pattern in a line."""
    start_index = line.find(pattern) + len(pattern)
    end_index = line[start_index:].find('"')
    return int(line[start_index : start_index + end_index])


def handle_polygons_shape(mesh_dictionnary: dict, polygon_apex_idx: np.ndarray) -> np.ndarray:
    """Handles the shape of the polygons array."""
    if polygon_apex_idx.size > mesh_dictionnary["polygons"].shape[1]:
        Mat = np.zeros((mesh_dictionnary["polygons"].shape[0], polygon_apex_idx.size))
        Mat[:, : mesh_dictionnary["polygons"].shape[1]] = mesh_dictionnary["polygons"]
        diff = polygon_apex_idx.size - mesh_dictionnary["polygons"].shape[1]
        Mat[:, mesh_dictionnary["polygons"].shape[1] :] = np.repeat(
            mesh_dictionnary["polygons"][:, -1].reshape(-1, 1), diff, axis=1
        )
        mesh_dictionnary["polygons"] = Mat
    elif polygon_apex_idx.size < mesh_dictionnary["polygons"].shape[1]:
        diff = mesh_dictionnary["polygons"].shape[1] - polygon_apex_idx.size
        polygon_apex_idx = np.hstack([polygon_apex_idx, np.repeat(None, diff)])
    return polygon_apex_idx


def read_vtp_file(filename: str) -> dict:
    mesh_dictionnary = {"N_Obj": 1}  # Only 1 object per file

    with open(filename, "r") as file:
        content = file.readlines()

    type_ = None
    i = 0

    for ligne in content:
        if "<Piece" in ligne:
            num_points = extract_number_from_line(ligne, 'NumberOfPoints="')
            mesh_dictionnary["normals"] = np.zeros((num_points, 3))
            mesh_dictionnary["nodes"] = np.zeros((num_points, 3))

            num_polys = extract_number_from_line(ligne, 'NumberOfPolys="')
            mesh_dictionnary["polygons"] = np.zeros((num_polys, 3))

        elif '<PointData Normals="Normals">' in ligne:
            type_ = "normals"
            i = 0
        elif "<Points>" in ligne:
            type_ = "nodes"
            i = 0
        elif "<Polys>" in ligne:
            type_ = "polygons"
            i = 0
        elif 'Name="offsets"' in ligne:
            type_ = None
        elif "<" not in ligne and type_ is not None:
            i += 1
            tmp = np.fromstring(ligne, sep=" ")

            if type_ == "polygons":
                tmp = handle_polygons_shape(mesh_dictionnary=mesh_dictionnary, polygon_apex_idx=tmp)

            if type_ == "nodes" and tmp.shape[0] == 6:
                mesh_dictionnary[type_][i - 1, :] = tmp[0:3]
                i += 1
                mesh_dictionnary[type_][i - 1, :] = tmp[3:6]
            else:
                mesh_dictionnary[type_][i - 1, :] = tmp

    return mesh_dictionnary


def write_data(fid, data, format_string, indent_level=0):
    for row in data:
        fid.write("\t" * indent_level + format_string % tuple(row) + "\n")


def write_vtp_file(mesh_dictionnary: dict, file_path: str, filename: str):
    filepath = os.path.join(file_path, filename)

    with open(filepath, "w") as fid:
        fid.write('<?xml version="1.0"?>\n')
        fid.write(
            '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">\n'
        )
        fid.write("\t<PolyData>\n")

        nb_points = mesh_dictionnary["nodes"].shape[0]
        nb_polys = mesh_dictionnary["polygons"].shape[0]
        nb_nodes_polys = mesh_dictionnary["polygons"].shape[1]

        fid.write(
            f'\t\t<Piece NumberOfPoints="{nb_points}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{nb_polys}">\n'
        )

        fid.write('\t\t\t<PointData Normals="Normals">\n')
        fid.write('\t\t\t\t<DataArray type="Float32" Name="Normals" NumberOfComponents="3" format="ascii">\n')
        write_data(fid, mesh_dictionnary["normals"], "%8.6f %8.6f %8.6f", 4)
        fid.write("\t\t\t\t</DataArray>\n")
        fid.write("\t\t\t</PointData>\n")

        fid.write("\t\t\t<Points>\n")
        fid.write('\t\t\t\t<DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        write_data(fid, mesh_dictionnary["nodes"], "%8.6f %8.6f %8.6f", 4)
        fid.write("\t\t\t\t</DataArray>\n")
        fid.write("\t\t\t</Points>\n")

        fid.write("\t\t\t<Polys>\n")
        fid.write('\t\t\t\t<DataArray type="Int32" Name="connectivity" format="ascii">\n')
        format_chain = " ".join(["%i"] * nb_nodes_polys)
        write_data(fid, mesh_dictionnary["polygons"], format_chain, 5)
        fid.write("\t\t\t\t</DataArray>\n")

        fid.write('\t\t\t\t<DataArray type="Int32" Name="offsets" format="ascii">\n')
        fid.write("\t\t\t\t\t")
        poly_list = np.arange(1, len(mesh_dictionnary["polygons"]) + 1) * nb_nodes_polys
        fid.write(" ".join(map(str, poly_list)))
        fid.write("\n")
        fid.write("\t\t\t\t</DataArray>\n")
        fid.write("\t\t\t</Polys>\n")

        fid.write("\t\t</Piece>\n")
        fid.write("\t</PolyData>\n")
        fid.write("</VTKFile>\n")
