import os
import shutil
import numpy as np

from biobuddy.utils import norm2


class VtpParser:
    """
    Convert vtp mesh to triangles mesh
    """
    def __init__(self, geometry_path: str, geometry_cleaned_path: str):

        if not isinstance(geometry_path, str):
            raise ValueError("geometry_path must be a string")
        if not isinstance(geometry_cleaned_path, str):
            raise ValueError("geometry_cleaned_path must be a string")

        if not os.path.exists(geometry_cleaned_path):
            os.makedirs(geometry_cleaned_path)

        log_file_failed = []
        print("Cleaning vtp file into triangles: ")

        # select only files in such that filename.endswith('.vtp') or filename in self.vtp_files
        files = [
            filename
            for filename in os.listdir(geometry_path)
            if filename.endswith(".vtp")
        ]

        for filename in files:
            complete_path = os.path.join(geometry_path, filename)

            with open(complete_path, "r") as f:
                print(complete_path)
                try:
                    mesh = self.read_vtp_file(complete_path)
                    if mesh["polygons"].shape[1] == 3:  # it means it doesn't need to be converted into triangles
                        shutil.copy(complete_path, geometry_cleaned_path)
                    else:
                        poly, nodes, normals = self.transform_polygon_to_triangles(
                            mesh["polygons"], mesh["nodes"], mesh["normals"]
                        )
                        new_mesh = dict(polygons=poly, nodes=nodes, normals=normals)

                        self.write_vtp_file(new_mesh, geometry_cleaned_path, filename)

                except:
                    print(f"Error with {filename}")
                    log_file_failed.append(filename)
                    # if failed we just copy the file in the new folder
                    shutil.copy(complete_path, geometry_cleaned_path)

        if len(log_file_failed) > 0:
            print("Files failed to clean:")
            print(log_file_failed)

    @staticmethod
    def extract_number_from_line(line, pattern):
        """Extracts the number from a given pattern in a line."""
        start_index = line.find(pattern) + len(pattern)
        end_index = line[start_index:].find('"')
        return int(line[start_index : start_index + end_index])

    @staticmethod
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

    def read_vtp_file(self, filename: str) -> dict:
        mesh_dictionnary = {"N_Obj": 1}  # Only 1 object per file

        with open(filename, "r") as file:
            content = file.readlines()

        type_ = None
        i = 0

        for ligne in content:
            if "<Piece" in ligne:
                num_points = self.extract_number_from_line(ligne, 'NumberOfPoints="')
                mesh_dictionnary["normals"] = np.zeros((num_points, 3))
                mesh_dictionnary["nodes"] = np.zeros((num_points, 3))

                num_polys = self.extract_number_from_line(ligne, 'NumberOfPolys="')
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
                    tmp = self.handle_polygons_shape(mesh_dictionnary=mesh_dictionnary, polygon_apex_idx=tmp)

                if type_ == "nodes" and tmp.shape[0] == 6:
                    mesh_dictionnary[type_][i - 1, :] = tmp[0:3]
                    i += 1
                    mesh_dictionnary[type_][i - 1, :] = tmp[3:6]
                else:
                    mesh_dictionnary[type_][i - 1, :] = tmp

        return mesh_dictionnary

    @staticmethod
    def write_data(fid, data, format_string, indent_level=0):
        for row in data:
            fid.write("\t" * indent_level + format_string % tuple(row) + "\n")

    def write_vtp_file(self, mesh_dictionnary: dict, file_path: str, filename: str):
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
            self.write_data(fid, mesh_dictionnary["normals"], "%8.6f %8.6f %8.6f", 4)
            fid.write("\t\t\t\t</DataArray>\n")
            fid.write("\t\t\t</PointData>\n")

            fid.write("\t\t\t<Points>\n")
            fid.write('\t\t\t\t<DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
            self.write_data(fid, mesh_dictionnary["nodes"], "%8.6f %8.6f %8.6f", 4)
            fid.write("\t\t\t\t</DataArray>\n")
            fid.write("\t\t\t</Points>\n")

            fid.write("\t\t\t<Polys>\n")
            fid.write('\t\t\t\t<DataArray type="Int32" Name="connectivity" format="ascii">\n')
            format_chain = " ".join(["%i"] * nb_nodes_polys)
            self.write_data(fid, mesh_dictionnary["polygons"], format_chain, 5)
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


    def transform_polygon_to_triangles(self, polygons, nodes, normals) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform any polygons with more than 3 edges into polygons with 3 edges (triangles)."""

        if polygons.shape[1] == 3:
            return polygons, nodes, normals

        elif polygons.shape[1] == 4:
            return self.convert_quadrangles_to_triangles(polygons, nodes, normals)

        elif polygons.shape[1] > 4:
            return self.convert_polygon_to_triangles(polygons, nodes, normals)

        else:
            RuntimeError("The polygons array must have at least 3 columns.")

    @staticmethod
    def convert_quadrangles_to_triangles(polygons, nodes, normals) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform polygons with 4 edges (quadrangles) into polygons with 3 edges (triangles)."""
        # 1. Search for quadrangles
        quadrangles_idx = np.where((polygons[:, 3] != 0) & (~np.isnan(polygons[:, 3])))[0]
        triangles_idx = np.where(np.isnan(polygons[:, 3]))[0]

        # transform polygons[quadrangles, X] as a list of int
        polygons_0 = polygons[quadrangles_idx, 0].astype(int)
        polygons_1 = polygons[quadrangles_idx, 1].astype(int)
        polygons_2 = polygons[quadrangles_idx, 2].astype(int)
        polygons_3 = polygons[quadrangles_idx, 3].astype(int)

        # 2. Determine triangles to be made
        mH = 0.5 * (nodes[polygons_0] + nodes[polygons_2])  # Barycentres AC
        mK = 0.5 * (nodes[polygons_1] + nodes[polygons_3])  # Barycentres BD
        KH = mH - mK
        AC = -nodes[polygons_0] + nodes[polygons_2]  # Vector AC
        BD = -nodes[polygons_1] + nodes[polygons_3]  # Vector BD
        # Search for the optimal segment for the quadrangle cut
        type_ = np.sign((np.sum(KH * BD, axis=1) / norm2(BD)) ** 2 - (np.sum(KH * AC, axis=1) / norm2(AC)) ** 2)

        # 3. Creation of new triangles
        tBD = np.where(type_ >= 0)[0]
        tAC = np.where(type_ < 0)[0]
        # For BD
        PBD_1 = np.column_stack(
            [polygons[quadrangles_idx[tBD], 0], polygons[quadrangles_idx[tBD], 1], polygons[quadrangles_idx[tBD], 3]]
        )
        PBD_2 = np.column_stack(
            [polygons[quadrangles_idx[tBD], 1], polygons[quadrangles_idx[tBD], 2], polygons[quadrangles_idx[tBD], 3]]
        )
        # For AC
        PAC_1 = np.column_stack(
            [polygons[quadrangles_idx[tAC], 0], polygons[quadrangles_idx[tAC], 1], polygons[quadrangles_idx[tAC], 2]]
        )
        PAC_2 = np.column_stack(
            [polygons[quadrangles_idx[tAC], 2], polygons[quadrangles_idx[tAC], 3], polygons[quadrangles_idx[tAC], 0]]
        )

        # 4. Matrix of final polygons
        new_polygons = np.vstack([polygons[triangles_idx, :3], PBD_1, PBD_2, PAC_1, PAC_2])

        return new_polygons, nodes, normals

    @staticmethod
    def convert_polygon_to_triangles(polygons, nodes, normals) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform any polygons with more than 3 edges into polygons with 3 edges (triangles).
        """

        # Search for polygons with more than 3 edges
        polygons_with_more_than_3_edges = np.where((polygons[:, 3] != 0) & (~np.isnan(polygons[:, 3])))[0]
        polygons_with_3_edges = np.where(np.isnan(polygons[:, 3]))[0]

        triangles = []
        new_normals = []
        for j, poly_idx in enumerate(polygons_with_more_than_3_edges):
            # get only the non-nan values
            current_polygon = polygons[poly_idx, np.isnan(polygons[poly_idx]) == False]
            # Split the polygons into triangles
            # For simplicity, we'll use vertex 0 as the common vertex and form triangles:
            # (0, 1, 2), (0, 2, 3), (0, 3, 4), ..., (0, n-2, n-1)

            for i in range(1, current_polygon.shape[0] - 1):
                triangles.append(np.column_stack([polygons[poly_idx, 0], polygons[poly_idx, i], polygons[poly_idx, i + 1]]))

        return (
            np.vstack([polygons[polygons_with_3_edges, :3], *triangles]),
            nodes,
            normals,
        )
