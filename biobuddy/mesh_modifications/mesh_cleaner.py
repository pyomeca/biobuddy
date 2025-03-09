import numpy as np


def transform_polygon_to_triangles(polygons, nodes, normals) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform any polygons with more than 3 edges into polygons with 3 edges (triangles)."""

    if polygons.shape[1] == 3:
        return polygons, nodes, normals

    elif polygons.shape[1] == 4:
        return convert_quadrangles_to_triangles(polygons, nodes, normals)

    elif polygons.shape[1] > 4:
        return convert_polygon_to_triangles(polygons, nodes, normals)

    else:
        RuntimeError("The polygons array must have at least 3 columns.")


def norm2(v):
    """Compute the squared norm of each row of the matrix v."""
    return np.sum(v**2, axis=1)


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
