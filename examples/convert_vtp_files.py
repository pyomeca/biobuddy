"""
This example shows how to convert vtp files to triangles.
"""

from pathlib import Path

from biobuddy import VtpParser


if __name__ == "__main__":

    # Paths
    current_path_file = Path(__file__).parent
    geometry_path = f"{current_path_file}/models/Geometry"
    geometry_cleaned_path = f"{current_path_file}/models/Geometry_cleaned"

    # Convert the files
    VtpParser(geometry_path=geometry_path, geometry_cleaned_path=geometry_cleaned_path)
