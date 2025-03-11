import os
import pytest
from pathlib import Path

from biobuddy import VtpParser


def test_cleaning_vtp_files():

    # Paths
    current_path_file = Path(__file__).parent
    geometry_path = f"{current_path_file}/../examples/models/Geometry"
    geometry_cleaned_path = f"{current_path_file}/../examples/models/Geometry_cleaned"

    # First, remove the files
    for file in os.listdir(geometry_cleaned_path):
        os.remove(os.path.join(geometry_cleaned_path, file))
    assert len(os.listdir(geometry_cleaned_path)) == 0  # No files left

    # Then, convert vtp files
    VtpParser(geometry_path, geometry_cleaned_path)
    assert len(os.listdir(geometry_path)) == 344
    assert len(os.listdir(geometry_cleaned_path)) == 344
