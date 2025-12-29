import pytest
import numpy as np
import numpy.testing as npt
from pathlib import Path

from biobuddy.utils.marker_data import MarkerData, C3dData, CsvData



def test_marker_data_csv():

    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    # Load MarkerData from CSV
    marker_data = CsvData(csv_path=csv_path)