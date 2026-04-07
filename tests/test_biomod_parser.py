import pytest
import numpy as np
import numpy.testing as npt
import os

from biobuddy import (
    BiomechanicalModelReal,
)


def test_pi_parsing():
    """
    Read a simple biomod file that contains pi expressions
    """

    # Paths
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_path = parent_path + "/examples/models"
    biomod_filepath = f"{root_path}/pendulum.bioMod"

    # Load the model
    model = BiomechanicalModelReal().from_biomod(filepath=biomod_filepath)

    # Check the pi values
    q_min = model.segments["Seg1"].q_ranges.min_bound
    q_max = model.segments["Seg1"].q_ranges.max_bound
    npt.assert_almost_equal(q_min, np.array([-1, -2*np.pi]))
    npt.assert_almost_equal(q_max, np.array([5, 2*np.pi]))

def test_problematic_parsing():
    """
    Read a simple biomod file that contains an unknown expressions
    """

    # Paths
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_path = parent_path + "/examples/models"
    biomod_filepath = f"{root_path}/pendulum.bioMod"

    # Replace a line with a wrong expression
    with open(biomod_filepath, "r") as f:
        content = f.read()
    content = content.replace("com  -0.0005 0.0688 -0.9542", "com  -0.0005 0.0688 bad_name")

    bad_file_name = biomod_filepath.replace(".bioMod", "_bad.bioMod")
    with open(bad_file_name, "w") as f:
        f.write(content)

    # Load the model
    with pytest.raises(ValueError, match="could not convert string to float: 'bad_name'"):
        model = BiomechanicalModelReal().from_biomod(filepath=bad_file_name)

    # Delete the bad file created
    os.remove(bad_file_name)
