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

    # Convert osim to biomod
    model = BiomechanicalModelReal().from_biomod(filepath=biomod_filepath)

    # Check the pi values
    q_min = model.segments["Seg1"].q_ranges.min_bound
    q_max = model.segments["Seg1"].q_ranges.max_bound
    npt.assert_almost_equal(q_min, np.array([-1, -2*np.pi]))
    npt.assert_almost_equal(q_max, np.array([5, 2*np.pi]))

