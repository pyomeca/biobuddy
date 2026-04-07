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
    npt.assert_almost_equal(q_min, np.array([-1, -2 * np.pi]))
    npt.assert_almost_equal(q_max, np.array([5, 2 * np.pi]))


def test_variable_parsing():
    """
    Read a simple biomod file that contains variable expressions
    """

    # Paths
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_path = parent_path + "/examples/models"
    biomod_filepath = f"{root_path}/pendulum.bioMod"

    # Load the model
    model = BiomechanicalModelReal().from_biomod(filepath=biomod_filepath)

    # Test the variable values
    mass = model.segments["Seg1"].inertia_parameters.mass
    npt.assert_almost_equal(mass, 1.0)

    inertia = model.segments["Seg1"].inertia_parameters.inertia
    npt.assert_almost_equal(inertia[0, 0], 0.0391)
    npt.assert_almost_equal(
        inertia,
        np.array(
            [[0.0391, 0.0, 0.0, 0.0], [0.0, 0.0335, -0.0032, 0.0], [0.0, -0.0032, 0.009, 0.0], [0.0, 0.0, 0.0, 1.0]]
        ),
    )

    marker = model.segments["Seg1"].markers["marker_1"].position
    npt.assert_almost_equal(
        marker.reshape(
            4,
        ),
        np.array([-1.2, 0, 0, 1]),
    )


def test_block_comment_parsing():
    """
    Read a simple biomod file that contains a block comment
    """

    # Paths
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_path = parent_path + "/examples/models"
    biomod_filepath = f"{root_path}/pendulum.bioMod"

    # Load the model
    model = BiomechanicalModelReal().from_biomod(filepath=biomod_filepath)

    # Check that there is only one segment, because the second one is between block comment
    assert model.nb_segments == 2
    assert model.segment_names == ["root", "Seg1"]


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
    with pytest.raises(ValueError, match="Invalid expression detected in your biomod file: bad_name"):
        model = BiomechanicalModelReal().from_biomod(filepath=bad_file_name)

    # Delete the bad file created
    os.remove(bad_file_name)
