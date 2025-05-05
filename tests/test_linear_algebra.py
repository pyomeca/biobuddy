import numpy as np
import numpy.testing as npt

import biorbd

from biobuddy import Rotations
from biobuddy.utils.linear_algebra import euler_and_translation_to_matrix


def test_rt():

    np.random.seed(42)

    for angle_sequence in Rotations:
        if angle_sequence != Rotations.NONE:
            nb_angles = len(angle_sequence.value)
            angles = np.random.rand(nb_angles) * 2 * np.pi
            translations = np.random.rand(3)

            rt_biobuddy = euler_and_translation_to_matrix(
                angles=angles, angle_sequence=angle_sequence.value, translations=translations
            )
            rot_biobuddy = rt_biobuddy[:3, :3]
            rot_biorbd = biorbd.Rotation.fromEulerAngles(angles, angle_sequence.value).to_array()

            npt.assert_almost_equal(
                rot_biobuddy,
                rot_biorbd,
            )
            npt.assert_almost_equal(translations, rt_biobuddy[:3, 3])
