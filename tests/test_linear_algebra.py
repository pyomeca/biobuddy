import numpy as np
import numpy.testing as npt
import pytest

import biorbd

from biobuddy import Rotations
from biobuddy.utils.linear_algebra import euler_and_translation_to_matrix, RotoTransMatrix


def test_rt():

    np.random.seed(42)

    for angle_sequence in Rotations:
        if angle_sequence != Rotations.NONE:
            nb_angles = len(angle_sequence.value)
            angles = np.random.rand(nb_angles) * 2 * np.pi
            translations = np.random.rand(3)

            # --- rt from translations and Euler angles --- #
            # TODO: remove when the uniformization is completed
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

            # TODO: Leave this section though
            rt_biobuddy = RotoTransMatrix()
            rt_biobuddy.from_euler_angles_and_translation(angles=angles, angle_sequence=angle_sequence.value, translation=translations)

            rot_biobuddy = rt_biobuddy.rotation_matrix

            rotation_matrix_biorbd = biorbd.Rotation.fromEulerAngles(angles, angle_sequence.value)
            rot_biorbd = rotation_matrix_biorbd.to_array()

            npt.assert_almost_equal(
                rot_biobuddy,
                rot_biorbd,
            )
            npt.assert_almost_equal(translations, rt_biobuddy.translation)

            # --- Euler angles from rotation matrix --- #
            if angle_sequence == Rotations.XYZ:
                angles_biobuddy = rt_biobuddy.euler_angles(angle_sequence=angle_sequence.value)
                angles_biorbd = biorbd.Rotation.toEulerAngles(rotation_matrix_biorbd, angle_sequence.value).to_array()

                npt.assert_almost_equal(
                    angles_biobuddy,
                    angles_biorbd,
                )
            else:
                with pytest.raises(NotImplementedError, match="This angle_sequence is not implemented yet"):
                    angles_biobuddy = rt_biobuddy.euler_angles(angle_sequence=angle_sequence.value)

