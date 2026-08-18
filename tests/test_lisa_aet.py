import numpy as np

from tv_pspline_psd.lisa_aet import (
    XYZ_TO_AET,
    xyz_covariance_to_aet_diagonal,
    xyz_to_aet_series,
)


def test_xyz_to_aet_is_orthonormal_and_preserves_sample_power():
    np.testing.assert_allclose(XYZ_TO_AET @ XYZ_TO_AET.T, np.eye(3), atol=1e-15)
    xyz = np.arange(24, dtype=float).reshape(3, 8)
    aet = xyz_to_aet_series(xyz)
    np.testing.assert_allclose(np.sum(aet**2, axis=0), np.sum(xyz**2, axis=0))


def test_full_covariance_rotation_returns_only_correct_aet_diagonal():
    covariance = np.asarray(
        [
            [2.0, 0.4 + 0.1j, -0.2j],
            [0.4 - 0.1j, 3.0, 0.3],
            [0.2j, 0.3, 5.0],
        ]
    )
    expected = np.diag(XYZ_TO_AET @ covariance @ XYZ_TO_AET.T).real
    np.testing.assert_allclose(
        xyz_covariance_to_aet_diagonal(covariance), expected
    )
