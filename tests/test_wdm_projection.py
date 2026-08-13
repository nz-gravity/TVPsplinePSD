import numpy as np
import pytest
from wdm_transform import get_backend
from wdm_transform.windows import phi_unit

from tv_pspline_psd.wdm_projection import (
    collapse_wdm_frequency_projection,
    wdm_frequency_projection_grid,
    wdm_frequency_quadrature,
)


def test_wdm_quadrature_is_symmetric_normalized_and_compact():
    offsets, weights = wdm_frequency_quadrature(24)
    assert np.allclose(offsets, -offsets[::-1])
    assert np.allclose(weights, weights[::-1])
    assert np.all(weights > 0.0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-14)
    assert np.max(np.abs(offsets)) < 2.0 / 3.0
    upstream_window = np.asarray(
        phi_unit(get_backend("numpy"), offsets, 1.0 / 3.0, 1.0)
    )
    _, legendre_weights = np.polynomial.legendre.leggauss(offsets.size)
    expected = legendre_weights * upstream_window**2
    expected /= expected.sum()
    assert np.allclose(weights, expected)


def test_projection_preserves_constant_and_linear_spectra():
    frequency = np.linspace(0.01, 0.1, 12)
    grid, weights = wdm_frequency_projection_grid(frequency, 2.5e-4)
    constant = collapse_wdm_frequency_projection(np.full_like(grid, 7.0), weights)
    linear = collapse_wdm_frequency_projection(2.0 + 3.0 * grid, weights)
    assert np.allclose(constant, 7.0)
    assert np.allclose(linear, 2.0 + 3.0 * frequency)


def test_projection_matches_quadratic_curvature_formula():
    frequency = np.array([0.03, 0.06])
    delta_f = 3.25e-5
    grid, weights = wdm_frequency_projection_grid(frequency, delta_f, n_nodes=32)
    projected = collapse_wdm_frequency_projection(grid**2, weights)
    offsets, same_weights = wdm_frequency_quadrature(32)
    expected = frequency**2 + delta_f**2 * np.sum(same_weights * offsets**2)
    assert np.allclose(projected, expected, rtol=2e-14, atol=0.0)


def test_projection_regularizes_a_deep_but_smooth_null_without_distant_leakage():
    center = np.array([0.03])
    delta_f = 3.255208333333333e-5
    grid, weights = wdm_frequency_projection_grid(center, delta_f, n_nodes=32)
    width = 1.0e-3
    floor = 1.0e-12
    spectrum = floor + ((grid - center[:, None]) / width) ** 2
    projected = collapse_wdm_frequency_projection(spectrum, weights)[0]
    assert projected > 1.0e3 * floor
    assert projected < 1.0e-3
    assert np.max(np.abs(grid - center[:, None])) < 2.0 * delta_f / 3.0


def test_projection_rejects_dc_crossing_and_bad_shapes():
    with pytest.raises(ValueError, match="crosses zero"):
        wdm_frequency_projection_grid(np.array([1.0e-6]), 1.0e-3)
    with pytest.raises(ValueError, match="final values dimension"):
        collapse_wdm_frequency_projection(np.ones((2, 3)), np.ones(2) / 2.0)
