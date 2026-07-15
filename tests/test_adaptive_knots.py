"""Tests for experimental data-adaptive knot placement."""

from __future__ import annotations

import numpy as np

from tv_pspline_psd.adaptive_knots import fit_running_median_chi2_knots


def _notched_power() -> tuple[np.ndarray, np.ndarray]:
    freq = np.linspace(1e-4, 0.1, 2400)
    log_psd = 0.3 * np.log(freq / 0.02) ** 2
    for center, width, depth in ((0.03, 7e-4, 5.0), (0.06, 9e-4, 6.0)):
        log_psd -= depth * np.exp(-0.5 * ((freq - center) / width) ** 2)
    base = np.exp(log_psd)
    modulation = np.exp(0.04 * np.sin(np.arange(24)[:, None] * 1.7))
    return freq, modulation * base[None, :]


def test_running_median_chi2_knots_concentrate_near_notches() -> None:
    freq, power = _notched_power()
    result = fit_running_median_chi2_knots(
        power,
        freq,
        30,
        median_window_hz=5e-4,
    )
    uniform = np.linspace(freq[0], freq[-1], 32)[1:-1]

    assert result.knots.shape == (30,)
    assert np.all(np.diff(result.knots) > 0)
    assert result.knots[0] > freq[0]
    assert result.knots[-1] < freq[-1]
    for center in (0.03, 0.06):
        adaptive_count = np.count_nonzero(np.abs(result.knots - center) < 0.002)
        uniform_count = np.count_nonzero(np.abs(uniform - center) < 0.002)
        assert adaptive_count > uniform_count


def test_running_median_chi2_knots_are_scale_invariant() -> None:
    freq, power = _notched_power()
    reference = fit_running_median_chi2_knots(
        power,
        freq,
        24,
        median_window_hz=1e-3,
    )
    rescaled = fit_running_median_chi2_knots(
        power * 1e-34,
        freq,
        24,
        median_window_hz=1e-3,
    )

    np.testing.assert_allclose(reference.knots, rescaled.knots, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        reference.running_median_log_power - reference.running_median_log_power[0],
        rescaled.running_median_log_power - rescaled.running_median_log_power[0],
        rtol=0.0,
        atol=1e-12,
    )
