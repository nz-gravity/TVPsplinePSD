"""FD->WDM front end must match the time-domain path exactly."""

from __future__ import annotations

import numpy as np
import pytest

from tv_pspline_psd import (
    PSplineConfig,
    wdm_analysis_coefficients,
    wdm_analysis_coefficients_from_fd,
)


def test_fd_path_matches_time_domain_path_exactly() -> None:
    rng = np.random.default_rng(0)
    n, dt, nt = 512, 5.0, 16
    x = rng.standard_normal(n)
    config = PSplineConfig(
        trim_time_bins=1, trim_low_freq_channels=1, trim_high_freq_channels=1
    )

    td_coeffs, td_time, td_freq = wdm_analysis_coefficients(x, dt, nt, config)
    fd = dt * np.fft.rfft(x)  # h(f) = dt * rfft(x)
    fd_coeffs, fd_time, fd_freq = wdm_analysis_coefficients_from_fd(fd, dt, nt, config)

    np.testing.assert_array_equal(td_time, fd_time)
    np.testing.assert_array_equal(td_freq, fd_freq)
    np.testing.assert_allclose(fd_coeffs, td_coeffs, rtol=0, atol=1e-10)


def test_fd_path_rejects_bad_input() -> None:
    config = PSplineConfig()
    with pytest.raises(ValueError, match="one-dimensional"):
        wdm_analysis_coefficients_from_fd(np.zeros((2, 65), dtype=complex), 5.0, 16, config)
    bad = np.zeros(257, dtype=complex)
    bad[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        wdm_analysis_coefficients_from_fd(bad, 5.0, 16, config)
