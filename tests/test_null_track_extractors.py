"""Focused tests for the Figure 7 null-track diagnostics."""

from __future__ import annotations

import numpy as np

from studies.ollie_tdi.fit_aet_fullband import _quadratic_minimum_track


def test_quadratic_minimum_track_recovers_subchannel_vertex() -> None:
    freq = np.linspace(0.055, 0.065, 101)
    truth = np.array([0.05943, 0.06017])
    surface = np.stack([(freq - center) ** 2 for center in truth])
    surface = surface[None, :, :]
    recovered = _quadratic_minimum_track(surface, freq)[0]
    np.testing.assert_allclose(recovered, truth, atol=2e-10)
