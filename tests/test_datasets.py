from __future__ import annotations

import numpy as np

from tv_pspline_psd.datasets import simulate_ls2


def test_simulate_ls2_returns_expected_shape() -> None:
    sample = simulate_ls2(64, rng=np.random.default_rng(0))
    assert sample.shape == (64,)
