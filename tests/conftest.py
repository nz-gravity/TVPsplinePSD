from __future__ import annotations

import os

import numpy as np
import pytest

from tv_pspline_psd import PSplineConfig
from tv_pspline_psd.datasets import simulate_ls2

OUTDIR = os.path.join(os.path.dirname(__file__), "out_tests")

@pytest.fixture
def ls2_smoke_data() -> np.ndarray:
	return simulate_ls2(512, rng=np.random.default_rng(0))


@pytest.fixture
def smoke_config() -> PSplineConfig:
	return PSplineConfig(
		n_interior_knots_time=12,
		n_interior_knots_freq=12,
		freq_knot_strategy="linear",
	)


@pytest.fixture
def plot_outdir() -> str:
	os.makedirs(OUTDIR, exist_ok=True)
	return OUTDIR
