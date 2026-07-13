"""Fast regression tests for Mojito segment windows and CV folds."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

STUDY_DIR = Path(__file__).resolve().parents[1] / "studies" / "ollie_tdi"
sys.path.insert(0, str(STUDY_DIR))
fit_mojito_segment = importlib.import_module("fit_mojito_segment")
mojito_experiments = importlib.import_module("mojito_experiments")
mojito_validation = importlib.import_module("mojito_validation")


@pytest.fixture
def synthetic_mojito(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create ten days of tiny X/Y/Z data at four samples per day."""
    path = tmp_path / "mojito.h5"
    with h5py.File(path, "w") as h:
        grp = h.create_group("processed/segment0")
        x = np.arange(40, dtype=float)
        grp.create_dataset("X", data=x)
        grp.create_dataset("Y", data=x + 100.0)
        grp.create_dataset("Z", data=x + 200.0)

    monkeypatch.setattr(fit_mojito_segment, "DATA", path)
    monkeypatch.setattr(fit_mojito_segment, "DT", 86400.0 / 4.0)
    return path


def test_load_segment_returns_exact_wdm_length(synthetic_mojito: Path) -> None:
    series, start_used = fit_mojito_segment.load_segment(
        "X", nt=2, start_day=1.0, days=2.0
    )

    assert start_used == 1.0
    assert series.shape == (8,)
    np.testing.assert_array_equal(series, np.arange(4.0, 12.0))


def test_load_segment_rejects_out_of_range_window(synthetic_mojito: Path) -> None:
    with pytest.raises(ValueError, match="extends past EOF"):
        fit_mojito_segment.load_segment("X", nt=2, start_day=8.25, days=2.0)

    with pytest.raises(ValueError, match="outside the data span"):
        fit_mojito_segment.load_segment("X", nt=2, start_day=-0.25, days=1.0)


def test_full_cv_fold_is_two_disjoint_348_day_windows() -> None:
    assert mojito_experiments.EXPERIMENTS["full"]["cv"] == (
        (18.0, 348.0),
        (366.0, 348.0),
    )


def test_cross_val_uses_explicit_fold_and_rejects_grid_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, tuple[float, float]] = {}

    def fake_fit(channel, nt, time_knots, start_day, days):
        calls["train"] = (start_day, days)
        result = {
            "freq_grid": np.array([0.1, 0.2]),
            "psd_mean": np.ones((2, 2)),
            "coeffs": np.ones((2, 2)),
        }
        return result, object(), start_day

    def fake_load(channel, nt, start_day, days):
        calls["test"] = (start_day, days)
        return np.ones(2 * nt), start_day

    monkeypatch.setattr(mojito_validation, "fit_channel", fake_fit)
    monkeypatch.setattr(mojito_validation, "load_segment", fake_load)
    monkeypatch.setattr(
        mojito_validation,
        "wdm_analysis_coefficients",
        lambda *args: (
            np.ones((2, 2)),
            np.array([0.25, 0.75]),
            np.array([0.1, 0.21]),
        ),
    )

    with pytest.raises(ValueError, match="frequency grids differ"):
        mojito_validation.cross_val_whitening("full", "X", tmp_path)

    assert calls == {"train": (18.0, 348.0), "test": (366.0, 348.0)}
