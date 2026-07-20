from __future__ import annotations

from tv_pspline_psd import PSplineConfig, datasets, fit_log_pspline_surface


def test_public_api_imports() -> None:
    assert callable(fit_log_pspline_surface)
    assert PSplineConfig().degree_time == 3


def test_dataset_public_exports_exist() -> None:
    assert all(hasattr(datasets, name) for name in datasets.__all__)
