from __future__ import annotations

from tv_pspline_psd import PSplineConfig, fit_log_pspline_surface


def test_public_api_imports() -> None:
    assert callable(fit_log_pspline_surface)
    assert PSplineConfig().degree_time == 3