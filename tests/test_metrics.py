import numpy as np
import pytest

from tv_pspline_psd.metrics import bias_log_psd, mse_log_psd, rmse_log_psd


def test_bias_is_the_signed_mean_log_error():
    """Bias must keep its sign; RMSE cannot distinguish an offset from scatter."""
    truth = np.array([1.0, 2.0, 4.0, 8.0])
    offsets = np.array([0.2, 0.1, 0.3, 0.2])
    estimate = truth * np.exp(offsets)
    assert bias_log_psd(truth, estimate) == pytest.approx(offsets.mean())
    # A pure offset has |bias| == rmse only when the offset is constant.
    constant = truth * np.exp(0.25)
    assert bias_log_psd(truth, constant) == pytest.approx(0.25)
    assert rmse_log_psd(truth, constant) == pytest.approx(0.25)


def test_zero_mean_scatter_has_rmse_without_bias():
    truth = np.array([1.0, 2.0, 4.0, 8.0])
    estimate = truth * np.exp(np.array([0.3, -0.3, 0.3, -0.3]))
    assert bias_log_psd(truth, estimate) == pytest.approx(0.0, abs=1e-12)
    assert rmse_log_psd(truth, estimate) == pytest.approx(0.3)


def test_rmse_is_the_root_of_mse():
    truth = np.array([1.0, 3.0, 9.0])
    estimate = np.array([1.1, 2.7, 10.0])
    assert rmse_log_psd(truth, estimate) ** 2 == pytest.approx(mse_log_psd(truth, estimate))


def test_bias_rejects_mismatched_or_nonpositive_input():
    with pytest.raises(ValueError):
        bias_log_psd(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        bias_log_psd(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
