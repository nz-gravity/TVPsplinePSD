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


def test_recalibration_preconditioner_is_a_single_block():
    """The recalibration model must sample one K-vector, not two.

    Guards the reduction: a(f) multiplies the whole reference, so the metric is
    a single K x K block with no TM/OMS cross term and therefore no degenerate
    ridge to leave behind.
    """
    from tv_pspline_psd.multichannel import component_noise_preconditioner

    rng = np.random.default_rng(3)
    n_freq, n_knot, n_time = 60, 10, 20
    frequency = np.geomspace(1e-4, 1e-1, n_freq)
    centres = np.linspace(np.log(frequency[0]), np.log(frequency[-1]), n_knot)
    width = (centres[1] - centres[0]) * 1.2
    basis = np.exp(-0.5 * ((np.log(frequency)[:, None] - centres[None, :]) / width) ** 2)
    penalty = np.eye(n_knot)
    transfer_tm = np.abs(rng.normal(1.0, 0.1, (3, n_time, n_freq)))
    transfer_oms = np.abs(rng.normal(1.0, 0.1, (3, n_time, n_freq)))
    template = np.abs(rng.normal(1.0, 0.1, (3, n_time, n_freq))) * 1e-2
    counts = np.full((3, n_time, n_freq), 8.0)
    spectrum_tm, spectrum_oms = 1.0 / frequency**2, np.ones_like(frequency)

    factor = component_noise_preconditioner(
        basis, transfer_tm, transfer_oms, spectrum_tm, spectrum_oms,
        template, counts, penalty, 1e4,
    )
    assert factor.shape == (n_knot, n_knot)
    hessian = np.linalg.inv(factor @ factor.T)
    assert np.linalg.cond(factor.T @ hessian @ factor) == pytest.approx(1.0, abs=1e-6)
