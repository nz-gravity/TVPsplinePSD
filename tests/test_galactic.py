import numpy as np
import pytest
import jax
import jax.numpy as jnp

from tv_pspline_psd.galactic import (simulation_parameters, galactic_psd,
    log_galactic_psd, log_galactic_psd_jax, GalacticParameters)


def test_equation_six_and_table_ii():
    p = simulation_parameters()
    assert p.amplitude == 1.14e-44
    assert p.f2_hz == .00031
    assert p.alpha == 1.8
    assert p.f_knee_hz == 10**-2.47
    f = np.geomspace(1e-5,.004,100)
    literal = p.amplitude/2*f**(-7/3)*np.exp(-(f/p.f1_hz)**p.alpha)*(1+np.tanh((p.f_knee_hz-f)/p.f2_hz))
    np.testing.assert_allclose(galactic_psd(f,p),literal,rtol=1e-12,atol=0)
    np.testing.assert_allclose(simulation_parameters(4).f1_hz/p.f1_hz,4**-.25)
    np.testing.assert_allclose(simulation_parameters(4).f_knee_hz/p.f_knee_hz,4**-.27)


def test_tail_and_jax_gradients():
    p = simulation_parameters()
    f = np.geomspace(1e-4,.1,40)
    logs = np.log(list(p.to_dict().values()))
    np.testing.assert_allclose(log_galactic_psd_jax(jnp.array(f),*logs),log_galactic_psd(f,p),rtol=1e-12)
    fn = lambda x: log_galactic_psd_jax(jnp.array([.001,.003,.005]),*x).sum()
    analytic = np.asarray(jax.grad(fn)(jnp.array(logs)))
    step=1e-5
    numeric=np.array([(float(fn(logs+np.eye(5)[i]*step))-float(fn(logs-np.eye(5)[i]*step)))/(2*step) for i in range(5)])
    np.testing.assert_allclose(analytic,numeric,rtol=1e-6)


def test_invalid_parameters():
    with pytest.raises(ValueError): simulation_parameters(0)
    with pytest.raises(ValueError): galactic_psd([0],simulation_parameters())
    with pytest.raises(ValueError): GalacticParameters(-1,1,1,1,1)
