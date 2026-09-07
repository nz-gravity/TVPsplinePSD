import jax
import numpy as np
from numpyro import handlers
from tv_pspline_psd.parametric_lisa import parametric_lisa_model
from tv_pspline_psd.galactic import simulation_parameters, galactic_psd


def test_seven_physical_parameters_and_masked_likelihood():
    f=np.array([[.001,.002],[.003,.004]])
    response=np.ones((3,2,2,2))*1e40
    tm=np.ones((3,2,2)); oms=2*tm
    p=simulation_parameters()
    params={'log_'+k:np.log(v) for k,v in p.to_dict().items()}
    params.update(log_tm_scale=0.,log_oms_scale=0.)
    total=tm+oms+np.einsum('ctfq,fq->ctf',response,galactic_psd(f,p))
    counts=np.ones_like(tm)*8; counts[:,1]=0
    power=counts*total
    model=handlers.substitute(handlers.seed(parametric_lisa_model,jax.random.PRNGKey(1)),data=params)
    trace=handlers.trace(model).get_trace(power,counts,tm,oms,response,f)
    sites={k for k,v in trace.items() if v['type']=='sample' and not v['is_observed']}
    assert sites==set(params)
    expected=-.5*np.sum(counts*np.log(total)+power/total)
    np.testing.assert_allclose(trace['diagonal_whittle']['fn'].log_factor,expected,rtol=1e-12)
