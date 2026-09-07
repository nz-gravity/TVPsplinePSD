"""Seven-parameter foreground + instrument model for diagonal AET powers.

The supplied response weights already contain the WDM projection and pooling.
Evaluating the spectrum at their frequency nodes places the changing spectrum
inside those operations, rather than rescaling a pre-pooled template.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, init_to_value

from .galactic import log_galactic_psd_jax, simulation_parameters
from .multichannel import AETDiagonalPosterior


PARAMETER_NAMES = ("amplitude", "f1_hz", "f2_hz", "f_knee_hz", "alpha")


def parametric_lisa_model(power_sum, counts, tm, oms, response_weights, frequencies, prior_scale=1.):
    """Independent log-normal priors, fixed before observing the realization.

    Noise scales multiply PSD (an ASD amplitude scales as their square root).
    Foreground prior centers use the published one-year mean/SNR7 simulation;
    Eq. 7 does not couple any of the five sampled parameters.
    """
    centre = simulation_parameters().to_dict()
    widths = (1.0, .5, .5, .5, .35)
    logs = [numpyro.sample("log_" + name, dist.Normal(np.log(centre[name]), width*prior_scale))
            for name, width in zip(PARAMETER_NAMES, widths)]
    log_tm = numpyro.sample("log_tm_scale", dist.Normal(0., .5*prior_scale))
    log_oms = numpyro.sample("log_oms_scale", dist.Normal(0., .5*prior_scale))
    spectrum = jnp.exp(log_galactic_psd_jax(frequencies, *logs))
    galaxy = jnp.einsum("ctfq,fq->ctf", response_weights, spectrum)
    noise = jnp.exp(log_tm)*tm + jnp.exp(log_oms)*oms
    total = noise + galaxy
    likelihood = -.5*jnp.sum(counts*jnp.log(total) + power_sum/total)
    numpyro.factor("diagonal_whittle", likelihood)


def fit_parametric_lisa(observed, counts, tm, oms, response_weights, frequencies,
                        *, mask=None, n_warmup=500, n_samples=500, num_chains=2,
                        random_seed=20260906, progress_bar=True, prior_scale=1.,
                        target_accept_probability=.9, max_tree_depth=10):
    """Return physical parameter draws and the existing surface-summary interface."""
    if not np.isfinite(prior_scale) or prior_scale <= 0:
        raise ValueError("prior_scale must be finite and positive")
    observed, counts, tm, oms, response_weights, frequencies = map(
        np.asarray, (observed, counts, tm, oms, response_weights, frequencies))
    if observed.ndim != 3 or observed.shape[0] != 3:
        raise ValueError("observed must have shape (3,time,frequency)")
    if any(x.shape != observed.shape for x in (counts, tm, oms)):
        raise ValueError("counts and noise references must match observed")
    if response_weights.shape[:3] != observed.shape or response_weights.ndim != 4:
        raise ValueError("response weights must have shape (3,time,frequency,node)")
    if frequencies.shape != response_weights.shape[2:]:
        raise ValueError("frequencies must have shape (frequency,node)")
    if num_chains < 2:
        raise ValueError("use at least two chains")
    for x in (counts, tm, oms, response_weights):
        if np.any(~np.isfinite(x)) or np.any(x < 0):
            raise ValueError("weights, counts and references must be finite and nonnegative")
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError("frequency nodes must be finite and positive")
    valid = counts > 0
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid) or np.any(~np.isfinite(observed[valid])) or np.any(observed[valid] < 0):
        raise ValueError("invalid retained powers")
    if np.any(tm + oms <= 0):
        raise ValueError("noise reference must be positive, including held-out cells")
    fit_counts = np.where(valid, counts, 0.)
    summed = fit_counts*np.where(valid, observed, 0.)
    init = {"log_"+k: np.log(v) for k,v in simulation_parameters().to_dict().items()}
    init.update(log_tm_scale=0., log_oms_scale=0.)
    sampler = MCMC(NUTS(parametric_lisa_model, init_strategy=init_to_value(values=init),
                        dense_mass=True, target_accept_prob=target_accept_probability, max_tree_depth=max_tree_depth),
                   num_warmup=n_warmup, num_samples=n_samples, num_chains=num_chains,
                   chain_method="sequential", progress_bar=progress_bar)
    start = time.perf_counter()
    sampler.run(jax.random.PRNGKey(random_seed), *map(jnp.asarray,
                (summed, fit_counts, tm, oms, response_weights, frequencies)),
                prior_scale=prior_scale, extra_fields=("diverging", "accept_prob", "num_steps", "energy"))
    runtime = time.perf_counter()-start
    samples = {k: np.asarray(v) for k,v in sampler.get_samples().items()}
    diagnostics = summary(sampler.get_samples(group_by_chain=True))
    extra = sampler.get_extra_fields(group_by_chain=True)
    energy = np.asarray(extra['energy'])
    diag = dict(num_chains=num_chains,
                divergences=int(np.sum(extra['diverging'])),
                max_r_hat=float(max(np.max(v['r_hat']) for v in diagnostics.values())),
                min_effective_sample_size=float(min(np.min(v['n_eff']) for v in diagnostics.values())),
                tree_depth_saturation_fraction=float(np.mean(np.asarray(extra['num_steps']) >= 2**max_tree_depth-1)),
                min_ebfmi=float(np.min(np.mean(np.diff(energy,axis=1)**2,axis=1)/np.var(energy,axis=1))))
    # Summarize bounded channel/frequency chunks: never allocate draws x full mission.
    intervals = [np.empty((3,)+observed.shape) for _ in range(3)]
    all_logs = jnp.asarray(np.stack([samples['log_'+name] for name in PARAMETER_NAMES],axis=1))
    for c in range(3):
        for lo in range(0,observed.shape[2],16):
            hi = min(lo+16,observed.shape[2])
            freq = jnp.asarray(frequencies[lo:hi])
            spectra = np.asarray(jax.vmap(lambda x: jnp.exp(log_galactic_psd_jax(freq,*x)))(all_logs))
            g = np.einsum('tfq,dfq->dtf',response_weights[c,:,lo:hi],spectra,optimize=True)
            n = (np.exp(samples['log_tm_scale'])[:,None,None]*tm[c,:,lo:hi]
                 + np.exp(samples['log_oms_scale'])[:,None,None]*oms[c,:,lo:hi])
            for out,draws in zip(intervals,(n,g,n+g)):
                out[:,c,:,lo:hi] = np.quantile(draws,[.5,.05,.95],axis=0)
    return AETDiagonalPosterior(
        *intervals[0], *intervals[1], *intervals[2],
        amplitude_draws=np.exp(samples['log_amplitude']),
        f_knee_draws_hz=np.exp(samples['log_f_knee_hz']), diagnostics=diag,
        samples=samples, mcmc=sampler, runtime_seconds=runtime,
        phi_time=float('nan'), phi_frequency=float('nan'), noise_level_log_sd=.5)
