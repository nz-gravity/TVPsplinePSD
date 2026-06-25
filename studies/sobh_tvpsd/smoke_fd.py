"""Validate the FD stationary baseline: PSD recovery + dL recovery."""
import numpy as np
from datasets.lisa import lisa_instrument_psd
from datasets.lisa_tdi import _draw_colored
from datasets.sobh import SOBHParams, sobh_strain_td
from studies.sobh_tvpsd.fd_baseline import estimate_stationary_psd, fd_dL_posterior

def main():
    n, dt = 2**13, 4.0
    freq = np.fft.rfftfreq(n, d=dt); fe = freq.copy(); fe[0]=fe[1]
    S_phys = lisa_instrument_psd(fe)            # physical one-sided 1/Hz
    rng = np.random.default_rng(0)
    noise = _draw_colored(S_phys, n, dt, rng)   # one-sided PSD = S_phys
    est = estimate_stationary_psd(noise, dt, n_knots=24, n_warmup=200, n_samples=200)
    interior = (freq > 0) & (freq < freq[-1])
    ratio = est["S_stat"][interior] / S_phys[interior]
    print(f"[psd] K={est['K']:.3e}  recovered/physical median={np.median(ratio):.3f} "
          f"IQR {np.percentile(ratio,25):.3f}-{np.percentile(ratio,75):.3f}")

    # dL recovery: inject SOBH at d_ref into the same noise.
    p = SOBHParams(); p.tc = 0.6*n*dt
    d_ref = p.distance
    h_ref = sobh_strain_td(n, dt, p)
    snr = np.sqrt(np.sum(4.0/(n*dt) * np.abs(dt*np.fft.rfft(h_ref))[1:]**2 / S_phys[1:]))
    # scale to a moderate SNR by moving distance
    target_snr = 25.0
    dl_true = d_ref * snr / target_snr
    p.distance = dl_true
    data = sobh_strain_td(n, dt, p) + noise
    post = fd_dL_posterior(data, h_ref, dt, S_stat=est["S_stat"], d_ref=d_ref,
                           dl_ref=dl_true, dl_scale=0.5, n_warmup=400, n_samples=800)
    dl = post["dL"]
    print(f"[dL] true={dl_true:.4g}  post={dl.mean():.4g} +/- {dl.std():.3g}  "
          f"z={(dl.mean()-dl_true)/dl.std():+.2f}  (inj SNR~{target_snr})")

if __name__ == "__main__":
    main()
