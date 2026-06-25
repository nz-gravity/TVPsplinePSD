"""Validation gate for datasets/sobh.py: waveform, SNR convention, WDM track.

Checks, at smoke scale:
  1. ripple waveform evaluates over the LISA band (in-band frequency range);
  2. the FFT embedding is self-consistent (round-trip SNR matches direct);
  3. the LW antenna patterns modulate annually with a sane sky-averaged level;
  4. the WDM transform of the signal shows a rising chirp track.

    uv run python studies/sobh_tvpsd/smoke.py
"""

from __future__ import annotations

import numpy as np

from datasets.lisa import lisa_instrument_psd
from datasets.sobh import (
    SOBHParams,
    lisa_lw_antenna,
    sobh_optimal_snr,
    sobh_strain_fd,
    sobh_strain_td,
)

_C = 299_792_458.0
_G = 6.674e-11
_MSUN = 1.989e30
_YEAR = 365.25 * 86400.0


def f_isco(m1: float, m2: float) -> float:
    return _C**3 / (_G * (m1 + m2) * _MSUN * 6.0**1.5 * np.pi)


def main() -> None:
    params = SOBHParams()
    fisco = f_isco(params.m1, params.m2)
    print(f"[source] Mc={params.chirp_mass:.0f} Msun  eta={params.eta:.3f}  "
          f"f_isco={fisco:.2e} Hz")

    # Sampling: Nyquist comfortably above ISCO; ~40-day window placing the merger
    # near tc=0.8 T. n a multiple of 2*nt for the WDM factorisation.
    dt = 0.4 / fisco
    nt = 32
    T = 40.0 * 86400.0
    n = int(round(T / dt / (2 * nt))) * (2 * nt)
    T = n * dt
    params.tc = 0.8 * T
    print(f"[grid] dt={dt:.1f}s  n={n}  T={T/86400:.1f} days  "
          f"f_Nyq={0.5/dt:.2e} Hz  tc={params.tc/86400:.1f} days")

    # 1. Waveform support over the band.
    freqs = np.fft.rfftfreq(n, d=dt)[1:]
    hp, hc = sobh_strain_fd(freqs, params)
    support = np.abs(hp) > 0
    fmin, fmax = freqs[support].min(), freqs[support].max()
    print(f"[waveform] in-band support {fmin:.2e} - {fmax:.2e} Hz  "
          f"({support.sum()} bins, finite={np.all(np.isfinite(hp[support]))})")

    # 2. FFT-embedding convention: embed h+ alone, transform back, compare.
    from datasets.sobh import _embed_fd
    psd = lisa_instrument_psd(np.fft.rfftfreq(n, d=dt))
    x_plus = _embed_fd(hp, freqs, n, dt)
    h_rt = dt * np.fft.rfft(x_plus)  # continuous FT of the round-tripped series
    rel = np.max(np.abs(h_rt[1:] - hp)) / np.max(np.abs(hp))
    snr_fd = np.sqrt(np.sum(4.0 * np.abs(hp) ** 2 / psd[1:] / (n * dt)))
    snr_rt = sobh_optimal_snr(x_plus, dt, psd)
    print(f"[convention] +pol embed round-trip rel-err={rel:.1e}  "
          f"SNR direct={snr_fd:.2f} round-trip={snr_rt:.2f}")

    # Responded single-channel series and its (lower, antenna-modulated) SNR.
    signal = sobh_strain_td(n, dt, params)
    snr_td = sobh_optimal_snr(signal, dt, psd)
    print(f"[snr] LW-responded single-channel SNR={snr_td:.2f}")

    # 3. Annual antenna modulation over a full year, sane sky-averaged level.
    t_year = np.linspace(0.0, _YEAR, 2000)
    fp, fc = lisa_lw_antenna(t_year, params)
    resp = fp**2 + fc**2
    print(f"[antenna] <F+^2+Fx^2>={resp.mean():.3f}  "
          f"swing max/min={resp.max() / max(resp.min(), 1e-6):.1f}x")

    # 4. WDM chirp track: energy-weighted mean frequency should rise toward tc.
    from tv_pspline_psd import PSplineConfig, wdm_analysis_coefficients
    coeffs, tg, fg = wdm_analysis_coefficients(signal, dt, nt, PSplineConfig())
    energy = coeffs**2
    fbar = (energy @ fg) / np.maximum(energy.sum(axis=1), 1e-300)  # per time bin
    # Restrict to bins carrying signal (top-half by total energy).
    e_t = energy.sum(axis=1)
    track = e_t > np.median(e_t)
    rho = np.corrcoef(tg[track], fbar[track])[0, 1]
    print(f"[wdm] coeffs {coeffs.shape}  freq {fg[0]:.2e}-{fg[-1]:.2e} Hz  "
          f"mean-freq vs time corr={rho:+.2f} (>0 => rising chirp track)")


if __name__ == "__main__":
    main()
