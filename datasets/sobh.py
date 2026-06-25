"""Chirping black-hole binary in the LISA band: ripple waveform + LW response.

A massive/intermediate-mass black-hole binary sweeps a *diagonal chirp track*
through the LISA time-frequency plane over the observation and (for the masses we
use) coalesces within it. The merger epoch ``tc`` places that track against the
**cyclostationary** Galactic-confusion noise of :mod:`datasets.lisa`, so the
source is measured against a noise floor that changes by several-fold over the
year -- the time-variation our WDM estimator resolves and a stationary PSD does
not. True stellar-mass (~30 Msun) systems sit at the top of the LISA band and
barely chirp over a year; an intermediate-mass system (total ~1e4-1e5 Msun)
gives the cleanest in-band track, so that is the default here.

The intrinsic frequency-domain strain ``h+(f), hx(f)`` comes from ``ripple``
(jax-native IMRPhenomD, differentiable -- so the joint source+noise fit can use
NUTS end to end). The single-channel LISA output is built in the **long-
wavelength approximation**: analytic LISA spacecraft orbits give the two arm
unit vectors, hence the Michelson detector tensor ``D(t) = 1/2 (n1 (x) n1 -
n2 (x) n2)``, whose contraction with the (psi-rotated) polarization tensors gives
the time-dependent antenna patterns ``F+(t), Fx(t)``. The detector output is

    h(t) = F+(t) h+(t) + Fx(t) hx(t),

valid while the response varies slowly compared with the GW phase (the LW
regime, good across most of the LISA band).

FFT convention (matching :mod:`datasets.lisa_tdi`): for a real series of length
``n`` at spacing ``dt`` the continuous Fourier transform is ``h(f_k) = dt *
rfft(x)_k`` and the one-sided PSD is ``S(f_k)``; ripple's ``h(f)`` is the
continuous transform, embedded as ``rfft_k = h/dt``. The matched-filter SNR is
``SNR^2 = 4 df sum_k |h_k|^2 / S(f_k)`` with ``df = 1/(n dt)``.

Requires the optional ``[lisa]`` extra (``uv pip install -e '.[lisa]'``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_MSUN_S = 4.925490947e-6  # G Msun / c^3 in seconds
_MPC_M = 3.085677581e22   # 1 Mpc in metres
_C = 299_792_458.0
_AU = 1.495978707e11      # metres
_YEAR = 365.25 * 86400.0


def _require_ripple():
    try:
        import ripplegw.waveforms.IMRPhenomD  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The SOBH demo needs the [lisa] extra (ripplegw): "
            "uv pip install -e '.[lisa]'"
        ) from exc


@dataclass
class SOBHParams:
    """Source parameters for the chirping LISA binary.

    Masses are detector-frame solar masses, spins are dimensionless aligned
    components, ``distance`` is luminosity distance in Mpc, ``tc`` is the
    coalescence time in seconds from the start of the observation, ``phic`` the
    coalescence phase, ``inc`` the inclination, and ``(ecl_lon, ecl_lat, psi)``
    the ecliptic sky location and polarization for the LISA response.
    """

    m1: float = 3.0e4
    m2: float = 1.0e4
    chi1: float = 0.0
    chi2: float = 0.0
    distance: float = 3.0e3      # Mpc
    tc: float = 0.85 * _YEAR     # seconds from start of observation
    phic: float = 0.0
    inc: float = 0.5
    ecl_lon: float = 1.2         # ecliptic longitude (rad)
    ecl_lat: float = 0.3         # ecliptic latitude (rad)
    psi: float = 0.7             # polarization angle (rad)
    f_ref: float = 1.0e-3        # reference frequency (Hz)

    @property
    def chirp_mass(self) -> float:
        return (self.m1 * self.m2) ** 0.6 / (self.m1 + self.m2) ** 0.2

    @property
    def eta(self) -> float:
        return (self.m1 * self.m2) / (self.m1 + self.m2) ** 2

    def ripple_theta(self) -> np.ndarray:
        """``[Mc, eta, chi1, chi2, D(Mpc), tc, phic, inc]`` for ripple hphc."""
        return np.array([
            self.chirp_mass, self.eta, self.chi1, self.chi2,
            self.distance, self.tc, self.phic, self.inc,
        ], dtype=float)


def sobh_strain_fd(
    freqs: np.ndarray, params: SOBHParams
) -> tuple[np.ndarray, np.ndarray]:
    """Frequency-domain plus/cross strain ``h+(f), hx(f)`` from ripple IMRPhenomD.

    Returns complex arrays on ``freqs`` (Hz); entries outside the waveform's
    support are zero. ``freqs`` must be strictly positive (drop the DC bin).
    """
    _require_ripple()
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc

    f = np.asarray(freqs, dtype=float)
    if np.any(f <= 0.0):
        raise ValueError("freqs must be strictly positive (exclude the DC bin).")
    hp, hc = gen_IMRPhenomD_hphc(
        jnp.asarray(f), jnp.asarray(params.ripple_theta()), float(params.f_ref)
    )
    return np.asarray(hp), np.asarray(hc)


def _lisa_spacecraft_positions(t: np.ndarray, *, arm_length: float, kappa: float = 0.0,
                               orbit_lambda: float = 0.0) -> np.ndarray:
    """Analytic first-order LISA spacecraft positions ``(3, 3, n_t)`` [metres].

    Cornish & Rubbo (2003) cartwheel orbit: the constellation barycentre orbits
    the Sun at 1 AU with annual period while the equilateral triangle rotates.
    Returns positions ``r[sc, axis, t]`` for ``sc = 0, 1, 2``.
    """
    t = np.asarray(t, dtype=float)
    alpha = 2.0 * np.pi * t / _YEAR + kappa
    ecc = arm_length / (2.0 * np.sqrt(3.0) * _AU)
    pos = np.empty((3, 3, t.size))
    for k in range(3):
        beta_k = 2.0 * np.pi * k / 3.0 + orbit_lambda
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta_k), np.sin(beta_k)
        pos[k, 0] = _AU * ca + _AU * ecc * (sa * ca * sb - (1.0 + sa**2) * cb)
        pos[k, 1] = _AU * sa + _AU * ecc * (sa * ca * cb - (1.0 + ca**2) * sb)
        pos[k, 2] = -np.sqrt(3.0) * _AU * ecc * np.cos(alpha - beta_k)
    return pos


def _polarization_tensors(ecl_lon: float, ecl_lat: float, psi: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Plus/cross polarization tensors ``e+, ex`` (3x3) in ecliptic coordinates."""
    # Unit vector toward the source; propagation is -u.
    clat, slat = np.cos(ecl_lat), np.sin(ecl_lat)
    clon, slon = np.cos(ecl_lon), np.sin(ecl_lon)
    u = np.array([clat * clon, clat * slon, slat])
    # Wave-frame basis (p, q) transverse to the propagation direction.
    p = np.array([slon, -clon, 0.0])
    p = p / np.linalg.norm(p)
    q = np.cross(u, p)
    c2, s2 = np.cos(2.0 * psi), np.sin(2.0 * psi)
    p_psi = c2 * p + s2 * q
    q_psi = -s2 * p + c2 * q
    e_plus = np.outer(p_psi, p_psi) - np.outer(q_psi, q_psi)
    e_cross = np.outer(p_psi, q_psi) + np.outer(q_psi, p_psi)
    return e_plus, e_cross


def lisa_lw_antenna(
    t: np.ndarray, params: SOBHParams, *, arm_length: float = 2.5e9
) -> tuple[np.ndarray, np.ndarray]:
    """Long-wavelength LISA antenna patterns ``F+(t), Fx(t)`` for one Michelson.

    Built constructively: analytic spacecraft orbits give the two arm unit
    vectors at spacecraft 0, hence the detector tensor ``D = 1/2(n1 (x) n1 -
    n2 (x) n2)``, contracted with the polarization tensors.
    """
    t = np.asarray(t, dtype=float)
    pos = _lisa_spacecraft_positions(t, arm_length=arm_length)
    r01 = pos[1] - pos[0]
    r02 = pos[2] - pos[0]
    n1 = r01 / np.linalg.norm(r01, axis=0)
    n2 = r02 / np.linalg.norm(r02, axis=0)
    e_plus, e_cross = _polarization_tensors(params.ecl_lon, params.ecl_lat, params.psi)
    # D_ij = 1/2 (n1_i n1_j - n2_i n2_j); F = D_ij e_ij, vectorised over time.
    d = 0.5 * (np.einsum("it,jt->ijt", n1, n1) - np.einsum("it,jt->ijt", n2, n2))
    f_plus = np.einsum("ijt,ij->t", d, e_plus)
    f_cross = np.einsum("ijt,ij->t", d, e_cross)
    return f_plus, f_cross


def _embed_fd(h_fd: np.ndarray, freqs_pos: np.ndarray, n: int, dt: float) -> np.ndarray:
    """Time series from a positive-frequency continuous-FT strain (rfft = h/dt)."""
    spectrum = np.zeros(n // 2 + 1, dtype=complex)
    spectrum[1:] = np.nan_to_num(h_fd) / dt
    return np.fft.irfft(spectrum, n=n)


def sobh_strain_td(n: int, dt: float, params: SOBHParams, *, arm_length: float = 2.5e9
                   ) -> np.ndarray:
    """Single-channel LISA time series ``h(t) = F+(t) h+(t) + Fx(t) hx(t)``.

    The polarizations are formed in the time domain by inverse-transforming the
    ripple FD strain, then modulated by the slowly varying LW antenna patterns.
    """
    freqs = np.fft.rfftfreq(n, d=dt)
    hp_fd, hc_fd = sobh_strain_fd(freqs[1:], params)
    x_plus = _embed_fd(hp_fd, freqs[1:], n, dt)
    x_cross = _embed_fd(hc_fd, freqs[1:], n, dt)
    t = np.arange(n, dtype=float) * dt
    f_plus, f_cross = lisa_lw_antenna(t, params, arm_length=arm_length)
    return f_plus * x_plus + f_cross * x_cross


def sobh_optimal_snr(signal: np.ndarray, dt: float, psd_onesided: np.ndarray) -> float:
    """Matched-filter optimal SNR of a real time series against a one-sided PSD.

    ``psd_onesided`` is evaluated on ``rfftfreq(len(signal), dt)`` (same grid the
    embedding uses). The DC bin is excluded.
    """
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=dt)
    h = dt * np.fft.rfft(signal)  # continuous FT
    df = 1.0 / (n * dt)
    mask = freq > 0
    integrand = 4.0 * np.abs(h[mask]) ** 2 / psd_onesided[mask] * df
    return float(np.sqrt(np.sum(integrand)))
