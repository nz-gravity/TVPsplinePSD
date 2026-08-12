"""Measure the WDM transform's effective frequency kernel K(m, f).

For stationary noise, E[w_nm^2] = int |g_nm(f)|^2 S(f) df, so the time-summed
response of channel m to a unit tone at f0 IS the kernel evaluated at f0.
Injecting tones and reading the coefficients measures the kernel of the code as
implemented -- discretisation sidelobes included -- with no window algebra.
"""
import numpy as np
from wdm_transform import TimeSeries

NT = 32
NF = 4096
DT = 2.0
N = NT * NF
DF = 1.0 / (2.0 * NF * DT)          # WDM channel spacing
BINS_PER_CHANNEL = NT // 2          # rfft bins spanned by one WDM channel

t = np.arange(N) * DT
m0 = NF // 2                        # a mid-band channel, far from DC/Nyquist edges


def channel_response(offset_bins: int) -> np.ndarray:
    """Sum_n w_nm^2 for a unit tone offset `offset_bins` rfft bins above f_m0."""
    k = m0 * BINS_PER_CHANNEL + offset_bins
    tone = np.sqrt(2.0) * np.cos(2.0 * np.pi * (k / (N * DT)) * t)
    coeffs = np.asarray(TimeSeries(tone, dt=DT).to_wdm(nt=NT).coeffs)
    if coeffs.ndim == 3:
        coeffs = coeffs[0]
    return (coeffs**2).sum(axis=0)


# Sweep one full channel spacing; combined with the m-axis this tiles the whole
# kernel profile at a resolution of DF / BINS_PER_CHANNEL.
u_list, k_list = [], []
for offset in range(BINS_PER_CHANNEL):
    response = channel_response(offset)
    m = np.arange(response.size)
    u_list.append(offset / BINS_PER_CHANNEL - (m - m0))   # (f0 - f_m) / DF
    k_list.append(response)

u = np.concatenate(u_list)
kernel = np.concatenate(k_list)
order = np.argsort(u)
u, kernel = u[order], kernel[order]
kernel = kernel / kernel.max()

np.savez("wdm_kernel.npz", u=u, kernel=kernel, nt=NT, nf=NF, dt=DT, df=DF)

print(f"nt={NT} nf={NF} df={DF:.4e} Hz   kernel samples={u.size}")
print("\nkernel vs offset u = (f - f_m)/df   [normalised to peak]")
for lo, hi in [(0.0, 0.5), (0.5, 0.7), (0.7, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]:
    sel = (np.abs(u) >= lo) & (np.abs(u) < hi)
    if sel.any():
        print(f"  {lo:4.1f} <= |u| < {hi:<6.1f} n={sel.sum():4d}  "
              f"max={kernel[sel].max():.3e}  median={np.median(kernel[sel]):.3e}")

ideal_support = np.abs(u) <= 2.0 / 3.0
outside = kernel[~ideal_support]
print(f"\nideal window support is |u| <= 2/3 (a=1/3)")
print(f"  power fraction inside  : {kernel[ideal_support].sum() / kernel.sum():.10f}")
print(f"  power fraction outside : {outside.sum() / kernel.sum():.3e}")
print(f"  worst sidelobe outside : {outside.max():.3e}")

norm = kernel.sum()
print(f"\neffective bandwidth sigma_W / df = "
      f"{np.sqrt((kernel * u**2).sum() / norm):.4f}")
