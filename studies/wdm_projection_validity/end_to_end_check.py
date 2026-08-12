"""End-to-end: does the pilot's WDM->PSD pipeline recover a known steep S(f)?

Draws stationary Gaussian noise with the *real* analytic A/E/T PSD, pushes it
through the exact transform and calibration the pilot uses, and compares the
recovered PSD to the input. Any bias here is a projection/calibration artifact,
because there is no physics in the input beyond S(f) itself.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import median_filter

LDG = Path("/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")
PKG = Path("/Users/avi/Documents/projects/wdm_psd/wdm_psd")
sys.path.insert(0, str(LDG))
sys.path.insert(0, str(PKG))

from run_aet_diagonal_pilot import analytic_aet_noise_psd
from aet_diagonal import AET_CHANNELS
from tv_pspline_psd import PSplineConfig, wdm_analysis_coefficients

ORBITS = LDG / "noise2a" / "orbits.h5"
ARCHIVE = LDG / "combined_esa_xyz.h5"

NT = 32
NF = 65_536
DT = 2.0
N = NT * NF
DF = 1.0 / (2.0 * NF * DT)

with h5py.File(ARCHIVE, "r") as hdf:
    epoch = float(hdf["truth/time_tcb"][30])

# Input PSD on the full rfft grid; the pipeline's convention is one-sided,
# S_white = 2 sigma^2 dt, so E[|rfft(x)_k|^2] = N S_k / (2 dt).
rfft_frequency = np.fft.rfftfreq(N, DT)
positive = rfft_frequency > 0
psd_input = np.zeros((3, rfft_frequency.size))
psd_input[:, positive] = analytic_aet_noise_psd(
    ORBITS, np.array([epoch]), rfft_frequency[positive]
)[:, 0, :]

rng = np.random.default_rng(20260812)
config = PSplineConfig(
    n_interior_knots_time=3, n_interior_knots_freq=40,
    trim_time_bins=1, trim_low_freq_channels=1, trim_high_freq_channels=0,
    freq_knot_strategy="log", centered=True,
)

print(f"nt={NT} nf={NF} df={DF:.4e} Hz  N={N}  epoch={epoch:.1f}\n")
print(f"{'chan':>5} {'band [Hz]':>20} {'mean ratio':>12} {'median ratio':>13} "
      f"{'expected sd':>12}")

for channel, name in enumerate(AET_CHANNELS):
    spectrum = np.sqrt(N * psd_input[channel] / (2.0 * DT))
    amplitude = spectrum * (
        rng.standard_normal(spectrum.size) + 1j * rng.standard_normal(spectrum.size)
    ) / np.sqrt(2.0)
    amplitude[0] = 0.0
    amplitude[-1] = amplitude[-1].real * np.sqrt(2.0)
    series = np.fft.irfft(amplitude, n=N)

    coefficients, _, frequency = wdm_analysis_coefficients(series, DT, NT, config)
    recovered = coefficients**2 * (2.0 * DT / N)          # pilot's conversion
    truth = np.interp(frequency, rfft_frequency, psd_input[channel])

    ratio = recovered.mean(axis=0) / truth                # average over time bins
    n_time = coefficients.shape[0]
    for lo, hi in [(1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1)]:
        band = (frequency >= lo) & (frequency < hi)
        print(f"{name:>5} {f'{lo:.0e} - {hi:.0e}':>20} {ratio[band].mean():12.4f} "
              f"{np.median(ratio[band]):13.4f} {np.sqrt(2.0 / n_time):12.4f}")

    # The hypothesis predicts the damage is concentrated in response nulls, so
    # look there specifically: cells sitting far below their local continuum.
    continuum = median_filter(np.log(truth), size=501, mode="nearest")
    depth = np.log(truth) - continuum
    for threshold in (-1.0, -3.0, -5.0):
        null = depth < threshold
        if null.sum() > 5:
            print(f"{name:>5} {f'null cells <{threshold:.0f} nat':>20} "
                  f"{ratio[null].mean():12.4f} {np.median(ratio[null]):13.4f} "
                  f"{f'n={null.sum()}':>12}")
