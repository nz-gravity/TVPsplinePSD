"""Predicted WDM projection bias S_eff(m)/S(f_m) for the real analytic A/E/T PSD.

S_eff(m) = int K(f - f_m) S(f) df / int K, with K the measured WDM kernel
(compact support |f - f_m| <= 2/3 df). Pure prediction: no data, nothing fitted.
"""
import sys
from pathlib import Path

import h5py
import numpy as np

LDG = Path("/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")
sys.path.insert(0, str(LDG))

from run_aet_diagonal_pilot import analytic_aet_noise_psd, wdm_valid_length
from tv_pspline_psd.lisa_aet import AET_CHANNELS

ARCHIVE = LDG / "combined_esa_xyz.h5"
ORBITS = LDG / "noise2a" / "orbits.h5"

kernel_data = np.load(Path(__file__).parent / "wdm_kernel.npz")
u_kernel, k_kernel = kernel_data["u"], kernel_data["kernel"]

with h5py.File(ARCHIVE, "r") as hdf:
    dt = float(hdf.attrs["dt_seconds"])
    n_archive = int(hdf.attrs["n_samples"])
    truth_time = hdf["truth/time_tcb"][:]

# Sub-cell quadrature nodes, weighted by the measured kernel.
U = np.linspace(-2.0 / 3.0, 2.0 / 3.0, 65)
W = np.interp(U, u_kernel, k_kernel)
W /= W.sum()

times = truth_time[::15][:5]
CENTRES = np.geomspace(1.0e-4, 2.0e-2, 160)

for nt in (32, 64, 128):
    n_total = wdm_valid_length(n_archive, nt)
    nf = n_total // nt
    df = 1.0 / (2.0 * nf * dt)
    # Snap the probe frequencies onto exact WDM cell centres for this grid.
    centres = np.unique(np.round(CENTRES / df).astype(np.int64)) * df

    grid = (centres[:, None] + U[None, :] * df).ravel()
    psd = analytic_aet_noise_psd(ORBITS, times, np.concatenate([centres, grid]))
    psd_centre = psd[:, :, : centres.size]
    psd_grid = psd[:, :, centres.size :].reshape(3, times.size, centres.size, U.size)

    ratio = (psd_grid * W).sum(axis=-1) / psd_centre

    print(f"\n=== nt={nt}  nf={nf}  df={df:.4e} Hz  "
          f"(kernel half-width {2/3*df:.3e} Hz) ===")
    print(f"{'chan':>5} {'max|R-1|':>12} {'at f [Hz]':>12} {'median|R-1|':>13}")
    for c, name in enumerate(AET_CHANNELS):
        excess = np.abs(ratio[c] - 1.0)
        flat = excess.max(axis=0)
        i = int(np.argmax(flat))
        print(f"{name:>5} {flat[i]:12.3e} {centres[i]:12.4e} "
              f"{np.median(excess):13.3e}")
