"""Fast smoke test for the vendored LISA galactic-binary study.

Overrides the heavy production constants for speed, runs one seed through both
the frequency-domain and WDM-domain samplers, and prints a per-parameter table
of the freq-vs-WDM posterior means and their |freq - wdm| difference in sigma.

SUCCESS: runs without error, low divergences, freq-vs-WDM means agree to ~1-2 sigma.

    uv run python studies/lisa_gb/smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lisa_gb_study as m  # noqa: E402

# ── Lightweight overrides (speed) ──────────────────────────────────────────────
m.T_OBS = 30 * 24 * 3600
m.NBLOCKS = 256
m.N_WARMUP = 300
m.N_DRAWS = 300
m.NUM_CHAINS = 1

import numpyro  # noqa: E402

numpyro.set_host_device_count(1)


def main() -> None:
    grid = m._make_grid()
    jgb = m.make_jgb(grid)
    print(
        f"[smoke] N={grid['n_total']} T_obs={grid['t_obs'] / 86400:.1f}d "
        f"dt={grid['dt']:.2f}s channels={m.CHANNELS}"
    )

    t0 = time.perf_counter()
    res = m.run_one_seed(0, grid=grid, jgb=jgb)
    runtime = time.perf_counter() - t0

    freq_rows = {r["label"]: r for r in res["freq"]}
    wdm_rows = {r["label"]: r for r in res["wdm"]}
    freq_div = res["diagnostics"]["freq"]["divergences"]
    wdm_div = res["diagnostics"]["wdm"]["divergences"]

    print("\n===== FREQ vs WDM agreement (seed 0) =====")
    print(f"{'parameter':<20} {'truth':>12} {'freq_mean':>12} {'wdm_mean':>12} "
          f"{'|f-w|/sigma':>12}")
    max_delta = 0.0
    for label in m.POSTERIOR_LABELS:
        fr, wd = freq_rows[label], wdm_rows[label]
        s = 0.5 * (fr["std"] + wd["std"])
        delta = abs(fr["mean"] - wd["mean"]) / s if s > 0 else 0.0
        max_delta = max(max_delta, delta)
        print(f"{label:<20} {fr['truth']:>12.5f} {fr['mean']:>12.5f} "
              f"{wd['mean']:>12.5f} {delta:>12.3f}")

    print(f"\ndivergences: freq={freq_div} wdm={wdm_div}")
    print(f"max |freq-wdm| disagreement: {max_delta:.3f} sigma")
    print(f"runtime: {runtime:.1f}s")

    ok = max_delta <= 2.0 and freq_div == 0 and wdm_div == 0
    print("RESULT:", "PASS" if ok else "CHECK (see numbers above)")


if __name__ == "__main__":
    main()
