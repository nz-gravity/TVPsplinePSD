"""Estimate local WDM coefficient dependence for the LS2 ensemble.

The manuscript LS2 study uses 100 independent records at each observation
length.  This companion diagnostic regenerates those records, standardizes
each WDM cell by its ensemble marginal standard deviation, and pools the
correlation at each local offset.  WDM parity is retained because correlations
with opposite signs can otherwise cancel when neighbouring anchors are pooled.

Run from the repository root with::

    .venv/bin/python studies/ls2_local_dependence.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tv_pspline_psd import PSplineConfig, wdm_analysis_coefficients
from tv_pspline_psd.datasets import simulate_ls2


DT = 0.1
NF = 32
NT_VALUES = (32, 64, 128, 256, 512, 1024)
MAX_TIME_LAG = 1
MAX_FREQUENCY_LAG = 8


def _shifted_views(
    array: np.ndarray, delta_time: int, delta_frequency: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned ``(record, time, frequency)`` views at one offset."""
    time_0 = slice(
        max(0, -delta_time), array.shape[1] - max(0, delta_time)
    )
    time_1 = slice(
        max(0, delta_time), array.shape[1] - max(0, -delta_time)
    )
    frequency_0 = slice(
        max(0, -delta_frequency),
        array.shape[2] - max(0, delta_frequency),
    )
    frequency_1 = slice(
        max(0, delta_frequency),
        array.shape[2] - max(0, -delta_frequency),
    )
    return (
        array[:, time_0, frequency_0],
        array[:, time_1, frequency_1],
    )


def local_dependence(nt: int, repeats: int, seed_start: int) -> dict[str, object]:
    """Return the largest parity-stratified pooled local correlation."""
    config = PSplineConfig()
    coefficients = []
    for repeat in range(repeats):
        data = simulate_ls2(nt * NF, rng=np.random.default_rng(seed_start + repeat))
        wdm, _, _ = wdm_analysis_coefficients(data, DT, nt, config)
        coefficients.append(wdm)
    values = np.stack(coefficients)

    # The diagnostic concerns dependence after the marginal power has been
    # accounted for.  Estimate that finite-resolution marginal separately at
    # every WDM cell from the same 100-record ensemble used by the LS2 study.
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1)
    z = (values - mean[None, :, :]) / scale[None, :, :]
    parity = np.indices(values.shape[1:]).sum(axis=0) % 2

    correlations: list[dict[str, float | int]] = []
    for delta_time in range(-MAX_TIME_LAG, MAX_TIME_LAG + 1):
        for delta_frequency in range(
            -MAX_FREQUENCY_LAG, MAX_FREQUENCY_LAG + 1
        ):
            if delta_time == 0 and delta_frequency == 0:
                continue
            first, second = _shifted_views(z, delta_time, delta_frequency)
            parity_first, _ = _shifted_views(
                parity[None, :, :], delta_time, delta_frequency
            )
            for anchor_parity in (0, 1):
                selected = np.broadcast_to(
                    parity_first == anchor_parity, first.shape
                )
                correlation = float(
                    np.corrcoef(first[selected], second[selected])[0, 1]
                )
                correlations.append(
                    {
                        "delta_time": delta_time,
                        "delta_frequency": delta_frequency,
                        "anchor_parity": anchor_parity,
                        "correlation": correlation,
                        "n_pairs": int(selected.sum()),
                    }
                )

    largest = max(correlations, key=lambda item: abs(float(item["correlation"])))
    return {
        "nt": nt,
        "n_total": nt * NF,
        "repeats": repeats,
        "wdm_shape": list(values.shape[1:]),
        "max_abs_local_correlation": abs(float(largest["correlation"])),
        "max_correlation_entry": largest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=6000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "overleaf"
        / "data"
        / "ls2_local_dependence.json",
    )
    args = parser.parse_args()
    results = [
        local_dependence(nt, args.repeats, args.seed_start) for nt in NT_VALUES
    ]
    payload = {
        "definition": (
            "maximum absolute parity-stratified pooled Pearson correlation "
            "over |delta_time|<=1, |delta_frequency|<=8, excluding zero lag"
        ),
        "marginal_standardization": (
            "cellwise mean and standard deviation over the independent LS2 records"
        ),
        "dt": DT,
        "nf": NF,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
