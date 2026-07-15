"""Reproducibility metadata for fit artifacts and study outputs."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_PACKAGES = ("tv_pspline_psd", "jax", "numpyro", "wdm-transform")


def _json_safe(value: Any) -> Any:
    """Return nested metadata using only JSON-native values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "Binning metadata must contain only JSON-serializable values; "
        f"got {type(value).__name__}."
    )


def binning_provenance(
    *,
    n_time: int,
    n_freq: int,
    time_bin: int = 1,
    freq_bin: int = 1,
    freq_bin_starts: np.ndarray | None = None,
    selector_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe a separable likelihood partition completely and reproducibly.

    Explicit starts and widths are retained even for regular bins. This makes
    ragged edge bins unambiguous and records the realised adaptive partition,
    independently of whether the pilot used to choose it remains available.
    ``selector_metadata`` can additionally retain the pilot algorithm and its
    settings.
    """
    time_starts = np.arange(0, int(n_time), int(time_bin), dtype=int)
    if freq_bin_starts is None:
        frequency_mode = "identity" if freq_bin == 1 else "uniform"
        frequency_starts = np.arange(0, int(n_freq), int(freq_bin), dtype=int)
    else:
        frequency_mode = "variable"
        frequency_starts = np.asarray(freq_bin_starts, dtype=int)

    time_widths = np.diff(np.r_[time_starts, int(n_time)])
    frequency_widths = np.diff(np.r_[frequency_starts, int(n_freq)])
    recipe: dict[str, Any] = {
        "schema_version": 1,
        "input_shape": [int(n_time), int(n_freq)],
        "output_shape": [int(time_starts.size), int(frequency_starts.size)],
        "time": {
            "mode": "identity" if time_bin == 1 else "uniform",
            "bin_size": int(time_bin),
            "starts": time_starts.tolist(),
            "widths": time_widths.tolist(),
        },
        "frequency": {
            "mode": frequency_mode,
            "bin_size": int(freq_bin),
            "starts": frequency_starts.tolist(),
            "widths": frequency_widths.tolist(),
        },
    }
    if selector_metadata is not None:
        recipe["selector"] = _json_safe(selector_metadata)
    # Fail here, close to the caller, rather than later while saving a run.
    json.dumps(recipe)
    return recipe


def provenance(
    *,
    seed: int | None = None,
    dt: float | None = None,
    nt: int | None = None,
    trims: dict[str, int] | None = None,
    config: object | None = None,
    calibration: dict[str, Any] | None = None,
    source_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable package, git, fit, and source metadata."""
    packages: dict[str, str | None] = {}
    for package in _PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None

    repo = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        git_commit: str | None = completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        git_commit = None

    out: dict[str, Any] = {"packages": packages, "git_commit": git_commit}
    optional = {
        "seed": seed,
        "dt": dt,
        "nt": nt,
        "trims": trims,
        "config": asdict(config) if config is not None and is_dataclass(config) else config,
        "calibration": calibration,
        "source_data": source_data,
    }
    out.update({key: value for key, value in optional.items() if value is not None})
    return out


__all__ = ["binning_provenance", "provenance"]
