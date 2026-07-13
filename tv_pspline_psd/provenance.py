"""Reproducibility metadata for fit artifacts and study outputs."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_PACKAGES = ("tv_pspline_psd", "jax", "numpyro", "wdm-transform")


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


__all__ = ["provenance"]
