from __future__ import annotations

import numpy as np
import pytest

from studies.paper_figures.scripts import make_sim_study_figures as study


def _fake_metrics(start: int, size: int) -> dict[str, list[float]]:
    values = [float(i) for i in range(start, start + size)]
    return {key: values.copy() for key in study.METRIC_KEYS}


def test_chunk_checkpoints_resume_and_merge(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(study, "FIG_DIR", tmp_path)
    monkeypatch.setattr(study, "NF", 32)
    n_total = 32 * 32

    for start in (0, 2):
        path = study._chunk_path(n_total, 6, start, 2)
        study._atomic_savez(
            path,
            **study._checkpoint_arrays(
                _fake_metrics(start, 2),
                n_total=n_total,
                freq_knots=6,
                repeat_start=start,
                repeats_target=2,
            ),
        )

    resumed = study._load_checkpoint(
        study._chunk_path(n_total, 6, 2, 2),
        n_total=n_total,
        freq_knots=6,
        repeat_start=2,
        repeats_target=2,
    )
    assert resumed["wm"] == [2.0, 3.0]

    study._merge_chunks([32], 6, total_repeats=4, chunk_size=2)
    with np.load(study._shard_path(n_total, 6)) as merged:
        np.testing.assert_array_equal(merged["repeat_ids"], np.arange(4))
        np.testing.assert_array_equal(merged["wm_samples"], np.arange(4.0))


def test_reference_cache_rejects_incompatible_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(study, "NF", 32)
    path = study._reference_cache_path(tmp_path, 32, 6, 2)
    metadata = study._reference_metadata(32, 6, 2)
    metadata["tang_m"] = study.TANG_M + 1
    study._atomic_savez(
        path,
        **metadata,
        cal_wdm=np.ones(2),
        ref_wdm=np.ones((2, 2)),
        ref_tang=np.ones(2),
    )

    with pytest.raises(ValueError, match="tang_m"):
        study._read_reference_cache(path, 32, 6, 2)
