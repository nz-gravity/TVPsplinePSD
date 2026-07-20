# Studies

This directory contains the reproducible study workflows supporting the paper.
Generated data, sampler outputs, plots, and logs belong under `studies/results/`
and are not source artifacts unless deliberately retained as a small summary.

## Active results

### LS2 known-truth simulations

`paper_figures/scripts/make_sim_study_figures.py` is the primary simulation
driver.  Its Slurm workflow is documented in `slurm/README.md` and submitted
through `slurm/submit_jobs.sh`.  It establishes recovery of the known surface,
Bayesian interval calibration, the data-size trend, and the effect of exact
coarse-graining.

`ls2_coarse_graining_study.py` is the paired validation of time- and
frequency-likelihood pooling.  It supports checkpointed Slurm-array execution
and merging of completed shards.

### LISA noise with drifting unequal arms

`ollie_tdi/` contains the active Mojito/Ollie noise-only workflow.  The core
entry points are `check_XYZ.py`, `null_zoom.py`, `fit_mojito_segment.py`,
`mojito_experiments.py`, and `mojito_validation.py`; the latter two provide the
posterior-whitening and cross-validation checks.  `fit_aet_fullband.py`,
`gap_compare.py`, `gap_ensemble.py`, and `verify_aet_fit.py` support the
unequal-arm null-drift and gap-robustness claims.  The LISA adaptive
coarse-graining study is `ollie_tdi/lisa_coarse_graining_study.py`.

Use `ollie_tdi/LISA_COARSE_GRAINING_STUDY.md` and `slurm/README.md` for the
current execution recipes.

## Supporting and future work

`lisa_gb/` and `paper_figures/scripts/` contain supporting comparison and
figure-generation workflows.  `tv_pspline_psd/joint.py` is retained for future
joint signal-plus-noise inference; the target use case is a broadband EMRI or
MBH-like track, not a nearly monochromatic Galactic binary.

The VI warm-start path and its benchmark were retired: it added runtime without
improving the NUTS fits, and is intentionally absent from the active workflow.
