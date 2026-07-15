# OzStar study workflows

## Matched-knot LS2 comparison (Figure 3)

From the repository root:

```bash
job_id=$(sbatch --parsable studies/slurm/ls2_sim_study.sh)
sbatch --dependency=afterok:${job_id} studies/slurm/ls2_render.sh
```

The array covers seven durations (`n = 576, ..., 36864`) and common interior
frequency-knot counts 6, 8, and 10. Both WDM and moving-periodogram fits use
the same count and physical knot locations. Production defaults are 100 data
realisations and 1000 Monte Carlo draws for each finite-resolution reference.
Override them at submission time when needed:

```bash
REPEATS=20 REFERENCE_DRAWS=200 sbatch studies/slurm/ls2_sim_study.sh
```

The primary paper figure uses 8 frequency knots. The render job also writes
`sim_knot_sensitivity.png` from every available 6/8/10-knot shard.

## Figure 7 oscillation diagnostics

```bash
job_id=$(sbatch --parsable studies/slurm/fig7_sensitivity.sh)
sbatch --dependency=afterok:${job_id} studies/slurm/fig7_render.sh
```

This runs ungapped and gapped fits with 8, 12, 16, and 20 interior time knots.
Each fit saves centroid, quadratic-minimum, and dense spline-minimum null tracks,
including per-chain summaries. The render job compares extractor and chain
stability, whitening, the original all-arm proxy, and an A-channel-specific arm
proxy. Diagnostic tags prevent the production Figure 7 fits from being
overwritten.
