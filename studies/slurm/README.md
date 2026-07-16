# OzStar study workflows

## Matched-knot LS2 comparison (Figure 3)

Start with the inexpensive pilot. It uses one core per task, 9 jobs, 20--30
realisations, and only 200 Monte Carlo reference draws:

```bash
job_id=$(sbatch --parsable studies/slurm/ls2_matched_pilot.sh)
sbatch --dependency=afterok:${job_id} studies/slurm/ls2_render.sh
```

This is sufficient to decide whether the common 8-knot configuration and the
endpoint 6/10-knot sensitivity are materially different. Use the larger sweep
below only if that pilot indicates that the paper result needs tighter bands.

### Final production sweep

From the repository root:

```bash
job_id=$(sbatch --parsable studies/slurm/ls2_fig3_production.sh)
sbatch --dependency=afterok:${job_id} studies/slurm/ls2_fig3_render.sh
```

The array covers exactly `n = 2^10, ..., 2^15`, holding the WDM frequency axis
at 32 channels and varying its time axis from 32 to 1024 bins.  Six common
interior frequency knots were selected by the pilot; both likelihoods use the
same physical locations.  Production defaults are 100 data realisations and
200 Monte Carlo draws for the representation-specific diagnostic.  The common
latent-PSD MSE is the paper metric. Override the repeat count when needed:

```bash
REPEATS=20 sbatch studies/slurm/ls2_fig3_production.sh
```

The primary paper figure uses six matched frequency knots. Existing pilot
shards remain separate because their `n` grid differs from this production run.

## Figure 7 oscillation diagnostics

Start with the three one-core ungapped fits:

```bash
sbatch studies/slurm/fig7_pilot.sh
```

They test the time-knot explanation directly at 8, 12, and 16 knots. The
existing 16-knot gapped fit already covers the gap-robustness question. Run the
full eight-fit extractor/chain/gap study below only if the pilot shows a
substantive knot dependence.

### Full diagnostic sweep

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
