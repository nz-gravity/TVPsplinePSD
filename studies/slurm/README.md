# OzStar study workflows

## LISA coarse-graining benchmark

`lisa_coarse_graining.sh` runs the ten paired exact/coarse fits on intact and
gapped 30-day LISA A-channel data. Its canonical output directory is
`studies/results/ollie_tdi/coarse_graining_lisa/`. The completed production
fits ran on 2026-07-16 and were copied back from OzStar on 2026-07-20.

```bash
sbatch studies/slurm/lisa_coarse_graining.sh
```

### Outputs and protocol

The output directory contains:

- `summary.json`: complete configuration, gap schedule, grid shapes and metrics;
- `summary.csv`: compact fit-level metric table;
- `pooling_cells.{png,pdf}`: gap-aware adaptive pooling around the 0.06 Hz null;
- `ozstar_export_2026-07-16.tar.gz`: the untouched SCP transfer bundle; and
- one resumable `.npz` fit per condition and setting on OzStar.

The study uses 30 days of simulated LISA TDI A-channel noise over
0.000123--0.1 Hz. The fine grid has 120 x 4,046 cells intact and 101 x 4,046
after removing 19 gap-affected rows. Every fit uses the same centered P-spline
surface (16 time and 94 frequency interior knots) and two sequential NUTS
chains with 300 warm-up and 300 retained draws. It compares the exact grid,
uniform frequency x12, adaptive-frequency tolerances 0.2 and 0.3, and adaptive
0.3 plus time x2, for both intact and deterministically gapped data.

### Production results

| Condition | Setting | Cell reduction | Speedup | Normalized surface RMS | Cells >1 pooled SD | Null RMS | Mean z^2 | Null mean z^2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Intact | exact | 1.00x | 1.00x | 0.000 | 0.00% | 0.000 | 0.9975 | 0.9795 |
| Intact | uniform frequency x12 | 11.97x | 7.72x | 0.508 | 2.83% | 0.103 | 0.9914 | 0.9351 |
| Intact | adaptive frequency 0.2 | 7.68x | 5.78x | 0.191 | 0.83% | 0.030 | 0.9962 | 0.9712 |
| Intact | adaptive 0.3 + time x2 | 23.39x | 10.67x | 0.351 | 1.70% | 0.058 | 0.9955 | 0.9672 |
| Gapped | exact | 1.00x | 1.00x | 0.000 | 0.00% | 0.000 | 0.9972 | 0.9787 |
| Gapped | uniform frequency x12 | 11.97x | 6.60x | 0.503 | 2.34% | 0.106 | 0.9911 | 0.9336 |
| Gapped | adaptive frequency 0.2 | 6.37x | 4.68x | 0.162 | 0.45% | 0.025 | 0.9962 | 0.9726 |
| Gapped | adaptive 0.3 + time x2 | 18.71x | 7.40x | 0.293 | 1.40% | 0.050 | 0.9954 | 0.9698 |

All ten fits had zero divergences and a mean of 31 leapfrog steps. Adaptive
tolerance 0.2 is the conservative choice: it preserves the null region much
better than uniform x12 while giving a 4.7--5.8x speedup. Adaptive 0.3 plus
time x2 is the higher-throughput choice, giving 7.4--10.7x speedup with a
larger surface displacement.

This is a paired accuracy/runtime study on one fixed 30-day realization. It
supports the coarse-likelihood and gap-partitioning claims, but it is not an
ensemble coverage result. Setting-specific sampler seeds mean the measured
posterior displacement also contains finite-chain Monte Carlo variation. See
`studies/ollie_tdi/LISA_COARSE_GRAINING_STUDY.md` for implementation details.

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

The submission wrapper has two modes. First smoke-test the most expensive
chunk (`N=2^15`, seeds 90--99) in the OzStar environment:

```bash
studies/slurm/submit_jobs.sh smoke
```

The wrapper prints the `sacct` command for checking elapsed time and peak
resident memory. If the smoke job completes comfortably within 4 GB, submit
the complete workflow with:

```bash
studies/slurm/submit_jobs.sh full
```

Running `full` after `smoke` is safe: the validated reference cache and
completed seed chunk are reused.

The fit array covers exactly `n = 2^10, ..., 2^15`, holding the WDM frequency axis
at 32 channels and varying its time axis from 32 to 1024 bins.  Six common
interior frequency knots were selected by the pilot; both likelihoods use the
same physical locations.  Production defaults are 100 data realisations and
200 Monte Carlo draws for the representation-specific diagnostic. The common
latent-PSD MSE is the paper metric.

The reference array creates one validated deterministic cache per data size.
The fit comprises 60 independently schedulable one-core jobs: ten seed chunks
for each of six sizes. Jobs are submitted in resource tiers so inexpensive
sizes can backfill without inheriting the largest fit's request:

| Array tasks | Data sizes | Memory | Time | Max concurrent |
|---|---|---:|---:|---:|
| 0--29 | `N=2^10..2^12` | 3 GB | 1 h | 6 |
| 30--49 | `N=2^13..2^14` | 4 GB | 1.5 h | 4 |
| 50--59 | `N=2^15` | 4 GB | 2.5 h | 2 |

Each chunk atomically checkpoints after every seed and resumes incomplete
work. The final job depends on all three tiers and refuses to render unless
all seed IDs 0--99 occur exactly once for every size.

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
