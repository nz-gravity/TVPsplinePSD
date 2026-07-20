# LISA coarse-graining study

This is a paired accuracy/speed study on the 30-day A-channel WDM
coefficients. If the optional WDM cache is absent, the script creates it from
`datasets/ollie_data/simulated_noise_30_days_L1_ext.h5` (using `tdis/A2`, or
constructing A from X2 and Z2 as a fallback) and reuses it on later submissions.
It runs the same spline model with five likelihood grids:

1. exact fine grid;
2. uniform frequency pooling (`F=12`);
3. adaptive frequency pooling with log-range tolerance 0.2;
4. adaptive frequency pooling with tolerance 0.3; and
5. tolerance 0.3 combined with nominal two-row time pooling.

All five fits are repeated for intact data and for a deterministic LISA-like
gap schedule. In the gapped case, corrupted WDM rows are removed exactly as in
the existing gap analysis. The adaptive pilot is smoothed independently in
each contiguous time run, and time pooling restarts on each side of every gap.
No pooled block can therefore bridge missing data.

The study uses two sequential chains. Ten fits run inside one small Slurm job,
instead of competing as an array. Every fit has its own deterministic seed and
is saved immediately to `studies/results/ollie_tdi/coarse_graining_lisa/`.
Resubmitting the job skips completed `.npz` files.

Submit from the repository root:

```bash
sbatch studies/slurm/lisa_coarse_graining.sh
```

For a cheap end-to-end smoke test before submission:

```bash
.venv/bin/python studies/ollie_tdi/lisa_coarse_graining_study.py \
  --warmup 10 --samples 10 --chains 1 \
  --output-dir /tmp/lisa_coarse_smoke
```

The production outputs are `summary.json`, `summary.csv`, one resumable fit per
condition/setting, and `pooling_cells.{png,pdf}`. The summary compares every
coarse posterior with the exact posterior for the same retained data and
reports cell reduction, runtime, speedup, posterior-standardized surface
displacement, null-region displacement, whitening, divergences, and seed.

## Production result (2026-07-16)

The completed two-chain, 300/300 OzStar run is summarized under the LISA
coarse-graining section of `studies/slurm/README.md`, with complete
machine-readable metrics in `summary.json` and `summary.csv` under the output
directory. Adaptive
frequency pooling at tolerance 0.2 reduced the likelihood grid by 7.68x
(intact) and 6.37x (gapped), producing 5.78x and 4.68x NUTS speedups while
moving the posterior-mean log surface by 0.191 and 0.162 pooled posterior
standard deviations RMS. Uniform frequency x12 was faster but displaced the
null corridors substantially more. Adaptive tolerance 0.3 plus time x2 gave
10.67x and 7.40x speedups at normalized surface RMS 0.351 and 0.293. All ten
fits had zero divergences.

This is a paired accuracy/runtime study on one 30-day realization, including a
fixed gap stress test. It validates the computational approximation and gap
partitioning, but should not be described as an ensemble coverage result.
