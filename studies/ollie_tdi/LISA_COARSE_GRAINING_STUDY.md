# LISA coarse-graining study

This is a paired accuracy/speed study on the cached 30-day A-channel WDM
coefficients. It runs the same spline model with five likelihood grids:

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
