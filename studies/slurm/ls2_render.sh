#!/bin/bash
#SBATCH --job-name=ls2_render
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ls2_render_%j.out

# Make the single-realization triptych (Fig 1) and re-render Fig 2 from the
# per-duration shards. Run after the ls2_sim_study.sh array completes:
#   sbatch --dependency=afterok:<array_job_id> studies/slurm/ls2_render.sh
# Then copy studies/paper_figures/figures/sim_* back locally.

module purge
module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/TVPsplinePSD/.venv/bin/activate

set -euo pipefail

export JAX_PLATFORMS=cpu
# --nt with no values: fit nothing, make the single-realisation panel, render
# the primary 8-knot Figure 3, and render the 6/8/10-knot sensitivity figure.
python studies/paper_figures/scripts/make_sim_study_figures.py \
    --nt --freq-knots 8 --reference-draws "${REFERENCE_DRAWS:-1000}"
