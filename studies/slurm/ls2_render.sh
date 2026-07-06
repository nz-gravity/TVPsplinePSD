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
# Then copy notes/figures/sim_*.png and sim_metrics_nt*.npz back locally.

module purge
module load gcc/13.3.0 python/3.12.3

set -euo pipefail
source "${VENV:-$PWD/.venv-ozstar}/bin/activate"

export JAX_PLATFORMS=cpu
# --nt with no values: fit nothing, make Fig 1, render Fig 2 from shards.
python notes/scripts/make_sim_study_figures.py --nt
