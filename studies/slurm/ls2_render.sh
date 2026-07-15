#!/bin/bash
#SBATCH --job-name=ls2_render
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ls2_render_%j.out

# Re-render Figure 3 from the per-duration shards. Run after either LS2 array
# completes; this deliberately does not launch an additional largest-duration
# MCMC fit merely to refresh the illustrative triptych.
#   sbatch --dependency=afterok:<array_job_id> studies/slurm/ls2_render.sh
# Then copy studies/paper_figures/figures/sim_* back locally.

module purge
module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/TVPsplinePSD/.venv/bin/activate

set -euo pipefail

export JAX_PLATFORMS=cpu
# Primary 8-knot Figure 3. A complete 6/8/10 sensitivity plot is emitted only
# after all three knot counts share the same duration grid.
python studies/paper_figures/scripts/make_sim_study_figures.py \
    --render-only --freq-knots 8
