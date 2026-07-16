#!/bin/bash
#SBATCH --job-name=ls2_fig3_render
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ls2_fig3_render_%j.out

# Render the matched six-knot production Figure 3 after ls2_fig3_production.

module purge
module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/TVPsplinePSD/.venv/bin/activate

set -euo pipefail
export JAX_PLATFORMS=cpu
python studies/paper_figures/scripts/make_sim_study_figures.py \
    --render-only --freq-knots 6 --nf 32
