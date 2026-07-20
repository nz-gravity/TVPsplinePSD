#!/bin/bash
#SBATCH --job-name=ls2_fig3_render
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ls2_fig3_render_%j.out

# Validate and merge all seed chunks, then render the matched six-knot Figure 3.

module purge
module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/TVPsplinePSD/.venv/bin/activate

set -euo pipefail
export JAX_PLATFORMS=cpu
python studies/paper_figures/scripts/make_sim_study_figures.py \
    --merge-chunks --nt 32 64 128 256 512 1024 \
    --nf 32 --freq-knots 6 --repeats 100 --chunk-size 10

python studies/paper_figures/scripts/make_sim_study_figures.py \
    --render-only --freq-knots 6 --nf 32
