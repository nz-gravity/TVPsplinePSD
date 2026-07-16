#!/bin/bash
#SBATCH --job-name=ls2_fig3
#SBATCH --array=0-5%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ls2_fig3_%A_%a.out

# Final Figure 3 study: N = 2^10, ..., 2^15.  We hold nf=32 fixed and vary
# nt, so every WDM transform is valid (both axes even) and the total record
# length is exactly a power of two.  Skipping N=512 avoids the edge-case
# nt=16 time grid. Both likelihoods use the selected six interior frequency
# knots at exactly the same physical locations.
#
# This is seven one-core jobs, capped at three concurrent jobs.  Each task
# writes its own shard and is safe to restart after a failed array element.

module purge
module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/TVPsplinePSD/.venv/bin/activate

set -euo pipefail
mkdir -p logs
export JAX_PLATFORMS=cpu

# N = nt * 32 = 2^10, ..., 2^15.
NT_VALUES=(32 64 128 256 512 1024)
NT=${NT_VALUES[$SLURM_ARRAY_TASK_ID]}

python studies/paper_figures/scripts/make_sim_study_figures.py \
    --nt "$NT" --nf 32 --freq-knots 6 \
    --repeats "${REPEATS:-100}" \
    --reference-draws "${REFERENCE_DRAWS:-200}" \
    --skip-fig1 --skip-render
