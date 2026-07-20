#!/bin/bash
#SBATCH --job-name=ls2_fig3
#SBATCH --array=0-59%6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:30:00
#SBATCH --output=logs/ls2_fig3_%A_%a.out

# Final Figure 3 study: N = 2^10, ..., 2^15.  We hold nf=32 fixed and vary
# nt, so every WDM transform is valid (both axes even) and the total record
# length is exactly a power of two. Skipping total length N=512 avoids the
# edge-case nt=16 time grid. Both likelihoods use the selected six frequency
# knots at exactly the same physical locations.
#
# This is 60 short one-core jobs (6 sizes x 10 seed chunks), capped at six
# concurrent jobs. Each task atomically checkpoints its own ten-seed shard and
# resumes it safely after a failed array element.

module purge
module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/TVPsplinePSD/.venv/bin/activate

set -euo pipefail
mkdir -p logs
export JAX_PLATFORMS=cpu

# N = nt * 32 = 2^10, ..., 2^15. Ten chunks cover seeds 0,...,99.
NT_VALUES=(32 64 128 256 512 1024)
N_CHUNKS=10
SIZE_INDEX=$((SLURM_ARRAY_TASK_ID / N_CHUNKS))
CHUNK_INDEX=$((SLURM_ARRAY_TASK_ID % N_CHUNKS))
NT=${NT_VALUES[$SIZE_INDEX]}
REPEAT_START=$((CHUNK_INDEX * 10))

python studies/paper_figures/scripts/make_sim_study_figures.py \
    --nt "$NT" --nf 32 --freq-knots 6 \
    --repeat-start "$REPEAT_START" --repeats 10 \
    --reference-draws "${REFERENCE_DRAWS:-200}" \
    --require-reference-cache --skip-fig1 --skip-render
