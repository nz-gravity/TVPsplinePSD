#!/bin/bash
#SBATCH --job-name=ls2_coarse
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/ls2_coarse_%j.out

# Submit from the repository root:
#   sbatch studies/slurm/ls2_coarse_graining.sh
# This launches ten independent shards of ten paired realizations each. After
# all tasks finish, run the merge command documented below.

set -euo pipefail
mkdir -p logs
module load gcc/13.3.0 python/3.12.3
export JAX_PLATFORMS=cpu

TOTAL_REPEATS="${REPEATS:-100}"
REPEATS_PER_TASK="${REPEATS_PER_TASK:-10}"
START=$((SLURM_ARRAY_TASK_ID * REPEATS_PER_TASK))
STOP=$((START + REPEATS_PER_TASK))
if (( START >= TOTAL_REPEATS )); then
    exit 0
fi
if (( STOP > TOTAL_REPEATS )); then
    STOP=$TOTAL_REPEATS
fi

OUTDIR="studies/results/ls2/coarse_graining_16384/shards/task_${SLURM_ARRAY_TASK_ID}"
.venv/bin/python studies/ls2_coarse_graining_study.py \
    --repeats "$TOTAL_REPEATS" \
    --repeat-start "$START" --repeat-stop "$STOP" \
    --output-dir "$OUTDIR"

# Merge after every array task has succeeded:
# .venv/bin/python studies/ls2_coarse_graining_study.py --merge-shards
