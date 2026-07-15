#!/bin/bash
#SBATCH --job-name=fig7_sens
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig7_sens_%A_%a.out

# Figure 7 diagnostic: 4 time-knot counts x ungapped/gapped. All outputs carry
# a fig7_ktXX suffix and cannot overwrite the production artifacts.

module purge
module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/TVPsplinePSD/.venv/bin/activate

set -euo pipefail
mkdir -p logs
export JAX_PLATFORMS=cpu

TIME_KNOTS=(8 12 16 20)
KNOT_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
GAP_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
KNOTS=${TIME_KNOTS[$KNOT_INDEX]}
GAP_ARGS=()
if (( GAP_INDEX == 1 )); then
    GAP_ARGS+=(--gaps)
fi

python studies/ollie_tdi/fit_aet_fullband.py \
    --channel A --data full \
    --time-knots "$KNOTS" \
    --tag-suffix "fig7_kt$(printf '%02d' "$KNOTS")" \
    --n-warmup 300 --n-samples 300 --num-chains 2 \
    "${GAP_ARGS[@]}"
