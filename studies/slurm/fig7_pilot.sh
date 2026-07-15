#!/bin/bash
#SBATCH --job-name=fig7_pilot
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig7_pilot_%A_%a.out

# First-pass Figure 7 knot sensitivity. These are ungapped fits only: the
# existing gapped result already establishes gap robustness at 16 knots. Run a
# gapped confirmation only for the preferred knot count after inspecting these.

module purge
module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/TVPsplinePSD/.venv/bin/activate

set -euo pipefail
mkdir -p logs
export JAX_PLATFORMS=cpu

TIME_KNOTS=(8 12 16)
KNOTS=${TIME_KNOTS[$SLURM_ARRAY_TASK_ID]}

python studies/ollie_tdi/fit_aet_fullband.py \
    --channel A --data full --time-knots "$KNOTS" \
    --tag-suffix "fig7_pilot_kt$(printf '%02d' "$KNOTS")" \
    --n-warmup 200 --n-samples 200 --num-chains 2
