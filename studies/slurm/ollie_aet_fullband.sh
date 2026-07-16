#!/bin/bash
#SBATCH --job-name=ollie_aet
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ollie_aet_%A_%a.out

# LISA-band (1e-4..0.1 Hz) TV-PSD fit of the AET channels built from Ollie's
# 30-day unequal-arm XYZ TDI, one channel per array task.
# Uses the centered parameterization (script default): the non-centered form
# froze phi on this grid (r_hat 2.9-9.0); centered mixes at r_hat ~ 1.00.
# Submit from the repository root:  sbatch studies/slurm/ollie_aet_fullband.sh

module purge
module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/TVPsplinePSD/.venv/bin/activate

set -euo pipefail
mkdir -p logs

# Tasks: A, E, and the A-channel gap-robustness demo.
TASK_ARGS=("--channel A" "--channel E" "--channel A --gaps")

export JAX_PLATFORMS=cpu
# shellcheck disable=SC2086  # word-splitting of the per-task args is intended
python studies/ollie_tdi/fit_aet_fullband.py \
    --data full ${TASK_ARGS[$SLURM_ARRAY_TASK_ID]} \
    --n-warmup 500 --n-samples 500 --num-chains 2
