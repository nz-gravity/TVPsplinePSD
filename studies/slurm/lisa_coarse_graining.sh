#!/bin/bash
#SBATCH --job-name=lisa_coarse
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lisa_coarse_%j.out

# One scheduler-friendly serial job: ten paired fits (five intact, five gapped).
# Each fit is checkpointed separately, so resubmitting resumes automatically.
# Submit from the repository root:
#   sbatch studies/slurm/lisa_coarse_graining.sh

set -euo pipefail
mkdir -p logs
module load gcc/13.3.0 python/3.12.3

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

.venv/bin/python studies/ollie_tdi/lisa_coarse_graining_study.py \
    --warmup "${WARMUP:-300}" \
    --samples "${SAMPLES:-300}" \
    --chains "${CHAINS:-2}"
