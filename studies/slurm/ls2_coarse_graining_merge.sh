#!/bin/bash
#SBATCH --job-name=ls2_coarse_merge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/ls2_coarse_merge_%j.out

# Submit after the array job succeeds:
#   sbatch --dependency=afterok:<array-job-id> studies/slurm/ls2_coarse_graining_merge.sh

set -euo pipefail
mkdir -p logs
module load gcc/13.3.0 python/3.12.3

.venv/bin/python studies/ls2_coarse_graining_study.py --merge-shards
