#!/bin/bash
#SBATCH --job-name=fig7_render
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fig7_render_%j.out

module purge
module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/TVPsplinePSD/.venv/bin/activate

set -euo pipefail
export JAX_PLATFORMS=cpu

python studies/ollie_tdi/fig7_sensitivity.py
