#!/bin/bash
#SBATCH --job-name=ls2_fig3_refs
#SBATCH --array=0-5%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ls2_fig3_refs_%A_%a.out

# Prepare one deterministic calibration/reference cache per Figure 3 size.
# Fit chunks require these files and never recompute them concurrently.

module purge
module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/TVPsplinePSD/.venv/bin/activate

set -euo pipefail
mkdir -p logs
export JAX_PLATFORMS=cpu

NT_VALUES=(32 64 128 256 512 1024)
NT=${NT_VALUES[$SLURM_ARRAY_TASK_ID]}

python studies/paper_figures/scripts/make_sim_study_figures.py \
    --prepare-references --nt "$NT" --nf 32 --freq-knots 6 \
    --reference-draws "${REFERENCE_DRAWS:-200}" --skip-fig1
