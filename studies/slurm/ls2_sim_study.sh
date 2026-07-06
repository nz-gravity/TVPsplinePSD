#!/bin/bash
#SBATCH --job-name=ls2_sim
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ls2_sim_%A_%a.out

# LS2 simulation study, one duration shard per array task.
# Submit from the repository root:  sbatch studies/slurm/ls2_sim_study.sh
# After all shards finish:          sbatch studies/slurm/ls2_render.sh

module purge
module load gcc/13.3.0 python/3.12.3
source 

set -euo pipefail
mkdir -p logs

NT_VALUES=(24 48 96 192 384)
NT=${NT_VALUES[$SLURM_ARRAY_TASK_ID]}

export JAX_PLATFORMS=cpu
python notes/scripts/make_sim_study_figures.py \
    --nt "$NT" --repeats "${REPEATS:-100}" --skip-fig1
