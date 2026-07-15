#!/bin/bash
#SBATCH --job-name=ls2_sim
#SBATCH --array=0-20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ls2_sim_%A_%a.out

# Matched-knot LS2 simulation study: 7 durations x 3 common frequency-knot
# counts. Each array task writes one duration/knot-count shard.
# Submit from the repository root:  sbatch studies/slurm/ls2_sim_study.sh
# After all shards finish:          sbatch studies/slurm/ls2_render.sh

module purge
module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/TVPsplinePSD/.venv/bin/activate

set -euo pipefail
mkdir -p logs

NT_VALUES=(24 48 96 192 384 768 1536)
FREQ_KNOT_VALUES=(6 8 10)
N_DURATION=${#NT_VALUES[@]}
KNOT_INDEX=$((SLURM_ARRAY_TASK_ID / N_DURATION))
NT_INDEX=$((SLURM_ARRAY_TASK_ID % N_DURATION))
NT=${NT_VALUES[$NT_INDEX]}
FREQ_KNOTS=${FREQ_KNOT_VALUES[$KNOT_INDEX]}

export JAX_PLATFORMS=cpu
python studies/paper_figures/scripts/make_sim_study_figures.py \
    --nt "$NT" \
    --freq-knots "$FREQ_KNOTS" \
    --repeats "${REPEATS:-100}" \
    --reference-draws "${REFERENCE_DRAWS:-1000}" \
    --skip-fig1 --skip-render
