#!/bin/bash
#SBATCH --job-name=ls2_matched_pilot
#SBATCH --array=0-8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ls2_matched_pilot_%A_%a.out

# Cheap first pass for Figure 3. Five primary 8-knot durations establish the
# matched comparison; 6/10-knot runs only at the two endpoints test whether
# frequency flexibility changes the conclusion.

module purge
module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/TVPsplinePSD/.venv/bin/activate

set -euo pipefail
mkdir -p logs
export JAX_PLATFORMS=cpu

NT_VALUES=(24 48 96 192 384 24 384 24 384)
KNOT_VALUES=(8 8 8 8 8 6 6 10 10)
REPEAT_VALUES=(30 30 30 30 30 20 20 20 20)
NT=${NT_VALUES[$SLURM_ARRAY_TASK_ID]}
KNOTS=${KNOT_VALUES[$SLURM_ARRAY_TASK_ID]}
REPEATS=${REPEAT_VALUES[$SLURM_ARRAY_TASK_ID]}

python studies/paper_figures/scripts/make_sim_study_figures.py \
    --nt "$NT" --freq-knots "$KNOTS" --repeats "$REPEATS" \
    --reference-draws "${REFERENCE_DRAWS:-200}" \
    --skip-fig1 --skip-render
