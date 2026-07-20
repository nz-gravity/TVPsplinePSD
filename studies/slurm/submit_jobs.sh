#!/bin/bash

# Submit the matched-knot Figure 3 workflow from the repository root.
#
# Usage:
#   studies/slurm/submit_jobs.sh smoke
#   studies/slurm/submit_jobs.sh full

set -euo pipefail

SUBMIT_MODE=${1:-full}
SLURM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SLURM_DIR}/../.." && pwd)
cd "$REPO_ROOT"

case "$SUBMIT_MODE" in
    smoke)
        REF_ID=$(sbatch --parsable --array=5 \
            studies/slurm/ls2_fig3_references.sh)
        SMOKE_ID=$(sbatch --parsable \
            --dependency="afterok:${REF_ID}" \
            --array=59 \
            studies/slurm/ls2_fig3_production.sh)

        echo "reference job: ${REF_ID}"
        echo "worst-case smoke job: ${SMOKE_ID}"
        echo "inspect after completion:"
        echo "  sacct -j ${SMOKE_ID} --format=JobID,State,Elapsed,MaxRSS,ReqMem"
        ;;

    full)
        REF_ID=$(sbatch --parsable studies/slurm/ls2_fig3_references.sh)

        SMALL_ID=$(sbatch --parsable \
            --dependency="afterok:${REF_ID}" \
            --array=0-29%6 --mem=3G --time=01:00:00 \
            studies/slurm/ls2_fig3_production.sh)

        MEDIUM_ID=$(sbatch --parsable \
            --dependency="afterok:${REF_ID}" \
            --array=30-49%4 --mem=4G --time=01:30:00 \
            studies/slurm/ls2_fig3_production.sh)

        LARGE_ID=$(sbatch --parsable \
            --dependency="afterok:${REF_ID}" \
            --array=50-59%2 --mem=4G --time=02:30:00 \
            studies/slurm/ls2_fig3_production.sh)

        RENDER_ID=$(sbatch --parsable \
            --dependency="afterok:${SMALL_ID}:${MEDIUM_ID}:${LARGE_ID}" \
            studies/slurm/ls2_fig3_render.sh)

        echo "reference array: ${REF_ID}"
        echo "small fit array (N=2^10..2^12): ${SMALL_ID}"
        echo "medium fit array (N=2^13..2^14): ${MEDIUM_ID}"
        echo "large fit array (N=2^15): ${LARGE_ID}"
        echo "merge/render job: ${RENDER_ID}"
        ;;

    *)
        echo "Usage: $0 {smoke|full}" >&2
        exit 2
        ;;
esac
