#!/usr/bin/env bash
#SBATCH --job-name=task4_block
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=main
#SBATCH --array=0-5%3
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=sophont

set -euo pipefail

ROOT="/data/connor/fomo_tune"
cd $ROOT

set -a
source .env
set +a

EXP_DIR="experiments/fomo_tune_baseline_task4"
OUT_DIR="${EXP_DIR}/output"

# Blocks of 24, at the default geometry. The final post-norm output is the `s2_c4` run of
# launch_geometry.sh, so it is not repeated here.
blocks=(23 19 15 11 7 3)

block="${blocks[${SLURM_ARRAY_TASK_ID}]}"
name="$(printf "blk%02d" "${block}")"

if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
    echo "result ${name} exists; skipping"
    exit 0
fi

echo "=== ${name} ==="
uv run --no-sync python -m fomo_tune.main_task4 train \
    output_root="${OUT_DIR}" \
    name="${name}" \
    scale=2 \
    subcell=4 \
    block="${block}"
