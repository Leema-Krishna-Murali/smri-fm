#!/usr/bin/env bash
#SBATCH --job-name=task4_geometry
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=main
#SBATCH --array=0-9%5
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

# name, scale, subcell, indexed by array task. The current default first, so it starts first
# on a busy queue; then the rest of scale 2, then scale 1, then the scale 3 probe.
runs=(
    "s2_c4 2 4"
    "s2_c1 2 1"
    "s2_c2 2 2"
    "s2_c8 2 8"
    "s1_c1 1 1"
    "s1_c2 1 2"
    "s1_c4 1 4"
    "s1_c8 1 8"
    "s3_c2 3 2"
    "s3_c4 3 4"
)

read -r name scale subcell <<<"${runs[${SLURM_ARRAY_TASK_ID}]}"

if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
    echo "result ${name} exists; skipping"
    exit 0
fi

echo "=== ${name} ==="
uv run --no-sync python -m fomo_tune.main_task4 train \
    output_root="${OUT_DIR}" \
    name="${name}" \
    scale="${scale}" \
    subcell="${subcell}"
