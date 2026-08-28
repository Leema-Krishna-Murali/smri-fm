#!/usr/bin/env bash
#SBATCH --job-name=task3_perturb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --account=sophont
#SBATCH --qos=high

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd $ROOT

set -a
source .env
set +a

EXP_DIR="experiments/task3_perturb"
OUT_DIR="${EXP_DIR}/output"

EVALS="[camcan,camcan-thick_slice_5mm,camcan-acquired_at_2mm,camcan-random_scale]"

# name, train_views
runs=(
    "noaug []"
    "aug [thick_slice_5mm,acquired_at_2mm]"
    "aug_v2 [thick_slice_5mm,acquired_at_2mm,random_scale]"
)

for run in "${runs[@]}"; do
    read -r name train_views <<<"${run}"

    if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
        echo "result ${name} exists; skipping"
        continue
    fi

    echo "=== ${name} ==="
    uv run --no-sync python -m fomo_tune.main_task3 train \
        output_root="${OUT_DIR}" \
        name="${name}" \
        train_views="${train_views}" \
        workers=12 \
        evals="${EVALS}"
done

echo "=== results ==="
uv run --no-sync python "${EXP_DIR}/collect.py"
