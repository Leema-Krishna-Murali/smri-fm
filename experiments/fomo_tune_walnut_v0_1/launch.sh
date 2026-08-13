#!/usr/bin/env bash
#SBATCH --job-name=fomo_tune_walnut_v0_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --account=sophont

set -euo pipefail

ROOT="/data/connor/fomo_tune"
cd $ROOT

set -a
source .env
set +a

EXP_DIR="experiments/fomo_tune_walnut_v0_1"
OUT_DIR="${EXP_DIR}/output"

# same protocol as fomo_tune_baseline, only the backbone differs
CKPT="hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"

# name, module. Cheapest first, so a broken environment fails in 90s.
runs=(
    "task1 main_task1"
    "task5 main_task5"
    "task3 main_task3"
    "task2 main_task2"
)

for run in "${runs[@]}"; do
    read -r name module <<<"${run}"

    if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
        echo "result ${name} exists; skipping"
        continue
    fi

    echo "=== ${name} ==="
    uv run --no-sync python -m "fomo_tune.${module}" train \
        ckpt_path="${CKPT}" \
        output_root="${OUT_DIR}" \
        name="${name}"
done

echo "=== results ==="
cat "${OUT_DIR}"/*/metrics.json
