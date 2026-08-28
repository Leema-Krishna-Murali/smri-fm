#!/usr/bin/env bash
#SBATCH --job-name=fomo_tune_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=2:00:00
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

EXP_DIR="experiments/fomo_tune_v2"
OUT_DIR="${EXP_DIR}/output"

runs=(
    "task1 main_task1"
    "task5 main_task5"
)

for run in "${runs[@]}"; do
    read -r name module <<<"${run}"

    if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
        echo "result ${name} exists; skipping"
        continue
    fi

    echo "=== ${name} ==="
    uv run --no-sync python -m "fomo_tune.${module}" train \
        output_root="${OUT_DIR}" \
        name="${name}"
done

echo "=== results ==="
cat "${OUT_DIR}"/*/metrics.json
