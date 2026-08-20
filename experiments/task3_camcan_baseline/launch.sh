#!/usr/bin/env bash
#SBATCH --job-name=task3_camcan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --account=sophont

set -euo pipefail

# Infer the root directory (nb, breaks convention of hard-coded path in other scripts).
ROOT="$(git rev-parse --show-toplevel)"
cd $ROOT

EXP_DIR="experiments/task3_camcan_baseline"
OUT_DIR="${EXP_DIR}/output"

name="task3_camcan"

if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
    echo "result ${name} exists; skipping"
    exit 0
fi

# quoted: bash would read the unquoted brackets as a glob
uv run --no-sync python -m fomo_tune.main_task3 train \
    output_root="${OUT_DIR}" \
    name="${name}" \
    'evals=[camcan]'

echo "=== results ==="
cat "${OUT_DIR}/${name}/metrics.json"
