#!/usr/bin/env bash
#SBATCH --job-name=task4_depth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --partition=main
#SBATCH --array=0-8
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=sophont
#SBATCH --qos=high

set -euo pipefail

ROOT="/data/connor/fomo_tune"
cd $ROOT

set -a
source .env
set +a

EXP_DIR="experiments/fomo_tune_baseline_task4"
OUT_DIR="${EXP_DIR}/output"

# name, scale, depth, indexed by array task. `subcell=4` throughout: it won its column at both
# scales in sweep A. The two anchors re-measure a sweep A config under the per-label cut, so they
# go first; then the shallow depths at the cheap scale, then the combination never yet run.
runs=(
    "s3_dfinal 3 null"
    "s2_d04    2 4"
    "s2_d00    2 0"
    "s2_d01    2 1"
    "s2_d02    2 2"
    "s3_d00    3 0"
    "s3_d01    3 1"
    "s3_d02    3 2"
    "s3_d04    3 4"
)

read -r name scale depth <<<"${runs[${SLURM_ARRAY_TASK_ID}]}"

if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
    echo "result ${name} exists; skipping"
    exit 0
fi

echo "=== ${name} ==="
uv run --no-sync python -m fomo_tune.main_task4 train \
    output_root="${OUT_DIR}" \
    name="${name}" \
    scale="${scale}" \
    subcell=4 \
    depth="${depth}"
