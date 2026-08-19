#!/usr/bin/env bash
#SBATCH --job-name=task4_depth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --partition=main
#SBATCH --exclude=n-3
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --account=sophont
#SBATCH --qos=top

set -euo pipefail

ROOT="/data/connor/fomo_tune"
cd $ROOT

set -a
source .env
set +a

EXP_DIR="experiments/fomo_tune_baseline_task4"
OUT_DIR="${EXP_DIR}/output"

N_PARALLEL=4

# name, scale, depth
runs=(
    "s2_d06 2 6"
    "s2_d08 2 8"
    "s2_d10 2 10"
    "s2_d12 2 12"
    "s3_d06 3 6"
    "s3_d08 3 8"
    "s3_d10 3 10"
    "s3_d12 3 12"
)

run_one() {
    local name="$1" scale="$2" depth="$3"
    if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
        echo "result ${name} exists; skipping"
        return 0
    fi
    echo "=== ${name} ==="
    uv run --no-sync python -m fomo_tune.main_task4 train \
        output_root="${OUT_DIR}" \
        name="${name}" \
        scale="${scale}" \
        subcell=4 \
        depth="${depth}"
}
export -f run_one
export OUT_DIR

printf '%s\n' "${runs[@]}" |
    parallel --will-cite --colsep ' ' --jobs "${N_PARALLEL}" --line-buffer --tagstring '{1}' \
        run_one {1} {2} {3}
