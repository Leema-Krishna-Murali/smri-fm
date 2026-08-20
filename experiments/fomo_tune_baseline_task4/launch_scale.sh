#!/usr/bin/env bash
#SBATCH --job-name=task4_scale
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

N_PARALLEL=3

# name, scale, subcell, depth
runs=(
    "s2_c2_d04 2 2 4"
    "s2_c4_d04 2 4 4"
    "s2_c8_d04 2 8 4"
    "s3_c2_d04 3 2 4"
    "s3_c4_d04 3 4 4"
    "s3_c8_d04 3 8 4"
    "s4_c2_d04 4 2 4"
    "s4_c4_d04 4 4 4"
    "s4_c8_d04 4 8 4"
)

run_one() {
    local name="$1" scale="$2" subcell="$3" depth="$4"
    if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
        echo "result ${name} exists; skipping"
        return 0
    fi
    echo "=== ${name} ==="
    uv run --no-sync python -m fomo_tune.main_task4 train \
        output_root="${OUT_DIR}" \
        name="${name}" \
        scale="${scale}" \
        subcell="${subcell}" \
        depth="${depth}"
}
export -f run_one
export OUT_DIR

printf '%s\n' "${runs[@]}" |
    parallel --will-cite --colsep ' ' --jobs "${N_PARALLEL}" --line-buffer --tagstring '{1}' \
        run_one {1} {2} {3} {4}
