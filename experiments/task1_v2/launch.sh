#!/usr/bin/env bash
#SBATCH --job-name=task1_v2
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

EXP_DIR="experiments/task1_v2"
OUT_DIR="${EXP_DIR}/output"

N_PARALLEL=4

# name, normalize_volume, normalize_test_volume, masking, pooling
runs=(
    "mask-mean_norm-none_pool-mean false false mean mean"
    "mask-zero_norm-none_pool-mean false false zero mean"
    "mask-zero_norm-test_pool-mean false true zero mean"
    "mask-zero_norm-train_pool-mean true false zero mean"
    "mask-zero_norm-both_pool-mean true true zero mean"
    "mask-zero_norm-none_pool-local false false zero local"
    "mask-zero_norm-both_pool-local true true zero local"
    "mask-zero_norm-none_pool-ensemble false false zero ensemble"
    "mask-zero_norm-both_pool-ensemble true true zero ensemble"
)

run_one() {
    local name="$1" normalize="$2" normalize_test="$3" masking="$4" pooling="$5"
    if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
        echo "result ${name} exists; skipping"
        return 0
    fi
    echo "=== ${name} ==="
    uv run --no-sync python -m fomo_tune.main_task1 train \
        output_root="${OUT_DIR}" \
        name="${name}" \
        masking="${masking}" \
        pooling="${pooling}" \
        normalize_volume="${normalize}" \
        normalize_test_volume="${normalize_test}"
}
export -f run_one
export OUT_DIR

printf '%s\n' "${runs[@]}" |
    parallel --will-cite --colsep ' ' --jobs "${N_PARALLEL}" --line-buffer --tagstring '{1}' \
        run_one {1} {2} {3} {4} {5}
