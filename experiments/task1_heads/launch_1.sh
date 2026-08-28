#!/usr/bin/env bash
#SBATCH --job-name=task1_heads
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

EXP_DIR="experiments/task1_heads"
OUT_DIR="${EXP_DIR}/output"

N_PARALLEL=8

CKPT="hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"

# task1_v2's best walnut run: ensemble pooling, both volume normalizations, zero masking.
# `head` sets both the global and the local probe, so the sweep moves them together.
# name, head, head_scoring, head_C, scaler
heads=(
    "head-cv-auc logistic_cv roc_auc 1.0 true"
    "head-cv-auc-noscale logistic_cv roc_auc 1.0 false"
    "head-cv-logloss logistic_cv neg_log_loss 1.0 true"
    "head-fixed-C0.01 logistic roc_auc 0.01 true"
    "head-fixed-C1 logistic roc_auc 1.0 true"
    "head-fixed-C1-noscale logistic roc_auc 1.0 false"
    "head-fixed-C100 logistic roc_auc 100.0 true"
    "head-lda lda roc_auc 1.0 true"
    "head-lda-noscale lda roc_auc 1.0 false"
)

run_one() {
    local name="$1" head="$2" scoring="$3" head_c="$4" scaler="$5"
    local full="walnut-ensemble_${name}"
    if [[ -f "${OUT_DIR}/${full}/metrics.json" ]]; then
        echo "result ${full} exists; skipping"
        return 0
    fi
    echo "=== ${full} ==="
    uv run --no-sync python -m fomo_tune.main_task1 train \
        output_root="${OUT_DIR}" \
        ckpt_path="${CKPT}" \
        name="${full}" \
        masking=zero \
        pooling=ensemble \
        normalize_volume=true \
        normalize_test_volume=true \
        head="${head}" \
        head_scoring="${scoring}" \
        head_C="${head_c}" \
        scaler="${scaler}"
}
export -f run_one
export OUT_DIR CKPT

printf '%s\n' "${heads[@]}" |
    parallel --will-cite --colsep ' ' --jobs "${N_PARALLEL}" --line-buffer --tagstring '{1}' \
        run_one {1} {2} {3} {4} {5}
