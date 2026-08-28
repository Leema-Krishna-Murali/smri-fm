#!/usr/bin/env bash
#SBATCH --job-name=task1_heads
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

EXP_DIR="experiments/task1_heads"
OUT_DIR="${EXP_DIR}/output"

N_PARALLEL=4

WALNUT="hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"

# the embedding config is fixed at task1_v2's headline run; only the head varies
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

runs=()
for head in "${heads[@]}"; do
    runs+=("${head} default")
    runs+=("${head} walnut")
done

run_one() {
    local name="$1" head="$2" scoring="$3" head_c="$4" scaler="$5" ckpt="$6"
    local full="${name}"
    local args=()
    if [[ "${ckpt}" == "walnut" ]]; then
        full="ckpt-walnut_${full}"
        args+=(ckpt_path="${WALNUT}")
    fi
    if [[ -f "${OUT_DIR}/${full}/metrics.json" ]]; then
        echo "result ${full} exists; skipping"
        return 0
    fi
    echo "=== ${full} ==="
    uv run --no-sync python -m fomo_tune.main_task1 train \
        output_root="${OUT_DIR}" \
        name="${full}" \
        masking=zero \
        pooling=mean \
        head="${head}" \
        head_scoring="${scoring}" \
        head_C="${head_c}" \
        scaler="${scaler}" \
        "${args[@]}"
}
export -f run_one
export OUT_DIR WALNUT

printf '%s\n' "${runs[@]}" |
    parallel --will-cite --colsep ' ' --jobs "${N_PARALLEL}" --line-buffer --tagstring '{1}/{6}' \
        run_one {1} {2} {3} {4} {5} {6}
