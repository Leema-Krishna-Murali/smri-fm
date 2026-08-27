#!/usr/bin/env bash
#SBATCH --job-name=task3_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=main
# #SBATCH --exclude=n-1,n-3
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=sophont
#SBATCH --qos=high

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd $ROOT

set -a
source .env
set +a

EXP_DIR="experiments/task3_v2"
OUT_DIR="${EXP_DIR}/output"

CKPT="hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"

depth=null
label="bal_ckpt-walnut_depth-final"

# suffix, train_aug, test_aug
runs=(
    "aug-none false false"
    "aug-both true true"
    "aug-train true false"
    "aug-test false true"
)

for run in "${runs[@]}"; do
    read -r suffix train_aug test_aug <<<"${run}"
    name="${label}_${suffix}"

    if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
        echo "result ${name} exists; skipping"
        continue
    fi

    echo "=== ${name} ==="
    uv run --no-sync python -m fomo_tune.main_task3 train \
        output_root="${OUT_DIR}" \
        ckpt_path="${CKPT}" \
        name="${name}" \
        depth="${depth}" \
        train_aug="${train_aug}" \
        test_aug="${test_aug}" \
        balance_age=true \
        workers=12 \
        evals="[camcan]"
done

echo "=== results ==="
cat "${OUT_DIR}/${label}"_*/metrics.json
