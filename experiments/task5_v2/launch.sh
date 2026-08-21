#!/usr/bin/env bash
#SBATCH --job-name=task5_v2
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

EXP_DIR="experiments/task5_v2"
OUT_DIR="${EXP_DIR}/output"

N_PARALLEL=3

PT_FULL="hf://medarc/walnut/checkpoints/pretrain_full_90_10_h100/checkpoint-last.pth"
WALNUT="hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"

# `crop_ap` alone would equal crop-both: `features` crops on the way in, so a cropped training
# set implies a cropped test set. Only these three combinations are distinct.
# name, ckpt, crop_ap, crop_test_ap
runs=(
    "ckpt-ptfull_crop-none ${PT_FULL} false false"
    "ckpt-ptfull_crop-test ${PT_FULL} false true"
    "ckpt-ptfull_crop-both ${PT_FULL} true true"
    "ckpt-walnut_crop-none ${WALNUT} false false"
    "ckpt-walnut_crop-test ${WALNUT} false true"
    "ckpt-walnut_crop-both ${WALNUT} true true"
)

run_one() {
    local name="$1" ckpt="$2" crop="$3" crop_test="$4"
    if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
        echo "result ${name} exists; skipping"
        return 0
    fi
    echo "=== ${name} ==="
    uv run --no-sync python -m fomo_tune.main_task5 train \
        output_root="${OUT_DIR}" \
        ckpt_path="${ckpt}" \
        name="${name}" \
        crop_ap="${crop}" \
        crop_test_ap="${crop_test}"
}
export -f run_one
export OUT_DIR

printf '%s\n' "${runs[@]}" |
    parallel --will-cite --colsep ' ' --jobs "${N_PARALLEL}" --line-buffer --tagstring '{1}' \
        run_one {1} {2} {3} {4}
