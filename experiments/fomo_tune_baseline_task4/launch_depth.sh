#!/usr/bin/env bash
#SBATCH --job-name=task4_depth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --partition=main
#SBATCH --requeue
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

# One GPU, several runs at once: a run peaks at ~3GB of 80 and leaves the device idle between
# bursts, so the cap is host cores rather than the device. 16 cores is what a 1-GPU job gets by
# default here (DefCpuPerGPU), asked for explicitly since this job depends on having them.
# On preemption `--requeue` reruns the job, and the guard below skips whatever already finished.
N_PARALLEL=5

# name, scale, depth. `subcell=4` throughout: it won its column at both scales in sweep A. The two
# anchors re-measure a sweep A config under the per-label cut, so they lead.
runs=(
    "s3_dfinal 3 null"
    "s2_d04 2 4"
    "s2_d00 2 0"
    "s2_d01 2 1"
    "s2_d02 2 2"
    "s3_d00 3 0"
    "s3_d01 3 1"
    "s3_d02 3 2"
    "s3_d04 3 4"
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
