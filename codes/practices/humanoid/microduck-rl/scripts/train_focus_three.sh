#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/home/huangxiao/.local/bin:${PATH}"
export TASK_REGEX='^Mjlab-(StandUp-Flat|GroundPick-Flat|BallKick-Flat)-MicroDuck$'
export MICRODUCK_COMPLETION=1
export MICRODUCK_EXPERIMENT="${MICRODUCK_EXPERIMENT:-focus_three_completion}"
echo "Focused tasks: StandUp-Flat, GroundPick-Flat, BallKick-Flat"
echo "Experiment: ${MICRODUCK_EXPERIMENT}"
./scripts/train_stable_matrix_stage1.sh
./scripts/train_stable_matrix_stage2.sh
