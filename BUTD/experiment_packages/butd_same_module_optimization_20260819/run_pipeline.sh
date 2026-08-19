#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"

exec > >(tee -a "${STATE_ROOT}/pipeline.log") 2>&1
echo "[$(timestamp)] BUTD same-module optimization pipeline started."
bash "${HERE}/01_pause_ablation_after_checkpoint.sh"
bash "${HERE}/02_optimize_scanrefer_same_module.sh"
bash "${HERE}/03_train_nr3d_sr3d_same_module.sh"
echo "[$(timestamp)] BUTD same-module optimization pipeline completed."
touch "${STATE_ROOT}/PIPELINE_COMPLETED"

