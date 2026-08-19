#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}" || exit 1

DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT}
OUTPUT_DIR=${OUTPUT_DIR:-logs/new_method_v2/tuning/eval_fusion_sweep}
GPU=${GPU:-0}

HAS_CHECKPOINT_ARG=0
for arg in "$@"; do
  case "${arg}" in
    --checkpoint|--checkpoint=*|--checkpoint_path|--checkpoint_path=*)
      HAS_CHECKPOINT_ARG=1
      ;;
  esac
done

ARGS=(
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --gpu "${GPU}"
)

if [ "${HAS_CHECKPOINT_ARG}" = "0" ] && [ -n "${CHECKPOINT:-}" ]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
fi

if [ "${HAS_CHECKPOINT_ARG}" = "0" ] && [ -z "${CHECKPOINT:-}" ]; then
  echo "Eval-only sweep skipped: no checkpoint was provided."
  echo "Run:"
  echo "python scripts/new_method_v2/tuning/scanrefer_two_stage_eval_fusion_sweep.py --checkpoint /path/to/existing_checkpoint.pth --data-root ${DATA_ROOT} --output-dir ${OUTPUT_DIR} --gpu ${GPU}"
  exit 0
fi

python scripts/new_method_v2/tuning/scanrefer_two_stage_eval_fusion_sweep.py \
  "${ARGS[@]}" \
  "$@"
