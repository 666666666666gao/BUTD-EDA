#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
DUMP="${ROOT}/stage75_stage16_e8_val_geometry_semantic_dump/stage16_e8_val_geometry_semantic_frozen_legacy.pt"
LOCK="${ROOT}/stage92_stage29_continuous_box_residual/locked_box_residual_policy.json"
RESULT="${ROOT}/stage93_continuous_box_residual_locked_val_eval.json"

cd "${R}"
[ -f "${DUMP}" ]
[ -f "${LOCK}" ]
[ ! -e "${RESULT}" ] || { echo "refusing to overwrite ${RESULT}" >&2; exit 293; }
"${PYTHON}" - "${LOCK}" <<'PY'
import json
import sys

lock = json.load(open(sys.argv[1], encoding='utf-8'))
assert lock['external_eval_worthy'] is True, 'Stage92 internal gate did not pass'
PY
"${PYTHON}" -u "${P}/train_stage29_continuous_box_residual.py" evaluate \
  "${DUMP}" "${LOCK}" "${RESULT}" --device cuda \
  2>&1 | tee "${P}/stage93_continuous_box_residual_locked_val_eval.log"
[ -f "${RESULT}" ]
sha256sum "${RESULT}" > "${RESULT}.sha256"
echo STAGE93_CONTINUOUS_BOX_RESIDUAL_LOCKED_VAL_EVALUATED
