#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
TRAIN_DUMP="${ROOT}/stage24_stage16_e8_train_geometry_dump/stage16_e8_train_geometry.pt"
OUT="${ROOT}/stage33_pointwise_ranker"
VAL_DUMP="${ROOT}/stage17_stage16_e8_geometry_dump/stage16_e8_geometry.pt"
VAL_RESULT="${ROOT}/stage34_pointwise_locked_val_eval.json"

cd "${R}"
[ -f "${TRAIN_DUMP}" ] && [ -f "${VAL_DUMP}" ]
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 331; }
[ ! -e "${VAL_RESULT}" ] || { echo "refusing to overwrite ${VAL_RESULT}" >&2; exit 332; }

"${PYTHON}" "${P}/train_joint_option_ranker.py" self-test
"${PYTHON}" "${P}/train_joint_option_ranker.py" pointwise-train \
  "${TRAIN_DUMP}" "${OUT}" --max-candidates 8 --num-threads 16 \
  > "${P}/stage33_pointwise_train.log" 2>&1

[ -f "${OUT}/pointwise_option_model.txt" ]
[ -f "${OUT}/locked_pointwise_policy.json" ]
sha256sum "${OUT}/locked_pointwise_policy.json" \
  > "${ROOT}/stage33_lock_sha256_before_val.txt"

"${PYTHON}" "${P}/train_joint_option_ranker.py" evaluate \
  "${VAL_DUMP}" "${OUT}/pointwise_option_model.txt" \
  "${OUT}/locked_pointwise_policy.json" "${VAL_RESULT}" \
  > "${P}/stage34_pointwise_locked_val_eval.log" 2>&1

"${PYTHON}" - "${VAL_RESULT}" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], 'r'))
selected = result['selected']
print('STAGE34_POINTWISE_LOCKED_VAL acc025={:.10f} acc050={:.10f} '
      'offline_goal={}'.format(
          selected['acc025'], selected['acc050'],
          result['goal_achieved_offline']))
PY
