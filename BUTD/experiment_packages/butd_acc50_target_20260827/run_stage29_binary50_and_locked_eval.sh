#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
TRAIN_DUMP="${ROOT}/stage24_stage16_e8_train_geometry_dump/stage16_e8_train_geometry.pt"
OUT="${ROOT}/stage29_binary50_ranker"
VAL_DUMP="${ROOT}/stage17_stage16_e8_geometry_dump/stage16_e8_geometry.pt"
VAL_RESULT="${ROOT}/stage30_binary50_locked_val_eval.json"

cd "${R}"
[ -f "${TRAIN_DUMP}" ] && [ -f "${VAL_DUMP}" ]
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 291; }
[ ! -e "${VAL_RESULT}" ] || { echo "refusing to overwrite ${VAL_RESULT}" >&2; exit 292; }

"${PYTHON}" "${P}/train_joint_option_ranker.py" self-test
"${PYTHON}" "${P}/train_joint_option_ranker.py" binary50-train \
  "${TRAIN_DUMP}" "${OUT}" --max-candidates 8 --num-threads 16 \
  > "${P}/stage29_binary50_train.log" 2>&1

[ -f "${OUT}/binary50_option_ranker.txt" ]
[ -f "${OUT}/locked_binary50_policy.json" ]
sha256sum "${OUT}/locked_binary50_policy.json" \
  > "${ROOT}/stage29_lock_sha256_before_val.txt"

"${PYTHON}" "${P}/train_joint_option_ranker.py" evaluate \
  "${VAL_DUMP}" "${OUT}/binary50_option_ranker.txt" \
  "${OUT}/locked_binary50_policy.json" "${VAL_RESULT}" \
  > "${P}/stage30_binary50_locked_val_eval.log" 2>&1

"${PYTHON}" - "${VAL_RESULT}" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], 'r'))
selected = result['selected']
print('STAGE30_BINARY50_LOCKED_VAL acc025={:.10f} acc050={:.10f} '
      'offline_goal={}'.format(
          selected['acc025'], selected['acc050'],
          result['goal_achieved_offline']))
PY
