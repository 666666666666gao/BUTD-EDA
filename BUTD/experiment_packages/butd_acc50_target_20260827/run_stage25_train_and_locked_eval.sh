#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
TRAIN_DUMP="${ROOT}/stage24_stage16_e8_train_geometry_dump/stage16_e8_train_geometry.pt"
TRAIN_OUT="${ROOT}/stage25_joint_option_ranker"
VAL_DUMP="${ROOT}/stage17_stage16_e8_geometry_dump/stage16_e8_geometry.pt"
VAL_RESULT="${ROOT}/stage26_locked_val_eval.json"
LOCK_HASH="${ROOT}/stage25_lock_sha256_before_val.txt"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"

cd "${R}"
[ -f "${TRAIN_DUMP}" ]
[ -f "${VAL_DUMP}" ]
[ ! -e "${TRAIN_OUT}" ] || { echo "refusing to overwrite ${TRAIN_OUT}" >&2; exit 251; }
[ ! -e "${VAL_RESULT}" ] || { echo "refusing to overwrite ${VAL_RESULT}" >&2; exit 252; }
grep -q 'STAGE24_TRAIN_DUMP_PASS rows=36665' "${P}/stage24_train_dump.log"

"${PYTHON}" "${P}/train_joint_option_ranker.py" self-test
"${PYTHON}" "${P}/train_joint_option_ranker.py" train \
  "${TRAIN_DUMP}" "${TRAIN_OUT}" \
  --max-candidates 8 --num-threads 16 \
  > "${P}/stage25_ranker_train.log" 2>&1

[ -f "${TRAIN_OUT}/joint_option_ranker.txt" ]
[ -f "${TRAIN_OUT}/locked_policy.json" ]
sha256sum "${TRAIN_OUT}/locked_policy.json" > "${LOCK_HASH}"

# First and only application of the locked train-only policy to ScanRefer val.
"${PYTHON}" "${P}/train_joint_option_ranker.py" evaluate \
  "${VAL_DUMP}" \
  "${TRAIN_OUT}/joint_option_ranker.txt" \
  "${TRAIN_OUT}/locked_policy.json" \
  "${VAL_RESULT}" \
  > "${P}/stage26_locked_val_eval.log" 2>&1

"${PYTHON}" - "${VAL_RESULT}" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], 'r'))
selected = result['selected']
print('STAGE26_LOCKED_VAL_RESULT acc025={:.10f} acc050={:.10f} '
      'offline_goal={}'.format(
          selected['acc025'], selected['acc050'],
          result['goal_achieved_offline']))
PY
