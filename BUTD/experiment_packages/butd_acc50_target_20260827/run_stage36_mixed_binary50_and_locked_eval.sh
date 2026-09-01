#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
CLEAN_DUMP="${ROOT}/stage24_stage16_e8_train_geometry_dump/stage16_e8_train_geometry.pt"
AUG_DUMP="${ROOT}/stage35_stage16_e8_augmented_train_dump/stage16_e8_augmented_train_geometry.pt"
OUT="${ROOT}/stage36_mixed_binary50_ranker"
VAL_DUMP="${ROOT}/stage17_stage16_e8_geometry_dump/stage16_e8_geometry.pt"
VAL_RESULT="${ROOT}/stage37_mixed_locked_val_eval.json"

cd "${R}"
[ -f "${CLEAN_DUMP}" ] && [ -f "${AUG_DUMP}" ] && [ -f "${VAL_DUMP}" ]
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 361; }
[ ! -e "${VAL_RESULT}" ] || { echo "refusing to overwrite ${VAL_RESULT}" >&2; exit 362; }
grep -q 'STAGE35_AUGMENTED_TRAIN_DUMP_PASS rows=36665' "${P}/stage35_augmented_dump.log"

"${PYTHON}" "${P}/train_joint_option_ranker.py" self-test
"${PYTHON}" "${P}/train_joint_option_ranker.py" mixed-binary50-train \
  "${CLEAN_DUMP}" "${AUG_DUMP}" "${OUT}" \
  --max-candidates 8 --num-threads 16 \
  > "${P}/stage36_mixed_binary50_train.log" 2>&1

[ -f "${OUT}/mixed_binary50_ranker.txt" ]
[ -f "${OUT}/locked_mixed_policy.json" ]
sha256sum "${OUT}/locked_mixed_policy.json" \
  > "${ROOT}/stage36_lock_sha256_before_val.txt"

"${PYTHON}" "${P}/train_joint_option_ranker.py" evaluate \
  "${VAL_DUMP}" "${OUT}/mixed_binary50_ranker.txt" \
  "${OUT}/locked_mixed_policy.json" "${VAL_RESULT}" \
  > "${P}/stage37_mixed_locked_val_eval.log" 2>&1

"${PYTHON}" - "${VAL_RESULT}" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], 'r'))
selected = result['selected']
print('STAGE37_MIXED_LOCKED_VAL acc025={:.10f} acc050={:.10f} '
      'offline_goal={}'.format(
          selected['acc025'], selected['acc050'],
          result['goal_achieved_offline']))
PY
