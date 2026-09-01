#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
TRAIN_DUMP="${ROOT}/stage24_stage16_e8_train_geometry_dump/stage16_e8_train_geometry.pt"
ORDINAL_MODEL="${ROOT}/stage25_joint_option_ranker/joint_option_ranker.txt"
ORDINAL_LOCK="${ROOT}/stage25_joint_option_ranker/locked_policy.json"
BINARY_MODEL="${ROOT}/stage29_binary50_ranker/binary50_option_ranker.txt"
BINARY_LOCK="${ROOT}/stage29_binary50_ranker/locked_binary50_policy.json"
OUT="${ROOT}/stage31_ordinal_binary_blend"
VAL_DUMP="${ROOT}/stage17_stage16_e8_geometry_dump/stage16_e8_geometry.pt"
VAL_RESULT="${ROOT}/stage32_blend_locked_val_eval.json"

cd "${R}"
for path in "${TRAIN_DUMP}" "${ORDINAL_MODEL}" "${ORDINAL_LOCK}" \
            "${BINARY_MODEL}" "${BINARY_LOCK}" "${VAL_DUMP}"; do
  [ -f "${path}" ]
done
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 311; }
[ ! -e "${VAL_RESULT}" ] || { echo "refusing to overwrite ${VAL_RESULT}" >&2; exit 312; }

"${PYTHON}" "${P}/train_joint_option_ranker.py" self-test
"${PYTHON}" "${P}/train_joint_option_ranker.py" blend-train \
  "${TRAIN_DUMP}" "${ORDINAL_MODEL}" "${ORDINAL_LOCK}" \
  "${BINARY_MODEL}" "${BINARY_LOCK}" "${OUT}" \
  > "${P}/stage31_blend_train.log" 2>&1

[ -f "${OUT}/locked_blend_policy.json" ]
sha256sum "${OUT}/locked_blend_policy.json" \
  > "${ROOT}/stage31_lock_sha256_before_val.txt"

"${PYTHON}" "${P}/train_joint_option_ranker.py" blend-evaluate \
  "${VAL_DUMP}" "${OUT}/locked_blend_policy.json" "${VAL_RESULT}" \
  > "${P}/stage32_blend_locked_val_eval.log" 2>&1

"${PYTHON}" - "${VAL_RESULT}" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], 'r'))
selected = result['selected']
print('STAGE32_BLEND_LOCKED_VAL acc025={:.10f} acc050={:.10f} '
      'offline_goal={}'.format(
          selected['acc025'], selected['acc050'],
          result['goal_achieved_offline']))
PY
