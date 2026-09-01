#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
TRAIN_DUMP="${ROOT}/stage24_stage16_e8_train_geometry_dump/stage16_e8_train_geometry.pt"
SOURCE_LOCK="${ROOT}/stage25_joint_option_ranker/locked_policy.json"
OUT="${ROOT}/stage27_crossfit_ensemble"
VAL_DUMP="${ROOT}/stage17_stage16_e8_geometry_dump/stage16_e8_geometry.pt"
VAL_RESULT="${ROOT}/stage28_crossfit_locked_val_eval.json"

cd "${R}"
[ -f "${TRAIN_DUMP}" ] && [ -f "${SOURCE_LOCK}" ] && [ -f "${VAL_DUMP}" ]
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 271; }
[ ! -e "${VAL_RESULT}" ] || { echo "refusing to overwrite ${VAL_RESULT}" >&2; exit 272; }

"${PYTHON}" "${P}/train_joint_option_ranker.py" self-test
"${PYTHON}" "${P}/train_joint_option_ranker.py" crossfit-train \
  "${TRAIN_DUMP}" "${SOURCE_LOCK}" "${OUT}" \
  --num-folds 5 --num-threads 16 \
  > "${P}/stage27_crossfit_train.log" 2>&1

[ -f "${OUT}/locked_ensemble_policy.json" ]
sha256sum "${OUT}/locked_ensemble_policy.json" \
  > "${ROOT}/stage27_lock_sha256_before_val.txt"

"${PYTHON}" "${P}/train_joint_option_ranker.py" ensemble-evaluate \
  "${VAL_DUMP}" "${OUT}/locked_ensemble_policy.json" "${VAL_RESULT}" \
  > "${P}/stage28_crossfit_locked_val_eval.log" 2>&1

"${PYTHON}" - "${VAL_RESULT}" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], 'r'))
selected = result['selected']
print('STAGE28_CROSSFIT_LOCKED_VAL acc025={:.10f} acc050={:.10f} '
      'offline_goal={}'.format(
          selected['acc025'], selected['acc050'],
          result['goal_achieved_offline']))
PY
