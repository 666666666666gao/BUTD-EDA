#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
SCRIPT="${P}/train_joint_option_ranker_top2.py"
SCRIPT_SHA="dc1de9fa44df7fa9fc339591fb13533e1383ea793d76406ed13a8609c752e2c7"
TRAIN_DUMP="${ROOT}/stage24_stage16_e8_train_geometry_dump/stage16_e8_train_geometry.pt"
TRAIN_SHA="6df4cbd9a2177e1470ecf2209d614a12cd66e51d89e6c0cc20745b70a1bf70fb"
VAL_DUMP="${ROOT}/stage17_stage16_e8_geometry_dump/stage16_e8_geometry.pt"
OUT="${ROOT}/stage49_top2_binary50_ranker"
VAL_RESULT="${ROOT}/stage50_top2_binary50_locked_val_eval.json"

cd "${R}"
[ "$(sha256sum "${SCRIPT}" | awk '{print $1}')" = "${SCRIPT_SHA}" ]
[ "$(sha256sum "${TRAIN_DUMP}" | awk '{print $1}')" = "${TRAIN_SHA}" ]
[ -f "${VAL_DUMP}" ]
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 249; }
[ ! -e "${VAL_RESULT}" ] || { echo "refusing to overwrite ${VAL_RESULT}" >&2; exit 250; }

"${PYTHON}" "${SCRIPT}" self-test
"${PYTHON}" -u "${SCRIPT}" binary50-train \
  "${TRAIN_DUMP}" "${OUT}" --max-candidates 8 --num-threads 16 \
  2>&1 | tee "${P}/stage49_top2_binary50_train.log"

[ -f "${OUT}/binary50_option_ranker.txt" ]
[ -f "${OUT}/locked_binary50_policy.json" ]
sha256sum "${OUT}/binary50_option_ranker.txt" \
  "${OUT}/locked_binary50_policy.json" \
  > "${ROOT}/stage49_top2_lock_sha256_before_val.txt"

"${PYTHON}" -u "${SCRIPT}" evaluate \
  "${VAL_DUMP}" "${OUT}/binary50_option_ranker.txt" \
  "${OUT}/locked_binary50_policy.json" "${VAL_RESULT}" \
  2>&1 | tee "${P}/stage50_top2_binary50_locked_val_eval.log"

"${PYTHON}" - "${VAL_RESULT}" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
s = r['selected']
print('STAGE50_TOP2_BINARY50_LOCKED_VAL acc025={:.10f} acc050={:.10f} '
      'offline_goal={}'.format(
          s['acc025'], s['acc050'], r['goal_achieved_offline']))
PY
