#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
TRAIN_DUMP="${ROOT}/stage73_stage16_e8_train_geometry_semantic_dump/stage16_e8_train_geometry_semantic.pt"
OUT="${ROOT}/stage88_crossfit_projected_oof_gate"
LOG="${P}/stage88_crossfit_projected_oof_train.log"

cd "${R}"
[ -f "${TRAIN_DUMP}" ]
[ -f "${P}/train_crossfit_projected_override_gate.py" ]
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 288; }

avail_kb="$(df --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')"
[ "${avail_kb}" -ge 2097152 ] || {
  echo "insufficient /root/autodl-tmp free space: ${avail_kb} KiB" >&2
  exit 188
}

"${PYTHON}" -u "${P}/train_crossfit_projected_override_gate.py" train \
  "${TRAIN_DUMP}" \
  "${P}/train_joint_option_ranker.py" \
  "${P}/train_joint_option_ranker_semantic.py" \
  "${ROOT}/stage29_binary50_ranker/binary50_option_ranker.txt" \
  "${ROOT}/stage74_semantic_binary50_ranker/binary50_option_ranker.txt" \
  "${ROOT}/stage29_binary50_ranker/locked_binary50_policy.json" \
  "${ROOT}/stage74_semantic_binary50_ranker/locked_binary50_policy.json" \
  "${OUT}" --num-folds 5 --num-threads 32 --preserve-acc025-drop 0.0 \
  2>&1 | tee "${LOG}"

[ -f "${OUT}/oof_override_gate.txt" ]
[ -f "${OUT}/locked_oof_override_gate.json" ]
[ -f "${OUT}/oof_meta_features.npz" ]
sha256sum \
  "${OUT}/oof_override_gate.txt" \
  "${OUT}/locked_oof_override_gate.json" \
  "${OUT}/oof_meta_features.npz" \
  > "${OUT}/sha256.txt"

"${PYTHON}" - "${OUT}/locked_oof_override_gate.json" <<'PY'
import json
import sys

lock = json.load(open(sys.argv[1], encoding='utf-8'))
test = lock['internal']['test']
net50 = int(test['fix050_count']) - int(test['break050_count'])
net25 = int(test['fix025_count']) - int(test['break025_count'])
print('STAGE88_INTERNAL_TEST_NET', {'acc025': net25, 'acc050': net50})
print('STAGE88_EXTERNAL_EVAL_WORTHY', bool(net50 >= 19 and net25 >= 0))
PY

echo STAGE88_CROSSFIT_OOF_GATE_LOCKED
