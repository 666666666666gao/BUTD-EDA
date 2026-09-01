#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
DUMP="${ROOT}/stage75_stage16_e8_val_geometry_semantic_dump/stage16_e8_val_geometry_semantic_frozen_legacy.pt"
MODEL="${ROOT}/stage88_crossfit_projected_oof_gate/oof_override_gate.txt"
LOCK="${ROOT}/stage88_crossfit_projected_oof_gate/locked_oof_override_gate.json"
RESULT="${ROOT}/stage89_crossfit_projected_oof_locked_val_eval.json"

cd "${R}"
[ -f "${DUMP}" ]
[ -f "${MODEL}" ]
[ -f "${LOCK}" ]
[ ! -e "${RESULT}" ] || { echo "refusing to overwrite ${RESULT}" >&2; exit 289; }

"${PYTHON}" -u "${P}/train_crossfit_projected_override_gate.py" evaluate \
  "${DUMP}" "${MODEL}" "${LOCK}" "${RESULT}" \
  2>&1 | tee "${P}/stage89_crossfit_projected_oof_locked_val_eval.log"

[ -f "${RESULT}" ]
sha256sum "${RESULT}" > "${RESULT}.sha256"
echo STAGE89_CROSSFIT_OOF_LOCKED_VAL_EVALUATED
