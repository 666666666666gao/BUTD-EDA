#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
DUMP="${ROOT}/stage104_stage95_e11_calibrated_geometry_dump/stage95_e11_geometry.pt"
SCRIPT="${P}/train_joint_option_ranker.py"
OUT="${ROOT}/stage106_stage95_calibrated_candidate_oracle.json"
LOG="${P}/stage106_stage95_calibrated_candidate_oracle.log"

test ! -e "${OUT}"
test "$(sha256sum "${DUMP}" | awk '{print $1}')" = \
  "5bf6a572e33acb9b3b523286d44b317920c6f3db0d76b000687a8c55d8febca0"
test "$(sha256sum "${SCRIPT}" | awk '{print $1}')" = \
  "67b0c8ea0f0baaab57ca961bc4cd01c6f6128d21fb3b82db5e670d07e293b407"

cd "${R}"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" oracle \
  "${DUMP}" "${OUT}" --max-candidates 8 2>&1 | tee "${LOG}"
test -s "${OUT}"
chmod 0444 "${OUT}"
