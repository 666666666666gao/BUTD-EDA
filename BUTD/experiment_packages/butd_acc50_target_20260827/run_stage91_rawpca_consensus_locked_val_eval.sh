#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
DUMP="${ROOT}/stage83_raw_query_pca_val/stage16_e8_val_geometry_semantic_rawpca.pt"
LOCK="${ROOT}/stage90_crossfit_rawpca_consensus/locked_rawpca_consensus_policy.json"
RESULT="${ROOT}/stage91_rawpca_consensus_locked_val_eval.json"

cd "${R}"
[ -f "${DUMP}" ]
[ -f "${LOCK}" ]
[ ! -e "${RESULT}" ] || { echo "refusing to overwrite ${RESULT}" >&2; exit 291; }
"${PYTHON}" - "${LOCK}" <<'PY'
import json
import sys

lock = json.load(open(sys.argv[1], encoding='utf-8'))
assert lock['external_eval_worthy'] is True, 'Stage90 internal gate did not pass'
PY
"${PYTHON}" -u "${P}/train_crossfit_rawpca_consensus.py" evaluate \
  "${DUMP}" "${LOCK}" "${RESULT}" \
  2>&1 | tee "${P}/stage91_rawpca_consensus_locked_val_eval.log"
[ -f "${RESULT}" ]
sha256sum "${RESULT}" > "${RESULT}.sha256"
echo STAGE91_RAWPCA_CONSENSUS_LOCKED_VAL_EVALUATED
