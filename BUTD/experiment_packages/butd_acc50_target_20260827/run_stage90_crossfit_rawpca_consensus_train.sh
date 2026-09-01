#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
RAW_DUMP="${ROOT}/stage81_raw_query_pca_train/stage16_e8_train_geometry_semantic_rawpca.pt"
STAGE88="${ROOT}/stage88_crossfit_projected_oof_gate"
OUT="${ROOT}/stage90_crossfit_rawpca_consensus"
LOG="${P}/stage90_crossfit_rawpca_consensus_train.log"

cd "${R}"
[ -f "${RAW_DUMP}" ]
[ -f "${STAGE88}/oof_meta_features.npz" ]
[ -f "${STAGE88}/locked_oof_override_gate.json" ]
[ ! -e "${OUT}" ] || { echo "refusing to overwrite ${OUT}" >&2; exit 290; }
avail_kb="$(df --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')"
[ "${avail_kb}" -ge 2097152 ] || {
  echo "insufficient /root/autodl-tmp free space: ${avail_kb} KiB" >&2
  exit 190
}

"${PYTHON}" -u "${P}/train_crossfit_rawpca_consensus.py" train \
  "${RAW_DUMP}" \
  "${STAGE88}/oof_meta_features.npz" \
  "${STAGE88}/locked_oof_override_gate.json" \
  "${P}/train_crossfit_projected_override_gate.py" \
  "${P}/train_joint_option_ranker_semantic_rawpca.py" \
  "${ROOT}/stage82_rawpca_binary50_ranker/binary50_option_ranker.txt" \
  "${ROOT}/stage82_rawpca_binary50_ranker/locked_binary50_policy.json" \
  "${OUT}" --num-folds 5 --num-threads 32 \
  2>&1 | tee "${LOG}"

[ -f "${OUT}/locked_rawpca_consensus_policy.json" ]
[ -f "${OUT}/rawpca_oof_consensus_evidence.npz" ]
sha256sum \
  "${OUT}/locked_rawpca_consensus_policy.json" \
  "${OUT}/rawpca_oof_consensus_evidence.npz" \
  > "${OUT}/sha256.txt"
"${PYTHON}" - "${OUT}/locked_rawpca_consensus_policy.json" <<'PY'
import json
import sys

lock = json.load(open(sys.argv[1], encoding='utf-8'))
print('STAGE90_OOF_NET', {'acc025': lock['net025'], 'acc050': lock['net050']})
print('STAGE90_REQUIRED_NET050', lock['required_scaled_net050'])
print('STAGE90_EXTERNAL_EVAL_WORTHY', lock['external_eval_worthy'])
PY
echo STAGE90_RAWPCA_CONSENSUS_LOCKED
