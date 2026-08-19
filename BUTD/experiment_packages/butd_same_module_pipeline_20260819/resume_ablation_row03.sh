#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PAUSE_JSON="${STATE_ROOT}/ablation_row03_pause_receipt.json"
[ -f "${PAUSE_JSON}" ] || { echo "Missing ${PAUSE_JSON}" >&2; exit 2; }
CHECKPOINT="$(${PYTHON} - "${PAUSE_JSON}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['checkpoint'])
PY
)"
assert_checkpoint "${CHECKPOINT}" 10
wait_gpu_idle

export NMV2_MAX_EPOCH=65
export ABLATION_LOG_ROOT="${REPO_ROOT}/logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_resume_from_pause"
exec bash "${REPO_ROOT}/scripts/ablations/scanrefer_20260814/03_no_sacr_rapf_scanrefer_20260814.sh" \
  --checkpoint_path "${CHECKPOINT}"

