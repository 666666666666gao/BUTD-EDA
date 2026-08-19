#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ACTIVE_RUN="${REPO_ROOT}/logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_from_official_init/03_no_sacr_rapf_qahnl_base/scanrefer_spacy/1787104780"
RECEIPT="${ACTIVE_RUN}/best_primary.json"
CHECKPOINT="${ACTIVE_RUN}/ckpt_best_primary.pth"
MIN_PAUSE_EPOCH="${MIN_PAUSE_EPOCH:-10}"
PAUSE_JSON="${STATE_ROOT}/ablation_row03_pause_receipt.json"

echo "[$(timestamp)] Waiting for row03 checkpoint at epoch >= ${MIN_PAUSE_EPOCH}; polling every ${POLL_SECONDS}s."
while true; do
  if [ -f "${RECEIPT}" ] && [ -f "${CHECKPOINT}" ]; then
    epoch="$(${PYTHON} - "${RECEIPT}" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))['epoch']))
PY
)"
    if [ "${epoch}" -ge "${MIN_PAUSE_EPOCH}" ] && [ -f "${ACTIVE_RUN}/eval_epoch_${epoch}.log" ]; then
      break
    fi
    echo "[$(timestamp)] Latest resumable row03 checkpoint is epoch ${epoch}; waiting."
  else
    echo "[$(timestamp)] Row03 checkpoint receipt not ready; waiting."
  fi
  sleep "${POLL_SECONDS}"
done

assert_checkpoint "${CHECKPOINT}" "${MIN_PAUSE_EPOCH}"
sha="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
"${PYTHON}" - "${RECEIPT}" "${CHECKPOINT}" "${sha}" "${PAUSE_JSON}" <<'PY'
import json, sys
receipt_path, checkpoint, sha, output = sys.argv[1:]
receipt = json.load(open(receipt_path))
payload = {
    'status': 'resumable_pause_ready',
    'job_id': '03_no_sacr_rapf_qahnl_base',
    'checkpoint': checkpoint,
    'checkpoint_sha256': sha,
    'epoch': int(receipt['epoch']),
    'metric': receipt['metric'],
    'score': float(receipt['score']),
    'resume_script': '/home/gb/new butd/butd_detr-main/experiment_packages/butd_same_module_optimization_20260819/resume_ablation_row03.sh',
}
open(output, 'w').write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, sort_keys=True))
PY

echo "[$(timestamp)] Checkpoint is durable; pausing ablation queue and its restart watchdog."
screen -S scanrefer_ablation_watchdog_20260814 -X quit >/dev/null 2>&1 || true
screen -S scanrefer_ablation_retrain_20260814_v2 -X quit >/dev/null 2>&1 || true

for _ in $(seq 1 12); do
  if ! pgrep -af 'train_dist_mod.py' | grep -F -- "${ACTIVE_RUN%/scanrefer_spacy/1787104780}" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

mapfile -t remaining < <(pgrep -f "${ACTIVE_RUN%/scanrefer_spacy/1787104780}" || true)
if [ "${#remaining[@]}" -gt 0 ]; then
  echo "[$(timestamp)] Terminating remaining row03 processes: ${remaining[*]}"
  kill -TERM "${remaining[@]}" 2>/dev/null || true
  sleep 10
fi

if pgrep -af 'train_dist_mod.py' | grep -F -- "${ACTIVE_RUN%/scanrefer_spacy/1787104780}" >/dev/null 2>&1; then
  echo "ERROR: row03 training still active after scoped termination." >&2
  exit 51
fi

touch "${REPO_ROOT}/logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_queue/status/03_no_sacr_rapf_qahnl_base.paused"
echo "[$(timestamp)] Ablation row03 paused safely. Resume receipt: ${PAUSE_JSON}"

