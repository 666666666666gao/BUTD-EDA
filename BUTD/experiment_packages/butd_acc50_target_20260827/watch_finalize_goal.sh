#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/home/gb/new butd/butd_detr-main"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
STAGE0_RECEIPT="${ROOT}/stage0_reload_verify/goal_receipt.json"
STAGE1_RECEIPT="${ROOT}/stage1_reload_verify/goal_receipt.json"
STAGE2_RECEIPT="${ROOT}/stage2_reload_verify/goal_receipt.json"
STAGE3_RECEIPT="${ROOT}/stage3_reload_verify/goal_receipt.json"
STAGE4_RECEIPT="${ROOT}/stage4_reload_verify/goal_receipt.json"
STAGE5_RECEIPT="${ROOT}/stage5_reload_verify/goal_receipt.json"
FINAL_RECEIPT="${ROOT}/FINAL_GOAL_RECEIPT.json"
CLEANUP_RECEIPT="${ROOT}/cleanup_receipt.json"
HANDOFF="${REPO_ROOT}/docs/EXPERIMENT_HANDOFF_MASTER.md"
QUEUE_PARENT_PID=114992
ORIGINAL_SHA="20f289d98657be242530e379fb23a3bea8137ef392dc7cd8f28675151dd805e4"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
STATUS_FILE="${PACKAGE_ROOT}/finalize_watch_status.txt"

read_achieved() {
  "${PYTHON}" -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["goal_achieved"])).lower())' "$1" 2>/dev/null || true
}

stage0_achieved=""
while [ -z "${stage0_achieved}" ]; do
  if [ -s "${STAGE0_RECEIPT}" ]; then
    stage0_achieved="$(read_achieved "${STAGE0_RECEIPT}")"
  fi
  if [ -z "${stage0_achieved}" ]; then
    printf '%s waiting_for_stage0_receipt\n' "$(date -Is)" > "${STATUS_FILE}"
    sleep "${WAIT_SECONDS}"
  fi
done

if [ "${stage0_achieved}" = "true" ]; then
  WINNER="stage0"
  WINNER_RECEIPT="${STAGE0_RECEIPT}"
else
stage1_achieved=""
while [ -z "${stage1_achieved}" ]; do
  if [ -s "${STAGE1_RECEIPT}" ]; then
    stage1_achieved="$(read_achieved "${STAGE1_RECEIPT}")"
  fi
  if [ -z "${stage1_achieved}" ]; then
    printf '%s waiting_for_stage1_receipt\n' "$(date -Is)" > "${STATUS_FILE}"
    sleep "${WAIT_SECONDS}"
  fi
done

if [ "${stage1_achieved}" = "true" ]; then
  WINNER="stage1"
  WINNER_RECEIPT="${STAGE1_RECEIPT}"
else
  stage2_achieved=""
  while [ -z "${stage2_achieved}" ]; do
    stage2_status="$(cat "${PACKAGE_ROOT}/stage2_watch_status.txt" 2>/dev/null || true)"
    if printf '%s' "${stage2_status}" | grep -q 'stage2_full_finetune_failed'; then
      printf '%s stopped; stage2_failed\n' "$(date -Is)" > "${STATUS_FILE}"
      exit 130
    fi
    if [ -s "${STAGE2_RECEIPT}" ]; then
      stage2_achieved="$(read_achieved "${STAGE2_RECEIPT}")"
    fi
    if [ -z "${stage2_achieved}" ]; then
      printf '%s waiting_for_stage2_receipt\n' "$(date -Is)" > "${STATUS_FILE}"
      sleep "${WAIT_SECONDS}"
    fi
  done
  if [ "${stage2_achieved}" = "true" ]; then
    WINNER="stage2"
    WINNER_RECEIPT="${STAGE2_RECEIPT}"
  else
    stage3_achieved=""
    while [ -z "${stage3_achieved}" ]; do
      stage3_status="$(cat "${PACKAGE_ROOT}/stage3_watch_status.txt" 2>/dev/null || true)"
      if printf '%s' "${stage3_status}" | grep -q 'stage3_quality_grid_failed'; then
        printf '%s stopped; stage3_failed\n' "$(date -Is)" > "${STATUS_FILE}"
        exit 133
      fi
      if [ -s "${STAGE3_RECEIPT}" ]; then
        stage3_achieved="$(read_achieved "${STAGE3_RECEIPT}")"
      fi
      if [ -z "${stage3_achieved}" ]; then
        printf '%s waiting_for_stage3_receipt\n' "$(date -Is)" > "${STATUS_FILE}"
        sleep "${WAIT_SECONDS}"
      fi
    done
    if [ "${stage3_achieved}" = "true" ]; then
      WINNER="stage3"
      WINNER_RECEIPT="${STAGE3_RECEIPT}"
    else
      stage4_achieved=""
      while [ -z "${stage4_achieved}" ]; do
        stage4_status="$(cat "${PACKAGE_ROOT}/stage4_watch_status.txt" 2>/dev/null || true)"
        if printf '%s' "${stage4_status}" | grep -q 'stage4_quality_head_failed'; then
          printf '%s stopped; stage4_failed\n' "$(date -Is)" > "${STATUS_FILE}"
          exit 134
        fi
        if [ -s "${STAGE4_RECEIPT}" ]; then
          stage4_achieved="$(read_achieved "${STAGE4_RECEIPT}")"
        fi
        if [ -z "${stage4_achieved}" ]; then
          printf '%s waiting_for_stage4_receipt\n' "$(date -Is)" > "${STATUS_FILE}"
          sleep "${WAIT_SECONDS}"
        fi
      done
      if [ "${stage4_achieved}" = "true" ]; then
        WINNER="stage4"
        WINNER_RECEIPT="${STAGE4_RECEIPT}"
      else
        stage5_achieved=""
        while [ -z "${stage5_achieved}" ]; do
          stage5_status="$(cat "${PACKAGE_ROOT}/stage5_watch_status.txt" 2>/dev/null || true)"
          if printf '%s' "${stage5_status}" | grep -q 'stage5_quality_logits_failed'; then
            printf '%s stopped; stage5_failed\n' "$(date -Is)" > "${STATUS_FILE}"
            exit 135
          fi
          if [ -s "${STAGE5_RECEIPT}" ]; then
            stage5_achieved="$(read_achieved "${STAGE5_RECEIPT}")"
          fi
          if [ -z "${stage5_achieved}" ]; then
            printf '%s waiting_for_stage5_receipt\n' "$(date -Is)" > "${STATUS_FILE}"
            sleep "${WAIT_SECONDS}"
          fi
        done
        if [ "${stage5_achieved}" != "true" ]; then
          printf '%s target_not_achieved_after_stage5\n' "$(date -Is)" > "${STATUS_FILE}"
          exit 0
        fi
        WINNER="stage5"
        WINNER_RECEIPT="${STAGE5_RECEIPT}"
      fi
    fi
  fi
fi
fi

while pgrep -af '[t]rain_dist_mod.py' >/dev/null 2>&1; do
  printf '%s waiting_for_final_gpu_idle\n' "$(date -Is)" > "${STATUS_FILE}"
  sleep "${WAIT_SECONDS}"
done

evaluator_sha="$(sha256sum "${REPO_ROOT}/src/grounding_evaluator.py" | awk '{print $1}')"
if [ "${evaluator_sha}" != "${ORIGINAL_SHA}" ]; then
  printf '%s refused; evaluator_sha=%s\n' "$(date -Is)" "${evaluator_sha}" > "${STATUS_FILE}"
  exit 131
fi

"${PYTHON}" - "${WINNER}" "${WINNER_RECEIPT}" "${FINAL_RECEIPT}" "${HANDOFF}" <<'PY'
import datetime
import hashlib
import json
import os
import sys

winner, receipt_path, final_path, handoff_path = sys.argv[1:]
with open(receipt_path) as f:
    source = json.load(f)
if not bool(source.get('goal_achieved')):
    raise SystemExit('refusing to finalize a non-achieved receipt')
checkpoint = os.path.realpath(source['checkpoint'])
if not os.path.isfile(checkpoint):
    raise SystemExit('winner checkpoint missing: ' + checkpoint)
sha = hashlib.sha256()
with open(checkpoint, 'rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        sha.update(block)
actual_sha = sha.hexdigest()
if actual_sha != source['checkpoint_sha256']:
    raise SystemExit('winner checkpoint SHA mismatch')
acc025 = float(source['overall_acc0.25'])
acc050 = float(source['overall_acc0.50'])
if not (acc025 > 0.5391 and acc050 > 0.4241):
    raise SystemExit('winner metrics fail strict thresholds')
payload = dict(source)
payload.update({
    'winner_stage': winner,
    'completion_audit': 'independent_full_reload_strict_dual_threshold',
    'selection_policy': 'dual_threshold_then_maximize_overall_acc0.25',
    'finalized_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
})
os.makedirs(os.path.dirname(final_path), exist_ok=True)
tmp = final_path + '.tmp'
with open(tmp, 'w') as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write('\n')
os.replace(tmp, final_path)
marker = '<!-- BUTD_ACC50_FINAL_' + actual_sha + ' -->'
with open(handoff_path, encoding='utf-8') as f:
    handoff = f.read()
if marker not in handoff:
    with open(handoff_path, 'a', encoding='utf-8') as f:
        f.write('\n\n' + marker + '\n')
        f.write('### BUTD Acc@0.50 target achieved\n\n')
        f.write('- Winner stage: `{}`.\n'.format(winner))
        f.write('- Reloaded Overall Acc@0.25 / Acc@0.50: `{:.4f}/{:.4f}`.\n'.format(acc025 * 100.0, acc050 * 100.0))
        f.write('- Checkpoint: `{}`.\n'.format(checkpoint))
        f.write('- SHA256: `{}`.\n'.format(actual_sha))
        if 'rapf_quality_weight' in source:
            f.write(
                '- RAPF quality weight: `{:.4f}`.\n'.format(
                    float(source['rapf_quality_weight'])
                )
            )
        f.write('- Audit: independent full reload, both strict thresholds PASS; selector preserves Acc@0.25 after feasibility.\n')
print(json.dumps(payload, indent=2, sort_keys=True))
PY

if [ "${WINNER}" = "stage2" ] || [ "${WINNER}" = "stage3" ] || [ "${WINNER}" = "stage4" ] || [ "${WINNER}" = "stage5" ]; then
  STAGE1_CHECKPOINT="$("${PYTHON}" -c 'import json,os,sys; print(os.path.realpath(json.load(open(sys.argv[1]))["checkpoint"]))' "${STAGE1_RECEIPT}")"
  case "${STAGE1_CHECKPOINT}" in
    "${ROOT}/stage1_qahnl_iou50_universal_only"/scanrefer_spacy/*/ckpt_best_primary.pth)
      if [ -f "${STAGE1_CHECKPOINT}" ]; then
        "${PYTHON}" - "${STAGE1_CHECKPOINT}" "${CLEANUP_RECEIPT}" <<'PY'
import datetime, json, os, sys
path, receipt = sys.argv[1:]
size = os.path.getsize(path)
os.remove(path)
payload = {
    'removed_unused_checkpoint': path,
    'removed_bytes': size,
    'reason': 'Stage-2 independently verified winner supersedes failed Stage-1 checkpoint',
    'removed_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
}
tmp = receipt + '.tmp'
with open(tmp, 'w') as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write('\n')
os.replace(tmp, receipt)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
      fi
      ;;
    *)
      echo "ERROR: refusing cleanup outside exact Stage-1 root: ${STAGE1_CHECKPOINT}" >&2
      exit 132
      ;;
  esac
fi

parent_state="$(awk '/^State:/ {print $2}' "/proc/${QUEUE_PARENT_PID}/status" 2>/dev/null || true)"
if [ "${parent_state}" = "T" ]; then
  kill -CONT "${QUEUE_PARENT_PID}"
fi
printf '%s finalized winner=%s; ablation_queue_resumed=1\n' "$(date -Is)" "${WINNER}" > "${STATUS_FILE}"
