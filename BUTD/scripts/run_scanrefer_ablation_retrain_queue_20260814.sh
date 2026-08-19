#!/bin/bash

set -euo pipefail

REPO_ROOT="/home/gb/new butd/butd_detr-main"
cd "${REPO_ROOT}"

export PATH="/root/miniconda3/envs/bdetr/bin:${PATH}"
export LD_LIBRARY_PATH="/root/miniconda3/envs/bdetr/lib/python3.7/site-packages/torch/lib:/root/miniconda3/envs/bdetr/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/pointnet2"

export DATA_ROOT="/root/autodl-tmp/DATA_ROOT"
export PP_CHECKPOINT="${DATA_ROOT}/gf_detector_l6o256.pth"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NMV2_LOG_ROOT="./logs/butd_universal_target/scanrefer_ablation_retrain_20260814_from_official_init"
export NMV2_BATCH_SIZE="${NMV2_BATCH_SIZE:-24}"
export NMV2_MAX_EPOCH="${NMV2_MAX_EPOCH:-100}"
export NMV2_VAL_FREQ="${NMV2_VAL_FREQ:-5}"
export NMV2_SAVE_FREQ="${NMV2_SAVE_FREQ:-1000}"
export NMV2_PRINT_FREQ="${NMV2_PRINT_FREQ:-1000}"
export NMV2_LR_DECAY_EPOCHS="${NMV2_LR_DECAY_EPOCHS:-65}"
export DIAG="${DIAG:-1}"

COMMON_EXTRA_ARGS=(
  --rng_seed 0
  --best_checkpoint_only
  --best_checkpoint_metric last__bbs_acc0.25_top1
  --best_checkpoint_min_delta 0
)
export EXTRA_ARGS="${COMMON_EXTRA_ARGS[*]}"

QUEUE_ROOT="${REPO_ROOT}/logs/butd_universal_target/scanrefer_ablation_retrain_20260814_from_official_init_queue"
STATUS_DIR="${QUEUE_ROOT}/status"
LAUNCH_LOG_DIR="${QUEUE_ROOT}/launcher_logs"
SUMMARY_TSV="${QUEUE_ROOT}/summary.tsv"
MANIFEST="${QUEUE_ROOT}/manifest.tsv"
mkdir -p "${STATUS_DIR}" "${LAUNCH_LOG_DIR}"

if [ ! -f "${SUMMARY_TSV}" ]; then
  printf 'job_id\tstatus\tstart_time\tend_time\trun_dir\tmetric\tbest_checkpoint\tsha256\n' > "${SUMMARY_TSV}"
fi

cat > "${MANIFEST}" <<'MANIFEST_EOF'
01_baseline	scripts/new_method_v2/scanrefer/two_stage/01_baseline_scanrefer_2stage.sh	Official BUTD-DETR detector-only two-stage baseline; no SACR/RAPF/QAHNL.
05_full_sacr_rapf_qahnl	scripts/new_method_v2/scanrefer/two_stage/05_full_sacr_rapf_qahnl_scanrefer_2stage.sh	Full paper-facing SACR+RAPF+QAHNL with fused primary.
08_full_no_qahnl	scripts/new_method_v2/scanrefer/two_stage/08_full_no_qahnl_scanrefer_2stage.sh	Remove QAHNL while keeping SACR+RAPF+quality fused primary.
07_full_no_quality	scripts/new_method_v2/scanrefer/two_stage/07_full_no_quality_scanrefer_2stage.sh	Remove quality head / quality path from full while keeping SACR+RAPF+QAHNL fused primary.
06_full_no_gate_supervision	scripts/new_method_v2/scanrefer/two_stage/06_full_no_gate_supervision_scanrefer_2stage.sh	Remove RAPF gate supervision while keeping the full architecture.
03_sacr_only	scripts/new_method_v2/scanrefer/two_stage/03_sacr_only_scanrefer_2stage.sh	Use direct SACR structured scores without RAPF fusion; tests whether reliability fusion is necessary.
09_sacr_no_relation	scripts/new_method_v2/scanrefer/two_stage/09_sacr_no_relation_scanrefer_2stage.sh	Disable SACR relation-anchor branch; tests whether relation modeling matters.
MANIFEST_EOF

record_failure() {
  local job_id="$1"
  local start_time="$2"
  local end_time
  end_time=$(date -Is)
  printf '%s\tfailed\t%s\t%s\t\t\t\t\n' "${job_id}" "${start_time}" "${end_time}" >> "${SUMMARY_TSV}"
  touch "${STATUS_DIR}/${job_id}.failed"
}

parse_and_record_success() {
  local job_id="$1"
  local start_time="$2"
  local script_path="$3"
  local end_time
  end_time=$(date -Is)
  /root/miniconda3/envs/bdetr/bin/python - "${job_id}" "${start_time}" "${end_time}" "${script_path}" "${NMV2_LOG_ROOT}" "${SUMMARY_TSV}" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

job_id, start_time, end_time, script_path, log_root, summary_tsv = sys.argv[1:]
root = Path(log_root) / "scanrefer" / "two_stage"
slug = Path(script_path).name
slug = slug.replace("_scanrefer_2stage.sh", "")
slug = slug[3:] if slug[:2].isdigit() and slug[2] == "_" else slug
run_parent = root / slug / "scanrefer_spacy"
best_files = sorted(run_parent.glob("*/best_primary.json"), key=lambda p: p.stat().st_mtime)
if not best_files:
    raise SystemExit(f"missing best_primary.json under {run_parent}")
best_json = best_files[-1]
data = json.loads(best_json.read_text())
ckpt = Path(data["checkpoint"])
if not ckpt.is_file():
    raise SystemExit(f"missing checkpoint {ckpt}")
sha = hashlib.sha256()
with ckpt.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        sha.update(chunk)
metric = f'{data.get("metric")}={data.get("score")}'
with open(summary_tsv, "a", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow([job_id, "completed", start_time, end_time, str(best_json.parent), metric, str(ckpt), sha.hexdigest()])
print(f"RECORDED {job_id} {metric} {ckpt} {sha.hexdigest()}")
PY
  touch "${STATUS_DIR}/${job_id}.done"
}

validate_dry_run() {
  local job_id="$1"
  local script_path="$2"
  local dry_log="${LAUNCH_LOG_DIR}/${job_id}.dryrun.txt"
  bash "${script_path}" --dry-run > "${dry_log}"
  if grep -q -- "--checkpoint_path" "${dry_log}"; then
    echo "ERROR: ${job_id} dry-run unexpectedly contains --checkpoint_path" >&2
    exit 2
  fi
  if ! grep -q -- "--pp_checkpoint ${PP_CHECKPOINT}" "${dry_log}"; then
    echo "ERROR: ${job_id} dry-run does not use the required official pp checkpoint" >&2
    exit 2
  fi
  if ! grep -q -- "--best_checkpoint_only" "${dry_log}"; then
    echo "ERROR: ${job_id} dry-run does not enable best_checkpoint_only" >&2
    exit 2
  fi
  if ! grep -q -- "--best_checkpoint_metric last__bbs_acc0.25_top1" "${dry_log}"; then
    echo "ERROR: ${job_id} dry-run does not use the ScanRefer primary best metric" >&2
    exit 2
  fi
}

sha256sum "${PP_CHECKPOINT}" > "${QUEUE_ROOT}/official_init.sha256"
env | sort > "${QUEUE_ROOT}/launcher_env.txt"

while IFS=$'\t' read -r job_id script_path description; do
  [ -n "${job_id}" ] || continue
  if [ -f "${STATUS_DIR}/${job_id}.done" ]; then
    echo "SKIP completed ${job_id}"
    continue
  fi
  if [ -f "${STATUS_DIR}/${job_id}.failed" ]; then
    echo "STOP previously failed ${job_id}; remove ${STATUS_DIR}/${job_id}.failed after inspection to retry" >&2
    exit 3
  fi
  if [ ! -x "${script_path}" ]; then
    echo "ERROR: missing or non-executable script ${script_path}" >&2
    exit 2
  fi

  validate_dry_run "${job_id}" "${script_path}"
  start_time=$(date -Is)
  echo "START ${job_id} ${start_time} ${description}"
  printf '%s\trunning\t%s\t\t\t\t\t\n' "${job_id}" "${start_time}" >> "${SUMMARY_TSV}"
  set +e
  bash "${script_path}" 2>&1 | tee "${LAUNCH_LOG_DIR}/${job_id}.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "${rc}" -ne 0 ]; then
    echo "FAILED ${job_id} exit=${rc}" >&2
    record_failure "${job_id}" "${start_time}"
    exit "${rc}"
  fi
  parse_and_record_success "${job_id}" "${start_time}" "${script_path}"
done < "${MANIFEST}"

echo "ALL_SCANREFER_ABLATIONS_COMPLETED $(date -Is)"
