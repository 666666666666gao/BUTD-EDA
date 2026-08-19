#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/gb/new butd/butd_detr-main"
cd "${REPO_ROOT}"
export PATH="/root/miniconda3/envs/bdetr/bin:${PATH}"
export LD_LIBRARY_PATH="/root/miniconda3/envs/bdetr/lib/python3.7/site-packages/torch/lib:/root/miniconda3/envs/bdetr/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/pointnet2"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export DATA_ROOT="/root/autodl-tmp/DATA_ROOT"
export PP_CHECKPOINT="${DATA_ROOT}/gf_detector_l6o256.pth"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ABLATION_LOG_ROOT="${REPO_ROOT}/logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_from_official_init"
export NMV2_BATCH_SIZE=24 NMV2_MAX_EPOCH=100 NMV2_VAL_FREQ=5 NMV2_PRINT_FREQ=100 NMV2_LR_DECAY_EPOCHS=65
export NMV2_EARLY_STOP_MIN_EPOCH=35 NMV2_EARLY_STOP_PATIENCE=4 NMV2_EARLY_STOP_MIN_DELTA=0.001

ORIGINAL_QUEUE="${REPO_ROOT}/logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_queue"
QUEUE_ROOT="${REPO_ROOT}/logs/butd_universal_target/scanrefer_ablation_extension_20260815_queue"
STATUS_DIR="${QUEUE_ROOT}/status"
LAUNCH_LOG_DIR="${QUEUE_ROOT}/launcher_logs"
SUMMARY_TSV="${QUEUE_ROOT}/summary.tsv"
MANIFEST="${QUEUE_ROOT}/manifest.tsv"
ROW5="${REPO_ROOT}/scripts/ablations/scanrefer_20260814/05_no_quality_scanrefer_20260814.sh"
ROW5_ORIGINAL="${ROW5}.original_20260815"
ROW5_ORIGINAL_SHA="89af902e547d11f81e82cb10b98c30243b7025ee0e31aacae490e5954135cc66"
MODULE_GATE="${QUEUE_ROOT}/MODULE_PRIORITY_PASS"
mkdir -p "${STATUS_DIR}" "${LAUNCH_LOG_DIR}"

cat > "${MANIFEST}" <<'EOF'
08_sacr_only	scripts/ablations/scanrefer_20260815_extension/08_sacr_only_scanrefer_20260815.sh
09_sacr_qahnl	scripts/ablations/scanrefer_20260815_extension/09_sacr_qahnl_scanrefer_20260815.sh
10_full_qahnl_base_source	scripts/ablations/scanrefer_20260815_extension/10_full_qahnl_base_source_scanrefer_20260815.sh
EOF

if [ ! -f "${SUMMARY_TSV}" ]; then
  printf 'job_id\tstatus\tstart_time\tend_time\trun_dir\tmetric\tbest_checkpoint\tsha256\n' > "${SUMMARY_TSV}"
fi

sha256sum "${PP_CHECKPOINT}" > "${QUEUE_ROOT}/official_init.sha256"
sha256sum \
  main_utils.py src/grounding_evaluator.py models/bdetr.py models/losses.py models/reliability_fusion.py \
  scripts/ablations/scanrefer_20260814/scanrefer_ablation_common_20260814.sh \
  scripts/ablations/scanrefer_20260815_extension/*.sh \
  scripts/run_scanrefer_ablation_extension_queue_20260815.sh \
  tests/test_scanrefer_ablation_extension_launchers.py tests/test_optional_module_rng_isolation.py tests/test_validation_early_stopping.py \
  tools/audit_scanrefer_ablation_extension_completion.py \
  tools/audit_scanrefer_ablation_master_completion.py \
  tools/render_scanrefer_ablation_paper_table.py \
  > "${QUEUE_ROOT}/code_and_launchers.sha256"
printf '%s  %s\n' "${ROW5_ORIGINAL_SHA}" "${ROW5_ORIGINAL}" > "${QUEUE_ROOT}/row5_original.sha256"
env | sort > "${QUEUE_ROOT}/launcher_env.txt"
/root/miniconda3/envs/bdetr/bin/python -m pytest -q \
  tests/test_scanrefer_ablation_extension_launchers.py \
  tests/test_optional_module_rng_isolation.py \
  tests/test_validation_early_stopping.py \
  > "${QUEUE_ROOT}/preflight_pytest.log" 2>&1

while IFS=$'\t' read -r job_id script_path; do
  DRY_RUN=1 bash "${script_path}" > "${LAUNCH_LOG_DIR}/${job_id}.dryrun.txt"
  grep -q -- "--pp_checkpoint ${PP_CHECKPOINT}" "${LAUNCH_LOG_DIR}/${job_id}.dryrun.txt"
  grep -q -- '--rng_seed 0' "${LAUNCH_LOG_DIR}/${job_id}.dryrun.txt"
  grep -q -- '--max_epoch 100' "${LAUNCH_LOG_DIR}/${job_id}.dryrun.txt"
  grep -q -- '--best_checkpoint_only' "${LAUNCH_LOG_DIR}/${job_id}.dryrun.txt"
  grep -q -- '--early_stopping' "${LAUNCH_LOG_DIR}/${job_id}.dryrun.txt"
  grep -q -- '--early_stopping_metric last__bbs_acc0.25_top1' "${LAUNCH_LOG_DIR}/${job_id}.dryrun.txt"
  grep -q -- '--early_stopping_min_epoch 35' "${LAUNCH_LOG_DIR}/${job_id}.dryrun.txt"
  grep -q -- '--early_stopping_patience 4' "${LAUNCH_LOG_DIR}/${job_id}.dryrun.txt"
  grep -q -- '--early_stopping_min_delta 0.001' "${LAUNCH_LOG_DIR}/${job_id}.dryrun.txt"
  if grep -q -- '--checkpoint_path' "${LAUNCH_LOG_DIR}/${job_id}.dryrun.txt"; then
    echo "ERROR: ${job_id} contains --checkpoint_path" >&2; exit 2
  fi
done < "${MANIFEST}"
sha256sum -c "${QUEUE_ROOT}/row5_original.sha256"
echo "PREFLIGHT_PASS $(date -Is)"

record_success() {
  local job_id="$1" started="$2" ended
  ended=$(date -Is)
  /root/miniconda3/envs/bdetr/bin/python - "${job_id}" "${started}" "${ended}" "${ABLATION_LOG_ROOT}" "${SUMMARY_TSV}" <<'PY'
import csv, hashlib, json, os, sys
from pathlib import Path
job_id, started, ended, root, summary = sys.argv[1:]
receipts = sorted((Path(root) / job_id / 'scanrefer_spacy').glob('*/best_primary.json'), key=lambda p: p.stat().st_mtime)
if len(receipts) != 1:
    raise SystemExit('expected one best receipt for {}, got {}'.format(job_id, len(receipts)))
receipt = receipts[0]; data = json.loads(receipt.read_text()); ckpt = Path(data['checkpoint'])
if not ckpt.is_file(): raise SystemExit('missing checkpoint {}'.format(ckpt))
cfg = json.loads((receipt.parent / 'config.json').read_text())
assert cfg.get('checkpoint_path') is None
assert cfg.get('pp_checkpoint') == '/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth'
assert cfg.get('rng_seed') == 0
assert (receipt.parent / 'eval_epoch_last.log').is_file()
assert (receipt.parent / 'early_stopping.json').is_file()
h = hashlib.sha256()
with ckpt.open('rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
digest = h.hexdigest()
tmp = receipt.parent / ('ckpt_best_primary.sha256.tmp.{}'.format(os.getpid()))
tmp.write_text('{}  {}\n'.format(digest, ckpt.name)); os.replace(str(tmp), str(receipt.parent / 'ckpt_best_primary.sha256'))
with open(summary, 'a', newline='') as f:
    csv.writer(f, delimiter='\t').writerow([job_id, 'completed', started, ended, str(receipt.parent), '{}={}'.format(data['metric'], data['score']), str(ckpt), digest])
print('RECORDED {} {}'.format(job_id, digest))
PY
}

run_job() {
  local job_id="$1" script_path="$2" started rc
  if [ -f "${STATUS_DIR}/${job_id}.done" ]; then echo "SKIP ${job_id}"; return 0; fi
  started=$(date -Is); echo "START ${job_id} ${started}"
  set +e
  bash "${script_path}" 2>&1 | tee "${LAUNCH_LOG_DIR}/${job_id}.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "${rc}" -ne 0 ]; then
    printf '%s\tfailed\t%s\t%s\t\t\t\t\n' "${job_id}" "${started}" "$(date -Is)" >> "${SUMMARY_TSV}"
    touch "${STATUS_DIR}/${job_id}.failed"; exit "${rc}"
  fi
  record_success "${job_id}" "${started}"
  touch "${STATUS_DIR}/${job_id}.done"
}

# Phase A: wait until the original queue has completed all module-level rows.
while [ ! -f "${ORIGINAL_QUEUE}/status/04_no_qahnl.done" ]; do
  if [ -f "${ORIGINAL_QUEUE}/WATCHDOG_ALERT" ] || [ -f "${ORIGINAL_QUEUE}/FINALIZER_ALERT" ]; then
    echo "ORIGINAL_QUEUE_ALERT_BEFORE_MODULE_HANDOFF $(date -Is)" >&2; exit 10
  fi
  echo "WAITING_FOR_ORIGINAL_MODULE_ROWS $(date -Is)"
  sleep 300
done

run_job "08_sacr_only" "scripts/ablations/scanrefer_20260815_extension/08_sacr_only_scanrefer_20260815.sh"
run_job "09_sacr_qahnl" "scripts/ablations/scanrefer_20260815_extension/09_sacr_qahnl_scanrefer_20260815.sh"

# Restore the exact frozen original row-5 launcher before releasing internal ablations.
actual=$(sha256sum "${ROW5_ORIGINAL}" | awk '{print $1}')
[ "${actual}" = "${ROW5_ORIGINAL_SHA}" ]
cp -p "${ROW5_ORIGINAL}" "${ROW5}.restore.$$"
mv "${ROW5}.restore.$$" "${ROW5}"
chmod 755 "${ROW5}"
[ "$(sha256sum "${ROW5}" | awk '{print $1}')" = "${ROW5_ORIGINAL_SHA}" ]
touch "${MODULE_GATE}"
echo "MODULE_PRIORITY_6_OF_6_COMPLETED $(date -Is)"

# Phase B: original internal rows 05--07 run next; row 10 is the final QAHNL-internal test.
while [ ! -f "${ORIGINAL_QUEUE}/COMPLETION_AUDIT_PASS" ]; do
  if [ -f "${ORIGINAL_QUEUE}/WATCHDOG_ALERT" ] || [ -f "${ORIGINAL_QUEUE}/FINALIZER_ALERT" ]; then
    echo "ORIGINAL_QUEUE_AUDIT_ALERT $(date -Is)" >&2; exit 11
  fi
  echo "WAITING_FOR_ORIGINAL_INTERNAL_ROWS $(date -Is)"
  sleep 300
done

run_job "10_full_qahnl_base_source" "scripts/ablations/scanrefer_20260815_extension/10_full_qahnl_base_source_scanrefer_20260815.sh"
/root/miniconda3/envs/bdetr/bin/python tools/audit_scanrefer_ablation_extension_completion.py > "${QUEUE_ROOT}/completion_auditor.log" 2>&1
/root/miniconda3/envs/bdetr/bin/python tools/render_scanrefer_ablation_paper_table.py >> "${QUEUE_ROOT}/paper_table.log" 2>&1
/root/miniconda3/envs/bdetr/bin/python tools/audit_scanrefer_ablation_master_completion.py > "${QUEUE_ROOT}/master_auditor.log" 2>&1
echo "ALL_SCANREFER_ABLATIONS_10_OF_10_COMPLETED $(date -Is)"
