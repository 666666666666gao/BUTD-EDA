#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/gb/new butd/butd_detr-main"
cd "${REPO_ROOT}"

export PATH="/root/miniconda3/envs/bdetr/bin:${PATH}"
export LD_LIBRARY_PATH="/root/miniconda3/envs/bdetr/lib/python3.7/site-packages/torch/lib:/root/miniconda3/envs/bdetr/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/pointnet2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export DATA_ROOT="/root/autodl-tmp/DATA_ROOT"
export PP_CHECKPOINT="${DATA_ROOT}/gf_detector_l6o256.pth"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ABLATION_LOG_ROOT="${REPO_ROOT}/logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_from_official_init"
export NMV2_BATCH_SIZE="${NMV2_BATCH_SIZE:-24}"
export NMV2_MAX_EPOCH="${NMV2_MAX_EPOCH:-100}"
export NMV2_VAL_FREQ="${NMV2_VAL_FREQ:-5}"
export NMV2_PRINT_FREQ="${NMV2_PRINT_FREQ:-100}"
export NMV2_LR_DECAY_EPOCHS="${NMV2_LR_DECAY_EPOCHS:-65}"

QUEUE_ROOT="${REPO_ROOT}/logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_queue"
STATUS_DIR="${QUEUE_ROOT}/status"
LAUNCH_LOG_DIR="${QUEUE_ROOT}/launcher_logs"
SUMMARY_TSV="${QUEUE_ROOT}/summary.tsv"
MANIFEST="${QUEUE_ROOT}/manifest.tsv"
mkdir -p "${STATUS_DIR}" "${LAUNCH_LOG_DIR}"

if [ ! -f "${SUMMARY_TSV}" ]; then
  printf 'job_id\tstatus\tstart_time\tend_time\trun_dir\tmetric\tbest_checkpoint\tsha256\n' > "${SUMMARY_TSV}"
fi

cat > "${MANIFEST}" <<'MANIFEST_EOF'
01_baseline	scripts/ablations/scanrefer_20260814/01_baseline_scanrefer_20260814.sh
02_full_sacr_rapf_qahnl	scripts/ablations/scanrefer_20260814/02_full_scanrefer_20260814.sh
03_no_sacr_rapf_qahnl_base	scripts/ablations/scanrefer_20260814/03_no_sacr_rapf_scanrefer_20260814.sh
04_no_qahnl	scripts/ablations/scanrefer_20260814/04_no_qahnl_scanrefer_20260814.sh
05_no_quality	scripts/ablations/scanrefer_20260814/05_no_quality_scanrefer_20260814.sh
06_no_gate_supervision	scripts/ablations/scanrefer_20260814/06_no_gate_supervision_scanrefer_20260814.sh
07_no_relation	scripts/ablations/scanrefer_20260814/07_no_relation_scanrefer_20260814.sh
MANIFEST_EOF

validate_dry_run() {
  local job_id="$1"
  local script_path="$2"
  local dry_log="${LAUNCH_LOG_DIR}/${job_id}.dryrun.txt"
  DRY_RUN=1 bash "${script_path}" > "${dry_log}"
  if grep -q -- '--checkpoint_path' "${dry_log}"; then
    echo "ERROR: ${job_id} unexpectedly contains --checkpoint_path" >&2
    return 2
  fi
  grep -q -- "--pp_checkpoint ${PP_CHECKPOINT}" "${dry_log}"
  grep -q -- '--rng_seed 0' "${dry_log}"
  grep -q -- '--best_checkpoint_only' "${dry_log}"
  grep -q -- '--best_checkpoint_metric last__bbs_acc0.25_top1' "${dry_log}"
}

record_success() {
  local job_id="$1"
  local start_time="$2"
  local end_time
  end_time=$(date -Is)
  /root/miniconda3/envs/bdetr/bin/python - "${job_id}" "${start_time}" "${end_time}" "${ABLATION_LOG_ROOT}" "${SUMMARY_TSV}" <<'PY'
import csv, hashlib, json, sys
from pathlib import Path

job_id, started, ended, log_root, summary_path = sys.argv[1:]
parent = Path(log_root) / job_id / 'scanrefer_spacy'
receipts = sorted(parent.glob('*/best_primary.json'), key=lambda p: p.stat().st_mtime)
if not receipts:
    raise SystemExit('missing best_primary.json under {}'.format(parent))
receipt = receipts[-1]
data = json.loads(receipt.read_text())
ckpt = Path(data['checkpoint'])
if not ckpt.is_file():
    raise SystemExit('missing checkpoint {}'.format(ckpt))
config = receipt.parent / 'config.json'
final_eval = receipt.parent / 'eval_epoch_last.log'
if not config.is_file() or not final_eval.is_file():
    raise SystemExit('missing config or final best-model evaluation')
cfg = json.loads(config.read_text())
assert cfg.get('checkpoint_path') is None
assert cfg.get('pp_checkpoint') == '/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth'
assert cfg.get('rng_seed') == 0
sha = hashlib.sha256()
with ckpt.open('rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        sha.update(chunk)
metric = '{}={}'.format(data['metric'], data['score'])
with open(summary_path, 'a', newline='') as f:
    csv.writer(f, delimiter='\t').writerow(
        [job_id, 'completed', started, ended, str(receipt.parent), metric, str(ckpt), sha.hexdigest()]
    )
print('RECORDED {} {} {}'.format(job_id, metric, sha.hexdigest()))
PY
}

sha256sum "${PP_CHECKPOINT}" > "${QUEUE_ROOT}/official_init.sha256"
sha256sum main_utils.py src/grounding_evaluator.py models/bdetr.py models/losses.py models/reliability_fusion.py scripts/ablations/scanrefer_20260814/*.sh scripts/run_scanrefer_ablation_retrain_queue_20260814_v2.sh tests/test_scanrefer_ablation_launchers.py tests/test_bbs_subset_metrics.py tests/test_best_checkpoint_final_reload.py tests/test_optional_module_rng_isolation.py tools/best_checkpoint_reload_smoke.py tools/model_init_parity_scanrefer.py > "${QUEUE_ROOT}/code_and_launchers.sha256"
env | sort > "${QUEUE_ROOT}/launcher_env.txt"
/root/miniconda3/envs/bdetr/bin/python -m pytest -q \
  tests/test_scanrefer_ablation_launchers.py \
  tests/test_bbs_subset_metrics.py \
  tests/test_best_checkpoint_final_reload.py \
  tests/test_optional_module_rng_isolation.py \
  > "${QUEUE_ROOT}/preflight_pytest.log" 2>&1
/root/miniconda3/envs/bdetr/bin/python tools/model_init_parity_scanrefer.py \
  > "${QUEUE_ROOT}/model_init_parity.log" 2>&1
grep -q '^MODEL_INIT_PARITY_PASS ' "${QUEUE_ROOT}/model_init_parity.log"

while IFS=$'\t' read -r job_id script_path; do
  [ -n "${job_id}" ] || continue
  if [ -f "${STATUS_DIR}/${job_id}.done" ]; then
    echo "SKIP completed ${job_id}"
    continue
  fi
  validate_dry_run "${job_id}" "${script_path}"
  started=$(date -Is)
  echo "START ${job_id} ${started}"
  set +e
  bash "${script_path}" 2>&1 | tee "${LAUNCH_LOG_DIR}/${job_id}.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "${rc}" -ne 0 ]; then
    printf '%s\tfailed\t%s\t%s\t\t\t\t\n' "${job_id}" "${started}" "$(date -Is)" >> "${SUMMARY_TSV}"
    touch "${STATUS_DIR}/${job_id}.failed"
    exit "${rc}"
  fi
  record_success "${job_id}" "${started}"
  touch "${STATUS_DIR}/${job_id}.done"
done < "${MANIFEST}"

echo "ALL_SCANREFER_ABLATIONS_COMPLETED $(date -Is)"
