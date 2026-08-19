#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/gb/new butd/butd_detr-main"
PACKAGE_ROOT="${REPO_ROOT}/experiment_packages/butd_same_module_optimization_20260819"
STATE_ROOT="${PACKAGE_ROOT}/state"
DATA_ROOT="/root/autodl-tmp/DATA_ROOT"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"

export PATH="/root/miniconda3/envs/bdetr/bin:${PATH}"
export LD_LIBRARY_PATH="/root/miniconda3/envs/bdetr/lib/python3.7/site-packages/torch/lib:/root/miniconda3/envs/bdetr/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/pointnet2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

POLL_SECONDS="${POLL_SECONDS:-120}"
mkdir -p "${STATE_ROOT}"
cd "${REPO_ROOT}"

timestamp() {
  date -Is
}

gpu_used_mib() {
  nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
}

wait_gpu_idle() {
  local used
  while true; do
    used="$(gpu_used_mib)"
    if [ "${used}" -lt 500 ]; then
      echo "[$(timestamp)] GPU 0 is idle (${used} MiB)."
      return 0
    fi
    echo "[$(timestamp)] GPU 0 still busy (${used} MiB); next check in ${POLL_SECONDS}s."
    sleep "${POLL_SECONDS}"
  done
}

latest_run_dir() {
  local root="$1"
  "${PYTHON}" - "$root" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
configs = list(root.rglob('config.json')) if root.exists() else []
if not configs:
    raise SystemExit(1)
print(max(configs, key=lambda p: p.stat().st_mtime).parent)
PY
}

assert_checkpoint() {
  local checkpoint="$1"
  local min_epoch="${2:-0}"
  "${PYTHON}" - "$checkpoint" "$min_epoch" <<'PY'
import sys, torch
path, min_epoch = sys.argv[1], int(sys.argv[2])
obj = torch.load(path, map_location='cpu')
for key in ('model', 'epoch'):
    if key not in obj:
        raise SystemExit('missing checkpoint key: ' + key)
if int(obj['epoch']) < min_epoch:
    raise SystemExit('checkpoint epoch {} < {}'.format(obj['epoch'], min_epoch))
if not obj['model']:
    raise SystemExit('empty model state')
print('checkpoint_ok epoch={} tensors={}'.format(obj['epoch'], len(obj['model'])))
PY
}

assert_three_module_boundary() {
  local checkpoint="$1"
  local output_json="$2"
  "${PYTHON}" - "$checkpoint" "$output_json" <<'PY'
import json, sys, torch
from pathlib import Path
checkpoint, output = sys.argv[1], Path(sys.argv[2])
obj = torch.load(checkpoint, map_location='cpu')
keys = list(obj['model'])
required = {
    'SACR': any('sacr_head' in k or 'structured_slot_builder' in k for k in keys),
    'RAPF': any('reliability_fusion' in k for k in keys),
    'QAHNL_support': any('quality_head' in k for k in keys),
}
forbidden_tokens = (
    'source_pool_selector', 'detector_policy_adapter', 'semantic_support_adapter',
    'source_choice', 'selector_choice',
)
forbidden = [k for k in keys if any(token in k.lower() for token in forbidden_tokens)]
payload = {
    'checkpoint': checkpoint,
    'epoch': int(obj.get('epoch', -1)),
    'tensor_count': len(keys),
    'required_modules_present': required,
    'forbidden_extra_tensor_count': len(forbidden),
    'forbidden_extra_tensors': forbidden[:50],
    'boundary_pass': all(required.values()) and not forbidden,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
if not payload['boundary_pass']:
    raise SystemExit('three-module boundary audit failed: {}'.format(payload))
print(json.dumps(payload, sort_keys=True))
PY
}

safe_remove_transient_weight() {
  local path="$1"
  case "${path}" in
    "${STATE_ROOT}"/*.pth|"${REPO_ROOT}"/logs/butd_universal_target/main_results_20260819/*/ckpt_epoch_last.pth)
      [ -f "${path}" ] && rm -f -- "${path}"
      ;;
    *)
      echo "Refusing to remove unexpected weight path: ${path}" >&2
      return 1
      ;;
  esac
}

