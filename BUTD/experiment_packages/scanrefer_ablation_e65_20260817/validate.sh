#!/bin/bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mapfile -t SCRIPTS < <(find "${PACKAGE_ROOT}/launch" -maxdepth 1 -type f -name '*.sh' | sort)
if [ "${#SCRIPTS[@]}" -ne 9 ]; then
  echo "ERROR: expected 9 trainable ablation launchers, found ${#SCRIPTS[@]}" >&2
  exit 70
fi

bash -n "${PACKAGE_ROOT}/common.sh" "${PACKAGE_ROOT}/register_external_baseline.sh" \
  "${PACKAGE_ROOT}/run_all_serial.sh" "${PACKAGE_ROOT}/start_all_in_screen.sh" "${PACKAGE_ROOT}/validate.sh" "${SCRIPTS[@]}"

for script in "${SCRIPTS[@]}"; do
  output="$(DRY_RUN=1 bash "${script}")"
  grep -q -- '--pp_checkpoint /root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth' <<< "${output}"
  grep -q -- '--rng_seed 0' <<< "${output}"
  grep -q -- '--dataset scanrefer_spacy' <<< "${output}"
  grep -q -- '--test_dataset scanrefer_spacy' <<< "${output}"
  grep -q -- '--max_epoch 65' <<< "${output}"
  grep -q -- '--val_freq 5' <<< "${output}"
  grep -q -- '--lr_decay_epochs 55 60' <<< "${output}"
  grep -q -- '--best_checkpoint_only' <<< "${output}"
  grep -q -- '--best_checkpoint_metric last__bbs_acc0.25_top1' <<< "${output}"
  if grep -q -- '--checkpoint_path' <<< "${output}"; then
    echo "ERROR: resume checkpoint found in ${script}" >&2
    exit 71
  fi
  if grep -q -- '--early_stopping' <<< "${output}"; then
    echo "ERROR: early stopping found in ${script}" >&2
    exit 72
  fi
done

for script in "${SCRIPTS[@]}"; do
  if grep -q -- '--use_rapf' "${script}" && ! grep -q -- '--use_sacr' "${script}"; then
    echo "ERROR: invalid RAPF-without-SACR launcher: ${script}" >&2
    exit 73
  fi
done

echo "PACKAGE_PREFLIGHT_PASS: 9 launchers; fixed E65/LR55/LR60; fresh official initialization; no early stopping; strict-best O@0.25 retention."

