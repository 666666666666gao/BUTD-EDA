#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/home/gb/new butd/butd_detr-main"
PACKAGE="${REPO}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
SCRIPT="${PACKAGE}/stage154_subgroup_metrics.py"
TEST="${PACKAGE}/test_stage154_subgroup_metrics.py"
RAW_VAL="${ROOT}/stage136c_stage135c_raw_val_geometry_dump/stage135c_e12_raw_val_geometry.pt"
SOURCE="${ROOT}/stage154b_stage150_e13_val_source_dump/stage150_e13_val_source_features.pt"
COMPACT="${ROOT}/stage154b_stage150_e13_val_source_dump/stage150_e13_val_adapter_features.pt"
STAGE31_LOCK="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33_LOCK="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
STAGE142_LOCK="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"
POLICY_LOCK="${ROOT}/stage154a_train_only_oof_source_selector/locked_oof_source_selector.json"
VALIDATION_RESULT="${ROOT}/stage154b_oof_selector_validation_result.json"
OUTPUT="${ROOT}/stage154e_locked_subgroup_report.json"
STATUS="${PACKAGE}/stage154e_dependency_status.txt"
LOG="${PACKAGE}/stage154e_locked_subgroup_report.log"

fail_status() {
  local rc=$?
  printf 'stage154e_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" > "${STATUS}"
  exit "${rc}"
}
trap fail_status ERR INT TERM

cd "${PACKAGE}"
test "$(sha256sum "${SCRIPT}" | awk '{print $1}')" = \
  "2032659d85ff23712add211771017747fa71aec75152484f9ee12ab5c78abb59"
test "$(sha256sum "${TEST}" | awk '{print $1}')" = \
  "167367042413188ea79545e3e162e0c092b7bd7e23422fcd6b89d1e79b2dc950"
test "$(sha256sum "${VALIDATION_RESULT}" | awk '{print $1}')" = \
  "983b74dd3dfb816992ffe8a854c9428836721620d52b24c6693e2d3ab653f816"
for path in "${RAW_VAL}" "${SOURCE}" "${COMPACT}" "${STAGE31_LOCK}" \
  "${STAGE33_LOCK}" "${STAGE142_LOCK}" "${POLICY_LOCK}"; do
  test -s "${path}"
done
test ! -e "${OUTPUT}"

PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" -W error::ResourceWarning -m unittest -v \
  test_stage154_subgroup_metrics.py
printf 'stage154e_running_posthoc_subgroup_only %s\n' \
  "$(date --iso-8601=seconds)" > "${STATUS}"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
nice -n 19 ionice -c 3 "${PYTHON}" "${SCRIPT}" \
  "${RAW_VAL}" "${SOURCE}" "${COMPACT}" \
  "${STAGE31_LOCK}" "${STAGE33_LOCK}" "${STAGE142_LOCK}" \
  "${POLICY_LOCK}" "${VALIDATION_RESULT}" "${OUTPUT}" \
  2>&1 | tee "${LOG}"

test -s "${OUTPUT}"
OUTPUT_SHA="$(sha256sum "${OUTPUT}" | awk '{print $1}')"
printf 'stage154e_complete output_sha256=%s at=%s\n' \
  "${OUTPUT_SHA}" "$(date --iso-8601=seconds)" > "${STATUS}"
chmod 0444 "${OUTPUT}" "${STATUS}"
trap - ERR INT TERM
