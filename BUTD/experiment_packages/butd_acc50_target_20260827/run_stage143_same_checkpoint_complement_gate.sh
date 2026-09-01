#!/usr/bin/env bash
set -Eeuo pipefail

R='/home/gb/new butd/butd_detr-main'
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT=/root/autodl-tmp/logs/butd_acc50_target_20260827
PY=/root/miniconda3/envs/bdetr/bin/python
TRAIN_DIR="${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump"
TRAIN_RECEIPT="${TRAIN_DIR}/stage141_receipt.json"
TRAIN_DUMP="${TRAIN_DIR}/stage135c_e12_raw_train_geometry_with_ids.pt"
VAL_DUMP="${ROOT}/stage136c_stage135c_raw_val_geometry_dump/stage135c_e12_raw_val_geometry.pt"
STAGE31_LOCK="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33_LOCK="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
STAGE142_LOCK="${ROOT}/stage142_stage135c_same_domain_nested_blend/locked_same_domain_nested_blend_policy.json"
OUTDIR="${ROOT}/stage143_stage135c_same_checkpoint_complement_gate"
ARTIFACTS="${OUTDIR}/artifacts"
POLICY="${ARTIFACTS}/locked_same_checkpoint_gate.json"
RESULT="${OUTDIR}/stage143_on_stage135c_eval.json"
LOG="${OUTDIR}/stage143.log"

EXPECTED_CKPT_SHA=a367318ccccedfb9fb4345b03044521f67e7cb50dbc9c089c037c9f86f98de2b
EXPECTED_TRAIN_DUMP_SHA=66de6d6eb5f4b2059233c9211d514149cab073d14b0ca9252eed04434994025d
EXPECTED_VAL_SHA=6a837f903f69b0ec15f43bf0544230344352adf14303c6ea0d13a7e842825508
EXPECTED_STAGE31_LOCK_SHA=4c1be1199fe2bc62dc3e4679c4ad26af4af193db1f5dda551b8cc559620b83c9
EXPECTED_STAGE33_LOCK_SHA=da1e020bc190d9792a6df57bf83b3d8be41a7e754eb353ab350a40d11588705d
EXPECTED_STAGE142_LOCK_SHA=422dfdb9d0289ec79a3fd9fab623992c6b112be8add3e7149491f25e041c8dc3
EXPECTED_TRAINER_SHA=67b0c8ea0f0baaab57ca961bc4cd01c6f6128d21fb3b82db5e670d07e293b407
EXPECTED_STAGE140_SHA=0a17816dc5285dee56fdaab333b3818a856367c3cbd3235ef3cbd8833c86d7ff
EXPECTED_STAGE143_SHA=e22328fb409ba2f8ce876b83e89a2f8e3a51da29a20cd6d2ff367957b1dc7df1

check_sha() {
  local path="$1" expected="$2"
  test -s "${path}"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

test ! -e "${OUTDIR}"
check_sha "${TRAIN_DUMP}" "${EXPECTED_TRAIN_DUMP_SHA}"
check_sha "${VAL_DUMP}" "${EXPECTED_VAL_SHA}"
check_sha "${STAGE31_LOCK}" "${EXPECTED_STAGE31_LOCK_SHA}"
check_sha "${STAGE33_LOCK}" "${EXPECTED_STAGE33_LOCK_SHA}"
check_sha "${STAGE142_LOCK}" "${EXPECTED_STAGE142_LOCK_SHA}"
check_sha "${P}/train_joint_option_ranker.py" "${EXPECTED_TRAINER_SHA}"
check_sha "${P}/stage140_train_eval_nested_blend.py" "${EXPECTED_STAGE140_SHA}"
check_sha "${P}/stage143_same_checkpoint_complement_gate.py" "${EXPECTED_STAGE143_SHA}"
test -s "${TRAIN_RECEIPT}"

"${PY}" - "${TRAIN_RECEIPT}" "${EXPECTED_CKPT_SHA}" "${EXPECTED_TRAIN_DUMP_SHA}" <<'PY'
import json
import sys
path, checkpoint_sha, dump_sha = sys.argv[1:]
receipt = json.load(open(path, encoding='utf-8'))
assert receipt['status'] == 'complete'
assert receipt['rows'] == 36665
assert receipt['unique_stable_keys'] == 36665
assert receipt['source_checkpoint_sha256'] == checkpoint_sha
assert receipt['corrected_dump_sha256'] == dump_sha
assert receipt['validation_labels_used'] is False
PY

available_kb="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')"
test "${available_kb}" -ge 5242880
mkdir -p "${OUTDIR}"
record_exit() {
  code=$?
  printf '%s\n' "${code}" > "${OUTDIR}/exit_code.txt"
}
trap record_exit EXIT

{
  printf 'stage=143_stage135c_same_checkpoint_complement_gate\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'selection_scope=scanrefer_train_scene70_fit_scene15_dev_lock_scene15_test\n'
  printf 'validation_labels_used_for_selection=false\n'
  printf 'source_checkpoint_sha256=%s\n' "${EXPECTED_CKPT_SHA}"
  printf 'train_dump_sha256=%s\n' "${EXPECTED_TRAIN_DUMP_SHA}"
  printf 'val_dump_sha256=%s\n' "${EXPECTED_VAL_SHA}"
  printf 'stage142_lock_sha256=%s\n' "${EXPECTED_STAGE142_LOCK_SHA}"
  printf 'script_sha256=%s\n' "${EXPECTED_STAGE143_SHA}"
  printf 'gpu_visible=none\n'
} > "${OUTDIR}/launch_manifest.txt"

cd "${P}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
"${PY}" stage143_same_checkpoint_complement_gate.py train \
  "${TRAIN_DUMP}" "${STAGE31_LOCK}" "${STAGE33_LOCK}" \
  "${STAGE142_LOCK}" "${ARTIFACTS}" --num-threads 16 \
  2>&1 | tee "${LOG}"
test -s "${POLICY}"
sha256sum "${POLICY}" > "${OUTDIR}/locked_same_checkpoint_gate.json.sha256"

AUTHORIZED="$("${PY}" - "${POLICY}" <<'PY'
import json
import sys
lock = json.load(open(sys.argv[1], encoding='utf-8'))
print('true' if lock['validation_evaluation_authorized'] else 'false')
PY
)"
printf 'validation_evaluation_authorized=%s\n' "${AUTHORIZED}" >> "${OUTDIR}/launch_manifest.txt"
if [ "${AUTHORIZED}" = true ]; then
  "${PY}" stage143_same_checkpoint_complement_gate.py evaluate \
    "${VAL_DUMP}" "${POLICY}" "${RESULT}" 2>&1 | tee -a "${LOG}"
  test -s "${RESULT}"
  sha256sum "${RESULT}" > "${OUTDIR}/stage143_on_stage135c_eval.json.sha256"
else
  printf 'validation_not_consumed_internal_scene_test_gate_failed\n' | tee -a "${LOG}"
fi
printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >> "${OUTDIR}/launch_manifest.txt"
