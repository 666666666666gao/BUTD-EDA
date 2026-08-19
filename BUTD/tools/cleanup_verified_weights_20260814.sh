#!/bin/bash
set -euo pipefail

ROOT="/home/gb/new butd/butd_detr-main"
cd "${ROOT}"

REPORT="reports/tuning/weight_cleanup_20260814.tsv"
TEMP="${REPORT}.tmp.$$"
mkdir -p "$(dirname "${REPORT}")"
printf 'deleted_at\tsize_bytes\tsha256\tpath\treason\n' > "${TEMP}"

deleted_bytes=0

delete_verified() {
  local expected_sha="$1"
  local expected_size="$2"
  local relative="$3"
  local reason="$4"
  local absolute actual_sha actual_size

  absolute=$(realpath -e -- "${relative}")
  case "${absolute}" in
    "${ROOT}"/*) ;;
    *) echo "Refusing path outside repository: ${absolute}" >&2; exit 10 ;;
  esac
  [ -f "${absolute}" ] || { echo "Not a regular file: ${absolute}" >&2; exit 11; }
  actual_size=$(stat -c '%s' -- "${absolute}")
  [ "${actual_size}" = "${expected_size}" ] || {
    echo "Size mismatch: ${relative}: ${actual_size} != ${expected_size}" >&2
    exit 12
  }
  actual_sha=$(sha256sum -- "${absolute}" | awk '{print $1}')
  [ "${actual_sha}" = "${expected_sha}" ] || {
    echo "SHA256 mismatch: ${relative}: ${actual_sha} != ${expected_sha}" >&2
    exit 13
  }
  if fuser "${absolute}" >/dev/null 2>&1; then
    echo "File is open by a process: ${relative}" >&2
    exit 14
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "${actual_size}" "${actual_sha}" "${relative}" "${reason}" >> "${TEMP}"
  rm -f -- "${absolute}"
  [ ! -e "${absolute}" ] || { echo "Deletion failed: ${relative}" >&2; exit 15; }
  deleted_bytes=$((deleted_bytes + actual_size))
}

# Exact duplicate of the preserved Nr3D 46.5755% best checkpoint.
delete_verified \
  79e7dc5ce9112133b021ecfc77eaf1c81f558f3eda55f476b857217179ba05db \
  770377022 \
  logs/butd_universal_target/officialdet_finetune_e1/nr3d/03_full_sacr_rapf_qahnl/nr3d_spacy/1786638278/ckpt_epoch_1.pth \
  duplicate_of_preserved_nr3d_best

# Non-best terminal checkpoint from the same Nr3D run.
delete_verified \
  6167ae83022a577f4418b2c75488b9b7a59b7b9887bd0e965bd84485da5500a8 \
  770377022 \
  logs/butd_universal_target/officialdet_finetune_e1/nr3d/03_full_sacr_rapf_qahnl/nr3d_spacy/1786638278/ckpt_epoch_last.pth \
  nonbest_terminal_duplicate_run_state

# Superseded e0 checkpoints (official BBS 45.65% < preserved 46.5755%).
delete_verified \
  83e5954b9148821b0822cfd878000b48e37d5ec1ca3246776c5a46e731409282 \
  770374974 \
  logs/butd_universal_target/officialdet_finetune_e0/nr3d/03_full_sacr_rapf_qahnl/nr3d_spacy/1786626252/ckpt_epoch_0.pth \
  superseded_nr3d_45.65pct
delete_verified \
  26661f2f55c91c3a347afeb2172c5d50d6e8b62f16f462ef9590926d5deda4e1 \
  770375038 \
  logs/butd_universal_target/officialdet_finetune_e0/nr3d/03_full_sacr_rapf_qahnl/nr3d_spacy/1786626252/ckpt_epoch_last.pth \
  superseded_nr3d_45.65pct_terminal

# Calibration run regressed to 44.93% and is superseded by the preserved result.
delete_verified \
  23975d8ee2a2bdea6e1f03afa5b6b6c62f13e0865f60c0d2ca8469fa3bbc1b97 \
  598532272 \
  logs/butd_universal_target/universal_calibration_qw0_e2/nr3d/03_full_sacr_rapf_qahnl/nr3d_spacy/1786641214/ckpt_best_primary.pth \
  superseded_nr3d_calibration_44.93pct

# Old intermediate universal initialization; current runs use the official detector init.
delete_verified \
  a82c0f40f93ab38dcd4975bbe8d902f3b9ddfa94e80c67d31a2d1d02a0b6ba00 \
  589347392 \
  checkpoints/universal/butd_nr3d_det_to_sacr_rapf_qahnl_init.pth \
  obsolete_intermediate_initialization

# Required retained weights must still exist after cleanup.
test -f checkpoints/preserved/nr3d_sacr_rapf_qahnl_46.5755/best_official_bbs_epoch1.pth
test -f logs/new_method_v2/scanrefer/two_stage/05_full_sacr_rapf_qahnl_candidateA_trial32_full/scanrefer_spacy/1777820240/ckpt_epoch_85.pth
test -f logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_from_official_init/01_baseline/scanrefer_spacy/1786657178/ckpt_best_primary.pth
test -f EDA-master/checkpoint/ScanRefer_54_59.pth
test -f checkpoints/official/bdetr_nr3d_43.2_new.pth
test -f /root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth

mv -f -- "${TEMP}" "${REPORT}"
printf 'WEIGHT_CLEANUP_PASS deleted_files=6 deleted_bytes=%s report=%s\n' \
  "${deleted_bytes}" "${ROOT}/${REPORT}"
