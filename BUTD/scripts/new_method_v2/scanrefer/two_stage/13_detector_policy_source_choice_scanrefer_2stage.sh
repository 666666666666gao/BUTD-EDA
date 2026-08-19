#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
BASE_SCRIPT="${SCRIPT_DIR}/12_full_quality_primary_scanrefer_2stage.sh"

PROTECTED_CKPT=${PROTECTED_CKPT:-${REPO_ROOT}/logs/new_method_v2/scanrefer/two_stage/05_full_sacr_rapf_qahnl_candidateA_trial32_full/scanrefer_spacy/1777820240/ckpt_epoch_85.pth}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export DIAG=${DIAG:-1}
export NMV2_BATCH_SIZE=${NMV2_BATCH_SIZE:-24}
export NMV2_MAX_EPOCH=${NMV2_MAX_EPOCH:-86}
export NMV2_VAL_FREQ=${NMV2_VAL_FREQ:-1}
export NMV2_SAVE_FREQ=${NMV2_SAVE_FREQ:-1}
export NMV2_PRINT_FREQ=${NMV2_PRINT_FREQ:-500}
export NMV2_LOG_ROOT=${NMV2_LOG_ROOT:-/root/autodl-tmp/butd_logs/new_method_v2}

"${BASE_SCRIPT}" \
  --checkpoint_path "${PROTECTED_CKPT}" \
  --log_dir "${NMV2_LOG_ROOT}/scanrefer/two_stage/187_detector_policy_source_choice_epoch85_to_86" \
  --use_source_pool_selector \
  --source_pool_selector_direct_choice \
  --source_pool_selector_train_only \
  --source_pool_selector_lr "${SOURCE_POOL_SELECTOR_LR:-0.001}" \
  --source_pool_selector_loss_weight "${SOURCE_POOL_SELECTOR_WEIGHT:-1.0}" \
  --source_pool_selector_source source_choice \
  --source_pool_selector_choice_target "${SOURCE_POOL_SELECTOR_CHOICE_TARGET:-threshold_utility}" \
  --source_pool_selector_pairwise_weight "${SOURCE_POOL_SELECTOR_PAIRWISE_WEIGHT:-0.1}" \
  --source_pool_selector_choice_balance \
  --source_pool_selector_oracle_prior_weight "${SOURCE_POOL_SELECTOR_ORACLE_PRIOR_WEIGHT:-0.25}" \
  --source_pool_selector_include_contrastive_choice \
  --source_pool_selector_include_detector_policy_choice \
  --source_pool_selector_rank_features \
  --source_pool_selector_pairdelta_features \
  --source_pool_selector_text_context \
  --source_pool_selector_metadata_context \
  --source_pool_selector_context_features \
  --eval_use_selector_choice_scores \
  "$@"
