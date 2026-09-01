#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"

export RUN_ID="stage157d"
export UPSTREAM_PREFIX="stage157b"
export UPSTREAM_STATUS_OVERRIDE="${P}/stage157b_dependency_status.txt"
export FINAL_STATUS_OVERRIDE="${P}/stage157d_dependency_status.txt"
export LOCK_ROOT_OVERRIDE="${ROOT}/stage157a_all_train_refit_selector"
export SELECTOR_LOCK_OVERRIDE="${LOCK_ROOT_OVERRIDE}/locked_all_train_refit.json"
export LOCKED_VAL_ROOT_OVERRIDE="${ROOT}"
export LOCKED_VAL_RESULT_OVERRIDE="${ROOT}/stage157b_all_train_refit_selector_validation_result.json"
export BUNDLE_OVERRIDE="${ROOT}/stage157d_final_artifact_bundle"
export TEMP_ALT_CKPT_OVERRIDE="${ROOT}/stage157d_materialized_stage135c_tmp.pth"
export ALT_OUT_OVERRIDE="${ROOT}/stage157d_fresh_bundle_stage135c_raw_val"
export PRIMARY_OUT_OVERRIDE="${ROOT}/stage157d_fresh_bundle_stage150_source_val"
export FRESH_RESULT_OVERRIDE="${ROOT}/stage157d_fresh_bundle_selector_validation_result.json"
export RELOAD_RECEIPT_OVERRIDE="${ROOT}/stage157d_fresh_bundle_reload_receipt.json"
export SELECTOR_RUNTIME_OVERRIDE="${P}/stage157_all_train_refit_selector.py"
export EXTRA_SELECTOR_RUNTIME_OVERRIDE="${P}/stage154_oof_source_selector.py"
export EXTRA_SELECTOR_RUNTIME_2_OVERRIDE="${P}/stage153_train_source_selector.py"
export RECEIPT_STAGE_OVERRIDE="stage157d_fresh_bundle_reload"

exec bash "${P}/run_stage153d_final_bundle_reload_if_goal_met.sh"
