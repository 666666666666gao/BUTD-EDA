#!/usr/bin/env bash
set -euo pipefail

ROW="${1:?usage: run_row.sh ROW [extra train args...]}"
shift

FULL="--use_structured_slots --use_sacr --use_rapf --use_reliability_gate --use_quality_head --rapf_use_quality --use_qahnl --qahnl_score_source fused --eval_use_fused_scores --rapf_quality_weight 0.75 --rapf_struct_residual_clip 0.25 --rapf_gate_loss_weight 0.1 --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1"
NO_QAHNL="--use_structured_slots --use_sacr --sacr_rank_loss_weight 0.2 --use_rapf --use_reliability_gate --use_quality_head --rapf_use_quality --eval_use_fused_scores --rapf_quality_weight 0.75 --rapf_struct_residual_clip 0.25 --rapf_gate_loss_weight 0.1 --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1"
FIXED_FUSION="--use_structured_slots --use_sacr --use_rapf --use_quality_head --rapf_use_quality --use_qahnl --qahnl_score_source fused --eval_use_fused_scores --rapf_quality_weight 0.75 --rapf_struct_residual_clip 0.25 --rapf_gate_loss_weight 0 --rapf_initial_gate_bias -2.1972246 --rapf_generic_gate_cap 0.1 --rapf_fixed_alpha 0.1"
NO_QUALITY="--use_structured_slots --use_sacr --use_rapf --use_reliability_gate --use_qahnl --qahnl_score_source fused --eval_use_fused_scores --rapf_struct_residual_clip 0.25 --rapf_gate_loss_weight 0.1 --rapf_initial_gate_bias -2.5 --rapf_generic_gate_cap 0.1"

case "${ROW}" in
  M1)
    ABLATION_ID=08_sacr_only
    ABLATION_FLAGS="--use_structured_slots --use_sacr --sacr_rank_loss_weight 0.2 --eval_use_structured_scores"
    ;;
  M2)
    ABLATION_ID=04_sacr_rapf_no_qahnl
    ABLATION_FLAGS="${NO_QAHNL}"
    ;;
  S0)
    ABLATION_ID=11_sacr_no_target_attribute
    ABLATION_FLAGS="${FULL} --sacr_disable_target_attr"
    ;;
  S1)
    ABLATION_ID=07_no_relation_anchor
    ABLATION_FLAGS="${FULL} --sacr_disable_relation"
    ;;
  S2)
    ABLATION_ID=12_sacr_no_pairwise_geometry
    ABLATION_FLAGS="${FULL} --sacr_geo_dim 0"
    ;;
  S3)
    ABLATION_ID=13_sacr_hard_top1_anchor
    ABLATION_FLAGS="${FULL} --sacr_anchor_aggregation hard"
    ;;
  R0)
    ABLATION_ID=15_rapf_fixed_fusion_g01
    ABLATION_FLAGS="${FIXED_FUSION}"
    ;;
  R1)
    ABLATION_ID=05_rapf_no_query_quality
    ABLATION_FLAGS="${NO_QUALITY}"
    ;;
  R2)
    ABLATION_ID=17_rapf_no_parser_anchor_cues
    ABLATION_FLAGS="${FULL} --rapf_disable_parser_anchor_cues"
    ;;
  R3)
    ABLATION_ID=06_rapf_no_gate_supervision
    ABLATION_FLAGS="${FULL/--rapf_gate_loss_weight 0.1/--rapf_gate_loss_weight 0}"
    ;;
  *)
    echo "ERROR: ${ROW} is outside the ten trainable three-table rows." >&2
    exit 2
    ;;
esac

export ABLATION_ID ABLATION_FLAGS
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh" "$@"
