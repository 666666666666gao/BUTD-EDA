# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------
"""Shared utilities for all main scripts."""

import argparse
import json
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from models import HungarianMatcher, SetCriterion, compute_hungarian_loss
from models.detector_policy_sources import DETECTOR_POLICY_SOURCE_NAMES
from utils import get_scheduler, setup_logger


def _deprecated_alias_warning(alias_flag, canonical_flag):
    warnings.warn(
        "{} is deprecated; use {} instead".format(alias_flag, canonical_flag),
        DeprecationWarning,
        stacklevel=3,
    )


def _resolve_deprecated_numeric_alias(
        args, canonical_attr, alias_attr, default, canonical_flag, alias_flag,
        public_alias_attr=None, is_int=False):
    canonical_value = getattr(args, canonical_attr)
    alias_value = getattr(args, alias_attr)
    if canonical_value is not None and alias_value is not None:
        if is_int:
            conflict = int(canonical_value) != int(alias_value)
        else:
            conflict = abs(float(canonical_value) - float(alias_value)) > 1e-12
        if conflict:
            raise ValueError(
                "{} and deprecated alias {} were both supplied with "
                "different values".format(canonical_flag, alias_flag)
            )
    if alias_value is not None:
        _deprecated_alias_warning(alias_flag, canonical_flag)
    value = canonical_value if canonical_value is not None else alias_value
    if value is None:
        value = default
    value = int(value) if is_int else float(value)
    setattr(args, canonical_attr, value)
    if public_alias_attr is not None:
        setattr(args, public_alias_attr, value)
    if hasattr(args, alias_attr):
        delattr(args, alias_attr)
    return value


def _resolve_deprecated_bool_alias(
        args, canonical_attr, alias_attr, canonical_flag, alias_flag,
        public_alias_attr=None):
    canonical_value = bool(getattr(args, canonical_attr))
    alias_value = bool(getattr(args, alias_attr))
    if alias_value:
        _deprecated_alias_warning(alias_flag, canonical_flag)
    value = canonical_value or alias_value
    setattr(args, canonical_attr, value)
    if public_alias_attr is not None:
        setattr(args, public_alias_attr, value)
    if hasattr(args, alias_attr):
        delattr(args, alias_attr)
    return value


def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--num_target', type=int, default=256,
                        help='Proposal number')
    parser.add_argument('--sampling', default='kps', type=str,
                        help='Query points sampling method (kps, fps)')

    # Transformer
    parser.add_argument('--num_encoder_layers', default=3, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--self_position_embedding', default='loc_learned',
                        type=str, help='(none, xyz_learned, loc_learned)')
    parser.add_argument('--self_attend', action='store_true')

    # Loss
    parser.add_argument('--query_points_obj_topk', default=4, type=int)
    parser.add_argument('--use_contrastive_align', action='store_true')
    parser.add_argument('--use_soft_token_loss', action='store_true')
    parser.add_argument('--detect_intermediate', action='store_true')
    parser.add_argument('--joint_det', action='store_true')

    # Data
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch Size during training')
    parser.add_argument('--dataset', type=str, default=['sr3d'],
                        nargs='+', help='list of datasets to train on')
    parser.add_argument('--test_dataset', default='sr3d')
    parser.add_argument('--data_root', default='./')
    parser.add_argument('--use_height', action='store_true',
                        help='Use height signal in input.')
    parser.add_argument('--use_color', action='store_true',
                        help='Use RGB color in input.')
    parser.add_argument('--use_multiview', action='store_true')
    parser.add_argument('--butd', action='store_true')
    parser.add_argument('--butd_gt', action='store_true')
    parser.add_argument('--butd_cls', action='store_true')
    parser.add_argument('--augment_det', action='store_true')
    parser.add_argument('--disable_box_jitter', action='store_true')
    parser.add_argument(
        '--disable_train_augmentation',
        action='store_true',
        default=False,
        help=(
            'Disable train-split point-cloud, target-box, scene-box, and '
            'detector corruption augmentation. Useful for eval_train feature '
            'dumps that should match validation-time augmentation state.'
        ),
    )
    parser.add_argument(
        '--spacy_relation_free_yaw_only_aug',
        action='store_true',
        default=False,
        help=(
            'For _spacy datasets, route relation-free non-view samples to '
            'yaw-only geometry augmentation instead of full pitch/roll.'
        ),
    )
    parser.add_argument(
        '--spacy_relation_free_view_guard_aug',
        action='store_true',
        default=False,
        help=(
            'For _spacy relation-free samples already routed to yaw-only, '
            'route explicit raw view-word samples to no-rotation geometry.'
        ),
    )
    parser.add_argument(
        '--spacy_relation_free_stable_yaw_aug',
        action='store_true',
        default=False,
        help=(
            'For _spacy relation-free samples routed away from full natural '
            'augmentation, keep yaw/flip geometry but suppress point, global '
            'shift/scale, and color jitter.'
        ),
    )
    parser.add_argument(
        '--spacy_relation_free_view_small_yaw_aug',
        action='store_true',
        default=False,
        help=(
            'For _spacy relation-free samples routed to yaw-only, route explicit '
            'raw view-word samples to small yaw jitter without 90-degree '
            'rotation or flips.'
        ),
    )
    parser.add_argument(
        '--spacy_relation_free_rawview_global_only_train',
        action='store_true',
        default=False,
        help=(
            'For _spacy train samples with relation-free raw view words and '
            'parse-noise evidence, train SACR/RAPF as global-only while leaving '
            'eval scoring rule-free.'
        ),
    )
    parser.add_argument(
        '--spacy_relation_free_none_aug',
        action='store_true',
        default=False,
        help=(
            'For _spacy relation-free samples routed away from full natural '
            'augmentation, use no-rotation geometry augmentation.'
        ),
    )
    parser.add_argument(
        '--spacy_relation_free_compass_guard_aug',
        action='store_true',
        default=False,
        help=(
            'For _spacy relation-free samples routed to yaw-only, route '
            'absolute compass-direction samples to no-rotation geometry.'
        ),
    )
    parser.add_argument(
        '--spacy_direction_sensitive_no_jitter_aug',
        action='store_true',
        default=False,
        help=(
            'For _spacy samples with relation/view/compass direction semantics, '
            'keep the chosen rotation policy but suppress point, global '
            'shift/scale, and color jitter.'
        ),
    )
    parser.add_argument(
        '--scanrefer_inject_spacy_decomp',
        action='store_true',
        default=False,
        help=(
            'For the raw ScanRefer dataset, inject matching refined spaCy '
            'decomposition fields while keeping the dataset identity scanrefer.'
        ),
    )
    parser.add_argument(
        '--text_target_alias_policy',
        type=str,
        default='strict',
        choices=['strict', 'run325_high_precision', 'run325_no_computer'],
        help=(
            'Optional deployable target-slot class alias remap for '
            'text_target_cid. strict keeps exact class aliases only; '
            'run325_* policies are explicit diagnostics from Run325.'
        ),
    )
    parser.add_argument('--num_workers', type=int, default=8)

    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--optimizer', type=str, default='adamW')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_backbone", default=1e-4, type=float)
    parser.add_argument("--text_encoder_lr", default=1e-5, type=float)
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"])
    parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340],
                        nargs='+', help='when to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for lr')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--syncbn', action='store_true')
    parser.add_argument('--warmup-epoch', type=int, default=-1)
    parser.add_argument('--warmup-multiplier', type=int, default=100)
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision training (default: True)')

    # io
    parser.add_argument('--checkpoint_path', default=None,
                        help='Model checkpoint path')
    parser.add_argument('--log_dir', default='log',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--tensorboard_root', default=None,
                        help='Root directory for TensorBoard event files')
    parser.add_argument('--tensorboard_flush_secs', type=int, default=30,
                        help='TensorBoard flush interval in seconds')
    parser.add_argument('--print_freq', type=int, default=10)  # batch-wise
    parser.add_argument('--save_freq', type=int, default=10)  # epoch-wise
    parser.add_argument('--val_freq', type=int, default=5)  # epoch-wise
    parser.add_argument(
        '--best_checkpoint_only', action='store_true',
        help=(
            'Keep only ckpt_best_primary.pth, selected after validation by '
            '--best_checkpoint_metric. Intended for long ablation runs where '
            'retaining every periodic checkpoint is prohibitively large.'
        ),
    )
    parser.add_argument(
        '--best_checkpoint_metric', type=str,
        default='last__bbs_acc0.25_top1',
        help='Validation result key used to select ckpt_best_primary.pth.',
    )
    parser.add_argument(
        '--best_checkpoint_min_delta', type=float, default=0.0,
        help='Required strict improvement over the current best score.',
    )
    parser.add_argument(
        '--early_stopping', action='store_true',
        help=(
            'Stop after validation-metric saturation. The maximum epoch remains '
            '--max_epoch; decisions are made only after complete validations.'
        ),
    )
    parser.add_argument(
        '--early_stopping_metric', type=str,
        default='last__bbs_acc0.25_top1',
        help='Validation result key monitored for saturation.',
    )
    parser.add_argument(
        '--early_stopping_min_epoch', type=int, default=35,
        help='Do not count stale validations before this epoch.',
    )
    parser.add_argument(
        '--early_stopping_patience', type=int, default=4,
        help='Number of stale validation events required to stop.',
    )
    parser.add_argument(
        '--early_stopping_min_delta', type=float, default=0.001,
        help=(
            'Minimum increase that resets saturation patience. For a [0,1] '
            'accuracy, 0.001 equals 0.10 percentage point.'
        ),
    )

    parser.add_argument('--use_s2s_aux_loss', action='store_true', default=False,
                        help='Enable direct supervision on slot_dict using decomposed span annotations (default: False)')
    parser.add_argument('--s2s_aux_weight', type=float, default=1.0,
                        help='Global weight for S2S auxiliary slot supervision (default: 1.0)')

    # S2S-ACD-DHC: Structured slot / decoder flags
    parser.add_argument('--use_structured_slots', action='store_true', default=False,
                        help='Enable S2S structured slot memory (default: False)')
    parser.add_argument('--use_late_acd', action='store_true', default=False,
                        help='Enable late ACD head for anchor-conditioned reasoning (default: False)')
    parser.add_argument('--use_dhc', action='store_true', default=False,
                        help='Enable DHC structured supervision (default: False)')

    # S2S slot settings
    parser.add_argument('--max_rel_anchor_pairs', type=int, default=3,
                        help='Max relation-anchor pairs per sample (default: 3)')
    parser.add_argument('--slot_pooling', type=str, default='attention',
                        choices=['mean', 'attention', 'boundary'],
                        help='Slot pooling method (default: attention)')


    # ACD settings
    parser.add_argument('--acd_top_m_targets', type=int, default=32,
                        help='Top M target candidates for ACD (default: 32)')
    parser.add_argument('--acd_top_k_anchors', type=int, default=16,
                        help='Top K anchor candidates for ACD (default: 16)')
    parser.add_argument('--acd_geo_dim', type=int, default=16,
                        help='Geometry encoding dimension (default: 16)')
    parser.add_argument('--acd_hidden_dim', type=int, default=288,
                        help='ACD hidden dimension (default: 288)')
    parser.add_argument('--acd_global_residual_alpha', type=float, default=0.5,
                        help='Global residual alpha for ACD fusion (default: 0.5)')
    parser.add_argument('--acd_use_confidence_fusion', action='store_true', default=False,
                        help='Use confidence-aware fusion in ACD (default: False)')
    parser.add_argument('--acd_warmup_steps', type=int, default=5000,
                        help='Steps to linearly warm up ACD residual alpha (default: 5000)')
    parser.add_argument('--acd_initial_alpha', type=float, default=0.05,
                        help='Initial ACD alpha before warmup completes (default: 0.05)')
    parser.add_argument('--acd_ea_scale', type=float, default=1.0,
                        help='Scale factor on the target-attribute ACD term (default: 1.0)')
    parser.add_argument('--acd_pool_ea_multiplier', type=float, default=1.0,
                        help='Multiplier on scaled_s_ea for target pool selection (default: 1.0)')
    parser.add_argument('--acd_final_ea_multiplier', type=float, default=1.0,
                        help='Multiplier on scaled_s_ea for final score (default: 1.0)')
    parser.add_argument('--acd_disable_struct_rerank', action='store_true', default=False,
                        help='Keep computing structured branch but do not fuse s_struct into final ACD score (default: False)')
    parser.add_argument('--acd_base_score_source', type=str, default='contrastive',
                        choices=['contrastive', 'quality'],
                        help='Base score source for late ACD final scores (default: contrastive)')
    parser.add_argument('--acd_rank_weight', type=float, default=1.0,
                        help='Weight for ACD target-vs-hard-negative rank loss (default: 1.0)')
    parser.add_argument('--acd_train_only', action='store_true', default=False,
                        help='Freeze all non-ACD parameters during training (default: False)')
    parser.add_argument('--acd_lr', type=float, default=0.001,
                        help='Learning rate used when --acd_train_only is enabled (default: 0.001)')

    # DHC settings
    parser.add_argument('--dhc_consistency_weight', type=float, default=0.2,
                        help='DHC consistency loss weight (default: 0.2)')
    parser.add_argument('--dhc_ent_hardneg_weight', type=float, default=0.2,
                        help='DHC entity hard negative weight (default: 0.2)')
    parser.add_argument('--dhc_attr_hardneg_weight', type=float, default=0.2,
                        help='DHC attribute hard negative weight (default: 0.2)')
    parser.add_argument('--dhc_rel_hardneg_weight', type=float, default=0.2,
                        help='DHC relation hard negative weight (default: 0.2)')
    parser.add_argument('--dhc_margin_min', type=float, default=0.0,
                        help='Minimum floor applied to all learned DHC margins; <=0 disables flooring (default: 0.0)')
    parser.add_argument('--dhc_temperature_max', type=float, default=0.0,
                        help='Maximum cap applied to learned DHC temperature; <=0 disables capping (default: 0.0)')

    # Debug
    parser.add_argument('--structured_debug', action='store_true', default=False,
                        help='Enable structured reasoning debug logging (default: False)')
    parser.add_argument('--eval_use_acd_scores', action='store_true', default=False,
                        help='Use ACD final scores for last-layer bbs evaluation ranking (default: False)')
    parser.add_argument('--eval_use_structured_scores', action='store_true', default=False,
                        help='Use SACR structured scores for last-layer bbs evaluation ranking (default: False)')
    parser.add_argument('--eval_use_quality_scores', action='store_true', default=False,
                        help='Use quality IoU scores for last-layer bbs evaluation ranking (default: False)')
    parser.add_argument('--eval_use_fused_scores', action='store_true', default=False,
                        help='Use RAPF fused scores for last-layer bbs evaluation ranking (default: False)')
    parser.add_argument('--eval_use_selector_scores', action='store_true', default=False,
                        help='Use source-pool selector scores for last-layer bbs evaluation ranking (default: False)')
    parser.add_argument('--eval_use_selector_pool_scores', action='store_true', default=False,
                        help='Use source-pool top-K candidates reranked by selector scores for last-layer bbs evaluation ranking (default: False)')
    parser.add_argument('--eval_use_selector_choice_scores', action='store_true', default=False,
                        help='Use direct source-choice selector scores for last-layer bbs evaluation ranking (default: False)')
    parser.add_argument('--eval_use_selector_choice_hybrid_scores', action='store_true', default=False,
                        help='Use direct source-choice scores only when confident, otherwise fall back to a configured source (default: False)')
    parser.add_argument('--eval_use_selector_choice_quality_override_scores', action='store_true', default=False,
                        help='Use direct source-choice scores to override quality only when non-quality confidence clears the configured margin (default: False)')
    parser.add_argument('--eval_use_detector_policy_adapter_scores',
                        action='store_true', default=False,
                        help='Use detector-policy adapter scores for last-layer bbs evaluation ranking (default: False)')
    parser.add_argument('--eval_primary_score_source', type=str, default='base',
                        choices=[
                            'base', 'structured', 'quality', 'fused',
                            'acd', 'selector', 'selector_pool',
                            'selector_choice', 'selector_choice_hybrid',
                            'selector_choice_quality_override',
                            'detector_policy_adapter',
                            'detector_countboost', 'detector_run174boost',
                            'detector_countsplit',
                            'detector_countsplit_lowonly',
                            'detector_countsplit_guarded',
                            'detector_countsplit_guarded_allcount',
                            'detector_jointtight', 'detector_strongcoarse',
                            'detector_confblend035', 'detector_confblend05',
                        ],
                        help='Explicit primary score source for last-layer bbs evaluation ranking (default: base). This supersedes the legacy eval_use_* flags when not base.')
    parser.add_argument('--eval_selector_pool_k', type=int, default=1,
                        help='Top-K candidates per source used by --eval_use_selector_pool_scores (default: 1)')
    parser.add_argument('--eval_selector_choice_min_margin', type=float, default=0.0,
                        help='Minimum top1-top2 source-choice logit margin required before selector-choice hybrid overrides fallback (default: 0.0)')
    parser.add_argument('--eval_selector_choice_hybrid_fallback', type=str, default='quality',
                        choices=['base', 'fused', 'quality'],
                        help='Fallback score source for --eval_use_selector_choice_hybrid_scores (default: quality)')
    parser.add_argument('--eval_selector_choice_source_bias_base', type=float, default=0.0,
                        help='Additive eval-time bias for selector-choice base source logits (default: 0.0)')
    parser.add_argument('--eval_selector_choice_source_bias_fused', type=float, default=0.0,
                        help='Additive eval-time bias for selector-choice fused source logits (default: 0.0)')
    parser.add_argument('--eval_selector_choice_source_bias_quality', type=float, default=0.0,
                        help='Additive eval-time bias for selector-choice quality source logits (default: 0.0)')
    parser.add_argument('--eval_selector_choice_source_bias_contrastive_base', type=float, default=0.0,
                        help='Additive eval-time bias for selector-choice contrastive-base source logits (default: 0.0)')
    parser.add_argument('--eval_selector_choice_source_bias_detector_countboost', type=float, default=0.0,
                        help='Additive eval-time bias for selector-choice detector-countboost source logits (default: 0.0)')
    parser.add_argument('--eval_selector_choice_source_bias_detector_countsplit', type=float, default=0.0,
                        help='Additive eval-time bias for selector-choice detector-countsplit source logits (default: 0.0)')
    parser.add_argument('--eval_selector_choice_source_bias_detector_jointtight', type=float, default=0.0,
                        help='Additive eval-time bias for selector-choice detector-jointtight source logits (default: 0.0)')
    parser.add_argument('--eval_selector_choice_source_bias_detector_strongcoarse', type=float, default=0.0,
                        help='Additive eval-time bias for selector-choice detector-strongcoarse source logits (default: 0.0)')
    parser.add_argument('--eval_selector_choice_source_bias_detector_confblend035', type=float, default=0.0,
                        help='Additive eval-time bias for selector-choice detector-confblend035 source logits (default: 0.0)')
    parser.add_argument('--eval_selector_choice_source_bias_detector_confblend05', type=float, default=0.0,
                        help='Additive eval-time bias for selector-choice detector-confblend05 source logits (default: 0.0)')
    parser.add_argument('--eval_selector_choice_use_override_head',
                        action='store_true', default=False,
                        help='Use selector_choice_override_logit to decide keep-default vs override-default during selector-choice eval (default: False)')
    parser.add_argument('--eval_selector_choice_override_threshold',
                        type=float, default=0.0,
                        help='Override-head logit threshold for selector-choice eval; lower values select more non-base overrides (default: 0.0)')
    parser.add_argument('--eval_selector_choice_override_default_source',
                        type=str, default='base',
                        choices=[
                            'base', 'fused', 'quality', 'contrastive_base',
                            'detector_countboost', 'detector_run174boost',
                            'detector_countsplit',
                            'detector_countsplit_lowonly',
                            'detector_countsplit_guarded',
                            'detector_countsplit_guarded_allcount',
                            'detector_jointtight', 'detector_strongcoarse',
                            'detector_confblend035', 'detector_confblend05'
                        ],
                        help='Default source kept when selector-choice override head does not fire (default: base)')
    parser.add_argument('--eval_report_diagnostic_scores', action='store_true', default=False,
                        help='Also report diagnostic metrics for all available score sources (default: False)')
    parser.add_argument('--eval_target_cid_source', type=str, default='gt',
                        choices=['gt', 'text'],
                        help='Target class source for target-class diagnostic score sources: gt uses dataset target_cid; text uses deployable text_target_cid and does not fall back to gt (default: gt)')
    parser.add_argument('--eval_report_spacy_source_scores',
                        action='store_true', default=False,
                        help='Report lightweight spacy bucket accuracies for selected score sources without enabling the full diagnostic sweep (default: False)')
    parser.add_argument('--eval_spacy_source_score_sources', type=str,
                        default='base,contrastive_base,quality,fused,detector_jointtight,detector_countsplit_guarded_allcount,detector_strongcoarse',
                        help='Comma-separated score sources to audit inside spacy augmentation/profile buckets when --eval_report_spacy_source_scores is set')
    parser.add_argument('--eval_dump_source_choice_features_path', type=str, default=None,
                        help='Optional torch-save path for frozen source-choice feature rows during eval')
    parser.add_argument('--eval_dump_source_choice_topk', type=int, default=1,
                        help='When dumping source-choice features, dump a union of each source top-k candidate rows if >1 (default: 1)')
    parser.add_argument('--eval_results_json_path', type=str, default=None,
                        help='Optional JSON path for rank0 eval_results returned by the grounding evaluator')
    parser.add_argument('--verbose_diagnostics', action='store_true', default=False,
                        help='Log the full dbg_/diag_/eval_ diagnostic set (default: core diagnostics only)')

    # SACR / RAPF / QA-HNL mainline
    parser.add_argument('--use_quality_head', action='store_true', default=False,
                        help='Enable the query quality IoU head (default: False)')
    parser.add_argument('--freeze_quality_head', action='store_true', default=False,
                        help='Freeze quality_head parameters during training diagnostics (default: False)')
    parser.add_argument('--quality_loss_weight', type=float, default=1.0,
                        help='Weight for quality IoU supervision (default: 1.0)')
    parser.add_argument('--quality_iou_threshold', type=float, default=0.25,
                        help='IoU threshold used for quality diagnostics (default: 0.25)')
    parser.add_argument('--quality_topk_rerank_weight', type=float, default=0.0,
                        help='Weight for quality-score top-K rerank supervision (default: 0.0)')
    parser.add_argument('--quality_topk_rerank_source', type=str, default='fused',
                        choices=[
                            'base', 'structured', 'quality', 'fused',
                            'source_pool', 'external_pool',
                            'detector_policy_adapter'
                        ] + list(DETECTOR_POLICY_SOURCE_NAMES),
                        help='Candidate score source for quality top-K rerank supervision (default: fused)')
    parser.add_argument('--quality_topk_rerank_k', type=int, default=5,
                        help='Candidate pool size for quality top-K rerank supervision (default: 5)')
    parser.add_argument('--quality_topk_rerank_margin', type=float, default=0.05,
                        help='Margin for quality top-K rerank supervision (default: 0.05)')
    parser.add_argument('--quality_topk_rerank_min_iou_gap', type=float, default=0.02,
                        help='Minimum IoU gap for a top-K candidate to be a rerank competitor (default: 0.02)')
    parser.add_argument('--use_source_pool_selector', action='store_true', default=False,
                        help='Enable an independent source-pool selector head (default: False)')
    parser.add_argument('--source_pool_selector_hidden_dim', type=int, default=288,
                        help='Hidden dimension for source-pool selector head (default: 288)')
    parser.add_argument('--source_pool_selector_loss_weight', type=float, default=0.0,
                        help='Weight for source-pool selector top-K supervision (default: 0.0)')
    parser.add_argument('--source_pool_selector_source', type=str, default='source_pool',
                        choices=[
                            'base', 'structured', 'quality', 'fused',
                            'source_pool', 'external_pool',
                            'detector_policy_adapter', 'source_choice'
                        ],
                        help='Candidate score source for source-pool selector supervision (default: source_pool)')
    parser.add_argument('--source_pool_selector_k', type=int, default=5,
                        help='Candidate pool size for source-pool selector supervision (default: 5)')
    parser.add_argument('--source_pool_selector_temperature', type=float, default=1.0,
                        help='Softmax temperature for source-pool selector CE loss (default: 1.0)')
    parser.add_argument('--source_pool_selector_min_iou_gap', type=float, default=0.02,
                        help='Minimum IoU gap for selector candidate supervision (default: 0.02)')
    parser.add_argument('--source_pool_selector_source_min_iou_gaps',
                        type=str, default=None,
                        help='Comma-separated per-source IoU gaps for source_choice targets, e.g. quality:0.02,detector_jointtight:0.10. Sources not listed use --source_pool_selector_min_iou_gap.')
    parser.add_argument('--source_pool_selector_train_only', action='store_true', default=False,
                        help='Freeze all non-selector parameters during training (default: False)')
    parser.add_argument('--source_pool_selector_lr', type=float, default=0.001,
                        help='Learning rate used when --source_pool_selector_train_only is enabled (default: 0.001)')
    parser.add_argument('--source_pool_selector_candidate_aware', action='store_true', default=False,
                        help='Score source-pool candidates with source-aware selector logits (default: False)')
    parser.add_argument('--source_pool_selector_direct_choice', action='store_true', default=False,
                        help='Directly choose among base/fused/quality top-1 candidates (default: False)')
    parser.add_argument('--source_pool_selector_include_contrastive_choice',
                        action='store_true', default=False,
                        help='Include contrastive/BBF top-1 as a direct source-choice candidate (default: False)')
    parser.add_argument('--source_pool_selector_include_detector_policy_choice',
                        action='store_true', default=False,
                        help='Include deployable detector-policy top-1 candidates in source-choice training (default: False)')
    parser.add_argument('--source_pool_selector_choice_sources',
                        type=str, default=None,
                        help='Comma-separated explicit source-choice candidates, e.g. base,quality,detector_countboost. Overrides the include_contrastive/include_detector_policy expansion when set.')
    parser.add_argument('--source_pool_selector_rank_features',
                        action='store_true', default=False,
                        help='Add scale-normalized cross-source rank features to source-aware selector candidates (default: False)')
    parser.add_argument('--source_pool_selector_pairdelta_features',
                        action='store_true', default=False,
                        help='Append source-pair delta features to source-aware selector candidates (default: False)')
    parser.add_argument('--source_pool_selector_candidate_context',
                        action='store_true', default=False,
                        help='Add source-pool summary context features to candidate-aware selector candidates (default: False)')
    parser.add_argument('--source_pool_selector_candidate_context_k',
                        type=int, default=5,
                        help='Top-K source-pool candidates used to build candidate-aware selector context (default: 5)')
    parser.add_argument('--source_pool_selector_text_context',
                        action='store_true', default=False,
                        help='Add pooled text features as selector-choice context bias (default: False)')
    parser.add_argument('--source_pool_selector_metadata_context',
                        action='store_true', default=False,
                        help='Add parse/coverage scalar metadata as selector-choice context bias (default: False)')
    parser.add_argument('--source_pool_selector_context_features',
                        action='store_true', default=False,
                        help='Append selector context directly to source-pool selector MLP inputs instead of only using additive bias (default: False)')
    parser.add_argument('--source_pool_selector_separate_override_head',
                        action='store_true', default=False,
                        help='Add a separate binary keep-base vs override-base head for direct source-choice selector training (default: False)')
    parser.add_argument('--source_pool_selector_override_initial_bias',
                        type=float, default=-1.5,
                        help='Initial final-layer bias for the separate selector-choice override head (default: -1.5)')
    parser.add_argument('--use_detector_policy_adapter', action='store_true',
                        default=False,
                        help='Enable a small trainable detector-policy score adapter (default: False)')
    parser.add_argument('--detector_policy_adapter_train_only',
                        action='store_true', default=False,
                        help='Freeze all non-adapter parameters while training detector-policy adapter (default: False)')
    parser.add_argument('--detector_policy_adapter_lr', type=float,
                        default=0.001,
                        help='Learning rate for detector-policy adapter train-only mode (default: 0.001)')
    parser.add_argument('--detector_policy_adapter_context',
                        action='store_true', default=False,
                        help='Condition detector-policy adapter weights on pooled text/metadata context (default: False)')
    parser.add_argument('--detector_policy_adapter_hidden_dim', type=int,
                        default=32,
                        help='Hidden dimension for detector-policy adapter context MLP (default: 32)')
    parser.add_argument('--detector_policy_adapter_delta_scale', type=float,
                        default=0.25,
                        help='Maximum tanh-scaled delta added to each detector-policy prior weight (default: 0.25)')
    parser.add_argument('--detector_policy_adapter_loss_weight', type=float,
                        default=0.0,
                        help='Weight for detector-policy adapter top-K supervision (default: 0.0)')
    parser.add_argument('--detector_policy_adapter_k', type=int, default=5,
                        help='Top-K adapter candidates supervised by GT IoU (default: 5)')
    parser.add_argument('--detector_policy_adapter_margin', type=float,
                        default=0.05,
                        help='Ranking margin for detector-policy adapter supervision (default: 0.05)')
    parser.add_argument('--detector_policy_adapter_min_iou_gap', type=float,
                        default=0.02,
                        help='Minimum IoU gap for detector-policy adapter competitors (default: 0.02)')
    parser.add_argument('--detector_policy_adapter_reg_weight', type=float,
                        default=0.0,
                        help='Weight for keeping adapter weights near prior values (default: 0.0)')
    parser.add_argument('--source_pool_selector_pairwise_weight', type=float, default=0.5,
                        help='Pairwise margin loss weight for source-pool selector supervision (default: 0.5)')
    parser.add_argument('--source_pool_selector_choice_balance', action='store_true', default=False,
                        help='Use inverse-frequency source class weights for source_choice selector CE loss (default: False)')
    parser.add_argument('--source_pool_selector_choice_balance_power', type=float, default=1.0,
                        help='Exponent for source_choice inverse-frequency class weights when choice balance is enabled (default: 1.0)')
    parser.add_argument('--source_pool_selector_oracle_prior_weight', type=float, default=0.0,
                        help='Batch-level oracle source distribution prior weight for source_choice selector training (default: 0.0)')
    parser.add_argument('--source_pool_selector_override_prior_weight', type=float, default=0.0,
                        help='Batch-level oracle override-rate prior weight for source_choice override-head training (default: 0.0)')
    parser.add_argument('--source_pool_selector_override_source_prior_weight', type=float, default=0.0,
                        help='Batch-level non-default oracle source distribution prior weight for override-style source_choice training (default: 0.0)')
    parser.add_argument('--source_pool_selector_override_margin_weight', type=float, default=0.0,
                        help='Per-sample margin loss weight for source_choice override-head logits (default: 0.0)')
    parser.add_argument('--source_pool_selector_override_margin', type=float, default=1.0,
                        help='Signed logit margin for source_choice override-head decisions (default: 1.0)')
    parser.add_argument('--source_pool_selector_quality_base_margin_weight', type=float, default=0.0,
                        help='Per-sample margin loss weight requiring quality source logits to beat base logits when quality top-1 has higher GT IoU (default: 0.0)')
    parser.add_argument('--source_pool_selector_quality_base_margin', type=float, default=0.75,
                        help='Logit margin for quality-over-base source-choice supervision (default: 0.75)')
    parser.add_argument('--source_pool_selector_quality_default_margin_weight', type=float, default=0.0,
                        help='Per-sample margin loss weight requiring quality source logits to beat the configured default source logits when quality top-1 has higher GT IoU (default: 0.0)')
    parser.add_argument('--source_pool_selector_quality_default_margin', type=float, default=0.75,
                        help='Logit margin for quality-over-default source-choice supervision (default: 0.75)')
    parser.add_argument('--source_pool_selector_quality_default_bidirectional_margin_weight', type=float, default=0.0,
                        help='Per-sample margin loss weight requiring the higher-IoU source to win between quality and the configured default source on clear different-query gaps (default: 0.0)')
    parser.add_argument('--source_pool_selector_quality_default_bidirectional_margin', type=float, default=0.75,
                        help='Logit margin for bidirectional quality-vs-default source-choice supervision (default: 0.75)')
    parser.add_argument('--source_pool_selector_iou_aux_weight', type=float, default=0.0,
                        help='Weak same-threshold-bucket IoU auxiliary weight for ranking non-default source-choice logits without changing the default/override target (default: 0.0)')
    parser.add_argument('--source_pool_selector_iou_aux_margin', type=float, default=0.5,
                        help='Logit margin for same-threshold-bucket IoU auxiliary non-default source ranking (default: 0.5)')
    parser.add_argument('--source_pool_selector_false_base_weight', type=float, default=1.0,
                        help='Additional weight for samples that should override the default source but are currently treated as keep-base cases (default: 1.0)')
    parser.add_argument('--source_pool_selector_false_override_weight', type=float, default=1.0,
                        help='Additional weight for samples that should keep the default source but are currently treated as override cases (default: 1.0)')
    parser.add_argument('--source_pool_selector_sourcewise_negative_weight', type=float, default=1.0,
                        help='Negative-label weight inside base_override_sourcewise_focal_bce sourcewise BCE; keeps override/keep-base pressure decoupled from per-source negative pressure (default: 1.0)')
    parser.add_argument('--source_pool_selector_override_utility_gap_weight', type=float, default=0.0,
                        help='Extra per-sample weight proportional to the absolute oracle utility gap for keep-base vs override decisions (default: 0.0)')
    parser.add_argument('--source_pool_selector_override_default_source',
                        type=str, default='base',
                        choices=[
                            'base', 'fused', 'quality', 'contrastive_base',
                            'detector_countboost', 'detector_run174boost',
                            'detector_countsplit',
                            'detector_countsplit_lowonly',
                            'detector_countsplit_guarded',
                            'detector_countsplit_guarded_allcount',
                            'detector_jointtight', 'detector_strongcoarse',
                            'detector_confblend035', 'detector_confblend05'
                        ],
                        help='Default source for base-override source_choice targets; non-default sources become learned residual overrides (default: base)')
    parser.add_argument('--source_pool_selector_choice_target', type=str, default='iou',
                        choices=[
                            'iou', 'threshold_utility',
                            'threshold_bucket', 'threshold_bucket_bce',
                            'threshold_bucket_argmax',
                            'threshold_bucket_unique',
                            'threshold_bucket_margin',
                            'threshold_utility_softmax',
                            'threshold_utility_regression',
                            'threshold_utility_hard',
                            'source_pool_lse',
                            'base_threshold_gain',
                            'base_override_bce',
                            'base_override_focal_bce',
                            'base_override_sourcewise_focal_bce',
                            'threshold_gain_default_sourcewise_focal_bce',
                            'threshold_gain_default_diffquery_sourcewise_focal_bce',
                            'precision_gain_default_sourcewise_focal_bce',
                            'quality_override'
                        ],
                        help='Target utility for source-choice/source-pool selector supervision (default: iou)')
    parser.add_argument('--use_sacr', action='store_true', default=False,
                        help='Enable structured anchor-compositional reasoning head (default: False)')
    parser.add_argument('--sacr_top_m_targets', type=int, default=32,
                        help='Top M target candidates for SACR relation scoring (default: 32)')
    parser.add_argument('--sacr_top_k_anchors', type=int, default=16,
                        help='Top K anchor candidates for SACR relation scoring (default: 16)')
    parser.add_argument('--sacr_hidden_dim', type=int, default=288,
                        help='Hidden dimension for SACR MLPs (default: 288)')
    parser.add_argument('--sacr_geo_dim', type=int, default=16,
                        help='Geometry encoding dimension for SACR relation pairs (default: 16)')
    parser.add_argument('--sacr_rank_loss_weight', type=float, default=0.0,
                        help='Weight for SACR target-vs-hard-negative rank loss (default: 0.0)')
    parser.add_argument('--sacr_rank_weight', type=float, dest='sacr_rank_loss_weight',
                        help='Deprecated alias for --sacr_rank_loss_weight')
    parser.add_argument('--sacr_rank_margin', type=float, default=0.2,
                        help='Margin for SACR target-vs-hard-negative rank loss (default: 0.2)')
    parser.add_argument('--sacr_disable_relation', action='store_true', default=False,
                        help='Disable SACR relation-anchor scores for ablation (requires --use_sacr; default: False)')
    parser.add_argument('--use_rapf', action='store_true', default=False,
                        help='Enable reliability-aware probabilistic fusion (default: False)')
    parser.add_argument('--universal_modules_train_only', action='store_true',
                        default=False,
                        help=('Freeze the BUTD backbone and train only structured slots, '
                              'SACR, RAPF, and quality modules (default: False)'))
    parser.add_argument('--universal_modules_lr', type=float, default=0.0001,
                        help=('Learning rate used when '
                              '--universal_modules_train_only is enabled '
                              '(default: 0.0001)'))
    parser.add_argument('--freeze_rapf', action='store_true', default=False,
                        help='Freeze reliability_fusion parameters during training diagnostics (default: False)')
    parser.add_argument('--use_reliability_gate', action='store_true', default=False,
                        help='Enable RAPF gate supervision (requires --use_rapf; default: False)')
    parser.add_argument('--rapf_hidden_dim', type=int, default=128,
                        help='Hidden dimension for RAPF gate MLP (default: 128)')
    parser.add_argument('--rapf_initial_gate_bias', type=float, default=-2.0,
                        help='Initial bias for RAPF gate logits (default: -2.0)')
    parser.add_argument('--rapf_use_quality', action='store_true', default=False,
                        help='Use quality scores as an input to RAPF (default: False)')
    parser.add_argument('--rapf_quality_weight', type=float, default=0.25,
                        help='Residual weight for normalized quality scores in RAPF (default: 0.25)')
    parser.add_argument('--rapf_quality_score_weight', type=float, dest='rapf_quality_weight',
                        help='Alias for --rapf_quality_weight')
    parser.add_argument('--rapf_struct_residual_clip', type=float, default=2.0,
                        help='Clip applied to structured-base normalized residual in RAPF (default: 2.0)')
    parser.add_argument('--rapf_gate_loss_weight', type=float, default=0.2,
                        help='Weight for RAPF gate supervision (default: 0.2)')
    parser.add_argument('--rapf_gate_iou_margin', type=float, default=0.02,
                        help='IoU improvement margin for RAPF gate labels (default: 0.02)')
    parser.add_argument('--rapf_generic_gate_cap', type=float, default=0.35,
                        help='Maximum RAPF gate value for generic-target samples (default: 0.35)')
    parser.add_argument('--rapf_max_gate_for_generic_target', type=float,
                        dest='rapf_generic_gate_cap',
                        help='Alias for --rapf_generic_gate_cap')
    parser.add_argument('--rapf_quality_anchor_structured_residual',
                        action='store_true', default=False,
                        help='Experimental tuning variant: anchor RAPF structured residual on base+quality instead of base (default: False)')
    parser.add_argument('--use_qahnl', action='store_true', default=False,
                        help='Enable query-aware hard negative learning (default: False)')
    parser.add_argument('--qahnl_score_source', type=str, default='fused',
                        choices=['base', 'structured', 'quality', 'fused'],
                        help='Score source for QA-HNL hard negative mining (default: fused)')
    parser.add_argument('--qahnl_num_hard_neg', type=int, default=16,
                        help='Number of hard negatives mined per sample for QA-HNL (default: 16)')
    parser.add_argument('--qahnl_pos_iou_thresh', type=float, default=None,
                        help='IoU threshold for QA-HNL positives (default: 0.25)')
    parser.add_argument('--qahnl_pos_iou_threshold', type=float,
                        dest='_qahnl_pos_iou_threshold_alias', default=None,
                        help='Deprecated alias for --qahnl_pos_iou_thresh')
    parser.add_argument('--qahnl_neg_iou_thresh', type=float, default=None,
                        help='Maximum IoU for QA-HNL hard negatives (default: 0.10)')
    parser.add_argument('--qahnl_neg_iou_threshold', type=float,
                        dest='_qahnl_neg_iou_threshold_alias', default=None,
                        help='Deprecated alias for --qahnl_neg_iou_thresh')
    parser.add_argument('--qahnl_topk_iou_pos', type=int, default=None,
                        help='Always include top-K IoU queries as QA-HNL positives (default: 3)')
    parser.add_argument('--qahnl_topk_iou', type=int,
                        dest='_qahnl_topk_iou_alias', default=None,
                        help='Deprecated alias for --qahnl_topk_iou_pos')
    parser.add_argument('--qahnl_margin_base', type=float, default=None,
                        help='Base margin for QA-HNL positive-vs-hard-negative loss (default: 0.2)')
    parser.add_argument('--qahnl_margin', type=float,
                        dest='_qahnl_margin_base_alias', default=None,
                        help='Deprecated alias for --qahnl_margin_base')
    parser.add_argument('--qahnl_margin_iou_lambda', type=float, default=0.5,
                        help='IoU-gap coefficient for QA-HNL adaptive margin (default: 0.5)')
    parser.add_argument('--qahnl_margin_min', type=float, default=0.05,
                        help='Minimum adaptive QA-HNL margin after clamping (default: 0.05)')
    parser.add_argument('--qahnl_margin_max', type=float, default=0.5,
                        help='Maximum adaptive QA-HNL margin after clamping (default: 0.5)')
    parser.add_argument('--qahnl_temperature', type=float, default=1.0,
                        help='QA-HNL softplus temperature before capping (default: 1.0)')
    parser.add_argument('--qahnl_temperature_max', type=float, default=6.0,
                        help='QA-HNL maximum softplus temperature (default: 6.0)')
    parser.add_argument('--qahnl_loss_weight', type=float, default=0.2,
                        help='Weight for QA-HNL loss (default: 0.2)')
    parser.add_argument('--qahnl_use_entity_hardneg', action='store_true', default=False,
                        help='Diagnostic mask flag for future QA-HNL entity hard negatives (default: False)')
    parser.add_argument('--qahnl_entity_hardneg', action='store_true',
                        dest='_qahnl_entity_hardneg_alias', default=False,
                        help='Deprecated alias for --qahnl_use_entity_hardneg')
    parser.add_argument('--qahnl_use_attr_hardneg', action='store_true', default=False,
                        help='Diagnostic mask flag for future QA-HNL attribute hard negatives (default: False)')
    parser.add_argument('--qahnl_attr_hardneg', action='store_true',
                        dest='_qahnl_attr_hardneg_alias', default=False,
                        help='Deprecated alias for --qahnl_use_attr_hardneg')
    parser.add_argument('--qahnl_use_relation_hardneg', action='store_true', default=False,
                        help='Diagnostic mask flag for future QA-HNL relation hard negatives (default: False)')
    parser.add_argument('--qahnl_rel_hardneg', action='store_true',
                        dest='_qahnl_rel_hardneg_alias', default=False,
                        help='Deprecated alias for --qahnl_use_relation_hardneg')

    # others
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank",
                        type=int, default=0,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5],
                        nargs='+', help='A list of AP IoU thresholds')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')
    parser.add_argument("--debug", action='store_true',
                        help="try to overfit few samples")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_train', action='store_true')
    parser.add_argument(
        '--eval_max_samples', type=int, default=-1,
        help=(
            'Limit eval/eval_train dataset annotations to the first N samples. '
            'Use -1 for the full split.'
        )
    )
    parser.add_argument('--pp_checkpoint', default=None)
    parser.add_argument('--reduce_lr', action='store_true')

    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError("Unknown command line arguments: {}".format(
            ' '.join(unknown)
        ))

    args.eval = args.eval or args.eval_train

    _resolve_deprecated_numeric_alias(
        args, 'qahnl_pos_iou_thresh', '_qahnl_pos_iou_threshold_alias',
        0.25, '--qahnl_pos_iou_thresh', '--qahnl_pos_iou_threshold',
        public_alias_attr='qahnl_pos_iou_threshold',
    )
    _resolve_deprecated_numeric_alias(
        args, 'qahnl_neg_iou_thresh', '_qahnl_neg_iou_threshold_alias',
        0.10, '--qahnl_neg_iou_thresh', '--qahnl_neg_iou_threshold',
        public_alias_attr='qahnl_neg_iou_threshold',
    )
    _resolve_deprecated_numeric_alias(
        args, 'qahnl_topk_iou_pos', '_qahnl_topk_iou_alias',
        3, '--qahnl_topk_iou_pos', '--qahnl_topk_iou',
        public_alias_attr='qahnl_topk_iou', is_int=True,
    )
    _resolve_deprecated_numeric_alias(
        args, 'qahnl_margin_base', '_qahnl_margin_base_alias',
        0.2, '--qahnl_margin_base', '--qahnl_margin',
        public_alias_attr='qahnl_margin',
    )
    _resolve_deprecated_bool_alias(
        args, 'qahnl_use_entity_hardneg', '_qahnl_entity_hardneg_alias',
        '--qahnl_use_entity_hardneg', '--qahnl_entity_hardneg',
        public_alias_attr='qahnl_entity_hardneg',
    )
    _resolve_deprecated_bool_alias(
        args, 'qahnl_use_attr_hardneg', '_qahnl_attr_hardneg_alias',
        '--qahnl_use_attr_hardneg', '--qahnl_attr_hardneg',
        public_alias_attr='qahnl_attr_hardneg',
    )
    _resolve_deprecated_bool_alias(
        args, 'qahnl_use_relation_hardneg', '_qahnl_rel_hardneg_alias',
        '--qahnl_use_relation_hardneg', '--qahnl_rel_hardneg',
        public_alias_attr='qahnl_rel_hardneg',
    )

    if args.use_sacr and not args.use_structured_slots:
        raise ValueError("--use_sacr requires --use_structured_slots")
    if args.sacr_disable_relation and not args.use_sacr:
        raise ValueError("--sacr_disable_relation requires --use_sacr")
    if args.use_rapf and not args.use_sacr:
        raise ValueError("--use_rapf requires --use_sacr")
    if args.universal_modules_train_only and not (
        args.use_structured_slots
        and args.use_sacr
        and args.use_rapf
        and args.use_quality_head
    ):
        raise ValueError(
            "--universal_modules_train_only requires --use_structured_slots, "
            "--use_sacr, --use_rapf, and --use_quality_head"
        )
    if (
        args.universal_modules_train_only
        and not args.eval
        and not args.checkpoint_path
    ):
        raise ValueError(
            "--universal_modules_train_only requires --checkpoint_path "
            "when training"
        )
    if args.universal_modules_train_only and (
        args.acd_train_only
        or args.source_pool_selector_train_only
        or args.detector_policy_adapter_train_only
    ):
        raise ValueError(
            "--universal_modules_train_only cannot be combined with another "
            "train-only mode"
        )
    if args.use_reliability_gate and not args.use_rapf:
        raise ValueError("--use_reliability_gate requires --use_rapf")
    if args.rapf_use_quality and not args.use_quality_head:
        raise ValueError("--rapf_use_quality requires --use_quality_head")
    if args.freeze_rapf and not args.use_rapf:
        raise ValueError("--freeze_rapf requires --use_rapf")
    if args.freeze_quality_head and not args.use_quality_head:
        raise ValueError("--freeze_quality_head requires --use_quality_head")
    if (
        (args.freeze_rapf or args.freeze_quality_head)
        and not args.eval
        and not args.checkpoint_path
    ):
        raise ValueError(
            "--freeze_rapf/--freeze_quality_head require --checkpoint_path "
            "when training"
        )
    if args.quality_topk_rerank_weight < 0:
        raise ValueError("--quality_topk_rerank_weight must be non-negative")
    if args.quality_topk_rerank_k < 1:
        raise ValueError("--quality_topk_rerank_k must be >= 1")
    if args.quality_topk_rerank_margin < 0:
        raise ValueError("--quality_topk_rerank_margin must be non-negative")
    if args.quality_topk_rerank_min_iou_gap < 0:
        raise ValueError(
            "--quality_topk_rerank_min_iou_gap must be non-negative"
        )
    if args.quality_topk_rerank_weight > 0 and not args.use_quality_head:
        raise ValueError(
            "--quality_topk_rerank_weight requires --use_quality_head"
        )
    if (
        args.quality_topk_rerank_weight > 0
        and args.quality_topk_rerank_source == "structured"
        and not args.use_sacr
    ):
        raise ValueError(
            "--quality_topk_rerank_source structured requires --use_sacr"
        )
    if (
        args.quality_topk_rerank_weight > 0
        and args.quality_topk_rerank_source == "fused"
        and not args.use_rapf
    ):
        raise ValueError(
            "--quality_topk_rerank_source fused requires --use_rapf"
        )
    if args.source_pool_selector_hidden_dim < 1:
        raise ValueError("--source_pool_selector_hidden_dim must be >= 1")
    if args.detector_policy_adapter_lr <= 0:
        raise ValueError("--detector_policy_adapter_lr must be positive")
    if args.detector_policy_adapter_hidden_dim < 1:
        raise ValueError("--detector_policy_adapter_hidden_dim must be >= 1")
    if args.detector_policy_adapter_delta_scale < 0:
        raise ValueError(
            "--detector_policy_adapter_delta_scale must be non-negative"
        )
    if args.detector_policy_adapter_loss_weight < 0:
        raise ValueError(
            "--detector_policy_adapter_loss_weight must be non-negative"
        )
    if args.detector_policy_adapter_k < 1:
        raise ValueError("--detector_policy_adapter_k must be >= 1")
    if args.detector_policy_adapter_margin < 0:
        raise ValueError(
            "--detector_policy_adapter_margin must be non-negative"
        )
    if args.detector_policy_adapter_min_iou_gap < 0:
        raise ValueError(
            "--detector_policy_adapter_min_iou_gap must be non-negative"
        )
    if args.detector_policy_adapter_reg_weight < 0:
        raise ValueError(
            "--detector_policy_adapter_reg_weight must be non-negative"
        )
    if (
        args.detector_policy_adapter_loss_weight > 0
        and not args.use_detector_policy_adapter
    ):
        raise ValueError(
            "--detector_policy_adapter_loss_weight requires "
            "--use_detector_policy_adapter"
        )
    if (
        args.detector_policy_adapter_train_only
        and not args.use_detector_policy_adapter
    ):
        raise ValueError(
            "--detector_policy_adapter_train_only requires "
            "--use_detector_policy_adapter"
        )
    if (
        args.detector_policy_adapter_context
        and not args.use_detector_policy_adapter
    ):
        raise ValueError(
            "--detector_policy_adapter_context requires "
            "--use_detector_policy_adapter"
        )
    if args.source_pool_selector_loss_weight < 0:
        raise ValueError("--source_pool_selector_loss_weight must be non-negative")
    if args.source_pool_selector_k < 1:
        raise ValueError("--source_pool_selector_k must be >= 1")
    if args.source_pool_selector_temperature <= 0:
        raise ValueError("--source_pool_selector_temperature must be positive")
    if args.source_pool_selector_min_iou_gap < 0:
        raise ValueError("--source_pool_selector_min_iou_gap must be non-negative")
    if args.source_pool_selector_source_min_iou_gaps:
        valid_gap_sources = {
            'base', 'fused', 'quality', 'contrastive_base',
            'detector_policy_adapter',
            'detector_countboost', 'detector_run174boost',
            'detector_countsplit',
            'detector_countsplit_lowonly',
            'detector_countsplit_guarded',
            'detector_countsplit_guarded_allcount',
            'detector_jointtight', 'detector_strongcoarse',
            'detector_confblend035', 'detector_confblend05',
        }
        parsed_gaps = {}
        for item in args.source_pool_selector_source_min_iou_gaps.split(','):
            item = item.strip()
            if not item:
                continue
            if ':' not in item:
                raise ValueError(
                    "--source_pool_selector_source_min_iou_gaps entries "
                    "must use source:value format"
                )
            source, value = item.split(':', 1)
            source = source.strip()
            if source not in valid_gap_sources:
                raise ValueError(
                    "--source_pool_selector_source_min_iou_gaps contains "
                    f"unknown source: {source}"
                )
            if source in parsed_gaps:
                raise ValueError(
                    "--source_pool_selector_source_min_iou_gaps must not "
                    f"repeat source: {source}"
                )
            try:
                gap = float(value)
            except ValueError as exc:
                raise ValueError(
                    "--source_pool_selector_source_min_iou_gaps values "
                    "must be numeric"
                ) from exc
            if gap < 0:
                raise ValueError(
                    "--source_pool_selector_source_min_iou_gaps values "
                    "must be non-negative"
                )
            parsed_gaps[source] = gap
        if not parsed_gaps:
            raise ValueError(
                "--source_pool_selector_source_min_iou_gaps must contain "
                "at least one source:value entry"
            )
        args.source_pool_selector_source_min_iou_gaps = parsed_gaps
    if args.source_pool_selector_lr <= 0:
        raise ValueError("--source_pool_selector_lr must be positive")
    if args.source_pool_selector_pairwise_weight < 0:
        raise ValueError("--source_pool_selector_pairwise_weight must be non-negative")
    if args.source_pool_selector_choice_balance_power < 0:
        raise ValueError("--source_pool_selector_choice_balance_power must be non-negative")
    if args.source_pool_selector_oracle_prior_weight < 0:
        raise ValueError(
            "--source_pool_selector_oracle_prior_weight must be non-negative"
        )
    if args.source_pool_selector_override_prior_weight < 0:
        raise ValueError(
            "--source_pool_selector_override_prior_weight must be non-negative"
        )
    if args.source_pool_selector_override_source_prior_weight < 0:
        raise ValueError(
            "--source_pool_selector_override_source_prior_weight must be "
            "non-negative"
        )
    if args.source_pool_selector_override_margin_weight < 0:
        raise ValueError(
            "--source_pool_selector_override_margin_weight must be non-negative"
        )
    if args.source_pool_selector_override_margin < 0:
        raise ValueError(
            "--source_pool_selector_override_margin must be non-negative"
        )
    if args.source_pool_selector_quality_base_margin_weight < 0:
        raise ValueError(
            "--source_pool_selector_quality_base_margin_weight must be "
            "non-negative"
        )
    if args.source_pool_selector_quality_base_margin < 0:
        raise ValueError(
            "--source_pool_selector_quality_base_margin must be non-negative"
        )
    if args.source_pool_selector_quality_default_margin_weight < 0:
        raise ValueError(
            "--source_pool_selector_quality_default_margin_weight must be "
            "non-negative"
        )
    if args.source_pool_selector_quality_default_margin < 0:
        raise ValueError(
            "--source_pool_selector_quality_default_margin must be "
            "non-negative"
        )
    if args.source_pool_selector_iou_aux_weight < 0:
        raise ValueError(
            "--source_pool_selector_iou_aux_weight must be non-negative"
        )
    if args.source_pool_selector_iou_aux_margin < 0:
        raise ValueError(
            "--source_pool_selector_iou_aux_margin must be non-negative"
        )
    if args.source_pool_selector_false_base_weight < 0:
        raise ValueError(
            "--source_pool_selector_false_base_weight must be non-negative"
        )
    if args.source_pool_selector_false_override_weight < 0:
        raise ValueError(
            "--source_pool_selector_false_override_weight must be "
            "non-negative"
        )
    if args.source_pool_selector_sourcewise_negative_weight < 0:
        raise ValueError(
            "--source_pool_selector_sourcewise_negative_weight must be "
            "non-negative"
        )
    if args.source_pool_selector_override_utility_gap_weight < 0:
        raise ValueError(
            "--source_pool_selector_override_utility_gap_weight must be "
            "non-negative"
        )
    if args.source_pool_selector_candidate_context_k < 1:
        raise ValueError(
            "--source_pool_selector_candidate_context_k must be >= 1"
        )
    if args.eval_selector_pool_k < 1:
        raise ValueError("--eval_selector_pool_k must be >= 1")
    if args.eval_selector_choice_min_margin < 0:
        raise ValueError("--eval_selector_choice_min_margin must be non-negative")
    if args.source_pool_selector_loss_weight > 0 and not args.use_source_pool_selector:
        raise ValueError(
            "--source_pool_selector_loss_weight requires --use_source_pool_selector"
        )
    if args.source_pool_selector_train_only and not args.use_source_pool_selector:
        raise ValueError(
            "--source_pool_selector_train_only requires --use_source_pool_selector"
        )
    if args.source_pool_selector_text_context and not args.use_source_pool_selector:
        raise ValueError(
            "--source_pool_selector_text_context requires --use_source_pool_selector"
        )
    if (
        args.source_pool_selector_metadata_context
        and not args.use_source_pool_selector
    ):
        raise ValueError(
            "--source_pool_selector_metadata_context requires "
            "--use_source_pool_selector"
        )
    if (
        args.source_pool_selector_context_features
        and not args.use_source_pool_selector
    ):
        raise ValueError(
            "--source_pool_selector_context_features requires "
            "--use_source_pool_selector"
        )
    if (
        args.source_pool_selector_context_features
        and not (
            args.source_pool_selector_text_context
            or args.source_pool_selector_metadata_context
        )
    ):
        raise ValueError(
            "--source_pool_selector_context_features requires "
            "--source_pool_selector_text_context or "
            "--source_pool_selector_metadata_context"
        )
    if (
        args.source_pool_selector_train_only
        and args.source_pool_selector_loss_weight <= 0
    ):
        raise ValueError(
            "--source_pool_selector_train_only requires "
            "--source_pool_selector_loss_weight > 0"
        )
    if args.acd_train_only and not args.use_late_acd:
        raise ValueError("--acd_train_only requires --use_late_acd")
    if args.acd_train_only and args.source_pool_selector_train_only:
        raise ValueError(
            "--acd_train_only cannot be combined with "
            "--source_pool_selector_train_only"
        )
    if args.detector_policy_adapter_train_only and args.acd_train_only:
        raise ValueError(
            "--detector_policy_adapter_train_only cannot be combined with "
            "--acd_train_only"
        )
    if args.eval_use_selector_scores and not args.use_source_pool_selector:
        raise ValueError("--eval_use_selector_scores requires --use_source_pool_selector")
    if args.eval_use_selector_pool_scores and not args.use_source_pool_selector:
        raise ValueError(
            "--eval_use_selector_pool_scores requires --use_source_pool_selector"
        )
    if args.eval_use_selector_choice_scores and not args.use_source_pool_selector:
        raise ValueError(
            "--eval_use_selector_choice_scores requires --use_source_pool_selector"
        )
    if args.eval_use_selector_choice_hybrid_scores and not args.use_source_pool_selector:
        raise ValueError(
            "--eval_use_selector_choice_hybrid_scores requires --use_source_pool_selector"
        )
    if (
        args.eval_use_selector_choice_quality_override_scores
        and not args.use_source_pool_selector
    ):
        raise ValueError(
            "--eval_use_selector_choice_quality_override_scores requires "
            "--use_source_pool_selector"
        )
    if (
        args.eval_use_detector_policy_adapter_scores
        and not args.use_detector_policy_adapter
    ):
        raise ValueError(
            "--eval_use_detector_policy_adapter_scores requires "
            "--use_detector_policy_adapter"
        )
    if args.source_pool_selector_direct_choice and not args.use_source_pool_selector:
        raise ValueError(
            "--source_pool_selector_direct_choice requires --use_source_pool_selector"
        )
    if (
        args.source_pool_selector_include_contrastive_choice
        and not (
            args.source_pool_selector_direct_choice
            or args.source_pool_selector_candidate_aware
        )
    ):
        raise ValueError(
            "--source_pool_selector_include_contrastive_choice requires "
            "--source_pool_selector_direct_choice or "
            "--source_pool_selector_candidate_aware"
        )
    if (
        args.source_pool_selector_include_detector_policy_choice
        and not (
            args.source_pool_selector_direct_choice
            or args.source_pool_selector_candidate_aware
        )
    ):
        raise ValueError(
            "--source_pool_selector_include_detector_policy_choice requires "
            "--source_pool_selector_direct_choice or "
            "--source_pool_selector_candidate_aware"
        )
    if args.source_pool_selector_choice_sources:
        valid_choice_sources = {
            'base', 'fused', 'quality', 'contrastive_base',
            'detector_policy_adapter',
            'detector_countboost', 'detector_run174boost',
            'detector_countsplit',
            'detector_countsplit_lowonly',
            'detector_countsplit_guarded',
            'detector_countsplit_guarded_allcount',
            'detector_jointtight', 'detector_strongcoarse',
            'detector_confblend035', 'detector_confblend05',
        }
        parsed_sources = [
            source.strip()
            for source in args.source_pool_selector_choice_sources.split(',')
            if source.strip()
        ]
        if not parsed_sources:
            raise ValueError(
                "--source_pool_selector_choice_sources must contain at least "
                "one source"
            )
        unknown_sources = [
            source for source in parsed_sources
            if source not in valid_choice_sources
        ]
        if unknown_sources:
            raise ValueError(
                "--source_pool_selector_choice_sources contains unknown "
                "sources: {}".format(','.join(unknown_sources))
            )
        if len(parsed_sources) != len(set(parsed_sources)):
            raise ValueError(
                "--source_pool_selector_choice_sources must not contain "
                "duplicate sources"
            )
        if not (
            args.source_pool_selector_direct_choice
            or args.source_pool_selector_candidate_aware
        ):
            raise ValueError(
                "--source_pool_selector_choice_sources requires "
                "--source_pool_selector_direct_choice or "
                "--source_pool_selector_candidate_aware"
            )
        if args.source_pool_selector_override_default_source not in parsed_sources:
            raise ValueError(
                "--source_pool_selector_override_default_source must be "
                "present in --source_pool_selector_choice_sources"
            )
        args.source_pool_selector_choice_sources = tuple(parsed_sources)
    if (
        args.source_pool_selector_rank_features
        and not (
            args.source_pool_selector_direct_choice
            or args.source_pool_selector_candidate_aware
        )
    ):
        raise ValueError(
            "--source_pool_selector_rank_features requires "
            "--source_pool_selector_direct_choice or "
            "--source_pool_selector_candidate_aware"
        )
    if (
        args.source_pool_selector_pairdelta_features
        and not (
            args.source_pool_selector_direct_choice
            or args.source_pool_selector_candidate_aware
        )
    ):
        raise ValueError(
            "--source_pool_selector_pairdelta_features requires "
            "--source_pool_selector_direct_choice or "
            "--source_pool_selector_candidate_aware"
        )
    if (
        args.source_pool_selector_candidate_context
        and not args.source_pool_selector_candidate_aware
    ):
        raise ValueError(
            "--source_pool_selector_candidate_context requires "
            "--source_pool_selector_candidate_aware"
        )
    if (
        args.source_pool_selector_separate_override_head
        and not (
            args.source_pool_selector_direct_choice
            or args.source_pool_selector_candidate_aware
        )
    ):
        raise ValueError(
            "--source_pool_selector_separate_override_head requires "
            "--source_pool_selector_direct_choice or "
            "--source_pool_selector_candidate_aware"
        )
    if (
        args.eval_selector_choice_use_override_head
        and not (
            args.source_pool_selector_direct_choice
            or args.source_pool_selector_candidate_aware
        )
    ):
        raise ValueError(
            "--eval_selector_choice_use_override_head requires "
            "--source_pool_selector_direct_choice or "
            "--source_pool_selector_candidate_aware"
        )
    if (
        args.source_pool_selector_loss_weight > 0
        and args.source_pool_selector_source == "source_choice"
        and not (
            args.source_pool_selector_direct_choice
            or args.source_pool_selector_candidate_aware
        )
    ):
        raise ValueError(
            "--source_pool_selector_source source_choice requires "
            "--source_pool_selector_direct_choice or "
            "--source_pool_selector_candidate_aware"
        )
    if (
        args.source_pool_selector_oracle_prior_weight > 0
        and not args.use_source_pool_selector
    ):
        raise ValueError(
            "--source_pool_selector_oracle_prior_weight requires "
            "--use_source_pool_selector"
        )
    if (
        args.source_pool_selector_oracle_prior_weight > 0
        and args.source_pool_selector_source != "source_choice"
    ):
        raise ValueError(
            "--source_pool_selector_oracle_prior_weight requires "
            "--source_pool_selector_source source_choice"
        )
    if (
        args.source_pool_selector_override_prior_weight > 0
        and not args.use_source_pool_selector
    ):
        raise ValueError(
            "--source_pool_selector_override_prior_weight requires "
            "--use_source_pool_selector"
        )
    if (
        args.source_pool_selector_override_prior_weight > 0
        and args.source_pool_selector_source != "source_choice"
    ):
        raise ValueError(
            "--source_pool_selector_override_prior_weight requires "
            "--source_pool_selector_source source_choice"
        )
    if (
        args.source_pool_selector_override_prior_weight > 0
        and args.source_pool_selector_choice_target not in (
            "base_override_bce",
            "base_override_focal_bce",
            "base_override_sourcewise_focal_bce",
            "threshold_gain_default_sourcewise_focal_bce",
            "threshold_gain_default_diffquery_sourcewise_focal_bce",
            "precision_gain_default_sourcewise_focal_bce",
            "quality_override",
            "threshold_utility_softmax",
        )
    ):
        raise ValueError(
            "--source_pool_selector_override_prior_weight requires an "
            "override-style or softmax source_choice target"
        )
    if (
        args.source_pool_selector_override_source_prior_weight > 0
        and not args.use_source_pool_selector
    ):
        raise ValueError(
            "--source_pool_selector_override_source_prior_weight requires "
            "--use_source_pool_selector"
        )
    if (
        args.source_pool_selector_override_source_prior_weight > 0
        and args.source_pool_selector_source != "source_choice"
    ):
        raise ValueError(
            "--source_pool_selector_override_source_prior_weight requires "
            "--source_pool_selector_source source_choice"
        )
    if (
        args.source_pool_selector_override_source_prior_weight > 0
        and args.source_pool_selector_choice_target not in (
            "base_override_bce",
            "base_override_focal_bce",
            "base_override_sourcewise_focal_bce",
            "threshold_gain_default_sourcewise_focal_bce",
            "threshold_gain_default_diffquery_sourcewise_focal_bce",
            "precision_gain_default_sourcewise_focal_bce",
            "quality_override",
            "threshold_utility_softmax",
        )
    ):
        raise ValueError(
            "--source_pool_selector_override_source_prior_weight requires "
            "an override-style or softmax source_choice target"
        )
    if (
        args.source_pool_selector_override_margin_weight > 0
        and not args.use_source_pool_selector
    ):
        raise ValueError(
            "--source_pool_selector_override_margin_weight requires "
            "--use_source_pool_selector"
        )
    if (
        args.source_pool_selector_override_margin_weight > 0
        and args.source_pool_selector_source != "source_choice"
    ):
        raise ValueError(
            "--source_pool_selector_override_margin_weight requires "
            "--source_pool_selector_source source_choice"
        )
    if (
        args.source_pool_selector_override_margin_weight > 0
        and args.source_pool_selector_choice_target not in (
            "base_override_bce",
            "base_override_focal_bce",
            "base_override_sourcewise_focal_bce",
            "threshold_gain_default_sourcewise_focal_bce",
            "threshold_gain_default_diffquery_sourcewise_focal_bce",
            "precision_gain_default_sourcewise_focal_bce",
            "quality_override",
        )
    ):
        raise ValueError(
            "--source_pool_selector_override_margin_weight requires an "
            "override-style source_choice target"
        )
    if (
        args.source_pool_selector_quality_base_margin_weight > 0
        and not args.use_source_pool_selector
    ):
        raise ValueError(
            "--source_pool_selector_quality_base_margin_weight requires "
            "--use_source_pool_selector"
        )
    if (
        args.source_pool_selector_quality_base_margin_weight > 0
        and args.source_pool_selector_source != "source_choice"
    ):
        raise ValueError(
            "--source_pool_selector_quality_base_margin_weight requires "
            "--source_pool_selector_source source_choice"
        )
    if (
        args.source_pool_selector_quality_default_margin_weight > 0
        and not args.use_source_pool_selector
    ):
        raise ValueError(
            "--source_pool_selector_quality_default_margin_weight requires "
            "--use_source_pool_selector"
        )
    if (
        args.source_pool_selector_quality_default_margin_weight > 0
        and args.source_pool_selector_source != "source_choice"
    ):
        raise ValueError(
            "--source_pool_selector_quality_default_margin_weight requires "
            "--source_pool_selector_source source_choice"
        )
    if (
        args.source_pool_selector_quality_default_bidirectional_margin_weight > 0
        and not args.use_source_pool_selector
    ):
        raise ValueError(
            "--source_pool_selector_quality_default_bidirectional_margin_weight "
            "requires --use_source_pool_selector"
        )
    if (
        args.source_pool_selector_quality_default_bidirectional_margin_weight > 0
        and args.source_pool_selector_source != "source_choice"
    ):
        raise ValueError(
            "--source_pool_selector_quality_default_bidirectional_margin_weight "
            "requires --source_pool_selector_source source_choice"
        )
    if args.source_pool_selector_quality_default_bidirectional_margin_weight < 0:
        raise ValueError(
            "--source_pool_selector_quality_default_bidirectional_margin_weight "
            "must be non-negative"
        )
    if args.source_pool_selector_quality_default_bidirectional_margin < 0:
        raise ValueError(
            "--source_pool_selector_quality_default_bidirectional_margin must "
            "be non-negative"
        )
    if (
        args.source_pool_selector_iou_aux_weight > 0
        and not args.use_source_pool_selector
    ):
        raise ValueError(
            "--source_pool_selector_iou_aux_weight requires "
            "--use_source_pool_selector"
        )
    if (
        args.source_pool_selector_iou_aux_weight > 0
        and args.source_pool_selector_source != "source_choice"
    ):
        raise ValueError(
            "--source_pool_selector_iou_aux_weight requires "
            "--source_pool_selector_source source_choice"
        )
    if (
        args.source_pool_selector_iou_aux_weight > 0
        and args.source_pool_selector_choice_target
        != "threshold_gain_default_sourcewise_focal_bce"
    ):
        raise ValueError(
            "--source_pool_selector_iou_aux_weight requires "
            "--source_pool_selector_choice_target "
            "threshold_gain_default_sourcewise_focal_bce"
        )
    if (
        (
            args.source_pool_selector_false_base_weight != 1.0
            or args.source_pool_selector_false_override_weight != 1.0
            or args.source_pool_selector_sourcewise_negative_weight != 1.0
            or args.source_pool_selector_override_utility_gap_weight > 0
        )
        and not args.use_source_pool_selector
    ):
        raise ValueError(
            "--source_pool_selector_false_base_weight, "
            "--source_pool_selector_false_override_weight, and "
            "--source_pool_selector_override_utility_gap_weight require "
            "--use_source_pool_selector"
        )
    if (
        (
            args.source_pool_selector_false_base_weight != 1.0
            or args.source_pool_selector_false_override_weight != 1.0
            or args.source_pool_selector_sourcewise_negative_weight != 1.0
            or args.source_pool_selector_override_utility_gap_weight > 0
        )
        and args.source_pool_selector_source != "source_choice"
    ):
        raise ValueError(
            "--source_pool_selector_false_base_weight, "
            "--source_pool_selector_false_override_weight, and "
            "--source_pool_selector_override_utility_gap_weight require "
            "--source_pool_selector_source source_choice"
        )
    if (
        (
            args.source_pool_selector_false_base_weight != 1.0
            or args.source_pool_selector_false_override_weight != 1.0
            or args.source_pool_selector_sourcewise_negative_weight != 1.0
            or args.source_pool_selector_override_utility_gap_weight > 0
        )
        and args.source_pool_selector_choice_target not in (
            "base_override_bce",
            "base_override_focal_bce",
            "base_override_sourcewise_focal_bce",
            "threshold_gain_default_sourcewise_focal_bce",
            "threshold_gain_default_diffquery_sourcewise_focal_bce",
            "precision_gain_default_sourcewise_focal_bce",
            "quality_override",
        )
    ):
        raise ValueError(
            "--source_pool_selector_false_base_weight, "
            "--source_pool_selector_false_override_weight, and "
            "--source_pool_selector_sourcewise_negative_weight, and "
            "--source_pool_selector_override_utility_gap_weight require an "
            "override-style source_choice target"
        )
    if (
        args.source_pool_selector_sourcewise_negative_weight != 1.0
        and args.source_pool_selector_choice_target not in (
            "base_override_sourcewise_focal_bce",
            "threshold_gain_default_sourcewise_focal_bce",
            "threshold_gain_default_diffquery_sourcewise_focal_bce",
            "precision_gain_default_sourcewise_focal_bce",
        )
    ):
        raise ValueError(
            "--source_pool_selector_sourcewise_negative_weight only applies "
            "to sourcewise focal BCE source_choice targets"
        )
    if (
        args.source_pool_selector_loss_weight > 0
        and args.source_pool_selector_source == "structured"
        and not args.use_sacr
    ):
        raise ValueError(
            "--source_pool_selector_source structured requires --use_sacr"
        )
    if (
        args.source_pool_selector_loss_weight > 0
        and args.source_pool_selector_source == "fused"
        and not args.use_rapf
    ):
        raise ValueError(
            "--source_pool_selector_source fused requires --use_rapf"
        )
    if args.use_qahnl and args.qahnl_score_source == "structured" and not args.use_sacr:
        raise ValueError("--qahnl_score_source structured requires --use_sacr")
    if args.use_qahnl and args.qahnl_score_source == "quality" and not args.use_quality_head:
        raise ValueError("--qahnl_score_source quality requires --use_quality_head")
    if args.use_qahnl and args.qahnl_score_source == "fused" and not args.use_rapf:
        raise ValueError("--qahnl_score_source fused requires --use_rapf")
    if args.use_qahnl and not args.use_contrastive_align:
        raise ValueError("--use_qahnl requires --use_contrastive_align")
    if args.use_qahnl and args.use_dhc:
        raise ValueError("--use_qahnl is mutually exclusive with legacy --use_dhc")

    if args.eval_use_selector_scores:
        args.eval_use_fused_scores = False
        args.eval_use_structured_scores = False
        args.eval_use_quality_scores = False
        args.eval_use_acd_scores = False
    if args.eval_use_selector_pool_scores:
        args.eval_use_fused_scores = False
        args.eval_use_structured_scores = False
        args.eval_use_quality_scores = False
        args.eval_use_acd_scores = False
        args.eval_use_selector_scores = False
    if args.eval_use_selector_choice_scores:
        args.eval_use_fused_scores = False
        args.eval_use_structured_scores = False
        args.eval_use_quality_scores = False
        args.eval_use_acd_scores = False
        args.eval_use_selector_scores = False
        args.eval_use_selector_pool_scores = False
        args.eval_use_selector_choice_hybrid_scores = False
        args.eval_use_selector_choice_quality_override_scores = False
    if args.eval_use_selector_choice_hybrid_scores:
        args.eval_use_fused_scores = False
        args.eval_use_structured_scores = False
        args.eval_use_quality_scores = False
        args.eval_use_acd_scores = False
        args.eval_use_selector_scores = False
        args.eval_use_selector_pool_scores = False
        args.eval_use_selector_choice_scores = False
        args.eval_use_selector_choice_quality_override_scores = False
    if args.eval_use_selector_choice_quality_override_scores:
        args.eval_use_fused_scores = False
        args.eval_use_structured_scores = False
        args.eval_use_quality_scores = False
        args.eval_use_acd_scores = False
        args.eval_use_selector_scores = False
        args.eval_use_selector_pool_scores = False
        args.eval_use_selector_choice_scores = False
        args.eval_use_selector_choice_hybrid_scores = False
        args.eval_use_detector_policy_adapter_scores = False
    if args.eval_use_detector_policy_adapter_scores:
        args.eval_use_fused_scores = False
        args.eval_use_structured_scores = False
        args.eval_use_quality_scores = False
        args.eval_use_acd_scores = False
        args.eval_use_selector_scores = False
        args.eval_use_selector_pool_scores = False
        args.eval_use_selector_choice_scores = False
        args.eval_use_selector_choice_hybrid_scores = False
        args.eval_use_selector_choice_quality_override_scores = False
    if args.eval_primary_score_source != 'base':
        args.eval_use_fused_scores = False
        args.eval_use_structured_scores = False
        args.eval_use_quality_scores = False
        args.eval_use_acd_scores = False
        args.eval_use_selector_scores = False
        args.eval_use_selector_pool_scores = False
        args.eval_use_selector_choice_scores = False
        args.eval_use_selector_choice_hybrid_scores = False
        args.eval_use_selector_choice_quality_override_scores = False
        args.eval_use_detector_policy_adapter_scores = False

    eval_score_flags = [
        args.eval_use_fused_scores,
        args.eval_use_structured_scores,
        args.eval_use_quality_scores,
        args.eval_use_acd_scores,
        args.eval_use_selector_scores,
        args.eval_use_selector_pool_scores,
        args.eval_use_selector_choice_scores,
        args.eval_use_selector_choice_hybrid_scores,
        args.eval_use_selector_choice_quality_override_scores,
        args.eval_use_detector_policy_adapter_scores,
    ]
    if sum(bool(flag) for flag in eval_score_flags) > 1:
        raise ValueError(
            "Only one eval primary score flag may be true among "
            "--eval_use_fused_scores/--eval_use_structured_scores/"
            "--eval_use_quality_scores/--eval_use_acd_scores/"
            "--eval_use_selector_scores/--eval_use_selector_pool_scores/"
            "--eval_use_selector_choice_scores/"
            "--eval_use_selector_choice_hybrid_scores/"
            "--eval_use_selector_choice_quality_override_scores/"
            "--eval_use_detector_policy_adapter_scores"
        )
    if (
        args.eval_primary_score_source == 'detector_policy_adapter'
        and not args.use_detector_policy_adapter
    ):
        raise ValueError(
            "--eval_primary_score_source detector_policy_adapter requires "
            "--use_detector_policy_adapter"
        )

    # 兼容 torchrun：环境变量显式覆盖默认 local_rank=0。
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    return args


def _strip_ddp_prefix(key):
    if key.startswith("module."):
        key = key[len("module."):]
    return key


def _is_selector_checkpoint_key(key):
    key = _strip_ddp_prefix(key)
    return key.startswith("source_pool_selector.")


def _is_detector_policy_adapter_checkpoint_key(key):
    key = _strip_ddp_prefix(key)
    return key.startswith("detector_policy_adapter.")


def _is_late_acd_checkpoint_key(key):
    key = _strip_ddp_prefix(key)
    return (
        key.startswith("acd_head.")
        or key.startswith("dhc_loss_module.")
    )


def _fresh_checkpoint_key_group(args, key):
    if (
        getattr(args, 'use_source_pool_selector', False)
        and _is_selector_checkpoint_key(key)
    ):
        return 'source_pool_selector'
    if (
        getattr(args, 'use_detector_policy_adapter', False)
        and _is_detector_policy_adapter_checkpoint_key(key)
    ):
        return 'detector_policy_adapter'
    if (
        getattr(args, 'use_late_acd', False)
        and _is_late_acd_checkpoint_key(key)
    ):
        return 'late_acd'
    return None


def _load_model_state_with_selector_compat(args, model, checkpoint_model):
    """Load model state, optionally allowing freshly initialized added heads."""
    allow_fresh_modules = bool(
        getattr(args, 'use_source_pool_selector', False)
        or getattr(args, 'use_detector_policy_adapter', False)
        or getattr(args, 'use_late_acd', False)
    )
    if not allow_fresh_modules:
        model.load_state_dict(checkpoint_model, strict=True)
        return False

    model_state = model.state_dict()
    incompatible_groups = set()
    for key, value in checkpoint_model.items():
        fresh_group = _fresh_checkpoint_key_group(args, key)
        if fresh_group is None:
            continue
        if key not in model_state:
            incompatible_groups.add(fresh_group)
            continue
        current_value = model_state[key]
        if hasattr(current_value, 'shape') and hasattr(value, 'shape'):
            if current_value.shape != value.shape:
                incompatible_groups.add(fresh_group)

    if incompatible_groups:
        checkpoint_model = {
            key: value
            for key, value in checkpoint_model.items()
            if _fresh_checkpoint_key_group(args, key) not in incompatible_groups
        }

    incompatible = model.load_state_dict(checkpoint_model, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = [
        key for key in incompatible.unexpected_keys
        if _fresh_checkpoint_key_group(args, key) is None
    ]
    disallowed_missing = [
        key for key in missing
        if _fresh_checkpoint_key_group(args, key) is None
    ]
    if disallowed_missing:
        raise RuntimeError(
            "Missing checkpoint keys outside fresh modules: {}".format(
                disallowed_missing
            )
        )
    if unexpected:
        raise RuntimeError(
            "Unexpected checkpoint keys: {}".format(unexpected)
        )
    return bool(
        incompatible_groups
        or any(_fresh_checkpoint_key_group(args, key) is not None for key in missing)
    )


def _freeze_non_selector_parameters(args, model):
    """Freeze everything except source_pool_selector parameters if requested."""
    if not getattr(args, 'source_pool_selector_train_only', False):
        return 0
    joint_adapter = getattr(args, 'detector_policy_adapter_train_only', False)
    trainable_count = 0
    for name, param in model.named_parameters():
        is_selector = 'source_pool_selector' in name
        is_adapter = 'detector_policy_adapter' in name
        keep_trainable = is_selector or (joint_adapter and is_adapter)
        param.requires_grad = keep_trainable
        if is_selector:
            trainable_count += param.numel()
    if trainable_count <= 0:
        raise ValueError(
            "--source_pool_selector_train_only found no selector parameters"
        )
    return trainable_count


def _freeze_non_detector_policy_adapter_parameters(args, model):
    """Freeze everything except detector_policy_adapter if requested."""
    if not getattr(args, 'detector_policy_adapter_train_only', False):
        return 0
    joint_selector = getattr(args, 'source_pool_selector_train_only', False)
    trainable_count = 0
    for name, param in model.named_parameters():
        is_adapter = 'detector_policy_adapter' in name
        is_selector = 'source_pool_selector' in name
        keep_trainable = is_adapter or (joint_selector and is_selector)
        param.requires_grad = keep_trainable
        if is_adapter:
            trainable_count += param.numel()
    if trainable_count <= 0:
        raise ValueError(
            "--detector_policy_adapter_train_only found no adapter "
            "parameters"
        )
    return trainable_count


def _freeze_non_acd_parameters(args, model):
    """Freeze everything except ACD / DHC parameters if requested."""
    if not getattr(args, 'acd_train_only', False):
        return 0
    trainable_count = 0
    for name, param in model.named_parameters():
        keep_trainable = (
            name.startswith('acd_head.') or name.startswith('dhc_loss_module.')
        )
        param.requires_grad = keep_trainable
        if keep_trainable:
            trainable_count += param.numel()
    if trainable_count <= 0:
        raise ValueError(
            "--acd_train_only found no ACD/DHC parameters"
        )
    return trainable_count


def _freeze_non_universal_module_parameters(args, model):
    """Freeze the BUTD backbone while calibrating the shared modules."""
    if not getattr(args, 'universal_modules_train_only', False):
        return 0
    trainable_prefixes = (
        'structured_slot_builder.',
        'sacr_head.',
        'reliability_fusion.',
        'quality_head.',
    )
    trainable_count = 0
    for name, param in model.named_parameters():
        keep_trainable = name.startswith(trainable_prefixes)
        param.requires_grad = keep_trainable
        if keep_trainable:
            trainable_count += param.numel()
    if trainable_count <= 0:
        raise ValueError(
            "--universal_modules_train_only found no shared module parameters"
        )
    return trainable_count


def _freeze_parameter_group(args, model, flag_name, name_fragment, cli_flag):
    """Freeze parameters whose names contain name_fragment when requested."""
    if not getattr(args, flag_name, False):
        return 0
    frozen_count = 0
    for name, param in model.named_parameters():
        if name_fragment in name:
            param.requires_grad = False
            frozen_count += param.numel()
    if frozen_count <= 0:
        raise ValueError(
            "{} found no {} parameters".format(cli_flag, name_fragment)
        )
    return frozen_count


def _freeze_rapf_parameters(args, model):
    """Freeze learned RAPF reliability_fusion parameters if requested."""
    return _freeze_parameter_group(
        args, model, 'freeze_rapf', 'reliability_fusion', '--freeze_rapf'
    )


def _freeze_quality_head_parameters(args, model):
    """Freeze learned quality_head parameters if requested."""
    return _freeze_parameter_group(
        args, model, 'freeze_quality_head', 'quality_head',
        '--freeze_quality_head'
    )


def _should_restore_checkpoint_train_state(args, has_fresh_params):
    """Whether continuation should restore optimizer/scheduler state."""
    return (
        not getattr(args, 'eval', False)
        and not getattr(args, 'reduce_lr', False)
        and not has_fresh_params
        and not getattr(args, 'source_pool_selector_train_only', False)
        and not getattr(args, 'detector_policy_adapter_train_only', False)
        and not getattr(args, 'universal_modules_train_only', False)
        and not getattr(args, 'freeze_rapf', False)
        and not getattr(args, 'freeze_quality_head', False)
    )


def _set_non_selector_modules_eval(model):
    """Keep frozen modules deterministic while training source_pool_selector."""
    base_model = model.module if hasattr(model, 'module') else model
    for name, module in base_model.named_modules():
        if name == '':
            continue
        if name.startswith('source_pool_selector'):
            module.train()
        else:
            module.eval()


def _set_non_detector_policy_adapter_modules_eval(model):
    """Keep frozen modules deterministic while training detector adapter."""
    base_model = model.module if hasattr(model, 'module') else model
    for name, module in base_model.named_modules():
        if name == '':
            continue
        if name.startswith('detector_policy_adapter'):
            module.train()
        else:
            module.eval()


def _set_non_selector_adapter_modules_eval(model):
    """Keep only selector and detector adapter modules in train mode."""
    base_model = model.module if hasattr(model, 'module') else model
    for name, module in base_model.named_modules():
        if name == '':
            continue
        if (
            name.startswith('source_pool_selector')
            or name.startswith('detector_policy_adapter')
        ):
            module.train()
        else:
            module.eval()


def _set_non_acd_modules_eval(model):
    """Keep frozen modules deterministic while training late ACD."""
    base_model = model.module if hasattr(model, 'module') else model
    for name, module in base_model.named_modules():
        if name == '':
            continue
        if name.startswith('acd_head') or name.startswith('dhc_loss_module'):
            module.train()
        else:
            module.eval()


def _set_non_universal_modules_eval(model):
    """Keep the frozen BUTD backbone deterministic during module calibration."""
    base_model = model.module if hasattr(model, 'module') else model
    trainable_prefixes = (
        'structured_slot_builder',
        'sacr_head',
        'reliability_fusion',
        'quality_head',
    )
    for name, module in base_model.named_modules():
        if name == '':
            continue
        if name.startswith(trainable_prefixes):
            module.train()
        else:
            module.eval()


def _dataloader_parallelism_kwargs(num_workers):
    """Return DataLoader multiprocessing-only kwargs."""
    num_workers = int(num_workers)
    kwargs = {'persistent_workers': num_workers > 0}
    if num_workers > 0:
        kwargs['prefetch_factor'] = 2
    return kwargs


def load_checkpoint(args, model, optimizer, scheduler):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    try:
        args.start_epoch = int(checkpoint['epoch']) + 1
    except Exception:
        args.start_epoch = 0
    has_fresh_params = _load_model_state_with_selector_compat(
        args, model, checkpoint['model']
    )
    if _should_restore_checkpoint_train_state(args, has_fresh_params):
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    elif has_fresh_params and not args.eval and not args.reduce_lr:
        print(
        "=> fresh optional module parameters are initialized; "
        "skipping optimizer/scheduler state from checkpoint"
        )
    elif (
        getattr(args, 'source_pool_selector_train_only', False)
        and getattr(args, 'detector_policy_adapter_train_only', False)
        and not args.eval
        and not args.reduce_lr
    ):
        print(
            "=> source_pool_selector_train_only + "
            "detector_policy_adapter_train_only enabled; skipping "
            "optimizer/scheduler state from checkpoint"
        )
    elif (
        getattr(args, 'source_pool_selector_train_only', False)
        and not args.eval
        and not args.reduce_lr
    ):
        print(
            "=> source_pool_selector_train_only enabled; skipping "
            "optimizer/scheduler state from checkpoint"
        )
    elif (
        getattr(args, 'detector_policy_adapter_train_only', False)
        and not args.eval
        and not args.reduce_lr
    ):
        print(
            "=> detector_policy_adapter_train_only enabled; skipping "
            "optimizer/scheduler state from checkpoint"
        )
    elif (
        getattr(args, 'universal_modules_train_only', False)
        and not args.eval
        and not args.reduce_lr
    ):
        print(
            "=> universal_modules_train_only enabled; skipping "
            "optimizer/scheduler state from checkpoint"
        )
    elif (
        (
            getattr(args, 'freeze_rapf', False)
            or getattr(args, 'freeze_quality_head', False)
        )
        and not args.eval
        and not args.reduce_lr
    ):
        enabled_freezes = []
        if getattr(args, 'freeze_rapf', False):
            enabled_freezes.append("freeze_rapf")
        if getattr(args, 'freeze_quality_head', False):
            enabled_freezes.append("freeze_quality_head")
        print(
            "=> {} enabled; skipping optimizer/scheduler state from "
            "checkpoint".format(" + ".join(enabled_freezes))
        )

    print("=> loaded successfully '{}' (epoch {})".format(
        args.checkpoint_path, checkpoint['epoch']
    ))

    del checkpoint
    torch.cuda.empty_cache()



class ValidationEarlyStopper:
    """Validation-event early stopping with an auditable deterministic state."""

    def __init__(self, metric, min_epoch, patience, min_delta, max_epoch,
                 val_freq):
        if not metric:
            raise ValueError("early_stopping_metric must be non-empty")
        if patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if min_epoch <= 0 or min_epoch > max_epoch:
            raise ValueError(
                "early_stopping_min_epoch must be in [1, max_epoch]"
            )
        if min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if val_freq <= 0:
            raise ValueError("val_freq must be positive")
        self.metric = metric
        self.min_epoch = int(min_epoch)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.max_epoch = int(max_epoch)
        self.val_freq = int(val_freq)
        self.reference_score = None
        self.reference_epoch = None
        self.stale_validations = 0
        self.history = []

    def update(self, epoch, eval_results):
        if eval_results is None or self.metric not in eval_results:
            raise KeyError(
                "Early-stopping metric {!r} is absent from evaluation "
                "results".format(self.metric)
            )
        score = float(eval_results[self.metric])
        if not np.isfinite(score):
            raise ValueError(
                "Early-stopping metric {!r} is not finite: {}".format(
                    self.metric, score
                )
            )
        meaningful = (
            self.reference_score is None
            or score > self.reference_score + self.min_delta
        )
        if meaningful:
            self.reference_score = score
            self.reference_epoch = int(epoch)
            self.stale_validations = 0
        elif int(epoch) >= self.min_epoch:
            self.stale_validations += 1
        else:
            self.stale_validations = 0
        should_stop = (
            int(epoch) >= self.min_epoch
            and self.stale_validations >= self.patience
        )
        event = {
            'epoch': int(epoch),
            'score': score,
            'meaningful_improvement': bool(meaningful),
            'reference_score': float(self.reference_score),
            'reference_epoch': int(self.reference_epoch),
            'stale_validations': int(self.stale_validations),
            'should_stop': bool(should_stop),
        }
        self.history.append(event)
        return event

    def receipt(self, status, stop_epoch=None, mechanism='native'):
        return {
            'enabled': True,
            'status': status,
            'mechanism': mechanism,
            'metric': self.metric,
            'mode': 'max',
            'min_epoch': self.min_epoch,
            'patience_validations': self.patience,
            'min_delta': self.min_delta,
            'validation_frequency_epochs': self.val_freq,
            'maximum_epoch': self.max_epoch,
            'reference_score': self.reference_score,
            'reference_epoch': self.reference_epoch,
            'stale_validations': self.stale_validations,
            'stop_epoch': (
                int(stop_epoch) if stop_epoch is not None else None
            ),
            'reason': (
                'validation_metric_saturated'
                if status == 'early_stopped'
                else (
                    'maximum_epoch_reached'
                    if status == 'max_epoch' else 'monitoring'
                )
            ),
            'history': list(self.history),
        }


def write_early_stopping_receipt(args, payload):
    """Atomically persist early-stopping state on rank zero."""
    path = os.path.join(args.log_dir, 'early_stopping.json')
    tmp = path + '.tmp.{}'.format(os.getpid())
    with open(tmp, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(tmp, path)
    return path


def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False):
    """Save checkpoint if requested."""
    if save_cur or epoch % args.save_freq == 0:
        state = {
            'config': args,
            'save_path': '',
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        spath = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        state['save_path'] = spath
        tmp_path = spath + ".tmp"
        try:
            # Atomic replace avoids leaving partially written checkpoints.
            torch.save(state, tmp_path)
            os.replace(tmp_path, spath)
            print("Saved in {}".format(spath))
        except (OSError, RuntimeError) as e:
            # Keep training even if storage is temporarily full.
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            print("[WARN] Failed to save checkpoint {}: {}".format(spath, e))
            print("[WARN] Training will continue without this checkpoint.")
    else:
        print("not saving checkpoint")


def save_best_checkpoint(args, epoch, model, optimizer, scheduler, eval_results):
    """Atomically retain the checkpoint with the best configured metric."""
    metric_key = getattr(
        args, 'best_checkpoint_metric', 'last__bbs_acc0.25_top1'
    )
    if eval_results is None or metric_key not in eval_results:
        raise KeyError(
            "Best-checkpoint metric {!r} is absent from evaluation results; "
            "available keys include {}".format(
                metric_key, sorted(eval_results or {})[:20]
            )
        )
    score = float(eval_results[metric_key])
    if not np.isfinite(score):
        raise ValueError(
            "Best-checkpoint metric {!r} is not finite: {}".format(
                metric_key, score
            )
        )

    receipt_path = os.path.join(args.log_dir, 'best_primary.json')
    checkpoint_path = os.path.join(args.log_dir, 'ckpt_best_primary.pth')
    previous_score = None
    if os.path.isfile(receipt_path):
        with open(receipt_path, 'r') as f:
            previous_score = float(json.load(f)['score'])
    min_delta = float(getattr(args, 'best_checkpoint_min_delta', 0.0))
    if previous_score is not None and score <= previous_score + min_delta:
        print(
            "Best checkpoint unchanged: {}={:.6f}, current best={:.6f}".format(
                metric_key, score, previous_score
            )
        )
        return False

    selection = {
        'metric': metric_key,
        'score': score,
        'epoch': int(epoch),
        'mode': 'max',
        'comparison': 'strict_greater_than',
        'min_delta': min_delta,
        'checkpoint': checkpoint_path,
    }
    state = {
        'config': args,
        'save_path': checkpoint_path,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': int(epoch),
        'best_checkpoint_selection': selection,
    }
    checkpoint_tmp = checkpoint_path + '.tmp'
    receipt_tmp = receipt_path + '.tmp'
    try:
        torch.save(state, checkpoint_tmp)
        os.replace(checkpoint_tmp, checkpoint_path)
        with open(receipt_tmp, 'w') as f:
            json.dump(selection, f, indent=2, sort_keys=True)
            f.write('\n')
        os.replace(receipt_tmp, receipt_path)
    except Exception:
        for tmp_path in (checkpoint_tmp, receipt_tmp):
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        raise
    print(
        "Updated best checkpoint: {}={:.6f} at epoch {} -> {}".format(
            metric_key, score, epoch, checkpoint_path
        )
    )
    return True


class BaseTrainTester:
    """Basic train/test class to be inherited."""

    def __init__(self, args):
        """Initialize."""
        name = args.log_dir.split('/')[-1]
        self.tb_writer = None
        self.tb_log_dir = None
        # Create log dir
        args.log_dir = os.path.join(
            args.log_dir,
            ','.join(args.dataset),
            f'{int(time.time())}'
        )
        os.makedirs(args.log_dir, exist_ok=True)

        # Create logger
        self.logger = setup_logger(
            output=args.log_dir, distributed_rank=dist.get_rank(),
            name=name
        )
        # Save config file and initialize tb writer
        if dist.get_rank() == 0:
            path = os.path.join(args.log_dir, "config.json")
            with open(path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            self.logger.info("Full config saved to {}".format(path))
            self.logger.info(str(vars(args)))

            if args.tensorboard_root:
                rel_parts = [
                    part for part in os.path.normpath(args.log_dir).split(os.sep)
                    if part not in ('', '.')
                ]
                if rel_parts and rel_parts[0] == 'logs':
                    rel_parts = rel_parts[1:]
                self.tb_log_dir = os.path.join(args.tensorboard_root, *rel_parts)
                os.makedirs(self.tb_log_dir, exist_ok=True)
                if SummaryWriter is None:
                    self.logger.warning(
                        'TensorBoard requested but torch.utils.tensorboard is unavailable'
                    )
                else:
                    self.tb_writer = SummaryWriter(
                        log_dir=self.tb_log_dir,
                        flush_secs=args.tensorboard_flush_secs
                    )
                    self.tb_writer.add_text('meta/log_dir', args.log_dir, 0)
                    self.logger.info(
                        f'TensorBoard events will be written to {self.tb_log_dir}'
                    )

    @staticmethod
    def get_datasets(args, include_train=True):
        """Initialize datasets."""
        train_dataset = None
        test_dataset = None
        return train_dataset, test_dataset

    def get_loaders(self, args, include_train=True):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        def _joint_det_collate(batch):
            """Collate while preserving variable-length span annotations."""
            preserve_keys = {
                'target_slot', 'entity_spans', 'attr_spans', 'rel_spans',
                'coverage_stats', 'decomposition_status',
                'description', 'scene_id', 'object_id', 'object_name',
                'ann_id', 'attr_slot', 'rel_slots', 'anchor_slots',
                'slot_mask', 'parse_confidence',
                'decomp_global_only_mask', 'decomp_weak_generic_mask',
                'decomposition_error_flags_count',
                'global_only_due_to_parse_error',
                'target_generic_reference',
                'decomposition_error_flags',
                'metadata_conflict_examples',
            }
            collated = {}
            for key in batch[0].keys():
                values = [sample[key] for sample in batch]
                if key in preserve_keys:
                    # Keep nested lists (list[list[dict]]) untouched.
                    collated[key] = values
                else:
                    collated[key] = default_collate(values)
            return collated
        # Datasets
        train_dataset, test_dataset = self.get_datasets(
            args, include_train=include_train
        )
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)
        train_loader = None
        if include_train:
            train_sampler = DistributedSampler(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                worker_init_fn=seed_worker,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True,
                generator=g,
                collate_fn=_joint_det_collate,
                **_dataloader_parallelism_kwargs(args.num_workers)
            )
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            sampler=test_sampler,
            drop_last=False,
            generator=g,
            collate_fn=_joint_det_collate,
            **_dataloader_parallelism_kwargs(args.num_workers)
        )
        return train_loader, test_loader

    @staticmethod
    def get_model(args):
        """Initialize the model."""
        return None

    @staticmethod
    def get_criterion(args):
        """Get loss criterion for training."""
        matcher = HungarianMatcher(1, 0, 2, args.use_soft_token_loss)
        losses = ['boxes', 'labels']
        if args.use_contrastive_align:
            losses.append('contrastive_align')
        set_criterion = SetCriterion(
            matcher=matcher,
            losses=losses, eos_coef=0.1, temperature=0.07
        )
        criterion = compute_hungarian_loss

        return criterion, set_criterion

    @staticmethod
    def get_optimizer(args, model):
        """Initialize optimizer."""
        if getattr(args, 'universal_modules_train_only', False):
            shared_params = [
                p for p in model.parameters() if p.requires_grad
            ]
            if not shared_params:
                raise ValueError(
                    "--universal_modules_train_only has no trainable "
                    "shared module parameters"
                )
            optimizer = optim.AdamW(
                [{"params": shared_params, "lr": args.universal_modules_lr}],
                lr=args.universal_modules_lr,
                weight_decay=args.weight_decay,
            )
            return optimizer
        if (
            getattr(args, 'source_pool_selector_train_only', False)
            and getattr(args, 'detector_policy_adapter_train_only', False)
        ):
            selector_params = [
                p for name, p in model.named_parameters()
                if p.requires_grad and 'source_pool_selector' in name
            ]
            adapter_params = [
                p for name, p in model.named_parameters()
                if p.requires_grad and 'detector_policy_adapter' in name
            ]
            if not selector_params:
                raise ValueError(
                    "--source_pool_selector_train_only has no trainable "
                    "selector parameters"
                )
            if not adapter_params:
                raise ValueError(
                    "--detector_policy_adapter_train_only has no trainable "
                    "adapter parameters"
                )
            optimizer = optim.AdamW(
                [
                    {
                        "params": selector_params,
                        "lr": args.source_pool_selector_lr,
                    },
                    {
                        "params": adapter_params,
                        "lr": args.detector_policy_adapter_lr,
                    },
                ],
                lr=args.source_pool_selector_lr,
                weight_decay=args.weight_decay,
            )
            return optimizer
        if getattr(args, 'source_pool_selector_train_only', False):
            selector_params = [
                p for p in model.parameters() if p.requires_grad
            ]
            if not selector_params:
                raise ValueError(
                    "--source_pool_selector_train_only has no trainable "
                    "selector parameters"
                )
            optimizer = optim.AdamW(
                [{"params": selector_params, "lr": args.source_pool_selector_lr}],
                lr=args.source_pool_selector_lr,
                weight_decay=args.weight_decay,
            )
            return optimizer
        if getattr(args, 'detector_policy_adapter_train_only', False):
            adapter_params = [
                p for p in model.parameters() if p.requires_grad
            ]
            if not adapter_params:
                raise ValueError(
                    "--detector_policy_adapter_train_only has no trainable "
                    "adapter parameters"
                )
            optimizer = optim.AdamW(
                [{
                    "params": adapter_params,
                    "lr": args.detector_policy_adapter_lr,
                }],
                lr=args.detector_policy_adapter_lr,
                weight_decay=args.weight_decay,
            )
            return optimizer
        if getattr(args, 'acd_train_only', False):
            acd_params = [
                p for p in model.parameters() if p.requires_grad
            ]
            if not acd_params:
                raise ValueError(
                    "--acd_train_only has no trainable ACD/DHC parameters"
                )
            optimizer = optim.AdamW(
                [{"params": acd_params, "lr": args.acd_lr}],
                lr=args.acd_lr,
                weight_decay=args.weight_decay,
            )
            return optimizer
        param_dicts = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "backbone_net" not in n and "text_encoder" not in n
                    and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "backbone_net" in n and p.requires_grad
                ],
                "lr": args.lr_backbone
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "text_encoder" in n and p.requires_grad
                ],
                "lr": args.text_encoder_lr
            }
        ]
        optimizer = optim.AdamW(param_dicts,
                                lr=args.lr,
                                weight_decay=args.weight_decay)
        return optimizer

    def main(self, args):
        """Run main training/testing pipeline."""
        # Get loaders
        if args.eval:
            train_loader = None
            _, test_loader = self.get_loaders(args, include_train=False)
            n_data = len(test_loader.dataset)
            self.logger.info(f"length of testing dataset: {n_data}")
        else:
            train_loader, test_loader = self.get_loaders(args, include_train=True)
            n_data = len(train_loader.dataset)
            self.logger.info(f"length of training dataset: {n_data}")
            n_data = len(test_loader.dataset)
            self.logger.info(f"length of testing dataset: {n_data}")

        # Get model
        model = self.get_model(args)
        selector_param_count = _freeze_non_selector_parameters(args, model)
        if selector_param_count:
            self.logger.info(
                "source_pool_selector_train_only: trainable selector "
                "parameters {}".format(selector_param_count)
            )
        adapter_param_count = _freeze_non_detector_policy_adapter_parameters(
            args, model
        )
        if adapter_param_count:
            self.logger.info(
                "detector_policy_adapter_train_only: trainable adapter "
                "parameters {}".format(adapter_param_count)
            )
        acd_param_count = _freeze_non_acd_parameters(args, model)
        if acd_param_count:
            self.logger.info(
                "acd_train_only: trainable ACD/DHC parameters {}".format(
                    acd_param_count
                )
            )
        shared_module_param_count = _freeze_non_universal_module_parameters(
            args, model
        )
        if shared_module_param_count:
            self.logger.info(
                "universal_modules_train_only: trainable shared module "
                "parameters {}".format(shared_module_param_count)
            )
        rapf_frozen_param_count = _freeze_rapf_parameters(args, model)
        if rapf_frozen_param_count:
            self.logger.info(
                "freeze_rapf: frozen reliability_fusion parameters {}".format(
                    rapf_frozen_param_count
                )
            )
        quality_frozen_param_count = _freeze_quality_head_parameters(args, model)
        if quality_frozen_param_count:
            self.logger.info(
                "freeze_quality_head: frozen quality_head parameters {}".format(
                    quality_frozen_param_count
                )
            )

        # Get criterion
        criterion, set_criterion = self.get_criterion(args)

        optimizer = None
        scheduler = None
        if not args.eval:
            # Get optimizer
            optimizer = self.get_optimizer(args, model)

            # Get scheduler
            scheduler = get_scheduler(optimizer, len(train_loader), args)

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda()
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=(
                args.use_structured_slots or args.use_late_acd
                or args.use_sacr or args.use_rapf or args.use_quality_head
                or args.use_source_pool_selector
                or args.use_detector_policy_adapter
            )
        )

        # Just eval and end execution
        if args.eval:
            # Check for a checkpoint
            if args.checkpoint_path:
                assert os.path.isfile(args.checkpoint_path)
                load_checkpoint(args, model, optimizer, scheduler)
            print("Testing evaluation.....................")
            self.evaluate_one_epoch(
                args.start_epoch, test_loader,
                model, criterion, set_criterion, args
            )
            return

        # Check for a checkpoint
        if args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)

        # Training loop
        early_stopper = None
        if getattr(args, 'early_stopping', False):
            if not getattr(args, 'best_checkpoint_only', False):
                raise ValueError(
                    'early_stopping requires best_checkpoint_only so the '
                    'strict best model is retained for final evaluation'
                )
            early_stopper = ValidationEarlyStopper(
                metric=args.early_stopping_metric,
                min_epoch=args.early_stopping_min_epoch,
                patience=args.early_stopping_patience,
                min_delta=args.early_stopping_min_delta,
                max_epoch=args.max_epoch,
                val_freq=args.val_freq,
            )
            if dist.get_rank() == 0:
                write_early_stopping_receipt(
                    args, early_stopper.receipt('monitoring')
                )
                self.logger.info(
                    'Early stopping enabled: metric=%s min_epoch=%d '
                    'patience_validations=%d min_delta=%.6f val_freq=%d',
                    early_stopper.metric, early_stopper.min_epoch,
                    early_stopper.patience, early_stopper.min_delta,
                    early_stopper.val_freq,
                )
        early_stopped = False
        last_completed_epoch = args.start_epoch - 1
        for epoch in range(args.start_epoch, args.max_epoch + 1):
            train_loader.sampler.set_epoch(epoch)
            tic = time.time()
            self.train_one_epoch(
                epoch, train_loader, model,
                criterion, set_criterion,
                optimizer, scheduler, args
            )
            epoch_time = time.time() - tic
            last_completed_epoch = epoch
            lr_base = optimizer.param_groups[0]['lr']
            lr_pointnet = (
                optimizer.param_groups[1]['lr']
                if len(optimizer.param_groups) > 1 else lr_base
            )
            self.logger.info(
                'epoch {}, total time {:.2f}, '
                'lr_base {:.5f}, lr_pointnet {:.5f}'.format(
                    epoch, epoch_time,
                    lr_base,
                    lr_pointnet
                )
            )
            self._tb_write_scalars('epoch', {
                'time_sec': epoch_time,
                'lr_base': lr_base,
                'lr_pointnet': lr_pointnet
            }, epoch)
            if epoch % args.val_freq == 0:
                if (
                    dist.get_rank() == 0
                    and not getattr(args, 'best_checkpoint_only', False)
                ):
                    save_checkpoint(args, epoch, model, optimizer, scheduler)
                print("Test evaluation.......")
                eval_results = self.evaluate_one_epoch(
                    epoch, test_loader,
                    model, criterion, set_criterion, args
                )
                # Write evaluation results to log file
                if dist.get_rank() == 0 and eval_results is not None:
                    eval_log_path = os.path.join(args.log_dir, f'eval_epoch_{epoch}.log')
                    with open(eval_log_path, 'w') as f:
                        f.write(f'==============================================\n')
                        f.write(f'Evaluation Results - Epoch {epoch}\n')
                        f.write(f'==============================================\n\n')
                        for key, value in sorted(eval_results.items()):
                            if isinstance(value, (float, int)):
                                f.write(f'{key}: {value:.10f}\n')
                            else:
                                f.write(f'{key}: {value}\n')
                        f.write(f'\n==============================================\n')
                    self.logger.info(f'Evaluation results saved to {eval_log_path}')
                    self._tb_write_scalars('eval', eval_results, epoch)
                    if getattr(args, 'best_checkpoint_only', False):
                        save_best_checkpoint(
                            args, epoch, model, optimizer, scheduler,
                            eval_results,
                        )
                if early_stopper is not None:
                    stop_signal = torch.zeros(
                        1, dtype=torch.int64,
                        device=next(model.parameters()).device,
                    )
                    if dist.get_rank() == 0:
                        event = early_stopper.update(epoch, eval_results)
                        write_early_stopping_receipt(
                            args, early_stopper.receipt('monitoring')
                        )
                        stop_signal[0] = int(event['should_stop'])
                    dist.broadcast(stop_signal, src=0)
                    if bool(stop_signal.item()):
                        early_stopped = True
                        if dist.get_rank() == 0:
                            self.logger.info(
                                'Early stopping triggered at epoch %d after '
                                '%d stale validations; reference %s=%.6f '
                                'at epoch %d',
                                epoch, early_stopper.stale_validations,
                                early_stopper.metric,
                                early_stopper.reference_score,
                                early_stopper.reference_epoch,
                            )
                        break


        # Training is over, evaluate
        final_eval_epoch = args.max_epoch
        if getattr(args, 'best_checkpoint_only', False):
            saved_path = os.path.join(args.log_dir, 'ckpt_best_primary.pth')
            if not os.path.isfile(saved_path):
                raise RuntimeError(
                    'best_checkpoint_only produced no best checkpoint; '
                    'ensure validation ran and the configured metric exists'
                )
            self.logger.info("Best checkpoint saved in {}".format(saved_path))
            checkpoint = torch.load(saved_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=True)
            final_eval_epoch = int(checkpoint['epoch'])
            self.logger.info(
                "Reloaded best-primary model from {} for final evaluation".format(
                    saved_path
                )
            )
        else:
            save_checkpoint(args, 'last', model, optimizer, scheduler, True)
            saved_path = os.path.join(args.log_dir, 'ckpt_epoch_last.pth')
            self.logger.info("Saved in {}".format(saved_path))
        eval_results = self.evaluate_one_epoch(
            final_eval_epoch, test_loader,
            model, criterion, set_criterion, args
        )
        # Write final evaluation results to log file
        if dist.get_rank() == 0 and eval_results is not None:
            eval_log_path = os.path.join(args.log_dir, 'eval_epoch_last.log')
            with open(eval_log_path, 'w') as f:
                f.write(f'==============================================\n')
                f.write(
                    f'Final Evaluation Results - Selected Epoch '
                    f'{final_eval_epoch}\n'
                )
                f.write(f'==============================================\n\n')
                for key, value in sorted(eval_results.items()):
                    if isinstance(value, (float, int)):
                        f.write(f'{key}: {value:.10f}\n')
                    else:
                        f.write(f'{key}: {value}\n')
                f.write(f'\n==============================================\n')
            self.logger.info(f'Final evaluation results saved to {eval_log_path}')
            self._tb_write_scalars('eval_final', eval_results, final_eval_epoch)
            if early_stopper is not None:
                final_status = (
                    'early_stopped' if early_stopped else 'max_epoch'
                )
                write_early_stopping_receipt(
                    args,
                    early_stopper.receipt(
                        final_status, stop_epoch=last_completed_epoch
                    ),
                )
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        return saved_path

    def _tb_write_scalars(self, prefix, scalars, step):
        if self.tb_writer is None or dist.get_rank() != 0:
            return
        for key, value in scalars.items():
            try:
                scalar = self._stat_to_float(value)
            except (TypeError, ValueError):
                continue
            self.tb_writer.add_scalar(f'{prefix}/{key}', scalar, step)

    @staticmethod
    def _to_gpu(data_dict):
        if torch.cuda.is_available():
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
        return data_dict

    @staticmethod
    def _get_inputs(batch_data):
        inputs = {
            'point_clouds': batch_data['point_clouds'].float(),
            'text': batch_data['utterances']
        }
        if 'all_detected_logits' in batch_data:
            inputs['det_logits'] = batch_data['all_detected_logits']
        # Add spans if available for structured slot building.
        if 'entity_spans' in batch_data:
            inputs['entity_spans'] = batch_data['entity_spans']
        if 'attr_spans' in batch_data:
            inputs['attr_spans'] = batch_data['attr_spans']
        if 'rel_spans' in batch_data:
            inputs['rel_spans'] = batch_data['rel_spans']
        if 'anchor_ids' in batch_data:
            inputs['anchor_ids'] = batch_data['anchor_ids'].long()
        for key in (
            'coverage_stats',
            'decomposition_status',
            'description',
            'scene_id',
            'object_id',
            'object_name',
            'ann_id',
            'attr_slot',
            'rel_slots',
            'anchor_slots',
            'slot_mask',
            'parse_confidence',
            'decomp_global_only_mask',
            'decomp_weak_generic_mask',
            'decomposition_error_flags_count',
            'global_only_due_to_parse_error',
            'target_generic_reference',
            'decomposition_error_flags',
            'metadata_conflict_examples',
            'spacy_rotation_mode_id',
            'spacy_augmentation_profile_id',
            'target_cid',
            'text_target_cid',
            'positive_map',
            'box_label_mask',
        ):
            if key in batch_data:
                inputs[key] = batch_data[key]
        return inputs

    @staticmethod
    def _compute_loss(end_points, criterion, set_criterion, args,
                      epoch=None, batch_idx=None, steps_per_epoch=None,
                      dhc_module=None, true_global_step=None):
        quality_topk_rerank_source = getattr(
            args, 'quality_topk_rerank_source', 'fused'
        )
        quality_topk_rerank_weight = getattr(
            args, 'quality_topk_rerank_weight', 0.0
        )
        if (
            epoch is None
            and quality_topk_rerank_source in DETECTOR_POLICY_SOURCE_NAMES
        ):
            quality_topk_rerank_weight = 0.0
        loss, end_points = criterion(
            end_points, args.num_decoder_layers,
            set_criterion,
            query_points_obj_topk=args.query_points_obj_topk,
            use_s2s_aux_loss=getattr(args, 'use_s2s_aux_loss', False),
            s2s_aux_weight=getattr(args, 's2s_aux_weight', 1.0),
            acd_rank_weight=getattr(args, 'acd_rank_weight', 1.0),
            use_quality_head=getattr(args, 'use_quality_head', False),
            quality_loss_weight=getattr(args, 'quality_loss_weight', 1.0),
            quality_iou_threshold=getattr(args, 'quality_iou_threshold', 0.25),
            use_sacr=getattr(args, 'use_sacr', False),
            sacr_rank_loss_weight=getattr(args, 'sacr_rank_loss_weight', 0.0),
            sacr_rank_margin=getattr(args, 'sacr_rank_margin', 0.2),
            use_rapf=getattr(args, 'use_rapf', False),
            use_reliability_gate=getattr(args, 'use_reliability_gate', False),
            rapf_gate_loss_weight=getattr(args, 'rapf_gate_loss_weight', 0.2),
            rapf_gate_iou_margin=getattr(args, 'rapf_gate_iou_margin', 0.02),
            use_qahnl=getattr(args, 'use_qahnl', False),
            qahnl_config={
                'score_source': getattr(args, 'qahnl_score_source', 'fused'),
                'num_hard_neg': getattr(args, 'qahnl_num_hard_neg', 16),
                'pos_iou_thresh': getattr(args, 'qahnl_pos_iou_thresh', 0.25),
                'neg_iou_thresh': getattr(args, 'qahnl_neg_iou_thresh', 0.10),
                'topk_iou_pos': getattr(args, 'qahnl_topk_iou_pos', 3),
                'margin_base': getattr(args, 'qahnl_margin_base', 0.2),
                'margin_iou_lambda': getattr(args, 'qahnl_margin_iou_lambda', 0.5),
                'margin_min': getattr(args, 'qahnl_margin_min', 0.05),
                'margin_max': getattr(args, 'qahnl_margin_max', 0.5),
                'temperature': getattr(args, 'qahnl_temperature', 1.0),
                'temperature_max': getattr(args, 'qahnl_temperature_max', 6.0),
                'loss_weight': getattr(args, 'qahnl_loss_weight', 0.2),
                'use_entity_hardneg': getattr(args, 'qahnl_use_entity_hardneg', False),
                'use_attr_hardneg': getattr(args, 'qahnl_use_attr_hardneg', False),
                'use_relation_hardneg': getattr(args, 'qahnl_use_relation_hardneg', False),
            } if getattr(args, 'use_qahnl', False) else None,
            quality_topk_rerank_weight=quality_topk_rerank_weight,
            quality_topk_rerank_source=quality_topk_rerank_source,
            quality_topk_rerank_k=getattr(
                args, 'quality_topk_rerank_k', 5
            ),
            quality_topk_rerank_margin=getattr(
                args, 'quality_topk_rerank_margin', 0.05
            ),
            quality_topk_rerank_min_iou_gap=getattr(
                args, 'quality_topk_rerank_min_iou_gap', 0.02
            ),
            source_pool_selector_loss_weight=getattr(
                args, 'source_pool_selector_loss_weight', 0.0
            ),
            source_pool_selector_source=getattr(
                args, 'source_pool_selector_source', 'source_pool'
            ),
            source_pool_selector_k=getattr(
                args, 'source_pool_selector_k', 5
            ),
            source_pool_selector_temperature=getattr(
                args, 'source_pool_selector_temperature', 1.0
            ),
            source_pool_selector_min_iou_gap=getattr(
                args, 'source_pool_selector_min_iou_gap', 0.02
            ),
            source_pool_selector_source_min_iou_gaps=getattr(
                args, 'source_pool_selector_source_min_iou_gaps', None
            ),
            source_pool_selector_pairwise_weight=getattr(
                args, 'source_pool_selector_pairwise_weight', 0.5
            ),
            source_pool_selector_choice_balance=getattr(
                args, 'source_pool_selector_choice_balance', False
            ),
            source_pool_selector_choice_balance_power=getattr(
                args, 'source_pool_selector_choice_balance_power', 1.0
            ),
            source_pool_selector_choice_target=getattr(
                args, 'source_pool_selector_choice_target', 'iou'
            ),
            source_pool_selector_override_default_source=getattr(
                args, 'source_pool_selector_override_default_source', 'base'
            ),
            source_pool_selector_oracle_prior_weight=getattr(
                args, 'source_pool_selector_oracle_prior_weight', 0.0
            ),
            source_pool_selector_override_prior_weight=getattr(
                args, 'source_pool_selector_override_prior_weight', 0.0
            ),
            source_pool_selector_override_source_prior_weight=getattr(
                args,
                'source_pool_selector_override_source_prior_weight',
                0.0,
            ),
            source_pool_selector_override_margin_weight=getattr(
                args,
                'source_pool_selector_override_margin_weight',
                0.0,
            ),
            source_pool_selector_override_margin=getattr(
                args, 'source_pool_selector_override_margin', 1.0
            ),
            source_pool_selector_quality_base_margin_weight=getattr(
                args,
                'source_pool_selector_quality_base_margin_weight',
                0.0,
            ),
            source_pool_selector_quality_base_margin=getattr(
                args,
                'source_pool_selector_quality_base_margin',
                0.75,
            ),
            source_pool_selector_quality_default_margin_weight=getattr(
                args,
                'source_pool_selector_quality_default_margin_weight',
                0.0,
            ),
            source_pool_selector_quality_default_margin=getattr(
                args,
                'source_pool_selector_quality_default_margin',
                0.75,
            ),
            source_pool_selector_quality_default_bidirectional_margin_weight=getattr(
                args,
                'source_pool_selector_quality_default_bidirectional_margin_weight',
                0.0,
            ),
            source_pool_selector_quality_default_bidirectional_margin=getattr(
                args,
                'source_pool_selector_quality_default_bidirectional_margin',
                0.75,
            ),
            source_pool_selector_iou_aux_weight=getattr(
                args, 'source_pool_selector_iou_aux_weight', 0.0
            ),
            source_pool_selector_iou_aux_margin=getattr(
                args, 'source_pool_selector_iou_aux_margin', 0.5
            ),
            source_pool_selector_false_base_weight=getattr(
                args, 'source_pool_selector_false_base_weight', 1.0
            ),
            source_pool_selector_false_override_weight=getattr(
                args, 'source_pool_selector_false_override_weight', 1.0
            ),
            source_pool_selector_sourcewise_negative_weight=getattr(
                args,
                'source_pool_selector_sourcewise_negative_weight',
                1.0,
            ),
            source_pool_selector_override_utility_gap_weight=getattr(
                args,
                'source_pool_selector_override_utility_gap_weight',
                0.0,
            ),
            detector_policy_adapter_loss_weight=getattr(
                args, 'detector_policy_adapter_loss_weight', 0.0
            ),
            detector_policy_adapter_k=getattr(
                args, 'detector_policy_adapter_k', 5
            ),
            detector_policy_adapter_margin=getattr(
                args, 'detector_policy_adapter_margin', 0.05
            ),
            detector_policy_adapter_min_iou_gap=getattr(
                args, 'detector_policy_adapter_min_iou_gap', 0.02
            ),
            detector_policy_adapter_reg_weight=getattr(
                args, 'detector_policy_adapter_reg_weight', 0.0
            ),
            use_dhc=getattr(args, 'use_dhc', False),
            dhc_config={
                'dhc_consistency_weight': getattr(args, 'dhc_consistency_weight', 0.2),
                'dhc_ent_hardneg_weight': getattr(args, 'dhc_ent_hardneg_weight', 0.2),
                'dhc_attr_hardneg_weight': getattr(args, 'dhc_attr_hardneg_weight', 0.2),
                'dhc_rel_hardneg_weight': getattr(args, 'dhc_rel_hardneg_weight', 0.2),
            } if getattr(args, 'use_dhc', False) else None,
            dhc_module=dhc_module
        )
        return loss, end_points

    @staticmethod
    def _accumulate_stats(stat_dict, end_points):
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key \
                    or key.startswith('avg_') \
                    or key.endswith('_active_rate') \
                    or (key.startswith('L') and ('gate_' in key or '_valid_ratio' in key)) \
                    or key.startswith('dbg_') \
                    or key.startswith('diag_') \
                    or key.startswith('eval_'):
                if isinstance(end_points[key], (list, str)):
                    continue
                if key not in stat_dict:
                    stat_dict[key] = 0
                if isinstance(end_points[key], (float, int)):
                    stat_dict[key] += end_points[key]
                else:
                    value = end_points[key].detach()
                    if torch.is_tensor(value) and value.numel() > 1:
                        value = value.float().mean()
                    stat_dict[key] += value
        return stat_dict

    @staticmethod
    def _stat_to_float(value):
        if isinstance(value, (float, int)):
            return float(value)
        if torch.is_tensor(value):
            return float(value.detach().cpu())
        return float(value)

    @staticmethod
    def _is_core_diagnostic_key(key):
        core_keys = {
            'metadata_conflict_ratio',
            'dbg_metadata_conflict_ratio',
            'positive_map_target_missing_ratio',
            'positive_map_fallback_used_ratio',
            'positive_map_global_only_target_empty_ratio',
            'positive_map_source_missing_ratio',
            'dbg_sacr_structured_valid_ratio',
            'dbg_sacr_global_only_ratio',
            'dbg_sacr_weak_generic_ratio',
            'dbg_sacr_relation_active_ratio',
            'dbg_rapf_gate_mean',
            'dbg_rapf_gate_std',
            'dbg_rapf_gate_min',
            'dbg_rapf_gate_max',
            'dbg_rapf_residual_clip_ratio',
            'dbg_rapf_iou_delta_mean',
            'dbg_rapf_wrong_to_right_ratio',
            'dbg_rapf_right_to_wrong_ratio',
            'dbg_quality_corr_pred_iou_target_iou',
            'dbg_rapf_quality_anchor_enabled',
            'dbg_rapf_safe_score_used',
            'dbg_quality_iou_corr',
            'dbg_quality_pred_target_iou_corr',
            'dbg_quality_top1_quality_improves_ratio',
            'dbg_quality_top1_iou_improvement',
            'dbg_qahnl_ambiguous_as_negative_ratio',
            'dbg_qahnl_pos_query_ratio',
            'dbg_qahnl_neg_query_ratio',
            'dbg_qahnl_iou_gap_mean',
            'dbg_qahnl_score_gap_mean',
            'dbg_qahnl_valid_batch_ratio',
            'dbg_qahnl_violation_ratio',
            'dbg_qahnl_global_only_used_ratio',
            'dbg_qahnl_global_only_skipped_structured_ratio',
            'dbg_qahnl_weak_generic_used_ratio',
            'dbg_source_pool_selector_valid_ratio',
            'dbg_source_pool_selector_violation_ratio',
            'dbg_source_pool_selector_score_gap',
            'dbg_source_pool_selector_pos_iou',
            'dbg_source_pool_selector_selected_iou',
            'dbg_source_pool_selector_oracle_prior_loss',
            'dbg_source_pool_selector_oracle_prior_weight',
            'dbg_source_pool_selector_override_prior_loss',
            'dbg_source_pool_selector_override_prior_weight',
            'dbg_source_pool_selector_override_source_prior_loss',
            'dbg_source_pool_selector_override_source_prior_weight',
            'dbg_source_pool_selector_override_margin_loss',
            'dbg_source_pool_selector_override_margin_weight',
            'dbg_source_pool_selector_override_margin',
            'dbg_source_pool_selector_quality_base_margin_loss',
            'dbg_source_pool_selector_quality_base_margin_weight',
            'dbg_source_pool_selector_quality_base_margin',
            'dbg_source_pool_selector_quality_base_margin_valid_ratio',
            'dbg_source_pool_selector_quality_base_margin_score_gap',
            'dbg_source_pool_selector_quality_base_margin_iou_gap',
            'dbg_source_pool_selector_quality_default_margin_loss',
            'dbg_source_pool_selector_quality_default_margin_weight',
            'dbg_source_pool_selector_quality_default_margin',
            'dbg_source_pool_selector_quality_default_margin_valid_ratio',
            'dbg_source_pool_selector_quality_default_margin_score_gap',
            'dbg_source_pool_selector_quality_default_margin_iou_gap',
            'dbg_source_pool_selector_quality_default_bidirectional_margin_loss',
            'dbg_source_pool_selector_quality_default_bidirectional_margin_weight',
            'dbg_source_pool_selector_quality_default_bidirectional_margin',
            'dbg_source_pool_selector_quality_default_bidirectional_margin_valid_ratio',
            'dbg_source_pool_selector_quality_default_bidirectional_margin_score_gap',
            'dbg_source_pool_selector_quality_default_bidirectional_margin_iou_gap',
            'dbg_source_pool_selector_iou_aux_loss',
            'dbg_source_pool_selector_iou_aux_weight',
            'dbg_source_pool_selector_iou_aux_margin',
            'dbg_source_pool_selector_iou_aux_valid_ratio',
            'dbg_source_pool_selector_iou_aux_score_gap',
            'dbg_source_pool_selector_iou_aux_iou_gap',
            'dbg_source_pool_selector_override_utility_gap',
            'dbg_source_pool_selector_sourcewise_negative_weight',
            'dbg_source_pool_selector_target_base_ratio',
            'dbg_source_pool_selector_target_fused_ratio',
            'dbg_source_pool_selector_target_quality_ratio',
            'dbg_source_pool_selector_target_contrastive_base_ratio',
            'dbg_source_pool_selector_selected_base_ratio',
            'dbg_source_pool_selector_selected_fused_ratio',
            'dbg_source_pool_selector_selected_quality_ratio',
            'dbg_source_pool_selector_selected_contrastive_base_ratio',
            'dbg_source_pool_selector_target_override_ratio',
            'dbg_source_pool_selector_selected_override_ratio',
            'dbg_source_pool_selector_diffquery_valid_ratio',
            'dbg_source_pool_selector_choice_target_threshold_gain_default_diffquery_sourcewise_focal_bce',
            'dbg_source_pool_selector_choice_target_precision_gain_default_sourcewise_focal_bce',
            'dbg_warn_global_only_target_positive_map_ratio',
            'eval_primary_score_source_id',
            'eval_bbs_score_source_id',
            'eval_bbf_score_source_id',
            'eval_target_row_uses_primary_score',
            'eval_anchor_rows_use_baseline_score',
            'eval_bbf_is_diagnostic_only',
        }
        core_prefixes = (
            'dbg_warn_',
            'dbg_data_',
            'dbg_positive_map_',
            'dbg_metadata_conflict_',
        )
        return key in core_keys or any(
            key.startswith(prefix) for prefix in core_prefixes
        )

    @classmethod
    def _diagnostic_keys_for_logging(cls, stat_dict, args):
        keys = [
            key for key in sorted(stat_dict.keys())
            if (
                key.startswith('dbg_')
                or key.startswith('diag_')
                or key.startswith('eval_')
                or key.startswith('positive_map_')
                or key.startswith('metadata_conflict_')
            )
        ]
        if getattr(args, 'verbose_diagnostics', False):
            return keys
        return [key for key in keys if cls._is_core_diagnostic_key(key)]

    @classmethod
    def _log_diagnostics(cls, logger, stat_dict, denom, args, tag):
        diag_keys = cls._diagnostic_keys_for_logging(stat_dict, args)
        if not diag_keys:
            return
        warn_keys = [key for key in diag_keys if key.startswith('dbg_warn_')]
        metric_keys = [key for key in diag_keys if key not in warn_keys]
        if metric_keys:
            logger.info('[{}-diagnostics] {}'.format(tag, '  '.join([
                '{}={:.4f}'.format(
                    key, cls._stat_to_float(stat_dict[key]) / denom
                )
                for key in metric_keys
            ])))
        active_warns = [
            key for key in warn_keys
            if (cls._stat_to_float(stat_dict[key]) / denom) > 0.0
        ]
        if active_warns:
            logger.warning('[{}-warnings] {}'.format(tag, '  '.join([
                '{}={:.4f}'.format(
                    key, cls._stat_to_float(stat_dict[key]) / denom
                )
                for key in active_warns
            ])))

    def train_one_epoch(self, epoch, train_loader, model,
                        criterion, set_criterion,
                        optimizer, scheduler, args):
        """
        Run a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        stat_dict = {}  # collect statistics
        model.train()  # set model to training mode
        if (
            getattr(args, 'source_pool_selector_train_only', False)
            and getattr(args, 'detector_policy_adapter_train_only', False)
        ):
            _set_non_selector_adapter_modules_eval(model)
        elif getattr(args, 'source_pool_selector_train_only', False):
            _set_non_selector_modules_eval(model)
        elif getattr(args, 'detector_policy_adapter_train_only', False):
            _set_non_detector_policy_adapter_modules_eval(model)
        if getattr(args, 'acd_train_only', False):
            _set_non_acd_modules_eval(model)
        if getattr(args, 'universal_modules_train_only', False):
            _set_non_universal_modules_eval(model)

        # Enable per-step gate stats collection in debug mode.
        # Always write the flag so that a previous True from an earlier epoch
        # or from train→eval within the same process is properly cleared.
        _debug_decomposed = False
        _m = model.module if hasattr(model, 'module') else model
        _m._collect_gate_stats = _debug_decomposed

        # Initialize AMP scaler if enabled
        scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

        # Calculate steps per epoch for warmup
        steps_per_epoch = len(train_loader)

        # Loop over batches
        for batch_idx, batch_data in enumerate(train_loader):
            # Move to GPU
            batch_data = self._to_gpu(batch_data)
            inputs = self._get_inputs(batch_data)

            # Absolute training step (epochs are 1-indexed, so epoch-1 gives 0-based count).
            # This is resume-safe: epoch 11 → step (11-1)*N+idx, warmup already passed.
            _true_global_step = (epoch - 1) * steps_per_epoch + batch_idx

            # Set ACD global step for warmup (read inside forward via getattr)
            _base_model = model.module if hasattr(model, 'module') else model
            _base_model._acd_global_step = _true_global_step

            # Forward pass with AMP
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    end_points = model(inputs)
                    # Compute loss
                    for key in batch_data:
                        if key not in end_points:
                            end_points[key] = batch_data[key]
                    loss, end_points = self._compute_loss(
                        end_points, criterion, set_criterion, args,
                        epoch=epoch, batch_idx=batch_idx, steps_per_epoch=steps_per_epoch,
                        dhc_module=getattr(_base_model, 'dhc_loss_module', None),
                        true_global_step=_true_global_step
                    )
            else:
                # Forward pass without AMP
                end_points = model(inputs)
                # Compute loss
                for key in batch_data:
                    if key not in end_points:
                        end_points[key] = batch_data[key]
                loss, end_points = self._compute_loss(
                    end_points, criterion, set_criterion, args,
                    epoch=epoch, batch_idx=batch_idx, steps_per_epoch=steps_per_epoch,
                    dhc_module=getattr(_base_model, 'dhc_loss_module', None),
                    true_global_step=_true_global_step
                )

            # Backward pass with AMP
            optimizer.zero_grad(set_to_none=True)
            stepped = False
            if args.use_amp:
                scaler.scale(loss).backward()
                if args.clip_norm > 0:
                    scaler.unscale_(optimizer)
                    grad_total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_norm
                    )
                    stat_dict['grad_norm'] = grad_total_norm
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= scale_before
            else:
                loss.backward()
                if args.clip_norm > 0:
                    grad_total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_norm
                    )
                    stat_dict['grad_norm'] = grad_total_norm
                optimizer.step()
                stepped = True

            if stepped:
                scheduler.step()

            # Accumulate statistics and print out
            stat_dict = self._accumulate_stats(stat_dict, end_points)

            if (batch_idx + 1) % args.print_freq == 0:
                global_step = _true_global_step + 1
                averaged_stats = {
                    key: self._stat_to_float(stat_dict[key]) / args.print_freq
                    for key in sorted(stat_dict.keys())
                }
                # Terminal logs
                self.logger.info(
                    f'Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  '
                )
                self.logger.info(''.join([
                    f'{key} {averaged_stats[key]:.4f} \t'
                    for key in sorted(averaged_stats.keys())
                    if 'loss' in key and 'proposal_' not in key
                    and 'last_' not in key and 'head_' not in key
                ]))
                self._tb_write_scalars('train', averaged_stats, global_step)
                lr_base = optimizer.param_groups[0]['lr']
                lr_pointnet = (
                    optimizer.param_groups[1]['lr']
                    if len(optimizer.param_groups) > 1 else lr_base
                )
                self._tb_write_scalars('train_lr', {
                    'base': lr_base,
                    'pointnet': lr_pointnet
                }, global_step)

                # Debug: log decomposed gate statistics (only when enabled)
                # Keys follow L{i}_gate_{type}_mean / L{i}_{type}_valid_ratio
                if _debug_decomposed:
                    gate_keys = [
                        k for k in sorted(stat_dict.keys())
                        if k.startswith('L') and ('gate_' in k or '_valid_ratio' in k)
                    ]
                    if gate_keys:
                        self.logger.info('[decomposed] ' + '  '.join([
                            f'{k}='
                            f'{self._stat_to_float(stat_dict[k]) / args.print_freq:.4f}'
                            for k in gate_keys
                        ]))

                self._log_diagnostics(
                    self.logger, stat_dict, float(args.print_freq), args, 'train'
                )

                for key in sorted(stat_dict.keys()):
                    stat_dict[key] = 0

    @torch.no_grad()
    def _main_eval_branch(self, batch_idx, batch_data, test_loader, model,
                          stat_dict,
                          criterion, set_criterion, args):
        # Ensure gate stats collection is off during eval to avoid
        # .item() CUDA sync overhead (it may have been left on by training).
        if batch_idx == 0:
            _m = model.module if hasattr(model, 'module') else model
            _m._collect_gate_stats = False
            # Reset ACD step so eval uses full alpha (no warmup)
            _m._acd_global_step = None

        # Move to GPU
        batch_data = self._to_gpu(batch_data)
        inputs = self._get_inputs(batch_data)
        if "train" not in inputs:
            inputs.update({"train": False})
        else:
            inputs["train"] = False
        inputs["eval_primary_score_source"] = getattr(
            args, "eval_primary_score_source", "base"
        )
        inputs["eval_target_cid_source"] = getattr(
            args, "eval_target_cid_source", "gt"
        )

        # Forward pass
        end_points = model(inputs)

        # Compute loss
        for key in batch_data:
            if key not in end_points:
                end_points[key] = batch_data[key]
        _base = model.module if hasattr(model, 'module') else model
        _, end_points = self._compute_loss(
            end_points, criterion, set_criterion, args,
            dhc_module=getattr(_base, 'dhc_loss_module', None)
        )
        explicit_primary_source = getattr(
            args, 'eval_primary_score_source', 'base'
        )
        explicit_primary_override = explicit_primary_source != 'base'
        end_points['eval_use_acd_scores'] = bool(
            getattr(args, 'eval_use_acd_scores', False)
        ) and not explicit_primary_override
        end_points['eval_use_structured_scores'] = bool(
            getattr(args, 'eval_use_structured_scores', False)
        ) and not explicit_primary_override
        end_points['eval_use_quality_scores'] = bool(
            getattr(args, 'eval_use_quality_scores', False)
        ) and not explicit_primary_override
        end_points['eval_use_fused_scores'] = bool(
            getattr(args, 'eval_use_fused_scores', False)
        ) and not explicit_primary_override
        end_points['eval_use_selector_scores'] = bool(
            getattr(args, 'eval_use_selector_scores', False)
        ) and not explicit_primary_override
        end_points['eval_use_selector_pool_scores'] = bool(
            getattr(args, 'eval_use_selector_pool_scores', False)
        ) and not explicit_primary_override
        end_points['eval_use_selector_choice_scores'] = bool(
            getattr(args, 'eval_use_selector_choice_scores', False)
        ) and not explicit_primary_override
        end_points['eval_use_selector_choice_hybrid_scores'] = bool(
            getattr(args, 'eval_use_selector_choice_hybrid_scores', False)
        ) and not explicit_primary_override
        end_points['eval_use_selector_choice_quality_override_scores'] = bool(
            getattr(args, 'eval_use_selector_choice_quality_override_scores', False)
        ) and not explicit_primary_override
        end_points['eval_use_detector_policy_adapter_scores'] = bool(
            getattr(args, 'eval_use_detector_policy_adapter_scores', False)
        ) and not explicit_primary_override
        end_points['eval_primary_score_source'] = getattr(
            args, 'eval_primary_score_source', 'base'
        )
        end_points['eval_selector_pool_k'] = int(
            getattr(args, 'eval_selector_pool_k', 1)
        )
        end_points['eval_selector_choice_min_margin'] = float(
            getattr(args, 'eval_selector_choice_min_margin', 0.0)
        )
        end_points['eval_selector_choice_hybrid_fallback'] = getattr(
            args, 'eval_selector_choice_hybrid_fallback', 'quality'
        )
        for source in (
            'base',
            'fused',
            'quality',
            'contrastive_base',
            'detector_countboost',
            'detector_run174boost',
            'detector_countsplit',
            'detector_countsplit_lowonly',
            'detector_countsplit_guarded',
            'detector_countsplit_guarded_allcount',
            'detector_jointtight',
            'detector_strongcoarse',
            'detector_confblend035',
            'detector_confblend05',
        ):
            key = f'eval_selector_choice_source_bias_{source}'
            end_points[key] = float(getattr(args, key, 0.0))
        end_points['eval_selector_choice_use_override_head'] = bool(
            getattr(args, 'eval_selector_choice_use_override_head', False)
        )
        end_points['eval_selector_choice_override_threshold'] = float(
            getattr(args, 'eval_selector_choice_override_threshold', 0.0)
        )
        end_points['eval_selector_choice_override_default_source'] = getattr(
            args, 'eval_selector_choice_override_default_source', 'base'
        )
        end_points['eval_report_diagnostic_scores'] = bool(
            getattr(args, 'eval_report_diagnostic_scores', False)
        )
        end_points['eval_target_cid_source'] = getattr(
            args, 'eval_target_cid_source', 'gt'
        )
        end_points['eval_report_spacy_source_scores'] = bool(
            getattr(args, 'eval_report_spacy_source_scores', False)
        )
        end_points['eval_spacy_source_score_sources'] = getattr(
            args,
            'eval_spacy_source_score_sources',
            (
                'base,contrastive_base,quality,fused,detector_jointtight,'
                'detector_countsplit_guarded_allcount,detector_strongcoarse'
            ),
        )
        for key in end_points:
            if 'pred_size' in key:
                end_points[key] = torch.clamp(end_points[key], min=1e-6)

        # Accumulate statistics and print out
        stat_dict = self._accumulate_stats(stat_dict, end_points)
        if (batch_idx + 1) % args.print_freq == 0:
            denom = float(batch_idx + 1)
            self.logger.info(f'Eval: [{batch_idx + 1}/{len(test_loader)}]  ')
            self.logger.info(''.join([
                f'{key} {self._stat_to_float(stat_dict[key]) / denom:.4f} \t'
                for key in sorted(stat_dict.keys())
                if 'loss' in key and 'proposal_' not in key
                and 'last_' not in key and 'head_' not in key
            ]))
            self._log_diagnostics(self.logger, stat_dict, denom, args, 'eval')
        return stat_dict, end_points

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader,
                           model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        return None
