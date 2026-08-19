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

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from models import HungarianMatcher, SetCriterion, compute_hungarian_loss
from utils import get_scheduler, setup_logger

from utils import record_tensorboard

from tqdm import tqdm

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
    parser.add_argument('--num_decoder_layers', default=6, type=int)    # 6
    parser.add_argument('--self_position_embedding', default='loc_learned',
                        type=str, help='(none, xyz_learned, loc_learned)')
    parser.add_argument('--self_attend', action='store_true')
    parser.add_argument(
        '--use_spatial_backbone_adapter', action='store_true', default=False,
        help=(
            'Use checkpoint-compatible language-conditioned spatial '
            'self-attention in the EDA encoder and decoder.'
        ),
    )

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
                        help='Use RGB color in input.')     # color
    parser.add_argument('--use_multiview', action='store_true')
    parser.add_argument('--wo_obj_name', default='None')    # grounding without object name
    parser.add_argument('--butd', action='store_true')
    parser.add_argument('--butd_gt', action='store_true')
    parser.add_argument('--butd_cls', action='store_true')
    parser.add_argument('--augment_det', action='store_true')
    parser.add_argument('--disable_box_jitter', action='store_true')
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
            'For _spacy relation-free samples already routed away from full '
            'natural augmentation, keep yaw/flip geometry but suppress point, '
            'global shift/scale, and color jitter.'
        ),
    )
    parser.add_argument(
        '--spacy_relation_free_view_small_yaw_aug',
        action='store_true',
        default=False,
        help=(
            'For _spacy relation-free samples routed to yaw-only, route explicit '
            'raw view-word samples to small yaw jitter without 90-degree rotation '
            'or flips.'
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
    parser.add_argument('--num_workers', type=int, default=4)

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

    # io
    parser.add_argument('--checkpoint_path', default=None,
                        help='Model checkpoint path')
    parser.add_argument(
        '--train_partial_checkpoint_init',
        action='store_true',
        default=False,
        help=(
            'Initialize training from compatible checkpoint tensors only; '
            'keeps missing/new model tensors at their current initialization.'
        ),
    )
    parser.add_argument('--log_dir', default='log',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--print_freq', type=int, default=10)  # batch-wise
    parser.add_argument('--save_freq', type=int, default=10)  # epoch-wise
    parser.add_argument('--val_freq', type=int, default=5)  # epoch-wise

    # SACR / RAPF / QA-HNL
    parser.add_argument('--use_structured_slots', action='store_true', default=False,
                        help='Enable offline span-to-slot structured text memory.')
    parser.add_argument('--slot_pooling', type=str, default='attention',
                        choices=['mean', 'attention'],
                        help='Pooling method for structured text spans.')
    parser.add_argument('--max_rel_anchor_pairs', type=int, default=3,
                        help='Maximum relation-anchor pairs for structured slots.')
    parser.add_argument('--structured_debug', action='store_true', default=False,
                        help='Log structured slot diagnostics.')
    parser.add_argument('--use_quality_head', action='store_true', default=False,
                        help='Enable query IoU quality head.')
    parser.add_argument('--quality_loss_weight', type=float, default=1.0,
                        help='Weight for quality IoU supervision.')
    parser.add_argument('--quality_iou_threshold', type=float, default=0.25,
                        help='IoU threshold for quality diagnostics.')
    parser.add_argument('--use_sacr', action='store_true', default=False,
                        help='Enable structured anchor-compositional reasoning.')
    parser.add_argument('--sacr_top_m_targets', type=int, default=32)
    parser.add_argument('--sacr_top_k_anchors', type=int, default=16)
    parser.add_argument('--sacr_hidden_dim', type=int, default=288)
    parser.add_argument('--sacr_geo_dim', type=int, default=16)
    parser.add_argument('--sacr_disable_relation', action='store_true', default=False)
    parser.add_argument('--sacr_rank_loss_weight', type=float, default=0.0)
    parser.add_argument('--sacr_rank_margin', type=float, default=0.2)
    parser.add_argument('--use_rapf', action='store_true', default=False,
                        help='Enable reliability-aware probabilistic fusion.')
    parser.add_argument('--use_reliability_gate', action='store_true', default=False,
                        help='Enable RAPF gate supervision.')
    parser.add_argument('--rapf_hidden_dim', type=int, default=128)
    parser.add_argument('--rapf_initial_gate_bias', type=float, default=-2.0)
    parser.add_argument('--rapf_use_quality', action='store_true', default=False)
    parser.add_argument('--rapf_quality_weight', type=float, default=0.25)
    parser.add_argument('--rapf_struct_residual_clip', type=float, default=2.0)
    parser.add_argument('--rapf_generic_gate_cap', type=float, default=0.35)
    parser.add_argument('--rapf_gate_loss_weight', type=float, default=0.2)
    parser.add_argument('--rapf_gate_iou_margin', type=float, default=0.02)
    parser.add_argument('--rapf_quality_anchor_structured_residual',
                        action='store_true', default=False)
    parser.add_argument('--use_qahnl', action='store_true', default=False,
                        help='Enable query-aware hard negative learning.')
    parser.add_argument('--qahnl_score_source', type=str, default='fused',
                        choices=[
                            'base', 'structured', 'quality', 'fused',
                            'semantic_rerank', 'semantic_support',
                        ])
    parser.add_argument('--aux_scores_use_contrastive_base',
                        action='store_true', default=False,
                        help=(
                            'Use contrastive query-token similarity as the '
                            'base score for SACR/RAPF/QA-HNL auxiliary '
                            'training losses when available.'
                        ))
    parser.add_argument('--aux_scores_use_semantic_eval_base',
                        action='store_true', default=False,
                        help=(
                            'Use the semantic-alignment evaluation formula '
                            'as the base score for SACR/RAPF/QA-HNL auxiliary '
                            'training losses when available.'
                        ))
    parser.add_argument('--qahnl_num_hard_neg', type=int, default=16)
    parser.add_argument('--qahnl_pos_iou_thresh', type=float, default=0.25)
    parser.add_argument('--qahnl_neg_iou_thresh', type=float, default=0.10)
    parser.add_argument('--qahnl_topk_iou_pos', type=int, default=3)
    parser.add_argument('--qahnl_margin_base', type=float, default=0.2)
    parser.add_argument('--qahnl_margin_iou_lambda', type=float, default=0.5)
    parser.add_argument('--qahnl_margin_min', type=float, default=0.05)
    parser.add_argument('--qahnl_margin_max', type=float, default=0.5)
    parser.add_argument('--qahnl_temperature', type=float, default=1.0)
    parser.add_argument('--qahnl_temperature_max', type=float, default=6.0)
    parser.add_argument('--qahnl_loss_weight', type=float, default=0.2)
    parser.add_argument('--qahnl_use_entity_hardneg',
                        action='store_true', default=False)
    parser.add_argument('--qahnl_use_attr_hardneg',
                        action='store_true', default=False)
    parser.add_argument('--qahnl_use_relation_hardneg',
                        action='store_true', default=False)
    parser.add_argument('--use_sem_iou_rank', action='store_true', default=False,
                        help=(
                            'Enable training-only semantic-alignment IoU '
                            'ranking loss on projection scores.'
                        ))
    parser.add_argument('--sem_iou_rank_loss_weight', type=float, default=0.05)
    parser.add_argument('--sem_iou_rank_pos_iou_thresh',
                        type=float, default=0.50)
    parser.add_argument('--sem_iou_rank_neg_iou_thresh',
                        type=float, default=0.25)
    parser.add_argument('--sem_iou_rank_topk_iou_pos',
                        type=int, default=1)
    parser.add_argument('--sem_iou_rank_num_hard_neg',
                        type=int, default=16)
    parser.add_argument('--sem_iou_rank_margin', type=float, default=0.1)
    parser.add_argument('--sem_iou_rank_temperature',
                        type=float, default=1.0)
    parser.add_argument('--use_sem_iou_listwise', action='store_true',
                        default=False,
                        help=(
                            'Enable training-only listwise semantic-IoU '
                            'calibration over semantic alignment scores.'
                        ))
    parser.add_argument('--sem_iou_listwise_loss_weight',
                        type=float, default=0.05)
    parser.add_argument('--sem_iou_listwise_topk', type=int, default=32)
    parser.add_argument('--sem_iou_listwise_score_temperature',
                        type=float, default=0.25)
    parser.add_argument('--sem_iou_listwise_target_iou_power',
                        type=float, default=2.0)
    parser.add_argument('--sem_iou_listwise_high_iou_threshold',
                        type=float, default=0.50)
    parser.add_argument('--sem_iou_listwise_high_iou_weight',
                        type=float, default=2.0)
    parser.add_argument('--sem_iou_listwise_min_target_iou',
                        type=float, default=0.01)
    parser.add_argument('--use_sem_iou_top1', action='store_true',
                        default=False,
                        help=(
                            'Enable training-only semantic-IoU Top-1 swap '
                            'correction over semantic alignment scores.'
                        ))
    parser.add_argument('--sem_iou_top1_loss_weight',
                        type=float, default=0.02)
    parser.add_argument('--sem_iou_top1_pos_iou_thresh',
                        type=float, default=0.50)
    parser.add_argument('--sem_iou_top1_iou_gap',
                        type=float, default=0.05)
    parser.add_argument('--sem_iou_top1_margin',
                        type=float, default=0.1)
    parser.add_argument('--sem_iou_top1_temperature',
                        type=float, default=0.5)
    parser.add_argument('--use_sem_eval_margin', action='store_true',
                        default=False,
                        help=(
                            'Enable training-only matched-query margin loss '
                            'on the exact semantic-eval score composition.'
                        ))
    parser.add_argument('--sem_eval_margin_loss_weight',
                        type=float, default=0.02)
    parser.add_argument('--sem_eval_margin_min_pos_iou',
                        type=float, default=0.50)
    parser.add_argument('--sem_eval_margin_neg_iou_thresh',
                        type=float, default=0.45)
    parser.add_argument('--sem_eval_margin_num_hard_neg',
                        type=int, default=8)
    parser.add_argument('--sem_eval_margin_margin',
                        type=float, default=0.08)
    parser.add_argument('--sem_eval_margin_temperature',
                        type=float, default=0.5)
    parser.add_argument('--sem_align_use_eval_weights',
                        action='store_true', default=False,
                        help=(
                            'Use official eval-style full component weights '
                            'for training position/semantic alignment losses.'
                        ))
    parser.add_argument('--use_semantic_rerank_head',
                        action='store_true', default=False,
                        help=(
                            'Enable a learned residual head for last-layer '
                            'semantic-alignment reranking.'
                        ))
    parser.add_argument('--semantic_rerank_loss_weight',
                        type=float, default=0.02)
    parser.add_argument('--semantic_rerank_listwise_weight',
                        type=float, default=1.0)
    parser.add_argument('--semantic_rerank_threshold_mass_weight',
                        type=float, default=0.0,
                        help=(
                            'Training-only weight for probability-mass losses '
                            'at the evaluation IoU thresholds 0.25 and 0.50.'
                        ))
    parser.add_argument('--semantic_rerank_failure_margin_weight',
                        type=float, default=0.0,
                        help=(
                            'Training-only weight for a margin loss applied '
                            'only when the current top-1 query fails an '
                            'evaluation IoU threshold and a positive candidate '
                            'exists.'
                        ))
    parser.add_argument('--semantic_rerank_failure_margin',
                        type=float, default=0.1)
    parser.add_argument(
        '--semantic_rerank_train_use_support_scores',
        action='store_true', default=False,
        help=(
            'Train the rerank head through the final parameter-free semantic '
            'support scores. Requires --use_semantic_support_adapter.'
        ),
    )
    parser.add_argument('--semantic_rerank_topk', type=int, default=16)
    parser.add_argument('--semantic_rerank_residual_scale',
                        type=float, default=0.1)
    parser.add_argument('--semantic_rerank_temperature',
                        type=float, default=0.5)
    parser.add_argument('--semantic_rerank_hidden_dim',
                        type=int, default=128)
    parser.add_argument('--semantic_rerank_use_target_conditioning',
                        action='store_true', default=False)
    parser.add_argument(
        '--semantic_rerank_aux_checkpoint',
        type=str,
        default='',
        help=(
            'Evaluation-only checkpoint whose semantic rerank head is '
            'blended with the primary rerank head on the same queries.'
        ),
    )
    parser.add_argument(
        '--semantic_rerank_aux_weight',
        type=float,
        default=0.5,
        help='Auxiliary rerank residual weight in [0, 1].',
    )
    parser.add_argument(
        '--semantic_rerank_snapshot_steps',
        type=str,
        default='',
        help=(
            'Comma-separated optimizer steps at which to save lightweight '
            'semantic-rerank-head-only checkpoints during the current run.'
        ),
    )
    parser.add_argument('--use_semantic_threshold_head',
                        action='store_true', default=False)
    parser.add_argument('--semantic_threshold_hidden_dim',
                        type=int, default=64)
    parser.add_argument('--semantic_threshold_residual_scale',
                        type=float, default=0.25)
    parser.add_argument('--semantic_threshold_loss_weight',
                        type=float, default=1.0)
    parser.add_argument('--semantic_threshold_bce_weight',
                        type=float, default=1.0)
    parser.add_argument('--semantic_threshold_pairwise_weight',
                        type=float, default=0.0)
    parser.add_argument('--semantic_threshold_pairwise_margin',
                        type=float, default=0.25)
    parser.add_argument('--semantic_threshold_pairwise_temperature',
                        type=float, default=0.5)
    parser.add_argument('--semantic_threshold_pairwise_high_weight',
                        type=float, default=1.0)
    parser.add_argument('--semantic_threshold_focal_gamma',
                        type=float, default=2.0)
    parser.add_argument('--semantic_rerank_target_iou_power',
                        type=float, default=2.0)
    parser.add_argument('--semantic_rerank_min_target_iou',
                        type=float, default=0.01)
    parser.add_argument('--semantic_rerank_hard_sample_weight',
                        type=float, default=0.0)
    parser.add_argument('--semantic_rerank_multi_sample_weight',
                        type=float, default=0.0)
    parser.add_argument('--semantic_rerank_high_iou_threshold',
                        type=float, default=0.5)
    parser.add_argument('--semantic_rerank_high_iou_weight',
                        type=float, default=1.0)
    parser.add_argument('--semantic_rerank_top1_margin_weight',
                        type=float, default=0.0)
    parser.add_argument('--semantic_rerank_top1_margin',
                        type=float, default=0.1)
    parser.add_argument('--semantic_rerank_top1_neg_iou_threshold',
                        type=float, default=0.25)
    parser.add_argument('--use_semantic_component_calibration',
                        action='store_true', default=False,
                        help=(
                            'Enable tiny learned calibration over official '
                            'semantic-alignment component scores.'
                        ))
    parser.add_argument('--semantic_component_loss_weight',
                        type=float, default=0.02)
    parser.add_argument('--semantic_component_topk', type=int, default=16)
    parser.add_argument('--semantic_component_temperature',
                        type=float, default=1.0)
    parser.add_argument('--semantic_component_target_iou_power',
                        type=float, default=2.0)
    parser.add_argument('--semantic_component_min_target_iou',
                        type=float, default=0.01)
    parser.add_argument('--semantic_component_max_delta',
                        type=float, default=0.25)
    parser.add_argument('--semantic_component_use_eda_score',
                        action='store_true', default=False,
                        help=(
                            'Add a learned low-capacity residual from the '
                            'EDA position-alignment score to semantic '
                            'component calibration.'
                        ))
    parser.add_argument('--semantic_component_extra_max_weight',
                        type=float, default=0.25)
    parser.add_argument('--semantic_component_hard_sample_weight',
                        type=float, default=0.0)
    parser.add_argument('--semantic_component_multi_sample_weight',
                        type=float, default=0.0)
    parser.add_argument('--eval_use_structured_scores',
                        action='store_true', default=False)
    parser.add_argument('--eval_use_quality_scores',
                        action='store_true', default=False)
    parser.add_argument('--eval_use_fused_scores',
                        action='store_true', default=False)
    parser.add_argument(
        '--eval_use_fused_semantic_scores',
        action='store_true',
        default=False,
        help=(
            'Use RAPF fused query scores for last-layer semantic alignment. '
            'This is an EDA evaluator adapter; RAPF itself is unchanged.'
        ),
    )
    parser.add_argument('--eval_use_semantic_rerank_scores',
                        action='store_true', default=False)
    parser.add_argument('--use_semantic_support_adapter',
                        action='store_true', default=False)
    parser.add_argument('--semantic_support_overlap_weight',
                        type=float, default=0.6075)
    parser.add_argument('--semantic_support_position_weight',
                        type=float, default=0.1075)
    parser.add_argument('--semantic_support_overlap_power',
                        type=float, default=0.5)
    parser.add_argument('--semantic_support_use_learned_gate',
                        action='store_true', default=False)
    parser.add_argument('--semantic_support_gate_hidden_dim',
                        type=int, default=16)
    parser.add_argument('--semantic_support_gate_max',
                        type=float, default=2.0)
    parser.add_argument('--semantic_support_gate_use_query_features',
                        action='store_true', default=False)
    parser.add_argument('--semantic_support_gate_loss_weight',
                        type=float, default=0.0)
    parser.add_argument('--semantic_support_gate_loss_beta',
                        type=float, default=0.25)
    parser.add_argument('--semantic_support_gate_low_iou_threshold',
                        type=float, default=0.25)
    parser.add_argument('--semantic_support_gate_high_iou_threshold',
                        type=float, default=0.5)
    parser.add_argument('--eval_use_semantic_support_scores',
                        action='store_true', default=False)
    parser.add_argument('--eval_use_semantic_component_scores',
                        action='store_true', default=False)
    parser.add_argument('--eval_report_diagnostic_scores',
                        action='store_true', default=False)
    parser.add_argument(
        '--eval_diagnostic_dump_path',
        type=str,
        default='',
        help=(
            'Optional .npz path for per-query semantic diagnostic tensors. '
            'Intended for offline score calibration analysis only.'
        ),
    )
    parser.add_argument('--eval_report_box_blend_diagnostics',
                        action='store_true', default=False)
    parser.add_argument('--verbose_diagnostics',
                        action='store_true', default=False)

    # others
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')  # note
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5],
                        nargs='+', help='A list of AP IoU thresholds')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')
    parser.add_argument("--debug", action='store_true',
                        help="try to overfit few samples")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_train', action='store_true')
    parser.add_argument('--pp_checkpoint', default=None)    # pointnet checkpoint
    parser.add_argument('--reduce_lr', action='store_true')
    parser.add_argument(
        '--freeze_base_train_heads',
        action='store_true',
        default=False,
        help=(
            'Freeze the pretrained base model and train only structured slot, '
            'quality, SACR, and RAPF head parameters.'
        ),
    )
    parser.add_argument(
        '--freeze_base_train_align_heads',
        action='store_true',
        default=False,
        help=(
            'Freeze the pretrained base model and train innovation heads plus '
            'contrastive alignment projection heads.'
        ),
    )
    parser.add_argument(
        '--freeze_base_train_box_align_heads',
        action='store_true',
        default=False,
        help=(
            'Freeze the pretrained base model and train box prediction heads, '
            'innovation heads, and contrastive alignment projection heads.'
        ),
    )
    parser.add_argument(
        '--freeze_base_train_component_head',
        action='store_true',
        default=False,
        help=(
            'Freeze the pretrained base model and train only the semantic '
            'component calibration head.'
        ),
    )
    parser.add_argument(
        '--freeze_base_train_rerank_head',
        action='store_true',
        default=False,
        help=(
            'Freeze the pretrained base model and train only the semantic '
            'rerank head.'
        ),
    )
    parser.add_argument(
        '--freeze_base_train_rerank_align_heads',
        action='store_true',
        default=False,
        help=(
            'Freeze the pretrained base and train only the semantic rerank '
            'head plus contrastive alignment projection heads.'
        ),
    )
    parser.add_argument(
        '--freeze_base_train_threshold_head',
        action='store_true',
        default=False,
        help=(
            'Freeze all existing parameters and train only the dual-IoU '
            'threshold query head inside the semantic rerank adapter.'
        ),
    )
    parser.add_argument(
        '--freeze_base_train_rerank_conditioner',
        action='store_true',
        default=False,
        help=(
            'Freeze all existing parameters and train only the target-conditioned '
            'semantic rerank branch.'
        ),
    )
    parser.add_argument(
        '--freeze_base_train_support_gate',
        action='store_true',
        default=False,
        help=(
            'Freeze all existing parameters and train only the query-level '
            'RAPF semantic support gate.'
        ),
    )

    args, _ = parser.parse_known_args()

    args.eval = args.eval or args.eval_train

    if args.semantic_rerank_aux_checkpoint:
        if not args.eval:
            parser.error('--semantic_rerank_aux_checkpoint is evaluation-only')
        if not args.use_semantic_rerank_head:
            parser.error(
                '--semantic_rerank_aux_checkpoint requires '
                '--use_semantic_rerank_head'
            )
        if not 0.0 <= args.semantic_rerank_aux_weight <= 1.0:
            parser.error('--semantic_rerank_aux_weight must be in [0, 1]')

    try:
        snapshot_steps = _parse_positive_int_set(
            args.semantic_rerank_snapshot_steps
        )
    except ValueError as exc:
        parser.error(str(exc))
    if snapshot_steps:
        if args.eval:
            parser.error('--semantic_rerank_snapshot_steps is training-only')
        if not args.use_semantic_rerank_head:
            parser.error(
                '--semantic_rerank_snapshot_steps requires '
                '--use_semantic_rerank_head'
            )
        if not args.freeze_base_train_rerank_head:
            parser.error(
                '--semantic_rerank_snapshot_steps requires '
                '--freeze_base_train_rerank_head'
            )

    return args


def _parse_positive_int_set(value):
    """Parse a comma-separated set of strictly positive integer steps."""
    text = str(value or '').strip()
    if not text:
        return set()
    try:
        steps = {int(item.strip()) for item in text.split(',')}
    except ValueError:
        raise ValueError(
            '--semantic_rerank_snapshot_steps must contain only integers'
        )
    if any(step <= 0 for step in steps):
        raise ValueError(
            '--semantic_rerank_snapshot_steps must contain positive integers'
        )
    return steps

HEAD_ONLY_TRAINABLE_PREFIXES = (
    'structured_slot_builder.',
    'quality_head.',
    'sacr_head.',
    'reliability_fusion.',
    'semantic_rerank_head.',
    'semantic_component_calibrator.',
)

COMPONENT_HEAD_TRAINABLE_PREFIXES = (
    'semantic_component_calibrator.',
)

RERANK_HEAD_TRAINABLE_PREFIXES = (
    'semantic_rerank_head.',
)

RERANK_CONDITIONER_TRAINABLE_PREFIXES = (
    'semantic_rerank_head.conditioning_mlp.',
)

THRESHOLD_HEAD_TRAINABLE_PREFIXES = (
    'semantic_rerank_head.threshold_head.',
)

SUPPORT_GATE_TRAINABLE_PREFIXES = (
    'semantic_support_adapter.support_gate.',
)

ALIGN_HEAD_TRAINABLE_PREFIXES = (
    'contrastive_align_projection_image.',
    'contrastive_align_projection_text.',
)

BBOX_HEAD_TRAINABLE_PREFIXES = (
    'proposal_head.',
    'prediction_heads.',
)


def _is_head_only_trainable_name(name):
    return any(
        name == prefix[:-1] or name.startswith(prefix)
        for prefix in HEAD_ONLY_TRAINABLE_PREFIXES
    )


def _is_component_head_trainable_name(name):
    return any(
        name == prefix[:-1] or name.startswith(prefix)
        for prefix in COMPONENT_HEAD_TRAINABLE_PREFIXES
    )


def _is_rerank_head_trainable_name(name):
    return any(
        name == prefix[:-1] or name.startswith(prefix)
        for prefix in RERANK_HEAD_TRAINABLE_PREFIXES
    )


def _is_rerank_conditioner_trainable_name(name):
    return any(
        name == prefix[:-1] or name.startswith(prefix)
        for prefix in RERANK_CONDITIONER_TRAINABLE_PREFIXES
    )


def _is_support_gate_trainable_name(name):
    return any(
        name == prefix[:-1] or name.startswith(prefix)
        for prefix in SUPPORT_GATE_TRAINABLE_PREFIXES
    )


def _is_threshold_head_trainable_name(name):
    return any(
        name == prefix[:-1] or name.startswith(prefix)
        for prefix in THRESHOLD_HEAD_TRAINABLE_PREFIXES
    )


def _is_align_head_trainable_name(name):
    return any(
        name == prefix[:-1] or name.startswith(prefix)
        for prefix in ALIGN_HEAD_TRAINABLE_PREFIXES
    )


def _is_bbox_head_trainable_name(name):
    return any(
        name == prefix[:-1] or name.startswith(prefix)
        for prefix in BBOX_HEAD_TRAINABLE_PREFIXES
    )


def _uses_frozen_base_training(args):
    return (
        bool(getattr(args, 'freeze_base_train_component_head', False))
        or bool(getattr(args, 'freeze_base_train_rerank_head', False))
        or bool(getattr(args, 'freeze_base_train_rerank_align_heads', False))
        or bool(getattr(args, 'freeze_base_train_rerank_conditioner', False))
        or bool(getattr(args, 'freeze_base_train_threshold_head', False))
        or bool(getattr(args, 'freeze_base_train_support_gate', False))
        or
        bool(getattr(args, 'freeze_base_train_heads', False))
        or bool(getattr(args, 'freeze_base_train_align_heads', False))
        or bool(getattr(args, 'freeze_base_train_box_align_heads', False))
    )


def _is_frozen_base_trainable_name(args, name):
    if bool(getattr(args, 'freeze_base_train_support_gate', False)):
        return _is_support_gate_trainable_name(name)
    if bool(getattr(args, 'freeze_base_train_threshold_head', False)):
        return _is_threshold_head_trainable_name(name)
    if bool(getattr(args, 'freeze_base_train_rerank_conditioner', False)):
        return _is_rerank_conditioner_trainable_name(name)
    if bool(getattr(args, 'freeze_base_train_component_head', False)):
        return _is_component_head_trainable_name(name)
    if bool(getattr(args, 'freeze_base_train_rerank_align_heads', False)):
        return (
            _is_rerank_head_trainable_name(name)
            or _is_align_head_trainable_name(name)
        )
    if bool(getattr(args, 'freeze_base_train_rerank_head', False)):
        return _is_rerank_head_trainable_name(name)
    if _is_head_only_trainable_name(name):
        return True
    return (
        (
            bool(getattr(args, 'freeze_base_train_align_heads', False))
            or bool(getattr(args, 'freeze_base_train_box_align_heads', False))
        )
        and _is_align_head_trainable_name(name)
    ) or (
        bool(getattr(args, 'freeze_base_train_box_align_heads', False))
        and _is_bbox_head_trainable_name(name)
    )


def _freeze_base_trainable_heads(args, model):
    """Freeze the pretrained base, leaving only innovation heads trainable."""
    if not _uses_frozen_base_training(args):
        return 0
    trainable_count = 0
    for name, param in model.named_parameters():
        keep_trainable = _is_frozen_base_trainable_name(args, name)
        param.requires_grad = keep_trainable
        if keep_trainable:
            trainable_count += param.numel()
    if trainable_count <= 0:
        raise ValueError(
            "frozen-base training found no trainable head "
            "parameters"
        )
    return trainable_count


def _set_frozen_base_modules_eval(model, args=None):
    """Keep frozen base modules deterministic while training head modules."""
    base_model = model.module if hasattr(model, 'module') else model
    for name, module in base_model.named_modules():
        if name == '':
            continue
        if args is None:
            keep_trainable = _is_head_only_trainable_name(name)
        else:
            keep_trainable = _is_frozen_base_trainable_name(args, name)
        if not keep_trainable:
            module.eval()


# BRIEF load checkpoint.
def load_checkpoint(args, model, optimizer, scheduler):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    partial_init = getattr(args, 'train_partial_checkpoint_init', False)
    try:
        checkpoint_epoch = int(checkpoint['epoch'])
    except Exception:
        checkpoint_epoch = -1
    if not partial_init:
        args.start_epoch = checkpoint_epoch + 1
    if partial_init:
        model_state = model.state_dict()
        checkpoint_state = checkpoint['model']
        compatible_state = {}
        skipped_shape = []
        for name, tensor in checkpoint_state.items():
            if name not in model_state:
                continue
            if tuple(model_state[name].shape) != tuple(tensor.shape):
                skipped_shape.append(name)
                continue
            compatible_state[name] = tensor
        model_state.update(compatible_state)
        model.load_state_dict(model_state, strict=True)
        print(
            "=> partially initialized {} / {} model tensors".format(
                len(compatible_state), len(model_state)
            )
        )
        if skipped_shape:
            print(
                "=> skipped {} shape-mismatched checkpoint tensors".format(
                    len(skipped_shape)
                )
            )
    else:
        checkpoint_state = checkpoint['model']
        if getattr(args, 'semantic_rerank_aux_checkpoint', ''):
            checkpoint_state = dict(checkpoint_state)
            model_state = model.state_dict()
            auxiliary_keys = [
                name for name in model_state
                if 'semantic_rerank_aux_head.' in name
            ]
            if not auxiliary_keys:
                raise ValueError(
                    'auxiliary rerank checkpoint requested but the model '
                    'has no auxiliary rerank head'
                )
            for name in auxiliary_keys:
                checkpoint_state[name] = model_state[name]
        model.load_state_dict(checkpoint_state, strict=True)
    if (
        not args.eval
        and not args.reduce_lr
        and not _uses_frozen_base_training(args)
        and not partial_init
    ):
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])

    print("=> loaded successfully '{}' (epoch {})".format(
        args.checkpoint_path, checkpoint_epoch
    ))

    del checkpoint
    torch.cuda.empty_cache()


def load_semantic_rerank_aux_checkpoint(args, model):
    """Load only a rerank head into the evaluation-only auxiliary branch."""
    checkpoint_path = str(
        getattr(args, 'semantic_rerank_aux_checkpoint', '') or ''
    )
    if not checkpoint_path:
        return 0
    if not getattr(args, 'eval', False):
        raise ValueError('auxiliary rerank checkpoint is evaluation-only')
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    base_model = model.module if hasattr(model, 'module') else model
    auxiliary_head = getattr(base_model, 'semantic_rerank_aux_head', None)
    if auxiliary_head is None:
        raise ValueError('model has no auxiliary semantic rerank head')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_state = checkpoint['model']
    source_prefixes = (
        'module.semantic_rerank_head.',
        'semantic_rerank_head.',
    )
    auxiliary_state = {}
    for name, tensor in checkpoint_state.items():
        for prefix in source_prefixes:
            if name.startswith(prefix):
                auxiliary_state[name[len(prefix):]] = tensor
                break
    expected = auxiliary_head.state_dict()
    if set(auxiliary_state) != set(expected):
        missing = sorted(set(expected) - set(auxiliary_state))
        unexpected = sorted(set(auxiliary_state) - set(expected))
        raise ValueError(
            'auxiliary rerank state mismatch: missing={} unexpected={}'
            .format(missing, unexpected)
        )
    auxiliary_head.load_state_dict(auxiliary_state, strict=True)
    del checkpoint
    torch.cuda.empty_cache()
    print(
        "=> loaded auxiliary semantic rerank head '{}' ({} tensors)".format(
            checkpoint_path, len(auxiliary_state)
        )
    )
    return len(auxiliary_state)


# BRIEF save model.
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
        torch.save(state, spath)
        print("Saved in {}".format(spath))
    else:
        print("not saving checkpoint")


def save_semantic_rerank_head_snapshot(args, model, step):
    """Save only the primary rerank head for cheap early-step evaluation."""
    base_model = model.module if hasattr(model, 'module') else model
    rerank_head = getattr(base_model, 'semantic_rerank_head', None)
    if rerank_head is None:
        raise ValueError('model has no semantic rerank head')
    state = {
        'model': {
            'module.semantic_rerank_head.' + name: tensor.detach().cpu()
            for name, tensor in rerank_head.state_dict().items()
        },
        'epoch': 'step_{}'.format(int(step)),
        'step': int(step),
    }
    path = os.path.join(
        args.log_dir, 'semantic_rerank_head_step_{}.pth'.format(int(step))
    )
    torch.save(state, path)
    print("Saved semantic rerank head in {}".format(path))
    return path


class BaseTrainTester:
    """Basic train/test class to be inherited."""

    # logger.
    def __init__(self, args):
        """Initialize."""
        name = args.log_dir.split('/')[-1]  # log_dir: './logs/eda', name: eda
        
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

        # tensorboard
        self.tensorboard = record_tensorboard.TensorBoard(args.log_dir, distributed_rank=dist.get_rank())

        # Save config file and initialize tb writer
        if dist.get_rank() == 0:
            path = os.path.join(args.log_dir, "config.json")
            with open(path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            self.logger.info("Full config saved to {}".format(path))
            self.logger.info(str(vars(args)))

    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""
        train_dataset = None
        test_dataset = None
        return train_dataset, test_dataset


    # BRIEF dataloader.
    def get_loaders(self, args):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        def _joint_det_collate(batch):
            preserve_keys = {
                'target_slot', 'entity_spans', 'attr_spans', 'rel_spans',
                'coverage_stats', 'description', 'scene_id', 'object_id',
                'object_name', 'ann_id', 'attr_slot', 'rel_slots',
                'anchor_slots', 'slot_mask',
            }
            collated = {}
            for key in batch[0].keys():
                values = [sample[key] for sample in batch]
                if key in preserve_keys:
                    collated[key] = values
                else:
                    collated[key] = default_collate(values)
            return collated

        # Datasets
        train_dataset, test_dataset = self.get_datasets(args)
        
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)

        if args.eval:
            train_loader = None
        else:
            train_sampler = DistributedSampler(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,      # TODO 
                num_workers=args.num_workers,
                worker_init_fn=seed_worker,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True,
                generator=g,
                collate_fn=_joint_det_collate
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
            collate_fn=_joint_det_collate
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
            losses=losses, eos_coef=0.1, temperature=0.07,
            sem_align_use_eval_weights=getattr(
                args, 'sem_align_use_eval_weights', False
            ),
        )
        criterion = compute_hungarian_loss

        return criterion, set_criterion

    @staticmethod
    def get_optimizer(args, model):
        """Initialize optimizer."""
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

    @staticmethod
    def _ddp_find_unused_parameters(args):
        return bool(getattr(args, 'use_structured_slots', False))


    # BRIEF main training/testing
    def main(self, args):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, test_loader = self.get_loaders(args)
        if not args.eval:
            n_data = len(train_loader.dataset)
            self.logger.info(f"length of training dataset: {n_data}")
        n_data = len(test_loader.dataset)
        self.logger.info(f"length of testing dataset: {n_data}")

        # Get model
        model = self.get_model(args)
        head_param_count = _freeze_base_trainable_heads(args, model)
        if head_param_count:
            freeze_mode = (
                'freeze_base_train_support_gate'
                if getattr(args, 'freeze_base_train_support_gate', False)
                else (
                    'freeze_base_train_threshold_head'
                    if getattr(
                        args, 'freeze_base_train_threshold_head', False
                    ) else (
                    'freeze_base_train_rerank_conditioner'
                    if getattr(args, 'freeze_base_train_rerank_conditioner', False)
                    else (
                        'freeze_base_train_component_head'
                        if getattr(args, 'freeze_base_train_component_head', False)
                        else (
                            'freeze_base_train_rerank_align_heads'
                            if getattr(args, 'freeze_base_train_rerank_align_heads', False)
                            else (
                            'freeze_base_train_rerank_head'
                            if getattr(args, 'freeze_base_train_rerank_head', False)
                            else (
                                'freeze_base_train_box_align_heads'
                                if getattr(args, 'freeze_base_train_box_align_heads', False)
                                else (
                                    'freeze_base_train_align_heads'
                                    if getattr(args, 'freeze_base_train_align_heads', False)
                                    else 'freeze_base_train_heads'
                                )
                            )
                            )
                        )
                    )
                    )
                )
            )
            self.logger.info(
                "{}: trainable head parameters {}".format(
                    freeze_mode, head_param_count
                )
            )

        # Get criterion
        criterion, set_criterion = self.get_criterion(args)

        # Get optimizer
        optimizer = self.get_optimizer(args, model)

        # Get scheduler
        if not args.eval:
            scheduler = get_scheduler(optimizer, len(train_loader), args)
        else:
            scheduler = None
        
        # Move model to devices
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                # synBN
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
            else:
                model = model.cuda()

        # note Distributed Data-Parallel Training (DDP)
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=self._ddp_find_unused_parameters(args)
        )

        # Check for a checkpoint
        if args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)
        load_semantic_rerank_aux_checkpoint(args, model)
        
        # ##############################################
        # NOTE [eval-only] Just eval and end execution #
        # ##############################################
        if args.eval:
            print("Testing evaluation.....................")
            self.evaluate_one_epoch(
                args.start_epoch, test_loader,
                model, criterion, set_criterion, args
            )
            return

        # ##############################
        # NOTE Training and Validation #
        # ##############################
        for epoch in range(args.start_epoch, args.max_epoch + 1):
            train_loader.sampler.set_epoch(epoch)
            tic = time.time()

            # train *
            self.train_one_epoch(
                epoch, train_loader, model,
                criterion, set_criterion,
                optimizer, scheduler, args
            )
            
            # log
            self.logger.info(
                'epoch {}, total time {:.2f}, '
                'lr_base {:.5f}, lr_pointnet {:.5f}'.format(
                    epoch, (time.time() - tic),
                    optimizer.param_groups[0]['lr'],
                    optimizer.param_groups[1]['lr']
                )
            )

            # save model and validate
            if epoch % args.val_freq == 0:
                if dist.get_rank() == 0:
                    save_checkpoint(args, epoch, model, optimizer, scheduler)
                
                # validate *
                print("Test evaluation.......")
                self.evaluate_one_epoch(
                    epoch, test_loader,
                    model, criterion, set_criterion, args
                )

        # Training is over
        save_checkpoint(args, 'last', model, optimizer, scheduler, True)
        saved_path = os.path.join(args.log_dir, 'ckpt_epoch_last.pth')
        self.logger.info("Saved in {}".format(saved_path))
        self.evaluate_one_epoch(
            args.max_epoch, test_loader,
            model, criterion, set_criterion, args
        )
        return saved_path

    @staticmethod
    def _to_gpu(data_dict):
        if torch.cuda.is_available():
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
        return data_dict

    @staticmethod
    def _merge_batch_data(end_points, batch_data):
        for key in batch_data:
            if key in end_points:
                continue
            end_points[key] = batch_data[key]
        return end_points

    @staticmethod
    def _get_inputs(batch_data):
        return {
            'point_clouds': batch_data['point_clouds'].float(),
            'text': batch_data['utterances']
        }

    @staticmethod
    def _compute_loss(end_points, criterion, set_criterion, args):
        loss, end_points = criterion(
            end_points, args.num_decoder_layers,
            set_criterion,
            query_points_obj_topk=args.query_points_obj_topk,
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
                'use_entity_hardneg': getattr(
                    args, 'qahnl_use_entity_hardneg', False
                ),
                'use_attr_hardneg': getattr(
                    args, 'qahnl_use_attr_hardneg', False
                ),
                'use_relation_hardneg': getattr(
                    args, 'qahnl_use_relation_hardneg', False
                ),
            } if getattr(args, 'use_qahnl', False) else None,
            use_sem_iou_rank=getattr(args, 'use_sem_iou_rank', False),
            sem_iou_rank_config={
                'loss_weight': getattr(args, 'sem_iou_rank_loss_weight', 0.05),
                'pos_iou_thresh': getattr(
                    args, 'sem_iou_rank_pos_iou_thresh', 0.50
                ),
                'neg_iou_thresh': getattr(
                    args, 'sem_iou_rank_neg_iou_thresh', 0.25
                ),
                'topk_iou_pos': getattr(
                    args, 'sem_iou_rank_topk_iou_pos', 1
                ),
                'num_hard_neg': getattr(
                    args, 'sem_iou_rank_num_hard_neg', 16
                ),
                'margin': getattr(args, 'sem_iou_rank_margin', 0.1),
                'temperature': getattr(args, 'sem_iou_rank_temperature', 1.0),
            } if getattr(args, 'use_sem_iou_rank', False) else None,
            use_sem_iou_listwise=getattr(args, 'use_sem_iou_listwise', False),
            sem_iou_listwise_config={
                'loss_weight': getattr(
                    args, 'sem_iou_listwise_loss_weight', 0.05
                ),
                'topk': getattr(args, 'sem_iou_listwise_topk', 32),
                'score_temperature': getattr(
                    args, 'sem_iou_listwise_score_temperature', 0.25
                ),
                'target_iou_power': getattr(
                    args, 'sem_iou_listwise_target_iou_power', 2.0
                ),
                'high_iou_threshold': getattr(
                    args, 'sem_iou_listwise_high_iou_threshold', 0.50
                ),
                'high_iou_weight': getattr(
                    args, 'sem_iou_listwise_high_iou_weight', 2.0
                ),
                'min_target_iou': getattr(
                    args, 'sem_iou_listwise_min_target_iou', 0.01
                ),
            } if getattr(args, 'use_sem_iou_listwise', False) else None,
            use_sem_iou_top1=getattr(args, 'use_sem_iou_top1', False),
            sem_iou_top1_config={
                'loss_weight': getattr(
                    args, 'sem_iou_top1_loss_weight', 0.02
                ),
                'pos_iou_thresh': getattr(
                    args, 'sem_iou_top1_pos_iou_thresh', 0.50
                ),
                'iou_gap': getattr(args, 'sem_iou_top1_iou_gap', 0.05),
                'margin': getattr(args, 'sem_iou_top1_margin', 0.1),
                'temperature': getattr(
                    args, 'sem_iou_top1_temperature', 0.5
                ),
            } if getattr(args, 'use_sem_iou_top1', False) else None,
            use_sem_eval_margin=getattr(args, 'use_sem_eval_margin', False),
            sem_eval_margin_config={
                'loss_weight': getattr(
                    args, 'sem_eval_margin_loss_weight', 0.02
                ),
                'min_pos_iou': getattr(
                    args, 'sem_eval_margin_min_pos_iou', 0.50
                ),
                'neg_iou_thresh': getattr(
                    args, 'sem_eval_margin_neg_iou_thresh', 0.45
                ),
                'num_hard_neg': getattr(
                    args, 'sem_eval_margin_num_hard_neg', 8
                ),
                'margin': getattr(args, 'sem_eval_margin_margin', 0.08),
                'temperature': getattr(
                    args, 'sem_eval_margin_temperature', 0.5
                ),
            } if getattr(args, 'use_sem_eval_margin', False) else None,
            use_semantic_rerank_head=getattr(
                args, 'use_semantic_rerank_head', False
            ),
            semantic_rerank_config={
                'loss_weight': getattr(
                    args, 'semantic_rerank_loss_weight', 0.02
                ),
                'listwise_weight': getattr(
                    args, 'semantic_rerank_listwise_weight', 1.0
                ),
                'threshold_mass_weight': getattr(
                    args, 'semantic_rerank_threshold_mass_weight', 0.0
                ),
                'failure_margin_weight': getattr(
                    args, 'semantic_rerank_failure_margin_weight', 0.0
                ),
                'failure_margin': getattr(
                    args, 'semantic_rerank_failure_margin', 0.1
                ),
                'train_use_support_scores': getattr(
                    args, 'semantic_rerank_train_use_support_scores', False
                ),
                'topk': getattr(args, 'semantic_rerank_topk', 16),
                'temperature': getattr(
                    args, 'semantic_rerank_temperature', 0.5
                ),
                'target_iou_power': getattr(
                    args, 'semantic_rerank_target_iou_power', 2.0
                ),
                'min_target_iou': getattr(
                    args, 'semantic_rerank_min_target_iou', 0.01
                ),
                'hard_sample_weight': getattr(
                    args, 'semantic_rerank_hard_sample_weight', 0.0
                ),
                'multi_sample_weight': getattr(
                    args, 'semantic_rerank_multi_sample_weight', 0.0
                ),
                'high_iou_threshold': getattr(
                    args, 'semantic_rerank_high_iou_threshold', 0.5
                ),
                'high_iou_weight': getattr(
                    args, 'semantic_rerank_high_iou_weight', 1.0
                ),
                'top1_margin_weight': getattr(
                    args, 'semantic_rerank_top1_margin_weight', 0.0
                ),
                'top1_margin': getattr(
                    args, 'semantic_rerank_top1_margin', 0.1
                ),
                'top1_neg_iou_threshold': getattr(
                    args, 'semantic_rerank_top1_neg_iou_threshold', 0.25
                ),
            } if getattr(args, 'use_semantic_rerank_head', False) else None,
            semantic_threshold_config={
                'loss_weight': getattr(
                    args, 'semantic_threshold_loss_weight', 1.0
                ),
                'bce_weight': getattr(
                    args, 'semantic_threshold_bce_weight', 1.0
                ),
                'pairwise_weight': getattr(
                    args, 'semantic_threshold_pairwise_weight', 0.0
                ),
                'pairwise_margin': getattr(
                    args, 'semantic_threshold_pairwise_margin', 0.25
                ),
                'pairwise_temperature': getattr(
                    args, 'semantic_threshold_pairwise_temperature', 0.5
                ),
                'pairwise_high_weight': getattr(
                    args, 'semantic_threshold_pairwise_high_weight', 1.0
                ),
                'focal_gamma': getattr(
                    args, 'semantic_threshold_focal_gamma', 2.0
                ),
            } if getattr(
                args, 'use_semantic_threshold_head', False
            ) else None,
            semantic_support_gate_config={
                'loss_weight': getattr(
                    args, 'semantic_support_gate_loss_weight', 0.0
                ),
                'beta': getattr(
                    args, 'semantic_support_gate_loss_beta', 0.25
                ),
                'low_iou_threshold': getattr(
                    args, 'semantic_support_gate_low_iou_threshold', 0.25
                ),
                'high_iou_threshold': getattr(
                    args, 'semantic_support_gate_high_iou_threshold', 0.5
                ),
            } if getattr(
                args, 'semantic_support_gate_loss_weight', 0.0
            ) > 0 else None,
            use_semantic_component_calibration=getattr(
                args, 'use_semantic_component_calibration', False
            ),
            semantic_component_config={
                'loss_weight': getattr(
                    args, 'semantic_component_loss_weight', 0.02
                ),
                'topk': getattr(args, 'semantic_component_topk', 16),
                'temperature': getattr(
                    args, 'semantic_component_temperature', 1.0
                ),
                'target_iou_power': getattr(
                    args, 'semantic_component_target_iou_power', 2.0
                ),
                'min_target_iou': getattr(
                    args, 'semantic_component_min_target_iou', 0.01
                ),
                'hard_sample_weight': getattr(
                    args, 'semantic_component_hard_sample_weight', 0.0
                ),
                'multi_sample_weight': getattr(
                    args, 'semantic_component_multi_sample_weight', 0.0
                ),
            } if getattr(
                args, 'use_semantic_component_calibration', False
            ) else None,
        )
        return loss, end_points

    @staticmethod
    def _accumulate_stats(stat_dict, end_points):
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                if isinstance(end_points[key], (float, int)):
                    stat_dict[key] += end_points[key]
                elif isinstance(end_points[key], torch.Tensor):
                    value = end_points[key].detach()
                    if value.numel() == 1:
                        stat_dict[key] += value.item()
                    else:
                        stat_dict[key] += value.float().mean().item()
                else:
                    continue
        return stat_dict


    # BRIEF Training
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
        if _uses_frozen_base_training(args):
            _set_frozen_base_modules_eval(model, args)
        snapshot_steps = _parse_positive_int_set(
            getattr(args, 'semantic_rerank_snapshot_steps', '')
        )

        # Loop over batches
        train_loader = tqdm(train_loader)
        for batch_idx, batch_data in enumerate(train_loader):
            # Move to GPU
            batch_data = self._to_gpu(batch_data)
            # get the input data: pointcloud and text
            inputs = self._get_inputs(batch_data)

            # note Forward pass
            end_points = model(inputs)

            # note Compute loss and gradients, update parameters.
            end_points = self._merge_batch_data(end_points, batch_data)
            loss, end_points = self._compute_loss(
                end_points, criterion, set_criterion, args
            )

            optimizer.zero_grad()
            loss.backward()

            if args.clip_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_norm
                )
                stat_dict['grad_norm'] = grad_total_norm
            
            optimizer.step()
            scheduler.step()

            run_step = (
                (epoch - args.start_epoch) * len(train_loader)
                + batch_idx + 1
            )
            if run_step in snapshot_steps and dist.get_rank() == 0:
                save_semantic_rerank_head_snapshot(args, model, run_step)

            # Accumulate statistics and print out
            stat_dict = self._accumulate_stats(stat_dict, end_points)

            # print loss
            if (batch_idx + 1) % args.print_freq == 0:
                # Terminal logs
                self.logger.info(
                    f'Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  '  # Train: [30][2000/2432]
                )
                self.logger.info(''.join([
                    f'{key} {stat_dict[key] / (batch_idx + 1):.4f} \t'
                    for key in sorted(stat_dict.keys())
                    if 'loss' in key and 'proposal_' not in key
                    and 'last_' not in key and 'head_' not in key
                ])) # loss，loss_bbox，loss_ce，loss_sem_align，loss_giou，query_points_generation_loss

                # # reset stat_dict
                # for key in sorted(stat_dict.keys()):
                #     stat_dict[key] = 0
                
                if dist.get_rank() == 0:
                    for key in self.tensorboard.item["train_loss"]:
                        self.tensorboard.item["train_loss"][key] = stat_dict[key] / (batch_idx + 1)
                    self.tensorboard.dump_tensorboard("train_loss", (epoch-1)*len(train_loader)+batch_idx+1)

        # tensorboard
        if dist.get_rank() == 0:
            # loss
            for key in self.tensorboard.item["train_loss"]:
                self.tensorboard.item["train_loss"][key] = stat_dict[key] / len(train_loader)
            self.tensorboard.dump_tensorboard("train_loss", (epoch-1)*len(train_loader)+batch_idx+1)
            # lr
            self.tensorboard.item["train_lr"]["lr_base"] = optimizer.param_groups[0]['lr']
            self.tensorboard.item["train_lr"]["lr_pointnet"] = optimizer.param_groups[1]['lr']
            self.tensorboard.dump_tensorboard("train_lr", epoch)

    # BRIEF eval 
    @torch.no_grad()
    def _main_eval_branch(self, batch_idx, batch_data, test_loader, model,
                          stat_dict,
                          criterion, set_criterion, args):
        # Move to GPU
        batch_data = self._to_gpu(batch_data)
        inputs = self._get_inputs(batch_data)
        if "train" not in inputs:
            inputs.update({"train": False})
        else:
            inputs["train"] = False

        # STEP Forward pass
        end_points = model(inputs)

        # STEP Compute loss
        end_points = self._merge_batch_data(end_points, batch_data)
        _, end_points = self._compute_loss(
            end_points, criterion, set_criterion, args
        )
        for key in (
            'eval_use_structured_scores',
            'eval_use_quality_scores',
            'eval_use_fused_scores',
            'eval_use_fused_semantic_scores',
            'eval_use_semantic_rerank_scores',
            'eval_use_semantic_support_scores',
            'eval_use_semantic_component_scores',
            'eval_report_diagnostic_scores',
            'eval_report_box_blend_diagnostics',
        ):
            end_points[key] = bool(getattr(args, key, False))
        end_points['eval_diagnostic_dump_path'] = str(
            getattr(args, 'eval_diagnostic_dump_path', '') or ''
        )
        for key in end_points:
            if 'pred_size' in key:
                end_points[key] = torch.clamp(end_points[key], min=1e-6)

        # Accumulate statistics and print out
        stat_dict = self._accumulate_stats(stat_dict, end_points)
        if (batch_idx + 1) % args.print_freq == 0:
            self.logger.info(f'Eval: [{batch_idx + 1}/{len(test_loader)}]  ')
            self.logger.info(''.join([
                f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                for key in sorted(stat_dict.keys())
                if 'loss' in key and 'proposal_' not in key
                and 'last_' not in key and 'head_' not in key
            ]))
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
