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
"""Main script for language modulation."""

import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist

from main_utils import parse_option, BaseTrainTester
from data.model_util_scannet import ScannetDatasetConfig
from src.joint_det_dataset import Joint3DDataset
from src.grounding_evaluator import GroundingEvaluator, GroundingGTEvaluator
from models import BeaUTyDETR
from models import APCalculator, parse_predictions, parse_groundtruths
from models.detector_policy_sources import DETECTOR_POLICY_SOURCE_NAMES


import ipdb
st = ipdb.set_trace


def _eval_result_json_value(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_eval_results_json(path, eval_results):
    if not path or eval_results is None:
        return
    dump_dir = os.path.dirname(path)
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
    payload = {
        str(key): _eval_result_json_value(value)
        for key, value in eval_results.items()
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def configure_reproducibility(seed):
    """Configure best-effort deterministic training for a process."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TrainTester(BaseTrainTester):
    """Train/test a language grounder."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

    @staticmethod
    def get_datasets(args, include_train=True):
        """Initialize datasets."""
        dataset_dict = {}  # dict to use multiple datasets
        for dset in args.dataset:
            dataset_dict[dset] = 1
        if args.joint_det:
            dataset_dict['scannet'] = 10
        print('Loading datasets:', sorted(list(dataset_dict.keys())))
        train_dataset = None
        if include_train:
            train_dataset = Joint3DDataset(
                dataset_dict=dataset_dict,
                test_dataset=args.test_dataset,
                split='train' if not args.debug else 'val',
                use_color=args.use_color, use_height=args.use_height,
                overfit=args.debug,
                data_path=args.data_root,
                detect_intermediate=args.detect_intermediate,
                use_multiview=args.use_multiview,
                butd=args.butd,
                butd_gt=args.butd_gt,
                butd_cls=args.butd_cls,
                augment_det=args.augment_det,
                disable_box_jitter=args.disable_box_jitter,
                disable_train_augmentation=getattr(
                    args, 'disable_train_augmentation', False
                ),
                spacy_relation_free_yaw_only_aug=getattr(
                    args, 'spacy_relation_free_yaw_only_aug', False
                ),
                spacy_relation_free_view_guard_aug=getattr(
                    args, 'spacy_relation_free_view_guard_aug', False
                ),
                spacy_relation_free_stable_yaw_aug=getattr(
                    args, 'spacy_relation_free_stable_yaw_aug', False
                ),
                spacy_relation_free_view_small_yaw_aug=getattr(
                    args, 'spacy_relation_free_view_small_yaw_aug', False
                ),
                spacy_relation_free_rawview_global_only_train=getattr(
                    args, 'spacy_relation_free_rawview_global_only_train', False
                ),
                spacy_relation_free_none_aug=getattr(
                    args, 'spacy_relation_free_none_aug', False
                ),
                spacy_relation_free_compass_guard_aug=getattr(
                    args, 'spacy_relation_free_compass_guard_aug', False
                ),
                spacy_direction_sensitive_no_jitter_aug=getattr(
                    args, 'spacy_direction_sensitive_no_jitter_aug', False
                ),
                scanrefer_inject_spacy_decomp=getattr(
                    args, 'scanrefer_inject_spacy_decomp', False
                ),
                text_target_alias_policy=getattr(
                    args, 'text_target_alias_policy', 'strict'
                ),
            )
        test_dataset = Joint3DDataset(
            dataset_dict=dataset_dict,
            test_dataset=args.test_dataset,
            split='val' if not args.eval_train else 'train',
            use_color=args.use_color, use_height=args.use_height,
            overfit=args.debug,
            data_path=args.data_root,
            detect_intermediate=args.detect_intermediate,
            use_multiview=args.use_multiview,
            butd=args.butd,
            butd_gt=args.butd_gt,
            butd_cls=args.butd_cls,
            augment_det=args.augment_det,
            disable_box_jitter=args.disable_box_jitter,
            disable_train_augmentation=getattr(
                args, 'disable_train_augmentation', False
            ),
            spacy_relation_free_yaw_only_aug=getattr(
                args, 'spacy_relation_free_yaw_only_aug', False
            ),
            spacy_relation_free_view_guard_aug=getattr(
                args, 'spacy_relation_free_view_guard_aug', False
            ),
            spacy_relation_free_stable_yaw_aug=getattr(
                args, 'spacy_relation_free_stable_yaw_aug', False
            ),
            spacy_relation_free_view_small_yaw_aug=getattr(
                args, 'spacy_relation_free_view_small_yaw_aug', False
            ),
            spacy_relation_free_rawview_global_only_train=getattr(
                args, 'spacy_relation_free_rawview_global_only_train', False
            ),
            spacy_relation_free_none_aug=getattr(
                args, 'spacy_relation_free_none_aug', False
            ),
            spacy_relation_free_compass_guard_aug=getattr(
                args, 'spacy_relation_free_compass_guard_aug', False
            ),
            spacy_direction_sensitive_no_jitter_aug=getattr(
                args, 'spacy_direction_sensitive_no_jitter_aug', False
            ),
            scanrefer_inject_spacy_decomp=getattr(
                args, 'scanrefer_inject_spacy_decomp', False
            ),
            text_target_alias_policy=getattr(
                args, 'text_target_alias_policy', 'strict'
            ),
        )
        eval_max_samples = int(getattr(args, 'eval_max_samples', -1))
        if eval_max_samples >= 0 and hasattr(test_dataset, 'annos'):
            test_dataset.annos = test_dataset.annos[:eval_max_samples]
        return train_dataset, test_dataset

    @staticmethod
    def get_model(args):
        """Initialize the model."""
        num_input_channel = int(args.use_color) * 3
        if args.use_height:
            num_input_channel += 1
        if args.use_multiview:
            num_input_channel += 128
        if args.use_soft_token_loss:
            num_class = 256
        else:
            num_class = 19
        model = BeaUTyDETR(
            num_class=num_class,
            num_obj_class=485,
            input_feature_dim=num_input_channel,
            num_queries=args.num_target,
            num_decoder_layers=args.num_decoder_layers,
            self_position_embedding=args.self_position_embedding,
            contrastive_align_loss=args.use_contrastive_align,
            butd=args.butd or args.butd_gt or args.butd_cls,
            pointnet_ckpt=args.pp_checkpoint,
            self_attend=args.self_attend,
            # Structured reasoning
            use_structured_slots=args.use_structured_slots,
            use_late_acd=args.use_late_acd,
            slot_pooling=args.slot_pooling,
            max_rel_anchor_pairs=args.max_rel_anchor_pairs,
            acd_top_m_targets=args.acd_top_m_targets,
            acd_top_k_anchors=args.acd_top_k_anchors,
            acd_geo_dim=args.acd_geo_dim,
            acd_hidden_dim=args.acd_hidden_dim,
            acd_global_residual_alpha=args.acd_global_residual_alpha,
            acd_use_confidence_fusion=args.acd_use_confidence_fusion,
            acd_warmup_steps=args.acd_warmup_steps,
            acd_initial_alpha=args.acd_initial_alpha,
            acd_ea_scale=args.acd_ea_scale,
            acd_pool_ea_multiplier=args.acd_pool_ea_multiplier,
            acd_final_ea_multiplier=args.acd_final_ea_multiplier,
            acd_disable_struct_rerank=args.acd_disable_struct_rerank,
            acd_base_score_source=args.acd_base_score_source,
            dhc_margin_min=args.dhc_margin_min,
            dhc_temperature_max=args.dhc_temperature_max,
            structured_debug=args.structured_debug,
            use_quality_head=args.use_quality_head,
            use_sacr=args.use_sacr,
            sacr_top_m_targets=args.sacr_top_m_targets,
            sacr_top_k_anchors=args.sacr_top_k_anchors,
            sacr_hidden_dim=args.sacr_hidden_dim,
            sacr_geo_dim=args.sacr_geo_dim,
            sacr_disable_relation=args.sacr_disable_relation,
            use_rapf=args.use_rapf,
            rapf_hidden_dim=args.rapf_hidden_dim,
            rapf_initial_gate_bias=args.rapf_initial_gate_bias,
            rapf_use_quality=args.rapf_use_quality,
            rapf_quality_weight=args.rapf_quality_weight,
            rapf_struct_residual_clip=args.rapf_struct_residual_clip,
            rapf_generic_gate_cap=args.rapf_generic_gate_cap,
            rapf_quality_anchor_structured_residual=(
                args.rapf_quality_anchor_structured_residual
            ),
            use_qahnl=args.use_qahnl,
            qahnl_score_source=args.qahnl_score_source,
            use_source_pool_selector=args.use_source_pool_selector,
            source_pool_selector_hidden_dim=args.source_pool_selector_hidden_dim,
            source_pool_selector_candidate_aware=args.source_pool_selector_candidate_aware,
            source_pool_selector_direct_choice=args.source_pool_selector_direct_choice,
            source_pool_selector_include_contrastive_choice=(
                args.source_pool_selector_include_contrastive_choice
            ),
            source_pool_selector_rank_features=(
                getattr(args, 'source_pool_selector_rank_features', False)
            ),
            source_pool_selector_pairdelta_features=(
                getattr(args, 'source_pool_selector_pairdelta_features', False)
            ),
            source_pool_selector_candidate_context=(
                getattr(args, 'source_pool_selector_candidate_context', False)
            ),
            source_pool_selector_candidate_context_k=(
                getattr(args, 'source_pool_selector_candidate_context_k', 5)
            ),
            source_pool_selector_include_detector_policy_choice=(
                getattr(
                    args,
                    'source_pool_selector_include_detector_policy_choice',
                    False,
                )
            ),
            source_pool_selector_choice_sources=getattr(
                args, 'source_pool_selector_choice_sources', None
            ),
            source_pool_selector_text_context=(
                getattr(args, 'source_pool_selector_text_context', False)
            ),
            source_pool_selector_metadata_context=(
                getattr(args, 'source_pool_selector_metadata_context', False)
            ),
            source_pool_selector_context_features=(
                getattr(args, 'source_pool_selector_context_features', False)
            ),
            source_pool_selector_separate_override_head=(
                getattr(args, 'source_pool_selector_separate_override_head',
                        False)
            ),
            source_pool_selector_override_initial_bias=(
                getattr(args, 'source_pool_selector_override_initial_bias',
                        -1.5)
            ),
            use_detector_policy_adapter=(
                getattr(args, 'use_detector_policy_adapter', False)
            ),
            use_detector_policy_teacher=(
                getattr(args, 'quality_topk_rerank_weight', 0.0) > 0.0
                and getattr(args, 'quality_topk_rerank_source', 'fused')
                in DETECTOR_POLICY_SOURCE_NAMES
            ),
            detector_policy_adapter_context=(
                getattr(args, 'detector_policy_adapter_context', False)
            ),
            detector_policy_adapter_hidden_dim=(
                getattr(args, 'detector_policy_adapter_hidden_dim', 32)
            ),
            detector_policy_adapter_delta_scale=(
                getattr(args, 'detector_policy_adapter_delta_scale', 0.25)
            ),
        )
        return model

    @staticmethod
    def _get_inputs(batch_data):
        inputs = {
            'point_clouds': batch_data['point_clouds'].float(),
            'text': batch_data['utterances'],
            "det_boxes": batch_data['all_detected_boxes'],
            "det_bbox_label_mask": batch_data['all_detected_bbox_label_mask'],
            "det_class_ids": batch_data['all_detected_class_ids']
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
            # Required inside the model so SACR/RAPF are anchored to the
            # official BBS soft-token target score rather than silently
            # falling back to the diagnostic contrastive score.
            'positive_map',
            'box_label_mask',
        ):
            if key in batch_data:
                inputs[key] = batch_data[key]
        return inputs

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader,
                           model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        if args.test_dataset == 'scannet':
            return self.evaluate_one_epoch_det(
                epoch, test_loader, model,
                criterion, set_criterion, args
            )
        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)

        if args.num_decoder_layers > 0:
            prefixes = ['last_', 'proposal_']
            prefixes = ['last_']
            prefixes.append('proposal_')
        else:
            prefixes = ['proposal_']  # only proposal
        prefixes += [f'{i}head_' for i in range(args.num_decoder_layers - 1)]

        if args.butd_cls or args.butd_gt:
            evaluator = GroundingGTEvaluator(prefixes=prefixes)
        else:
            evaluator = GroundingEvaluator(
                only_root=True, thresholds=[0.25, 0.5],
                topks=[1, 5, 10], prefixes=prefixes,
                source_choice_dump_path=getattr(
                    args, 'eval_dump_source_choice_features_path', None
                ),
                source_choice_dump_topk=getattr(
                    args, 'eval_dump_source_choice_topk', 1
                ),
            )

        # Main eval branch
        for batch_idx, batch_data in enumerate(test_loader):
            stat_dict, end_points = self._main_eval_branch(
                batch_idx, batch_data, test_loader, model, stat_dict,
                criterion, set_criterion, args
            )
            if evaluator is not None:
                for prefix in prefixes:
                    evaluator.evaluate(end_points, prefix)
        evaluator.synchronize_between_processes()
        eval_results = None
        if dist.get_rank() == 0:
            if evaluator is not None:
                eval_results = evaluator.print_stats()
                write_eval_results_json(
                    getattr(args, 'eval_results_json_path', None),
                    eval_results,
                )
        return eval_results

    @torch.no_grad()
    def evaluate_one_epoch_det(self, epoch, test_loader,
                               model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        dataset_config = ScannetDatasetConfig(18)
        # Used for AP calculation
        CONFIG_DICT = {
            'remove_empty_box': False, 'use_3d_nms': True,
            'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
            'per_class_proposal': True, 'conf_thresh': 0.0,
            'dataset_config': dataset_config,
            'hungarian_loss': True
        }
        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)
        if set_criterion is not None:
            set_criterion.eval()

        if args.num_decoder_layers > 0:
            prefixes = ['last_', 'proposal_']
            prefixes += [
                f'{i}head_' for i in range(args.num_decoder_layers - 1)
            ]
        else:
            prefixes = ['proposal_']  # only proposal
        prefixes = ['last_']
        ap_calculator_list = [
            APCalculator(iou_thresh, dataset_config.class2type)
            for iou_thresh in args.ap_iou_thresholds
        ]
        mAPs = [
            [iou_thresh, {k: 0 for k in prefixes}]
            for iou_thresh in args.ap_iou_thresholds
        ]

        batch_pred_map_cls_dict = {k: [] for k in prefixes}
        batch_gt_map_cls_dict = {k: [] for k in prefixes}

        # Main eval branch
        wordidx = np.array([
            0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11,
            12, 13, 13, 14, 15, 16, 16, 17, 17, 18, 18
        ])
        tokenidx = np.array([
            1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19, 21, 23,
            25, 27, 29, 31, 32, 34, 36, 38, 39, 41, 42, 44, 45
        ])
        for batch_idx, batch_data in enumerate(test_loader):
            stat_dict, end_points = self._main_eval_branch(
                batch_idx, batch_data, test_loader, model, stat_dict,
                criterion, set_criterion, args
            )
            # contrast
            proj_tokens = end_points['proj_tokens']  # (B, tokens, 64)
            proj_queries = end_points['last_proj_queries']  # (B, Q, 64)
            sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
            sem_scores_ = sem_scores / 0.07  # (B, Q, tokens)
            # Pad to cover all token indices (dynamic, not hardcoded)
            max_tok_idx = max((int(t) for t in tokenidx), default=-1) + 1
            max_tok = max(sem_scores_.size(2), max_tok_idx)
            sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), max_tok)
            sem_scores = sem_scores.to(sem_scores_.device)
            sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_
            end_points['last_sem_cls_scores'] = sem_scores
            # end contrast
            sem_cls = torch.zeros_like(end_points['last_sem_cls_scores'])[..., :19]
            for w, t in zip(wordidx, tokenidx):
                sem_cls[..., w] += end_points['last_sem_cls_scores'][..., t]
            end_points['last_sem_cls_scores'] = sem_cls

            # Parse predictions
            # for prefix in prefixes:
            prefix = 'last_'
            batch_pred_map_cls = parse_predictions(
                end_points, CONFIG_DICT, prefix,
                size_cls_agnostic=True)
            batch_gt_map_cls = parse_groundtruths(
                end_points, CONFIG_DICT,
                size_cls_agnostic=True)
            batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
            batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

        mAP = 0.0
        # for prefix in prefixes:
        prefix = 'last_'
        for (batch_pred_map_cls, batch_gt_map_cls) in zip(
                batch_pred_map_cls_dict[prefix],
                batch_gt_map_cls_dict[prefix]):
            for ap_calculator in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        # Evaluate average precision
        for i, ap_calculator in enumerate(ap_calculator_list):
            metrics_dict = ap_calculator.compute_metrics()
            self.logger.info(
                '=====================>'
                f'{prefix} IOU THRESH: {args.ap_iou_thresholds[i]}'
                '<====================='
            )
            for key in metrics_dict:
                self.logger.info(f'{key} {metrics_dict[key]}')
            if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                mAP = metrics_dict['mAP']
            mAPs[i][1][prefix] = metrics_dict['mAP']
            ap_calculator.reset()

        for mAP in mAPs:
            self.logger.info(
                f'IoU[{mAP[0]}]:\t'
                + ''.join([
                    f'{key}: {mAP[1][key]:.4f} \t'
                    for key in sorted(mAP[1].keys())
                ])
            )

        return None


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    opt = parse_option()
    configure_reproducibility(opt.rng_seed)
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_tester = TrainTester(opt)
    ckpt_path = train_tester.main(opt)
