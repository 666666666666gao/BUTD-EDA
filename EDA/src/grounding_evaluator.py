# ------------------------------------------------------------------------
# Modification: EDA
# Created: 05/21/2022
# Author: Yanmin Wu
# E-mail: wuyanminmax@gmail.com
# https://github.com/yanmin-wu/EDA 
# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""A class to collect and evaluate language grounding results."""

import os

import torch

from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
import utils.misc as misc
import numpy as np

def softmax(x):
    """Numpy function for softmax."""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs

# BRIEF Evaluator
class GroundingEvaluator:
    """
    Evaluate language grounding.

    Args:
        only_root (bool): detect only the root noun
        thresholds (list): IoU thresholds to check
        topks (list): k to evaluate top--k accuracy
        prefixes (list): names of layers to evaluate
    """

    SPACY_AUGMENTATION_MODE_NAMES = {
        1: 'none',
        2: 'yaw_only',
        3: 'full_natural',
    }
    SPACY_AUGMENTATION_PROFILE_NAMES = {
        1: 'none',
        2: 'yaw_relation',
        3: 'full_natural_relation_free',
        4: 'yaw_relation_free',
        5: 'none_relation_free_view',
        6: 'yaw_relation_free_stable',
        7: 'small_yaw_relation_free_view',
        8: 'rawview_relation_free_global_only',
        9: 'none_relation_free',
        10: 'none_relation_free_compass',
    }
    RERANK_DIAGNOSTIC_SCALES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
    PREV_BOX_BLEND_DIAGNOSTIC_ALPHAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.25)
    DET_BOX_BLEND_DIAGNOSTIC_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
    BOX_SIZE_SCALE_DIAGNOSTIC_FACTORS = (
        0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30,
    )

    def __init__(self, only_root=True, thresholds=[0.25, 0.5],
                 topks=[1, 5, 10], prefixes=[], filter_non_gt_boxes=False):
        """Initialize accumulators."""
        self.only_root = only_root
        self.thresholds = thresholds
        self.topks = topks
        self.prefixes = prefixes
        self.filter_non_gt_boxes = filter_non_gt_boxes
        self.reset()

    def reset(self):
        """Reset accumulators to empty."""
        self.dets = {
            (prefix, t, k, mode): 0
            for prefix in self.prefixes
            for t in self.thresholds
            for k in self.topks
            for mode in ['bbs', 'bbf']
        }
        self.gts = dict(self.dets)

        self.dets.update({'vd': 0, 'vid': 0})
        self.dets.update({'hard': 0, 'easy': 0})
        self.dets.update({'multi': 0, 'unique': 0})
        self.gts.update({'vd': 1e-14, 'vid': 1e-14})
        self.gts.update({'hard': 1e-14, 'easy': 1e-14})
        self.gts.update({'multi': 1e-14, 'unique': 1e-14})
        self.dets.update({'vd50': 0, 'vid50': 0})
        self.dets.update({'hard50': 0, 'easy50': 0})
        self.dets.update({'multi50': 0, 'unique50': 0})
        self.gts.update({'vd50': 1e-14, 'vid50': 1e-14})
        self.gts.update({'hard50': 1e-14, 'easy50': 1e-14})
        self.gts.update({'multi50': 1e-14, 'unique50': 1e-14})
        for name in self.SPACY_AUGMENTATION_MODE_NAMES.values():
            self.dets[f'spacy_aug_{name}'] = 0
            self.gts[f'spacy_aug_{name}'] = 0
            self.dets[f'spacy_aug_{name}50'] = 0
            self.gts[f'spacy_aug_{name}50'] = 0
        for name in self.SPACY_AUGMENTATION_PROFILE_NAMES.values():
            self.dets[f'spacy_profile_{name}'] = 0
            self.gts[f'spacy_profile_{name}'] = 0
            self.dets[f'spacy_profile_{name}50'] = 0
            self.gts[f'spacy_profile_{name}50'] = 0
        self.dets.update({
            'diag_sem_total': 0,
            'diag_sem_top1_changed': 0,
            'diag_sem_fix25': 0,
            'diag_sem_break25': 0,
            'diag_sem_fix50': 0,
            'diag_sem_break50': 0,
            'diag_sem_base_top1_iou_sum': 0.0,
            'diag_sem_eval_top1_iou_sum': 0.0,
            'diag_sem_best_iou_sum': 0.0,
            'diag_sem_oracle_top10_iou_sum': 0.0,
        })
        for suffix in ('25', '50'):
            self.dets[f'diag_sem_fail{suffix}_total'] = 0
            self.dets[f'diag_sem_fail{suffix}_best_rankable'] = 0
            self.dets[f'diag_sem_fail{suffix}_top10_rankable'] = 0
            self.dets[f'diag_sem_fail{suffix}_unrankable'] = 0
        for scale in self.RERANK_DIAGNOSTIC_SCALES:
            scale_key = str(scale).replace('.', 'p')
            self.dets[f'diag_rerank_scale_{scale_key}_hit25'] = 0
            self.dets[f'diag_rerank_scale_{scale_key}_hit50'] = 0
        self.dets['diag_prev_box_blend_total'] = 0
        for alpha in self.PREV_BOX_BLEND_DIAGNOSTIC_ALPHAS:
            alpha_key = str(alpha).replace('.', 'p')
            self.dets[f'diag_prev_box_blend_{alpha_key}_hit25'] = 0
            self.dets[f'diag_prev_box_blend_{alpha_key}_hit50'] = 0
        self.dets['diag_det_box_blend_total'] = 0
        for alpha in self.DET_BOX_BLEND_DIAGNOSTIC_ALPHAS:
            alpha_key = str(alpha).replace('.', 'p')
            self.dets[f'diag_det_box_blend_{alpha_key}_hit25'] = 0
            self.dets[f'diag_det_box_blend_{alpha_key}_hit50'] = 0
        self.dets['diag_box_size_scale_total'] = 0
        for factor in self.BOX_SIZE_SCALE_DIAGNOSTIC_FACTORS:
            factor_key = str(factor).replace('.', 'p')
            self.dets[f'diag_box_size_scale_{factor_key}_hit25'] = 0
            self.dets[f'diag_box_size_scale_{factor_key}_hit50'] = 0
        self.dets.update({
            'diag_cls_total': 0,
            'diag_cls_eval_top_match': 0,
            'diag_cls_best_iou_match': 0,
            'diag_cls_fail25_total': 0,
            'diag_cls_fail25_eval_match': 0,
            'diag_cls_fail25_eval_mismatch': 0,
            'diag_cls_fail50_total': 0,
            'diag_cls_fail50_eval_match': 0,
            'diag_cls_fail50_eval_mismatch': 0,
        })
        for prefix in ('eval_top', 'best_iou'):
            for name in ('main', 'attr', 'pron', 'rel', 'other'):
                self.dets[f'diag_comp_{prefix}_{name}_sum'] = 0.0
        self.semantic_diagnostic_rows = []
        self.semantic_diagnostic_dump_path = ''

    def print_stats(self):
        """Print accumulated accuracies."""
        mode_str = {
            'bbs': 'position alignment',
            'bbf': 'semantic alignment'
        }
        for prefix in self.prefixes:
            for mode in ['bbs', 'bbf']:
                for t in self.thresholds:
                    print(
                        prefix, mode_str[mode], 'Acc%.2f:' % t,
                        ', '.join([
                            'Top-%d: %.5f' % (
                                k,
                                self.dets[(prefix, t, k, mode)]
                                / max(self.gts[(prefix, t, k, mode)], 1)
                            )
                            for k in self.topks
                        ])
                    )
        print('\nAnalysis')
        print('iou@0.25')
        for field in ['easy', 'hard', 'vd', 'vid', 'unique', 'multi']:
            print(field, self.dets[field] / self.gts[field])
        print('iou@0.50')
        for field in ['easy50', 'hard50', 'vd50', 'vid50', 'unique50', 'multi50']:
            print(field, self.dets[field] / self.gts[field])
        self._print_spacy_augmentation_stats()
        self._print_spacy_augmentation_profile_stats()
        self._print_diagnostic_stats()

    def _print_spacy_augmentation_stats(self):
        fields = [
            f'spacy_aug_{name}'
            for name in self.SPACY_AUGMENTATION_MODE_NAMES.values()
        ]
        if not any(self.gts[field] > 0 for field in fields):
            return
        print('spacy augmentation iou@0.25')
        for field in fields:
            print(field, self.dets[field] / max(self.gts[field], 1))
        print('spacy augmentation iou@0.50')
        for field in fields:
            field50 = f'{field}50'
            print(field50, self.dets[field50] / max(self.gts[field50], 1))

    def _print_spacy_augmentation_profile_stats(self):
        fields = [
            f'spacy_profile_{name}'
            for name in self.SPACY_AUGMENTATION_PROFILE_NAMES.values()
        ]
        if not any(self.gts[field] > 0 for field in fields):
            return
        print('spacy augmentation profile iou@0.25')
        for field in fields:
            print(field, self.dets[field] / max(self.gts[field], 1))
        print('spacy augmentation profile iou@0.50')
        for field in fields:
            field50 = f'{field}50'
            print(field50, self.dets[field50] / max(self.gts[field50], 1))

    def _print_diagnostic_stats(self):
        total = self.dets.get('diag_sem_total', 0)
        if total <= 0:
            return
        print('\nDiagnostics')
        print(
            'semantic_diag',
            'top1_changed %.5f' % (
                self.dets['diag_sem_top1_changed'] / max(total, 1)
            ),
            'fix25 %.5f' % (self.dets['diag_sem_fix25'] / max(total, 1)),
            'break25 %.5f' % (self.dets['diag_sem_break25'] / max(total, 1)),
            'fix50 %.5f' % (self.dets['diag_sem_fix50'] / max(total, 1)),
            'break50 %.5f' % (self.dets['diag_sem_break50'] / max(total, 1)),
        )
        print(
            'semantic_iou',
            'base_top1 %.5f' % (
                self.dets['diag_sem_base_top1_iou_sum'] / max(total, 1)
            ),
            'eval_top1 %.5f' % (
                self.dets['diag_sem_eval_top1_iou_sum'] / max(total, 1)
            ),
            'best %.5f' % (
                self.dets['diag_sem_best_iou_sum'] / max(total, 1)
            ),
            'oracle_top10 %.5f' % (
                self.dets['diag_sem_oracle_top10_iou_sum'] / max(total, 1)
            ),
        )
        for suffix in ('25', '50'):
            fail_total = self.dets[f'diag_sem_fail{suffix}_total']
            if fail_total <= 0:
                continue
            best_rankable = self.dets[f'diag_sem_fail{suffix}_best_rankable']
            top10_rankable = self.dets[f'diag_sem_fail{suffix}_top10_rankable']
            outside_top10 = max(best_rankable - top10_rankable, 0)
            print(
                f'semantic_failure{suffix}',
                'fail_rate %.5f' % (fail_total / max(total, 1)),
                'rankable_best %.5f' % (
                    best_rankable / max(fail_total, 1)
                ),
                'rankable_top10 %.5f' % (
                    top10_rankable / max(fail_total, 1)
                ),
                'outside_top10 %.5f' % (
                    outside_top10 / max(fail_total, 1)
                ),
                'unrankable %.5f' % (
                    self.dets[f'diag_sem_fail{suffix}_unrankable']
                    / max(fail_total, 1)
                ),
            )
        for prefix in ('eval_top', 'best_iou'):
            print(
                f'component_{prefix}',
                'main %.5f' % (
                    self.dets[f'diag_comp_{prefix}_main_sum'] / max(total, 1)
                ),
                'attr %.5f' % (
                    self.dets[f'diag_comp_{prefix}_attr_sum'] / max(total, 1)
                ),
                'pron %.5f' % (
                    self.dets[f'diag_comp_{prefix}_pron_sum'] / max(total, 1)
                ),
                'rel %.5f' % (
                    self.dets[f'diag_comp_{prefix}_rel_sum'] / max(total, 1)
                ),
                'other %.5f' % (
                    self.dets[f'diag_comp_{prefix}_other_sum'] / max(total, 1)
                ),
            )
        self._print_class_diagnostic_stats()
        self._print_rerank_scale_stats()
        self._print_box_blend_stats()
        self._save_semantic_diagnostic_dump()

    @staticmethod
    def _batch_item(value, bid, default=None):
        if value is None:
            return default
        if torch.is_tensor(value):
            if value.shape[0] <= bid:
                return default
            item = value[bid].detach().cpu()
            return item.item() if item.numel() == 1 else item.numpy()
        if isinstance(value, (list, tuple)):
            return value[bid] if len(value) > bid else default
        return value

    @staticmethod
    def _query_vector(end_points, key, bid, num_queries, device):
        value = end_points.get(key, None)
        if value is None or not torch.is_tensor(value):
            return torch.zeros(num_queries, device=device)
        value = value[bid].to(device=device, dtype=torch.float32).reshape(-1)
        if value.numel() >= num_queries:
            return value[:num_queries]
        return torch.cat([
            value,
            value.new_zeros(num_queries - value.numel()),
        ])

    def _record_semantic_diagnostic_dump(self, end_points, bid, query_iou,
                                         base_scores, eval_scores, pred_bbox):
        dump_path = str(end_points.get('eval_diagnostic_dump_path', '') or '')
        if not dump_path:
            return
        self.semantic_diagnostic_dump_path = dump_path
        num_queries = query_iou.numel()
        device = query_iou.device
        row = {
            'query_iou': query_iou.detach().float().cpu().numpy(),
            'semantic_base': base_scores[0].detach().float().cpu().numpy(),
            'semantic_eval': eval_scores[0].detach().float().cpu().numpy(),
            'pred_bbox': pred_bbox.detach().float().cpu().numpy(),
        }
        score_keys = (
            'semantic_rerank_base_scores',
            'semantic_rerank_residual',
            'semantic_rerank_primary_residual',
            'semantic_rerank_aux_residual',
            'fused_scores',
            'pred_iou',
            'structured_scores',
            'base_grounding_scores',
            'target_attr_scores',
            'relation_anchor_scores',
            'semantic_threshold_residual',
            'semantic_support_scores_without_threshold',
        )
        for key in score_keys:
            row[key] = self._query_vector(
                end_points, key, bid, num_queries, device
            ).detach().cpu().numpy()

        threshold_logits = end_points.get('semantic_threshold_logits', None)
        if torch.is_tensor(threshold_logits):
            threshold_row = threshold_logits[bid].to(
                device=device, dtype=torch.float32
            )
            for threshold_idx, threshold_name in enumerate(('25', '50')):
                row[f'semantic_threshold_logit{threshold_name}'] = (
                    threshold_row[:num_queries, threshold_idx]
                    .detach().cpu().numpy()
                )
        else:
            zeros = torch.zeros(num_queries, device=device)
            row['semantic_threshold_logit25'] = zeros.cpu().numpy()
            row['semantic_threshold_logit50'] = zeros.cpu().numpy()

        support_gate = end_points.get('semantic_support_gate', None)
        if torch.is_tensor(support_gate):
            row['semantic_support_gate'] = float(
                support_gate[bid].detach().float().reshape(-1)[0].cpu().item()
            )
        else:
            row['semantic_support_gate'] = 1.0

        # Keep only the two competing query representations and the target
        # language slot. This makes support-gate capacity auditable without
        # writing the full [Q, D] query tensor for every description.
        last_queries = end_points.get('last_queries', None)
        semantic_scores = end_points.get('semantic_rerank_scores', None)
        support_scores = end_points.get('semantic_support_scores', None)
        if (
            torch.is_tensor(last_queries)
            and torch.is_tensor(semantic_scores)
            and torch.is_tensor(support_scores)
        ):
            query_row = last_queries[bid].detach().float()
            semantic_top = int(semantic_scores[bid].argmax().item())
            support_top = int(support_scores[bid].argmax().item())
            row['semantic_top_query_feat'] = (
                query_row[semantic_top].cpu().numpy().astype(np.float16)
            )
            row['support_top_query_feat'] = (
                query_row[support_top].cpu().numpy().astype(np.float16)
            )

            fixed_support_scores = end_points.get(
                'semantic_support_scores_without_threshold', support_scores
            )
            fixed_order = torch.argsort(
                fixed_support_scores[bid], descending=True
            )
            fixed_top1 = int(fixed_order[0].item())
            fixed_top2 = int(fixed_order[1].item())
            row['fixed_support_top1_query_feat'] = (
                query_row[fixed_top1].cpu().numpy().astype(np.float16)
            )
            row['fixed_support_top2_query_feat'] = (
                query_row[fixed_top2].cpu().numpy().astype(np.float16)
            )

            slot_dict = end_points.get('slot_dict', {})
            target_slot = (
                slot_dict.get('target_slot', None)
                if isinstance(slot_dict, dict) else None
            )
            if torch.is_tensor(target_slot):
                target_feat = target_slot[bid].detach().float()
            else:
                target_feat = query_row.new_zeros(query_row.shape[-1])
            row['target_slot_feat'] = (
                target_feat.cpu().numpy().astype(np.float16)
            )

        components = end_points.get('semantic_component_raw_scores', None)
        if torch.is_tensor(components):
            component_row = components[bid].to(
                device=device, dtype=torch.float32
            )
            component_row = component_row[:num_queries, :5]
            if component_row.shape[0] < num_queries:
                component_row = torch.cat([
                    component_row,
                    component_row.new_zeros(
                        num_queries - component_row.shape[0],
                        component_row.shape[1],
                    ),
                ], dim=0)
        else:
            component_row = torch.zeros(num_queries, 5, device=device)
        row['semantic_components'] = component_row.detach().cpu().numpy()

        nearest_class = torch.full(
            (num_queries,), -1, dtype=torch.long, device=device
        )
        nearest_overlap = torch.zeros(num_queries, device=device)
        nearest_logit = torch.zeros(num_queries, device=device)
        detected_boxes = end_points.get('all_detected_boxes', None)
        detected_classes = end_points.get('all_detected_class_ids', None)
        detected_mask = end_points.get('all_detected_bbox_label_mask', None)
        if (
            torch.is_tensor(detected_boxes)
            and torch.is_tensor(detected_classes)
            and torch.is_tensor(detected_mask)
        ):
            valid_det = detected_mask[bid].bool()
            if valid_det.any():
                det_boxes = detected_boxes[bid][valid_det].to(device=device)
                det_classes = detected_classes[bid][valid_det].to(device=device)
                det_ious, _ = _iou3d_par(
                    box_cxcyczwhd_to_xyzxyz(det_boxes),
                    box_cxcyczwhd_to_xyzxyz(pred_bbox),
                )
                nearest_overlap, nearest_idx = det_ious.max(dim=0)
                nearest_class = det_classes[nearest_idx].long()
                detected_logits = end_points.get('all_detected_logits', None)
                if torch.is_tensor(detected_logits):
                    valid_logits = detected_logits[bid][valid_det].to(
                        device=device, dtype=torch.float32
                    )
                    if valid_logits.dim() == 1:
                        nearest_logit = valid_logits[nearest_idx]
                    elif valid_logits.dim() >= 2:
                        nearest_logit = valid_logits.max(dim=-1).values[nearest_idx]
        row['nearest_detector_class'] = nearest_class.cpu().numpy()
        row['nearest_detector_overlap'] = nearest_overlap.float().cpu().numpy()
        row['nearest_detector_logit'] = nearest_logit.float().cpu().numpy()

        target_slot = self._batch_item(end_points.get('target_slot'), bid, {})
        if isinstance(target_slot, dict):
            target_text = str(target_slot.get('text', ''))
        else:
            target_text = str(target_slot or '')
        row.update({
            'target_text': target_text,
            'description': str(self._batch_item(
                end_points.get('description'), bid, ''
            )),
            'scene_id': str(self._batch_item(
                end_points.get('scene_id'), bid, ''
            )),
            'object_id': str(self._batch_item(
                end_points.get('object_id'), bid, ''
            )),
            'target_cid': int(self._batch_item(
                end_points.get('target_cid'), bid, -1
            )),
            'is_unique': bool(self._batch_item(
                end_points.get('is_unique'), bid, False
            )),
            'is_hard': bool(self._batch_item(
                end_points.get('is_hard'), bid, False
            )),
            'is_view_dep': bool(self._batch_item(
                end_points.get('is_view_dep'), bid, False
            )),
            'parse_confidence': float(self._batch_item(
                end_points.get('parse_confidence'), bid, 0.0
            )),
        })
        self.semantic_diagnostic_rows.append(row)

    def _save_semantic_diagnostic_dump(self):
        path = self.semantic_diagnostic_dump_path
        rows = self.semantic_diagnostic_rows
        if not path or not rows:
            return
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        keys = rows[0].keys()
        payload = {}
        for key in keys:
            values = [row[key] for row in rows]
            first = values[0]
            if isinstance(first, np.ndarray):
                payload[key] = np.stack(values, axis=0)
            else:
                payload[key] = np.asarray(values)
        np.savez_compressed(path, **payload)
        print(
            'semantic_diagnostic_dump',
            path,
            'samples',
            len(rows),
        )

    def _print_class_diagnostic_stats(self):
        total = self.dets.get('diag_cls_total', 0)
        if total <= 0:
            return
        print(
            'detector_class_diag',
            'eval_top_match %.5f' % (
                self.dets['diag_cls_eval_top_match'] / max(total, 1)
            ),
            'best_iou_match %.5f' % (
                self.dets['diag_cls_best_iou_match'] / max(total, 1)
            ),
        )
        for suffix in ('25', '50'):
            fail_total = self.dets[f'diag_cls_fail{suffix}_total']
            if fail_total <= 0:
                continue
            print(
                f'detector_class_fail{suffix}',
                'eval_match %.5f' % (
                    self.dets[f'diag_cls_fail{suffix}_eval_match']
                    / max(fail_total, 1)
                ),
                'eval_mismatch %.5f' % (
                    self.dets[f'diag_cls_fail{suffix}_eval_mismatch']
                    / max(fail_total, 1)
                ),
            )
    def _print_rerank_scale_stats(self):
        rerank_total = self.dets.get('diag_sem_total', 0)
        if rerank_total > 0:
            for scale in self.RERANK_DIAGNOSTIC_SCALES:
                scale_key = str(scale).replace('.', 'p')
                hit25 = self.dets[
                    f'diag_rerank_scale_{scale_key}_hit25'
                ] / rerank_total
                hit50 = self.dets[
                    f'diag_rerank_scale_{scale_key}_hit50'
                ] / rerank_total
                print(
                    'rerank_scale_diag',
                    'scale %.2f' % scale,
                    'acc25 %.5f' % hit25,
                    'acc50 %.5f' % hit50,
                )

    def _print_box_blend_stats(self):
        families = (
            (
                'prev_box_blend_diag',
                'diag_prev_box_blend',
                self.PREV_BOX_BLEND_DIAGNOSTIC_ALPHAS,
            ),
            (
                'det_box_blend_diag',
                'diag_det_box_blend',
                self.DET_BOX_BLEND_DIAGNOSTIC_ALPHAS,
            ),
        )
        for label, key_prefix, alphas in families:
            total = self.dets.get(f'{key_prefix}_total', 0)
            if total <= 0:
                continue
            for alpha in alphas:
                alpha_key = str(alpha).replace('.', 'p')
                print(
                    label,
                    'last_alpha %.2f' % alpha,
                    'acc25 %.5f' % (
                        self.dets[f'{key_prefix}_{alpha_key}_hit25'] / total
                    ),
                    'acc50 %.5f' % (
                        self.dets[f'{key_prefix}_{alpha_key}_hit50'] / total
                    ),
                )
        size_total = self.dets.get('diag_box_size_scale_total', 0)
        if size_total > 0:
            for factor in self.BOX_SIZE_SCALE_DIAGNOSTIC_FACTORS:
                factor_key = str(factor).replace('.', 'p')
                print(
                    'box_size_scale_diag',
                    'size_factor %.2f' % factor,
                    'acc25 %.5f' % (
                        self.dets[
                            f'diag_box_size_scale_{factor_key}_hit25'
                        ] / size_total
                    ),
                    'acc50 %.5f' % (
                        self.dets[
                            f'diag_box_size_scale_{factor_key}_hit50'
                        ] / size_total
                    ),
                )

    def synchronize_between_processes(self):
        all_dets = misc.all_gather(self.dets)
        all_gts = misc.all_gather(self.gts)

        if misc.is_main_process():
            merged_predictions = {}
            for key in all_dets[0].keys():
                merged_predictions[key] = 0
                for p in all_dets:
                    merged_predictions[key] += p[key]
            self.dets = merged_predictions

            merged_predictions = {}
            for key in all_gts[0].keys():
                merged_predictions[key] = 0
                for p in all_gts:
                    merged_predictions[key] += p[key]
            self.gts = merged_predictions

    # BRIEF Evaluation
    def evaluate(self, end_points, prefix):
        """
        Evaluate all accuracies.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # NOTE Two Evaluation Ways: position alignment, semantic alignment
        self.evaluate_bbox_by_pos_align(end_points, prefix)
        self.evaluate_bbox_by_sem_align(end_points, prefix)

    def _record_semantic_diagnostics(self, end_points, prefix, bid,
                                     base_scores, eval_scores,
                                     pred_bbox, gt_bbox):
        if prefix != 'last_':
            return
        if not self._flag_enabled(
            end_points.get('eval_report_diagnostic_scores', False)
        ):
            return
        if base_scores.shape[0] != 1 or eval_scores.shape[0] != 1:
            return

        ious, _ = _iou3d_par(
            box_cxcyczwhd_to_xyzxyz(gt_bbox[:1]),
            box_cxcyczwhd_to_xyzxyz(pred_bbox),
        )
        query_iou = ious[0]
        base_rank = base_scores[0].argsort(descending=True)
        eval_rank = eval_scores[0].argsort(descending=True)
        base_top = base_rank[0]
        eval_top = eval_rank[0]
        best_top = query_iou.argmax()
        oracle_top10_iou = query_iou[eval_rank[:10]].max()
        base_iou = query_iou[base_top]
        eval_iou = query_iou[eval_top]
        best_iou = query_iou[best_top]

        self.dets['diag_sem_total'] += 1
        self.dets['diag_sem_top1_changed'] += int(base_top.item() != eval_top.item())
        self.dets['diag_sem_base_top1_iou_sum'] += float(base_iou.item())
        self.dets['diag_sem_eval_top1_iou_sum'] += float(eval_iou.item())
        self.dets['diag_sem_best_iou_sum'] += float(best_iou.item())
        self.dets['diag_sem_oracle_top10_iou_sum'] += float(
            oracle_top10_iou.item()
        )
        self._record_semantic_diagnostic_dump(
            end_points, bid, query_iou, base_scores, eval_scores, pred_bbox
        )
        self._record_semantic_class_diagnostics(
            end_points, bid, eval_top, best_top, eval_iou, pred_bbox
        )
        rerank_base = end_points.get('semantic_rerank_base_scores', None)
        rerank_residual = end_points.get('semantic_rerank_residual', None)
        if rerank_base is not None and rerank_residual is not None:
            rerank_base = rerank_base[bid].to(query_iou.device)
            rerank_residual = rerank_residual[bid].to(query_iou.device)
            for scale in self.RERANK_DIAGNOSTIC_SCALES:
                scale_key = str(scale).replace('.', 'p')
                scaled_top = (
                    rerank_base + scale * rerank_residual
                ).argmax()
                scaled_iou = float(query_iou[scaled_top].item())
                self.dets[
                    f'diag_rerank_scale_{scale_key}_hit25'
                ] += int(scaled_iou > 0.25)
                self.dets[
                    f'diag_rerank_scale_{scale_key}_hit50'
                ] += int(scaled_iou > 0.50)

        prev_center = end_points.get('4head_center', None)
        prev_size = end_points.get('4head_pred_size', None)
        report_box_blends = self._flag_enabled(
            end_points.get('eval_report_box_blend_diagnostics', False)
        )
        if report_box_blends:
            selected_bbox = pred_bbox[eval_top]
            factors = selected_bbox.new_tensor(
                self.BOX_SIZE_SCALE_DIAGNOSTIC_FACTORS
            )
            scaled = selected_bbox.unsqueeze(0).expand(
                len(self.BOX_SIZE_SCALE_DIAGNOSTIC_FACTORS), -1
            ).clone()
            scaled[:, 3:] = (
                scaled[:, 3:] * factors.unsqueeze(1)
            ).clamp(min=1e-6)
            scale_iou, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bbox[:1]),
                box_cxcyczwhd_to_xyzxyz(scaled),
            )
            self.dets['diag_box_size_scale_total'] += 1
            for factor, iou_value in zip(
                self.BOX_SIZE_SCALE_DIAGNOSTIC_FACTORS,
                scale_iou[0].detach().cpu().tolist(),
            ):
                factor_key = str(factor).replace('.', 'p')
                self.dets[
                    f'diag_box_size_scale_{factor_key}_hit25'
                ] += int(iou_value > 0.25)
                self.dets[
                    f'diag_box_size_scale_{factor_key}_hit50'
                ] += int(iou_value > 0.50)
        if (
            report_box_blends
            and prev_center is not None
            and prev_size is not None
        ):
            prev_bbox = torch.cat(
                [prev_center[bid], prev_size[bid]], dim=-1
            ).to(pred_bbox.device)
            last_selected = pred_bbox[eval_top]
            prev_selected = prev_bbox[eval_top]
            self.dets['diag_prev_box_blend_total'] += 1
            alphas = last_selected.new_tensor(
                self.PREV_BOX_BLEND_DIAGNOSTIC_ALPHAS
            ).unsqueeze(1)
            blended = (
                alphas * last_selected.unsqueeze(0)
                + (1.0 - alphas) * prev_selected.unsqueeze(0)
            )
            blended[:, 3:] = blended[:, 3:].clamp(min=1e-6)
            blend_iou, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bbox[:1]),
                box_cxcyczwhd_to_xyzxyz(blended),
            )
            blend_ious = blend_iou[0].detach().cpu().tolist()
            for alpha, iou_value in zip(
                self.PREV_BOX_BLEND_DIAGNOSTIC_ALPHAS, blend_ious
            ):
                alpha_key = str(alpha).replace('.', 'p')
                self.dets[
                    f'diag_prev_box_blend_{alpha_key}_hit25'
                ] += int(iou_value > 0.25)
                self.dets[
                    f'diag_prev_box_blend_{alpha_key}_hit50'
                ] += int(iou_value > 0.50)

        detected_boxes = end_points.get('all_detected_boxes', None)
        detected_mask = end_points.get('all_detected_bbox_label_mask', None)
        if (
            report_box_blends
            and detected_boxes is not None
            and detected_mask is not None
        ):
            valid_det = detected_mask[bid].bool()
            if valid_det.any():
                det_boxes = detected_boxes[bid][valid_det].to(pred_bbox.device)
                last_selected = pred_bbox[eval_top]
                det_ious, _ = _iou3d_par(
                    box_cxcyczwhd_to_xyzxyz(det_boxes),
                    box_cxcyczwhd_to_xyzxyz(last_selected.unsqueeze(0)),
                )
                nearest_det = det_boxes[det_ious[:, 0].argmax()]
                self.dets['diag_det_box_blend_total'] += 1
                alphas = last_selected.new_tensor(
                    self.DET_BOX_BLEND_DIAGNOSTIC_ALPHAS
                ).unsqueeze(1)
                blended = (
                    alphas * last_selected.unsqueeze(0)
                    + (1.0 - alphas) * nearest_det.unsqueeze(0)
                )
                blended[:, 3:] = blended[:, 3:].clamp(min=1e-6)
                blend_iou, _ = _iou3d_par(
                    box_cxcyczwhd_to_xyzxyz(gt_bbox[:1]),
                    box_cxcyczwhd_to_xyzxyz(blended),
                )
                blend_ious = blend_iou[0].detach().cpu().tolist()
                for alpha, iou_value in zip(
                    self.DET_BOX_BLEND_DIAGNOSTIC_ALPHAS, blend_ious
                ):
                    alpha_key = str(alpha).replace('.', 'p')
                    self.dets[
                        f'diag_det_box_blend_{alpha_key}_hit25'
                    ] += int(iou_value > 0.25)
                    self.dets[
                        f'diag_det_box_blend_{alpha_key}_hit50'
                    ] += int(iou_value > 0.50)

        for threshold, suffix in ((0.25, '25'), (0.50, '50')):
            base_correct = bool(base_iou.item() > threshold)
            eval_correct = bool(eval_iou.item() > threshold)
            best_rankable = bool(best_iou.item() > threshold)
            top10_rankable = bool(oracle_top10_iou.item() > threshold)
            self.dets[f'diag_sem_fix{suffix}'] += int(
                eval_correct and not base_correct
            )
            self.dets[f'diag_sem_break{suffix}'] += int(
                base_correct and not eval_correct
            )
            if not eval_correct:
                self.dets[f'diag_sem_fail{suffix}_total'] += 1
                self.dets[f'diag_sem_fail{suffix}_best_rankable'] += int(
                    best_rankable
                )
                self.dets[f'diag_sem_fail{suffix}_top10_rankable'] += int(
                    top10_rankable
                )
                self.dets[f'diag_sem_fail{suffix}_unrankable'] += int(
                    not best_rankable
                )

        components = end_points.get('semantic_component_raw_scores', None)
        if components is None:
            return
        components = components.to(device=pred_bbox.device)
        if components.dim() != 3 or components.shape[-1] < 5:
            return
        comp_names = ('main', 'attr', 'pron', 'rel', 'other')
        eval_components = components[bid, eval_top, :5]
        best_components = components[bid, best_top, :5]
        for idx, name in enumerate(comp_names):
            self.dets[f'diag_comp_eval_top_{name}_sum'] += float(
                eval_components[idx].item()
            )
            self.dets[f'diag_comp_best_iou_{name}_sum'] += float(
                best_components[idx].item()
            )

    def _record_semantic_class_diagnostics(self, end_points, bid,
                                           eval_top, best_top, eval_iou,
                                           pred_bbox):
        detected_boxes = end_points.get('all_detected_boxes', None)
        detected_classes = end_points.get('all_detected_class_ids', None)
        detected_mask = end_points.get('all_detected_bbox_label_mask', None)
        target_cid = self._batch_item_as_int(end_points.get('target_cid'), bid)
        if (
            detected_boxes is None
            or detected_classes is None
            or detected_mask is None
            or target_cid is None
            or target_cid < 0
        ):
            return
        valid_det = detected_mask[bid].bool()
        if not valid_det.any():
            return
        eval_idx = int(eval_top.item())
        best_idx = int(best_top.item())
        selected_queries = pred_bbox[[eval_idx, best_idx]]
        det_boxes = detected_boxes[bid][valid_det].to(pred_bbox.device)
        det_classes = detected_classes[bid][valid_det].to(pred_bbox.device)
        det_ious, _ = _iou3d_par(
            box_cxcyczwhd_to_xyzxyz(det_boxes),
            box_cxcyczwhd_to_xyzxyz(selected_queries),
        )
        overlap, nearest_det = det_ious.max(dim=0)
        nearest_classes = det_classes[nearest_det].long()
        class_matches = (
            (nearest_classes == int(target_cid)) & (overlap > 0.25)
        )
        eval_match = bool(class_matches[0].item())
        best_match = bool(class_matches[1].item())

        self.dets['diag_cls_total'] += 1
        self.dets['diag_cls_eval_top_match'] += int(eval_match)
        self.dets['diag_cls_best_iou_match'] += int(best_match)
        eval_iou_value = float(eval_iou.item())
        for threshold, suffix in ((0.25, '25'), (0.50, '50')):
            if eval_iou_value > threshold:
                continue
            self.dets[f'diag_cls_fail{suffix}_total'] += 1
            if eval_match:
                self.dets[f'diag_cls_fail{suffix}_eval_match'] += 1
            else:
                self.dets[f'diag_cls_fail{suffix}_eval_mismatch'] += 1

    @staticmethod
    def _flag_enabled(value):
        if torch.is_tensor(value):
            if value.numel() == 0:
                return False
            return bool(value.detach().reshape(-1)[0].item())
        return bool(value)

    @staticmethod
    def _batch_item_as_int(value, bid):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() <= bid:
                return None
            return int(value.detach().reshape(-1)[bid].item())
        if isinstance(value, (list, tuple)):
            if len(value) <= bid:
                return None
            return int(value[bid])
        return int(value)

    def _spacy_augmentation_bucket(self, end_points, bid):
        mode_id = self._batch_item_as_int(
            end_points.get('spacy_rotation_mode_id', None), bid
        )
        if mode_id is None:
            return None
        mode_name = self.SPACY_AUGMENTATION_MODE_NAMES.get(mode_id)
        if mode_name is None:
            return None
        return f'spacy_aug_{mode_name}'

    def _spacy_augmentation_profile_bucket(self, end_points, bid):
        profile_id = self._batch_item_as_int(
            end_points.get('spacy_augmentation_profile_id', None), bid
        )
        if profile_id is None:
            return None
        profile_name = self.SPACY_AUGMENTATION_PROFILE_NAMES.get(profile_id)
        if profile_name is None:
            return None
        return f'spacy_profile_{profile_name}'

    @staticmethod
    def _crop_or_pad_scores(scores, num_queries):
        if scores.shape[-1] == num_queries:
            return scores
        if scores.shape[-1] > num_queries:
            return scores[..., :num_queries]
        pad_shape = list(scores.shape)
        pad_shape[-1] = num_queries - scores.shape[-1]
        return torch.cat([scores, scores.new_zeros(pad_shape)], dim=-1)

    @classmethod
    def _position_score_source(cls, end_points, prefix):
        if prefix != 'last_':
            return None
        candidates = (
            ('eval_use_fused_scores', 'fused_scores'),
            ('eval_use_quality_scores', 'pred_iou'),
            ('eval_use_structured_scores', 'structured_scores'),
        )
        for flag, key in candidates:
            if cls._flag_enabled(end_points.get(flag, False)):
                if key not in end_points:
                    raise ValueError(
                        f"Evaluation requested {key} but it is missing"
                    )
                return key
        return None

    @classmethod
    def _semantic_score_source(cls, end_points, prefix):
        if prefix != 'last_':
            return None
        if cls._flag_enabled(
            end_points.get('eval_use_semantic_support_scores', False)
        ):
            key = 'semantic_support_scores'
            if key not in end_points:
                raise ValueError(
                    "Evaluation requested semantic support scores but they "
                    "are missing"
                )
            return key
        if cls._flag_enabled(
            end_points.get('eval_use_fused_semantic_scores', False)
        ):
            key = 'fused_scores'
            if key not in end_points:
                raise ValueError(
                    "Evaluation requested semantic RAPF fused scores but "
                    "they are missing"
                )
            return key
        if cls._flag_enabled(
            end_points.get('eval_use_semantic_rerank_scores', False)
        ):
            key = 'semantic_rerank_scores'
            if key not in end_points:
                raise ValueError(
                    "Evaluation requested semantic rerank scores but they "
                    "are missing"
                )
            return key
        if cls._flag_enabled(
            end_points.get('eval_use_semantic_component_scores', False)
        ):
            key = 'semantic_component_scores'
            if key not in end_points:
                raise ValueError(
                    "Evaluation requested semantic component scores but they "
                    "are missing"
                )
            return key
        return None
    
    # BRIEF position alignment
    def evaluate_bbox_by_pos_align(self, end_points, prefix):
        """
        Evaluate bounding box IoU by position alignment

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # step get the position label and GT box 
        positive_map, modify_positive_map, pron_positive_map, other_entity_map, \
            auxi_entity_positive_map, rel_positive_map, gt_bboxes = self._parse_gt(end_points)    
        
        # Parse predictions
        sem_scores = end_points[f'{prefix}sem_cls_scores'].softmax(-1)

        if sem_scores.shape[-1] != positive_map.shape[-1]:
            sem_scores_ = torch.zeros(
                sem_scores.shape[0], sem_scores.shape[1],
                positive_map.shape[-1]).to(sem_scores.device)
            sem_scores_[:, :, :sem_scores.shape[-1]] = sem_scores
            sem_scores = sem_scores_

        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q=256, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1) # ([B, 256, 6])
        score_source_key = self._position_score_source(end_points, prefix)
        source_scores = None
        if score_source_key is not None:
            source_scores = end_points[score_source_key].to(
                device=pred_bbox.device,
                dtype=pred_bbox.dtype,
            )
            source_scores = self._crop_or_pad_scores(
                source_scores, pred_bbox.shape[1]
            )

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            is_correct = None
            if self.filter_non_gt_boxes:  # this works only for the target box
                ious, _ = _iou3d_par(
                    box_cxcyczwhd_to_xyzxyz(
                        end_points['all_detected_boxes'][bid][
                            end_points['all_detected_bbox_label_mask'][bid]
                        ]
                    ),  # (gt, 6)
                    box_cxcyczwhd_to_xyzxyz(pred_bbox[bid])  # (Q, 6)
                )  # (gt, Q)
                is_correct = (ious.max(0)[0] > 0.25) * 1.0
            
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores_main = (
                sem_scores[bid].unsqueeze(0)    
                * pmap.unsqueeze(1)             
            ).sum(-1)

            # score
            pmap_modi = modify_positive_map[bid, :1]
            pmap_pron = pron_positive_map[bid, :1]
            pmap_other = other_entity_map[bid, :1]
            pmap_rel = rel_positive_map[bid, :1]    # num_obj
            scores_modi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_modi.unsqueeze(1)             
            ).sum(-1)
            scores_pron = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_pron.unsqueeze(1)             
            ).sum(-1)
            scores_other = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_other.unsqueeze(1)             
            ).sum(-1)
            scores_rel = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_rel.unsqueeze(1)             
            ).sum(-1)

            scores = scores_main + scores_modi + scores_pron + scores_rel - scores_other
            if source_scores is not None:
                if scores.shape[0] != 1:
                    raise ValueError(
                        "Innovation score evaluation supports root-only "
                        "grounding rows"
                    )
                scores = source_scores[bid].unsqueeze(0)

            if is_correct is not None:
                scores = scores * is_correct[None]

            top = scores.argsort(1, True)[:, :10]
            pbox = pred_bbox[bid, top.reshape(-1)]

            ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]),  # (obj, 6)
                box_cxcyczwhd_to_xyzxyz(pbox)  # (obj*10, 6)
            )  # (obj, obj*10)
            ious = ious.reshape(top.size(0), top.size(0), top.size(1))
            ious = ious[torch.arange(len(ious)), torch.arange(len(ious))]   # ([1, 10])

            # step Measure IoU>threshold, ious are (obj, 10)
            topks = self.topks
            for t in self.thresholds:
                thresholded = ious > t
                for k in topks:
                    found = thresholded[:, :k].any(1)
                    self.dets[(prefix, t, k, 'bbs')] += found.sum().item()
                    self.gts[(prefix, t, k, 'bbs')] += len(thresholded)

    # BRIEF semantic alignment
    def evaluate_bbox_by_sem_align(self, end_points, prefix):
        """
        Evaluate bounding box IoU by semantic alignment.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # step get the position label and GT box 
        positive_map, modify_positive_map, pron_positive_map, other_entity_map, \
            auxi_entity_positive_map, rel_positive_map, gt_bboxes = self._parse_gt(end_points)    
        
        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)

        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        score_source_key = self._semantic_score_source(end_points, prefix)
        source_scores = None
        if score_source_key is not None:
            source_scores = end_points[score_source_key].to(
                device=pred_bbox.device,
                dtype=pred_bbox.dtype,
            )
            source_scores = self._crop_or_pad_scores(
                source_scores, pred_bbox.shape[1]
            )
        
        # step compute similarity between vision and text
        proj_tokens = end_points['proj_tokens']             # text feature   (B, 256, 64)
        proj_queries = end_points[f'{prefix}proj_queries']  # vision feature (B, 256, 64)
        sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))  # similarity ([B, 256, L]) 
        sem_scores_ = (sem_scores / 0.07).softmax(-1)                           # softmax ([B, 256, L])
        sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256) # ([B, 256, 256])
        sem_scores = sem_scores.to(sem_scores_.device)
        sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_ # ([B, P=256, L=256])

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            is_correct = None
            if self.filter_non_gt_boxes:  # this works only for the target box
                ious, _ = _iou3d_par(
                    box_cxcyczwhd_to_xyzxyz(
                        end_points['all_detected_boxes'][bid][
                            end_points['all_detected_bbox_label_mask'][bid]
                        ]
                    ),  # (gt, 6)
                    box_cxcyczwhd_to_xyzxyz(pred_bbox[bid])  # (Q, 6)
                )  # (gt, Q)
                is_correct = (ious.max(0)[0] > 0.25) * 1.0
            
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores_main = (
                sem_scores[bid].unsqueeze(0)  # (1, Q, 256)
                * pmap.unsqueeze(1)  # (obj, 1, 256)
            ).sum(-1)  # (obj, Q)
            
            # score
            pmap_modi = modify_positive_map[bid, :1]
            pmap_pron = pron_positive_map[bid, :1]
            pmap_other = other_entity_map[bid, :1]
            pmap_auxi = auxi_entity_positive_map[bid, :1]
            pmap_rel = rel_positive_map[bid, :1]
            scores_modi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_modi.unsqueeze(1)             
            ).sum(-1)
            scores_pron = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_pron.unsqueeze(1)             
            ).sum(-1)
            scores_other = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_other.unsqueeze(1)             
            ).sum(-1)
            scores_auxi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_auxi.unsqueeze(1)             
            ).sum(-1)
            scores_rel = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_rel.unsqueeze(1)             
            ).sum(-1)

            # total score
            base_scores = (
                scores_main + scores_modi + scores_pron + scores_rel
                - scores_other
            )
            scores = base_scores
            if source_scores is not None:
                if scores.shape[0] != 1:
                    raise ValueError(
                        "Semantic rerank score evaluation supports root-only "
                        "grounding rows"
                    )
                scores = source_scores[bid].unsqueeze(0)

            if is_correct is not None:
                base_scores = base_scores * is_correct[None]
                scores = scores * is_correct[None]

            self._record_semantic_diagnostics(
                end_points, prefix, bid, base_scores, scores,
                pred_bbox[bid], gt_bboxes[bid][:num_obj],
            )

            # 10 predictions per gt box
            top = scores.argsort(1, True)[:, :10]  # (obj, 10)
            pbox = pred_bbox[bid, top.reshape(-1)]

            # IoU
            ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]),  # (obj, 6)
                box_cxcyczwhd_to_xyzxyz(pbox)  # (obj*10, 6)
            )  # (obj, obj*10)
            ious = ious.reshape(top.size(0), top.size(0), top.size(1))
            ious = ious[torch.arange(len(ious)), torch.arange(len(ious))]

            # step Measure IoU>threshold, ious are (obj, 10)
            for t in self.thresholds:
                thresholded = ious > t
                for k in self.topks:
                    found = thresholded[:, :k].any(1)
                    self.dets[(prefix, t, k, 'bbf')] += found.sum().item()
                    self.gts[(prefix, t, k, 'bbf')] += len(thresholded)
                    if prefix == 'last_':
                        found = found[0].item()
                        if k == 1 and t == self.thresholds[0]:
                            if end_points['is_view_dep'][bid]:
                                self.gts['vd'] += 1
                                self.dets['vd'] += found
                            else:
                                self.gts['vid'] += 1
                                self.dets['vid'] += found
                            if end_points['is_hard'][bid]:
                                self.gts['hard'] += 1
                                self.dets['hard'] += found
                            else:
                                self.gts['easy'] += 1
                                self.dets['easy'] += found
                            if end_points['is_unique'][bid]:
                                self.gts['unique'] += 1
                                self.dets['unique'] += found
                            else:
                                self.gts['multi'] += 1
                                self.dets['multi'] += found
                            bucket = self._spacy_augmentation_bucket(
                                end_points, bid
                            )
                            if bucket is not None:
                                self.gts[bucket] += 1
                                self.dets[bucket] += found
                            profile_bucket = self._spacy_augmentation_profile_bucket(
                                end_points, bid
                            )
                            if profile_bucket is not None:
                                self.gts[profile_bucket] += 1
                                self.dets[profile_bucket] += found
                        if k == 1 and t == self.thresholds[1]:
                            if end_points['is_view_dep'][bid]:
                                self.gts['vd50'] += 1
                                self.dets['vd50'] += found
                            else:
                                self.gts['vid50'] += 1
                                self.dets['vid50'] += found
                            if end_points['is_hard'][bid]:
                                self.gts['hard50'] += 1
                                self.dets['hard50'] += found
                            else:
                                self.gts['easy50'] += 1
                                self.dets['easy50'] += found
                            if end_points['is_unique'][bid]:
                                self.gts['unique50'] += 1
                                self.dets['unique50'] += found
                            else:
                                self.gts['multi50'] += 1
                                self.dets['multi50'] += found
                            bucket = self._spacy_augmentation_bucket(
                                end_points, bid
                            )
                            if bucket is not None:
                                self.gts[f'{bucket}50'] += 1
                                self.dets[f'{bucket}50'] += found
                            profile_bucket = self._spacy_augmentation_profile_bucket(
                                end_points, bid
                            )
                            if profile_bucket is not None:
                                self.gts[f'{profile_bucket}50'] += 1
                                self.dets[f'{profile_bucket}50'] += found


    # BRIEF Get the postion label of the decoupled text component.
    def _parse_gt(self, end_points):
        positive_map = torch.clone(end_points['positive_map'])                  # main
        modify_positive_map = torch.clone(end_points['modify_positive_map'])    # attribute
        pron_positive_map = torch.clone(end_points['pron_positive_map'])        # pron
        other_entity_map = torch.clone(end_points['other_entity_map'])          # other(including auxi)
        auxi_entity_positive_map = torch.clone(end_points['auxi_entity_positive_map'])  # auxi
        rel_positive_map = torch.clone(end_points['rel_positive_map'])

        positive_map[positive_map > 0] = 1                      
        gt_center = end_points['center_label'][:, :, 0:3]       
        gt_size = end_points['size_gts']                        
        gt_bboxes = torch.cat([gt_center, gt_size], dim=-1)     # GT box cxcyczwhd
        
        if self.only_root:
            positive_map = positive_map[:, :1]  # (B, 1, 256)
            gt_bboxes = gt_bboxes[:, :1]        # (B, 1, 6)
        
        return positive_map, modify_positive_map, pron_positive_map, other_entity_map, auxi_entity_positive_map, \
            rel_positive_map, gt_bboxes
