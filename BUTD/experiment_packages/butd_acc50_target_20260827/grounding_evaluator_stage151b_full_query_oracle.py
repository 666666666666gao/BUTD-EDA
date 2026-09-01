# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""A class to collect and evaluate language grounding results."""

import gc
import os
import shutil
import torch

from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
from models.detector_policy_sources import DETECTOR_POLICY_SCORE_KEYS
import utils.misc as misc

import ipdb
st = ipdb.set_trace

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
        -1: "not_spacy",
        1: "none",
        2: "yaw_only",
        3: "full_natural",
    }
    SPACY_AUGMENTATION_PROFILE_NAMES = {
        -1: "not_spacy",
        1: "none",
        2: "yaw_relation",
        3: "full_natural_relation_free",
        4: "yaw_relation_free",
        5: "none_relation_free_view",
        6: "yaw_relation_free_stable",
        7: "small_yaw_relation_free_view",
        8: "rawview_relation_free_global_only",
        9: "none_relation_free",
        10: "none_relation_free_compass",
        11: "stable_direction_sensitive",
    }

    def __init__(self, only_root=True, thresholds=[0.25, 0.5],
                 topks=[1, 5, 10], prefixes=[],
                 source_choice_dump_path=None,
                 source_choice_dump_topk=1):
        """Initialize accumulators."""
        self.only_root = only_root
        self.thresholds = thresholds
        self.topks = topks
        self.prefixes = prefixes
        self.source_choice_dump_path = source_choice_dump_path
        self.source_choice_dump_topk = int(source_choice_dump_topk)
        self.source_choice_dump_flush_rows = 5000
        self.source_choice_dump_streaming = (
            source_choice_dump_path is not None
            and misc.get_world_size() == 1
        )

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
        self.bbs_subset_dets = {
            (subset, t): 0
            for subset in ('unique', 'multiple')
            for t in self.thresholds
        }
        self.bbs_subset_gts = dict(self.bbs_subset_dets)
        self.primary_score_source = 'base'
        self.bbs_score_source = 'base'
        self.bbf_score_source = 'contrastive_base'
        self.diagnostic_dets = {}
        self.diagnostic_gts = {}
        self.decomposition_dets = {}
        self.decomposition_gts = {}
        self.decomposition_status_counts = {}
        self.spacy_augmentation_dets = {}
        self.spacy_augmentation_gts = {}
        self.spacy_augmentation_counts = {}
        self.spacy_source_dets = {}
        self.spacy_source_gts = {}
        self.spacy_source_counts = {}
        self.per_layer_score_source = {}
        self.per_layer_has_fused_scores = {}
        self.score_alignment_sums = {}
        self.score_alignment_gts = {}
        self.selector_choice_source_names_for_logging = None
        self.source_choice_dump_source_names_for_logging = None
        self.source_choice_feature_rows = []
        self.source_choice_example_count = 0
        self.source_choice_dump_total_rows = 0
        self.source_choice_dump_shards = []
        self.source_choice_dump_shard_dir = None
        self.source_choice_dump_shard_dir_prepared = False
        self.detector_topk_compact_dump_path = os.environ.get(
            'NMV2_DETECTOR_TOPK_COMPACT_DUMP_PATH', ''
        ).strip()
        self.detector_topk_compact_rows = []
        if self.source_choice_dump_streaming:
            self.source_choice_dump_shard_dir = (
                self.source_choice_dump_path + '.parts'
            )

    def _next_source_choice_example_id(self):
        """Return a per-sample dump id that is stable across row counts."""
        example_id = misc.get_rank() * 1000000000 + self.source_choice_example_count
        self.source_choice_example_count += 1
        return example_id

    def _prepare_source_choice_dump_shard_dir(self):
        if not self.source_choice_dump_streaming:
            return
        if self.source_choice_dump_shard_dir_prepared:
            return
        if os.path.isdir(self.source_choice_dump_shard_dir):
            shutil.rmtree(self.source_choice_dump_shard_dir)
        os.makedirs(self.source_choice_dump_shard_dir, exist_ok=True)
        self.source_choice_dump_shard_dir_prepared = True

    def _flush_source_choice_feature_rows(self, force=False):
        if not self.source_choice_dump_streaming:
            return
        if not self.source_choice_feature_rows:
            return
        if (
            not force
            and len(self.source_choice_feature_rows)
            < self.source_choice_dump_flush_rows
        ):
            return

        self._prepare_source_choice_dump_shard_dir()
        shard_idx = len(self.source_choice_dump_shards)
        shard_name = 'rows_{:06d}.pt'.format(shard_idx)
        shard_path = os.path.join(self.source_choice_dump_shard_dir, shard_name)
        torch.save({'rows': self.source_choice_feature_rows}, shard_path)
        dump_dir = os.path.dirname(self.source_choice_dump_path) or '.'
        self.source_choice_dump_shards.append(
            os.path.relpath(shard_path, dump_dir)
        )
        self.source_choice_dump_total_rows += len(
            self.source_choice_feature_rows
        )
        self.source_choice_feature_rows = []
        gc.collect()

    def _source_choice_dump_source_names(self):
        source_names = self.source_choice_dump_source_names_for_logging
        if source_names is None:
            source_names = self.selector_choice_source_names_for_logging
        if source_names is None:
            return None
        return tuple(str(source) for source in source_names)

    def print_stats(self):
        """Print accumulated accuracies and return results as dict."""
        mode_str = {
            'bbs': 'Box given span (soft-token)',
            'bbf': 'Box given span (contrastive)'
        }
        results = {}
        for prefix in self.prefixes:
            for mode in ['bbs', 'bbf']:
                for t in self.thresholds:
                    line = '%s %s Acc%.2f: %s' % (
                        prefix, mode_str[mode], t,
                        ', '.join([
                            'Top-%d: %.3f' % (
                                k,
                                self.dets[(prefix, t, k, mode)]
                                / max(self.gts[(prefix, t, k, mode)], 1)
                            )
                            for k in self.topks
                        ])
                    )
                    print(line)
                    # Store results
                    for k in self.topks:
                        key = f'{prefix}_{mode}_acc{t:.2f}_top{k}'
                        results[key] = self.dets[(prefix, t, k, mode)] / max(self.gts[(prefix, t, k, mode)], 1)

        print(f'\nPrimary score source: {self.primary_score_source}')
        print(f'BBS score source: {self.bbs_score_source}')
        print(f'BBF score source: {self.bbf_score_source}')
        results['eval_primary_score_source'] = self.primary_score_source
        results['eval_bbs_score_source'] = self.bbs_score_source
        results['eval_bbf_score_source'] = self.bbf_score_source
        results['eval_primary_metric_family'] = 'bbs'
        results['eval_target_row_uses_primary_score'] = True
        results['eval_anchor_rows_use_baseline_score'] = True
        results['eval_bbf_is_diagnostic_only'] = True
        results['primary_score_source_id'] = {
            'base': 0.0, 'structured': 1.0, 'quality': 2.0,
            'fused': 3.0, 'acd': 4.0, 'selector': 5.0,
            'selector_pool': 6.0, 'selector_choice': 7.0,
            'selector_choice_hybrid': 8.0,
            'selector_choice_quality_override': 9.0,
            'detector_policy_adapter': 10.0,
        }.get(self.primary_score_source, -1.0)
        results['eval_primary_score_source_id'] = results['primary_score_source_id']
        results['eval_bbs_score_source_id'] = results['primary_score_source_id']
        results['eval_bbf_score_source_id'] = 0.0

        if self.per_layer_score_source or self.score_alignment_sums:
            print('\nPer-layer score alignment')
            for prefix in self.prefixes:
                source = self.per_layer_score_source.get(prefix, 'unknown')
                has_fused = float(self.per_layer_has_fused_scores.get(prefix, False))
                results[f'{prefix}per_layer_bbs_score_source'] = source
                results[f'{prefix}per_layer_bbs_score_source_id'] = self._score_source_id(source)
                results[f'{prefix}per_layer_has_fused_scores'] = has_fused
                print(
                    f'{prefix} bbs_source={source} '
                    f'has_fused_scores={has_fused:.0f}'
                )
                alignment_fields = [
                    'per_layer_top1_query_bbs',
                    'per_layer_top1_query_bbf',
                    'bbs_vs_bbf_top1_disagree_ratio',
                    'bbs_top1_iou',
                    'bbf_top1_iou',
                    'selector_choice_selected_source_id',
                    'selector_choice_oracle_source_id',
                    'selector_choice_oracle_agree',
                    'selector_choice_selected_iou',
                    'selector_choice_oracle_iou',
                    'selector_choice_iou_gap_to_oracle',
                    'selector_choice_target_override_ratio',
                    'selector_choice_selected_override_ratio',
                    'selector_choice_false_base_ratio',
                    'selector_choice_false_override_ratio',
                    'selector_choice_override_agreement_ratio',
                    'selector_choice_selected_base_ratio',
                    'selector_choice_selected_fused_ratio',
                    'selector_choice_selected_quality_ratio',
                    'selector_choice_selected_contrastive_base_ratio',
                    'selector_choice_oracle_base_ratio',
                    'selector_choice_oracle_fused_ratio',
                    'selector_choice_oracle_quality_ratio',
                    'selector_choice_oracle_contrastive_base_ratio',
                    'selector_choice_selected_hit025',
                    'selector_choice_selected_hit050',
                    'selector_choice_oracle_hit025',
                    'selector_choice_oracle_hit050',
                    'selector_pool_selected_query_id',
                    'selector_pool_oracle_query_id',
                    'selector_pool_oracle_agree',
                    'selector_pool_selected_iou',
                    'selector_pool_oracle_iou',
                    'selector_pool_iou_gap_to_oracle',
                    'selector_pool_score_gap_to_oracle',
                    'selector_pool_selected_hit025',
                    'selector_pool_selected_hit050',
                    'selector_pool_oracle_hit025',
                    'selector_pool_oracle_hit050',
                ]
                for source in self._selector_choice_logging_source_names():
                    alignment_fields.append(
                        f'selector_choice_selected_{source}_ratio'
                    )
                    alignment_fields.append(
                        f'selector_choice_oracle_{source}_ratio'
                    )
                    alignment_fields.append(
                        f'selector_choice_logit_{source}_mean'
                    )
                    alignment_fields.append(
                        f'selector_choice_logit_margin_{source}_mean'
                    )
                    for label in ('025', '050'):
                        alignment_fields.append(
                            f'selector_choice_{source}_hit{label}_ratio'
                        )
                        alignment_fields.append(
                            f'selector_choice_{source}_unique_hit{label}_ratio'
                        )
                for field in alignment_fields:
                    key = (prefix, field)
                    if key not in self.score_alignment_sums:
                        continue
                    value = (
                        self.score_alignment_sums[key]
                        / max(self.score_alignment_gts.get(key, 1), 1)
                    )
                    results[f'{prefix}{field}'] = value
                    print(f'{prefix} {field}: {value:.4f}')

        if self.diagnostic_dets:
            print('\nDiagnostic score sources')
            for key in sorted(self.diagnostic_dets.keys()):
                value = self.diagnostic_dets[key] / max(self.diagnostic_gts.get(key, 1), 1)
                print(key, value)
                source, prefix, t, k = key
                results[f'{prefix}_diag_{source}_acc{t:.2f}_top{k}'] = value
                if prefix == 'last_' and k == 1:
                    results[f'diag_{source}@{t:.2f}'] = value

        if self.decomposition_dets:
            print('\nDecomposition status')
            for key in sorted(self.decomposition_dets.keys()):
                value = self.decomposition_dets[key] / max(self.decomposition_gts.get(key, 1), 1)
                print(key, value)
                status, prefix, t, k = key
                safe_status = str(status).replace('/', '_').replace(' ', '_')
                results[f'{prefix}_decomp_{safe_status}_acc{t:.2f}_top{k}'] = value
            for (status, prefix), count in sorted(self.decomposition_status_counts.items()):
                safe_status = str(status).replace('/', '_').replace(' ', '_')
                results[f'{prefix}_decomp_{safe_status}_count'] = count
                if prefix == 'last_':
                    results[f'eval_decomp_{safe_status}_count'] = count

        if self.spacy_augmentation_dets:
            print('\nSpacy augmentation audit')
            for key in sorted(self.spacy_augmentation_dets.keys()):
                value = (
                    self.spacy_augmentation_dets[key]
                    / max(self.spacy_augmentation_gts.get(key, 1), 1)
                )
                print(key, value)
                bucket, prefix, t, k = key
                results[f'{prefix}_{bucket}_acc{t:.2f}_top{k}'] = value
            for (bucket, prefix), count in sorted(self.spacy_augmentation_counts.items()):
                print((bucket, prefix, 'count'), count)
                results[f'{prefix}_{bucket}_count'] = count
                if prefix == 'last_':
                    results[f'eval_{bucket}_count'] = count

        if self.spacy_source_dets:
            print('\nSpacy source score audit')
            for key in sorted(self.spacy_source_dets.keys()):
                value = (
                    self.spacy_source_dets[key]
                    / max(self.spacy_source_gts.get(key, 1), 1)
                )
                print(key, value)
                bucket, source, prefix, t, k = key
                safe_bucket = str(bucket).replace('/', '_').replace(' ', '_')
                safe_source = str(source).replace('/', '_').replace(' ', '_')
                results[
                    f'{prefix}_spacy_source_{safe_bucket}_{safe_source}'
                    f'_acc{t:.2f}_top{k}'
                ] = value
                if prefix == 'last_':
                    results[
                        f'eval_spacy_source_{safe_bucket}_{safe_source}'
                        f'_acc{t:.2f}_top{k}'
                    ] = value
            for (bucket, source, prefix), count in sorted(
                self.spacy_source_counts.items()
            ):
                print((bucket, source, prefix, 'count'), count)
                safe_bucket = str(bucket).replace('/', '_').replace(' ', '_')
                safe_source = str(source).replace('/', '_').replace(' ', '_')
                results[
                    f'{prefix}_spacy_source_{safe_bucket}_{safe_source}_count'
                ] = count
                if prefix == 'last_':
                    results[
                        f'eval_spacy_source_{safe_bucket}_{safe_source}_count'
                    ] = count

        print('\nAnalysis')
        for field in ['easy', 'hard', 'vd', 'vid', 'unique', 'multi']:
            value = self.dets[field] / self.gts[field]
            print(field, value)
            results[field] = value

        if self.bbs_subset_gts:
            print('\nOfficial BBS Unique/Multiple analysis')
            for subset in ('unique', 'multiple'):
                for t in self.thresholds:
                    key = (subset, t)
                    count = self.bbs_subset_gts[key]
                    value = (
                        self.bbs_subset_dets[key] / max(count, 1)
                    )
                    print(
                        '{} Acc{:.2f}: {:.6f} ({}/{})'.format(
                            subset, t, value,
                            self.bbs_subset_dets[key], count,
                        )
                    )
                    results[
                        'last__bbs_{}_acc{:.2f}_top1'.format(subset, t)
                    ] = value
                    results[
                        'last__bbs_{}_count_acc{:.2f}'.format(subset, t)
                    ] = count

        if self.source_choice_dump_path and misc.is_main_process():
            dump_dir = os.path.dirname(self.source_choice_dump_path)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
            if self.source_choice_dump_streaming:
                self._flush_source_choice_feature_rows(force=True)
                manifest = {
                    'format': 'source_choice_feature_dump_sharded_v1',
                    'row_count': self.source_choice_dump_total_rows,
                    'shards': self.source_choice_dump_shards,
                    'topk': self.source_choice_dump_topk,
                }
                source_names = self._source_choice_dump_source_names()
                if source_names is not None:
                    manifest['source_names'] = source_names
                    selector_source_names = (
                        self.selector_choice_source_names_for_logging
                    )
                    if selector_source_names is None:
                        selector_source_names = source_names
                    manifest['selector_choice_source_names'] = tuple(
                        str(source) for source in selector_source_names
                    )
                torch.save(manifest, self.source_choice_dump_path)
                print(
                    (
                        '\nSaved source-choice feature dump: {} rows '
                        'in {} shards -> {}'
                    ).format(
                        self.source_choice_dump_total_rows,
                        len(self.source_choice_dump_shards),
                        self.source_choice_dump_path,
                    )
                )
                results['source_choice_feature_rows'] = (
                    self.source_choice_dump_total_rows
                )
            else:
                torch.save(
                    {'rows': self.source_choice_feature_rows},
                    self.source_choice_dump_path,
                )
                print(
                    '\nSaved source-choice feature dump: {} rows -> {}'.format(
                        len(self.source_choice_feature_rows),
                        self.source_choice_dump_path,
                    )
                )
                results['source_choice_feature_rows'] = len(
                    self.source_choice_feature_rows
                )

        if self.detector_topk_compact_dump_path and misc.is_main_process():
            compact_dir = os.path.dirname(
                self.detector_topk_compact_dump_path
            )
            if compact_dir:
                os.makedirs(compact_dir, exist_ok=True)
            torch.save(
                {
                    'format': 'detector_topk_compact_v1',
                    'rows': self.detector_topk_compact_rows,
                },
                self.detector_topk_compact_dump_path,
            )
            results['detector_topk_compact_rows'] = len(
                self.detector_topk_compact_rows
            )
            print(
                '\nSaved detector top-K compact dump: {} rows -> {}'.format(
                    len(self.detector_topk_compact_rows),
                    self.detector_topk_compact_dump_path,
                )
            )

        return results

    def synchronize_between_processes(self):
        all_dets = misc.all_gather(self.dets)
        all_gts = misc.all_gather(self.gts)
        all_diag_dets = misc.all_gather(self.diagnostic_dets)
        all_diag_gts = misc.all_gather(self.diagnostic_gts)
        all_bbs_subset_dets = misc.all_gather(self.bbs_subset_dets)
        all_bbs_subset_gts = misc.all_gather(self.bbs_subset_gts)
        all_decomp_dets = misc.all_gather(self.decomposition_dets)
        all_decomp_gts = misc.all_gather(self.decomposition_gts)
        all_decomp_counts = misc.all_gather(self.decomposition_status_counts)
        all_spacy_aug_dets = misc.all_gather(self.spacy_augmentation_dets)
        all_spacy_aug_gts = misc.all_gather(self.spacy_augmentation_gts)
        all_spacy_aug_counts = misc.all_gather(self.spacy_augmentation_counts)
        all_spacy_source_dets = misc.all_gather(self.spacy_source_dets)
        all_spacy_source_gts = misc.all_gather(self.spacy_source_gts)
        all_spacy_source_counts = misc.all_gather(self.spacy_source_counts)
        all_score_sources = misc.all_gather(self.per_layer_score_source)
        all_has_fused = misc.all_gather(self.per_layer_has_fused_scores)
        all_alignment_sums = misc.all_gather(self.score_alignment_sums)
        all_alignment_gts = misc.all_gather(self.score_alignment_gts)
        if self.source_choice_dump_streaming:
            all_source_choice_rows = [self.source_choice_feature_rows]
        else:
            all_source_choice_rows = misc.all_gather(
                self.source_choice_feature_rows
            )

        if misc.is_main_process():
            merged_predictions = {}
            for key in set().union(*[p.keys() for p in all_dets]):
                merged_predictions[key] = 0
                for p in all_dets:
                    merged_predictions[key] += p.get(key, 0)
            self.dets = merged_predictions

            merged_predictions = {}
            for key in set().union(*[p.keys() for p in all_gts]):
                merged_predictions[key] = 0
                for p in all_gts:
                    merged_predictions[key] += p.get(key, 0)
            self.gts = merged_predictions

            merged_predictions = {}
            for key in set().union(*[p.keys() for p in all_bbs_subset_dets]):
                merged_predictions[key] = sum(
                    p.get(key, 0) for p in all_bbs_subset_dets
                )
            self.bbs_subset_dets = merged_predictions
            merged_predictions = {}
            for key in set().union(*[p.keys() for p in all_bbs_subset_gts]):
                merged_predictions[key] = sum(
                    p.get(key, 0) for p in all_bbs_subset_gts
                )
            self.bbs_subset_gts = merged_predictions

            self.diagnostic_dets = {}
            self.diagnostic_gts = {}
            diag_keys = set().union(*[p.keys() for p in all_diag_dets]) if all_diag_dets else []
            for key in diag_keys:
                self.diagnostic_dets[key] = sum(p.get(key, 0) for p in all_diag_dets)
                self.diagnostic_gts[key] = sum(p.get(key, 0) for p in all_diag_gts)

            self.decomposition_dets = {}
            self.decomposition_gts = {}
            decomp_keys = set().union(*[p.keys() for p in all_decomp_dets]) if all_decomp_dets else []
            for key in decomp_keys:
                self.decomposition_dets[key] = sum(p.get(key, 0) for p in all_decomp_dets)
                self.decomposition_gts[key] = sum(p.get(key, 0) for p in all_decomp_gts)
            self.decomposition_status_counts = {}
            count_keys = set().union(*[p.keys() for p in all_decomp_counts]) if all_decomp_counts else []
            for key in count_keys:
                self.decomposition_status_counts[key] = sum(
                    p.get(key, 0) for p in all_decomp_counts
                )
            self.spacy_augmentation_dets = {}
            self.spacy_augmentation_gts = {}
            spacy_aug_keys = (
                set().union(*[p.keys() for p in all_spacy_aug_dets])
                if all_spacy_aug_dets else []
            )
            for key in spacy_aug_keys:
                self.spacy_augmentation_dets[key] = sum(
                    p.get(key, 0) for p in all_spacy_aug_dets
                )
                self.spacy_augmentation_gts[key] = sum(
                    p.get(key, 0) for p in all_spacy_aug_gts
                )
            self.spacy_augmentation_counts = {}
            spacy_count_keys = (
                set().union(*[p.keys() for p in all_spacy_aug_counts])
                if all_spacy_aug_counts else []
            )
            for key in spacy_count_keys:
                self.spacy_augmentation_counts[key] = sum(
                    p.get(key, 0) for p in all_spacy_aug_counts
                )
            self.spacy_source_dets = {}
            self.spacy_source_gts = {}
            spacy_source_keys = (
                set().union(*[p.keys() for p in all_spacy_source_dets])
                if all_spacy_source_dets else []
            )
            for key in spacy_source_keys:
                self.spacy_source_dets[key] = sum(
                    p.get(key, 0) for p in all_spacy_source_dets
                )
                self.spacy_source_gts[key] = sum(
                    p.get(key, 0) for p in all_spacy_source_gts
                )
            self.spacy_source_counts = {}
            spacy_source_count_keys = (
                set().union(*[p.keys() for p in all_spacy_source_counts])
                if all_spacy_source_counts else []
            )
            for key in spacy_source_count_keys:
                self.spacy_source_counts[key] = sum(
                    p.get(key, 0) for p in all_spacy_source_counts
                )
            self.per_layer_score_source = {}
            for item in all_score_sources:
                self.per_layer_score_source.update(item)
            self.per_layer_has_fused_scores = {}
            for item in all_has_fused:
                self.per_layer_has_fused_scores.update(item)
            self.score_alignment_sums = {}
            sum_keys = set().union(*[p.keys() for p in all_alignment_sums]) if all_alignment_sums else []
            for key in sum_keys:
                self.score_alignment_sums[key] = sum(
                    p.get(key, 0.0) for p in all_alignment_sums
                )
            self.score_alignment_gts = {}
            gts_keys = set().union(*[p.keys() for p in all_alignment_gts]) if all_alignment_gts else []
            for key in gts_keys:
                self.score_alignment_gts[key] = sum(
                    p.get(key, 0) for p in all_alignment_gts
                )
            self.source_choice_feature_rows = [
                row for rows in all_source_choice_rows for row in rows
            ]

    def evaluate(self, end_points, prefix):
        """
        Evaluate all accuracies.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        self.evaluate_bbox_by_span(end_points, prefix)
        self.evaluate_bbox_by_contrast(end_points, prefix)
        self.evaluate_score_alignment(end_points, prefix)

    @staticmethod
    def _get_eval_targets(end_points, positive_map, gt_bboxes, bid):
        """Slice the GT tensors to the objects actually being evaluated."""
        num_obj = min(
            int(end_points['box_label_mask'][bid].sum()),
            positive_map.shape[1],
            gt_bboxes.shape[1]
        )
        pmap = positive_map[bid, :num_obj]
        gt_boxes = gt_bboxes[bid, :num_obj]
        return pmap, gt_boxes

    @staticmethod
    def _aligned_topk_ious(gt_boxes, pred_boxes, top):
        """Return per-object IoUs for that object's own top-k predictions."""
        num_obj, topk = top.shape
        pbox = pred_boxes[top.reshape(-1)]
        ious, _ = _iou3d_par(
            box_cxcyczwhd_to_xyzxyz(gt_boxes),
            box_cxcyczwhd_to_xyzxyz(pbox)
        )
        ious = ious.reshape(num_obj, num_obj, topk)
        ious = ious[torch.arange(num_obj), torch.arange(num_obj)]
        return ious

    @staticmethod
    def _decode_status_value(value):
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode('utf-8', errors='ignore')
        elif torch.is_tensor(value):
            if value.numel() != 1:
                return None
            value = value.detach().cpu().item()
        value = str(value)
        return value if value else None

    @classmethod
    def _status_from_container(cls, container, bid):
        if isinstance(container, str):
            return container
        if isinstance(container, (list, tuple)) and bid < len(container):
            return cls._decode_status_value(container[bid])
        if isinstance(container, dict):
            status = container.get('decomposition_status', None)
            if isinstance(status, (list, tuple)) and bid < len(status):
                return cls._decode_status_value(status[bid])
            if torch.is_tensor(status):
                if status.ndim > 0 and bid < status.shape[0]:
                    return cls._decode_status_value(status[bid])
                return cls._decode_status_value(status)
            return cls._decode_status_value(status)
        return None

    @staticmethod
    def _coverage_value(item, key, bid, default=0):
        if not isinstance(item, dict):
            return default
        value = item.get(key, default)
        if torch.is_tensor(value):
            if value.ndim > 0 and bid < value.shape[0]:
                value = value[bid]
            if value.numel() == 1:
                return value.detach().cpu().item()
            return default
        if isinstance(value, (list, tuple)):
            if bid < len(value):
                return value[bid]
            return default
        return value

    @classmethod
    def _decomposition_status(cls, end_points, bid):
        coverage = end_points.get('coverage_stats', None)
        item = None
        if isinstance(coverage, dict):
            item = coverage
        elif isinstance(coverage, (list, tuple)) and bid < len(coverage):
            item = coverage[bid]
        coverage_status = cls._status_from_container(item, bid)
        if coverage_status is not None:
            return coverage_status
        if isinstance(item, dict):
            has_target = bool(
                cls._coverage_value(item, 'has_target', bid, 1)
            )
            generic = bool(
                cls._coverage_value(
                    item, 'overgeneric_target_remaining', bid, 0
                )
                or cls._coverage_value(
                    item, 'target_overgeneric_canonical', bid, 0
                )
                or cls._coverage_value(
                    item, 'generic_target', bid, 0
                )
            )
            global_only = (
                (not has_target)
                or bool(cls._coverage_value(
                    item, 'global_only_due_to_parse_error', bid, 0
                ))
                or bool(cls._coverage_value(
                    item, 'missing_target', bid, 0
                ))
            )
            if global_only:
                return 'global_only_target_unresolved'
            if generic:
                return 'weak_generic_target_recovered'
            return 'ok'
        explicit = cls._status_from_container(
            end_points.get('decomposition_status', None), bid
        )
        if explicit is not None:
            return explicit
        return 'unknown'

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
    def _primary_source(end_points, prefix):
        if prefix != 'last_':
            return 'base'
        explicit_source = str(
            end_points.get('eval_primary_score_source', 'base')
        )
        if explicit_source and explicit_source != 'base':
            return explicit_source
        flags = [
            ('fused', bool(end_points.get('eval_use_fused_scores', False))),
            ('sacr_residual', bool(end_points.get('eval_use_sacr_residual_scores', False))),
            ('structured', bool(end_points.get('eval_use_structured_scores', False))),
            ('quality', bool(end_points.get('eval_use_quality_scores', False))),
            ('acd', bool(end_points.get('eval_use_acd_scores', False))),
            ('selector', bool(end_points.get('eval_use_selector_scores', False))),
            ('selector_pool', bool(end_points.get('eval_use_selector_pool_scores', False))),
            ('selector_choice', bool(end_points.get('eval_use_selector_choice_scores', False))),
            ('selector_choice_hybrid', bool(end_points.get('eval_use_selector_choice_hybrid_scores', False))),
            ('selector_choice_quality_override', bool(end_points.get('eval_use_selector_choice_quality_override_scores', False))),
            ('detector_policy_adapter', bool(end_points.get('eval_use_detector_policy_adapter_scores', False))),
        ]
        enabled = [name for name, flag in flags if flag]
        return enabled[0] if enabled else 'base'

    @staticmethod
    def _score_source_id(source):
        return {
            'base': 0.0,
            'contrastive_base': 0.0,
            'structured': 1.0,
            'sacr_residual': 1.5,
            'quality': 2.0,
            'fused': 3.0,
            'acd': 4.0,
            'selector': 5.0,
            'selector_pool': 6.0,
            'selector_choice': 7.0,
            'selector_choice_hybrid': 8.0,
            'selector_choice_quality_override': 9.0,
            'detector_policy_adapter': 10.0,
        }.get(source, -1.0)

    @staticmethod
    def _diagnostic_focus():
        return os.environ.get('NMV2_EVAL_DIAG_FOCUS', '').strip().lower()

    @staticmethod
    def _single_source_scores(end_points, source, bid, num_obj, base_scores=None):
        if base_scores is not None and source == 'base':
            return base_scores.clone()
        key_map = {
            'structured': 'structured_scores',
            'sacr_residual': 'sacr_residual_scores',
            'quality': 'pred_iou',
            'fused': 'fused_scores',
            'contrastive_base': 'bbf_base_grounding_scores',
            'acd': 'acd_final_scores',
            'selector': 'selector_scores',
            'detector_policy_adapter': 'detector_policy_adapter_scores',
        }
        key_map.update(DETECTOR_POLICY_SCORE_KEYS)
        key = key_map[source]
        if key not in end_points:
            raise ValueError(f"Evaluation requested {source} scores but {key} is missing")
        source_scores = end_points[key][bid]
        if base_scores is None:
            if num_obj != 1:
                raise ValueError(
                    "base_scores is required for multi-object evaluation score rows"
                )
            return source_scores.unsqueeze(0).expand(num_obj, -1)
        scores = base_scores.clone()
        if num_obj > 0:
            scores[0] = source_scores
        return scores

    @staticmethod
    def _candidate_source_scores(
        end_points, bid, num_obj, base_scores, extra_source_scores=None
    ):
        source_scores = {'base': base_scores}
        for source in (
            'fused',
            'quality',
            'contrastive_base',
            'acd',
            'detector_policy_adapter',
        ) + tuple(DETECTOR_POLICY_SCORE_KEYS.keys()):
            key_map = {
                'fused': 'fused_scores',
                'quality': 'pred_iou',
                'contrastive_base': 'bbf_base_grounding_scores',
                'acd': 'acd_final_scores',
                'detector_policy_adapter': (
                    'detector_policy_adapter_scores'
                ),
            }
            key_map.update(DETECTOR_POLICY_SCORE_KEYS)
            key = key_map[source]
            if key not in end_points:
                continue
            source_scores[source] = GroundingEvaluator._single_source_scores(
                end_points, source, bid, num_obj, base_scores=base_scores
            )
        if extra_source_scores:
            for source, scores in extra_source_scores.items():
                if scores is not None:
                    source_scores[source] = scores
        return source_scores

    @classmethod
    def _target_cid(cls, end_points, bid):
        source = str(end_points.get('eval_target_cid_source', 'gt')).strip().lower()
        key = 'text_target_cid' if source == 'text' else 'target_cid'
        value = cls._batch_value(end_points, key, bid, default=None)
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() < 1:
                return None
            value = value.reshape(-1)[0].detach().cpu().item()
        try:
            target_cid = int(value)
        except (TypeError, ValueError):
            return None
        return target_cid if target_cid >= 0 else None

    @classmethod
    def _target_semantic_scores(
        cls, end_points, prefix, bid, num_obj, base_scores, sem_scores
    ):
        target_cid = cls._target_cid(end_points, bid)
        if target_cid is None or sem_scores.dim() != 3:
            return None
        if bid >= sem_scores.shape[0] or target_cid >= sem_scores.shape[-1]:
            return None
        scores = base_scores.clone()
        if num_obj > 0:
            scores[0] = sem_scores[bid, :, target_cid].float()
        return scores

    @classmethod
    def _target_detector_class_overlap_scores(
        cls, end_points, bid, num_obj, base_scores, pred_bbox
    ):
        sources = cls._target_detector_overlap_score_sources(
            end_points, bid, num_obj, base_scores, pred_bbox, topks=()
        )
        return sources.get('target_detector_class_overlap', None)

    @classmethod
    def _target_detector_class_match_count(
        cls, end_points, bid, min_conf=None
    ):
        target_cid = cls._target_cid(end_points, bid)
        if target_cid is None:
            return None
        class_ids = end_points.get('all_detected_class_ids', None)
        logits = end_points.get('all_detected_logits', None)
        mask = end_points.get('all_detected_bbox_label_mask', None)
        if class_ids is None:
            class_ids = end_points.get('det_class_ids', None)
        if logits is None:
            logits = end_points.get('det_logits', None)
        if mask is None:
            mask = end_points.get('det_bbox_label_mask', None)
        if (
            not torch.is_tensor(class_ids)
            or not torch.is_tensor(mask)
            or class_ids.dim() < 2
            or mask.dim() < 2
            or bid >= class_ids.shape[0]
            or bid >= mask.shape[0]
            or class_ids.shape[1] != mask.shape[1]
        ):
            return None

        det_mask = mask[bid].bool()
        if not bool(det_mask.any().detach().cpu().item()):
            return 0
        det_classes = class_ids[bid].long()[det_mask]
        class_match = det_classes == int(target_cid)
        if min_conf is not None:
            if (
                not torch.is_tensor(logits)
                or logits.dim() < 3
                or bid >= logits.shape[0]
                or logits.shape[1] != mask.shape[1]
                or target_cid >= logits.shape[-1]
            ):
                return None
            det_logits = logits[bid].float()[det_mask]
            target_conf = det_logits.softmax(dim=-1)[:, int(target_cid)]
            class_match = class_match & (target_conf > float(min_conf))
        return int(class_match.sum().detach().cpu().item())

    @classmethod
    def _target_detector_overlap_score_sources(
        cls, end_points, bid, num_obj, base_scores, pred_bbox, topks=(2, 3, 5)
    ):
        target_cid = cls._target_cid(end_points, bid)
        if target_cid is None:
            return {}
        boxes = end_points.get('all_detected_boxes', None)
        class_ids = end_points.get('all_detected_class_ids', None)
        logits = end_points.get('all_detected_logits', None)
        mask = end_points.get('all_detected_bbox_label_mask', None)
        if boxes is None:
            boxes = end_points.get('det_boxes', None)
        if class_ids is None:
            class_ids = end_points.get('det_class_ids', None)
        if logits is None:
            logits = end_points.get('det_logits', None)
        if mask is None:
            mask = end_points.get('det_bbox_label_mask', None)
        if (
            not torch.is_tensor(boxes)
            or not torch.is_tensor(mask)
            or bid >= boxes.shape[0]
            or bid >= mask.shape[0]
        ):
            return {}

        det_mask = mask[bid].bool()
        if not bool(det_mask.any().detach().cpu().item()):
            return {}
        det_boxes = boxes[bid].float()[det_mask]
        pred_boxes = pred_bbox.float()
        pair_ious, _ = _iou3d_par(
            box_cxcyczwhd_to_xyzxyz(pred_boxes),
            box_cxcyczwhd_to_xyzxyz(det_boxes),
        )
        sources = {}

        def score_from_overlap(overlap_scores):
            scores = base_scores.clone()
            if num_obj > 0:
                scores[0] = overlap_scores.to(
                    device=base_scores.device, dtype=base_scores.dtype
                )
            return scores

        if (
            torch.is_tensor(class_ids)
            and bid < class_ids.shape[0]
            and class_ids.shape[1] == mask.shape[1]
        ):
            det_classes = class_ids[bid].long()[det_mask]
            class_match = det_classes == int(target_cid)
            if bool(class_match.any().detach().cpu().item()):
                sources['target_detector_class_overlap'] = (
                    score_from_overlap(
                        pair_ious[:, class_match].max(dim=1).values
                    )
                )

        if (
            torch.is_tensor(logits)
            and bid < logits.shape[0]
            and logits.shape[1] == mask.shape[1]
            and target_cid < logits.shape[-1]
        ):
            det_logits = logits[bid].float()[det_mask]
            target_conf = det_logits.softmax(dim=-1)[:, int(target_cid)]
            sources['target_detector_logit_overlap'] = (
                score_from_overlap(
                    (pair_ious * target_conf.unsqueeze(0)).max(dim=1).values
                )
            )
            topks = tuple(int(topk) for topk in topks if int(topk) > 0)
            if topks:
                max_topk = min(max(topks), int(det_logits.shape[-1]))
                det_topk = det_logits.topk(max_topk, dim=-1).indices
                for topk in topks:
                    k = min(topk, max_topk)
                    target_in_topk = (
                        det_topk[:, :k] == int(target_cid)
                    ).any(dim=-1)
                    if not bool(target_in_topk.any().detach().cpu().item()):
                        continue
                    sources[
                        f'target_detector_logit_top{topk}_overlap'
                    ] = score_from_overlap(
                        pair_ious[:, target_in_topk].max(dim=1).values
                    )
            if (
                'target_detector_class_overlap' in sources
                and torch.is_tensor(class_ids)
                and bid < class_ids.shape[0]
                and class_ids.shape[1] == mask.shape[1]
            ):
                det_classes = class_ids[bid].long()[det_mask]
                class_match = det_classes == int(target_cid)
                for min_conf, suffix in (
                    (0.20, 'gt0p2'),
                    (0.30, 'gt0p3'),
                    (0.40, 'gt0p4'),
                    (0.50, 'gt0p5'),
                ):
                    conf_match = class_match & (target_conf > min_conf)
                    if not bool(conf_match.any().detach().cpu().item()):
                        continue
                    sources[
                        f'target_detector_class_conf_{suffix}_overlap'
                    ] = score_from_overlap(
                        pair_ious[:, conf_match].max(dim=1).values
                    )
        return sources

    def _record_detector_topk_compact(
        self, end_points, bid, num_obj, base_scores, contrastive_scores,
        gt_boxes, pred_bbox, topk=5
    ):
        """Record a compact, diagnostic-only row for deployable rerank rules.

        Candidate selection features use only model/detector/text outputs.  IoU
        fields are labels for offline auditing and are never consumed by the
        inference rule.
        """
        if not self.detector_topk_compact_dump_path or num_obj <= 0:
            return
        if 'pred_iou' not in end_points:
            return

        quality_scores = self._single_source_scores(
            end_points, 'quality', bid, num_obj, base_scores=base_scores
        )[0].float()
        fused_scores = (
            self._single_source_scores(
                end_points, 'fused', bid, num_obj, base_scores=base_scores
            )[0].float()
            if 'fused_scores' in end_points
            else quality_scores
        )
        detector_sources = self._target_detector_overlap_score_sources(
            end_points, bid, num_obj, base_scores, pred_bbox,
            topks=(2, 3, 5),
        )
        class_scores = detector_sources.get(
            'target_detector_class_overlap', None
        )
        if class_scores is None:
            class_scores = torch.zeros_like(base_scores)

        k = max(1, min(int(topk), quality_scores.numel()))
        quality_top = quality_scores.argsort(descending=True)[:k]
        candidate_ious = self._aligned_topk_ious(
            gt_boxes[:1], pred_bbox, quality_top.view(1, -1)
        )[0]
        # Stage151b diagnostic-only full-query oracle.  The adapter candidate
        # mask contains only its deployed top-K pool, so max IoU inside that
        # mask is not the Oracle@t defined over every predicted query.
        all_query_count = max(
            0, min(int(num_obj), int(pred_bbox.shape[0]))
        )
        all_query_indices = torch.arange(
            all_query_count, device=pred_bbox.device, dtype=torch.long
        )
        all_query_ious = self._aligned_topk_ious(
            gt_boxes[:1], pred_bbox, all_query_indices.view(1, -1)
        )[0]

        def values_at(source_name):
            scores = detector_sources.get(source_name, None)
            if scores is None:
                return [0.0] * k
            return [
                self._as_float(v)
                for v in scores[0].float()[quality_top]
            ]

        def top_summary(scores):
            query = scores.argmax().long()
            iou = self._aligned_topk_ious(
                gt_boxes[:1], pred_bbox, query.view(1, 1)
            )[0, 0]
            return {
                'query': int(query.detach().cpu().item()),
                'iou': self._as_float(iou),
            }

        row = {
            'example_id': int(len(self.detector_topk_compact_rows)),
            'quality_topk_query': [
                int(v) for v in quality_top.detach().cpu().tolist()
            ],
            'quality_topk_score': [
                self._as_float(v) for v in quality_scores[quality_top]
            ],
            'quality_topk_iou': [
                self._as_float(v) for v in candidate_ious
            ],
            'base_at_quality_topk': [
                self._as_float(v) for v in base_scores[0].float()[quality_top]
            ],
            'fused_at_quality_topk': [
                self._as_float(v) for v in fused_scores[quality_top]
            ],
            'contrastive_at_quality_topk': [
                self._as_float(v)
                for v in contrastive_scores[0].float()[quality_top]
            ],
            'detector_class_at_quality_topk': values_at(
                'target_detector_class_overlap'
            ),
            'detector_logit_at_quality_topk': values_at(
                'target_detector_logit_overlap'
            ),
            'detector_logit_top2_at_quality_topk': values_at(
                'target_detector_logit_top2_overlap'
            ),
            'detector_logit_top3_at_quality_topk': values_at(
                'target_detector_logit_top3_overlap'
            ),
            'detector_conf20_at_quality_topk': values_at(
                'target_detector_class_conf_gt0p2_overlap'
            ),
            'detector_conf30_at_quality_topk': values_at(
                'target_detector_class_conf_gt0p3_overlap'
            ),
            'detector_conf40_at_quality_topk': values_at(
                'target_detector_class_conf_gt0p4_overlap'
            ),
            'detector_conf50_at_quality_topk': values_at(
                'target_detector_class_conf_gt0p5_overlap'
            ),
            'base_top': top_summary(base_scores[0].float()),
            'quality_top': top_summary(quality_scores),
            'fused_top': top_summary(fused_scores),
            'contrastive_top': top_summary(contrastive_scores[0].float()),
            'detector_class_count': self._target_detector_class_match_count(
                end_points, bid, min_conf=None
            ),
            'detector_conf20_count': self._target_detector_class_match_count(
                end_points, bid, min_conf=0.20
            ),
            'detector_conf30_count': self._target_detector_class_match_count(
                end_points, bid, min_conf=0.30
            ),
            'detector_conf40_count': self._target_detector_class_match_count(
                end_points, bid, min_conf=0.40
            ),
            'detector_conf50_count': self._target_detector_class_match_count(
                end_points, bid, min_conf=0.50
            ),
            'decomposition_status': self._decomposition_status(
                end_points, bid
            ),
            'spacy_augmentation_bucket': self._spacy_augmentation_bucket(
                end_points, bid
            ),
            'spacy_profile_bucket': self._spacy_augmentation_profile_bucket(
                end_points, bid
            ),
            'is_unique_label_only': bool(end_points['is_unique'][bid]),
            'all_query_count': int(all_query_count),
            'all_query_iou_max': (
                self._as_float(all_query_ious.max())
                if all_query_ious.numel() else 0.0
            ),
            'all_query_iou_ge_025_count': int(
                (all_query_ious >= 0.25).sum().detach().cpu().item()
            ),
            'all_query_iou_ge_050_count': int(
                (all_query_ious >= 0.50).sum().detach().cpu().item()
            ),
        }
        adapter_mask = end_points.get(
            'detector_policy_adapter_candidate_mask', None
        )
        adapter_delta = end_points.get(
            'detector_policy_adapter_rerank_delta', None
        )
        adapter_scores = end_points.get(
            'detector_policy_adapter_scores', None
        )
        if (
            adapter_mask is not None and adapter_delta is not None
            and adapter_scores is not None
        ):
            mask = adapter_mask[bid].bool()
            candidate = torch.nonzero(mask, as_tuple=False).flatten()
            candidate_iou = self._aligned_topk_ious(
                gt_boxes[:1], pred_bbox, candidate.view(1, -1)
            )[0]
            row.update({
                'adapter_candidate_query': [
                    int(v) for v in candidate.detach().cpu().tolist()
                ],
                'adapter_fused_at_candidate': [
                    self._as_float(v) for v in fused_scores[candidate]
                ],
                'adapter_delta_at_candidate': [
                    self._as_float(v)
                    for v in adapter_delta[bid][candidate]
                ],
                'adapter_score_at_candidate': [
                    self._as_float(v)
                    for v in adapter_scores[bid][candidate]
                ],
                'adapter_iou_at_candidate': [
                    self._as_float(v) for v in candidate_iou
                ],
                'adapter_box_at_candidate': [
                    [self._as_float(value) for value in box]
                    for box in pred_bbox[candidate]
                ],
                'gt_box': [
                    self._as_float(value) for value in gt_boxes[0]
                ],
            })
            for endpoint_key, row_key in (
                (
                    'detector_policy_adapter_hit25_logits',
                    'adapter_hit25_logit_at_candidate',
                ),
                (
                    'detector_policy_adapter_hit50_logits',
                    'adapter_hit50_logit_at_candidate',
                ),
                (
                    'detector_policy_adapter_rescue_logits',
                    'adapter_rescue_logit_at_candidate',
                ),
            ):
                values = end_points.get(endpoint_key, None)
                if torch.is_tensor(values):
                    row[row_key] = [
                        self._as_float(v) for v in values[bid][candidate]
                    ]
            for endpoint_key, row_key in (
                (
                    'detector_policy_adapter_rescue_gate',
                    'adapter_rescue_gate',
                ),
                (
                    'detector_policy_adapter_rescue_query',
                    'adapter_rescue_query',
                ),
                (
                    'detector_policy_adapter_fallback_query',
                    'adapter_fallback_query',
                ),
            ):
                value = end_points.get(endpoint_key, None)
                if torch.is_tensor(value):
                    scalar = value[bid].detach().cpu().item()
                    row[row_key] = (
                        bool(scalar) if row_key.endswith('_gate')
                        else int(scalar)
                    )
        detected_boxes = end_points.get('all_detected_boxes', None)
        detected_mask = end_points.get(
            'all_detected_bbox_label_mask', None
        )
        detected_classes = end_points.get('all_detected_class_ids', None)
        detected_logits = end_points.get('all_detected_logits', None)
        if detected_boxes is None:
            detected_boxes = end_points.get('det_boxes', None)
        if detected_mask is None:
            detected_mask = end_points.get('det_bbox_label_mask', None)
        if detected_classes is None:
            detected_classes = end_points.get('det_class_ids', None)
        if detected_logits is None:
            detected_logits = end_points.get('det_logits', None)
        if (
            torch.is_tensor(detected_boxes)
            and torch.is_tensor(detected_mask)
            and bid < detected_boxes.shape[0]
            and bid < detected_mask.shape[0]
        ):
            valid_detector = detected_mask[bid].bool()
            boxes = detected_boxes[bid].float()[valid_detector]
            row['detected_box'] = [
                [self._as_float(value) for value in box] for box in boxes
            ]
            if (
                torch.is_tensor(detected_classes)
                and bid < detected_classes.shape[0]
            ):
                row['detected_class_id'] = [
                    int(value) for value in
                    detected_classes[bid].long()[valid_detector].cpu().tolist()
                ]
            target_cid = self._target_cid(end_points, bid)
            row['text_target_cid'] = (
                int(target_cid) if target_cid is not None else None
            )
            if (
                target_cid is not None
                and torch.is_tensor(detected_logits)
                and bid < detected_logits.shape[0]
                and target_cid < detected_logits.shape[-1]
            ):
                confidence = detected_logits[bid].float()[
                    valid_detector
                ].softmax(dim=-1)[:, int(target_cid)]
                row['detected_target_confidence'] = [
                    self._as_float(value) for value in confidence
                ]
        self.detector_topk_compact_rows.append(row)

    @classmethod
    def _target_scene_class_overlap_scores(
        cls, end_points, bid, num_obj, base_scores, pred_bbox
    ):
        target_cid = cls._target_cid(end_points, bid)
        if target_cid is None:
            return None
        boxes = end_points.get('all_bboxes', None)
        class_ids = end_points.get('all_class_ids', None)
        mask = end_points.get('all_bbox_label_mask', None)
        if (
            not torch.is_tensor(boxes)
            or not torch.is_tensor(class_ids)
            or not torch.is_tensor(mask)
            or bid >= boxes.shape[0]
            or bid >= class_ids.shape[0]
            or bid >= mask.shape[0]
        ):
            return None

        scene_boxes = boxes[bid].float()
        scene_classes = class_ids[bid].long()
        scene_mask = mask[bid].bool() & (scene_classes == int(target_cid))
        if not bool(scene_mask.any().detach().cpu().item()):
            return None
        target_boxes = scene_boxes[scene_mask]
        pred_boxes = pred_bbox.float()
        pair_ious, _ = _iou3d_par(
            box_cxcyczwhd_to_xyzxyz(pred_boxes),
            box_cxcyczwhd_to_xyzxyz(target_boxes),
        )
        overlap_scores = pair_ious.max(dim=1).values.to(
            device=base_scores.device, dtype=base_scores.dtype
        )
        scores = base_scores.clone()
        if num_obj > 0:
            scores[0] = overlap_scores
        return scores

    @classmethod
    def _target_detector_logit_overlap_scores(
        cls, end_points, bid, num_obj, base_scores, pred_bbox
    ):
        sources = cls._target_detector_overlap_score_sources(
            end_points, bid, num_obj, base_scores, pred_bbox, topks=()
        )
        return sources.get('target_detector_logit_overlap', None)

    @classmethod
    def _target_detector_logit_topk_overlap_scores(
        cls, end_points, bid, num_obj, base_scores, pred_bbox, topk
    ):
        sources = cls._target_detector_overlap_score_sources(
            end_points, bid, num_obj, base_scores, pred_bbox, topks=(topk,)
        )
        return sources.get(
            f'target_detector_logit_top{int(topk)}_overlap', None
        )

    @staticmethod
    def _proposal_scene_overlap_scores_from_anchor_scores(
        source, base_scores, pred_bbox, anchor_scores,
        anchor_rank_scores=None, candidate_k=5, min_anchor_score=0.05
    ):
        if base_scores.numel() == 0 or anchor_scores is None:
            return None
        num_obj, num_queries = base_scores.shape
        if num_obj <= 0 or num_queries <= 0:
            return None
        rows = []
        pred_boxes = pred_bbox.float()
        for obj_idx in range(num_obj):
            valid = anchor_scores[obj_idx] > float(min_anchor_score)
            if not bool(valid.any().detach().cpu().item()):
                return None
            candidate_idx = torch.nonzero(valid, as_tuple=False).flatten()
            candidate_scores = anchor_scores[obj_idx, candidate_idx]
            k = max(1, min(int(candidate_k), int(candidate_idx.numel())))
            order = candidate_scores.argsort(0, True)[:k]
            selected_idx = candidate_idx[order]
            if anchor_rank_scores is not None:
                rank_values = anchor_rank_scores[obj_idx, selected_idx]
                rank_order = rank_values.argsort(0, True)
                selected_idx = selected_idx[rank_order]
            pseudo_boxes = pred_boxes[selected_idx]
            pair_ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(pred_boxes),
                box_cxcyczwhd_to_xyzxyz(pseudo_boxes),
            )
            rows.append(pair_ious.max(dim=1).values)
        overlap_scores = torch.stack(rows, dim=0).to(
            device=base_scores.device, dtype=base_scores.dtype
        )
        scores = base_scores.clone()
        scores[:num_obj] = overlap_scores
        return source, scores

    @staticmethod
    def _selector_choice_source_names(num_sources, end_points=None):
        source_names = None
        if end_points is not None:
            source_names = end_points.get('selector_choice_source_names', None)
        if source_names is None:
            source_names = (
                'base', 'fused', 'quality', 'contrastive_base', 'acd'
            )
        return tuple(str(source) for source in source_names)[:num_sources]

    def _selector_choice_logging_source_names(self):
        source_names = self.selector_choice_source_names_for_logging
        if source_names is None:
            return self._selector_choice_source_names(5)
        return tuple(str(source) for source in source_names)

    @staticmethod
    def _candidate_dump_source_names(extra_source_scores=None, end_points=None):
        configured_source_names = None
        if end_points is not None:
            configured_source_names = end_points.get(
                'selector_choice_source_names', None
            )
        if configured_source_names is not None:
            source_names = [
                str(source) for source in configured_source_names
            ]
        else:
            source_names = list(
                GroundingEvaluator._selector_choice_source_names(5)
            )
        if extra_source_scores is not None:
            for source in extra_source_scores:
                if source not in source_names:
                    source_names.append(source)
        return tuple(source_names)

    @staticmethod
    def _selector_choice_source_biases(end_points, num_sources, device, dtype):
        source_names = GroundingEvaluator._selector_choice_source_names(
            num_sources, end_points=end_points
        )
        biases = torch.zeros(num_sources, device=device, dtype=dtype)
        for source_idx, source in enumerate(source_names):
            key = f'eval_selector_choice_source_bias_{source}'
            biases[source_idx] = float(end_points.get(key, 0.0))
        return biases

    @staticmethod
    def _source_choice_selector_logit_features(
        end_points, bid, source_scores
    ):
        row = {}
        raw_scores = end_points.get('selector_choice_scores', None)
        if not torch.is_tensor(raw_scores) or raw_scores.dim() < 2:
            return row
        if bid >= raw_scores.shape[0]:
            return row

        raw_scores = raw_scores[bid].float()
        if raw_scores.dim() != 1:
            raw_scores = raw_scores.flatten()
        num_sources = raw_scores.shape[0]
        if num_sources <= 0:
            return row

        source_names = GroundingEvaluator._selector_choice_source_names(
            num_sources, end_points=end_points
        )
        raw_scores = raw_scores[:len(source_names)]
        biases = GroundingEvaluator._selector_choice_source_biases(
            end_points, len(source_names), raw_scores.device, raw_scores.dtype
        )
        biased_scores = raw_scores + biases
        available = []
        for source in source_names:
            scores = source_scores.get(source, None)
            available.append(
                scores is not None
                and torch.is_tensor(scores)
                and scores.dim() == 2
                and scores.shape[1] > 0
            )
        available_t = torch.tensor(
            available, device=raw_scores.device, dtype=torch.bool
        )
        masked_value = raw_scores.new_tensor(-1e30)
        applied_scores = biased_scores.masked_fill(~available_t, masked_value)
        if bool(available_t.any().detach().cpu().item()):
            selected_source = GroundingEvaluator._selector_choice_selected_source_index(
                end_points, bid, applied_scores
            )
        else:
            selected_source = -1
        row['selector_choice_selected_source_id'] = float(selected_source)
        override_logit = GroundingEvaluator._batch_value(
            end_points, 'selector_choice_override_logit', bid, default=None
        )
        if override_logit is not None:
            default_source = str(
                end_points.get(
                    'eval_selector_choice_override_default_source', 'base'
                )
            )
            default_idx = (
                source_names.index(default_source)
                if default_source in source_names
                else 0
            )
            row['selector_choice_override_logit'] = float(override_logit)
            row['selector_choice_override_threshold'] = float(
                end_points.get('eval_selector_choice_override_threshold', 0.0)
            )
            row['selector_choice_override_default_source_id'] = float(
                default_idx
            )

        for source_idx, source in enumerate(source_names):
            raw_logit = raw_scores[source_idx]
            bias = biases[source_idx]
            biased_logit = biased_scores[source_idx]
            applied_logit = applied_scores[source_idx]
            other_available = available_t.clone()
            other_available[source_idx] = False
            source_available = bool(available_t[source_idx].detach().cpu().item())
            if source_available and bool(other_available.any().detach().cpu().item()):
                other_raw = raw_scores.masked_fill(~other_available, masked_value)
                other_applied = applied_scores.masked_fill(
                    ~other_available, masked_value
                )
                raw_margin = raw_logit - other_raw.max()
                applied_margin = applied_logit - other_applied.max()
            else:
                raw_margin = raw_scores.new_tensor(0.0)
                applied_margin = raw_scores.new_tensor(0.0)

            row[f'selector_choice_logit_{source}'] = (
                GroundingEvaluator._as_float(raw_logit)
            )
            row[f'selector_choice_logit_bias_{source}'] = (
                GroundingEvaluator._as_float(bias)
            )
            row[f'selector_choice_biased_logit_{source}'] = (
                GroundingEvaluator._as_float(biased_logit)
            )
            row[f'selector_choice_applied_logit_{source}'] = (
                GroundingEvaluator._as_float(applied_logit)
            )
            row[f'selector_choice_logit_margin_{source}'] = (
                GroundingEvaluator._as_float(raw_margin)
            )
            row[f'selector_choice_applied_logit_margin_{source}'] = (
                GroundingEvaluator._as_float(applied_margin)
            )
        return row

    @staticmethod
    def _selector_choice_selected_source_index(
        end_points, bid, choice_logits, override_threshold=None
    ):
        use_override_head = bool(
            end_points.get('eval_selector_choice_use_override_head', False)
        )
        if override_threshold is not None:
            use_override_head = True
        else:
            override_threshold = float(
                end_points.get('eval_selector_choice_override_threshold', 0.0)
            )
        source_names = GroundingEvaluator._selector_choice_source_names(
            choice_logits.shape[0], end_points=end_points
        )
        default_source = str(
            end_points.get(
                'eval_selector_choice_override_default_source', 'base'
            )
        )
        default_idx = (
            source_names.index(default_source)
            if default_source in source_names
            else 0
        )
        if (
            use_override_head
            and 'selector_choice_override_logit' in end_points
            and choice_logits.shape[0] > 1
        ):
            override_logit = end_points['selector_choice_override_logit'][bid]
            if (
                float(override_logit.detach().cpu().item())
                <= float(override_threshold)
            ):
                return default_idx
            nonbase_logits = choice_logits.clone()
            nonbase_logits[default_idx] = torch.finfo(
                choice_logits.dtype
            ).min
            nonbase_value, nonbase_source = nonbase_logits.max(dim=0)
            if (
                float(nonbase_value.detach().cpu().item())
                <= torch.finfo(choice_logits.dtype).min / 2.0
            ):
                return 0
            return int(nonbase_source.detach().cpu().item())
        return int(choice_logits.argmax().detach().cpu().item())

    @staticmethod
    def _as_float(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    @staticmethod
    def _batch_value(end_points, key, bid, default=None):
        value = end_points.get(key, default)
        if torch.is_tensor(value):
            if value.ndim == 0:
                return value.detach().cpu().item()
            if bid >= value.shape[0]:
                return default
            item = value[bid]
            if torch.is_tensor(item) and item.numel() == 1:
                return item.detach().cpu().item()
            return item
        if isinstance(value, (list, tuple)):
            if bid >= len(value):
                return default
            item = value[bid]
            if (
                isinstance(item, (list, tuple))
                and len(item) == 1
                and not isinstance(item[0], dict)
            ):
                return item[0]
            return item
        return value

    @staticmethod
    def _value_as_float(value, default=0.0):
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            if value.numel() != 1:
                return float(default)
            value = value.detach().cpu().item()
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {'true', 'yes', 'y'}:
                return 1.0
            if lowered in {'false', 'no', 'n'}:
                return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @classmethod
    def _batch_float(cls, end_points, key, bid, default=0.0):
        return cls._value_as_float(
            cls._batch_value(end_points, key, bid, default), default
        )

    @classmethod
    def _coverage_float(cls, end_points, key, bid, default=0.0):
        coverage = end_points.get('coverage_stats', None)
        value = default
        if isinstance(coverage, dict):
            value = coverage.get(key, default)
        elif isinstance(coverage, (list, tuple)) and bid < len(coverage):
            item = coverage[bid]
            if isinstance(item, dict):
                value = item.get(key, default)
        return cls._value_as_float(value, default)

    @classmethod
    def _slot_float(cls, end_points, key, bid, default=0.0):
        slot_dict = end_points.get('slot_dict', None)
        if not isinstance(slot_dict, dict) or key not in slot_dict:
            return float(default)
        value = slot_dict[key]
        if torch.is_tensor(value):
            if value.ndim == 0:
                return cls._value_as_float(value, default)
            if bid < value.shape[0]:
                return cls._value_as_float(value[bid], default)
            return float(default)
        return cls._value_as_float(value, default)

    @classmethod
    def _span_count(cls, end_points, key, bid):
        value = cls._batch_value(end_points, key, bid, [])
        if value is None:
            return 0.0
        if torch.is_tensor(value):
            return float(value.numel())
        if isinstance(value, (list, tuple)):
            return float(len(value))
        if isinstance(value, dict):
            return 1.0
        return 0.0

    @classmethod
    def _slot_mask_count(cls, end_points, bid):
        value = cls._batch_value(end_points, 'slot_mask', bid, [])
        if value is None:
            return 0.0
        if torch.is_tensor(value):
            return float((value.float() > 0).sum().detach().cpu().item())
        if isinstance(value, (list, tuple)):
            count = 0
            for item in value:
                try:
                    count += 1 if float(item) > 0 else 0
                except (TypeError, ValueError):
                    count += 1 if item else 0
            return float(count)
        return 1.0 if bool(value) else 0.0

    @classmethod
    def _source_choice_context_features(cls, end_points, bid):
        explicit_status = cls._status_from_container(
            end_points.get('decomposition_status', None), bid
        )
        status = str(
            explicit_status or cls._decomposition_status(end_points, bid)
            or 'unknown'
        )
        parse_confidence = cls._slot_float(
            end_points, 'parse_confidence', bid, default=-1.0
        )
        if parse_confidence < 0.0:
            parse_confidence = cls._coverage_float(
                end_points, 'parse_confidence', bid, default=-1.0
            )
        if parse_confidence < 0.0:
            parse_confidence = cls._batch_float(
                end_points, 'parse_confidence', bid, default=1.0
            )

        row = {
            'context_parse_confidence': parse_confidence,
            'context_has_target': cls._coverage_float(
                end_points, 'has_target', bid, default=1.0
            ),
            'context_num_attrs': cls._coverage_float(
                end_points, 'num_attrs', bid, default=0.0
            ),
            'context_num_pairs': cls._coverage_float(
                end_points, 'num_pairs', bid, default=0.0
            ),
            'context_num_parse_errors': cls._coverage_float(
                end_points, 'num_parse_errors', bid, default=0.0
            ),
            'context_decomposition_error_flags_count': cls._batch_float(
                end_points, 'decomposition_error_flags_count', bid,
                default=0.0,
            ),
            'context_status_ok': 1.0 if status == 'ok' else 0.0,
            'context_status_repaired': (
                1.0 if status.startswith('repaired') else 0.0
            ),
            'context_status_weak_generic': (
                1.0 if status == 'weak_generic_target_recovered' else 0.0
            ),
            'context_status_global_only': (
                1.0 if status == 'global_only_target_unresolved' else 0.0
            ),
            'context_status_unknown': 1.0 if status == 'unknown' else 0.0,
            'context_global_only_mask': cls._batch_float(
                end_points, 'decomp_global_only_mask', bid, default=0.0
            ),
            'context_weak_generic_mask': cls._batch_float(
                end_points, 'decomp_weak_generic_mask', bid, default=0.0
            ),
            'context_global_only_due_to_parse_error': cls._batch_float(
                end_points, 'global_only_due_to_parse_error', bid,
                default=cls._coverage_float(
                    end_points, 'global_only_due_to_parse_error', bid,
                    default=0.0,
                ),
            ),
            'context_target_generic_reference': cls._batch_float(
                end_points, 'target_generic_reference', bid,
                default=cls._coverage_float(
                    end_points, 'target_generic_reference', bid,
                    default=0.0,
                ),
            ),
            'context_entity_span_count': cls._span_count(
                end_points, 'entity_spans', bid
            ),
            'context_attr_span_count': cls._span_count(
                end_points, 'attr_spans', bid
            ),
            'context_rel_span_count': cls._span_count(
                end_points, 'rel_spans', bid
            ),
            'context_anchor_slot_count': cls._span_count(
                end_points, 'anchor_slots', bid
            ),
            'context_slot_mask_count': cls._slot_mask_count(end_points, bid),
            'context_is_view_dep': cls._batch_float(
                end_points, 'is_view_dep', bid, default=0.0
            ),
            'context_metadata_conflict_ratio': cls._batch_float(
                end_points, 'metadata_conflict_ratio', bid, default=0.0
            ),
        }
        positive_map_fields = {
            'target_missing_ratio': 'positive_map_target_missing_ratio',
            'fallback_used_ratio': 'positive_map_fallback_used_ratio',
            'global_only_target_empty_ratio': (
                'positive_map_global_only_target_empty_ratio'
            ),
            'explicit_target_slot_ratio': (
                'positive_map_source_explicit_target_slot_ratio'
            ),
            'entity_exact_match_ratio': (
                'positive_map_source_entity_exact_match_ratio'
            ),
            'lexical_exact_match_ratio': (
                'positive_map_source_lexical_exact_match_ratio'
            ),
            'missing_ratio': 'positive_map_source_missing_ratio',
        }
        for suffix, source_key in positive_map_fields.items():
            row[f'context_positive_map_{suffix}'] = cls._batch_float(
                end_points, source_key, bid, default=0.0
            )
        row.update(cls._source_choice_spacy_augmentation_features(end_points, bid))
        return row

    @classmethod
    def _source_choice_spacy_augmentation_features(cls, end_points, bid):
        mode_id = cls._batch_item_as_int(
            end_points.get('spacy_rotation_mode_id', None), bid
        )
        profile_id = cls._batch_item_as_int(
            end_points.get('spacy_augmentation_profile_id', None), bid
        )
        if mode_id is None:
            mode_id = -1
        if profile_id is None:
            profile_id = -1

        row = {
            'spacy_rotation_mode_id': float(mode_id),
            'spacy_augmentation_profile_id': float(profile_id),
        }
        for known_id, name in cls.SPACY_AUGMENTATION_MODE_NAMES.items():
            row[f'spacy_aug_{name}'] = 1.0 if mode_id == known_id else 0.0
        for known_id, name in cls.SPACY_AUGMENTATION_PROFILE_NAMES.items():
            row[f'spacy_profile_{name}'] = (
                1.0 if profile_id == known_id else 0.0
            )
        return row

    @staticmethod
    def _source_choice_intrinsic_inputs(
        end_points, bid, num_queries, prefix='last_'
    ):
        sem_key = f'{prefix}sem_cls_scores'
        sem_probs = None
        if sem_key in end_points and end_points[sem_key].dim() == 3:
            sem_probs = end_points[sem_key][bid].float().softmax(-1)

        objectness_key = f'{prefix}objectness_scores'
        objectness_scores = None
        if (
            objectness_key in end_points
            and end_points[objectness_key].dim() >= 2
        ):
            objectness_scores = end_points[objectness_key][bid].float()

        seed_objectness_scores = None
        if (
            'seeds_obj_cls_logits' in end_points
            and 'query_points_sample_inds' in end_points
        ):
            seed_logits = end_points['seeds_obj_cls_logits']
            sample_inds = end_points['query_points_sample_inds']
            if (
                torch.is_tensor(seed_logits)
                and torch.is_tensor(sample_inds)
                and bid < seed_logits.shape[0]
                and bid < sample_inds.shape[0]
            ):
                seed_row = seed_logits[bid].float()
                if seed_row.dim() > 1:
                    seed_row = seed_row.squeeze(0)
                query_seed_inds = sample_inds[bid].long()
                valid_seed = (
                    query_seed_inds >= 0
                ) & (query_seed_inds < seed_row.shape[0])
                if query_seed_inds.numel() >= num_queries and bool(
                    valid_seed[:num_queries].all().detach().cpu().item()
                ):
                    seed_objectness_scores = seed_row[
                        query_seed_inds[:num_queries]
                    ].float()

        return sem_probs, objectness_scores, seed_objectness_scores

    @staticmethod
    def _source_choice_query_intrinsic_features(
        end_points, bid, query_idx, pred_bbox, num_queries,
        field_prefix='candidate', prefix='last_', intrinsic_inputs=None
    ):
        row = {
            f'{field_prefix}_center_x': 0.0,
            f'{field_prefix}_center_y': 0.0,
            f'{field_prefix}_center_z': 0.0,
            f'{field_prefix}_size_x': 0.0,
            f'{field_prefix}_size_y': 0.0,
            f'{field_prefix}_size_z': 0.0,
            f'{field_prefix}_volume': 0.0,
            f'{field_prefix}_sem_cls_available': 0.0,
            f'{field_prefix}_sem_cls_max': 0.0,
            f'{field_prefix}_sem_cls_margin': 0.0,
            f'{field_prefix}_sem_cls_entropy': 0.0,
            f'{field_prefix}_objectness_available': 0.0,
            f'{field_prefix}_objectness_score': 0.0,
            f'{field_prefix}_objectness_prob': 0.0,
            f'{field_prefix}_seed_objectness_available': 0.0,
            f'{field_prefix}_seed_objectness_score': 0.0,
            f'{field_prefix}_seed_objectness_prob': 0.0,
            f'{field_prefix}_seed_objectness_rank': 0.0,
            f'{field_prefix}_seed_objectness_delta_to_top': 0.0,
        }
        if query_idx is None:
            return row
        if torch.is_tensor(query_idx):
            query_idx = query_idx.detach().cpu().flatten()
            if query_idx.numel() == 0:
                return row
            query_idx = int(query_idx[0].item())
        else:
            query_idx = int(query_idx)
        if query_idx < 0 or query_idx >= pred_bbox.shape[0]:
            return row

        candidate_box = pred_bbox[query_idx].float()
        candidate_center = candidate_box[:3]
        candidate_size = candidate_box[3:6]
        candidate_volume = candidate_size.clamp(min=0.0).prod()
        row[f'{field_prefix}_center_x'] = GroundingEvaluator._as_float(
            candidate_center[0]
        )
        row[f'{field_prefix}_center_y'] = GroundingEvaluator._as_float(
            candidate_center[1]
        )
        row[f'{field_prefix}_center_z'] = GroundingEvaluator._as_float(
            candidate_center[2]
        )
        row[f'{field_prefix}_size_x'] = GroundingEvaluator._as_float(
            candidate_size[0]
        )
        row[f'{field_prefix}_size_y'] = GroundingEvaluator._as_float(
            candidate_size[1]
        )
        row[f'{field_prefix}_size_z'] = GroundingEvaluator._as_float(
            candidate_size[2]
        )
        row[f'{field_prefix}_volume'] = GroundingEvaluator._as_float(
            candidate_volume
        )

        if intrinsic_inputs is None:
            intrinsic_inputs = GroundingEvaluator._source_choice_intrinsic_inputs(
                end_points, bid, num_queries, prefix=prefix
            )
        sem_probs, objectness_scores, seed_objectness_scores = intrinsic_inputs

        if sem_probs is not None and query_idx < sem_probs.shape[0]:
            candidate_sem = sem_probs[query_idx]
            top_sem = candidate_sem.topk(min(2, candidate_sem.numel()))
            sem_max = top_sem.values[0]
            sem_margin = (
                top_sem.values[0] - top_sem.values[1]
                if top_sem.values.numel() > 1
                else top_sem.values[0]
            )
            sem_entropy = -(
                candidate_sem * candidate_sem.clamp(min=1e-12).log()
            ).sum()
            row[f'{field_prefix}_sem_cls_available'] = 1.0
            row[f'{field_prefix}_sem_cls_max'] = GroundingEvaluator._as_float(
                sem_max
            )
            row[f'{field_prefix}_sem_cls_margin'] = (
                GroundingEvaluator._as_float(sem_margin)
            )
            row[f'{field_prefix}_sem_cls_entropy'] = (
                GroundingEvaluator._as_float(sem_entropy)
            )

        if (
            objectness_scores is not None
            and query_idx < objectness_scores.shape[0]
        ):
            objectness_score = objectness_scores[query_idx]
            row[f'{field_prefix}_objectness_available'] = 1.0
            row[f'{field_prefix}_objectness_score'] = (
                GroundingEvaluator._as_float(objectness_score)
            )
            row[f'{field_prefix}_objectness_prob'] = (
                GroundingEvaluator._as_float(objectness_score.sigmoid())
            )

        if (
            seed_objectness_scores is not None
            and query_idx < seed_objectness_scores.shape[0]
        ):
            seed_objectness_score = seed_objectness_scores[query_idx]
            seed_top_score = seed_objectness_scores.max()
            seed_rank = (
                int(
                    (
                        seed_objectness_scores
                        > seed_objectness_score
                    ).sum().item()
                )
                + 1
            )
            row[f'{field_prefix}_seed_objectness_available'] = 1.0
            row[f'{field_prefix}_seed_objectness_score'] = (
                GroundingEvaluator._as_float(seed_objectness_score)
            )
            row[f'{field_prefix}_seed_objectness_prob'] = (
                GroundingEvaluator._as_float(seed_objectness_score.sigmoid())
            )
            row[f'{field_prefix}_seed_objectness_rank'] = float(seed_rank)
            row[f'{field_prefix}_seed_objectness_delta_to_top'] = (
                GroundingEvaluator._as_float(
                    seed_top_score - seed_objectness_score
                )
            )

        return row

    @staticmethod
    def _decoder_layer_prefixes(end_points):
        prefixes = []
        for key in end_points.keys():
            if not key.endswith('sem_cls_scores'):
                continue
            prefix = key[:-len('sem_cls_scores')]
            if prefix == 'proposal_':
                continue
            if prefix == 'last_' or prefix.endswith('head_'):
                prefixes.append(prefix)
        return sorted(
            set(prefixes),
            key=GroundingEvaluator._layer_prefix_sort_key,
        )

    @staticmethod
    def _layer_prefix_sort_key(prefix):
        if prefix == 'last_':
            return (1, 1_000_000)
        if prefix.endswith('head_'):
            head_idx = prefix[:-len('head_')]
            if head_idx.isdigit():
                return (0, int(head_idx))
        return (0, 1_000_001)

    @staticmethod
    def _source_choice_layer_stability_features(
        end_points, bid, num_queries, target_map
    ):
        device = target_map.device if torch.is_tensor(target_map) else None
        if target_map is None or num_queries <= 0:
            zero_query = torch.zeros(
                num_queries, device=device, dtype=torch.float32
            )
            row = {
                'layer_stability_bbs_num_layers': 0.0,
                'layer_stability_bbs_top1_change_count': 0.0,
                'layer_stability_bbs_top1_stable_tail_ratio': 0.0,
                'layer_stability_bbf_num_layers': 0.0,
                'layer_stability_bbf_top1_change_count': 0.0,
                'layer_stability_bbf_top1_stable_tail_ratio': 0.0,
                'layer_stability_bbs_bbf_top1_agreement_count': 0.0,
                'layer_stability_bbs_bbf_top1_agreement_ratio': 0.0,
                'layer_stability_bbs_bbf_top1_disagreement_count': 0.0,
                'layer_stability_bbs_bbf_top1_disagreement_ratio': 0.0,
            }
            query_features = {
                'layer_stability_bbs_score_mean': zero_query,
                'layer_stability_bbs_score_std': zero_query,
                'layer_stability_bbs_score_min': zero_query,
                'layer_stability_bbs_score_max': zero_query,
                'layer_stability_bbs_score_last_minus_first': zero_query,
                'layer_stability_bbs_top1_count': zero_query,
                'layer_stability_bbs_top1_ratio': zero_query,
                'layer_stability_bbf_score_mean': zero_query,
                'layer_stability_bbf_score_std': zero_query,
                'layer_stability_bbf_score_min': zero_query,
                'layer_stability_bbf_score_max': zero_query,
                'layer_stability_bbf_score_last_minus_first': zero_query,
                'layer_stability_bbf_top1_count': zero_query,
                'layer_stability_bbf_top1_ratio': zero_query,
            }
            return row, query_features

        if torch.is_tensor(target_map):
            target_map = target_map.detach().float()
            if target_map.dim() > 1:
                target_map = target_map.view(-1, target_map.shape[-1])[0]
        else:
            target_map = torch.as_tensor(
                target_map, dtype=torch.float32, device=device
            )
        device = target_map.device
        target_map = target_map.to(device=device, dtype=torch.float32)

        def _align_token_scores(scores):
            scores = scores.float()
            if scores.shape[-1] == target_map.shape[-1]:
                return scores
            aligned = scores.new_zeros(
                scores.shape[:-1] + (target_map.shape[-1],)
            )
            copy_dim = min(scores.shape[-1], target_map.shape[-1])
            aligned[..., :copy_dim] = scores[..., :copy_dim]
            return aligned

        layer_prefixes = GroundingEvaluator._decoder_layer_prefixes(end_points)
        layer_scores = {'bbs': [], 'bbf': []}
        layer_top_queries = {'bbs': [], 'bbf': []}

        if 'proj_tokens' not in end_points or bid >= end_points['proj_tokens'].shape[0]:
            proj_tokens = None
        else:
            proj_tokens = end_points['proj_tokens'][bid].float()

        for prefix in layer_prefixes:
            sem_key = f'{prefix}sem_cls_scores'
            if sem_key in end_points and bid < end_points[sem_key].shape[0]:
                sem_probs = end_points[sem_key][bid].float().softmax(-1)
                sem_probs = _align_token_scores(sem_probs)
                bbs_scores = (sem_probs * target_map.unsqueeze(0)).sum(-1)
                layer_scores['bbs'].append((prefix, bbs_scores))
                layer_top_queries['bbs'].append(
                    (prefix, bbs_scores.argmax(dim=0))
                )

            query_key = f'{prefix}proj_queries'
            if (
                proj_tokens is not None
                and query_key in end_points
                and bid < end_points[query_key].shape[0]
            ):
                proj_queries = end_points[query_key][bid].float()
                token_scores = torch.matmul(
                    proj_queries, proj_tokens.transpose(-1, -2)
                )
                token_scores = (token_scores / 0.07).softmax(-1)
                token_scores = _align_token_scores(token_scores)
                bbf_scores = (token_scores * target_map.unsqueeze(0)).sum(-1)
                layer_scores['bbf'].append((prefix, bbf_scores))
                layer_top_queries['bbf'].append(
                    (prefix, bbf_scores.argmax(dim=0))
                )

        row = {}
        query_features = {}
        zero_query = torch.zeros(
            num_queries, device=device, dtype=torch.float32
        )

        def _add_source_stats(label, source_layers, source_tops):
            stacked = (
                torch.stack([scores for _, scores in source_layers], dim=0)
                if source_layers else None
            )
            top_query_stack = (
                torch.stack([top for _, top in source_tops], dim=0)
                if source_tops else None
            )
            row[f'layer_stability_{label}_num_layers'] = float(
                stacked.shape[0] if stacked is not None else 0.0
            )
            if top_query_stack is not None and top_query_stack.shape[0] > 1:
                top1_change_count = (
                    top_query_stack[1:] != top_query_stack[:-1]
                ).float().sum()
            else:
                top1_change_count = torch.tensor(0.0, device=device)
            row[f'layer_stability_{label}_top1_change_count'] = (
                GroundingEvaluator._as_float(top1_change_count)
            )
            if top_query_stack is not None and top_query_stack.shape[0] > 0:
                stable_tail_ratio = (
                    top_query_stack == top_query_stack[-1]
                ).float().mean()
            else:
                stable_tail_ratio = torch.tensor(0.0, device=device)
            row[f'layer_stability_{label}_top1_stable_tail_ratio'] = (
                GroundingEvaluator._as_float(stable_tail_ratio)
            )
            if stacked is None:
                query_features[f'layer_stability_{label}_score_mean'] = zero_query
                query_features[f'layer_stability_{label}_score_std'] = zero_query
                query_features[f'layer_stability_{label}_score_min'] = zero_query
                query_features[f'layer_stability_{label}_score_max'] = zero_query
                query_features[
                    f'layer_stability_{label}_score_last_minus_first'
                ] = zero_query
                query_features[f'layer_stability_{label}_top1_count'] = zero_query
                query_features[f'layer_stability_{label}_top1_ratio'] = zero_query
                return

            query_features[f'layer_stability_{label}_score_mean'] = (
                stacked.mean(dim=0)
            )
            query_features[f'layer_stability_{label}_score_std'] = (
                stacked.std(dim=0, unbiased=False)
            )
            query_features[f'layer_stability_{label}_score_min'] = (
                stacked.min(dim=0).values
            )
            query_features[f'layer_stability_{label}_score_max'] = (
                stacked.max(dim=0).values
            )
            query_features[
                f'layer_stability_{label}_score_last_minus_first'
            ] = stacked[-1] - stacked[0]
            top_counts = torch.bincount(
                top_query_stack.detach().long().cpu(),
                minlength=num_queries,
            ).to(device=device, dtype=stacked.dtype)
            query_features[f'layer_stability_{label}_top1_count'] = top_counts
            query_features[f'layer_stability_{label}_top1_ratio'] = (
                top_counts / float(stacked.shape[0])
            )

        _add_source_stats('bbs', layer_scores['bbs'], layer_top_queries['bbs'])
        _add_source_stats('bbf', layer_scores['bbf'], layer_top_queries['bbf'])

        common_prefixes = sorted(
            set(prefix for prefix, _ in layer_scores['bbs'])
            & set(prefix for prefix, _ in layer_scores['bbf']),
            key=GroundingEvaluator._layer_prefix_sort_key,
        )
        if common_prefixes:
            bbs_top_by_prefix = {
                prefix: top for prefix, top in layer_top_queries['bbs']
            }
            bbf_top_by_prefix = {
                prefix: top for prefix, top in layer_top_queries['bbf']
            }
            agree_count = 0.0
            for prefix in common_prefixes:
                agree_count += float(
                    int(bbs_top_by_prefix[prefix].detach().cpu().item())
                    == int(bbf_top_by_prefix[prefix].detach().cpu().item())
                )
            agree_ratio = agree_count / float(len(common_prefixes))
        else:
            agree_count = 0.0
            agree_ratio = 0.0
        row['layer_stability_bbs_bbf_top1_agreement_count'] = float(agree_count)
        row['layer_stability_bbs_bbf_top1_agreement_ratio'] = float(agree_ratio)
        row['layer_stability_bbs_bbf_top1_disagreement_count'] = float(
            len(common_prefixes) - agree_count
        )
        row['layer_stability_bbs_bbf_top1_disagreement_ratio'] = float(
            1.0 - agree_ratio
        )
        return row, query_features

    @staticmethod
    def _source_choice_rapf_features(end_points, bid, query_idx):
        row = {
            'rapf_gate': 0.0,
            'rapf_base_norm': 0.0,
            'rapf_structured_norm': 0.0,
            'rapf_quality_norm': 0.0,
            'rapf_safe_anchor': 0.0,
            'rapf_delta': 0.0,
            'rapf_base_entropy': 0.0,
            'rapf_base_top1_margin': 0.0,
            'rapf_top1_disagreement': 0.0,
            'rapf_js_divergence': 0.0,
            'rapf_quality_max': 0.0,
            'rapf_residual_abs_mean': 0.0,
            'rapf_residual_clip_ratio': 0.0,
            'rapf_top1_changed_ratio': 0.0,
            'rapf_quality_enabled': 0.0,
            'rapf_quality_anchor_enabled': 0.0,
            'rapf_residual_clip': 0.0,
            'rapf_structured_valid_ratio': 0.0,
            'rapf_global_only_ratio': 0.0,
            'rapf_generic_target_ratio': 0.0,
        }
        if query_idx is None:
            return row
        if torch.is_tensor(query_idx):
            query_idx = int(query_idx.detach().cpu().flatten()[0].item())
        else:
            query_idx = int(query_idx)
        if query_idx < 0:
            return row

        def _query_float(key):
            value = end_points.get(key, None)
            if not torch.is_tensor(value):
                return 0.0
            if value.dim() < 2 or bid >= value.shape[0] or query_idx >= value.shape[1]:
                return 0.0
            return GroundingEvaluator._as_float(value[bid, query_idx])

        for key in (
            'rapf_gate',
            'rapf_base_norm',
            'rapf_structured_norm',
            'rapf_quality_norm',
            'rapf_safe_anchor',
            'rapf_delta',
        ):
            row[key] = _query_float(key)

        scalar_keys = (
            'dbg_rapf_base_entropy_mean',
            'dbg_rapf_base_top1_margin_mean',
            'dbg_rapf_top1_disagreement_ratio',
            'dbg_rapf_js_divergence_mean',
            'dbg_rapf_quality_max_mean',
            'dbg_rapf_residual_abs_mean',
            'dbg_rapf_residual_clip_ratio',
            'dbg_rapf_top1_changed_ratio',
            'dbg_rapf_quality_enabled',
            'dbg_rapf_quality_anchor_enabled',
            'dbg_rapf_residual_clip',
            'dbg_rapf_structured_valid_ratio',
            'dbg_rapf_global_only_ratio',
            'dbg_rapf_generic_target_ratio',
        )
        for key in scalar_keys:
            row[key.replace('dbg_', '')] = GroundingEvaluator._batch_float(
                end_points, key, bid, default=0.0
            )
        return row

    @staticmethod
    def _source_choice_pairwise_numeric_deltas(
        row, source_names, fields, include_abs=False
    ):
        for left_idx, left in enumerate(source_names):
            left_available = GroundingEvaluator._value_as_float(
                row.get(f'{left}_available', 0.0)
            ) > 0.5
            for right in source_names[left_idx + 1:]:
                right_available = GroundingEvaluator._value_as_float(
                    row.get(f'{right}_available', 0.0)
                ) > 0.5
                both_available = left_available and right_available
                pair_prefix = f'source_pair_{left}_{right}'
                row[f'{pair_prefix}_both_available'] = (
                    1.0 if both_available else 0.0
                )
                for field in fields:
                    left_value = GroundingEvaluator._value_as_float(
                        row.get(f'{left}_{field}', 0.0)
                    )
                    right_value = GroundingEvaluator._value_as_float(
                        row.get(f'{right}_{field}', 0.0)
                    )
                    delta = left_value - right_value if both_available else 0.0
                    row[f'source_pair_{left}_minus_{right}_{field}'] = delta
                    if include_abs:
                        row[f'{pair_prefix}_{field}_abs_delta'] = abs(delta)

    @staticmethod
    def _source_choice_cross_score_deltas(row, source_names):
        for top_source in source_names:
            top_available = GroundingEvaluator._value_as_float(
                row.get(f'{top_source}_available', 0.0)
            ) > 0.5
            for left_idx, left in enumerate(source_names):
                left_available = GroundingEvaluator._value_as_float(
                    row.get(f'{left}_available', 0.0)
                ) > 0.5
                for right in source_names[left_idx + 1:]:
                    right_available = GroundingEvaluator._value_as_float(
                        row.get(f'{right}_available', 0.0)
                    ) > 0.5
                    if top_available and left_available and right_available:
                        left_value = GroundingEvaluator._value_as_float(
                            row.get(
                                f'{left}_score_at_{top_source}_top',
                                0.0,
                            )
                        )
                        right_value = GroundingEvaluator._value_as_float(
                            row.get(
                                f'{right}_score_at_{top_source}_top',
                                0.0,
                            )
                        )
                        delta = left_value - right_value
                    else:
                        delta = 0.0
                    row[
                        f'source_pair_{left}_minus_{right}'
                        f'_score_at_{top_source}_top'
                    ] = delta

    @staticmethod
    def _source_choice_top_geometry_deltas(row, source_names):
        axes = ('x', 'y', 'z')
        for left_idx, left in enumerate(source_names):
            left_available = GroundingEvaluator._value_as_float(
                row.get(f'{left}_available', 0.0)
            ) > 0.5
            for right in source_names[left_idx + 1:]:
                right_available = GroundingEvaluator._value_as_float(
                    row.get(f'{right}_available', 0.0)
                ) > 0.5
                both_available = left_available and right_available
                pair_prefix = f'source_pair_{left}_{right}'
                center_l1 = 0.0
                size_l1 = 0.0
                for axis in axes:
                    center_delta = 0.0
                    size_delta = 0.0
                    if both_available:
                        center_delta = (
                            GroundingEvaluator._value_as_float(
                                row.get(f'{left}_top_center_{axis}', 0.0)
                            )
                            - GroundingEvaluator._value_as_float(
                                row.get(f'{right}_top_center_{axis}', 0.0)
                            )
                        )
                        size_delta = (
                            GroundingEvaluator._value_as_float(
                                row.get(f'{left}_top_size_{axis}', 0.0)
                            )
                            - GroundingEvaluator._value_as_float(
                                row.get(f'{right}_top_size_{axis}', 0.0)
                            )
                        )
                    center_l1 += abs(center_delta)
                    size_l1 += abs(size_delta)
                    row[
                        f'source_pair_{left}_minus_{right}'
                        f'_top_center_{axis}'
                    ] = center_delta
                    row[
                        f'source_pair_{left}_minus_{right}'
                        f'_top_size_{axis}'
                    ] = size_delta

                if both_available:
                    volume_delta = (
                        GroundingEvaluator._value_as_float(
                            row.get(f'{left}_top_volume', 0.0)
                        )
                        - GroundingEvaluator._value_as_float(
                            row.get(f'{right}_top_volume', 0.0)
                        )
                    )
                else:
                    volume_delta = 0.0
                row[f'{pair_prefix}_top_center_l1_delta'] = center_l1
                row[f'{pair_prefix}_top_size_l1_delta'] = size_l1
                row[
                    f'source_pair_{left}_minus_{right}_top_volume'
                ] = volume_delta
                row[f'{pair_prefix}_top_volume_abs_delta'] = abs(volume_delta)

    @staticmethod
    def _source_choice_top_score_deltas(row, source_names):
        for left_idx, left in enumerate(source_names):
            left_available = GroundingEvaluator._value_as_float(
                row.get(f'{left}_available', 0.0)
            ) > 0.5
            left_top_score = (
                GroundingEvaluator._value_as_float(
                    row.get(f'{left}_score', 0.0)
                )
                + GroundingEvaluator._value_as_float(
                    row.get(f'{left}_delta_to_top', 0.0)
                )
            )
            if f'{left}_top_score' in row:
                left_top_score = GroundingEvaluator._value_as_float(
                    row.get(f'{left}_top_score', left_top_score)
                )
            for right in source_names[left_idx + 1:]:
                right_available = GroundingEvaluator._value_as_float(
                    row.get(f'{right}_available', 0.0)
                ) > 0.5
                if not (left_available and right_available):
                    delta = 0.0
                else:
                    right_top_score = (
                        GroundingEvaluator._value_as_float(
                            row.get(f'{right}_score', 0.0)
                        )
                        + GroundingEvaluator._value_as_float(
                            row.get(f'{right}_delta_to_top', 0.0)
                        )
                    )
                    if f'{right}_top_score' in row:
                        right_top_score = GroundingEvaluator._value_as_float(
                            row.get(f'{right}_top_score', right_top_score)
                        )
                    delta = left_top_score - right_top_score
                row[f'source_pair_{left}_minus_{right}_top_score'] = delta
                row[
                    f'source_pair_{left}_{right}_top_score_abs_delta'
                ] = abs(delta)

    @staticmethod
    def _source_choice_feature_row(
        end_points, bid, num_obj, base_scores, gt_boxes, pred_bbox,
        extra_source_scores=None, prefix='last_', target_map=None
    ):
        source_scores = GroundingEvaluator._candidate_source_scores(
            end_points, bid, num_obj, base_scores,
            extra_source_scores=extra_source_scores,
        )
        source_names = GroundingEvaluator._candidate_dump_source_names(
            extra_source_scores, end_points=end_points
        )
        device = base_scores.device
        num_queries = base_scores.shape[1]
        target_gt = gt_boxes[:1]
        row = {}
        row.update(
            GroundingEvaluator._source_choice_context_features(
                end_points, bid
            )
        )
        row.update(
            GroundingEvaluator._source_choice_selector_logit_features(
                end_points, bid, source_scores
            )
        )
        top_queries = {}
        top_ious = []
        available = []
        intrinsic_inputs = GroundingEvaluator._source_choice_intrinsic_inputs(
            end_points, bid, num_queries, prefix=prefix
        )

        for source in source_names:
            scores = source_scores.get(source, None)
            is_available = (
                scores is not None
                and scores.dim() == 2
                and scores.shape[1] == num_queries
            )
            available.append(bool(is_available))
            row[f'{source}_available'] = 1.0 if is_available else 0.0
            if not is_available:
                top_queries[source] = None
                top_ious.append(torch.tensor(-1.0, device=device))
                row[f'{source}_top_query'] = -1.0
                row[f'{source}_top_score'] = 0.0
                row[f'{source}_top_margin'] = 0.0
                row[f'{source}_top_iou'] = -1.0
                row.update(
                    GroundingEvaluator._source_choice_query_intrinsic_features(
                        end_points, bid, None, pred_bbox, num_queries,
                        field_prefix=f'{source}_top', prefix=prefix,
                        intrinsic_inputs=intrinsic_inputs,
                    )
                )
                continue

            source_row = scores[0].float()
            top_count = min(2, num_queries)
            top_values, top_indices = torch.topk(source_row, top_count, dim=0)
            top_query = top_indices[0].long()
            top_queries[source] = top_query
            top_score = top_values[0]
            if top_count > 1:
                top_margin = top_values[0] - top_values[1]
            else:
                top_margin = top_values.new_tensor(0.0)
            top_iou = GroundingEvaluator._aligned_topk_ious(
                target_gt, pred_bbox, top_query.view(1, 1)
            )[0, 0]
            top_ious.append(top_iou)
            row[f'{source}_top_query'] = GroundingEvaluator._as_float(
                top_query
            )
            row[f'{source}_top_score'] = GroundingEvaluator._as_float(
                top_score
            )
            row[f'{source}_top_margin'] = GroundingEvaluator._as_float(
                top_margin
            )
            row[f'{source}_top_iou'] = GroundingEvaluator._as_float(top_iou)
            row.update(
                GroundingEvaluator._source_choice_query_intrinsic_features(
                    end_points, bid, top_query, pred_bbox, num_queries,
                    field_prefix=f'{source}_top', prefix=prefix,
                    intrinsic_inputs=intrinsic_inputs,
                )
            )
            rapf_features = GroundingEvaluator._source_choice_rapf_features(
                end_points, bid, top_query
            )
            row.update(rapf_features)
            row.update(
                {
                    f'{source}_{key}': value
                    for key, value in rapf_features.items()
                }
            )

        for top_source in source_names:
            top_query = top_queries.get(top_source, None)
            for score_source in source_names:
                key = f'{score_source}_score_at_{top_source}_top'
                scores = source_scores.get(score_source, None)
                if (
                    top_query is None
                    or scores is None
                    or scores.dim() != 2
                    or scores.shape[1] != num_queries
                ):
                    row[key] = 0.0
                    continue
                row[key] = GroundingEvaluator._as_float(scores[0, top_query])

        for left_idx, left in enumerate(source_names):
            left_query = top_queries.get(left, None)
            for right in source_names[left_idx + 1:]:
                right_query = top_queries.get(right, None)
                same = (
                    left_query is not None
                    and right_query is not None
                    and int(left_query.detach().cpu().item())
                    == int(right_query.detach().cpu().item())
                )
                row[f'{left}_{right}_same_query'] = 1.0 if same else 0.0

        GroundingEvaluator._source_choice_pairwise_numeric_deltas(
            row, source_names, ('top_score', 'top_margin'), include_abs=True
        )
        GroundingEvaluator._source_choice_cross_score_deltas(row, source_names)
        GroundingEvaluator._source_choice_top_geometry_deltas(
            row, source_names
        )

        top_iou_tensor = torch.stack(top_ious)
        available_tensor = torch.tensor(
            available, device=device, dtype=torch.bool
        )
        masked_iou = top_iou_tensor.masked_fill(~available_tensor, -1.0)
        oracle_source = masked_iou.argmax()
        threshold_utility = (
            masked_iou
            + (masked_iou > 0.25).float()
            + (masked_iou > 0.50).float()
        )
        threshold_utility = threshold_utility.masked_fill(
            ~available_tensor, -1.0
        )
        threshold_source = threshold_utility.argmax()
        oracle_iou = masked_iou[oracle_source]
        row['oracle_source_id'] = GroundingEvaluator._as_float(oracle_source)
        row['threshold_utility_source_id'] = GroundingEvaluator._as_float(
            threshold_source
        )
        row['oracle_iou'] = GroundingEvaluator._as_float(oracle_iou)
        row['oracle_hit025'] = 1.0 if row['oracle_iou'] > 0.25 else 0.0
        row['oracle_hit050'] = 1.0 if row['oracle_iou'] > 0.50 else 0.0
        return row

    @staticmethod
    def _source_choice_candidate_rows(
        end_points, bid, example_id, num_obj, base_scores, gt_boxes,
        pred_bbox, topk=5, extra_source_scores=None, prefix='last_',
        target_map=None
    ):
        source_scores = GroundingEvaluator._candidate_source_scores(
            end_points, bid, num_obj, base_scores,
            extra_source_scores=extra_source_scores,
        )
        source_names = GroundingEvaluator._candidate_dump_source_names(
            extra_source_scores, end_points=end_points
        )
        device = base_scores.device
        num_queries = base_scores.shape[1]
        k = max(1, min(int(topk), num_queries))
        target_gt = gt_boxes[:1]
        candidate_mask = torch.zeros(num_queries, device=device, dtype=torch.bool)
        source_top = {}

        for source in source_names:
            scores = source_scores.get(source, None)
            if scores is None or scores.dim() != 2 or scores.shape[1] != num_queries:
                source_top[source] = None
                continue
            top = scores[0].float().argsort(0, descending=True)[:k]
            source_top[source] = top
            candidate_mask.scatter_(0, top, True)

        if not bool(candidate_mask.any().detach().item()):
            return []

        candidate_queries = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        candidate_ious = GroundingEvaluator._aligned_topk_ious(
            target_gt, pred_bbox, candidate_queries.view(1, -1)
        )[0]
        oracle_pos = candidate_ious.argmax()
        threshold_utility = (
            candidate_ious
            + (candidate_ious > 0.25).float()
            + (candidate_ious > 0.50).float()
        )
        sem_key = f'{prefix}sem_cls_scores'
        sem_probs = None
        if sem_key in end_points and end_points[sem_key].dim() == 3:
            sem_probs = end_points[sem_key][bid].float().softmax(-1)
        objectness_key = f'{prefix}objectness_scores'
        objectness_scores = None
        if objectness_key in end_points and end_points[objectness_key].dim() >= 2:
            objectness_scores = end_points[objectness_key][bid].float()
        seed_objectness_scores = None
        if (
            'seeds_obj_cls_logits' in end_points
            and 'query_points_sample_inds' in end_points
        ):
            seed_logits = end_points['seeds_obj_cls_logits']
            sample_inds = end_points['query_points_sample_inds']
            if (
                torch.is_tensor(seed_logits)
                and torch.is_tensor(sample_inds)
                and bid < seed_logits.shape[0]
                and bid < sample_inds.shape[0]
            ):
                seed_row = seed_logits[bid].float()
                if seed_row.dim() > 1:
                    seed_row = seed_row.squeeze(0)
                query_seed_inds = sample_inds[bid].long()
                valid_seed = (
                    query_seed_inds >= 0
                ) & (query_seed_inds < seed_row.shape[0])
                if query_seed_inds.numel() >= num_queries and bool(
                    valid_seed[:num_queries].all().detach().cpu().item()
                ):
                    seed_objectness_scores = seed_row[
                        query_seed_inds[:num_queries]
                    ].float()
        intrinsic_inputs = (
            sem_probs,
            objectness_scores,
            seed_objectness_scores,
        )
        layer_stability_row, layer_stability_query_features = (
            GroundingEvaluator._source_choice_layer_stability_features(
                end_points, bid, num_queries, target_map
            )
        )
        rows = []

        for pos, query in enumerate(candidate_queries):
            query_idx = int(query.detach().cpu().item())
            candidate_box = pred_bbox[query_idx].float()
            candidate_center = candidate_box[:3]
            candidate_size = candidate_box[3:6]
            candidate_volume = candidate_size.clamp(min=0.0).prod()
            row = {
                'example_id': float(example_id),
                'candidate_query': GroundingEvaluator._as_float(query),
                'candidate_iou': GroundingEvaluator._as_float(
                    candidate_ious[pos]
                ),
                'candidate_hit025': (
                    1.0 if GroundingEvaluator._as_float(candidate_ious[pos]) > 0.25
                    else 0.0
                ),
                'candidate_hit050': (
                    1.0 if GroundingEvaluator._as_float(candidate_ious[pos]) > 0.50
                    else 0.0
                ),
                'threshold_utility': GroundingEvaluator._as_float(
                    threshold_utility[pos]
                ),
                'oracle_candidate': 1.0 if int(pos) == int(oracle_pos) else 0.0,
                'candidate_center_x': GroundingEvaluator._as_float(
                    candidate_center[0]
                ),
                'candidate_center_y': GroundingEvaluator._as_float(
                    candidate_center[1]
                ),
                'candidate_center_z': GroundingEvaluator._as_float(
                    candidate_center[2]
                ),
                'candidate_size_x': GroundingEvaluator._as_float(
                    candidate_size[0]
                ),
                'candidate_size_y': GroundingEvaluator._as_float(
                    candidate_size[1]
                ),
                'candidate_size_z': GroundingEvaluator._as_float(
                    candidate_size[2]
                ),
                'candidate_volume': GroundingEvaluator._as_float(
                    candidate_volume
                ),
            }
            row.update(
                GroundingEvaluator._source_choice_context_features(
                    end_points, bid
                )
            )
            row.update(layer_stability_row)
            row.update(
                GroundingEvaluator._source_choice_rapf_features(
                    end_points, bid, query_idx
                )
            )
            for key, value in layer_stability_query_features.items():
                if torch.is_tensor(value):
                    row[key] = GroundingEvaluator._as_float(value[query_idx])
                else:
                    row[key] = float(value)
            if sem_probs is not None and query_idx < sem_probs.shape[0]:
                candidate_sem = sem_probs[query_idx]
                top_sem = candidate_sem.topk(min(2, candidate_sem.numel()))
                sem_max = top_sem.values[0]
                sem_margin = (
                    top_sem.values[0] - top_sem.values[1]
                    if top_sem.values.numel() > 1
                    else top_sem.values[0]
                )
                sem_entropy = -(
                    candidate_sem
                    * candidate_sem.clamp(min=1e-12).log()
                ).sum()
                row['candidate_sem_cls_available'] = 1.0
                row['candidate_sem_cls_max'] = GroundingEvaluator._as_float(
                    sem_max
                )
                row['candidate_sem_cls_margin'] = GroundingEvaluator._as_float(
                    sem_margin
                )
                row['candidate_sem_cls_entropy'] = GroundingEvaluator._as_float(
                    sem_entropy
                )
            else:
                row['candidate_sem_cls_available'] = 0.0
                row['candidate_sem_cls_max'] = 0.0
                row['candidate_sem_cls_margin'] = 0.0
                row['candidate_sem_cls_entropy'] = 0.0
            if objectness_scores is not None and query_idx < objectness_scores.shape[0]:
                objectness_score = objectness_scores[query_idx]
                row['candidate_objectness_available'] = 1.0
                row['candidate_objectness_score'] = GroundingEvaluator._as_float(
                    objectness_score
                )
                row['candidate_objectness_prob'] = GroundingEvaluator._as_float(
                    objectness_score.sigmoid()
                )
            else:
                row['candidate_objectness_available'] = 0.0
                row['candidate_objectness_score'] = 0.0
                row['candidate_objectness_prob'] = 0.0
            if (
                seed_objectness_scores is not None
                and query_idx < seed_objectness_scores.shape[0]
            ):
                seed_objectness_score = seed_objectness_scores[query_idx]
                seed_top_score = seed_objectness_scores.max()
                seed_rank = (
                    int(
                        (
                            seed_objectness_scores
                            > seed_objectness_score
                        ).sum().item()
                    )
                    + 1
                )
                row['candidate_seed_objectness_available'] = 1.0
                row['candidate_seed_objectness_score'] = (
                    GroundingEvaluator._as_float(seed_objectness_score)
                )
                row['candidate_seed_objectness_prob'] = (
                    GroundingEvaluator._as_float(
                        seed_objectness_score.sigmoid()
                    )
                )
                row['candidate_seed_objectness_rank'] = float(seed_rank)
                row['candidate_seed_objectness_delta_to_top'] = (
                    GroundingEvaluator._as_float(
                        seed_top_score - seed_objectness_score
                    )
                )
            else:
                row['candidate_seed_objectness_available'] = 0.0
                row['candidate_seed_objectness_score'] = 0.0
                row['candidate_seed_objectness_prob'] = 0.0
                row['candidate_seed_objectness_rank'] = 0.0
                row['candidate_seed_objectness_delta_to_top'] = 0.0
            for source in source_names:
                top = source_top.get(source, None)
                row.update(
                    GroundingEvaluator._source_choice_query_intrinsic_features(
                        end_points, bid, top, pred_bbox, num_queries,
                        field_prefix=f'{source}_top', prefix=prefix,
                        intrinsic_inputs=intrinsic_inputs,
                    )
                )
            for source in source_names:
                scores = source_scores.get(source, None)
                top = source_top.get(source, None)
                available = (
                    scores is not None
                    and scores.dim() == 2
                    and scores.shape[1] == num_queries
                )
                row[f'{source}_available'] = 1.0 if available else 0.0
                if not available:
                    row[f'{source}_score'] = 0.0
                    row[f'{source}_rank'] = 0.0
                    row[f'{source}_delta_to_top'] = 0.0
                    row[f'{source}_in_topk'] = 0.0
                    continue
                source_row = scores[0].float()
                row[f'{source}_score'] = GroundingEvaluator._as_float(
                    source_row[query_idx]
                )
                top_score = source_row.max()
                row[f'{source}_delta_to_top'] = GroundingEvaluator._as_float(
                    top_score - source_row[query_idx]
                )
                rank = int((source_row > source_row[query_idx]).sum().item()) + 1
                row[f'{source}_rank'] = float(rank)
                row[f'{source}_in_topk'] = (
                    1.0
                    if top is not None and bool((top == query).any().item())
                    else 0.0
                )
            GroundingEvaluator._source_choice_pairwise_numeric_deltas(
                row, source_names,
                ('score', 'rank', 'delta_to_top', 'in_topk'),
            )
            GroundingEvaluator._source_choice_top_score_deltas(
                row, source_names
            )
            GroundingEvaluator._source_choice_top_geometry_deltas(
                row, source_names
            )
            rows.append(row)
        return rows

    @staticmethod
    def _contrastive_base_scores(end_points, prefix, bid, pmap):
        if 'proj_tokens' not in end_points:
            return None
        query_key = f'{prefix}proj_queries'
        if query_key not in end_points:
            return None
        proj_tokens = end_points['proj_tokens']
        proj_queries = end_points[query_key]
        sem_scores = torch.matmul(
            proj_queries, proj_tokens.transpose(-1, -2)
        )
        sem_scores = (sem_scores / 0.07).softmax(-1)
        if sem_scores.shape[-1] != pmap.shape[-1]:
            aligned = torch.zeros(
                sem_scores.shape[0],
                sem_scores.shape[1],
                pmap.shape[-1],
                device=sem_scores.device,
                dtype=sem_scores.dtype,
            )
            copy_dim = min(sem_scores.shape[-1], pmap.shape[-1])
            aligned[:, :, :copy_dim] = sem_scores[:, :, :copy_dim]
            sem_scores = aligned
        return (sem_scores[bid].unsqueeze(0) * pmap.unsqueeze(1)).sum(-1)

    @staticmethod
    def _selector_pool_scores(source_scores, selector_scores, candidate_k):
        valid_sources = [
            scores for scores in source_scores.values()
            if scores is not None and scores.shape == selector_scores.shape
        ]
        if not valid_sources:
            return selector_scores.clone()

        num_rows, num_queries = selector_scores.shape
        k = max(1, min(int(candidate_k), num_queries))
        mask = torch.zeros_like(selector_scores, dtype=torch.bool)
        for scores in valid_sources:
            top = scores.argsort(1, True)[:, :k]
            mask.scatter_(1, top, True)

        masked_scores = torch.full_like(selector_scores, -1e30)
        masked_scores[mask] = selector_scores[mask]
        return masked_scores

    @staticmethod
    def _selector_pool_primary_scores(end_points, bid, num_obj, base_scores):
        if 'selector_source_scores' in end_points:
            source_scores = GroundingEvaluator._candidate_source_scores(
                end_points, bid, num_obj, base_scores
            )
            raw_scores = end_points['selector_source_scores'][bid]
            source_names = GroundingEvaluator._selector_choice_source_names(
                raw_scores.shape[0], end_points=end_points
            )
            num_sources = len(source_names)
            num_rows, num_queries = base_scores.shape
            k = max(
                1,
                min(int(end_points.get('eval_selector_pool_k', 1)), num_queries)
            )
            scores = torch.full_like(base_scores, -1e30)
            for source_idx in range(num_sources):
                source = source_names[source_idx]
                if source not in source_scores:
                    continue
                top = source_scores[source].argsort(1, True)[:, :k]
                source_query_scores = raw_scores[source_idx].unsqueeze(0).expand(
                    num_rows, -1
                )
                gathered = torch.gather(source_query_scores, 1, top)
                current = torch.gather(scores, 1, top)
                scores.scatter_(1, top, torch.maximum(current, gathered))
            if num_obj > 1:
                scores[1:] = base_scores[1:]
            return scores

        selector_scores = GroundingEvaluator._single_source_scores(
            end_points, 'selector', bid, num_obj, base_scores=base_scores
        )
        source_scores = GroundingEvaluator._candidate_source_scores(
            end_points, bid, num_obj, base_scores
        )
        scores = GroundingEvaluator._selector_pool_scores(
            source_scores,
            selector_scores,
            int(end_points.get('eval_selector_pool_k', 1)),
        )
        if num_obj > 1:
            scores[1:] = base_scores[1:]
        return scores

    @staticmethod
    def _selector_pool_quality_override_primary_scores(
        end_points, bid, num_obj, base_scores, min_margin=None
    ):
        if min_margin is None:
            min_margin = end_points.get('eval_selector_pool_min_margin', 0.0)
        min_margin = float(min_margin)
        source_scores = GroundingEvaluator._candidate_source_scores(
            end_points, bid, num_obj, base_scores
        )
        quality_scores = source_scores.get('quality', base_scores).clone()
        if 'quality' not in source_scores or num_obj <= 0:
            return quality_scores

        selector_pool_scores = GroundingEvaluator._selector_pool_primary_scores(
            end_points, bid, num_obj, base_scores
        )
        pool_target_scores = selector_pool_scores[0]
        selected_value, selected_query = pool_target_scores.max(dim=0)
        quality_query = quality_scores[0].argmax(dim=0)
        if int(selected_query.detach().cpu().item()) == int(
            quality_query.detach().cpu().item()
        ):
            return quality_scores

        quality_pool_value = pool_target_scores[quality_query]
        invalid_floor = -1e20
        if float(selected_value.detach().cpu().item()) <= invalid_floor:
            return quality_scores
        if float(quality_pool_value.detach().cpu().item()) <= invalid_floor:
            return quality_scores
        margin = selected_value - quality_pool_value
        if float(margin.detach().cpu().item()) < min_margin:
            return quality_scores

        scores = quality_scores.clone()
        scores[0] = torch.full_like(scores[0], -1e30)
        scores[0, selected_query] = selected_value
        return scores

    @staticmethod
    def _selector_pool_quality_override_diagnostic_scores(
        end_points, bid, num_obj, base_scores
    ):
        if num_obj <= 0 or 'pred_iou' not in end_points:
            return {}
        if (
            'selector_scores' not in end_points
            and 'selector_source_scores' not in end_points
        ):
            return {}
        margins = (0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.50, 1.00)
        scores = {}
        for margin in margins:
            label = GroundingEvaluator._selector_choice_hybrid_margin_label(
                margin
            )
            source = f'selector_pool_quality_override_m{label}_quality'
            scores[source] = (
                GroundingEvaluator
                ._selector_pool_quality_override_primary_scores(
                    end_points,
                    bid,
                    num_obj,
                    base_scores,
                    min_margin=margin,
                )
            )
        return scores

    @staticmethod
    def _selector_pool_source_blend_primary_scores(
        end_points, bid, num_obj, base_scores, alpha=0.0
    ):
        if 'selector_source_scores' not in end_points:
            return GroundingEvaluator._selector_pool_primary_scores(
                end_points, bid, num_obj, base_scores
            )
        source_scores = GroundingEvaluator._candidate_source_scores(
            end_points, bid, num_obj, base_scores
        )
        raw_scores = end_points['selector_source_scores'][bid]
        source_names = GroundingEvaluator._selector_choice_source_names(
            raw_scores.shape[0], end_points=end_points
        )
        num_sources = len(source_names)
        num_rows, num_queries = base_scores.shape
        k = max(
            1,
            min(int(end_points.get('eval_selector_pool_k', 1)), num_queries)
        )
        scores = torch.full_like(base_scores, -1e30)
        for source_idx in range(num_sources):
            source = source_names[source_idx]
            if source not in source_scores:
                continue
            top = source_scores[source].argsort(1, True)[:, :k]
            selector_values = raw_scores[source_idx].unsqueeze(0).expand(
                num_rows, -1
            )
            blended_values = (
                selector_values + (float(alpha) * source_scores[source])
            )
            gathered = torch.gather(blended_values, 1, top)
            current = torch.gather(scores, 1, top)
            scores.scatter_(1, top, torch.maximum(current, gathered))
        if num_obj > 1:
            scores[1:] = base_scores[1:]
        return scores

    @staticmethod
    def _selector_pool_source_blend_diagnostic_scores(
        end_points, bid, num_obj, base_scores
    ):
        if num_obj <= 0 or 'selector_source_scores' not in end_points:
            return {}
        alphas = (0.00, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00)
        scores = {}
        for alpha in alphas:
            label = GroundingEvaluator._selector_choice_hybrid_margin_label(
                alpha
            )
            source = f'selector_pool_source_blend_a{label}'
            scores[source] = (
                GroundingEvaluator
                ._selector_pool_source_blend_primary_scores(
                    end_points,
                    bid,
                    num_obj,
                    base_scores,
                    alpha=alpha,
                )
            )
        return scores

    @staticmethod
    def _selector_pool_diagnostics(
        end_points, bid, num_obj, base_scores, gt_boxes, pred_bbox
    ):
        if num_obj <= 0:
            return {}
        num_queries = base_scores.shape[1]
        device = base_scores.device
        target_gt = gt_boxes[:1]
        all_queries = torch.arange(
            num_queries, device=device, dtype=torch.long
        ).view(1, -1)
        target_ious = GroundingEvaluator._aligned_topk_ious(
            target_gt, pred_bbox, all_queries
        )[0]
        candidate_mask = torch.zeros(
            num_queries, device=device, dtype=torch.bool
        )
        selector_values = torch.full(
            (num_queries,),
            torch.finfo(base_scores.dtype).min,
            device=device,
            dtype=base_scores.dtype,
        )
        k = max(
            1,
            min(int(end_points.get('eval_selector_pool_k', 1)), num_queries)
        )
        source_scores = GroundingEvaluator._candidate_source_scores(
            end_points, bid, num_obj, base_scores
        )

        if 'selector_source_scores' in end_points:
            raw_scores = end_points['selector_source_scores'][bid]
            source_names = GroundingEvaluator._selector_choice_source_names(
                raw_scores.shape[0], end_points=end_points
            )
            num_sources = len(source_names)
            for source_idx, source in enumerate(source_names[:num_sources]):
                if source not in source_scores:
                    continue
                top = source_scores[source][0].argsort(0, True)[:k]
                candidate_mask.scatter_(0, top, True)
                source_query_scores = raw_scores[source_idx, top].to(
                    dtype=selector_values.dtype
                )
                current = selector_values[top]
                better = source_query_scores > current
                if bool(better.any().detach().item()):
                    selector_values[top[better]] = source_query_scores[better]
        elif 'selector_scores' in end_points:
            selector_scores = GroundingEvaluator._single_source_scores(
                end_points, 'selector', bid, num_obj,
                base_scores=base_scores,
            )[0]
            for scores in source_scores.values():
                if scores is None or scores.shape != base_scores.shape:
                    continue
                top = scores[0].argsort(0, True)[:k]
                candidate_mask.scatter_(0, top, True)
            selector_values[candidate_mask] = selector_scores[candidate_mask]
        else:
            return {}

        if not bool(candidate_mask.any().detach().item()):
            return {}

        selected_query = selector_values.argmax(dim=0)
        oracle_ious = target_ious.masked_fill(~candidate_mask, -1.0)
        oracle_query = oracle_ious.argmax(dim=0)
        selected_iou = target_ious[selected_query]
        oracle_iou = target_ious[oracle_query]
        selected_score = selector_values[selected_query]
        oracle_score = selector_values[oracle_query]
        diagnostics = {
            'selected_query_id': selected_query.float(),
            'oracle_query_id': oracle_query.float(),
            'selector_pool_oracle_agree': (
                selected_query == oracle_query
            ).float(),
            'selector_pool_selected_iou': selected_iou,
            'selector_pool_oracle_iou': oracle_iou,
            'selector_pool_iou_gap_to_oracle': oracle_iou - selected_iou,
            'selector_pool_score_gap_to_oracle': (
                oracle_score - selected_score
            ),
        }
        for threshold, label in ((0.25, '025'), (0.50, '050')):
            diagnostics[f'selector_pool_selected_hit{label}'] = (
                selected_iou > threshold
            ).float()
            diagnostics[f'selector_pool_oracle_hit{label}'] = (
                oracle_iou > threshold
            ).float()
        return diagnostics

    @staticmethod
    def _selector_choice_primary_scores(
        end_points, bid, num_obj, base_scores, extra_source_scores=None,
        override_threshold=None,
    ):
        if 'selector_choice_scores' not in end_points:
            raise ValueError(
                "selector_choice primary eval requires selector_choice_scores"
            )
        source_scores = GroundingEvaluator._candidate_source_scores(
            end_points, bid, num_obj, base_scores,
            extra_source_scores=extra_source_scores,
        )
        raw_scores = end_points['selector_choice_scores'][bid]
        num_sources = raw_scores.shape[0]
        source_names = GroundingEvaluator._selector_choice_source_names(
            num_sources, end_points=end_points
        )
        choice_logits = raw_scores[:num_sources].clone()
        choice_logits = choice_logits + (
            GroundingEvaluator._selector_choice_source_biases(
                end_points, num_sources, choice_logits.device,
                choice_logits.dtype,
            )
        )
        for source_idx, source in enumerate(source_names[:num_sources]):
            if source not in source_scores:
                choice_logits[source_idx] = torch.finfo(choice_logits.dtype).min
        selected_source = GroundingEvaluator._selector_choice_selected_source_index(
            end_points, bid, choice_logits,
            override_threshold=override_threshold,
        )
        selected_name = source_names[selected_source]
        selected_scores = source_scores[selected_name]
        top_idx = int(selected_scores[0].argmax().detach().cpu().item())

        scores = torch.full_like(base_scores, -1e30)
        scores[0, top_idx] = raw_scores[selected_source]
        if num_obj > 1:
            scores[1:] = base_scores[1:]
        return scores

    @staticmethod
    def _selector_choice_hybrid_primary_scores(
        end_points, bid, num_obj, base_scores, min_margin=None, fallback=None,
        extra_source_scores=None,
    ):
        if 'selector_choice_scores' not in end_points:
            raise ValueError(
                "selector_choice hybrid eval requires selector_choice_scores"
            )
        if fallback is None:
            fallback = end_points.get(
                'eval_selector_choice_hybrid_fallback', 'quality'
            )
        if fallback not in ('base', 'fused', 'quality'):
            raise ValueError(
                "eval_selector_choice_hybrid_fallback must be one of "
                "base/fused/quality"
            )
        if min_margin is None:
            min_margin = end_points.get('eval_selector_choice_min_margin', 0.0)
        min_margin = float(min_margin)
        source_scores = GroundingEvaluator._candidate_source_scores(
            end_points, bid, num_obj, base_scores,
            extra_source_scores=extra_source_scores,
        )
        fallback_scores = source_scores.get(fallback, base_scores).clone()

        raw_scores = end_points['selector_choice_scores'][bid]
        num_sources = raw_scores.shape[0]
        source_names = GroundingEvaluator._selector_choice_source_names(
            num_sources, end_points=end_points
        )
        choice_logits = raw_scores[:num_sources].clone()
        choice_logits = choice_logits + (
            GroundingEvaluator._selector_choice_source_biases(
                end_points, num_sources, choice_logits.device,
                choice_logits.dtype,
            )
        )
        for source_idx, source in enumerate(source_names[:num_sources]):
            if source not in source_scores:
                choice_logits[source_idx] = torch.finfo(
                    choice_logits.dtype
                ).min
        if num_sources == 0:
            return fallback_scores

        top_count = min(2, num_sources)
        top_values, top_indices = torch.topk(choice_logits, top_count, dim=0)
        if top_count > 1:
            margin = top_values[0] - top_values[1]
        else:
            margin = torch.tensor(float('inf'), device=choice_logits.device)
        if float(margin.detach().cpu().item()) < min_margin:
            return fallback_scores

        selected_source = int(top_indices[0].detach().cpu().item())
        selected_name = source_names[selected_source]
        selected_scores = source_scores[selected_name]
        top_idx = int(selected_scores[0].argmax().detach().cpu().item())

        scores = torch.full_like(base_scores, -1e30)
        scores[0, top_idx] = raw_scores[selected_source]
        if num_obj > 1:
            scores[1:] = base_scores[1:]
        return scores

    @staticmethod
    def _selector_choice_quality_override_primary_scores(
        end_points, bid, num_obj, base_scores, min_margin=None,
        extra_source_scores=None, override_threshold=None,
    ):
        if 'selector_choice_scores' not in end_points:
            raise ValueError(
                "selector_choice quality override eval requires "
                "selector_choice_scores"
            )
        if min_margin is None:
            min_margin = end_points.get('eval_selector_choice_min_margin', 0.0)
        min_margin = float(min_margin)
        source_scores = GroundingEvaluator._candidate_source_scores(
            end_points, bid, num_obj, base_scores,
            extra_source_scores=extra_source_scores,
        )
        quality_scores = source_scores.get('quality', base_scores).clone()

        raw_scores = end_points['selector_choice_scores'][bid]
        num_sources = raw_scores.shape[0]
        source_names = GroundingEvaluator._selector_choice_source_names(
            num_sources, end_points=end_points
        )
        quality_idx = (
            source_names.index('quality') if 'quality' in source_names else -1
        )
        if (
            quality_idx < 0
            or num_sources <= quality_idx
            or 'quality' not in source_scores
        ):
            return quality_scores

        choice_logits = raw_scores[:num_sources].clone()
        choice_logits = choice_logits + (
            GroundingEvaluator._selector_choice_source_biases(
                end_points, num_sources, choice_logits.device,
                choice_logits.dtype,
            )
        )
        for source_idx, source in enumerate(source_names[:num_sources]):
            if source not in source_scores:
                choice_logits[source_idx] = torch.finfo(
                    choice_logits.dtype
                ).min

        use_override_head = bool(
            end_points.get('eval_selector_choice_use_override_head', False)
        )
        if override_threshold is not None:
            use_override_head = True
        else:
            override_threshold = float(
                end_points.get('eval_selector_choice_override_threshold', 0.0)
            )
        if use_override_head and 'selector_choice_override_logit' in end_points:
            override_logit = end_points['selector_choice_override_logit'][bid]
            if float(override_logit.detach().cpu().item()) <= override_threshold:
                return quality_scores
            nonquality_logits = choice_logits.clone()
            nonquality_logits[quality_idx] = torch.finfo(
                nonquality_logits.dtype
            ).min
            best_nonquality_value, best_nonquality_idx = nonquality_logits.max(
                dim=0
            )
            if (
                float(
                    best_nonquality_value.detach().cpu().item()
                ) <= torch.finfo(nonquality_logits.dtype).min / 2.0
            ):
                return quality_scores
            selected_source = int(best_nonquality_idx.detach().cpu().item())
            selected_name = source_names[selected_source]
            if selected_name not in source_scores:
                return quality_scores
            selected_scores = source_scores[selected_name]
            top_idx = int(selected_scores[0].argmax().detach().cpu().item())

            scores = torch.full_like(base_scores, -1e30)
            scores[0, top_idx] = raw_scores[selected_source]
            if num_obj > 1:
                scores[1:] = base_scores[1:]
            return scores

        nonquality_logits = choice_logits.clone()
        nonquality_logits[quality_idx] = torch.finfo(
            nonquality_logits.dtype
        ).min
        best_nonquality_value, best_nonquality_idx = nonquality_logits.max(
            dim=0
        )
        quality_value = choice_logits[quality_idx]
        if float(
            (best_nonquality_value - quality_value).detach().cpu().item()
        ) < min_margin:
            return quality_scores

        selected_source = int(best_nonquality_idx.detach().cpu().item())
        selected_name = source_names[selected_source]
        if selected_name not in source_scores:
            return quality_scores
        selected_scores = source_scores[selected_name]
        top_idx = int(selected_scores[0].argmax().detach().cpu().item())

        scores = torch.full_like(base_scores, -1e30)
        scores[0, top_idx] = raw_scores[selected_source]
        if num_obj > 1:
            scores[1:] = base_scores[1:]
        return scores

    @staticmethod
    def _selector_choice_hybrid_margin_label(margin):
        text = "{:.2f}".format(float(margin)).rstrip('0').rstrip('.')
        return text.replace('.', 'p')

    @staticmethod
    def _selector_choice_override_threshold_label(threshold):
        value = float(threshold)
        prefix = 'neg' if value < 0 else ''
        text = "{:.2f}".format(abs(value)).rstrip('0').rstrip('.')
        return prefix + text.replace('.', 'p')

    @staticmethod
    def _selector_choice_override_threshold_diagnostic_scores(
        end_points, bid, num_obj, base_scores, extra_source_scores=None
    ):
        if 'selector_choice_scores' not in end_points or num_obj <= 0:
            return {}
        if 'selector_choice_override_logit' not in end_points:
            return {}
        thresholds = (
            -2.00, -1.50, -1.00, -0.75, -0.50, -0.25, 0.00,
            0.25, 0.50, 0.75, 1.00, 1.50, 2.00,
        )
        scores = {}
        for threshold in thresholds:
            label = (
                GroundingEvaluator
                ._selector_choice_override_threshold_label(threshold)
            )
            source = f'selector_choice_override_t{label}'
            scores[source] = (
                GroundingEvaluator._selector_choice_primary_scores(
                    end_points,
                    bid,
                    num_obj,
                    base_scores,
                    extra_source_scores=extra_source_scores,
                    override_threshold=threshold,
                )
            )
        return scores

    @staticmethod
    def _selector_choice_hybrid_diagnostic_scores(
        end_points, bid, num_obj, base_scores, extra_source_scores=None
    ):
        if 'selector_choice_scores' not in end_points or num_obj <= 0:
            return {}
        fallback = 'quality' if 'pred_iou' in end_points else 'base'
        margins = (0.05, 0.10, 0.25, 0.50, 1.00)
        scores = {}
        for margin in margins:
            label = GroundingEvaluator._selector_choice_hybrid_margin_label(
                margin
            )
            source = f'selector_choice_hybrid_m{label}_{fallback}'
            scores[source] = (
                GroundingEvaluator._selector_choice_hybrid_primary_scores(
                    end_points,
                    bid,
                    num_obj,
                    base_scores,
                    min_margin=margin,
                    fallback=fallback,
                    extra_source_scores=extra_source_scores,
                )
            )
        return scores

    @staticmethod
    def _selector_choice_quality_override_diagnostic_scores(
        end_points, bid, num_obj, base_scores, extra_source_scores=None
    ):
        if 'selector_choice_scores' not in end_points or num_obj <= 0:
            return {}
        if 'pred_iou' not in end_points:
            return {}
        margins = (0.00, 0.05, 0.10, 0.25, 0.50, 1.00)
        scores = {}
        for margin in margins:
            label = GroundingEvaluator._selector_choice_hybrid_margin_label(
                margin
            )
            source = f'selector_choice_quality_override_m{label}_quality'
            scores[source] = (
                GroundingEvaluator
                ._selector_choice_quality_override_primary_scores(
                    end_points,
                    bid,
                    num_obj,
                    base_scores,
                    min_margin=margin,
                    extra_source_scores=extra_source_scores,
                )
            )
        if 'selector_choice_override_logit' in end_points:
            thresholds = (
                -2.00, -1.50, -1.00, -0.75, -0.50, -0.25, 0.00,
                0.25, 0.50, 0.75, 1.00, 1.50, 2.00,
            )
            for threshold in thresholds:
                label = (
                    GroundingEvaluator
                    ._selector_choice_override_threshold_label(threshold)
                )
                source = f'selector_choice_quality_override_t{label}_quality'
                scores[source] = (
                    GroundingEvaluator
                    ._selector_choice_quality_override_primary_scores(
                        end_points,
                        bid,
                        num_obj,
                        base_scores,
                        min_margin=0.0,
                        extra_source_scores=extra_source_scores,
                        override_threshold=threshold,
                    )
                )
        return scores

    @staticmethod
    def _selector_choice_diagnostics(
        end_points, bid, num_obj, base_scores, gt_boxes, pred_bbox,
        extra_source_scores=None, selection_mode='selector_choice'
    ):
        source_scores = GroundingEvaluator._candidate_source_scores(
            end_points, bid, num_obj, base_scores,
            extra_source_scores=extra_source_scores,
        )
        raw_scores = end_points['selector_choice_scores'][bid]
        num_sources = raw_scores.shape[0]
        source_names = GroundingEvaluator._selector_choice_source_names(
            num_sources, end_points=end_points
        )
        device = raw_scores.device
        choice_logits = raw_scores[:num_sources].clone()
        choice_logits = choice_logits + (
            GroundingEvaluator._selector_choice_source_biases(
                end_points, num_sources, choice_logits.device,
                choice_logits.dtype,
            )
        )
        top_ious = torch.zeros(num_sources, device=device)
        available = torch.zeros(num_sources, device=device, dtype=torch.bool)
        target_gt = gt_boxes[:1]
        for source_idx, source in enumerate(source_names[:num_sources]):
            if source not in source_scores:
                choice_logits[source_idx] = torch.finfo(choice_logits.dtype).min
                continue
            source_top = source_scores[source][0].argmax().view(1, 1)
            top_ious[source_idx] = GroundingEvaluator._aligned_topk_ious(
                target_gt, pred_bbox, source_top
            )[0, 0]
            available[source_idx] = True

        if selection_mode == 'selector_choice_quality_override':
            quality_idx = (
                source_names.index('quality')
                if 'quality' in source_names
                else -1
            )
            use_override_head = bool(
                end_points.get('eval_selector_choice_use_override_head', False)
            )
            override_threshold = float(
                end_points.get('eval_selector_choice_override_threshold', 0.0)
            )
            selected_source_idx = max(quality_idx, 0)
            if (
                quality_idx < 0
                or num_sources <= quality_idx
                or not bool(available[quality_idx])
            ):
                selected_source_idx = int(
                    choice_logits.argmax().detach().cpu().item()
                )
            elif (
                use_override_head
                and 'selector_choice_override_logit' in end_points
            ):
                override_logit = end_points['selector_choice_override_logit'][bid]
                if (
                    float(override_logit.detach().cpu().item())
                    > override_threshold
                ):
                    nonquality_logits = choice_logits.clone()
                    nonquality_logits[quality_idx] = torch.finfo(
                        nonquality_logits.dtype
                    ).min
                    selected_source_idx = int(
                        nonquality_logits.argmax().detach().cpu().item()
                    )
            else:
                nonquality_logits = choice_logits.clone()
                nonquality_logits[quality_idx] = torch.finfo(
                    nonquality_logits.dtype
                ).min
                best_nonquality_value, best_nonquality_idx = (
                    nonquality_logits.max(dim=0)
                )
                quality_value = choice_logits[quality_idx]
                min_margin = float(
                    end_points.get('eval_selector_choice_min_margin', 0.0)
                )
                if float(
                    (best_nonquality_value - quality_value)
                    .detach()
                    .cpu()
                    .item()
                ) >= min_margin:
                    selected_source_idx = int(
                        best_nonquality_idx.detach().cpu().item()
                    )
            selected_source = choice_logits.new_tensor(
                selected_source_idx,
                dtype=torch.long,
            )
        else:
            selected_source = choice_logits.new_tensor(
                GroundingEvaluator._selector_choice_selected_source_index(
                    end_points, bid, choice_logits
                ),
                dtype=torch.long,
            )
        oracle_source = top_ious.masked_fill(~available, -1.0).argmax()
        selected_iou = top_ious[selected_source]
        oracle_iou = top_ious[oracle_source]
        agree = (selected_source == oracle_source).float()
        override_default_idx = 0
        if (
            selection_mode == 'selector_choice_quality_override'
            and num_sources > 2
            and bool(available[2].detach().item())
        ):
            override_default_idx = 2
        target_override = oracle_source != override_default_idx
        selected_override = selected_source != override_default_idx
        diagnostics = {
            'selected_source_id': selected_source.float(),
            'oracle_source_id': oracle_source.float(),
            'source_choice_oracle_agree': agree,
            'source_choice_selected_iou': selected_iou,
            'source_choice_oracle_iou': oracle_iou,
            'source_choice_iou_gap_to_oracle': oracle_iou - selected_iou,
            'source_choice_target_override_ratio': target_override.float(),
            'source_choice_selected_override_ratio': selected_override.float(),
            'source_choice_false_base_ratio': (
                (target_override & ~selected_override).float()
            ),
            'source_choice_false_override_ratio': (
                (~target_override & selected_override).float()
            ),
            'source_choice_override_agreement_ratio': (
                (target_override == selected_override).float()
            ),
        }
        for threshold, label in ((0.25, '025'), (0.50, '050')):
            hit_mask = (top_ious > threshold) & available
            diagnostics[f'source_choice_selected_hit{label}'] = (
                hit_mask[selected_source].float()
            )
            diagnostics[f'source_choice_oracle_hit{label}'] = (
                hit_mask[oracle_source].float()
            )
            for source_idx, source in enumerate(source_names):
                source_hit = (
                    hit_mask[source_idx]
                    if source_idx < num_sources
                    else raw_scores.new_tensor(False, dtype=torch.bool)
                )
                other_hit = (
                    hit_mask.clone()
                    if source_idx < num_sources
                    else hit_mask
                )
                if source_idx < num_sources:
                    other_hit[source_idx] = False
                diagnostics[f'source_choice_{source}_hit{label}_ratio'] = (
                    source_hit.float()
                )
                diagnostics[
                    f'source_choice_{source}_unique_hit{label}_ratio'
                ] = (
                    (source_hit & ~other_hit.any()).float()
                )
        for source_idx, source in enumerate(source_names):
            suffix = source
            diagnostics[f'source_choice_selected_{suffix}_ratio'] = (
                (selected_source == source_idx).float()
                if source_idx < num_sources else raw_scores.new_tensor(0.0)
            )
            diagnostics[f'source_choice_oracle_{suffix}_ratio'] = (
                (oracle_source == source_idx).float()
                if source_idx < num_sources else raw_scores.new_tensor(0.0)
            )
            if (
                source_idx < num_sources
                and bool(available[source_idx].detach().item())
            ):
                source_logit = raw_scores[source_idx].float()
                other_available = available.clone()
                other_available[source_idx] = False
                if bool(other_available.any().detach().item()):
                    other_logits = raw_scores[:num_sources].float().masked_fill(
                        ~other_available, torch.finfo(raw_scores.dtype).min
                    )
                    margin = source_logit - other_logits.max()
                else:
                    margin = raw_scores.new_tensor(0.0)
                diagnostics[f'source_choice_logit_{suffix}_mean'] = (
                    source_logit
                )
                diagnostics[
                    f'source_choice_logit_margin_{suffix}_mean'
                ] = margin
            else:
                diagnostics[f'source_choice_logit_{suffix}_mean'] = (
                    raw_scores.new_tensor(0.0)
                )
                diagnostics[
                    f'source_choice_logit_margin_{suffix}_mean'
                ] = raw_scores.new_tensor(0.0)
        return diagnostics

    def _record_diagnostic(self, source, prefix, scores, gt_boxes, pred_bbox):
        top = scores.argsort(1, True)[:, :10]
        ious = self._aligned_topk_ious(gt_boxes, pred_bbox, top)
        for t in self.thresholds:
            thresholded = ious > t
            for k in self.topks:
                found = thresholded[:, :k].any(1).sum().item()
                key = (source, prefix, t, k)
                self.diagnostic_dets[key] = self.diagnostic_dets.get(key, 0) + found
                self.diagnostic_gts[key] = self.diagnostic_gts.get(key, 0) + len(thresholded)
        return ious

    def _record_diagnostic_source_pool_oracle(self, prefix, source_ious):
        if not source_ious:
            return
        valid_ious = [
            ious for ious in source_ious.values()
            if ious is not None and ious.numel() > 0
        ]
        if not valid_ious:
            return
        num_targets = valid_ious[0].shape[0]
        for t in self.thresholds:
            for k in self.topks:
                source_hits = []
                for ious in valid_ious:
                    if ious.shape[0] != num_targets:
                        continue
                    upto = min(k, ious.shape[1])
                    if upto <= 0:
                        continue
                    source_hits.append((ious[:, :upto] > t).any(1))
                if not source_hits:
                    continue
                found = torch.stack(source_hits, dim=0).any(0).sum().item()
                key = ('source_pool_oracle', prefix, t, k)
                self.diagnostic_dets[key] = (
                    self.diagnostic_dets.get(key, 0) + found
                )
                self.diagnostic_gts[key] = (
                    self.diagnostic_gts.get(key, 0) + num_targets
                )

    @staticmethod
    def _rerank_candidate_indices(candidate_scores, rerank_scores, candidate_k):
        candidate_top = candidate_scores.argsort(1, True)[:, :candidate_k]
        candidate_rerank_scores = torch.gather(
            rerank_scores, 1, candidate_top
        )
        rerank_order = candidate_rerank_scores.argsort(1, True)
        return torch.gather(candidate_top, 1, rerank_order)

    @staticmethod
    def _rerank_source_pool_candidate_indices(source_scores, rerank_scores,
                                             candidate_k):
        valid_sources = [
            scores for scores in source_scores.values()
            if scores is not None and scores.shape == rerank_scores.shape
        ]
        if not valid_sources:
            return rerank_scores.argsort(1, True)

        num_rows, num_queries = rerank_scores.shape
        k = max(1, min(int(candidate_k), num_queries))
        max_candidates = min(num_queries, k * len(valid_sources))
        source_tops = [
            scores.argsort(1, True)[:, :k].detach().cpu().tolist()
            for scores in valid_sources
        ]
        rows = []
        for row_idx in range(num_rows):
            seen = set()
            candidates = []
            for top in source_tops:
                for query_idx in top[row_idx]:
                    if query_idx in seen:
                        continue
                    seen.add(query_idx)
                    candidates.append(query_idx)
            candidates.sort(
                key=lambda query_idx: float(
                    rerank_scores[row_idx, query_idx].detach().cpu().item()
                ),
                reverse=True,
            )
            if not candidates:
                candidates = [
                    int(rerank_scores[row_idx].argmax().detach().cpu().item())
                ]
            while len(candidates) < max_candidates:
                candidates.append(candidates[-1])
            rows.append(torch.tensor(
                candidates[:max_candidates],
                device=rerank_scores.device,
                dtype=torch.long,
            ))
        return torch.stack(rows, dim=0)

    def _record_diagnostic_rerank(self, source, prefix, candidate_scores,
                                  rerank_scores, gt_boxes, pred_bbox,
                                  candidate_k):
        top = self._rerank_candidate_indices(
            candidate_scores, rerank_scores, candidate_k
        )[:, :10]
        ious = self._aligned_topk_ious(gt_boxes, pred_bbox, top)
        for t in self.thresholds:
            thresholded = ious > t
            for k in self.topks:
                found = thresholded[:, :k].any(1).sum().item()
                key = (source, prefix, t, k)
                self.diagnostic_dets[key] = (
                    self.diagnostic_dets.get(key, 0) + found
                )
                self.diagnostic_gts[key] = (
                    self.diagnostic_gts.get(key, 0) + len(thresholded)
                )

    def _record_diagnostic_source_pool_rerank(self, source, prefix,
                                             source_scores, rerank_scores,
                                             gt_boxes, pred_bbox,
                                             candidate_k):
        top = self._rerank_source_pool_candidate_indices(
            source_scores, rerank_scores, candidate_k
        )
        ious = self._aligned_topk_ious(gt_boxes, pred_bbox, top)
        for t in self.thresholds:
            thresholded = ious > t
            for k in self.topks:
                found = thresholded[:, :k].any(1).sum().item()
                key = (source, prefix, t, k)
                self.diagnostic_dets[key] = (
                    self.diagnostic_dets.get(key, 0) + found
                )
                self.diagnostic_gts[key] = (
                    self.diagnostic_gts.get(key, 0) + len(thresholded)
                )

    def _record_decomposition(self, status, prefix, ious):
        count_key = (status, prefix)
        self.decomposition_status_counts[count_key] = (
            self.decomposition_status_counts.get(count_key, 0) + ious.shape[0]
        )
        for t in self.thresholds:
            thresholded = ious > t
            for k in self.topks:
                found = thresholded[:, :k].any(1).sum().item()
                key = (status, prefix, t, k)
                self.decomposition_dets[key] = self.decomposition_dets.get(key, 0) + found
                self.decomposition_gts[key] = self.decomposition_gts.get(key, 0) + len(thresholded)

    def _record_spacy_augmentation(self, bucket, prefix, ious):
        if bucket is None:
            return
        count_key = (bucket, prefix)
        self.spacy_augmentation_counts[count_key] = (
            self.spacy_augmentation_counts.get(count_key, 0) + ious.shape[0]
        )
        for t in self.thresholds:
            thresholded = ious > t
            for k in self.topks:
                found = thresholded[:, :k].any(1).sum().item()
                key = (bucket, prefix, t, k)
                self.spacy_augmentation_dets[key] = (
                    self.spacy_augmentation_dets.get(key, 0) + found
                )
                self.spacy_augmentation_gts[key] = (
                    self.spacy_augmentation_gts.get(key, 0) + len(thresholded)
                )

    @staticmethod
    def _requested_spacy_source_score_names(end_points):
        raw = end_points.get('eval_spacy_source_score_sources', '')
        if raw is None:
            return ()
        if isinstance(raw, str):
            items = raw.split(',')
        elif isinstance(raw, (list, tuple)):
            items = raw
        else:
            items = [raw]
        names = []
        seen = set()
        for item in items:
            name = str(item).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        return tuple(names)

    def _spacy_source_audit_scores(self, end_points, source, bid, num_obj,
                                   base_scores, contrastive_scores):
        if source == 'base':
            return base_scores
        if source == 'contrastive_base':
            return contrastive_scores
        try:
            return self._single_source_scores(
                end_points, source, bid, num_obj, base_scores=base_scores
            )
        except (KeyError, ValueError):
            return None

    def _record_spacy_source_score(self, bucket, source, prefix, scores,
                                   gt_boxes, pred_bbox):
        if bucket is None or scores is None:
            return
        top = scores.argsort(1, True)[:, :10]
        ious = self._aligned_topk_ious(gt_boxes, pred_bbox, top)
        count_key = (bucket, source, prefix)
        self.spacy_source_counts[count_key] = (
            self.spacy_source_counts.get(count_key, 0) + ious.shape[0]
        )
        for t in self.thresholds:
            thresholded = ious > t
            for k in self.topks:
                found = thresholded[:, :k].any(1).sum().item()
                key = (bucket, source, prefix, t, k)
                self.spacy_source_dets[key] = (
                    self.spacy_source_dets.get(key, 0) + found
                )
                self.spacy_source_gts[key] = (
                    self.spacy_source_gts.get(key, 0) + len(thresholded)
                )

    def evaluate_bbox_by_span(self, end_points, prefix):
        """
        Evaluate bounding box IoU for top gt span detections.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        positive_map, gt_bboxes = self._parse_gt(end_points)

        primary_source = self._primary_source(end_points, prefix)
        if prefix == 'last_':
            self.primary_score_source = primary_source
            self.bbs_score_source = primary_source
        sem_scores = end_points[f'{prefix}sem_cls_scores'].softmax(-1)
        if sem_scores.shape[-1] != positive_map.shape[-1]:
            sem_scores_ = torch.zeros(
                sem_scores.shape[0], sem_scores.shape[1],
                positive_map.shape[-1]).to(sem_scores.device)
            sem_scores_[:, :, :sem_scores.shape[-1]] = sem_scores
            sem_scores = sem_scores_

        pred_center = end_points[f'{prefix}center']
        pred_size = end_points[f'{prefix}pred_size']
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        if (
            prefix == 'last_'
            and primary_source == 'detector_policy_adapter'
            and torch.is_tensor(end_points.get(
                'detector_policy_adapter_calibrated_boxes', None
            ))
        ):
            calibrated = end_points[
                'detector_policy_adapter_calibrated_boxes'
            ]
            if calibrated.shape == pred_bbox.shape:
                pred_bbox = calibrated.to(
                    device=pred_bbox.device, dtype=pred_bbox.dtype
                )

        for bid in range(len(positive_map)):
            pmap, gt_boxes = self._get_eval_targets(
                end_points, positive_map, gt_bboxes, bid
            )
            num_obj = gt_boxes.shape[0]

            base_scores = (
                sem_scores[bid].unsqueeze(0)
                * pmap.unsqueeze(1)
            ).sum(-1)
            extra_source_scores = {}
            contrastive_scores = self._contrastive_base_scores(
                end_points, prefix, bid, pmap
            )
            if contrastive_scores is not None:
                extra_source_scores['contrastive_base'] = contrastive_scores
            scores = base_scores
            if primary_source == 'selector_choice' and num_obj > 0:
                scores = self._selector_choice_primary_scores(
                    end_points, bid, num_obj, base_scores,
                    extra_source_scores=extra_source_scores,
                )
            elif primary_source == 'selector_choice_hybrid' and num_obj > 0:
                scores = self._selector_choice_hybrid_primary_scores(
                    end_points, bid, num_obj, base_scores,
                    extra_source_scores=extra_source_scores,
                )
            elif (
                primary_source == 'selector_choice_quality_override'
                and num_obj > 0
            ):
                scores = self._selector_choice_quality_override_primary_scores(
                    end_points, bid, num_obj, base_scores,
                    extra_source_scores=extra_source_scores,
                )
            elif primary_source == 'selector_pool' and num_obj > 0:
                scores = self._selector_pool_primary_scores(
                    end_points, bid, num_obj, base_scores
                )
            elif primary_source != 'base' and num_obj > 0:
                scores = self._single_source_scores(
                    end_points, primary_source, bid, num_obj,
                    base_scores=base_scores,
                )

            top = scores.argsort(1, True)[:, :10]
            ious = self._aligned_topk_ious(gt_boxes, pred_bbox[bid], top)
            if prefix == 'last_':
                spacy_buckets = [
                    self._spacy_augmentation_bucket(end_points, bid),
                    self._spacy_augmentation_profile_bucket(end_points, bid),
                ]
                self._record_decomposition(
                    self._decomposition_status(end_points, bid),
                    prefix,
                    ious,
                )
                self._record_spacy_augmentation(
                    spacy_buckets[0],
                    prefix,
                    ious,
                )
                self._record_spacy_augmentation(
                    spacy_buckets[1],
                    prefix,
                    ious,
                )
                if bool(end_points.get('eval_report_spacy_source_scores', False)):
                    for source in self._requested_spacy_source_score_names(
                        end_points
                    ):
                        source_scores = self._spacy_source_audit_scores(
                            end_points, source, bid, num_obj, base_scores,
                            contrastive_scores,
                        )
                        for bucket in spacy_buckets:
                            self._record_spacy_source_score(
                                bucket, source, prefix, source_scores,
                                gt_boxes, pred_bbox[bid],
                            )
                if bool(end_points.get('eval_report_diagnostic_scores', False)):
                    diag_focus = self._diagnostic_focus()
                    detector_logit_topk_focus = (
                        diag_focus == 'detector_logit_topk_overlap'
                    )
                    detector_combo_focus = (
                        diag_focus == 'detector_class_logit_top2_combo'
                    )
                    detector_proposal_scene_focus = (
                        diag_focus == 'detector_proposal_scene_overlap'
                    )
                    detector_local_agreement_focus = (
                        diag_focus == 'detector_local_agreement'
                    )
                    detector_conf_filter_focus = (
                        diag_focus == 'detector_conf_filter'
                    )
                    detector_conf_combo_focus = (
                        diag_focus == 'detector_conf_combo'
                    )
                    detector_quality_margin_guard_focus = (
                        diag_focus == 'detector_quality_margin_guard'
                    )
                    detector_conf_logit_combo_focus = (
                        diag_focus == 'detector_conf_logit_combo'
                    )
                    detector_count_gate_combo_focus = (
                        diag_focus == 'detector_count_gate_combo'
                    )
                    detector_count_boost_combo_focus = (
                        diag_focus == 'detector_count_boost_combo'
                    )
                    detector_count_boost_le2_combo_focus = (
                        diag_focus == 'detector_count_boost_le2_combo'
                    )
                    detector_count_boost_fine_combo_focus = (
                        diag_focus == 'detector_count_boost_fine_combo'
                    )
                    detector_count_topswitch_boost_combo_focus = (
                        diag_focus == 'detector_count_topswitch_boost_combo'
                    )
                    detector_count_switchdelta_boost_combo_focus = (
                        diag_focus == 'detector_count_switchdelta_boost_combo'
                    )
                    detector_jointtight_switch_combo_focus = (
                        diag_focus == 'detector_jointtight_switch_combo'
                    )
                    detector_coarsejoint_switch_combo_focus = (
                        diag_focus == 'detector_coarsejoint_switch_combo'
                    )
                    detector_countsplit_combo_focus = (
                        diag_focus == 'detector_countsplit_combo'
                    )
                    detector_strongcoarse_joint_switch_combo_focus = (
                        diag_focus
                        == 'detector_strongcoarse_joint_switch_combo'
                    )
                    detector_triswitch_combo_focus = (
                        diag_focus == 'detector_triswitch_combo'
                    )
                    detector_text_context_split_combo_focus = (
                        diag_focus == 'detector_text_context_split_combo'
                    )
                    detector_decomp_status_boost_combo_focus = (
                        diag_focus == 'detector_decomp_status_boost_combo'
                    )
                    query_semantic_scene_focus = (
                        diag_focus == 'query_semantic_scene_overlap'
                    )
                    detector_focused = (
                        detector_logit_topk_focus
                        or detector_combo_focus
                        or detector_proposal_scene_focus
                        or detector_local_agreement_focus
                        or detector_conf_filter_focus
                        or detector_conf_combo_focus
                        or detector_quality_margin_guard_focus
                        or detector_conf_logit_combo_focus
                        or detector_count_gate_combo_focus
                        or detector_count_boost_combo_focus
                        or detector_count_boost_le2_combo_focus
                        or detector_count_boost_fine_combo_focus
                        or detector_count_topswitch_boost_combo_focus
                        or detector_count_switchdelta_boost_combo_focus
                        or detector_jointtight_switch_combo_focus
                        or detector_coarsejoint_switch_combo_focus
                        or detector_countsplit_combo_focus
                        or detector_strongcoarse_joint_switch_combo_focus
                        or detector_triswitch_combo_focus
                        or detector_text_context_split_combo_focus
                        or detector_decomp_status_boost_combo_focus
                        or query_semantic_scene_focus
                    )
                    base_scores = (
                        sem_scores[bid].unsqueeze(0) * pmap.unsqueeze(1)
                    ).sum(-1)
                    diagnostic_ious = {}
                    diagnostic_scores = self._candidate_source_scores(
                        end_points, bid, num_obj, base_scores
                    )
                    diagnostic_ious['base'] = self._record_diagnostic(
                        'base', prefix, base_scores, gt_boxes, pred_bbox[bid]
                    )
                    if contrastive_scores is not None:
                        diagnostic_scores['contrastive_base'] = (
                            contrastive_scores
                        )
                        diagnostic_ious['contrastive_base'] = (
                            self._record_diagnostic(
                                'contrastive_base', prefix,
                                contrastive_scores, gt_boxes, pred_bbox[bid]
                            )
                        )
                    for source in [
                        'structured', 'quality', 'fused', 'acd', 'selector'
                    ]:
                        key = {
                            'structured': 'structured_scores',
                            'quality': 'pred_iou',
                            'fused': 'fused_scores',
                            'acd': 'acd_final_scores',
                            'selector': 'selector_scores',
                        }[source]
                        if key in end_points:
                            diag_scores = self._single_source_scores(
                                end_points, source, bid, num_obj,
                                base_scores=base_scores,
                            )
                            diagnostic_scores[source] = diag_scores
                            diagnostic_ious[source] = self._record_diagnostic(
                                source, prefix, diag_scores, gt_boxes, pred_bbox[bid]
                            )
                    if not detector_focused:
                        target_semantic_scores = self._target_semantic_scores(
                            end_points, prefix, bid, num_obj, base_scores,
                            sem_scores,
                        )
                        if target_semantic_scores is not None:
                            diagnostic_scores['target_semantic'] = (
                                target_semantic_scores
                            )
                            diagnostic_ious['target_semantic'] = (
                                self._record_diagnostic(
                                    'target_semantic', prefix,
                                    target_semantic_scores, gt_boxes,
                                    pred_bbox[bid]
                                )
                            )
                    detector_topks = (2, 3, 5)
                    if detector_logit_topk_focus:
                        detector_topks = (2, 3, 5)
                    target_detector_sources = (
                        self._target_detector_overlap_score_sources(
                            end_points, bid, num_obj, base_scores,
                            pred_bbox[bid], topks=detector_topks,
                        )
                    )
                    for detector_source, detector_scores in (
                        target_detector_sources.items()
                    ):
                        if (
                            detector_logit_topk_focus
                            and not detector_source.startswith(
                                'target_detector_logit'
                            )
                        ):
                            continue
                        if (
                            (
                                detector_conf_filter_focus
                                or detector_conf_combo_focus
                                or detector_quality_margin_guard_focus
                                or detector_conf_logit_combo_focus
                                or detector_count_gate_combo_focus
                                or detector_count_boost_combo_focus
                                or detector_count_boost_le2_combo_focus
                                or detector_count_boost_fine_combo_focus
                                or detector_count_topswitch_boost_combo_focus
                                or detector_count_switchdelta_boost_combo_focus
                                or detector_jointtight_switch_combo_focus
                                or detector_coarsejoint_switch_combo_focus
                                or detector_countsplit_combo_focus
                                or detector_strongcoarse_joint_switch_combo_focus
                                or detector_triswitch_combo_focus
                                or detector_text_context_split_combo_focus
                                or detector_decomp_status_boost_combo_focus
                            )
                            and detector_source
                            != 'target_detector_class_overlap'
                            and detector_source
                            != 'target_detector_logit_top2_overlap'
                            and not detector_source.startswith(
                                'target_detector_class_conf_'
                            )
                        ):
                            continue
                        diagnostic_scores[detector_source] = detector_scores
                        diagnostic_ious[detector_source] = (
                            self._record_diagnostic(
                                detector_source, prefix, detector_scores,
                                gt_boxes, pred_bbox[bid]
                            )
                        )
                    if (
                        detector_combo_focus
                        and 'target_detector_class_overlap' in diagnostic_scores
                        and 'target_detector_logit_top2_overlap' in diagnostic_scores
                    ):
                        class_scores = diagnostic_scores[
                            'target_detector_class_overlap'
                        ]
                        top2_scores = diagnostic_scores[
                            'target_detector_logit_top2_overlap'
                        ]
                        combo_scores = torch.maximum(class_scores, top2_scores)
                        combo_source = (
                            'target_detector_class_logit_top2_max'
                        )
                        diagnostic_scores[combo_source] = combo_scores
                        diagnostic_ious[combo_source] = (
                            self._record_diagnostic(
                                combo_source, prefix, combo_scores,
                                gt_boxes, pred_bbox[bid]
                            )
                        )
                    if (
                        detector_proposal_scene_focus
                        and 'target_detector_class_overlap' in diagnostic_scores
                    ):
                        proposal_scene = (
                            self
                            ._proposal_scene_overlap_scores_from_anchor_scores(
                                'target_detector_class_'
                                'proposal_scene_top5_overlap',
                                base_scores,
                                pred_bbox[bid],
                                diagnostic_scores[
                                    'target_detector_class_overlap'
                                ],
                                anchor_rank_scores=diagnostic_scores.get(
                                    'quality', None
                                ),
                                candidate_k=5,
                            )
                        )
                        if proposal_scene is not None:
                            scene_source, scene_scores = proposal_scene
                            diagnostic_scores[scene_source] = scene_scores
                            diagnostic_ious[scene_source] = (
                                self._record_diagnostic(
                                    scene_source, prefix, scene_scores,
                                    gt_boxes, pred_bbox[bid]
                                )
                            )
                    if (
                        query_semantic_scene_focus
                        and 'quality' in diagnostic_scores
                    ):
                        target_query_semantic_scores = (
                            self._target_semantic_scores(
                                end_points, prefix, bid, num_obj,
                                base_scores, sem_scores,
                            )
                        )
                        if target_query_semantic_scores is not None:
                            proposal_scene = (
                                self
                                ._proposal_scene_overlap_scores_from_anchor_scores(
                                    'target_query_semantic_'
                                    'proposal_scene_top5_overlap',
                                    base_scores,
                                    pred_bbox[bid],
                                    target_query_semantic_scores,
                                    anchor_rank_scores=diagnostic_scores[
                                        'quality'
                                    ],
                                    candidate_k=5,
                                    min_anchor_score=0.05,
                                )
                            )
                            if proposal_scene is not None:
                                scene_source, scene_scores = proposal_scene
                                diagnostic_scores[scene_source] = (
                                    scene_scores
                                )
                                diagnostic_ious[scene_source] = (
                                    self._record_diagnostic(
                                        scene_source, prefix, scene_scores,
                                        gt_boxes, pred_bbox[bid]
                                    )
                                )
                    if not detector_focused:
                        target_scene_scores = (
                            self._target_scene_class_overlap_scores(
                                end_points, bid, num_obj, base_scores,
                                pred_bbox[bid],
                            )
                        )
                        if target_scene_scores is not None:
                            diagnostic_scores['target_scene_class_overlap'] = (
                                target_scene_scores
                            )
                            diagnostic_ious['target_scene_class_overlap'] = (
                                self._record_diagnostic(
                                    'target_scene_class_overlap', prefix,
                                    target_scene_scores, gt_boxes,
                                    pred_bbox[bid]
                                )
                            )
                    if 'quality' in diagnostic_scores:
                        for prior_source in (
                            'target_semantic',
                            'target_scene_class_overlap',
                            'target_detector_class_overlap',
                            'target_detector_logit_overlap',
                            'target_detector_logit_top2_overlap',
                            'target_detector_logit_top3_overlap',
                            'target_detector_logit_top5_overlap',
                            'target_detector_class_logit_top2_max',
                            'target_detector_class_conf_gt0p2_overlap',
                            'target_detector_class_conf_gt0p3_overlap',
                            'target_detector_class_conf_gt0p4_overlap',
                            'target_detector_class_conf_gt0p5_overlap',
                            'target_detector_class_quality_'
                            'proposal_scene_top5_overlap',
                            'target_detector_class_'
                            'proposal_scene_top5_overlap',
                            'target_query_semantic_'
                            'proposal_scene_top5_overlap',
                        ):
                            prior_scores = diagnostic_scores.get(
                                prior_source, None
                            )
                            if prior_scores is None:
                                continue
                            for alpha, suffix in (
                                (0.10, 'a0p1'),
                                (0.15, 'a0p15'),
                                (0.20, 'a0p2'),
                                (0.25, 'a0p25'),
                                (0.35, 'a0p35'),
                                (0.5, 'a0p5'),
                                (0.75, 'a0p75'),
                                (1.0, 'a1'),
                                (1.5, 'a1p5'),
                                (2.0, 'a2'),
                            ):
                                blend_source = (
                                    'quality_'
                                    f'{prior_source}_blend_{suffix}'
                                )
                                blend_scores = (
                                    diagnostic_scores['quality']
                                    + alpha * prior_scores
                                )
                                diagnostic_scores[blend_source] = blend_scores
                                diagnostic_ious[blend_source] = (
                                    self._record_diagnostic(
                                        blend_source, prefix, blend_scores,
                                        gt_boxes, pred_bbox[bid]
                                    )
                                )
                    if (
                        detector_combo_focus
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                        and 'target_detector_logit_top2_overlap' in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores[
                            'target_detector_class_overlap'
                        ]
                        top2_scores = diagnostic_scores[
                            'target_detector_logit_top2_overlap'
                        ]
                        for class_alpha, class_suffix in (
                            (0.25, 'c0p25'),
                            (0.35, 'c0p35'),
                            (0.5, 'c0p5'),
                            (0.75, 'c0p75'),
                        ):
                            for logit_alpha, logit_suffix in (
                                (0.10, 'l0p1'),
                                (0.15, 'l0p15'),
                                (0.20, 'l0p2'),
                                (0.25, 'l0p25'),
                                (0.35, 'l0p35'),
                            ):
                                combo_source = (
                                    'quality_target_detector_'
                                    'class_logit_top2_sum_'
                                    f'{class_suffix}_{logit_suffix}'
                                )
                                combo_scores = (
                                    quality_scores
                                    + class_alpha * class_scores
                                    + logit_alpha * top2_scores
                                )
                                diagnostic_scores[combo_source] = combo_scores
                                diagnostic_ious[combo_source] = (
                                    self._record_diagnostic(
                                        combo_source, prefix, combo_scores,
                                        gt_boxes, pred_bbox[bid]
                                    )
                                )
                    if (
                        detector_conf_combo_focus
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                        and 'target_detector_class_conf_gt0p3_overlap'
                        in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores.get(
                            'target_detector_class_overlap',
                            torch.zeros_like(quality_scores),
                        )
                        conf_scores = diagnostic_scores.get(
                            'target_detector_class_conf_gt0p3_overlap',
                            torch.zeros_like(quality_scores),
                        )
                        for class_alpha, class_suffix in (
                            (0.15, 'c0p15'),
                            (0.25, 'c0p25'),
                            (0.35, 'c0p35'),
                        ):
                            for conf_alpha, conf_suffix in (
                                (0.15, 'f0p15'),
                                (0.25, 'f0p25'),
                            ):
                                combo_source = (
                                    'quality_target_detector_'
                                    'class_conf_gt0p3_sum_'
                                    f'{class_suffix}_{conf_suffix}'
                                )
                                combo_scores = (
                                    quality_scores
                                    + class_alpha * class_scores
                                    + conf_alpha * conf_scores
                                )
                                diagnostic_scores[combo_source] = (
                                    combo_scores
                                )
                                diagnostic_ious[combo_source] = (
                                    self._record_diagnostic(
                                        combo_source, prefix, combo_scores,
                                        gt_boxes, pred_bbox[bid]
                                    )
                                )
                    if (
                        detector_conf_logit_combo_focus
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                        and 'target_detector_class_conf_gt0p3_overlap'
                        in diagnostic_scores
                        and 'target_detector_logit_top2_overlap'
                        in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores.get(
                            'target_detector_class_overlap',
                            torch.zeros_like(quality_scores),
                        )
                        conf_scores = diagnostic_scores.get(
                            'target_detector_class_conf_gt0p3_overlap',
                            torch.zeros_like(quality_scores),
                        )
                        logit_top2_scores = diagnostic_scores[
                            'target_detector_logit_top2_overlap'
                        ]
                        for class_alpha, class_suffix in (
                            (0.15, 'c0p15'),
                            (0.25, 'c0p25'),
                        ):
                            for conf_alpha, conf_suffix in (
                                (0.15, 'f0p15'),
                                (0.25, 'f0p25'),
                            ):
                                for logit_alpha, logit_suffix in (
                                    (0.05, 'l0p05'),
                                    (0.10, 'l0p1'),
                                    (0.15, 'l0p15'),
                                ):
                                    combo_source = (
                                        'quality_target_detector_'
                                        'class_conf_gt0p3_logit_top2_sum_'
                                        f'{class_suffix}_{conf_suffix}_'
                                        f'{logit_suffix}'
                                    )
                                    combo_scores = (
                                        quality_scores
                                        + class_alpha * class_scores
                                        + conf_alpha * conf_scores
                                        + logit_alpha * logit_top2_scores
                                    )
                                    diagnostic_scores[combo_source] = (
                                        combo_scores
                                    )
                                    diagnostic_ious[combo_source] = (
                                        self._record_diagnostic(
                                            combo_source, prefix,
                                            combo_scores, gt_boxes,
                                            pred_bbox[bid]
                                        )
                                    )
                    if (
                        detector_count_boost_combo_focus
                        and 'quality' in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores.get(
                            'target_detector_class_overlap',
                            torch.zeros_like(quality_scores),
                        )
                        conf_scores = diagnostic_scores.get(
                            'target_detector_class_conf_gt0p3_overlap',
                            torch.zeros_like(quality_scores),
                        )
                        det_count = self._target_detector_class_match_count(
                            end_points, bid, min_conf=0.30
                        )
                        for count_limit, count_suffix in (
                            (1, 'le1'),
                            (2, 'le2'),
                            (4, 'le4'),
                        ):
                            active_weight = quality_scores.new_tensor(
                                1.0
                                if (
                                    det_count is not None
                                    and det_count <= count_limit
                                )
                                else 0.0
                            )
                            for class_alpha, class_suffix in (
                                (0.15, 'c0p15'),
                                (0.25, 'c0p25'),
                            ):
                                conf_alpha = 0.25
                                base_combo_scores = (
                                    quality_scores
                                    + class_alpha * class_scores
                                    + conf_alpha * conf_scores
                                )
                                for boost_alpha, boost_suffix in (
                                    (0.05, 'xc0p05'),
                                    (0.10, 'xc0p1'),
                                    (0.15, 'xc0p15'),
                                ):
                                    combo_source = (
                                        'quality_target_detector_'
                                        'class_conf_gt0p3_detcount_'
                                        f'{count_suffix}_boost_'
                                        f'{class_suffix}_f0p25_'
                                        f'{boost_suffix}'
                                    )
                                    combo_scores = (
                                        base_combo_scores
                                        + active_weight
                                        * boost_alpha
                                        * class_scores
                                    )
                                    diagnostic_scores[combo_source] = (
                                        combo_scores
                                    )
                                    diagnostic_ious[combo_source] = (
                                        self._record_diagnostic(
                                            combo_source, prefix,
                                            combo_scores, gt_boxes,
                                            pred_bbox[bid]
                                        )
                                    )
                    if (
                        detector_count_boost_le2_combo_focus
                        and 'quality' in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores.get(
                            'target_detector_class_overlap',
                            torch.zeros_like(quality_scores),
                        )
                        conf_scores = diagnostic_scores.get(
                            'target_detector_class_conf_gt0p3_overlap',
                            torch.zeros_like(quality_scores),
                        )
                        det_count = self._target_detector_class_match_count(
                            end_points, bid, min_conf=0.30
                        )
                        active_weight = quality_scores.new_tensor(
                            1.0
                            if det_count is not None and det_count <= 2
                            else 0.0
                        )
                        base_combo_scores = (
                            quality_scores
                            + 0.15 * class_scores
                            + 0.25 * conf_scores
                        )
                        for boost_alpha, boost_suffix in (
                            (0.20, 'xc0p2'),
                            (0.25, 'xc0p25'),
                            (0.35, 'xc0p35'),
                            (0.50, 'xc0p5'),
                        ):
                            combo_source = (
                                'quality_target_detector_'
                                'class_conf_gt0p3_detcount_le2_boost_'
                                'c0p15_f0p25_'
                                f'{boost_suffix}'
                            )
                            combo_scores = (
                                base_combo_scores
                                + active_weight
                                * boost_alpha
                                * class_scores
                            )
                            diagnostic_scores[combo_source] = combo_scores
                            diagnostic_ious[combo_source] = (
                                self._record_diagnostic(
                                    combo_source, prefix,
                                    combo_scores, gt_boxes,
                                    pred_bbox[bid]
                                )
                            )
                    if (
                        detector_count_boost_fine_combo_focus
                        and 'quality' in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores.get(
                            'target_detector_class_overlap',
                            torch.zeros_like(quality_scores),
                        )
                        conf_scores = diagnostic_scores.get(
                            'target_detector_class_conf_gt0p3_overlap',
                            torch.zeros_like(quality_scores),
                        )
                        det_count = self._target_detector_class_match_count(
                            end_points, bid, min_conf=0.30
                        )
                        for count_limit, count_suffix in (
                            (2, 'le2'),
                            (3, 'le3'),
                        ):
                            active_weight = quality_scores.new_tensor(
                                1.0
                                if (
                                    det_count is not None
                                    and det_count <= count_limit
                                )
                                else 0.0
                            )
                            for class_alpha, class_suffix in (
                                (0.10, 'c0p1'),
                                (0.12, 'c0p12'),
                                (0.15, 'c0p15'),
                                (0.18, 'c0p18'),
                            ):
                                conf_alpha = 0.25
                                base_combo_scores = (
                                    quality_scores
                                    + class_alpha * class_scores
                                    + conf_alpha * conf_scores
                                )
                                for boost_alpha, boost_suffix in (
                                    (0.08, 'xc0p08'),
                                    (0.12, 'xc0p12'),
                                    (0.15, 'xc0p15'),
                                    (0.18, 'xc0p18'),
                                ):
                                    combo_source = (
                                        'quality_target_detector_'
                                        'class_conf_gt0p3_detcount_'
                                        f'{count_suffix}_boostfine_'
                                        f'{class_suffix}_f0p25_'
                                        f'{boost_suffix}'
                                    )
                                    combo_scores = (
                                        base_combo_scores
                                        + active_weight
                                        * boost_alpha
                                        * class_scores
                                    )
                                    diagnostic_scores[combo_source] = (
                                        combo_scores
                                    )
                                    diagnostic_ious[combo_source] = (
                                        self._record_diagnostic(
                                            combo_source, prefix,
                                            combo_scores, gt_boxes,
                                            pred_bbox[bid]
                                        )
                                    )
                    if (
                        detector_count_topswitch_boost_combo_focus
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                        and 'target_detector_class_conf_gt0p3_overlap'
                        in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores[
                            'target_detector_class_overlap'
                        ]
                        conf_scores = diagnostic_scores[
                            'target_detector_class_conf_gt0p3_overlap'
                        ]
                        det_count = self._target_detector_class_match_count(
                            end_points, bid, min_conf=0.30
                        )
                        if det_count is not None:
                            active_count = det_count <= 2
                            base_combo_scores = (
                                quality_scores
                                + 0.15 * class_scores
                                + 0.25 * conf_scores
                            )
                            detector_support = torch.maximum(
                                class_scores, conf_scores
                            )
                            base_top = base_combo_scores.argmax(dim=1)
                            for weak_max, weak_suffix in (
                                (0.05, 'w0p05'),
                                (0.10, 'w0p1'),
                            ):
                                for strong_min, strong_suffix in (
                                    (0.25, 's0p25'),
                                    (0.50, 's0p5'),
                                ):
                                    for boost_alpha, boost_suffix in (
                                        (0.10, 'xc0p1'),
                                        (0.15, 'xc0p15'),
                                        (0.20, 'xc0p2'),
                                    ):
                                        boosted_scores = (
                                            base_combo_scores
                                            + boost_alpha * class_scores
                                        )
                                        boosted_top = boosted_scores.argmax(
                                            dim=1
                                        )
                                        base_support = torch.gather(
                                            detector_support,
                                            1,
                                            base_top.unsqueeze(1),
                                        ).squeeze(1)
                                        boosted_support = torch.gather(
                                            detector_support,
                                            1,
                                            boosted_top.unsqueeze(1),
                                        ).squeeze(1)
                                        active = (
                                            active_count
                                            and bool(
                                                (
                                                    (
                                                        boosted_top
                                                        != base_top
                                                    )
                                                    & (
                                                        base_support
                                                        <= weak_max
                                                    )
                                                    & (
                                                        boosted_support
                                                        >= strong_min
                                                    )
                                                )
                                                .any()
                                                .detach()
                                                .cpu()
                                                .item()
                                            )
                                        )
                                        combo_source = (
                                            'quality_target_detector_'
                                            'class_conf_gt0p3_topswitch_le2_'
                                            f'{weak_suffix}_{strong_suffix}_'
                                            'c0p15_f0p25_'
                                            f'{boost_suffix}'
                                        )
                                        combo_scores = (
                                            boosted_scores
                                            if active
                                            else base_combo_scores
                                        )
                                        diagnostic_scores[combo_source] = (
                                            combo_scores
                                        )
                                        diagnostic_ious[combo_source] = (
                                            self._record_diagnostic(
                                                combo_source, prefix,
                                                combo_scores, gt_boxes,
                                                pred_bbox[bid]
                                            )
                                        )
                    if (
                        detector_decomp_status_boost_combo_focus
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                        and 'target_detector_class_conf_gt0p3_overlap'
                        in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores[
                            'target_detector_class_overlap'
                        ]
                        conf_scores = diagnostic_scores[
                            'target_detector_class_conf_gt0p3_overlap'
                        ]
                        status = self._decomposition_status(end_points, bid)
                        status_active = {
                            'repaired': status == 'repaired_structured',
                            'weak': status == (
                                'weak_generic_target_recovered'
                            ),
                            'repairweak': status in (
                                'repaired_structured',
                                'weak_generic_target_recovered',
                            ),
                        }
                        for status_suffix, active in (
                            ('repaired', status_active['repaired']),
                            ('weak', status_active['weak']),
                            ('repairweak', status_active['repairweak']),
                        ):
                            active_weight = quality_scores.new_tensor(
                                1.0 if active else 0.0
                            )
                            for class_alpha, class_suffix in (
                                (0.15, 'c0p15'),
                                (0.25, 'c0p25'),
                            ):
                                conf_alpha = 0.25
                                base_combo_scores = (
                                    quality_scores
                                    + class_alpha * class_scores
                                    + conf_alpha * conf_scores
                                )
                                for boost_alpha, boost_suffix in (
                                    (0.10, 'xc0p1'),
                                    (0.25, 'xc0p25'),
                                    (0.50, 'xc0p5'),
                                    (0.75, 'xc0p75'),
                                ):
                                    combo_source = (
                                        'quality_target_detector_'
                                        'class_conf_gt0p3_decomp_'
                                        f'{status_suffix}_boost_'
                                        f'{class_suffix}_f0p25_'
                                        f'{boost_suffix}'
                                    )
                                    combo_scores = (
                                        base_combo_scores
                                        + active_weight
                                        * boost_alpha
                                        * class_scores
                                    )
                                    diagnostic_scores[combo_source] = (
                                        combo_scores
                                    )
                                    diagnostic_ious[combo_source] = (
                                        self._record_diagnostic(
                                            combo_source, prefix,
                                            combo_scores, gt_boxes,
                                            pred_bbox[bid]
                                        )
                                    )
                    if (
                        detector_count_switchdelta_boost_combo_focus
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                        and 'target_detector_class_conf_gt0p3_overlap'
                        in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores[
                            'target_detector_class_overlap'
                        ]
                        conf_scores = diagnostic_scores[
                            'target_detector_class_conf_gt0p3_overlap'
                        ]
                        det_count = self._target_detector_class_match_count(
                            end_points, bid, min_conf=0.30
                        )
                        if det_count is not None:
                            active_count = det_count <= 2
                            detector_support = torch.maximum(
                                class_scores, conf_scores
                            )
                            for class_alpha, class_suffix in (
                                (0.15, 'c0p15'),
                                (0.25, 'c0p25'),
                            ):
                                conf_alpha = 0.25
                                base_combo_scores = (
                                    quality_scores
                                    + class_alpha * class_scores
                                    + conf_alpha * conf_scores
                                )
                                base_top = base_combo_scores.argmax(dim=1)
                                base_support = torch.gather(
                                    detector_support,
                                    1,
                                    base_top.unsqueeze(1),
                                ).squeeze(1)
                                for support_delta, delta_suffix in (
                                    (0.00, 'd0'),
                                    (0.10, 'd0p1'),
                                    (0.25, 'd0p25'),
                                    (0.50, 'd0p5'),
                                ):
                                    for boost_alpha, boost_suffix in (
                                        (0.10, 'xc0p1'),
                                        (0.15, 'xc0p15'),
                                        (0.20, 'xc0p2'),
                                    ):
                                        boosted_scores = (
                                            base_combo_scores
                                            + boost_alpha * class_scores
                                        )
                                        boosted_top = boosted_scores.argmax(
                                            dim=1
                                        )
                                        boosted_support = torch.gather(
                                            detector_support,
                                            1,
                                            boosted_top.unsqueeze(1),
                                        ).squeeze(1)
                                        support_improvement = (
                                            boosted_support - base_support
                                        )
                                        allow_boost = (
                                            (boosted_top == base_top)
                                            | (
                                                support_improvement
                                                >= support_delta
                                            )
                                        )
                                        active = (
                                            active_count
                                            and bool(
                                                allow_boost.any()
                                                .detach()
                                                .cpu()
                                                .item()
                                            )
                                        )
                                        active_rows = allow_boost.float()
                                        if not active:
                                            active_rows = (
                                                active_rows * 0.0
                                            )
                                        combo_source = (
                                            'quality_target_detector_'
                                            'class_conf_gt0p3_'
                                            'switchdelta_le2_'
                                            f'{delta_suffix}_'
                                            f'{class_suffix}_f0p25_'
                                            f'{boost_suffix}'
                                        )
                                        combo_scores = (
                                            base_combo_scores
                                            + active_rows.unsqueeze(1)
                                            * boost_alpha
                                            * class_scores
                                        )
                                        diagnostic_scores[combo_source] = (
                                            combo_scores
                                        )
                                        diagnostic_ious[combo_source] = (
                                            self._record_diagnostic(
                                                combo_source, prefix,
                                                combo_scores, gt_boxes,
                                                pred_bbox[bid]
                                            )
                                        )
                    if (
                        detector_jointtight_switch_combo_focus
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                        and 'target_detector_class_conf_gt0p3_overlap'
                        in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores[
                            'target_detector_class_overlap'
                        ]
                        conf_scores = diagnostic_scores[
                            'target_detector_class_conf_gt0p3_overlap'
                        ]
                        det_count = self._target_detector_class_match_count(
                            end_points, bid, min_conf=0.30
                        )
                        if det_count is not None:
                            active_count = det_count <= 2
                            detector_support = torch.maximum(
                                class_scores, conf_scores
                            )
                            joint_scores = (
                                quality_scores
                                + 0.12 * class_scores
                                + 0.25 * conf_scores
                            )
                            if active_count:
                                joint_scores = (
                                    joint_scores
                                    + 0.18 * class_scores
                                )
                            tight_base_scores = (
                                quality_scores
                                + 0.15 * class_scores
                                + 0.25 * conf_scores
                            )
                            tight_base_top = tight_base_scores.argmax(dim=1)
                            tight_base_support = torch.gather(
                                detector_support,
                                1,
                                tight_base_top.unsqueeze(1),
                            ).squeeze(1)
                            tight_boosted_scores = (
                                tight_base_scores
                                + 0.20 * class_scores
                            )
                            tight_boosted_top = (
                                tight_boosted_scores.argmax(dim=1)
                            )
                            tight_boosted_support = torch.gather(
                                detector_support,
                                1,
                                tight_boosted_top.unsqueeze(1),
                            ).squeeze(1)
                            tight_support_improvement = (
                                tight_boosted_support - tight_base_support
                            )
                            tight_allow_boost = (
                                (tight_boosted_top == tight_base_top)
                                | (tight_support_improvement >= 0.25)
                            )
                            tight_scores = (
                                tight_base_scores
                                + tight_allow_boost.float().unsqueeze(1)
                                * 0.20
                                * class_scores
                            )
                            joint_top = joint_scores.argmax(dim=1)
                            tight_top = tight_scores.argmax(dim=1)
                            joint_support = torch.gather(
                                detector_support,
                                1,
                                joint_top.unsqueeze(1),
                            ).squeeze(1)
                            tight_support = torch.gather(
                                detector_support,
                                1,
                                tight_top.unsqueeze(1),
                            ).squeeze(1)
                            joint_quality = torch.gather(
                                quality_scores,
                                1,
                                joint_top.unsqueeze(1),
                            ).squeeze(1)
                            tight_quality = torch.gather(
                                quality_scores,
                                1,
                                tight_top.unsqueeze(1),
                            ).squeeze(1)
                            quality_gap = joint_quality - tight_quality
                            for support_gain, support_suffix in (
                                (0.00, 'sg0'),
                                (0.10, 'sg0p1'),
                                (0.25, 'sg0p25'),
                                (0.50, 'sg0p5'),
                            ):
                                for quality_margin, quality_suffix in (
                                    (0.10, 'qg0p1'),
                                    (0.25, 'qg0p25'),
                                    (0.50, 'qg0p5'),
                                    (1.00, 'qg1'),
                                ):
                                    use_tight = (
                                        tight_support
                                        >= joint_support + support_gain
                                    ) & (quality_gap <= quality_margin)
                                    if not active_count:
                                        use_tight = (
                                            use_tight
                                            & torch.zeros_like(
                                                use_tight, dtype=torch.bool
                                            )
                                        )
                                    combo_source = (
                                        'quality_target_detector_'
                                        'class_conf_gt0p3_'
                                        'jointtight_le2_'
                                        f'{support_suffix}_'
                                        f'{quality_suffix}'
                                    )
                                    combo_scores = torch.where(
                                        use_tight.unsqueeze(1),
                                        tight_scores,
                                        joint_scores,
                                    )
                                    diagnostic_scores[combo_source] = (
                                        combo_scores
                                    )
                                    diagnostic_ious[combo_source] = (
                                        self._record_diagnostic(
                                            combo_source, prefix,
                                            combo_scores, gt_boxes,
                                            pred_bbox[bid]
                                            )
                                        )
                    if (
                        detector_coarsejoint_switch_combo_focus
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                        and 'target_detector_class_conf_gt0p3_overlap'
                        in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores[
                            'target_detector_class_overlap'
                        ]
                        conf_scores = diagnostic_scores[
                            'target_detector_class_conf_gt0p3_overlap'
                        ]
                        det_count = self._target_detector_class_match_count(
                            end_points, bid, min_conf=0.30
                        )
                        if det_count is not None:
                            active_count = det_count <= 2
                            detector_support = torch.maximum(
                                class_scores, conf_scores
                            )
                            joint_scores = (
                                quality_scores
                                + 0.12 * class_scores
                                + 0.25 * conf_scores
                            )
                            if active_count:
                                joint_scores = (
                                    joint_scores
                                    + 0.18 * class_scores
                                )
                            coarse_scores = quality_scores.clone()
                            if active_count:
                                coarse_scores = (
                                    coarse_scores
                                    + 0.25 * class_scores
                                    + 0.25 * conf_scores
                                )
                            joint_top = joint_scores.argmax(dim=1)
                            coarse_top = coarse_scores.argmax(dim=1)
                            joint_support = torch.gather(
                                detector_support,
                                1,
                                joint_top.unsqueeze(1),
                            ).squeeze(1)
                            coarse_support = torch.gather(
                                detector_support,
                                1,
                                coarse_top.unsqueeze(1),
                            ).squeeze(1)
                            joint_quality = torch.gather(
                                quality_scores,
                                1,
                                joint_top.unsqueeze(1),
                            ).squeeze(1)
                            coarse_quality = torch.gather(
                                quality_scores,
                                1,
                                coarse_top.unsqueeze(1),
                            ).squeeze(1)
                            quality_gap = joint_quality - coarse_quality
                            for support_gain, support_suffix in (
                                (0.00, 'sg0'),
                                (0.10, 'sg0p1'),
                                (0.25, 'sg0p25'),
                                (0.50, 'sg0p5'),
                            ):
                                for quality_margin, quality_suffix in (
                                    (0.10, 'qg0p1'),
                                    (0.25, 'qg0p25'),
                                    (0.50, 'qg0p5'),
                                    (1.00, 'qg1'),
                                ):
                                    use_coarse = (
                                        coarse_support
                                        >= joint_support + support_gain
                                    ) & (quality_gap <= quality_margin)
                                    if not active_count:
                                        use_coarse = (
                                            use_coarse
                                            & torch.zeros_like(
                                                use_coarse, dtype=torch.bool
                                            )
                                        )
                                    combo_source = (
                                        'quality_target_detector_'
                                        'class_conf_gt0p3_'
                                        'coarsejoint_le2_'
                                        f'{support_suffix}_'
                                        f'{quality_suffix}'
                                    )
                                    combo_scores = torch.where(
                                        use_coarse.unsqueeze(1),
                                        coarse_scores,
                                        joint_scores,
                                    )
                                    diagnostic_scores[combo_source] = (
                                        combo_scores
                                    )
                                    diagnostic_ious[combo_source] = (
                                        self._record_diagnostic(
                                            combo_source, prefix,
                                            combo_scores, gt_boxes,
                                            pred_bbox[bid]
                                        )
                                    )
                    if (
                        detector_countsplit_combo_focus
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                        and 'target_detector_class_conf_gt0p3_overlap'
                        in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores[
                            'target_detector_class_overlap'
                        ]
                        conf_scores = diagnostic_scores[
                            'target_detector_class_conf_gt0p3_overlap'
                        ]
                        det_count = self._target_detector_class_match_count(
                            end_points, bid, min_conf=0.30
                        )
                        if det_count is not None:
                            low_count = quality_scores.new_tensor(
                                1.0 if det_count <= 2 else 0.0
                            )
                            high_count = 1.0 - low_count
                            low_configs = (
                                (0.20, 0.25, 'lc0p2_lf0p25'),
                                (0.25, 0.25, 'lc0p25_lf0p25'),
                                (0.30, 0.25, 'lc0p3_lf0p25'),
                                (0.25, 0.35, 'lc0p25_lf0p35'),
                            )
                            high_configs = (
                                (0.00, 0.00, 'hc0_hf0'),
                                (0.05, 0.10, 'hc0p05_hf0p10'),
                                (0.10, 0.25, 'hc0p1_hf0p25'),
                                (0.12, 0.25, 'hc0p12_hf0p25'),
                                (0.15, 0.25, 'hc0p15_hf0p25'),
                            )
                            for low_class, low_conf, low_suffix in low_configs:
                                for high_class, high_conf, high_suffix in (
                                    high_configs
                                ):
                                    class_weight = (
                                        low_count * low_class
                                        + high_count * high_class
                                    )
                                    conf_weight = (
                                        low_count * low_conf
                                        + high_count * high_conf
                                    )
                                    combo_source = (
                                        'quality_target_detector_'
                                        'class_conf_gt0p3_'
                                        'countsplit_le2_'
                                        f'{low_suffix}_{high_suffix}'
                                    )
                                    combo_scores = (
                                        quality_scores
                                        + class_weight * class_scores
                                        + conf_weight * conf_scores
                                    )
                                    diagnostic_scores[combo_source] = (
                                        combo_scores
                                    )
                                    diagnostic_ious[combo_source] = (
                                        self._record_diagnostic(
                                            combo_source, prefix,
                                            combo_scores, gt_boxes,
                                            pred_bbox[bid]
                                        )
                                    )
                    if (
                        detector_strongcoarse_joint_switch_combo_focus
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                        and 'target_detector_class_conf_gt0p3_overlap'
                        in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores[
                            'target_detector_class_overlap'
                        ]
                        conf_scores = diagnostic_scores[
                            'target_detector_class_conf_gt0p3_overlap'
                        ]
                        det_count = self._target_detector_class_match_count(
                            end_points, bid, min_conf=0.30
                        )
                        if det_count is not None:
                            active_count = det_count <= 2
                            detector_support = torch.maximum(
                                class_scores, conf_scores
                            )
                            joint_scores = (
                                quality_scores
                                + 0.12 * class_scores
                                + 0.25 * conf_scores
                            )
                            if active_count:
                                joint_scores = (
                                    joint_scores
                                    + 0.18 * class_scores
                                )
                            joint_top = joint_scores.argmax(dim=1)
                            joint_support = torch.gather(
                                detector_support,
                                1,
                                joint_top.unsqueeze(1),
                            ).squeeze(1)
                            joint_quality = torch.gather(
                                quality_scores,
                                1,
                                joint_top.unsqueeze(1),
                            ).squeeze(1)
                            for strong_class, strong_conf, strong_suffix in (
                                (0.25, 0.25, 'sc0p25_sf0p25'),
                                (0.30, 0.25, 'sc0p3_sf0p25'),
                                (0.25, 0.35, 'sc0p25_sf0p35'),
                            ):
                                coarse_scores = (
                                    quality_scores
                                    + strong_class * class_scores
                                    + strong_conf * conf_scores
                                )
                                coarse_top = coarse_scores.argmax(dim=1)
                                coarse_support = torch.gather(
                                    detector_support,
                                    1,
                                    coarse_top.unsqueeze(1),
                                ).squeeze(1)
                                coarse_quality = torch.gather(
                                    quality_scores,
                                    1,
                                    coarse_top.unsqueeze(1),
                                ).squeeze(1)
                                quality_gap = (
                                    joint_quality - coarse_quality
                                )
                                for support_gain, support_suffix in (
                                    (0.00, 'sg0'),
                                    (0.10, 'sg0p1'),
                                    (0.25, 'sg0p25'),
                                    (0.50, 'sg0p5'),
                                ):
                                    for quality_margin, quality_suffix in (
                                        (0.25, 'qg0p25'),
                                        (0.50, 'qg0p5'),
                                        (1.00, 'qg1'),
                                    ):
                                        use_coarse = (
                                            coarse_support
                                            >= joint_support + support_gain
                                        ) & (quality_gap <= quality_margin)
                                        if not active_count:
                                            use_coarse = (
                                                use_coarse
                                                & torch.zeros_like(
                                                    use_coarse,
                                                    dtype=torch.bool,
                                                )
                                            )
                                        combo_source = (
                                            'quality_target_detector_'
                                            'class_conf_gt0p3_'
                                            'strongcoarse_joint_le2_'
                                            f'{strong_suffix}_'
                                            f'{support_suffix}_'
                                            f'{quality_suffix}'
                                        )
                                        combo_scores = torch.where(
                                            use_coarse.unsqueeze(1),
                                            coarse_scores,
                                            joint_scores,
                                        )
                                        diagnostic_scores[combo_source] = (
                                            combo_scores
                                        )
                                        diagnostic_ious[combo_source] = (
                                            self._record_diagnostic(
                                                combo_source, prefix,
                                                combo_scores, gt_boxes,
                                                pred_bbox[bid]
                                            )
                                        )
                    if (
                        detector_triswitch_combo_focus
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                        and 'target_detector_class_conf_gt0p3_overlap'
                        in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores[
                            'target_detector_class_overlap'
                        ]
                        conf_scores = diagnostic_scores[
                            'target_detector_class_conf_gt0p3_overlap'
                        ]
                        det_count = self._target_detector_class_match_count(
                            end_points, bid, min_conf=0.30
                        )
                        if det_count is not None:
                            active_count = det_count <= 2
                            detector_support = torch.maximum(
                                class_scores, conf_scores
                            )
                            joint_scores = (
                                quality_scores
                                + 0.12 * class_scores
                                + 0.25 * conf_scores
                            )
                            if active_count:
                                joint_scores = (
                                    joint_scores
                                    + 0.18 * class_scores
                                )

                            tight_base_scores = (
                                quality_scores
                                + 0.15 * class_scores
                                + 0.25 * conf_scores
                            )
                            tight_base_top = tight_base_scores.argmax(dim=1)
                            tight_base_support = torch.gather(
                                detector_support,
                                1,
                                tight_base_top.unsqueeze(1),
                            ).squeeze(1)
                            tight_boosted_scores = (
                                tight_base_scores
                                + 0.20 * class_scores
                            )
                            tight_boosted_top = (
                                tight_boosted_scores.argmax(dim=1)
                            )
                            tight_boosted_support = torch.gather(
                                detector_support,
                                1,
                                tight_boosted_top.unsqueeze(1),
                            ).squeeze(1)
                            tight_support_improvement = (
                                tight_boosted_support - tight_base_support
                            )
                            tight_allow_boost = (
                                (tight_boosted_top == tight_base_top)
                                | (tight_support_improvement >= 0.25)
                            )
                            tight_scores = (
                                tight_base_scores
                                + tight_allow_boost.float().unsqueeze(1)
                                * 0.20
                                * class_scores
                            )

                            joint_top = joint_scores.argmax(dim=1)
                            tight_top = tight_scores.argmax(dim=1)
                            joint_support = torch.gather(
                                detector_support,
                                1,
                                joint_top.unsqueeze(1),
                            ).squeeze(1)
                            tight_support = torch.gather(
                                detector_support,
                                1,
                                tight_top.unsqueeze(1),
                            ).squeeze(1)
                            joint_quality = torch.gather(
                                quality_scores,
                                1,
                                joint_top.unsqueeze(1),
                            ).squeeze(1)
                            tight_quality = torch.gather(
                                quality_scores,
                                1,
                                tight_top.unsqueeze(1),
                            ).squeeze(1)
                            tight_quality_gap = joint_quality - tight_quality
                            for low_class, low_conf, low_suffix in (
                                (0.30, 0.25, 'lc0p3_lf0p25'),
                                (0.25, 0.35, 'lc0p25_lf0p35'),
                            ):
                                coarse_scores = (
                                    quality_scores
                                    + low_class * class_scores
                                    + low_conf * conf_scores
                                )
                                coarse_top = coarse_scores.argmax(dim=1)
                                coarse_support = torch.gather(
                                    detector_support,
                                    1,
                                    coarse_top.unsqueeze(1),
                                ).squeeze(1)
                                coarse_quality = torch.gather(
                                    quality_scores,
                                    1,
                                    coarse_top.unsqueeze(1),
                                ).squeeze(1)
                                for tight_support_gain, tight_support_suffix in (
                                    (0.25, 'tsg0p25'),
                                    (0.50, 'tsg0p5'),
                                ):
                                    for tight_quality_margin, tight_quality_suffix in (
                                        (0.50, 'tqg0p5'),
                                        (1.00, 'tqg1'),
                                    ):
                                        use_tight = (
                                            tight_support
                                            >= (
                                                joint_support
                                                + tight_support_gain
                                            )
                                        ) & (
                                            tight_quality_gap
                                            <= tight_quality_margin
                                        )
                                        if not active_count:
                                            use_tight = (
                                                use_tight
                                                & torch.zeros_like(
                                                    use_tight,
                                                    dtype=torch.bool,
                                                )
                                            )
                                        selected_scores = torch.where(
                                            use_tight.unsqueeze(1),
                                            tight_scores,
                                            joint_scores,
                                        )
                                        selected_support = torch.where(
                                            use_tight,
                                            tight_support,
                                            joint_support,
                                        )
                                        selected_quality = torch.where(
                                            use_tight,
                                            tight_quality,
                                            joint_quality,
                                        )
                                        coarse_quality_gap = (
                                            selected_quality - coarse_quality
                                        )
                                        for coarse_support_gain, coarse_support_suffix in (
                                            (0.00, 'csg0'),
                                            (0.25, 'csg0p25'),
                                            (0.50, 'csg0p5'),
                                        ):
                                            for coarse_quality_margin, coarse_quality_suffix in (
                                                (0.25, 'cqg0p25'),
                                                (0.50, 'cqg0p5'),
                                                (1.00, 'cqg1'),
                                            ):
                                                use_coarse = (
                                                    coarse_support
                                                    >= (
                                                        selected_support
                                                        + coarse_support_gain
                                                    )
                                                ) & (
                                                    coarse_quality_gap
                                                    <= coarse_quality_margin
                                                )
                                                if not active_count:
                                                    use_coarse = (
                                                        use_coarse
                                                        & torch.zeros_like(
                                                            use_coarse,
                                                            dtype=torch.bool,
                                                        )
                                                    )
                                                combo_source = (
                                                    'quality_target_detector_'
                                                    'class_conf_gt0p3_'
                                                    'triswitch_le2_'
                                                    f'{low_suffix}_'
                                                    f'{tight_support_suffix}_'
                                                    f'{tight_quality_suffix}_'
                                                    f'{coarse_support_suffix}_'
                                                    f'{coarse_quality_suffix}'
                                                )
                                                combo_scores = torch.where(
                                                    use_coarse.unsqueeze(1),
                                                    coarse_scores,
                                                    selected_scores,
                                                )
                                                diagnostic_scores[
                                                    combo_source
                                                ] = combo_scores
                                                diagnostic_ious[
                                                    combo_source
                                                ] = self._record_diagnostic(
                                                    combo_source, prefix,
                                                    combo_scores, gt_boxes,
                                                    pred_bbox[bid]
                                                )
                    if (
                        detector_text_context_split_combo_focus
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                        and 'target_detector_class_conf_gt0p3_overlap'
                        in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores[
                            'target_detector_class_overlap'
                        ]
                        conf_scores = diagnostic_scores[
                            'target_detector_class_conf_gt0p3_overlap'
                        ]
                        det_count = self._target_detector_class_match_count(
                            end_points, bid, min_conf=0.30
                        )
                        if det_count is not None:
                            rel_count = self._coverage_float(
                                end_points, 'num_pairs', bid, default=0.0
                            )
                            if rel_count <= 0.0:
                                rel_count = self._span_count(
                                    end_points, 'rel_slots', bid
                                )
                            anchor_count = self._span_count(
                                end_points, 'anchor_slots', bid
                            )
                            relation_active = (
                                rel_count > 0.0 or anchor_count > 0.0
                            )
                            low_count_relation = (
                                relation_active and det_count <= 2
                            )
                            joint_scores = (
                                quality_scores
                                + 0.12 * class_scores
                                + 0.25 * conf_scores
                            )
                            if det_count <= 2:
                                joint_scores = (
                                    joint_scores
                                    + 0.18 * class_scores
                                )
                            context_modes = (
                                ('rel', relation_active),
                                ('relle2', low_count_relation),
                            )
                            context_weights = (
                                (0.05, 0.10, 'c0p05_f0p10'),
                                (0.10, 0.15, 'c0p1_f0p15'),
                                (0.15, 0.25, 'c0p15_f0p25'),
                                (0.20, 0.25, 'c0p2_f0p25'),
                            )
                            for context_suffix, active in context_modes:
                                active_weight = quality_scores.new_tensor(
                                    1.0 if active else 0.0
                                )
                                for (
                                    class_alpha,
                                    conf_alpha,
                                    weight_suffix,
                                ) in context_weights:
                                    combo_source = (
                                        'quality_target_detector_'
                                        'class_conf_gt0p3_'
                                        f'textctx_{context_suffix}_'
                                        f'{weight_suffix}'
                                    )
                                    combo_scores = (
                                        joint_scores
                                        + active_weight * (
                                            class_alpha * class_scores
                                            + conf_alpha * conf_scores
                                        )
                                    )
                                    diagnostic_scores[combo_source] = (
                                        combo_scores
                                    )
                                    diagnostic_ious[combo_source] = (
                                        self._record_diagnostic(
                                            combo_source, prefix,
                                            combo_scores, gt_boxes,
                                            pred_bbox[bid]
                                        )
                                    )
                    if (
                        detector_count_gate_combo_focus
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                        and 'target_detector_class_conf_gt0p3_overlap'
                        in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores[
                            'target_detector_class_overlap'
                        ]
                        conf_scores = diagnostic_scores[
                            'target_detector_class_conf_gt0p3_overlap'
                        ]
                        det_count = self._target_detector_class_match_count(
                            end_points, bid, min_conf=0.30
                        )
                        if det_count is not None:
                            for active, count_suffix in (
                                (det_count <= 1, 'le1'),
                                (det_count <= 2, 'le2'),
                                (det_count <= 4, 'le4'),
                                (det_count > 2, 'gt2'),
                            ):
                                active_weight = quality_scores.new_tensor(
                                    1.0 if active else 0.0
                                )
                                for class_alpha, class_suffix in (
                                    (0.15, 'c0p15'),
                                    (0.25, 'c0p25'),
                                ):
                                    conf_alpha = 0.25
                                    combo_source = (
                                        'quality_target_detector_'
                                        'class_conf_gt0p3_detcount_'
                                        f'{count_suffix}_sum_'
                                        f'{class_suffix}_f0p25'
                                    )
                                    combo_scores = (
                                        quality_scores
                                        + active_weight * (
                                            class_alpha * class_scores
                                            + conf_alpha * conf_scores
                                        )
                                    )
                                    diagnostic_scores[combo_source] = (
                                        combo_scores
                                    )
                                    diagnostic_ious[combo_source] = (
                                        self._record_diagnostic(
                                            combo_source, prefix,
                                            combo_scores, gt_boxes,
                                            pred_bbox[bid]
                                        )
                                    )
                    if (
                        detector_quality_margin_guard_focus
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                        and 'target_detector_class_conf_gt0p3_overlap'
                        in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        class_scores = diagnostic_scores[
                            'target_detector_class_overlap'
                        ]
                        conf_scores = diagnostic_scores[
                            'target_detector_class_conf_gt0p3_overlap'
                        ]
                        if quality_scores.shape[1] > 1:
                            quality_top2 = quality_scores.topk(
                                2, dim=1
                            ).values
                            quality_margin = (
                                quality_top2[:, 0] - quality_top2[:, 1]
                            )
                        else:
                            quality_margin = torch.zeros(
                                quality_scores.shape[0],
                                device=quality_scores.device,
                                dtype=quality_scores.dtype,
                            )
                        for margin, margin_suffix in (
                            (0.25, 'le0p25'),
                            (0.50, 'le0p5'),
                            (1.0, 'le1'),
                        ):
                            active = (
                                quality_margin <= margin
                            ).float().unsqueeze(1)
                            for class_alpha, class_suffix in (
                                (0.15, 'c0p15'),
                                (0.25, 'c0p25'),
                            ):
                                conf_alpha = 0.25
                                combo_source = (
                                    'quality_target_detector_'
                                    'class_conf_gt0p3_qmargin_'
                                    f'{margin_suffix}_sum_'
                                    f'{class_suffix}_f0p25'
                                )
                                combo_scores = (
                                    quality_scores
                                    + active * (
                                        class_alpha * class_scores
                                        + conf_alpha * conf_scores
                                    )
                                )
                                diagnostic_scores[combo_source] = (
                                    combo_scores
                                )
                                diagnostic_ious[combo_source] = (
                                    self._record_diagnostic(
                                        combo_source, prefix, combo_scores,
                                        gt_boxes, pred_bbox[bid]
                                    )
                                )
                    if (
                        detector_local_agreement_focus
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                    ):
                        quality_scores = diagnostic_scores['quality']
                        detector_scores = diagnostic_scores[
                            'target_detector_class_overlap'
                        ]
                        num_queries = quality_scores.shape[1]
                        detector_top = detector_scores.argmax(dim=1)
                        quality_top = quality_scores.argmax(dim=1)
                        detector_top_score = torch.gather(
                            detector_scores, 1, detector_top.unsqueeze(1)
                        ).squeeze(1)
                        detector_at_quality_top = torch.gather(
                            detector_scores, 1, quality_top.unsqueeze(1)
                        ).squeeze(1)
                        quality_top_score = torch.gather(
                            quality_scores, 1, quality_top.unsqueeze(1)
                        ).squeeze(1)
                        quality_at_detector_top = torch.gather(
                            quality_scores, 1, detector_top.unsqueeze(1)
                        ).squeeze(1)
                        detector_margin = (
                            detector_top_score - detector_at_quality_top
                        )
                        local_quality_gap = (
                            quality_top_score - quality_at_detector_top
                        )
                        for candidate_k in (2, 3):
                            k = max(1, min(int(candidate_k), num_queries))
                            quality_topk = quality_scores.argsort(
                                dim=1, descending=True
                            )[:, :k]
                            candidate_mask = torch.zeros_like(
                                quality_scores, dtype=torch.bool
                            )
                            candidate_mask.scatter_(1, quality_topk, True)
                            masked_detector = torch.zeros_like(
                                detector_scores
                            )
                            masked_detector[candidate_mask] = (
                                detector_scores[candidate_mask]
                            )
                            for alpha, alpha_suffix in (
                                (0.35, 'a0p35'),
                                (0.5, 'a0p5'),
                            ):
                                blend_source = (
                                    'quality_target_detector_quality_'
                                    f'top{candidate_k}_mask_blend_'
                                    f'{alpha_suffix}'
                                )
                                blend_scores = (
                                    quality_scores
                                    + alpha * masked_detector
                                )
                                diagnostic_scores[blend_source] = (
                                    blend_scores
                                )
                                diagnostic_ious[blend_source] = (
                                    self._record_diagnostic(
                                        blend_source, prefix, blend_scores,
                                        gt_boxes, pred_bbox[bid]
                                    )
                                )
                        for gap, gap_suffix in ((0.05, 'le0p05'),):
                            local_enough = local_quality_gap <= gap
                            for margin, margin_suffix in (
                                (0.05, 'gt0p05'),
                                (0.10, 'gt0p1'),
                            ):
                                active = (
                                    local_enough
                                    & (detector_margin > margin)
                                ).float().unsqueeze(1)
                                for alpha, alpha_suffix in (
                                    (0.35, 'a0p35'),
                                    (0.5, 'a0p5'),
                                    (0.75, 'a0p75'),
                                ):
                                    blend_source = (
                                        'quality_target_detector_local_gap_'
                                        f'{gap_suffix}_margin_'
                                        f'{margin_suffix}_blend_'
                                        f'{alpha_suffix}'
                                    )
                                    blend_scores = (
                                        quality_scores
                                        + active * alpha * detector_scores
                                    )
                                    diagnostic_scores[blend_source] = (
                                        blend_scores
                                    )
                                    diagnostic_ious[blend_source] = (
                                        self._record_diagnostic(
                                            blend_source, prefix, blend_scores,
                                            gt_boxes, pred_bbox[bid]
                                        )
                                    )
                    if (
                        not detector_focused
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                    ):
                        detector_scores = diagnostic_scores[
                            'target_detector_class_overlap'
                        ]
                        quality_scores = diagnostic_scores['quality']
                        quality_top = quality_scores.argmax(dim=1)
                        detector_at_quality_top = torch.gather(
                            detector_scores, 1, quality_top.unsqueeze(1)
                        ).squeeze(1)
                        detector_top = detector_scores.max(dim=1).values
                        detector_margin = (
                            detector_top - detector_at_quality_top
                        )
                        for margin, margin_suffix in (
                            (0.05, 'gt0p05'),
                            (0.10, 'gt0p1'),
                            (0.15, 'gt0p15'),
                            (0.20, 'gt0p2'),
                            (0.25, 'gt0p25'),
                            (0.35, 'gt0p35'),
                            (0.50, 'gt0p5'),
                        ):
                            active = (
                                detector_margin > margin
                            ).float().unsqueeze(1)
                            for alpha, alpha_suffix in (
                                (0.25, 'a0p25'),
                                (0.5, 'a0p5'),
                                (0.75, 'a0p75'),
                                (1.0, 'a1'),
                            ):
                                blend_source = (
                                    'quality_target_detector_margin_'
                                    f'{margin_suffix}_blend_{alpha_suffix}'
                                )
                                blend_scores = (
                                    quality_scores
                                    + active * alpha * detector_scores
                                )
                                diagnostic_scores[blend_source] = blend_scores
                                diagnostic_ious[blend_source] = (
                                    self._record_diagnostic(
                                        blend_source, prefix, blend_scores,
                                        gt_boxes, pred_bbox[bid]
                                    )
                                )
                    if (
                        not detector_focused
                        and 'quality' in diagnostic_scores
                        and 'target_detector_class_overlap' in diagnostic_scores
                    ):
                        detector_scores = diagnostic_scores[
                            'target_detector_class_overlap'
                        ]
                        quality_scores = diagnostic_scores['quality']
                        for candidate_k in (2, 3, 5, 10):
                            self._record_diagnostic_rerank(
                                f'target_detector_top{candidate_k}_quality_rerank',
                                prefix,
                                detector_scores,
                                quality_scores,
                                gt_boxes,
                                pred_bbox[bid],
                                candidate_k,
                            )
                            self._record_diagnostic_rerank(
                                f'quality_top{candidate_k}_target_detector_rerank',
                                prefix,
                                quality_scores,
                                detector_scores,
                                gt_boxes,
                                pred_bbox[bid],
                                candidate_k,
                            )
                    if (
                        not detector_focused
                        and 'selector' in diagnostic_scores
                    ):
                        selector_pool_scores = self._selector_pool_primary_scores(
                            end_points, bid, num_obj, base_scores
                        )
                        diagnostic_scores['selector_pool'] = (
                            selector_pool_scores
                        )
                        self._record_diagnostic(
                            'selector_pool', prefix, selector_pool_scores,
                            gt_boxes, pred_bbox[bid]
                        )
                    if not detector_focused:
                        for source, diag_scores in (
                            self
                            ._selector_pool_quality_override_diagnostic_scores(
                                end_points, bid, num_obj, base_scores
                            ).items()
                        ):
                            diagnostic_scores[source] = diag_scores
                            self._record_diagnostic(
                                source, prefix, diag_scores, gt_boxes,
                                pred_bbox[bid]
                            )
                        for source, diag_scores in (
                            self
                            ._selector_pool_source_blend_diagnostic_scores(
                                end_points, bid, num_obj, base_scores
                            ).items()
                        ):
                            diagnostic_scores[source] = diag_scores
                            self._record_diagnostic(
                                source, prefix, diag_scores, gt_boxes,
                                pred_bbox[bid]
                            )
                        self._record_diagnostic_source_pool_oracle(
                            prefix, diagnostic_ious
                        )
                    if (
                        not detector_focused
                        and 'fused' in diagnostic_scores
                        and 'quality' in diagnostic_scores
                    ):
                        for candidate_k in (5, 10):
                            self._record_diagnostic_rerank(
                                f'fused_top{candidate_k}_quality_rerank',
                                prefix,
                                diagnostic_scores['fused'],
                                diagnostic_scores['quality'],
                                gt_boxes,
                                pred_bbox[bid],
                                candidate_k,
                            )
                    if (
                        not detector_focused
                        and 'quality' in diagnostic_scores
                    ):
                        for candidate_k in (5, 10):
                            self._record_diagnostic_source_pool_rerank(
                                f'source_pool_top{candidate_k}_quality_rerank',
                                prefix,
                                diagnostic_scores,
                                diagnostic_scores['quality'],
                                gt_boxes,
                                pred_bbox[bid],
                                candidate_k,
                            )
                    if not detector_focused:
                        for source, diag_scores in (
                            self._selector_choice_hybrid_diagnostic_scores(
                                end_points, bid, num_obj, base_scores,
                                extra_source_scores=extra_source_scores,
                            ).items()
                        ):
                            diagnostic_scores[source] = diag_scores
                            self._record_diagnostic(
                                source, prefix, diag_scores, gt_boxes,
                                pred_bbox[bid]
                            )
                        for source, diag_scores in (
                            self
                            ._selector_choice_quality_override_diagnostic_scores(
                                end_points, bid, num_obj, base_scores,
                                extra_source_scores=extra_source_scores,
                            ).items()
                        ):
                            diagnostic_scores[source] = diag_scores
                            self._record_diagnostic(
                                source, prefix, diag_scores,
                                gt_boxes, pred_bbox[bid]
                            )
                        for source, diag_scores in (
                            self
                            ._selector_choice_override_threshold_diagnostic_scores(
                                end_points, bid, num_obj, base_scores,
                                extra_source_scores=extra_source_scores,
                            ).items()
                        ):
                            diagnostic_scores[source] = diag_scores
                            self._record_diagnostic(
                                source, prefix, diag_scores,
                                gt_boxes, pred_bbox[bid]
                            )

            # Measure IoU>threshold, ious are (obj, 10)
            topks = self.topks
            for t in self.thresholds:
                thresholded = ious > t
                for k in topks:
                    found = thresholded[:, :k].any(1)
                    self.dets[(prefix, t, k, 'bbs')] += found.sum().item()
                    self.gts[(prefix, t, k, 'bbs')] += len(thresholded)
                    if prefix == 'last_' and k == 1:
                        subset = (
                            'unique'
                            if bool(end_points['is_unique'][bid])
                            else 'multiple'
                        )
                        self.bbs_subset_dets[(subset, t)] += found.sum().item()
                        self.bbs_subset_gts[(subset, t)] += len(thresholded)

    def _record_score_alignment(self, prefix, field, values):
        if values.numel() == 0:
            return
        key = (prefix, field)
        self.score_alignment_sums[key] = (
            self.score_alignment_sums.get(key, 0.0)
            + float(values.detach().float().sum().cpu().item())
        )
        self.score_alignment_gts[key] = (
            self.score_alignment_gts.get(key, 0) + int(values.numel())
        )

    def evaluate_score_alignment(self, end_points, prefix):
        positive_map, gt_bboxes = self._parse_gt(end_points)
        primary_source = self._primary_source(end_points, prefix)
        self.per_layer_score_source[prefix] = primary_source
        self.per_layer_has_fused_scores[prefix] = (
            prefix == 'last_' and 'fused_scores' in end_points
        )

        pred_center = end_points[f'{prefix}center']
        pred_size = end_points[f'{prefix}pred_size']
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        # The compact Stage149 diagnostic must use the exact same boxes as
        # the official BBS evaluator.  The previous dump was emitted from
        # this score-alignment path before detector-policy calibration, which
        # made its per-sample IoUs disagree with the reported BBS metrics.
        if (
            prefix == 'last_'
            and primary_source == 'detector_policy_adapter'
            and torch.is_tensor(end_points.get(
                'detector_policy_adapter_calibrated_boxes', None
            ))
        ):
            calibrated = end_points[
                'detector_policy_adapter_calibrated_boxes'
            ]
            if calibrated.shape == pred_bbox.shape:
                pred_bbox = calibrated.to(
                    device=pred_bbox.device, dtype=pred_bbox.dtype
                )

        bbs_sem_scores = end_points[f'{prefix}sem_cls_scores'].softmax(-1)
        if bbs_sem_scores.shape[-1] != positive_map.shape[-1]:
            padded = torch.zeros(
                bbs_sem_scores.shape[0], bbs_sem_scores.shape[1],
                positive_map.shape[-1],
            ).to(bbs_sem_scores.device)
            padded[:, :, :bbs_sem_scores.shape[-1]] = bbs_sem_scores
            bbs_sem_scores = padded

        proj_tokens = end_points['proj_tokens']
        proj_queries = end_points[f'{prefix}proj_queries']
        bbf_sem_scores = torch.matmul(
            proj_queries, proj_tokens.transpose(-1, -2)
        )
        bbf_sem_scores = (bbf_sem_scores / 0.07).softmax(-1)
        padded = torch.zeros(
            bbf_sem_scores.size(0),
            bbf_sem_scores.size(1),
            positive_map.shape[-1],
        ).to(bbf_sem_scores.device)
        padded[:, :bbf_sem_scores.size(1), :bbf_sem_scores.size(2)] = bbf_sem_scores
        bbf_sem_scores = padded

        for bid in range(len(positive_map)):
            pmap, gt_boxes = self._get_eval_targets(
                end_points, positive_map, gt_bboxes, bid
            )
            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue

            bbs_base_scores = (
                bbs_sem_scores[bid].unsqueeze(0) * pmap.unsqueeze(1)
            ).sum(-1)
            bbf_scores = (
                bbf_sem_scores[bid].unsqueeze(0) * pmap.unsqueeze(1)
            ).sum(-1)
            choice_extra_scores = {'contrastive_base': bbf_scores}
            target_detector_scores = (
                self._target_detector_class_overlap_scores(
                    end_points, bid, num_obj, bbs_base_scores,
                    pred_bbox[bid],
                )
            )
            if target_detector_scores is not None:
                choice_extra_scores['target_detector'] = (
                    target_detector_scores
                )
            target_detector_logit_scores = (
                self._target_detector_logit_overlap_scores(
                    end_points, bid, num_obj, bbs_base_scores,
                    pred_bbox[bid],
                )
            )
            if target_detector_logit_scores is not None:
                choice_extra_scores['target_detector_logit'] = (
                    target_detector_logit_scores
                )
            if prefix == 'last_':
                self._record_detector_topk_compact(
                    end_points, bid, num_obj, bbs_base_scores, bbf_scores,
                    gt_boxes, pred_bbox[bid], topk=5,
                )
            if prefix == 'last_' and self.source_choice_dump_path:
                self.source_choice_dump_source_names_for_logging = (
                    self._candidate_dump_source_names(
                        choice_extra_scores,
                        end_points=end_points,
                    )
                )
                if self.source_choice_dump_topk > 1:
                    example_id = self._next_source_choice_example_id()
                    self.source_choice_feature_rows.extend(
                        self._source_choice_candidate_rows(
                            end_points, bid, example_id, num_obj,
                            bbs_base_scores, gt_boxes, pred_bbox[bid],
                            topk=self.source_choice_dump_topk,
                            extra_source_scores=choice_extra_scores,
                            prefix=prefix,
                            target_map=pmap[0],
                        )
                    )
                    self._flush_source_choice_feature_rows()
                else:
                    self.source_choice_feature_rows.append(
                        self._source_choice_feature_row(
                            end_points, bid, num_obj, bbs_base_scores,
                            gt_boxes, pred_bbox[bid],
                            extra_source_scores=choice_extra_scores,
                        )
                    )
                    self._flush_source_choice_feature_rows()
            bbs_scores = bbs_base_scores
            if primary_source == 'selector_choice':
                bbs_scores = self._selector_choice_primary_scores(
                    end_points, bid, num_obj, bbs_base_scores,
                    extra_source_scores=choice_extra_scores,
                )
            elif primary_source == 'selector_choice_hybrid':
                bbs_scores = self._selector_choice_hybrid_primary_scores(
                    end_points, bid, num_obj, bbs_base_scores,
                    extra_source_scores=choice_extra_scores,
                )
            elif primary_source == 'selector_choice_quality_override':
                bbs_scores = self._selector_choice_quality_override_primary_scores(
                    end_points, bid, num_obj, bbs_base_scores,
                    extra_source_scores=choice_extra_scores,
                )
            elif primary_source == 'selector_pool':
                bbs_scores = self._selector_pool_primary_scores(
                    end_points, bid, num_obj, bbs_base_scores
                )
            elif primary_source != 'base':
                bbs_scores = self._single_source_scores(
                    end_points, primary_source, bid, num_obj,
                    base_scores=bbs_base_scores,
                )
            bbs_top1 = bbs_scores.argmax(dim=1)
            bbf_top1 = bbf_scores.argmax(dim=1)
            bbs_ious = self._aligned_topk_ious(
                gt_boxes, pred_bbox[bid], bbs_top1.unsqueeze(1)
            )[:, 0]
            bbf_ious = self._aligned_topk_ious(
                gt_boxes, pred_bbox[bid], bbf_top1.unsqueeze(1)
            )[:, 0]

            self._record_score_alignment(
                prefix, 'per_layer_top1_query_bbs', bbs_top1.float()
            )
            self._record_score_alignment(
                prefix, 'per_layer_top1_query_bbf', bbf_top1.float()
            )
            self._record_score_alignment(
                prefix,
                'bbs_vs_bbf_top1_disagree_ratio',
                (bbs_top1 != bbf_top1).float(),
            )
            self._record_score_alignment(prefix, 'bbs_top1_iou', bbs_ious)
            self._record_score_alignment(prefix, 'bbf_top1_iou', bbf_ious)
            if (
                prefix == 'last_'
                and primary_source in (
                    'selector_choice',
                    'selector_choice_hybrid',
                    'selector_choice_quality_override',
                )
                and 'selector_choice_scores' in end_points
            ):
                source_count = end_points['selector_choice_scores'].shape[-1]
                self.selector_choice_source_names_for_logging = (
                    self._selector_choice_source_names(
                        source_count,
                        end_points=end_points,
                    )
                )
                choice_extra_scores = {}
                contrastive_scores = self._contrastive_base_scores(
                    end_points, prefix, bid, pmap
                )
                if contrastive_scores is not None:
                    choice_extra_scores['contrastive_base'] = (
                        contrastive_scores
                    )
                choice_diag = self._selector_choice_diagnostics(
                    end_points, bid, num_obj, bbs_base_scores,
                    gt_boxes, pred_bbox[bid],
                    extra_source_scores=choice_extra_scores,
                    selection_mode=primary_source,
                )
                field_map = {
                    'selected_source_id': 'selector_choice_selected_source_id',
                    'oracle_source_id': 'selector_choice_oracle_source_id',
                    'source_choice_oracle_agree': (
                        'selector_choice_oracle_agree'
                    ),
                    'source_choice_selected_iou': (
                        'selector_choice_selected_iou'
                    ),
                    'source_choice_oracle_iou': 'selector_choice_oracle_iou',
                    'source_choice_iou_gap_to_oracle': (
                        'selector_choice_iou_gap_to_oracle'
                    ),
                    'source_choice_target_override_ratio': (
                        'selector_choice_target_override_ratio'
                    ),
                    'source_choice_selected_override_ratio': (
                        'selector_choice_selected_override_ratio'
                    ),
                    'source_choice_false_base_ratio': (
                        'selector_choice_false_base_ratio'
                    ),
                    'source_choice_false_override_ratio': (
                        'selector_choice_false_override_ratio'
                    ),
                    'source_choice_override_agreement_ratio': (
                        'selector_choice_override_agreement_ratio'
                    ),
                    'source_choice_selected_base_ratio': (
                        'selector_choice_selected_base_ratio'
                    ),
                    'source_choice_selected_fused_ratio': (
                        'selector_choice_selected_fused_ratio'
                    ),
                    'source_choice_selected_quality_ratio': (
                        'selector_choice_selected_quality_ratio'
                    ),
                    'source_choice_selected_contrastive_base_ratio': (
                        'selector_choice_selected_contrastive_base_ratio'
                    ),
                    'source_choice_oracle_base_ratio': (
                        'selector_choice_oracle_base_ratio'
                    ),
                    'source_choice_oracle_fused_ratio': (
                        'selector_choice_oracle_fused_ratio'
                    ),
                    'source_choice_oracle_quality_ratio': (
                        'selector_choice_oracle_quality_ratio'
                    ),
                    'source_choice_oracle_contrastive_base_ratio': (
                        'selector_choice_oracle_contrastive_base_ratio'
                    ),
                    'source_choice_selected_hit025': (
                        'selector_choice_selected_hit025'
                    ),
                    'source_choice_selected_hit050': (
                        'selector_choice_selected_hit050'
                    ),
                    'source_choice_oracle_hit025': (
                        'selector_choice_oracle_hit025'
                    ),
                    'source_choice_oracle_hit050': (
                        'selector_choice_oracle_hit050'
                    ),
                }
                for source in self.selector_choice_source_names_for_logging:
                    field_map[
                        f'source_choice_logit_{source}_mean'
                    ] = f'selector_choice_logit_{source}_mean'
                    field_map[
                        f'source_choice_logit_margin_{source}_mean'
                    ] = f'selector_choice_logit_margin_{source}_mean'
                    field_map[
                        f'source_choice_selected_{source}_ratio'
                    ] = f'selector_choice_selected_{source}_ratio'
                    field_map[
                        f'source_choice_oracle_{source}_ratio'
                    ] = f'selector_choice_oracle_{source}_ratio'
                    for label in ('025', '050'):
                        field_map[
                            f'source_choice_{source}_hit{label}_ratio'
                        ] = f'selector_choice_{source}_hit{label}_ratio'
                        field_map[
                            f'source_choice_{source}_unique_hit{label}_ratio'
                        ] = (
                            f'selector_choice_{source}_unique_hit{label}_ratio'
                        )
                for diag_key, field in field_map.items():
                    if diag_key not in choice_diag:
                        continue
                    self._record_score_alignment(
                        prefix, field, choice_diag[diag_key].view(1)
                    )
            if (
                prefix == 'last_'
                and primary_source == 'selector_pool'
                and (
                    'selector_scores' in end_points
                    or 'selector_source_scores' in end_points
                )
            ):
                pool_diag = self._selector_pool_diagnostics(
                    end_points, bid, num_obj, bbs_base_scores,
                    gt_boxes, pred_bbox[bid],
                )
                field_map = {
                    'selected_query_id': 'selector_pool_selected_query_id',
                    'oracle_query_id': 'selector_pool_oracle_query_id',
                    'selector_pool_oracle_agree': (
                        'selector_pool_oracle_agree'
                    ),
                    'selector_pool_selected_iou': (
                        'selector_pool_selected_iou'
                    ),
                    'selector_pool_oracle_iou': 'selector_pool_oracle_iou',
                    'selector_pool_iou_gap_to_oracle': (
                        'selector_pool_iou_gap_to_oracle'
                    ),
                    'selector_pool_score_gap_to_oracle': (
                        'selector_pool_score_gap_to_oracle'
                    ),
                    'selector_pool_selected_hit025': (
                        'selector_pool_selected_hit025'
                    ),
                    'selector_pool_selected_hit050': (
                        'selector_pool_selected_hit050'
                    ),
                    'selector_pool_oracle_hit025': (
                        'selector_pool_oracle_hit025'
                    ),
                    'selector_pool_oracle_hit050': (
                        'selector_pool_oracle_hit050'
                    ),
                }
                for diag_key, field in field_map.items():
                    if diag_key not in pool_diag:
                        continue
                    self._record_score_alignment(
                        prefix, field, pool_diag[diag_key].view(1)
                    )

    def evaluate_bbox_by_contrast(self, end_points, prefix):
        """
        Evaluate bounding box IoU by contrasting with span features.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        positive_map, gt_bboxes = self._parse_gt(end_points)

        pred_center = end_points[f'{prefix}center']
        pred_size = end_points[f'{prefix}pred_size']
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)

        # Always use contrastive path for bbf metric (preserves metric semantics)
        proj_tokens = end_points['proj_tokens']
        proj_queries = end_points[f'{prefix}proj_queries']
        sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
        sem_scores_ = (sem_scores / 0.07).softmax(-1)
        pad_c = positive_map.shape[-1]
        sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), pad_c)
        sem_scores = sem_scores.to(sem_scores_.device)
        sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_

        for bid in range(len(positive_map)):
            pmap, gt_boxes = self._get_eval_targets(
                end_points, positive_map, gt_bboxes, bid
            )
            scores = (
                sem_scores[bid].unsqueeze(0)
                * pmap.unsqueeze(1)
            ).sum(-1)

            top = scores.argsort(1, True)[:, :10]
            ious = self._aligned_topk_ious(gt_boxes, pred_bbox[bid], top)

            # Measure IoU>threshold, ious are (obj, 10)
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

    def _parse_gt(self, end_points):
        positive_map = torch.clone(end_points['positive_map'])  # (B, K, 256)
        positive_map[positive_map > 0] = 1
        gt_center = end_points['center_label'][:, :, 0:3]  # (B, K, 3)
        gt_size = end_points['size_gts']  # (B, K2,3)
        gt_bboxes = torch.cat([gt_center, gt_size], dim=-1)  # cxcyczwhd
        if self.only_root:
            positive_map = positive_map[:, :1]  # (B, 1, 256)
            gt_bboxes = gt_bboxes[:, :1]  # (B, 1, 6)
        return positive_map, gt_bboxes


class GroundingGTEvaluator:
    """
    Evaluate language grounding.

    Args:
        only_root (bool): detect only the root noun
        thresholds (list): IoU thresholds to check
        topks (list): k to evaluate top--k accuracy
        prefixes (list): names of layers to evaluate
    """

    def __init__(self, prefixes=[]):
        """Initialize accumulators."""
        self.prefixes = prefixes
        self.reset()

    def reset(self):
        """Reset accumulators to empty."""
        self.dets = {
            (prefix, mode): 0
            for prefix in self.prefixes
            for mode in ['bbs', 'bbf']
        }
        self.gts = dict(self.dets)

        self.dets.update({'vd': 0, 'vid': 0})
        self.dets.update({'hard': 0, 'easy': 0})
        self.dets.update({'multi': 0, 'unique': 0})
        self.gts.update({'vd': 1e-14, 'vid': 1e-14})
        self.gts.update({'hard': 1e-14, 'easy': 1e-14})
        self.gts.update({'multi': 1e-14, 'unique': 1e-14})
        self.primary_score_source = 'base'
        self.bbs_score_source = 'base'
        self.bbf_score_source = 'contrastive_base'
        self.diagnostic_dets = {}
        self.diagnostic_gts = {}
        self.decomposition_dets = {}
        self.decomposition_gts = {}
        self.decomposition_status_counts = {}

    def print_stats(self):
        """Print accumulated accuracies and return results as dict."""
        mode_str = {
            'bbs': 'Box given span (soft-token)',
            'bbf': 'Box given span (contrastive)'
        }
        results = {}
        for prefix in self.prefixes:
            for mode in ['bbs', 'bbf']:
                value = self.dets[(prefix, mode)] / self.gts[(prefix, mode)]
                line = f'{prefix} {mode_str[mode]} Acc: {value}'
                print(line)
                results[f'{prefix}_{mode}_acc'] = value

        print(f'\nPrimary score source: {self.primary_score_source}')
        print(f'BBS score source: {self.bbs_score_source}')
        print(f'BBF score source: {self.bbf_score_source}')
        results['eval_primary_score_source'] = self.primary_score_source
        results['eval_bbs_score_source'] = self.bbs_score_source
        results['eval_bbf_score_source'] = self.bbf_score_source
        results['eval_primary_metric_family'] = 'bbs'
        results['eval_target_row_uses_primary_score'] = True
        results['eval_anchor_rows_use_baseline_score'] = True
        results['eval_bbf_is_diagnostic_only'] = True
        results['primary_score_source_id'] = {
            'base': 0.0, 'structured': 1.0, 'quality': 2.0,
            'fused': 3.0, 'acd': 4.0, 'selector': 5.0,
            'selector_pool': 6.0, 'selector_choice': 7.0,
            'selector_choice_hybrid': 8.0,
            'selector_choice_quality_override': 9.0,
            'detector_policy_adapter': 10.0,
        }.get(self.primary_score_source, -1.0)
        results['eval_primary_score_source_id'] = results['primary_score_source_id']
        results['eval_bbs_score_source_id'] = results['primary_score_source_id']
        results['eval_bbf_score_source_id'] = 0.0

        if self.diagnostic_dets:
            print('\nDiagnostic score sources')
            for key in sorted(self.diagnostic_dets.keys()):
                value = self.diagnostic_dets[key] / max(self.diagnostic_gts.get(key, 1), 1)
                print(key, value)
                source, prefix = key
                results[f'{prefix}_diag_{source}_acc'] = value
                if prefix == 'last_':
                    results[f'diag_{source}'] = value

        if self.decomposition_dets:
            print('\nDecomposition status')
            for key in sorted(self.decomposition_dets.keys()):
                value = self.decomposition_dets[key] / max(self.decomposition_gts.get(key, 1), 1)
                print(key, value)
                status, prefix = key
                safe_status = str(status).replace('/', '_').replace(' ', '_')
                results[f'{prefix}_decomp_{safe_status}_acc'] = value
            for (status, prefix), count in sorted(self.decomposition_status_counts.items()):
                safe_status = str(status).replace('/', '_').replace(' ', '_')
                results[f'{prefix}_decomp_{safe_status}_count'] = count
                if prefix == 'last_':
                    results[f'eval_decomp_{safe_status}_count'] = count

        print('\nAnalysis')
        for field in ['easy', 'hard', 'vd', 'vid', 'unique', 'multi']:
            value = self.dets[field] / self.gts[field]
            print(field, value)
            results[field] = value

        return results

    def synchronize_between_processes(self):
        all_dets = misc.all_gather(self.dets)
        all_gts = misc.all_gather(self.gts)
        all_diag_dets = misc.all_gather(self.diagnostic_dets)
        all_diag_gts = misc.all_gather(self.diagnostic_gts)
        all_decomp_dets = misc.all_gather(self.decomposition_dets)
        all_decomp_gts = misc.all_gather(self.decomposition_gts)
        all_decomp_counts = misc.all_gather(self.decomposition_status_counts)

        if misc.is_main_process():
            merged_predictions = {}
            for key in set().union(*[p.keys() for p in all_dets]):
                merged_predictions[key] = 0
                for p in all_dets:
                    merged_predictions[key] += p.get(key, 0)
            self.dets = merged_predictions

            merged_predictions = {}
            for key in set().union(*[p.keys() for p in all_gts]):
                merged_predictions[key] = 0
                for p in all_gts:
                    merged_predictions[key] += p.get(key, 0)
            self.gts = merged_predictions

            self.diagnostic_dets = {}
            self.diagnostic_gts = {}
            diag_keys = set().union(*[p.keys() for p in all_diag_dets]) if all_diag_dets else []
            for key in diag_keys:
                self.diagnostic_dets[key] = sum(p.get(key, 0) for p in all_diag_dets)
                self.diagnostic_gts[key] = sum(p.get(key, 0) for p in all_diag_gts)

            self.decomposition_dets = {}
            self.decomposition_gts = {}
            decomp_keys = set().union(*[p.keys() for p in all_decomp_dets]) if all_decomp_dets else []
            for key in decomp_keys:
                self.decomposition_dets[key] = sum(p.get(key, 0) for p in all_decomp_dets)
                self.decomposition_gts[key] = sum(p.get(key, 0) for p in all_decomp_gts)

            self.decomposition_status_counts = {}
            count_keys = set().union(*[p.keys() for p in all_decomp_counts]) if all_decomp_counts else []
            for key in count_keys:
                self.decomposition_status_counts[key] = sum(
                    p.get(key, 0) for p in all_decomp_counts
                )

    def evaluate(self, end_points, prefix):
        """
        Evaluate all accuracies.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        self.evaluate_bbox_by_span(end_points, prefix)
        self.evaluate_bbox_by_contrast(end_points, prefix)

    @staticmethod
    def _found_from_gt_scores(scores, is_correct, pred_bbox_bid,
                              all_gt_boxes, all_bboxes_bid, gt_boxes_bid):
        ranked_scores = scores * is_correct[None]
        top = ranked_scores.argsort(1, True)[:, 0]
        pbox = pred_bbox_bid[top.reshape(-1)]
        ious, _ = _iou3d_par(
            all_gt_boxes,
            box_cxcyczwhd_to_xyzxyz(pbox)
        )
        snapped = all_bboxes_bid[ious.argmax()]
        return int((snapped == gt_boxes_bid).all())

    def _record_diagnostic(self, source, prefix, found):
        key = (source, prefix)
        self.diagnostic_dets[key] = self.diagnostic_dets.get(key, 0) + found
        self.diagnostic_gts[key] = self.diagnostic_gts.get(key, 0) + 1

    def _record_decomposition(self, status, prefix, found):
        key = (status, prefix)
        self.decomposition_dets[key] = self.decomposition_dets.get(key, 0) + found
        self.decomposition_gts[key] = self.decomposition_gts.get(key, 0) + 1
        self.decomposition_status_counts[key] = (
            self.decomposition_status_counts.get(key, 0) + 1
        )

    def evaluate_bbox_by_span(self, end_points, prefix):
        """
        Evaluate bounding box IoU for top gt span detections.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # Parse gt
        positive_map, gt_bboxes = self._parse_gt(end_points)

        primary_source = GroundingEvaluator._primary_source(end_points, prefix)
        if prefix == 'last_':
            self.primary_score_source = primary_source
            self.bbs_score_source = primary_source

        # Parse predictions
        sem_scores = end_points[f'{prefix}sem_cls_scores'].softmax(-1)

        if sem_scores.shape[-1] != positive_map.shape[-1]:
            sem_scores_ = torch.zeros(
                sem_scores.shape[0], sem_scores.shape[1],
                positive_map.shape[-1]).to(sem_scores.device)
            sem_scores_[:, :, :sem_scores.shape[-1]] = sem_scores
            sem_scores = sem_scores_

        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            all_gt_boxes = box_cxcyczwhd_to_xyzxyz(
                        end_points['all_bboxes'][bid][
                            end_points['all_bbox_label_mask'][bid]
                        ]
                    )

            # filter out boxes with low overlap
            ious, _ = _iou3d_par(all_gt_boxes,  # (gt, 6)
                box_cxcyczwhd_to_xyzxyz(pred_bbox[bid])  # (Q, 6)
            )  # (gt, Q)
            is_correct = (ious.max(0)[0] > 0.25) * 1.0

            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            base_scores = (
                sem_scores[bid].unsqueeze(0)  # (1, Q, 256)
                * pmap.unsqueeze(1)  # (obj, 1, 256)
            ).sum(-1)  # (obj, Q)
            scores = base_scores
            if primary_source == 'selector_choice' and pmap.shape[0] > 0:
                scores = GroundingEvaluator._selector_choice_primary_scores(
                    end_points, bid, pmap.shape[0], base_scores
                )
            elif (
                primary_source == 'selector_choice_hybrid'
                and pmap.shape[0] > 0
            ):
                scores = GroundingEvaluator._selector_choice_hybrid_primary_scores(
                    end_points, bid, pmap.shape[0], base_scores
                )
            elif (
                primary_source == 'selector_choice_quality_override'
                and pmap.shape[0] > 0
            ):
                scores = GroundingEvaluator._selector_choice_quality_override_primary_scores(
                    end_points, bid, pmap.shape[0], base_scores
                )
            elif primary_source == 'selector_pool' and pmap.shape[0] > 0:
                scores = GroundingEvaluator._selector_pool_primary_scores(
                    end_points, bid, pmap.shape[0], base_scores
                )
            elif primary_source != 'base' and pmap.shape[0] > 0:
                scores = GroundingEvaluator._single_source_scores(
                    end_points, primary_source, bid, pmap.shape[0],
                    base_scores=base_scores,
                )
            all_bboxes_bid = end_points['all_bboxes'][bid][
                end_points['all_bbox_label_mask'][bid]
            ]
            found = self._found_from_gt_scores(
                scores, is_correct, pred_bbox[bid],
                all_gt_boxes, all_bboxes_bid, gt_bboxes[bid],
            )
            self.dets[(prefix, 'bbs')] += found
            self.gts[(prefix, 'bbs')] += 1
            if prefix == 'last_':
                self._record_decomposition(
                    GroundingEvaluator._decomposition_status(end_points, bid),
                    prefix,
                    found,
                )
                if bool(end_points.get('eval_report_diagnostic_scores', False)):
                    self._record_diagnostic('base', prefix, self._found_from_gt_scores(
                        base_scores, is_correct, pred_bbox[bid],
                        all_gt_boxes, all_bboxes_bid, gt_bboxes[bid],
                    ))
                    for source in [
                        'structured', 'quality', 'fused', 'acd', 'selector'
                    ]:
                        key = {
                            'structured': 'structured_scores',
                            'quality': 'pred_iou',
                            'fused': 'fused_scores',
                            'acd': 'acd_final_scores',
                            'selector': 'selector_scores',
                        }[source]
                        if key in end_points:
                            diag_scores = GroundingEvaluator._single_source_scores(
                                end_points, source, bid, pmap.shape[0],
                                base_scores=base_scores,
                            )
                            self._record_diagnostic(source, prefix, self._found_from_gt_scores(
                                diag_scores, is_correct, pred_bbox[bid],
                                all_gt_boxes, all_bboxes_bid, gt_bboxes[bid],
                            ))

    def evaluate_bbox_by_contrast(self, end_points, prefix):
        """
        Evaluate bounding box IoU by contrasting with span features.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # Parse gt
        positive_map, gt_bboxes = self._parse_gt(end_points)

        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)

        proj_tokens = end_points['proj_tokens']  # (B, tokens, 64)
        proj_queries = end_points[f'{prefix}proj_queries']  # (B, Q, 64)
        sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
        sem_scores_ = (sem_scores / 0.07).softmax(-1)  # (B, Q, tokens)
        pad_c = positive_map.shape[-1]
        sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), pad_c)
        sem_scores = sem_scores.to(sem_scores_.device)
        sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            all_gt_boxes = box_cxcyczwhd_to_xyzxyz(
                    end_points['all_bboxes'][bid][
                        end_points['all_bbox_label_mask'][bid]
                    ]
                )
            ious, _ = _iou3d_par(all_gt_boxes,  # (gt, 6)
                box_cxcyczwhd_to_xyzxyz(pred_bbox[bid])  # (Q, 6)
            )  # (gt, Q)
            is_correct = (ious.max(0)[0] > 0.25) * 1.0
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores = (
                sem_scores[bid].unsqueeze(0)  # (1, Q, 256)
                * pmap.unsqueeze(1)  # (obj, 1, 256)
            ).sum(-1)  # (obj, Q)
            scores = scores * is_correct[None]

            # 10 predictions per gt box
            top = scores.argsort(1, True)[:, 0]  # (obj, 10)
            pbox = pred_bbox[bid, top.reshape(-1)]

            # IoU
            ious, _ = _iou3d_par(
                all_gt_boxes,  # (obj, 6)
                box_cxcyczwhd_to_xyzxyz(pbox)  # (obj*10, 6)
            )  # (obj, obj*10)
            pbox = end_points['all_bboxes'][bid][
                            end_points['all_bbox_label_mask'][bid]
                        ][ious.argmax()]
            found = int((pbox == gt_bboxes[bid]).all())

            # Measure IoU>threshold, ious are (obj, 10)
            self.dets[(prefix, 'bbf')] += found
            self.gts[(prefix, 'bbf')] += 1
            if prefix == 'last_':
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

    def _parse_gt(self, end_points):
        positive_map = torch.clone(end_points['positive_map'])  # (B, K, 256)
        positive_map[positive_map > 0] = 1
        gt_center = end_points['center_label'][:, :, 0:3]  # (B, K, 3)
        gt_size = end_points['size_gts']  # (B, K2,3)
        gt_bboxes = torch.cat([gt_center, gt_size], dim=-1)  # cxcyczwhd
        positive_map = positive_map[:, :1]  # (B, 1, 256)
        gt_bboxes = gt_bboxes[:, :1]  # (B, 1, 6)
        return positive_map, gt_bboxes
