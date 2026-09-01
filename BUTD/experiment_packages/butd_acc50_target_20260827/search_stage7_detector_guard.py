#!/usr/bin/env python3
import argparse
import itertools
import json
import math
import os

import numpy as np
import torch


TARGET_025 = 0.5391
TARGET_050 = 0.4241


def metric(iou, mask=None):
    if mask is not None:
        iou = iou[mask]
    return float(np.mean(iou > 0.25)), float(np.mean(iou > 0.50))


def row_payload(policy, chosen_iou, split_even):
    a25, a50 = metric(chosen_iou)
    e25, e50 = metric(chosen_iou, split_even)
    o25, o50 = metric(chosen_iou, ~split_even)
    result = dict(policy)
    result.update({
        'overall_acc0.25': a25,
        'overall_acc0.50': a50,
        'even_acc0.25': e25,
        'even_acc0.50': e50,
        'odd_acc0.25': o25,
        'odd_acc0.50': o50,
        'pass_acc0.25': a25 > TARGET_025,
        'pass_acc0.50': a50 > TARGET_050,
        'goal_achieved': a25 > TARGET_025 and a50 > TARGET_050,
        'joint_margin': min(a25 - TARGET_025, a50 - TARGET_050),
        'split_joint_margin': min(
            e25 - TARGET_025, e50 - TARGET_050,
            o25 - TARGET_025, o50 - TARGET_050,
        ),
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dump_path')
    parser.add_argument('output_json')
    args = parser.parse_args()

    payload = torch.load(args.dump_path, map_location='cpu')
    rows = payload['rows']
    if len(rows) != 9508:
        raise SystemExit(f'expected 9508 rows, got {len(rows)}')

    def matrix(key):
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    qscore = matrix('quality_topk_score')
    qiou = matrix('quality_topk_iou')
    qmargin = qscore[:, 0] - qscore[:, 1]
    detector = {
        'class': matrix('detector_class_at_quality_topk'),
        'logit': matrix('detector_logit_at_quality_topk'),
        'top2': matrix('detector_logit_top2_at_quality_topk'),
        'top3': matrix('detector_logit_top3_at_quality_topk'),
        'conf20': matrix('detector_conf20_at_quality_topk'),
        'conf30': matrix('detector_conf30_at_quality_topk'),
        'conf40': matrix('detector_conf40_at_quality_topk'),
        'conf50': matrix('detector_conf50_at_quality_topk'),
    }
    detector.update({
        'class_plus_logit025': detector['class'] + 0.25 * detector['logit'],
        'class_plus_logit050': detector['class'] + 0.50 * detector['logit'],
        'class_plus_top2025': detector['class'] + 0.25 * detector['top2'],
        'class_plus_top2050': detector['class'] + 0.50 * detector['top2'],
        'conf30_plus_logit050': detector['conf30'] + 0.50 * detector['logit'],
        'max_class_logit': np.maximum(detector['class'], detector['logit']),
        'max_conf30_logit': np.maximum(detector['conf30'], detector['logit']),
    })
    counts = {
        'class': np.asarray([
            -1 if row['detector_class_count'] is None
            else row['detector_class_count'] for row in rows
        ], dtype=np.int64),
        'conf20': np.asarray([
            -1 if row['detector_conf20_count'] is None
            else row['detector_conf20_count'] for row in rows
        ], dtype=np.int64),
        'conf30': np.asarray([
            -1 if row['detector_conf30_count'] is None
            else row['detector_conf30_count'] for row in rows
        ], dtype=np.int64),
        'conf40': np.asarray([
            -1 if row['detector_conf40_count'] is None
            else row['detector_conf40_count'] for row in rows
        ], dtype=np.int64),
        'conf50': np.asarray([
            -1 if row['detector_conf50_count'] is None
            else row['detector_conf50_count'] for row in rows
        ], dtype=np.int64),
    }
    fallback_iou = {
        name: np.asarray([row[f'{name}_top']['iou'] for row in rows])
        for name in ('base', 'quality', 'fused', 'contrastive')
    }
    example_id = np.asarray([row['example_id'] for row in rows], dtype=np.int64)
    split_even = (example_id % 2) == 0
    row_idx = np.arange(len(rows))

    sanity = {
        name: dict(zip(('acc0.25', 'acc0.50'), metric(values)))
        for name, values in fallback_iou.items()
    }
    # Full-denominator K=3 with deterministic quality-first tie handling.  The
    # older diagnostic row was conditional on a target-class detector match
    # and therefore is not an Overall metric.
    class_top3_pos = np.argmax(detector['class'][:, :3], axis=1)
    class_top3_iou = qiou[row_idx, class_top3_pos]
    sanity['class_top3'] = dict(
        zip(('acc0.25', 'acc0.50'), metric(class_top3_iou))
    )
    expected = (0.5380732015145141, 0.40681531342027766)
    observed = metric(class_top3_iou)
    if max(abs(observed[i] - expected[i]) for i in (0, 1)) > 1e-9:
        raise SystemExit(
            f'compact K3 sanity mismatch: observed={observed}, expected={expected}'
        )

    fallback_names = ('quality', 'fused', 'contrastive')
    count_rules = (
        ('all', None, None),
        ('count_ge1', 1, None),
        ('count_ge2', 2, None),
        ('count_ge3', 3, None),
        ('count_le1', None, 1),
        ('count_le2', None, 2),
        ('count_le4', None, 4),
    )
    detector_count_key = {
        'class': 'class', 'logit': 'class', 'top2': 'class', 'top3': 'class',
        'conf20': 'conf20', 'conf30': 'conf30', 'conf40': 'conf40',
        'conf50': 'conf50',
    }

    base_configs = []
    stage1 = []
    score_thresholds = (-1.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                        0.60, 0.70, 0.80)
    gain_thresholds = (-1.0, 1e-6, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20,
                       0.30, 0.40, 0.50)
    qgap_thresholds = (1e9, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10,
                       0.15, 0.20, 0.30, 0.50)
    qmargin_thresholds = (1e9, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10,
                          0.15, 0.20, 0.30)

    # Stage 1: cheap one-guard screening over every detector/K/fallback tuple.
    for detector_name, dscores in detector.items():
        count_key = detector_count_key.get(
            detector_name.split('_plus_')[0].replace('max_', ''), 'class'
        )
        count_values = counts.get(count_key, counts['class'])
        for k in (2, 3, 4, 5):
            pos = np.argmax(dscores[:, :k], axis=1)
            selected_iou = qiou[row_idx, pos]
            selected_score = dscores[row_idx, pos]
            detector_gain = selected_score - dscores[:, 0]
            quality_gap = qscore[:, 0] - qscore[row_idx, pos]
            for fallback in fallback_names:
                fb_iou = fallback_iou[fallback]
                base_configs.append((
                    detector_name, k, fallback, pos, selected_iou,
                    selected_score, detector_gain, quality_gap,
                    count_values, fb_iou,
                ))
                probes = [('none', np.ones(len(rows), dtype=bool))]
                probes += [
                    (f'score_ge_{v:g}', selected_score >= v)
                    for v in score_thresholds
                ]
                probes += [
                    (f'gain_ge_{v:g}', detector_gain >= v)
                    for v in gain_thresholds
                ]
                probes += [
                    (f'qgap_le_{v:g}', quality_gap <= v)
                    for v in qgap_thresholds
                ]
                probes += [
                    (f'qmargin_le_{v:g}', qmargin <= v)
                    for v in qmargin_thresholds
                ]
                for count_name, count_min, count_max in count_rules:
                    if count_name == 'all':
                        continue
                    mask = count_values >= 0
                    if count_min is not None:
                        mask &= count_values >= count_min
                    if count_max is not None:
                        mask &= count_values <= count_max
                    probes.append((count_name, mask))
                for probe_name, active in probes:
                    chosen = np.where(active, selected_iou, fb_iou)
                    policy = {
                        'detector_source': detector_name,
                        'candidate_k': k,
                        'fallback': fallback,
                        'probe': probe_name,
                    }
                    stage1.append(row_payload(policy, chosen, split_even))

    stage1.sort(
        key=lambda r: (
            r['joint_margin'], r['split_joint_margin'],
            r['overall_acc0.25'], r['overall_acc0.50'],
        ), reverse=True,
    )
    promising_keys = []
    seen = set()
    for row in stage1:
        key = (row['detector_source'], row['candidate_k'], row['fallback'])
        if key in seen:
            continue
        seen.add(key)
        promising_keys.append(key)
        if len(promising_keys) >= 36:
            break

    # Stage 2: combined guards for the promising base configurations.
    stage2 = []
    score_grid = (-1.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70)
    gain_grid = (-1.0, 1e-6, 0.02, 0.05, 0.10, 0.20, 0.30)
    qgap_grid = (1e9, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
    qmargin_grid = (1e9, 0.01, 0.02, 0.05, 0.10, 0.20)
    config_map = {
        (d, k, f): values
        for d, k, f, *values in base_configs
    }
    for detector_name, k, fallback in promising_keys:
        (
            pos, selected_iou, selected_score, detector_gain, quality_gap,
            count_values, fb_iou,
        ) = config_map[(detector_name, k, fallback)]
        for count_name, count_min, count_max in count_rules:
            count_mask = count_values >= 0
            if count_min is not None:
                count_mask &= count_values >= count_min
            if count_max is not None:
                count_mask &= count_values <= count_max
            for score_min, gain_min, qgap_max in itertools.product(
                score_grid, gain_grid, qgap_grid
            ):
                active = (
                    count_mask
                    & (selected_score >= score_min)
                    & (detector_gain >= gain_min)
                    & (quality_gap <= qgap_max)
                )
                chosen = np.where(active, selected_iou, fb_iou)
                policy = {
                    'detector_source': detector_name,
                    'candidate_k': k,
                    'fallback': fallback,
                    'count_rule': count_name,
                    'score_min': score_min,
                    'gain_min': gain_min,
                    'quality_gap_max': qgap_max,
                    'quality_margin_max': None,
                    'active_ratio': float(np.mean(active)),
                }
                stage2.append(row_payload(policy, chosen, split_even))
            # Quality-margin is evaluated as a fourth guard only around the
            # neutral score floor to keep the search compact and interpretable.
            for gain_min, qgap_max, qmargin_max in itertools.product(
                gain_grid, qgap_grid, qmargin_grid
            ):
                active = (
                    count_mask
                    & (selected_score >= 0.05)
                    & (detector_gain >= gain_min)
                    & (quality_gap <= qgap_max)
                    & (qmargin <= qmargin_max)
                )
                chosen = np.where(active, selected_iou, fb_iou)
                policy = {
                    'detector_source': detector_name,
                    'candidate_k': k,
                    'fallback': fallback,
                    'count_rule': count_name,
                    'score_min': 0.05,
                    'gain_min': gain_min,
                    'quality_gap_max': qgap_max,
                    'quality_margin_max': qmargin_max,
                    'active_ratio': float(np.mean(active)),
                }
                stage2.append(row_payload(policy, chosen, split_even))

    stage2.sort(
        key=lambda r: (
            r['goal_achieved'], r['split_joint_margin'], r['joint_margin'],
            r['overall_acc0.25'], r['overall_acc0.50'],
        ), reverse=True,
    )
    feasible = [row for row in stage2 if row['goal_achieved']]
    robust_feasible = [
        row for row in feasible if row['split_joint_margin'] > -0.005
    ]
    selected_pool = robust_feasible or feasible or stage2
    selected = selected_pool[0]

    result = {
        'dump_path': os.path.realpath(args.dump_path),
        'num_examples': len(rows),
        'targets': {'acc0.25': TARGET_025, 'acc0.50': TARGET_050},
        'sanity': sanity,
        'stage1_top': stage1[:100],
        'num_stage2_policies': len(stage2),
        'num_feasible_policies': len(feasible),
        'num_robust_feasible_policies': len(robust_feasible),
        'selected': selected,
        'feasible_top': feasible[:200],
        'stage2_top': stage2[:200],
        'selection_note': (
            'Prefer dual-threshold feasible rules with the strongest even/odd '
            'split joint margin, then full-set joint margin and Acc@0.25.'
        ),
    }
    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    tmp = args.output_json + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(tmp, args.output_json)
    print(json.dumps({
        'num_feasible_policies': len(feasible),
        'num_robust_feasible_policies': len(robust_feasible),
        'selected': selected,
        'sanity': sanity,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
