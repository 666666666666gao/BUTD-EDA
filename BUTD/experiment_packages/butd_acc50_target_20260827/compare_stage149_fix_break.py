#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def selected_record(row):
    queries = [int(value) for value in row['adapter_candidate_query']]
    selected = int(row['adapter_rescue_query'])
    if selected not in queries:
        raise ValueError(
            'selected query {} is not in candidate list'.format(selected)
        )
    offset = queries.index(selected)
    boxes = np.asarray(row['adapter_box_at_candidate'], dtype=np.float64)
    ious = np.asarray(row['adapter_iou_at_candidate'], dtype=np.float64)
    return {
        'query': selected,
        'box': boxes[offset],
        'iou': float(ious[offset]),
        'oracle_iou': float(ious.max()),
        'candidate_queries': tuple(queries),
    }


def safe_rate(count, total):
    return float(count) / max(1, int(total))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline_dump')
    parser.add_argument('candidate_dump')
    parser.add_argument('output_json')
    args = parser.parse_args()

    baseline_rows = torch.load(
        args.baseline_dump, map_location='cpu'
    )['rows']
    candidate_rows = torch.load(
        args.candidate_dump, map_location='cpu'
    )['rows']
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError('row count mismatch')

    thresholds = (0.25, 0.50)
    counts = {
        threshold: {
            'baseline_hit': 0,
            'candidate_hit': 0,
            'fix': 0,
            'break': 0,
            'fix_query_changed': 0,
            'break_query_changed': 0,
            'fix_same_query': 0,
            'break_same_query': 0,
        }
        for threshold in thresholds
    }
    query_changed = 0
    box_changed = 0
    same_query_box_changed = 0
    candidate_set_changed = 0
    oracle_changed = 0
    iou_delta = []
    query_changed_iou_delta = []
    same_query_iou_delta = []
    unique_stats = {
        'unique': {threshold: [0, 0, 0] for threshold in thresholds},
        'multiple': {threshold: [0, 0, 0] for threshold in thresholds},
    }

    for index, (baseline_row, candidate_row) in enumerate(zip(
        baseline_rows, candidate_rows
    )):
        baseline = selected_record(baseline_row)
        candidate = selected_record(candidate_row)
        if baseline['candidate_queries'] != candidate['candidate_queries']:
            candidate_set_changed += 1
        if not math.isclose(
            baseline['oracle_iou'], candidate['oracle_iou'],
            rel_tol=0.0, abs_tol=1e-7,
        ):
            oracle_changed += 1
        changed_query = baseline['query'] != candidate['query']
        changed_box = not np.allclose(
            baseline['box'], candidate['box'], rtol=0.0, atol=1e-7
        )
        query_changed += int(changed_query)
        box_changed += int(changed_box)
        same_query_box_changed += int((not changed_query) and changed_box)
        delta = candidate['iou'] - baseline['iou']
        iou_delta.append(delta)
        (query_changed_iou_delta if changed_query else same_query_iou_delta).append(
            delta
        )

        group = (
            'unique' if bool(candidate_row.get('is_unique_label_only', False))
            else 'multiple'
        )
        for threshold in thresholds:
            baseline_hit = baseline['iou'] >= threshold
            candidate_hit = candidate['iou'] >= threshold
            fix = (not baseline_hit) and candidate_hit
            broken = baseline_hit and (not candidate_hit)
            stat = counts[threshold]
            stat['baseline_hit'] += int(baseline_hit)
            stat['candidate_hit'] += int(candidate_hit)
            stat['fix'] += int(fix)
            stat['break'] += int(broken)
            stat['fix_query_changed'] += int(fix and changed_query)
            stat['break_query_changed'] += int(broken and changed_query)
            stat['fix_same_query'] += int(fix and not changed_query)
            stat['break_same_query'] += int(broken and not changed_query)
            group_stat = unique_stats[group][threshold]
            group_stat[0] += 1
            group_stat[1] += int(baseline_hit)
            group_stat[2] += int(candidate_hit)

    def describe(values):
        array = np.asarray(values, dtype=np.float64)
        if not len(array):
            return {'count': 0}
        return {
            'count': int(len(array)),
            'mean': float(array.mean()),
            'p10': float(np.quantile(array, 0.10)),
            'p50': float(np.quantile(array, 0.50)),
            'p90': float(np.quantile(array, 0.90)),
            'positive': int((array > 1e-7).sum()),
            'negative': int((array < -1e-7).sum()),
            'zero': int((np.abs(array) <= 1e-7).sum()),
        }

    rows = len(baseline_rows)
    threshold_report = {}
    for threshold, stat in counts.items():
        threshold_report[str(threshold)] = {
            **stat,
            'net': stat['fix'] - stat['break'],
            'baseline_accuracy': safe_rate(stat['baseline_hit'], rows),
            'candidate_accuracy': safe_rate(stat['candidate_hit'], rows),
        }
    group_report = {}
    for group, threshold_values in unique_stats.items():
        group_report[group] = {}
        for threshold, (total, baseline_hit, candidate_hit) in (
            threshold_values.items()
        ):
            group_report[group][str(threshold)] = {
                'rows': total,
                'baseline_hit': baseline_hit,
                'candidate_hit': candidate_hit,
                'net': candidate_hit - baseline_hit,
                'baseline_accuracy': safe_rate(baseline_hit, total),
                'candidate_accuracy': safe_rate(candidate_hit, total),
            }

    report = {
        'stage': '149_stage148_fix_break_diagnostic',
        'status': 'complete',
        'rows': rows,
        'baseline_dump': os.path.abspath(args.baseline_dump),
        'candidate_dump': os.path.abspath(args.candidate_dump),
        'baseline_dump_sha256': sha256(args.baseline_dump),
        'candidate_dump_sha256': sha256(args.candidate_dump),
        'candidate_set_changed': candidate_set_changed,
        'oracle_changed': oracle_changed,
        'query_changed': query_changed,
        'query_changed_ratio': safe_rate(query_changed, rows),
        'box_changed': box_changed,
        'box_changed_ratio': safe_rate(box_changed, rows),
        'same_query_box_changed': same_query_box_changed,
        'same_query_box_changed_ratio': safe_rate(
            same_query_box_changed, rows
        ),
        'iou_delta_all': describe(iou_delta),
        'iou_delta_query_changed': describe(query_changed_iou_delta),
        'iou_delta_same_query': describe(same_query_iou_delta),
        'thresholds': threshold_report,
        'groups': group_report,
        'gt_used_for_diagnostic_only': True,
        'deployable_inference_claim': False,
    }
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
