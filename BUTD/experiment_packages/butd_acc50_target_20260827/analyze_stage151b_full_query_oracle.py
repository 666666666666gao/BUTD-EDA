#!/usr/bin/env python3
import argparse
import hashlib
import json

import numpy as np
import torch


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _selected(row):
    queries = [int(value) for value in row['adapter_candidate_query']]
    query = int(row['adapter_rescue_query'])
    offset = queries.index(query)
    ious = np.asarray(row['adapter_iou_at_candidate'], dtype=np.float64)
    return query, float(ious[offset]), float(ious.max())


def _full_query_oracle(row):
    if 'all_query_iou_max' not in row:
        raise KeyError(
            'all_query_iou_max missing: this dump only supports adapter-pool '
            'Oracle and must not be reported as full-query Oracle'
        )
    return float(row['all_query_iou_max'])


def _tier(iou):
    if iou >= 0.50:
        return 'tier2_ge_050'
    if iou >= 0.25:
        return 'tier1_025_050'
    if iou > 0.10:
        return 'ambiguous_010_025'
    return 'tier0_le_010'


def _empty_bins():
    return {
        'tier2_ge_050': 0,
        'tier1_025_050': 0,
        'ambiguous_010_025': 0,
        'tier0_le_010': 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline_dump')
    parser.add_argument('candidate_dump')
    parser.add_argument('output_json')
    args = parser.parse_args()

    baseline_rows = torch.load(args.baseline_dump, map_location='cpu')['rows']
    candidate_rows = torch.load(args.candidate_dump, map_location='cpu')['rows']
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError('row count mismatch')

    report = {
        'stage': '151_stage150_remaining_tier_diagnostic',
        'rows': len(candidate_rows),
        'baseline_dump_sha256': _sha256(args.baseline_dump),
        'candidate_dump_sha256': _sha256(args.candidate_dump),
        'selected_iou_bins': _empty_bins(),
        'remaining_case_a_050_selected_bins': _empty_bins(),
        'remaining_case_a_025_selected_bins': _empty_bins(),
        'fix_050_candidate_bins': _empty_bins(),
        'break_050_candidate_bins': _empty_bins(),
        'fix_025_candidate_bins': _empty_bins(),
        'break_025_candidate_bins': _empty_bins(),
        'groups': {},
        'query_changed': 0,
        'candidate_set_changed': 0,
        'oracle_scope': 'all_predicted_queries',
    }
    for group in ('unique', 'multiple'):
        report['groups'][group] = {
            'rows': 0,
            'selected_iou_bins': _empty_bins(),
            'remaining_case_a_050_selected_bins': _empty_bins(),
            'remaining_case_a_025_selected_bins': _empty_bins(),
        }

    hit_counts = {
        'baseline_025': 0,
        'baseline_050': 0,
        'candidate_025': 0,
        'candidate_050': 0,
        'candidate_oracle_025': 0,
        'candidate_oracle_050': 0,
        'adapter_pool_oracle_025': 0,
        'adapter_pool_oracle_050': 0,
    }
    for baseline_row, candidate_row in zip(baseline_rows, candidate_rows):
        base_query, base_iou, _ = _selected(baseline_row)
        cand_query, cand_iou, adapter_pool_oracle = _selected(candidate_row)
        cand_oracle = _full_query_oracle(candidate_row)
        report['query_changed'] += int(base_query != cand_query)
        report['candidate_set_changed'] += int(
            tuple(int(v) for v in baseline_row['adapter_candidate_query'])
            != tuple(int(v) for v in candidate_row['adapter_candidate_query'])
        )
        group = (
            'unique' if bool(candidate_row.get('is_unique_label_only', False))
            else 'multiple'
        )
        report['groups'][group]['rows'] += 1
        selected_tier = _tier(cand_iou)
        report['selected_iou_bins'][selected_tier] += 1
        report['groups'][group]['selected_iou_bins'][selected_tier] += 1

        for threshold, suffix in ((0.25, '025'), (0.50, '050')):
            base_hit = base_iou > threshold
            cand_hit = cand_iou > threshold
            oracle_hit = cand_oracle > threshold
            hit_counts[f'baseline_{suffix}'] += int(base_hit)
            hit_counts[f'candidate_{suffix}'] += int(cand_hit)
            hit_counts[f'candidate_oracle_{suffix}'] += int(oracle_hit)
            hit_counts[f'adapter_pool_oracle_{suffix}'] += int(
                adapter_pool_oracle >= threshold
            )
            if oracle_hit and not cand_hit:
                key = f'remaining_case_a_{suffix}_selected_bins'
                report[key][selected_tier] += 1
                report['groups'][group][key][selected_tier] += 1
            if cand_hit and not base_hit:
                report[f'fix_{suffix}_candidate_bins'][selected_tier] += 1
            if base_hit and not cand_hit:
                report[f'break_{suffix}_candidate_bins'][selected_tier] += 1

    report['hits'] = hit_counts
    report['query_changed_ratio'] = (
        report['query_changed'] / max(1, report['rows'])
    )
    report['remaining_case_a_050'] = sum(
        report['remaining_case_a_050_selected_bins'].values()
    )
    report['remaining_case_a_025'] = sum(
        report['remaining_case_a_025_selected_bins'].values()
    )
    report['gt_used_for_diagnostic_only'] = True
    report['deployable_inference_claim'] = False
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
