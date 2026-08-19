#!/usr/bin/env python
"""Refine precomputed *_spacy decomposition files without running spaCy."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


EDA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDA_ROOT))

from src.joint_det_dataset import Joint3DDataset  # noqa: E402


JSON_FIELDS = (
    'entities',
    'attributes',
    'target_slot',
    'attr_slot',
    'rel_slots',
    'anchor_slots',
    'coverage_stats',
    'slot_mask',
)

SCALAR_FIELDS = (
    'decomp_global_only_mask',
    'decomp_weak_generic_mask',
    'decomp_train_global_only_mask',
    'decomp_train_weak_generic_mask',
    'decomp_train_global_only_reason',
    'decomposition_error_flags_count',
)


def _json_load(value, default):
    if value in (None, ''):
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def _compact_json(value):
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'))


def _compact_scalar(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return ''
    return str(value)


def _infer_dataset(path):
    name = path.name.lower()
    if 'scanrefer' in name:
        return 'scanrefer_spacy'
    if 'nr3d' in name:
        return 'nr3d_spacy'
    if 'sr3d' in name:
        return 'sr3d_spacy'
    raise ValueError('cannot infer dataset; pass --dataset')


def _update_stats(stats, refined):
    coverage = refined.get('coverage_stats', {}) or {}
    stats['rows'] += 1
    stats['kept_relations'] += int(coverage.get('refined_relation_count', 0))
    stats['dropped_relations'] += int(
        coverage.get('refined_dropped_relation_count', 0)
    )
    stats['weak_rows'] += int(bool(refined.get('decomp_weak_generic_mask', False)))
    stats['train_global_only_rows'] += int(
        bool(refined.get('decomp_train_global_only_mask', False))
    )
    stats['train_weak_rows'] += int(
        bool(refined.get('decomp_train_weak_generic_mask', False))
    )
    for key, value in coverage.items():
        if key.startswith('refined_') and key.endswith('_count'):
            try:
                stats[key] += int(value)
            except (TypeError, ValueError):
                pass


def _refine_record(record, dataset):
    working = dict(record)
    working['dataset'] = dataset
    refined = Joint3DDataset._refine_spacy_decomposition_fields(working)
    refined.pop('dataset', None)
    return refined


def _refine_json(input_path, output_path, dataset):
    with input_path.open() as f:
        rows = json.load(f)
    stats = Counter()
    refined_rows = []
    for row in rows:
        refined = _refine_record(row, dataset)
        _update_stats(stats, refined)
        refined_rows.append(refined)
    if output_path:
        with output_path.open('w') as f:
            json.dump(refined_rows, f, ensure_ascii=True)
    return stats


def _refine_csv(input_path, output_path, dataset):
    stats = Counter()
    with input_path.open(newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        output_fieldnames = list(fieldnames)
        for field in SCALAR_FIELDS:
            if field not in output_fieldnames:
                output_fieldnames.append(field)
        rows = []
        for row in reader:
            parsed = dict(row)
            for field in JSON_FIELDS:
                if field in parsed:
                    parsed[field] = _json_load(parsed[field], [] if field.endswith('s') else {})
            if 'parse_confidence' in parsed and parsed['parse_confidence']:
                parsed['parse_confidence'] = float(parsed['parse_confidence'])
            refined = _refine_record(parsed, dataset)
            _update_stats(stats, refined)
            out_row = dict(row)
            for field in JSON_FIELDS:
                if field in out_row and field in refined:
                    out_row[field] = _compact_json(refined[field])
            for field in SCALAR_FIELDS:
                out_row[field] = _compact_scalar(refined.get(field, ''))
            if 'parse_confidence' in out_row:
                out_row['parse_confidence'] = '%.6g' % float(
                    refined.get('parse_confidence', 1.0)
                )
            rows.append(out_row)
    if output_path:
        with output_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument(
        '--dataset',
        choices=('scanrefer_spacy', 'nr3d_spacy', 'sr3d_spacy'),
    )
    args = parser.parse_args()

    dataset = args.dataset or _infer_dataset(args.input)
    if args.input.suffix == '.json':
        stats = _refine_json(args.input, args.output, dataset)
    elif args.input.suffix == '.csv':
        stats = _refine_csv(args.input, args.output, dataset)
    else:
        raise ValueError('input must be .json or .csv')

    for key in sorted(stats):
        print(f'{key}: {stats[key]}')


if __name__ == '__main__':
    main()
