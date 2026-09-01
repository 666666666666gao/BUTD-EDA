#!/usr/bin/env python
"""Join stable ScanRefer IDs onto the Stage133 raw candidate dump.

Stage133 and Stage24 are both augmentation-disabled train-split passes with
the same 36,665 example_id values.  This script copies only scene_id,
object_id, and ann_id after proving exact ID coverage, exact GT boxes, and
matching non-prediction metadata.  No prediction, score, box, or label is
copied from the older dump.
"""

import argparse
import gc
import hashlib
import json
import os

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dump')
    parser.add_argument('id_source_dump')
    parser.add_argument('output_dump')
    parser.add_argument('receipt_json')
    args = parser.parse_args()
    if os.path.exists(args.output_dump) or os.path.exists(args.receipt_json):
        raise FileExistsError('refusing to overwrite Stage133 ID-repair outputs')

    raw_payload = torch.load(args.raw_dump, map_location='cpu')
    id_payload = torch.load(args.id_source_dump, map_location='cpu')
    raw_rows = raw_payload['rows']
    id_rows = id_payload['rows']
    expected = 36665
    if len(raw_rows) != expected or len(id_rows) != expected:
        raise ValueError((len(raw_rows), len(id_rows), expected))

    raw_ids = [int(row['example_id']) for row in raw_rows]
    source_ids = [int(row['example_id']) for row in id_rows]
    expected_ids = set(range(expected))
    if set(raw_ids) != expected_ids or len(set(raw_ids)) != expected:
        raise ValueError('Stage133 example_id coverage is not exact')
    if set(source_ids) != expected_ids or len(set(source_ids)) != expected:
        raise ValueError('Stage24 example_id coverage is not exact')
    source_by_id = {int(row['example_id']): row for row in id_rows}

    max_gt_abs_delta = 0.0
    metadata_mismatches = 0
    keys = []
    for row in raw_rows:
        source = source_by_id[int(row['example_id'])]
        gt_delta = float(np.max(np.abs(
            np.asarray(row['gt_box'], np.float32)
            - np.asarray(source['gt_box'], np.float32)
        )))
        max_gt_abs_delta = max(max_gt_abs_delta, gt_delta)
        for name in (
            'text_target_cid', 'decomposition_status',
            'spacy_augmentation_bucket',
        ):
            metadata_mismatches += int(row.get(name) != source.get(name))
        for name in ('scene_id', 'object_id', 'ann_id'):
            if name in row:
                raise ValueError('raw Stage133 row unexpectedly has {}'.format(name))
            row[name] = str(source[name])
        keys.append((row['scene_id'], row['object_id'], row['ann_id']))

    if max_gt_abs_delta >= 1e-6:
        raise ValueError('GT mismatch: {}'.format(max_gt_abs_delta))
    if metadata_mismatches:
        raise ValueError('metadata mismatches: {}'.format(metadata_mismatches))
    if len(set(keys)) != expected:
        raise ValueError('stable keys are not unique')

    raw_payload['stable_id_join'] = {
        'format': 'example_id_exact_join_v1',
        'copied_fields': ['scene_id', 'object_id', 'ann_id'],
        'source_dump_sha256': sha256(args.id_source_dump),
        'max_gt_abs_delta': max_gt_abs_delta,
        'metadata_mismatches': metadata_mismatches,
    }
    torch.save(raw_payload, args.output_dump)
    del raw_payload, raw_rows, id_payload, id_rows, source_by_id
    gc.collect()
    corrected = torch.load(args.output_dump, map_location='cpu')
    corrected_rows = corrected['rows']
    corrected_keys = [
        (str(row['scene_id']), str(row['object_id']), str(row['ann_id']))
        for row in corrected_rows
    ]
    if len(corrected_rows) != expected or len(set(corrected_keys)) != expected:
        raise AssertionError('reloaded corrected dump failed key audit')

    receipt = {
        'stage': '133b_id_repair',
        'status': 'complete',
        'protocol': 'exact_example_id_join_copying_only_stable_ids',
        'rows': expected,
        'unique_keys': len(set(corrected_keys)),
        'copied_fields': ['scene_id', 'object_id', 'ann_id'],
        'max_gt_abs_delta': max_gt_abs_delta,
        'metadata_mismatches': metadata_mismatches,
        'raw_dump': os.path.abspath(args.raw_dump),
        'raw_dump_sha256': sha256(args.raw_dump),
        'id_source_dump': os.path.abspath(args.id_source_dump),
        'id_source_dump_sha256': sha256(args.id_source_dump),
        'corrected_dump': os.path.abspath(args.output_dump),
        'corrected_dump_sha256': sha256(args.output_dump),
    }
    with open(args.receipt_json, 'w', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
