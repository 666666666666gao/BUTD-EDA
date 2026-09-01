#!/usr/bin/env python
"""Add an exact dataset-index lookup to an audited Stage29 map."""

import argparse
import hashlib
import json
import os

import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('v1_map')
    parser.add_argument('corrected_dump')
    parser.add_argument('v2_map')
    parser.add_argument('receipt_json')
    args = parser.parse_args()
    if os.path.exists(args.v2_map) or os.path.exists(args.receipt_json):
        raise FileExistsError('refusing to overwrite Stage29 map-v2 outputs')

    payload = torch.load(args.v1_map, map_location='cpu')
    if payload.get('format') != 'stage29_query_action_map_v1':
        raise ValueError('expected Stage29 query-map v1')
    entries = payload['entries']
    rows = torch.load(args.corrected_dump, map_location='cpu')['rows']
    if len(rows) != 36665 or len(entries) != 36665:
        raise ValueError((len(rows), len(entries)))
    entries_by_example_id = {}
    missing = []
    for row in rows:
        stable_key = '|'.join((
            str(row['scene_id']), str(row['object_id']), str(row['ann_id'])
        ))
        entry = entries.get(stable_key, None)
        if entry is None:
            missing.append(stable_key)
            continue
        example_id = str(int(row['example_id']))
        if example_id in entries_by_example_id:
            raise ValueError('duplicate example_id: {}'.format(example_id))
        entries_by_example_id[example_id] = entry
    if missing or len(entries_by_example_id) != 36665:
        raise ValueError((len(missing), len(entries_by_example_id), missing[:2]))
    if set(entries_by_example_id) != {str(index) for index in range(36665)}:
        raise ValueError('example_id coverage is not exactly 0..36664')

    payload['format'] = 'stage29_query_action_map_v2'
    payload['entries_by_example_id'] = entries_by_example_id
    payload.setdefault('metadata', {})['example_id_index'] = {
        'format': 'dataset_index_exact_0_36664_v1',
        'count': len(entries_by_example_id),
        'corrected_dump_sha256': sha256(args.corrected_dump),
        'source_v1_map_sha256': sha256(args.v1_map),
        'validation_behavior': 'disabled_fallback_only',
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.v2_map)), exist_ok=True)
    torch.save(payload, args.v2_map)
    reloaded = torch.load(args.v2_map, map_location='cpu')
    if (
        reloaded.get('format') != 'stage29_query_action_map_v2'
        or len(reloaded.get('entries_by_example_id', {})) != 36665
    ):
        raise AssertionError('reloaded v2 map failed audit')
    receipt = {
        'stage': '134b_example_id_map',
        'status': 'complete',
        'protocol': 'exact_stable_key_to_dataset_example_id_index',
        'rows': len(rows),
        'entries': len(entries),
        'entries_by_example_id': len(entries_by_example_id),
        'example_id_min': min(map(int, entries_by_example_id)),
        'example_id_max': max(map(int, entries_by_example_id)),
        'v1_map_sha256': sha256(args.v1_map),
        'corrected_dump_sha256': sha256(args.corrected_dump),
        'v2_map_sha256': sha256(args.v2_map),
        'v2_map': os.path.abspath(args.v2_map),
        'validation_behavior': 'mapping_disabled_fallback_only',
    }
    with open(args.receipt_json, 'w', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
