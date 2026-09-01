#!/usr/bin/env python
"""Attach candidate query features to a frozen legacy geometry dump.

The legacy rows remain the source of truth for every pre-existing field.  The
semantic dump contributes only candidate-aligned FP16 decoder representations,
which are remapped by query id before being appended.
"""

import argparse
import hashlib
import json
import os

import numpy as np
import torch


SEMANTIC_FIELDS = (
    ('adapter_last_proj_query_f16', 64),
    ('adapter_last_query_f16', 288),
)
IDENTITY_FIELDS = ('example_id', 'scene_id', 'ann_id', 'object_id')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def decode(row, key, expected_rows, expected_dim):
    shape = tuple(int(value) for value in row[key + '_shape'])
    assert shape == (int(expected_rows), int(expected_dim)), (
        key, shape, expected_rows, expected_dim
    )
    values = np.frombuffer(row[key], dtype=np.float16).reshape(shape)
    assert np.isfinite(values).all(), key
    return values


def merge_rows(legacy_rows, semantic_rows):
    assert len(legacy_rows) == len(semantic_rows), (
        len(legacy_rows), len(semantic_rows)
    )
    merged_rows = []
    exact_query_order = 0
    candidate_count = 0
    for index, (legacy, semantic) in enumerate(
        zip(legacy_rows, semantic_rows)
    ):
        for key in IDENTITY_FIELDS:
            assert legacy.get(key) == semantic.get(key), (
                index, key, legacy.get(key), semantic.get(key)
            )
        legacy_queries = [int(value) for value in legacy['adapter_candidate_query']]
        semantic_queries = [
            int(value) for value in semantic['adapter_candidate_query']
        ]
        assert len(semantic_queries) == len(set(semantic_queries)), (
            index, semantic_queries
        )
        semantic_position = {
            query: position for position, query in enumerate(semantic_queries)
        }
        assert set(legacy_queries).issubset(semantic_position), (
            index, legacy_queries, semantic_queries
        )
        if legacy_queries == semantic_queries:
            exact_query_order += 1

        merged = dict(legacy)
        for key, dim in SEMANTIC_FIELDS:
            source = decode(semantic, key, len(semantic_queries), dim)
            aligned = np.stack(
                [source[semantic_position[query]] for query in legacy_queries],
                axis=0,
            ).astype(np.float16, copy=False)
            merged[key] = aligned.tobytes(order='C')
            merged[key + '_shape'] = [len(legacy_queries), dim]
        for key, value in legacy.items():
            assert merged[key] == value, (index, key)
        merged_rows.append(merged)
        candidate_count += len(legacy_queries)
    return merged_rows, exact_query_order, candidate_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('legacy_dump')
    parser.add_argument('semantic_dump')
    parser.add_argument('output_dump')
    parser.add_argument('receipt_json')
    args = parser.parse_args()

    for path in (args.output_dump, args.receipt_json):
        assert not os.path.exists(path), path
    legacy_payload = torch.load(args.legacy_dump, map_location='cpu')
    semantic_payload = torch.load(args.semantic_dump, map_location='cpu')
    merged_rows, exact_query_order, candidate_count = merge_rows(
        legacy_payload['rows'], semantic_payload['rows']
    )
    output_payload = dict(legacy_payload)
    output_payload['rows'] = merged_rows
    output_payload['semantic_merge_provenance'] = {
        'legacy_dump': os.path.abspath(args.legacy_dump),
        'legacy_dump_sha256': sha256(args.legacy_dump),
        'semantic_dump': os.path.abspath(args.semantic_dump),
        'semantic_dump_sha256': sha256(args.semantic_dump),
        'semantic_fields': [key for key, _ in SEMANTIC_FIELDS],
        'identity_fields': list(IDENTITY_FIELDS),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_dump)), exist_ok=True)
    temporary = args.output_dump + '.tmp'
    assert not os.path.exists(temporary), temporary
    torch.save(output_payload, temporary)
    os.replace(temporary, args.output_dump)

    reloaded = torch.load(args.output_dump, map_location='cpu')['rows']
    assert len(reloaded) == len(legacy_payload['rows'])
    for index, (legacy, merged) in enumerate(
        zip(legacy_payload['rows'], reloaded)
    ):
        for key, value in legacy.items():
            assert merged[key] == value, (index, key)
        for key, dim in SEMANTIC_FIELDS:
            assert merged[key + '_shape'] == [
                len(legacy['adapter_candidate_query']), dim
            ]

    receipt = {
        'protocol': 'frozen_legacy_rows_plus_query_id_aligned_semantic_features',
        'legacy_dump': os.path.abspath(args.legacy_dump),
        'legacy_dump_sha256': sha256(args.legacy_dump),
        'semantic_dump': os.path.abspath(args.semantic_dump),
        'semantic_dump_sha256': sha256(args.semantic_dump),
        'output_dump': os.path.abspath(args.output_dump),
        'output_dump_sha256': sha256(args.output_dump),
        'row_count': len(reloaded),
        'candidate_count': candidate_count,
        'exact_query_order_rows': exact_query_order,
        'legacy_fields_preserved_exactly': True,
        'semantic_fields': [key for key, _ in SEMANTIC_FIELDS],
    }
    with open(args.receipt_json, 'w', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
