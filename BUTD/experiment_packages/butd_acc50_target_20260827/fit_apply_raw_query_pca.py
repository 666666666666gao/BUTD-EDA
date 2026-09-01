#!/usr/bin/env python
"""Fit a train-only PCA for raw decoder queries and append compact features."""

import argparse
import hashlib
import json
import os

import numpy as np
import torch


RAW_KEY = 'adapter_last_query_f16'
RAW_DIM = 288
PCA_KEY = 'adapter_last_query_pca64_f16'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scene_bucket(scene_id):
    value = int(hashlib.sha1(scene_id.encode('utf-8')).hexdigest()[:8], 16)
    return value % 100


def decode(row, key, expected_rows, expected_dim):
    shape = tuple(int(value) for value in row[key + '_shape'])
    assert shape == (int(expected_rows), int(expected_dim)), (
        key, shape, expected_rows, expected_dim
    )
    values = np.frombuffer(row[key], dtype=np.float16).reshape(shape)
    assert np.isfinite(values).all(), key
    return values.astype(np.float32)


def deterministic_signs(components):
    components = np.asarray(components, dtype=np.float32).copy()
    for index in range(len(components)):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0:
            components[index] *= -1.0
    return components


def fit_projector(rows, fit_bucket_end, sample_size, output_dim, seed):
    fit_vectors = []
    fit_candidate_count = 0
    fit_scene_ids = set()
    for row in rows:
        scene_id = str(row.get('scene_id', ''))
        assert scene_id
        if scene_bucket(scene_id) > int(fit_bucket_end):
            continue
        queries = row['adapter_candidate_query']
        values = decode(row, RAW_KEY, len(queries), RAW_DIM)
        fit_vectors.append(values)
        fit_candidate_count += len(values)
        fit_scene_ids.add(scene_id)
    matrix = np.concatenate(fit_vectors, axis=0)
    assert len(matrix) == fit_candidate_count
    rng = np.random.RandomState(int(seed))
    if len(matrix) > int(sample_size):
        sample_indices = np.sort(
            rng.choice(len(matrix), size=int(sample_size), replace=False)
        )
        sample = matrix[sample_indices]
    else:
        sample = matrix
    mean = sample.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = sample - mean
    covariance = (
        centered.T.astype(np.float64) @ centered.astype(np.float64)
    ) / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order[:int(output_dim)]].T
    components = deterministic_signs(components)
    explained_ratio = float(
        eigenvalues[:int(output_dim)].sum() / max(eigenvalues.sum(), 1e-12)
    )
    return {
        'mean': mean,
        'components': components,
        'eigenvalues': eigenvalues[:int(output_dim)].astype(np.float32),
        'fit_candidate_count': int(fit_candidate_count),
        'fit_scene_count': int(len(fit_scene_ids)),
        'sample_count': int(len(sample)),
        'explained_variance_ratio': explained_ratio,
    }


def save_projector(path, projector, metadata):
    assert not os.path.exists(path), path
    temporary = path + '.tmp'
    assert not os.path.exists(temporary), temporary
    with open(temporary, 'wb') as handle:
        np.savez(
            handle,
            mean=projector['mean'],
            components=projector['components'],
            eigenvalues=projector['eigenvalues'],
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    os.replace(temporary, path)


def load_projector(path):
    payload = np.load(path, allow_pickle=False)
    mean = np.asarray(payload['mean'], dtype=np.float32)
    components = np.asarray(payload['components'], dtype=np.float32)
    metadata = json.loads(str(payload['metadata_json'].item()))
    assert mean.shape == (RAW_DIM,), mean.shape
    assert components.ndim == 2 and components.shape[1] == RAW_DIM
    assert np.isfinite(mean).all() and np.isfinite(components).all()
    return mean, components, metadata


def transform_rows(rows, mean, components):
    transformed = []
    candidate_count = 0
    value_sum = value_sq_sum = 0.0
    value_count = 0
    for index, row in enumerate(rows):
        queries = row['adapter_candidate_query']
        raw = decode(row, RAW_KEY, len(queries), RAW_DIM)
        projected = ((raw - mean) @ components.T).astype(np.float16)
        assert np.isfinite(projected).all(), index
        merged = dict(row)
        merged[PCA_KEY] = projected.tobytes(order='C')
        merged[PCA_KEY + '_shape'] = [
            int(projected.shape[0]), int(projected.shape[1])
        ]
        for key, value in row.items():
            assert merged[key] == value, (index, key)
        transformed.append(merged)
        candidate_count += len(projected)
        values64 = projected.astype(np.float64)
        value_sum += float(values64.sum())
        value_sq_sum += float(np.square(values64).sum())
        value_count += int(values64.size)
    mean_value = value_sum / max(value_count, 1)
    std_value = max(
        value_sq_sum / max(value_count, 1) - mean_value * mean_value,
        0.0,
    ) ** 0.5
    assert std_value > 1e-6
    return transformed, {
        'candidate_count': int(candidate_count),
        'projected_value_mean': float(mean_value),
        'projected_value_std': float(std_value),
    }


def write_transformed(input_payload, rows, output_dump, provenance):
    assert not os.path.exists(output_dump), output_dump
    payload = dict(input_payload)
    payload['rows'] = rows
    payload['raw_query_pca_provenance'] = provenance
    temporary = output_dump + '.tmp'
    assert not os.path.exists(temporary), temporary
    torch.save(payload, temporary)
    os.replace(temporary, output_dump)
    reloaded = torch.load(output_dump, map_location='cpu')['rows']
    assert len(reloaded) == len(rows)
    for index, (expected, actual) in enumerate(zip(rows, reloaded)):
        assert actual[PCA_KEY + '_shape'] == expected[PCA_KEY + '_shape']
        assert actual[PCA_KEY] == expected[PCA_KEY], index


def fit_transform(args):
    for path in (args.projector, args.output_dump, args.receipt_json):
        assert not os.path.exists(path), path
    input_payload = torch.load(args.input_dump, map_location='cpu')
    rows = input_payload['rows']
    projector = fit_projector(
        rows, args.fit_bucket_end, args.sample_size, args.output_dim, args.seed
    )
    metadata = {
        'protocol': 'raw_query_pca_fit_train_scene_buckets_only',
        'input_dump': os.path.abspath(args.input_dump),
        'input_dump_sha256': sha256(args.input_dump),
        'fit_bucket_range': [0, int(args.fit_bucket_end)],
        'fit_candidate_count': projector['fit_candidate_count'],
        'fit_scene_count': projector['fit_scene_count'],
        'sample_count': projector['sample_count'],
        'seed': int(args.seed),
        'raw_dim': RAW_DIM,
        'output_dim': int(args.output_dim),
        'explained_variance_ratio': projector['explained_variance_ratio'],
    }
    save_projector(args.projector, projector, metadata)
    mean, components, loaded_metadata = load_projector(args.projector)
    assert loaded_metadata == metadata
    transformed, stats = transform_rows(rows, mean, components)
    provenance = dict(metadata)
    provenance.update({
        'projector': os.path.abspath(args.projector),
        'projector_sha256': sha256(args.projector),
        **stats,
    })
    write_transformed(input_payload, transformed, args.output_dump, provenance)
    receipt = dict(provenance)
    receipt.update({
        'output_dump': os.path.abspath(args.output_dump),
        'output_dump_sha256': sha256(args.output_dump),
        'row_count': len(transformed),
        'legacy_fields_preserved_exactly': True,
    })
    with open(args.receipt_json, 'w', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def apply_projector(args):
    for path in (args.output_dump, args.receipt_json):
        assert not os.path.exists(path), path
    input_payload = torch.load(args.input_dump, map_location='cpu')
    mean, components, metadata = load_projector(args.projector)
    transformed, stats = transform_rows(input_payload['rows'], mean, components)
    provenance = {
        'protocol': 'apply_locked_raw_query_pca',
        'input_dump': os.path.abspath(args.input_dump),
        'input_dump_sha256': sha256(args.input_dump),
        'projector': os.path.abspath(args.projector),
        'projector_sha256': sha256(args.projector),
        'projector_metadata': metadata,
        **stats,
    }
    write_transformed(input_payload, transformed, args.output_dump, provenance)
    receipt = dict(provenance)
    receipt.update({
        'output_dump': os.path.abspath(args.output_dump),
        'output_dump_sha256': sha256(args.output_dump),
        'row_count': len(transformed),
        'legacy_fields_preserved_exactly': True,
    })
    with open(args.receipt_json, 'w', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    fit_parser = sub.add_parser('fit-transform')
    fit_parser.add_argument('input_dump')
    fit_parser.add_argument('projector')
    fit_parser.add_argument('output_dump')
    fit_parser.add_argument('receipt_json')
    fit_parser.add_argument('--fit-bucket-end', type=int, default=69)
    fit_parser.add_argument('--sample-size', type=int, default=50000)
    fit_parser.add_argument('--output-dim', type=int, default=64)
    fit_parser.add_argument('--seed', type=int, default=0)
    apply_parser = sub.add_parser('apply')
    apply_parser.add_argument('input_dump')
    apply_parser.add_argument('projector')
    apply_parser.add_argument('output_dump')
    apply_parser.add_argument('receipt_json')
    args = parser.parse_args()
    if args.command == 'fit-transform':
        fit_transform(args)
    elif args.command == 'apply':
        apply_projector(args)
    else:
        raise AssertionError(args.command)


if __name__ == '__main__':
    main()
